"""Evaluate both policies and one frozen drift replay"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

from config import MODEL_FEATURES, NUMBER_OF_DAYS, SEED, START_DATE
from src import dataset, features, policy, risk_model

PRE_SCENARIO = "pre_drift_test"
DRIFT_SCENARIO = "new_device_rollout"
ROLLOUT_FRACTION = 0.70
REPLAY_OFFSET_DAYS = 14
BOOTSTRAP_REPLICATES = 4000
DECISION_BANDS = ("allow", "challenge", "block")
REPLAY_MODIFIED_FIELDS = ("event_id", "timestamp", "device_id", "anomaly_campaign_id")
DECISION_FIELDS = ["scenario", "source_event_id", *dataset.EVENT_FIELDS, *[
    "risk_score", "baseline_decision", "baseline_reason_codes", "model_decision",
    "final_decision", "decision_source", "override_codes", "step_up_control",
    "reason_codes", "top_model_contributors", "explanation",
]]


def decisions_for_requests(events, frame, model, resources, volume_threshold, scenario):
    """Export both policy decisions on the same request, including audit context."""
    event_by_id = {event["event_id"]: event for event in events}
    if len(frame) != len(event_by_id) or set(frame["event_id"]) != set(event_by_id):
        raise ValueError("Decision requests and feature IDs do not match one-to-one.")
    scores = risk_model.score_requests(model, frame)
    contributors = risk_model.positive_contributions(model, frame)
    thresholds = model["thresholds"]
    records = []
    for position, (_, row) in enumerate(frame.iterrows()):
        event = event_by_id[row["event_id"]]
        score = float(scores[position])
        result = policy.decide(
            score, policy.context_for_event(event, resources),
            thresholds["challenge_threshold"], thresholds["block_threshold"],
        )
        baseline, baseline_reasons = policy.evaluate_event(
            event, resources[event["resource_id"]]["resource_group"], row, volume_threshold,
        )
        if result["final_decision"] == "allow":
            reasons = ["LOW_CALIBRATED_RISK"]
        else:
            reasons = list(result["override_codes"])
            if result["model_decision"] != "allow":
                reasons += policy.reason_codes(row)
            reasons = list(dict.fromkeys(reasons))
        explanation = (
            f"Risk {score:.2f}/100; model band={result['model_decision']}; "
            f"decision={result['final_decision']} via {result['decision_source']}; indicators="
            + ", ".join(reasons[:4])
            + "; positive model contributions="
            + ", ".join(contributors[position].split("|"))
        )
        records.append({
            "scenario": scenario,
            "source_event_id": (
                event["event_id"].removeprefix("drift_")
                if scenario == DRIFT_SCENARIO else event["event_id"]
            ),
            **event,
            "risk_score": score, "baseline_decision": baseline,
            "baseline_reason_codes": "|".join(baseline_reasons),
            "model_decision": result["model_decision"],
            "final_decision": result["final_decision"],
            "decision_source": result["decision_source"],
            "override_codes": "|".join(result["override_codes"]) or "NONE",
            "step_up_control": result["step_up_control"],
            "reason_codes": "|".join(reasons),
            "top_model_contributors": contributors[position],
            "explanation": explanation,
        })
    return records


def safe_ratio(numerator, denominator):
    return round(float(numerator / denominator), 4) if denominator else None


def binary_metrics(y_true, y_pred, scores=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "f1": safe_ratio(2 * tp, 2 * tp + fp + fn) if tp + fn else None,
        "false_positive_rate": safe_ratio(fp, fp + tn),
        "false_negative_rate": safe_ratio(fn, fn + tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if scores is not None:
        result["pr_auc"] = round(float(average_precision_score(y_true, scores)), 4)
    return result


def wilson_interval(successes, trials):
    """Descriptive 95% binomial interval."""
    if trials == 0:
        return None, None
    z = 1.959964
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return (
        round(max(0.0, centre - half), 4),
        round(min(1.0, centre + half), 4),
    )


def f1_from_flags(actual, flagged) -> float:
    """Unrounded F1 for resampling, where a degenerate arm scores zero.

    Inputs are coerced to arrays: a list compared with ``==`` yields a scalar
    and would silently score every arm at zero.
    """
    actual = np.asarray(actual)
    flagged = np.asarray(flagged)
    true_positive = int(np.sum((flagged == 1) & (actual == 1)))
    false_positive = int(np.sum((flagged == 1) & (actual == 0)))
    false_negative = int(np.sum((flagged == 0) & (actual == 1)))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def mcnemar_exact_p(first_correct, second_correct) -> float | None:
    """Exact two-sided McNemar p over the discordant pairs.

    Sums the binomial(0.5) mass of every outcome at least as unlikely as the
    observed split, which is the standard exact two-sided form.
    """
    trials = first_correct + second_correct
    if trials == 0:
        return None

    def mass(successes: int) -> float:
        return math.comb(trials, successes) * 0.5 ** trials

    observed = mass(first_correct)
    return round(sum(mass(k) for k in range(trials + 1) if mass(k) <= observed + 1e-12), 4)


def cluster_bootstrap(actual, first, second, clusters, replicates, seed):
    """Resample users, because campaigns and repeat users make events dependent."""
    positions = {key: np.flatnonzero(clusters == key) for key in np.unique(clusters)}
    keys = list(positions)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates)
    for index in range(replicates):
        drawn = np.concatenate(
            [positions[key] for key in rng.choice(keys, len(keys), replace=True)]
        )
        deltas[index] = (
            f1_from_flags(actual[drawn], first[drawn])
            - f1_from_flags(actual[drawn], second[drawn])
        )
    return deltas


def comparison_uncertainty(data) -> dict:
    """Paired evidence for the AI-versus-baseline difference on one evaluation set."""
    actual = data["actual"].to_numpy()
    ai = data["ai_flagged"].to_numpy()
    baseline = data["baseline_flagged"].to_numpy()
    ai_only = (ai == 1) & (baseline == 0)
    baseline_only = (ai == 0) & (baseline == 1)
    ai_correct = int(np.sum(ai_only & (actual == 1)) + np.sum(baseline_only & (actual == 0)))
    baseline_correct = int(np.sum(ai_only & (actual == 0)) + np.sum(baseline_only & (actual == 1)))
    deltas = cluster_bootstrap(
        actual, ai, baseline, data["user_id"].to_numpy(), BOOTSTRAP_REPLICATES, SEED
    )
    return {
        "unit": "evaluation request; the bootstrap resamples users with replacement",
        "discordant_pairs": {
            "ai_intervened_baseline_allowed": int(ai_only.sum()),
            "baseline_intervened_ai_allowed": int(baseline_only.sum()),
            "ai_correct_baseline_wrong": ai_correct,
            "baseline_correct_ai_wrong": baseline_correct,
        },
        "mcnemar_exact_two_sided_p": mcnemar_exact_p(ai_correct, baseline_correct),
        "f1_difference": {
            "point_estimate": round(
                f1_from_flags(actual, ai) - f1_from_flags(actual, baseline), 4
            ),
            "cluster": "user_id",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
            "mean": round(float(deltas.mean()), 4),
            "ci95_low": round(float(np.percentile(deltas, 2.5)), 4),
            "ci95_high": round(float(np.percentile(deltas, 97.5)), 4),
            "share_at_or_below_zero": round(float(np.mean(deltas <= 0)), 4),
        },
        "interpretation": (
            "Both statistics are computed on the same frozen decisions as the headline "
            "metrics. McNemar treats requests as independent; the user-clustered "
            "bootstrap relaxes that assumption for the interval."
        ),
    }


def component_ablation(data, band) -> dict:
    """Separate the learned score from the mandatory and contextual controls."""
    mandatory = data["override_codes"].str.contains("MANDATORY_STATIC_BLOCK")
    contextual = data["decision_source"] == "contextual_override"
    arms = [
        ("score_band_only", band != "allow"),
        ("full_ai_policy", (data["final_decision"] != "allow").to_numpy()),
        ("mandatory_and_contextual_controls_only", (mandatory | contextual).to_numpy()),
        ("static_baseline", (data["baseline_decision"] != "allow").to_numpy()),
    ]
    rows = []
    for name, flagged in arms:
        row = {"variant": name, "interventions": int(flagged.sum())}
        row.update(binary_metrics(data["actual"].to_numpy(), flagged.astype(int)))
        rows.append(row)
    return {
        "variants": rows,
        "mandatory_block_events": {
            "events": int(mandatory.sum()),
            "already_in_block_band": int((mandatory & (band == "block")).sum()),
            "anomalies": int(data["actual"][mandatory].sum()),
            "normal_events": int((1 - data["actual"])[mandatory].sum()),
        },
        "contextual_override_events": {
            "events": int(contextual.sum()),
            "anomalies": int(data["actual"][contextual].sum()),
            "normal_events": int((1 - data["actual"])[contextual].sum()),
        },
        "interpretation": (
            "score_band_only drops both control layers and mandatory_and_contextual_controls_only "
            "drops the score, so the gap between score_band_only and full_ai_policy is the "
            "detection cost the control layers carry on this evaluation set."
        ),
    }


def score_resolution(data, band) -> dict:
    """Report how much graded information the calibrated score actually carries."""
    occupancy = {}
    for name in DECISION_BANDS:
        selected = band == name
        occupancy[name] = {
            "events": int(selected.sum()),
            "anomalies": int(data["actual"][selected].sum()),
            "share_of_events": round(float(selected.mean()), 4),
            "distinct_scores": int(data["risk_score"][selected].nunique()),
        }
    return {
        "evaluation_events": int(len(data)),
        "evaluation_distinct_scores": int(data["risk_score"].nunique()),
        "band_occupancy": occupancy,
        "interpretation": (
            "Isotonic calibration fitted on the calibration partition is a step function, so the "
            "exported score takes few distinct values and any threshold pair inside a gap between "
            "adjacent values selects the same partition. Within a band many requests share one "
            "score atom, so case selection sorts on event_id after risk_score."
        ),
    }


CASE_SORT_COLUMNS = ["risk_score", "event_id"]


def select_examples(data, band):
    """High and medium-risk explanations, each chosen by a fully specified rule.

    One highest-scoring intervened request per anomaly category supplies the
    high-risk cases. The medium-risk cases are the challenge-band anomalies, one
    per category, because that band can hold many requests on a single score
    atom and a score-only rule there would pick arbitrarily.
    """
    intervened = data[data["final_decision"].isin(["challenge", "block"])]
    detected = (
        intervened[intervened["actual"] == 1]
        .sort_values(CASE_SORT_COLUMNS, ascending=[False, True])
        .groupby("anomaly_type", group_keys=False)
        .head(1)
        .sort_values(CASE_SORT_COLUMNS, ascending=[False, True])
    )
    detected = detected.assign(case_role="detected_" + detected["anomaly_type"])
    medium_mask = (band == "challenge") & (data["actual"].to_numpy() == 1)
    medium = (
        data[medium_mask]
        .sort_values(CASE_SORT_COLUMNS, ascending=[False, True])
        .groupby("anomaly_type", group_keys=False)
        .head(1)
        .sort_values(CASE_SORT_COLUMNS, ascending=[False, True])
    )
    medium = medium.assign(case_role="medium_risk_" + medium["anomaly_type"])
    return pd.concat([detected, medium])


def select_error_cases(data):
    """The worst false positive and the worst false negative on the same rule."""
    intervened = data[data["final_decision"].isin(["challenge", "block"])]
    false_positive = (
        intervened[intervened["actual"] == 0]
        .sort_values(CASE_SORT_COLUMNS, ascending=[False, True])
        .head(1)
        .assign(case_role="highest_scoring_false_positive")
    )
    false_negative = (
        data[(data["actual"] == 1) & (data["final_decision"] == "allow")]
        .sort_values(CASE_SORT_COLUMNS, ascending=[True, True])
        .head(1)
        .assign(case_role="lowest_scoring_false_negative")
    )
    return pd.concat([false_positive, false_negative])


def department_metrics(frame) -> list[dict]:
    """The required group breakdown, using department."""
    rows = []
    for department, part in frame.groupby("department"):
        y = part["actual"].to_numpy()
        ai = part["ai_flagged"].to_numpy()
        baseline = part["baseline_flagged"].to_numpy()
        row = {
            "group_dimension": "department", "group": department,
            "events": int(len(part)), "anomalies": int(y.sum()),
            "normal_events": int(len(part) - y.sum()),
        }
        for prefix, predicted in [("ai", ai), ("baseline", baseline)]:
            for metric, value in binary_metrics(y, predicted).items():
                if metric != "confusion_matrix":
                    row[f"{prefix}_{metric}"] = value
        row["challenge_rate"] = round(float((part["final_decision"] == "challenge").mean()), 4)
        row["block_rate"] = round(float((part["final_decision"] == "block").mean()), 4)
        intervals = {
            "ai_recall": wilson_interval(
                int(np.sum((y == 1) & (ai == 1))), int(np.sum(y == 1))
            ),
            "ai_false_positive_rate": wilson_interval(
                int(np.sum((y == 0) & (ai == 1))), int(np.sum(y == 0))
            ),
            "block_rate": wilson_interval(
                int((part["final_decision"] == "block").sum()), len(part)
            ),
        }
        for metric, (low, high) in intervals.items():
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return rows


def evaluation_summary(records, feature_frame, volume_threshold, thresholds) -> dict:
    data = pd.DataFrame(records)
    evidence_fields = [
        "event_id", "outside_role_hours", "device_seen_before",
        "device_managed_numeric", "role_resource_allowed", "location_change_60m",
        "user_events_1h", "user_volume_1h_mb", "large_transfer",
    ]
    data = data.merge(feature_frame[evidence_fields], on="event_id", validate="one_to_one")
    if len(data) != len(records):
        raise ValueError("Evaluation join lost requests.")
    data["actual"] = (data["ground_truth"] == "anomalous").astype(int)
    data["ai_flagged"] = (data["final_decision"] != "allow").astype(int)
    data["baseline_flagged"] = (data["baseline_decision"] != "allow").astype(int)
    band = np.array([
        policy.model_band(
            score, thresholds["challenge_threshold"], thresholds["block_threshold"]
        )
        for score in data["risk_score"]
    ])
    ai = binary_metrics(data["actual"], data["ai_flagged"], data["risk_score"] / 100)
    baseline = binary_metrics(data["actual"], data["baseline_flagged"])
    for result, column in [(ai, "final_decision"), (baseline, "baseline_decision")]:
        result["challenge_rate"] = round(float((data[column] == "challenge").mean()), 4)
        result["block_rate"] = round(float((data[column] == "block").mean()), 4)
    examples = select_examples(data, band)
    error_cases = select_error_cases(data)
    example_fields = [
        "case_role", "event_id", "timestamp", "role", "department", "resource_id",
        "resource_sensitivity", "anomaly_type", "risk_score", "model_decision",
        "final_decision", "decision_source", "step_up_control", "reason_codes",
        "top_model_contributors", *evidence_fields[1:], "explanation",
    ]
    model_flagged = (data["model_decision"] != "allow").astype(int)
    return {
        "evaluation_set": {
            "events": int(len(data)), "anomalies": int(data["actual"].sum()),
            "anomaly_rate": round(float(data["actual"].mean()), 4),
        },
        "ai_assisted": ai, "static_baseline": baseline,
        "baseline_volume_threshold_mb": volume_threshold,
        "dynamic_decision_counts": {
            str(key): int(value) for key, value in data["final_decision"].value_counts().items()
        },
        "policy_diagnostics": {
            "decision_source_counts": {
                str(key): int(value)
                for key, value in data["decision_source"].value_counts().items()
            },
            "decisions_changed_by_context": int(
                (data["model_decision"] != data["final_decision"]).sum()
            ),
            "interventions_changed_by_context": int(
                (model_flagged != data["ai_flagged"]).sum()
            ),
        },
        "comparison_uncertainty": comparison_uncertainty(data),
        "component_ablation": component_ablation(data, band),
        "score_resolution": score_resolution(data, band),
        "departments": department_metrics(data),
        "case_selection": {
            "examples_rule": (
                "highest-scoring intervened request per anomaly category, then one "
                "challenge-band anomaly per category; risk_score descending with event_id "
                "breaking ties"
            ),
            "error_cases_rule": (
                "highest-scoring false positive and lowest-scoring false negative; "
                "risk_score then event_id breaking ties"
            ),
            "pool": "evaluation requests only; case_role labels each row",
            "examples": int(len(examples)),
            "error_cases": int(len(error_cases)),
        },
        "examples": examples[example_fields].to_dict(orient="records"),
        "error_cases": error_cases[example_fields].to_dict(orient="records"),
        "interpretation": {
            "positive_definition": "challenge or block",
            "negative_definition": "allow",
            "pr_auc_definition": "average precision on the calibrated risk score",
            "intervention_caveat": (
                "A challenge on a legitimate request counts as a detection false positive, "
                "even when verification is appropriate under the security policy. Normal "
                "personal-device activity therefore measures assurance burden as well as "
                "anomaly detection; normal does not mean unconditional permission."
            ),
            "group_uncertainty": (
                "Wilson intervals are descriptive event-level summaries. Repeated users "
                "and campaigns create dependence, so no inferential claim is drawn from a "
                "single department row; comparison_uncertainty resamples users instead."
            ),
        },
    }


def select_rollout_users(events):
    user_ids = sorted({event["user_id"] for event in events})
    return frozenset(dataset.seeded_random(11).sample(
        user_ids, round(ROLLOUT_FRACTION * len(user_ids))
    ))


def build_drift_events(evaluation_events, rollout_users):
    """Apply device changes by user/trust status."""
    output = []
    for source in sorted(evaluation_events, key=lambda event: (event["timestamp"], event["event_id"])):
        event = dict(source)
        event["event_id"] = f"drift_{source['event_id']}"
        event["timestamp"] = dataset.iso_utc(
            dataset.parse_utc(source["timestamp"]) + timedelta(days=REPLAY_OFFSET_DAYS)
        )
        if source["device_managed"] != "true":
            event["device_id"] = f"DRIFT-UNMANAGED-{source['device_id']}"
        elif source["user_id"] in rollout_users:
            event["device_id"] = f"DRIFT-ROLLOUT-{source['user_id']}"
        if source["anomaly_campaign_id"]:
            event["anomaly_campaign_id"] = f"drift_{source['anomaly_campaign_id']}"
        output.append(event)
    return output


def metric_row(system, scenario, records) -> dict:
    actual = np.asarray([record["ground_truth"] == "anomalous" for record in records]).astype(int)
    column = "final_decision" if system == "ai_assisted" else "baseline_decision"
    flagged = np.asarray([record[column] != "allow" for record in records]).astype(int)
    scores = (
        np.asarray([record["risk_score"] for record in records]) / 100
        if system == "ai_assisted" else None
    )
    return {
        "system": system, "scenario": scenario, "events": len(records),
        "anomalies": int(actual.sum()), **binary_metrics(actual, flagged, scores),
        "challenge_or_block_rate": round(float(np.mean(flagged)), 4),
    }


def replay_effect(original, changed) -> dict:
    """Measure what the replay changes per field and per decision, arm by arm."""
    by_source = {row["source_event_id"]: row for row in original}
    if len(by_source) != len(original):
        raise ValueError("Pre-drift records are not uniquely keyed by event identifier.")
    field_changes = Counter()
    baseline_decision_changes = 0
    baseline_reason_changes = 0
    for row in changed:
        source = by_source[row["source_event_id"]]
        for field in dataset.EVENT_FIELDS:
            if row[field] != source[field]:
                field_changes[field] += 1
        if row["baseline_decision"] != source["baseline_decision"]:
            baseline_decision_changes += 1
        if row["baseline_reason_codes"] != source["baseline_reason_codes"]:
            baseline_reason_changes += 1
    return {
        "pairs": len(changed),
        "fields_changed": {
            field: field_changes[field]
            for field in dataset.EVENT_FIELDS if field_changes.get(field)
        },
        "rule_relevant_fields_changed": {
            field: count for field, count in sorted(field_changes.items())
            if field not in REPLAY_MODIFIED_FIELDS
        },
        "baseline_decisions_changed": baseline_decision_changes,
        "baseline_reason_codes_changed": baseline_reason_changes,
        "interpretation": (
            "The replay modifies identifiers, timestamps and device identity only. No static "
            "rule reads device_id, a whole-day offset preserves hour of day, and preserved "
            "relative timing leaves the history features unchanged, so the baseline arm cannot "
            "respond to this scenario and its before and after rows restate one measurement. "
            "The AI arm responds through the device-novelty features."
        ),
    }


def drift_summary(original, changed, events, rollout_users) -> dict:
    original_ids = {record["event_id"] for record in original}
    if (
        len(original) != len(changed)
        or original_ids != {record["source_event_id"] for record in changed}
    ):
        raise ValueError("Pre-drift and drift evaluation supports differ.")
    return {
        "scenario_id": DRIFT_SCENARIO,
        "scenario": (
            "Evaluation requests replayed two weeks later. A fixed 70% user cohort "
            "moves managed requests to genuinely new managed devices, independently "
            "of event labels. Unmanaged devices receive fresh untrusted identities."
        ),
        "rollout": {
            "cohort_selection": "seeded sample of user IDs; labels and attack types ignored",
            "users_total": len({event["user_id"] for event in events}),
            "users_selected": len(rollout_users), "user_ids": sorted(rollout_users),
            "managed_requests_changed": sum(
                record["device_id"].startswith("DRIFT-ROLLOUT-") for record in changed
            ),
            "replay_offset_days": REPLAY_OFFSET_DAYS,
        },
        "comparison": [
            metric_row("ai_assisted", PRE_SCENARIO, original),
            metric_row("ai_assisted", DRIFT_SCENARIO, changed),
            metric_row("static_baseline", PRE_SCENARIO, original),
            metric_row("static_baseline", DRIFT_SCENARIO, changed),
        ],
        "replay_effect": replay_effect(original, changed),
        "adaptation_evaluated": False,
        "recommended_response": (
            "Review the approved rollout and device inventory first. "
            "Consider threshold recalibration or retraining only on reviewed change "
            "data, followed by a new untouched evaluation period."
        ),
        "limitations": [
            "A matched replay is not an independent future workload or new attack sample.",
            "First-seen-device attacks are all unmanaged, but benign personal-device requests now overlap in trust status.",
            "There is a gap with no generated activity before the replay.",
            "Device rollout and replay history both affect the before/after comparison; it is not a causal estimate.",
        ],
    }


def run_experiment():
    """Return dataset, decisions and summary, keeping all working state in memory."""
    events, enterprise = dataset.generate_dataset()
    validation = dataset.validate_dataset(events, enterprise)
    resources = {resource["resource_id"]: resource for resource in enterprise["resources"]}
    feature_rows = features.compute_features(events, resources)
    frame = pd.DataFrame(feature_rows)
    history = {row["event_id"]: row for row in feature_rows}
    selection = policy.select_volume_threshold(events, resources, history)
    volume = selection["volume_threshold_mb"]
    model = risk_model.train_model(frame)
    test_frame = frame[frame["split"] == "test"]
    test_ids = set(test_frame["event_id"])
    test_events = [event for event in events if event["event_id"] in test_ids]
    original = decisions_for_requests(
        test_events, test_frame, model, resources, volume, PRE_SCENARIO
    )
    rollout_users = select_rollout_users(events)
    drift_events = build_drift_events(test_events, rollout_users)
    old_devices = {event["device_id"] for event in events}
    new_devices = {
        event["device_id"] for event in drift_events
        if event["device_id"].startswith("DRIFT-")
    }
    if old_devices & new_devices:
        raise ValueError("Drift reused a supposedly fresh device identity.")
    combined_features = features.compute_features(events + drift_events, resources)
    drift_ids = {event["event_id"] for event in drift_events}
    drift_frame = pd.DataFrame([
        row for row in combined_features if row["event_id"] in drift_ids
    ])
    changed = decisions_for_requests(
        drift_events, drift_frame, model, resources, volume, DRIFT_SCENARIO
    )
    split_counts = defaultdict(lambda: {"normal": 0, "anomalous": 0})
    for row in feature_rows:
        split_counts[row["split"]][row["ground_truth"]] += 1
    summary = {
        "run": {"seed": SEED, "start_date": START_DATE.isoformat(), "history_days": NUMBER_OF_DAYS},
        "dataset": validation,
        "split": dict(split_counts),
        "baseline": policy.baseline_description(selection),
        "model": {**model["metadata"], "features": MODEL_FEATURES},
        "evaluation": evaluation_summary(original, test_frame, volume, model["thresholds"]),
        "drift": drift_summary(original, changed, events, rollout_users),
    }
    return events, original + changed, summary


def create_plots(summary, records, directory):
    """Render exactly the three approved figures, without a display dependency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    background, text = "#F7F6F2", "#28251D"
    teal, rust, dark_teal, gold = "#20808D", "#A84B2F", "#1B474D", "#D19900"

    def style_axis(ax):
        ax.set_facecolor(background)
        ax.figure.set_facecolor(background)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", color="#D4D1CA", linewidth=0.7)
        ax.tick_params(colors=text)
        ax.title.set_color(text)

    def save(fig, name):
        fig.savefig(directory / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    metrics, drift = summary["evaluation"], summary["drift"]
    scores = pd.DataFrame([record for record in records if record["scenario"] == PRE_SCENARIO])
    names = ["Precision", "Recall", "F1"]
    ai = [metrics["ai_assisted"][name.lower()] for name in names]
    baseline = [metrics["static_baseline"][name.lower()] for name in names]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    x = range(len(names))
    ax.bar([v - 0.18 for v in x], baseline, width=0.36,
           label="Static baseline", color=rust, hatch="//")
    ax.bar([v + 0.18 for v in x], ai, width=0.36,
           label="AI-assisted (score + context)", color=teal)
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Static baseline versus AI-assisted policy\nSame chronological evaluation set")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    style_axis(ax)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    save(fig, "baseline-vs-ai.png")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for label, color in [("normal", dark_teal), ("anomalous", gold)]:
        subset = scores[scores["ground_truth"] == label]["risk_score"]
        ax.hist(subset, bins=20, alpha=0.72, label=label.capitalize(), color=color)
    ax.set_xlabel("Calibrated risk score (0-100)")
    ax.set_ylabel("Evaluation events")
    ax.set_title("Risk-score distribution\nHeld-out days 12-14")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, "risk-score-distribution.png")

    rows = {(row["system"], row["scenario"]): row for row in drift["comparison"]}
    scenarios = [PRE_SCENARIO, DRIFT_SCENARIO]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for system, offset, color, label in [
        ("static_baseline", -0.18, rust, "Static baseline"),
        ("ai_assisted", 0.18, teal, "AI-assisted"),
    ]:
        rates = [rows[(system, scenario)]["challenge_or_block_rate"] for scenario in scenarios]
        bars = ax.bar([v + offset for v in range(2)], rates, width=0.36, color=color, label=label)
        ax.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates], padding=4)
    ax.set_xticks([0, 1], ["Pre-drift", "New-device rollout"])
    ax.set_ylim(0, max(
        rows[(system, scenario)]["challenge_or_block_rate"]
        for system in ["static_baseline", "ai_assisted"] for scenario in scenarios
    ) * 1.3 + 0.02)
    ax.set_ylabel("Challenge or block rate")
    ax.set_title("Operational impact of drift\nFrozen policies before and after context change")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, "drift-intervention-rate.png")
