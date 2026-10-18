"""Acceptance and regression tests for the six-output experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import unittest
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    DATA_DIR, DEPARTMENT_SPECS, MODEL_FEATURES, NUMBER_OF_DAYS,
    PERSONAL_DEVICE_REQUESTS_PER_USER, PERSONAL_DEVICE_USERS_PER_DEPARTMENT,
    RESULTS_DIR,
)
from src import dataset, evaluation, features, policy, risk_model


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def snapshot():
    """Rebuild working state in memory; do not create test checkpoint files."""
    events = read_rows(DATA_DIR / "access_events.csv")
    records = read_rows(RESULTS_DIR / "decisions.csv")
    with (RESULTS_DIR / "results.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    enterprise = dataset.generate_enterprise()
    resources = {row["resource_id"]: row for row in enterprise["resources"]}
    rows = features.compute_features(events, resources)
    frame = pd.DataFrame(rows)
    original = [row for row in records if row["scenario"] == evaluation.PRE_SCENARIO]
    changed = [row for row in records if row["scenario"] == evaluation.DRIFT_SCENARIO]
    drift_events = [
        {field: row[field] for field in dataset.EVENT_FIELDS} for row in changed
    ]
    combined = features.compute_features(events + drift_events, resources)
    return {
        "events": events, "records": records, "summary": summary,
        "enterprise": enterprise, "resources": resources, "frame": frame,
        "feature_by_id": {row["event_id"]: row for row in rows},
        "model": risk_model.train_model(frame),
        "original": original, "changed": changed, "drift_events": drift_events,
        "combined_by_id": {row["event_id"]: row for row in combined},
    }


class DatasetTests(unittest.TestCase):
    def test_dataset_minimums_and_reproducible_generation(self):
        state = snapshot()
        events = state["events"]
        regenerated, _ = dataset.generate_dataset()
        self.assertEqual(events, regenerated)
        self.assertGreaterEqual(len(events), 12000)
        self.assertGreaterEqual(len({row["user_id"] for row in events}), 50)
        self.assertGreaterEqual(len({row["department"] for row in events}), 4)
        anomalies = [row for row in events if row["ground_truth"] == "anomalous"]
        self.assertGreaterEqual(len({row["anomaly_type"] for row in anomalies}), 5)
        self.assertTrue(0.03 <= len(anomalies) / len(events) <= 0.10)

    def test_configured_date_coverage(self):
        dates = {row["timestamp"][:10] for row in snapshot()["events"]}
        self.assertEqual(len(dates), NUMBER_OF_DAYS)
        self.assertGreaterEqual(len(dates), 14)

    def test_chronological_split_boundaries_do_not_overlap(self):
        by_split = defaultdict(list)
        for row in snapshot()["feature_by_id"].values():
            by_split[row["split"]].append(dataset.parse_utc(row["timestamp"]))
        self.assertLess(max(by_split["train"]), min(by_split["calibration"]))
        self.assertLess(max(by_split["calibration"]), min(by_split["test"]))

    def test_first_user_requests_have_zero_device_history(self):
        first = {}
        for row in snapshot()["feature_by_id"].values():
            first.setdefault(row["user_id"], row["device_seen_before"])
        self.assertTrue(all(value == 0 for value in first.values()))

    def test_labels_identifiers_and_current_outcomes_are_not_model_inputs(self):
        forbidden = {
            "ground_truth", "anomaly_type", "anomaly_campaign_id", "event_id",
            "user_id", "device_id", "timestamp", "outcome",
        }
        self.assertTrue(forbidden.isdisjoint(MODEL_FEATURES))

    def test_csv_headers_and_in_memory_feature_schema_are_unique(self):
        for path, expected in [
            (DATA_DIR / "access_events.csv", dataset.EVENT_FIELDS),
            (RESULTS_DIR / "decisions.csv", evaluation.DECISION_FIELDS),
        ]:
            with path.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(header, expected)
            self.assertEqual(len(header), len(set(header)))
        self.assertEqual(len(features.FEATURE_FIELDS), len(set(features.FEATURE_FIELDS)))
        self.assertTrue(set(MODEL_FEATURES).issubset(snapshot()["frame"].columns))
        self.assertEqual(len(MODEL_FEATURES), len(set(MODEL_FEATURES)))

    def test_hour_windows_use_strictly_earlier_timestamps(self):
        state = snapshot()
        by_user = defaultdict(list)
        for event in state["events"]:
            by_user[event["user_id"]].append(event)
        for events in by_user.values():
            ordered = sorted(events, key=lambda event: event["timestamp"])
            times = [dataset.parse_utc(event["timestamp"]) for event in ordered]
            for event, timestamp in zip(ordered, times):
                first = bisect_left(times, timestamp - timedelta(hours=1))
                stop = bisect_left(times, timestamp)
                row = state["feature_by_id"][event["event_id"]]
                self.assertEqual(row["user_events_1h"], stop - first, event["event_id"])
                self.assertAlmostEqual(
                    row["user_volume_1h_mb"],
                    math.fsum(float(previous["data_volume_mb"]) for previous in ordered[first:stop]),
                    places=8, msg=event["event_id"],
                )

    def test_entity_references_and_resource_metadata(self):
        state = snapshot()
        users = {row["user_id"]: row for row in state["enterprise"]["users"]}
        self.assertGreaterEqual(len(state["resources"]), 25)
        for event in state["events"]:
            user = users[event["user_id"]]
            resource = state["resources"][event["resource_id"]]
            self.assertEqual(event["role"], user["role"])
            self.assertEqual(event["department"], user["department"])
            self.assertEqual(event["resource_name"], resource["resource_name"])
            self.assertEqual(event["resource_sensitivity"], resource["sensitivity"])
            self.assertIn(event["action"], resource["allowed_actions"].split("|"))
            self.assertIn(event["outcome"], {"success", "denied"})
            self.assertIn(event["ground_truth"], {"normal", "anomalous"})
            self.assertGreaterEqual(float(event["data_volume_mb"]), 0)

    def test_benign_counterexamples_exist(self):
        normal = [row for row in snapshot()["events"] if row["ground_truth"] == "normal"]
        self.assertTrue(any(row["network_zone"] == "foreign_public_network" for row in normal))
        self.assertTrue(any(
            int(row["timestamp"][11:13]) < 5 and row["resource_sensitivity"] == "privileged"
            for row in normal
        ))
        self.assertTrue(any(row["device_id"].startswith("ROLLOUT-") for row in normal))
        self.assertTrue(any(float(row["data_volume_mb"]) >= 80 for row in normal))
        self.assertTrue(any(
            (row["department"] == "Engineering" and row["resource_id"] == "RES-021")
            or (row["department"] == "Sales" and row["resource_id"] == "RES-007")
            for row in normal
        ))

    def test_personal_device_overlap_and_request_context(self):
        state = snapshot()
        personal = [
            row for row in state["events"] if row["device_id"].startswith("PERSONAL-")
        ]
        expected_users = len(DEPARTMENT_SPECS) * PERSONAL_DEVICE_USERS_PER_DEPARTMENT
        self.assertEqual(
            len(personal), expected_users * sum(PERSONAL_DEVICE_REQUESTS_PER_USER.values())
        )
        users_by_department = defaultdict(set)
        mix = Counter()
        for event in personal:
            self.assertEqual(event["ground_truth"], "normal")
            self.assertEqual(event["anomaly_type"], "none")
            self.assertEqual(event["device_managed"], "false")
            users_by_department[event["department"]].add(event["user_id"])
            feature = state["feature_by_id"][event["event_id"]]
            self.assertEqual(feature["outside_role_hours"], 0)
            self.assertEqual(feature["role_resource_allowed"], 1)
            if event["resource_sensitivity"] in {"public", "internal"}:
                mix["public_or_internal"] += 1
            else:
                self.assertEqual(event["resource_sensitivity"], "confidential")
                self.assertEqual(event["action"], "read")
                mix["confidential_read"] += 1
        self.assertEqual(set(users_by_department), set(DEPARTMENT_SPECS))
        self.assertTrue(all(
            len(users) == PERSONAL_DEVICE_USERS_PER_DEPARTMENT
            for users in users_by_department.values()
        ))
        self.assertEqual(dict(mix), {
            kind: expected_users * count
            for kind, count in PERSONAL_DEVICE_REQUESTS_PER_USER.items()
        })
        evidence = state["summary"]["dataset"]["checks"]["personal_device_activity"]
        self.assertEqual(evidence["events"], len(personal))
        self.assertEqual(evidence["users"], expected_users)
        self.assertEqual(evidence["public_or_internal_events"], mix["public_or_internal"])
        self.assertEqual(evidence["confidential_read_events"], mix["confidential_read"])
        for split in ["train", "calibration", "test"]:
            labels = {
                row["ground_truth"] for row in state["events"]
                if row["device_managed"] == "false"
                and state["feature_by_id"][row["event_id"]]["split"] == split
            }
            self.assertEqual(labels, {"normal", "anomalous"})

    def test_personal_devices_have_stable_prior_only_history_in_all_splits(self):
        state = snapshot()
        by_user = defaultdict(list)
        for row in state["events"]:
            if row["device_id"].startswith("PERSONAL-"):
                by_user[row["user_id"]].append(row)
        for user_id, events in by_user.items():
            self.assertEqual(
                {event["device_id"] for event in events}, {f"PERSONAL-{user_id}"}
            )
            self.assertEqual(len(events), sum(PERSONAL_DEVICE_REQUESTS_PER_USER.values()))
            observed = Counter()
            for event in events:
                feature = state["feature_by_id"][event["event_id"]]
                kind = (
                    "public_or_internal"
                    if event["resource_sensitivity"] in {"public", "internal"}
                    else "confidential_read"
                )
                observed[(kind, feature["split"])] += 1
                earlier = sum(other["timestamp"] < event["timestamp"] for other in events)
                self.assertEqual(feature["device_prior_count"], earlier)
                self.assertEqual(feature["device_seen_before"], int(earlier > 0))
            for kind, count in PERSONAL_DEVICE_REQUESTS_PER_USER.items():
                self.assertEqual(sum(
                    observed[(kind, split)] for split in ["train", "calibration", "test"]
                ), count)
                for split in ["train", "calibration", "test"]:
                    self.assertGreater(observed[(kind, split)], 0)

    def test_personal_conversion_preserves_other_fields_and_attack_records(self):
        state = snapshot()
        normal = dataset.generate_normal_events(state["enterprise"])
        anomalies = dataset.generate_anomalies(normal, state["enterprise"])
        current = {row["event_id"]: row for row in state["events"]}
        converted = 0
        for raw in normal + anomalies:
            event_id = "evt_" + hashlib.sha256(
                f"{dataset.SEED}:{raw['event_id']}".encode("utf-8")
            ).hexdigest()[:16]
            original = dict(raw, event_id=event_id)
            actual = current[event_id]
            changed = {field for field in dataset.EVENT_FIELDS if actual[field] != original[field]}
            if actual["device_id"].startswith("PERSONAL-"):
                self.assertEqual(original["ground_truth"], "normal")
                self.assertEqual(changed, {"device_id", "device_managed"})
                converted += 1
            else:
                self.assertEqual(changed, set())
        self.assertEqual(converted, len(DEPARTMENT_SPECS)
                         * PERSONAL_DEVICE_USERS_PER_DEPARTMENT
                         * sum(PERSONAL_DEVICE_REQUESTS_PER_USER.values()))
        forward = [dict(row) for row in normal]
        backward = [dict(row) for row in reversed(normal)]
        dataset.add_personal_device_activity(forward, state["enterprise"])
        dataset.add_personal_device_activity(backward, state["enterprise"])
        self.assertEqual(
            {row["event_id"]: row for row in forward},
            {row["event_id"]: row for row in backward},
        )

    def test_anomaly_feature_invariants(self):
        state = snapshot()
        categories = defaultdict(list)
        for event in state["events"]:
            if event["ground_truth"] == "anomalous":
                categories[event["anomaly_type"]].append(state["feature_by_id"][event["event_id"]])
        self.assertTrue(all(row["outside_role_hours"] == 1 for row in categories["off_hours_privileged_access"]))
        self.assertTrue(all(row["device_seen_before"] == 0 for row in categories["first_seen_device_sensitive_action"]))
        self.assertTrue(all(row["role_resource_allowed"] == 0 for row in categories["role_resource_mismatch"]))
        travel = categories["impossible_travel"]
        self.assertGreater(sum(row["location_change_60m"] == 1 for row in travel) / len(travel), 0.9)
        bulk = categories["bulk_download_spike"]
        self.assertGreater(sum(row["user_events_1h"] >= 5 for row in bulk) / len(bulk), 0.5)

    def test_anomaly_campaigns_do_not_cross_splits(self):
        state = snapshot()
        campaigns = defaultdict(set)
        for event in state["events"]:
            if event["anomaly_campaign_id"]:
                campaigns[event["anomaly_campaign_id"]].add(
                    state["feature_by_id"][event["event_id"]]["split"]
                )
        self.assertTrue(all(len(splits) == 1 for splits in campaigns.values()))

    def test_future_events_do_not_change_earlier_features(self):
        state = snapshot()
        ordered = sorted(state["events"], key=lambda row: (row["timestamp"], row["event_id"]))
        prefix = features.compute_features(ordered[:400], state["resources"])
        extended = features.compute_features(ordered[:400] + ordered[-50:], state["resources"])
        for before, after in zip(prefix, extended):
            self.assertEqual(before["event_id"], after["event_id"])
            for field in MODEL_FEATURES:
                self.assertEqual(before[field], after[field])


class PolicyAndModelTests(unittest.TestCase):
    def test_both_policies_cover_the_same_requests_in_both_scenarios(self):
        state = snapshot()
        test_ids = {
            row["event_id"] for row in state["feature_by_id"].values() if row["split"] == "test"
        }
        self.assertEqual({row["event_id"] for row in state["original"]}, test_ids)
        self.assertEqual({row["source_event_id"] for row in state["changed"]}, test_ids)
        for records in [state["original"], state["changed"]]:
            self.assertEqual(len(records), len(test_ids))
            self.assertEqual(
                {row["baseline_decision"] for row in records}, {"allow", "challenge", "block"}
            )

    def test_evaluation_scores_are_bounded_and_complete(self):
        state = snapshot()
        frame = state["frame"][state["frame"]["split"] == "test"]
        self.assertEqual(len(frame), len(state["original"]))
        expected = dict(zip(frame["event_id"], risk_model.score_requests(state["model"], frame)))
        for row in state["original"]:
            score = float(row["risk_score"])
            self.assertTrue(0 <= score <= 100)
            self.assertEqual(score, expected[row["event_id"]])

    def test_decisions_have_three_outcomes_and_required_fields(self):
        original = snapshot()["original"]
        self.assertEqual({row["final_decision"] for row in original}, {"allow", "challenge", "block"})
        for field in [
            "event_id", "risk_score", "baseline_decision", "baseline_reason_codes",
            "final_decision", "step_up_control", "reason_codes",
            "top_model_contributors", "explanation",
        ]:
            self.assertTrue(all(row[field] for row in original), field)

    def test_at_least_five_event_explanations(self):
        block = snapshot()["summary"]["evaluation"]
        examples = block["examples"]
        anomalous = [row for row in examples if row["anomaly_type"] != "none"]
        self.assertGreaterEqual(len(anomalous), 5)
        self.assertTrue(all(row["top_model_contributors"] for row in examples))
        self.assertTrue(
            all(row["final_decision"] in {"challenge", "block"} for row in examples),
            "Task 6 asks for high or medium risk cases, so examples exclude allows.",
        )
        detected = [row for row in examples if row["case_role"].startswith("detected_")]
        medium = [row for row in examples if row["case_role"].startswith("medium_risk_")]
        self.assertEqual(len(detected) + len(medium), len(examples))
        self.assertEqual(
            len({row["case_role"] for row in examples}), len(examples),
            "Each exported case must occupy exactly one selection slot.",
        )
        self.assertEqual(
            {row["model_decision"] for row in medium}, {"challenge"},
            "Medium-risk cases must come from the challenge band.",
        )
        self.assertEqual({row["model_decision"] for row in detected}, {"block"})
        error_cases = block["error_cases"]
        self.assertEqual(
            {row["case_role"] for row in error_cases},
            {"highest_scoring_false_positive", "lowest_scoring_false_negative"},
        )
        by_role = {row["case_role"]: row for row in error_cases}
        self.assertIn(
            by_role["highest_scoring_false_positive"]["final_decision"], {"challenge", "block"}
        )
        self.assertEqual(by_role["lowest_scoring_false_negative"]["final_decision"], "allow")

    def test_case_selection_follows_the_documented_rules(self):
        state = snapshot()
        original = state["original"]
        block = state["summary"]["evaluation"]
        thresholds = state["model"]["thresholds"]
        band = {
            row["event_id"]: policy.model_band(
                float(row["risk_score"]),
                thresholds["challenge_threshold"], thresholds["block_threshold"],
            )
            for row in original
        }

        def best(pool, descending):
            return min(
                pool,
                key=lambda item: (
                    -float(item["risk_score"]) if descending else float(item["risk_score"]),
                    item["event_id"],
                ),
            )

        intervened = [row for row in original if row["final_decision"] in {"challenge", "block"}]
        by_type = defaultdict(list)
        for row in intervened:
            by_type[row["anomaly_type"]].append(row)
        detected = [row for row in block["examples"] if row["case_role"].startswith("detected_")]
        self.assertEqual(
            {row["case_role"].removeprefix("detected_") for row in detected},
            {
                category for category, rows in by_type.items()
                if any(row["ground_truth"] == "anomalous" for row in rows)
            },
        )
        for row in detected:
            self.assertEqual(row["event_id"], best(by_type[row["anomaly_type"]], True)["event_id"])
        challenge_anomalies = defaultdict(list)
        for row in original:
            if band[row["event_id"]] == "challenge" and row["ground_truth"] == "anomalous":
                challenge_anomalies[row["anomaly_type"]].append(row)
        self.assertGreater(len(challenge_anomalies), 0)
        medium = [row for row in block["examples"] if row["case_role"].startswith("medium_risk_")]
        self.assertEqual(
            {row["case_role"].removeprefix("medium_risk_") for row in medium},
            set(challenge_anomalies),
        )
        for row in medium:
            self.assertEqual(
                row["event_id"], best(challenge_anomalies[row["anomaly_type"]], True)["event_id"]
            )
        truth = {row["event_id"]: row["ground_truth"] for row in state["events"]}
        false_positives = [row for row in intervened if row["ground_truth"] == "normal"]
        self.assertGreater(len(false_positives), 0)
        worst_fp = next(
            row for row in block["error_cases"]
            if row["case_role"] == "highest_scoring_false_positive"
        )
        self.assertEqual(worst_fp["event_id"], best(false_positives, True)["event_id"])
        self.assertEqual(truth[worst_fp["event_id"]], "normal")
        missed = [
            row for row in original
            if row["ground_truth"] == "anomalous" and row["final_decision"] == "allow"
        ]
        self.assertGreater(len(missed), 0)
        worst_fn = next(
            row for row in block["error_cases"]
            if row["case_role"] == "lowest_scoring_false_negative"
        )
        self.assertEqual(worst_fn["event_id"], best(missed, False)["event_id"])
        self.assertEqual(truth[worst_fn["event_id"]], "anomalous")

    def test_case_selection_is_stable_under_input_reordering(self):
        state = snapshot()
        rows = state["original"]
        thresholds = state["model"]["thresholds"]
        block = state["summary"]["evaluation"]

        def selection(order):
            frame = pd.DataFrame([rows[index] for index in order]).reset_index(drop=True)
            frame["risk_score"] = frame["risk_score"].astype(float)
            frame["actual"] = (frame["ground_truth"] == "anomalous").astype(int)
            band = np.array([
                policy.model_band(
                    score, thresholds["challenge_threshold"], thresholds["block_threshold"]
                )
                for score in frame["risk_score"]
            ])
            examples = evaluation.select_examples(frame, band)
            errors = evaluation.select_error_cases(frame)
            return (
                list(zip(examples["case_role"], examples["event_id"])),
                list(zip(errors["case_role"], errors["event_id"])),
            )

        reference = selection(range(len(rows)))
        self.assertEqual(
            reference[0], [(row["case_role"], row["event_id"]) for row in block["examples"]]
        )
        self.assertEqual(
            reference[1], [(row["case_role"], row["event_id"]) for row in block["error_cases"]]
        )
        for seed in (1, 2, 3):
            order = list(range(len(rows)))
            np.random.default_rng(seed).shuffle(order)
            self.assertEqual(selection(order), reference, f"input order seed {seed}")

    def test_metrics_use_the_same_evaluation_support(self):
        state = snapshot()
        metrics = state["summary"]["evaluation"]
        self.assertEqual(metrics["evaluation_set"]["events"], len(state["original"]))
        self.assertIn("pr_auc", metrics["ai_assisted"])
        self.assertNotIn("accuracy", metrics["ai_assisted"])
        self.assertNotIn("roc_auc", metrics["ai_assisted"])

    def test_evaluation_ids_are_unique_and_match_the_dataset(self):
        state = snapshot()
        ids = [row["event_id"] for row in state["original"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(ids).issubset({row["event_id"] for row in state["events"]}))
        for row in state["original"]:
            self.assertEqual(row["source_event_id"], row["event_id"])

    def test_baseline_is_not_a_perfect_comparator(self):
        metrics = snapshot()["summary"]["evaluation"]["static_baseline"]
        self.assertLess(metrics["precision"], 1)
        self.assertLess(metrics["recall"], 1)

    def test_department_breakdown_covers_the_required_group_dimension(self):
        state = snapshot()
        groups = state["summary"]["evaluation"]["departments"]
        self.assertEqual({row["group_dimension"] for row in groups}, {"department"})
        self.assertEqual(
            {row["group"] for row in groups},
            {row["department"] for row in state["original"]},
        )

    def test_drift_changes_context_without_requiring_a_particular_result(self):
        drift = snapshot()["summary"]["drift"]
        rows = [row for row in drift["comparison"] if row["system"] == "ai_assisted"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["events"], rows[1]["events"])
        self.assertEqual(rows[0]["anomalies"], rows[1]["anomalies"])
        self.assertGreater(drift["rollout"]["managed_requests_changed"], 0)
        self.assertFalse(drift["adaptation_evaluated"])

    def test_static_drift_decisions_and_metrics_recompute(self):
        state = snapshot()
        volume = state["summary"]["baseline"]["threshold_selection"]["volume_threshold_mb"]
        tp = fp = fn = 0
        for row in state["changed"]:
            decision, reasons = policy.evaluate_event(
                row, state["resources"][row["resource_id"]]["resource_group"],
                state["combined_by_id"][row["event_id"]], volume,
            )
            self.assertEqual(row["baseline_decision"], decision)
            self.assertEqual(row["baseline_reason_codes"], "|".join(reasons))
            actual, predicted = row["ground_truth"] == "anomalous", decision != "allow"
            tp += actual and predicted
            fp += not actual and predicted
            fn += actual and not predicted
        reported = next(
            row for row in state["summary"]["drift"]["comparison"]
            if row["system"] == "static_baseline" and row["scenario"] == evaluation.DRIFT_SCENARIO
        )
        self.assertAlmostEqual(reported["precision"], tp / (tp + fp), places=4)
        self.assertAlmostEqual(reported["recall"], tp / (tp + fn), places=4)
        self.assertAlmostEqual(reported["f1"], 2 * tp / (2 * tp + fp + fn), places=4)


class ContextAndOutputTests(unittest.TestCase):
    def test_policy_boundaries_failsafe_and_context_controls(self):
        base = {"mandatory_block_codes": [], "resource_sensitivity": "internal", "device_managed": True}
        for score, expected in [(5, "allow"), (15, "challenge"), (69.99, "challenge"), (70, "block")]:
            self.assertEqual(policy.decide(score, base, 15, 70)["final_decision"], expected)
        for invalid in [None, float("nan"), -1, 101, "not-a-number"]:
            result = policy.decide(invalid, base, 15, 70)
            self.assertEqual(result["final_decision"], "challenge")
            self.assertEqual(result["decision_source"], "fail_safe")
        mandatory = dict(base, mandatory_block_codes=["BLOCK_NON_IT_IDENTITY_CHANGE"])
        for score in [0, None]:
            self.assertEqual(policy.decide(score, mandatory, 15, 70)["final_decision"], "block")
        privileged = dict(base, resource_sensitivity="privileged")
        self.assertIn("manager approval", policy.decide(40, privileged, 15, 70)["step_up_control"])
        unmanaged = dict(base, resource_sensitivity="confidential", device_managed=False)
        result = policy.decide(0, unmanaged, 15, 70)
        self.assertEqual(result["final_decision"], "challenge")
        self.assertEqual(result["decision_source"], "contextual_override")

    def test_mandatory_rules_always_block(self):
        state = snapshot()
        seen = 0
        for row in state["records"]:
            codes = policy.context_for_event(row, state["resources"])["mandatory_block_codes"]
            if codes:
                seen += 1
                self.assertEqual(row["final_decision"], "block")
                self.assertIn("MANDATORY_STATIC_BLOCK", row["override_codes"])
        self.assertGreater(seen, 0)

    def test_decision_sources_are_valid_and_inspectable(self):
        records = snapshot()["original"]
        self.assertTrue({row["decision_source"] for row in records}.issubset(policy.DECISION_SOURCES))
        self.assertIn("contextual_override", {row["decision_source"] for row in records})
        for field in ["model_decision", "decision_source", "override_codes"]:
            self.assertTrue(all(row[field] for row in records))

    def test_mcnemar_exact_p_matches_the_binomial_tail(self):
        self.assertIsNone(evaluation.mcnemar_exact_p(0, 0))
        for first, second, expected in [(4, 0, 0.125), (15, 7, 0.1338), (10, 10, 1.0), (3, 9, 0.146)]:
            self.assertEqual(evaluation.mcnemar_exact_p(first, second), expected)
        self.assertEqual(evaluation.mcnemar_exact_p(7, 15), evaluation.mcnemar_exact_p(15, 7))
        self.assertLess(evaluation.mcnemar_exact_p(30, 0), 1e-4)

    def test_score_resolution_matches_the_exported_scores(self):
        state = snapshot()
        resolution = state["summary"]["evaluation"]["score_resolution"]
        scores = [float(row["risk_score"]) for row in state["original"]]
        self.assertEqual(resolution["evaluation_events"], len(scores))
        self.assertEqual(resolution["evaluation_distinct_scores"], len(set(scores)))
        thresholds = state["model"]["thresholds"]
        counts = Counter(
            policy.model_band(score, thresholds["challenge_threshold"], thresholds["block_threshold"])
            for score in scores
        )
        occupancy = resolution["band_occupancy"]
        self.assertEqual(set(occupancy), set(evaluation.DECISION_BANDS))
        for band, row in occupancy.items():
            self.assertEqual(row["events"], counts[band], band)
            self.assertLessEqual(row["anomalies"], row["events"])
        self.assertEqual(sum(row["events"] for row in occupancy.values()), len(scores))
        self.assertEqual(
            state["summary"]["model"]["distinct_calibration_scores"],
            state["model"]["metadata"]["distinct_calibration_scores"],
        )

    def test_threshold_ties_recompute_on_the_calibration_partition(self):
        state = snapshot()
        model = state["model"]
        calibration = state["frame"][state["frame"]["split"] == "calibration"]
        raw = model["pipeline"].predict_proba(calibration[MODEL_FEATURES])[:, 1]
        scores = model["calibrator"].predict(raw) * 100
        labels = (calibration["ground_truth"] == "anomalous").astype(int).to_numpy()
        rerun = risk_model.choose_thresholds(labels, scores)
        self.assertEqual(rerun, model["thresholds"])
        selected = state["summary"]["model"]["threshold_selection"]
        for key in ("grid_pairs_evaluated", "pairs_at_minimum_cost",
                    "pairs_selecting_the_identical_partition", "identical_partition_ranges"):
            self.assertEqual(selected[key], rerun[key], key)
        chosen = tuple(
            policy.model_band(score, rerun["challenge_threshold"], rerun["block_threshold"])
            for score in scores
        )
        identical = sum(
            1
            for challenge in range(10, 61, 5)
            for block in range(max(challenge + 10, 50), 96, 5)
            if tuple(
                policy.model_band(score, float(challenge), float(block)) for score in scores
            ) == chosen
        )
        self.assertEqual(identical, rerun["pairs_selecting_the_identical_partition"])
        self.assertGreaterEqual(
            rerun["pairs_at_minimum_cost"], rerun["pairs_selecting_the_identical_partition"]
        )
        self.assertGreaterEqual(
            rerun["grid_pairs_evaluated"], rerun["pairs_at_minimum_cost"]
        )
        for key, name in [("challenge", "challenge_threshold"), ("block", "block_threshold")]:
            low, high = rerun["identical_partition_ranges"][key]
            self.assertLessEqual(low, rerun[name])
            self.assertGreaterEqual(high, rerun[name])

    def test_baseline_threshold_ignores_evaluation_labels(self):
        state = snapshot()
        changed = []
        for event in state["events"]:
            copy = dict(event)
            if state["feature_by_id"][event["event_id"]]["split"] == "test":
                copy["ground_truth"] = "normal" if event["ground_truth"] == "anomalous" else "anomalous"
            changed.append(copy)
        original = policy.select_volume_threshold(state["events"], state["resources"], state["feature_by_id"])
        flipped = policy.select_volume_threshold(changed, state["resources"], state["feature_by_id"])
        self.assertEqual(original, flipped)
        selection = state["summary"]["baseline"]["threshold_selection"]
        self.assertEqual(original["volume_threshold_mb"], selection["volume_threshold_mb"])
        self.assertFalse(selection["evaluation_days_used"])

    def test_risk_model_training_ignores_evaluation_labels(self):
        state = snapshot()
        changed = state["frame"].copy()
        mask = changed["split"] == "test"
        changed.loc[mask, "ground_truth"] = changed.loc[mask, "ground_truth"].map(
            {"normal": "anomalous", "anomalous": "normal"}
        )
        retrained = risk_model.train_model(changed)
        self.assertEqual(retrained["thresholds"], state["model"]["thresholds"])
        np.testing.assert_allclose(
            retrained["pipeline"].named_steps["classifier"].coef_,
            state["model"]["pipeline"].named_steps["classifier"].coef_, rtol=0, atol=1e-12,
        )

    def test_both_headline_metrics_recompute_from_the_combined_csv(self):
        state = snapshot()
        actual = [row["ground_truth"] == "anomalous" for row in state["original"]]
        for system, column in [("ai_assisted", "final_decision"), ("static_baseline", "baseline_decision")]:
            predicted = [row[column] != "allow" for row in state["original"]]
            tp = sum(a and p for a, p in zip(actual, predicted))
            fp = sum(not a and p for a, p in zip(actual, predicted))
            fn = sum(a and not p for a, p in zip(actual, predicted))
            reported = state["summary"]["evaluation"][system]
            self.assertAlmostEqual(reported["precision"], tp / (tp + fp), places=4)
            self.assertAlmostEqual(reported["recall"], tp / (tp + fn), places=4)
            self.assertAlmostEqual(reported["f1"], 2 * tp / (2 * tp + fp + fn), places=4)
        ai_flag = [row["final_decision"] != "allow" for row in state["original"]]
        base_flag = [row["baseline_decision"] != "allow" for row in state["original"]]
        ai_only = [a and not b for a, b in zip(ai_flag, base_flag)]
        base_only = [b and not a for a, b in zip(ai_flag, base_flag)]
        pairs = state["summary"]["evaluation"]["comparison_uncertainty"]["discordant_pairs"]
        self.assertEqual(pairs["ai_intervened_baseline_allowed"], sum(ai_only))
        self.assertEqual(pairs["baseline_intervened_ai_allowed"], sum(base_only))
        self.assertEqual(
            pairs["ai_correct_baseline_wrong"],
            sum(flag and truth for flag, truth in zip(ai_only, actual))
            + sum(flag and not truth for flag, truth in zip(base_only, actual)),
        )
        self.assertEqual(
            pairs["baseline_correct_ai_wrong"],
            sum(flag and not truth for flag, truth in zip(ai_only, actual))
            + sum(flag and truth for flag, truth in zip(base_only, actual)),
        )
        self.assertEqual(
            pairs["ai_correct_baseline_wrong"] + pairs["baseline_correct_ai_wrong"],
            pairs["ai_intervened_baseline_allowed"] + pairs["baseline_intervened_ai_allowed"],
        )
        ablation = {
            row["variant"]: row
            for row in state["summary"]["evaluation"]["component_ablation"]["variants"]
        }
        self.assertEqual(
            set(ablation),
            {
                "score_band_only", "full_ai_policy",
                "mandatory_and_contextual_controls_only", "static_baseline",
            },
        )
        for variant, key in [("full_ai_policy", "ai_assisted"), ("static_baseline", "static_baseline")]:
            self.assertEqual(
                ablation[variant]["confusion_matrix"],
                state["summary"]["evaluation"][key]["confusion_matrix"],
                variant,
            )
        controls = state["summary"]["evaluation"]["component_ablation"]
        self.assertEqual(
            controls["mandatory_block_events"]["events"]
            + controls["contextual_override_events"]["events"],
            sum(
                row["interventions"]
                for row in controls["variants"]
                if row["variant"] == "mandatory_and_contextual_controls_only"
            ),
        )

    def test_undefined_metrics_and_descriptive_intervals(self):
        undefined = evaluation.binary_metrics([0, 0], [0, 0])
        self.assertIsNone(undefined["recall"])
        self.assertIsNone(undefined["precision"])
        self.assertEqual(evaluation.wilson_interval(0, 0), (None, None))
        for row in snapshot()["summary"]["evaluation"]["departments"]:
            for metric in ["ai_recall", "ai_false_positive_rate", "block_rate"]:
                self.assertLessEqual(row[f"{metric}_ci_low"], row[metric] + 1e-9)
                self.assertGreaterEqual(row[f"{metric}_ci_high"], row[metric] - 1e-9)

    def test_explanations_match_the_recomputed_feature_values(self):
        state = snapshot()
        block = state["summary"]["evaluation"]
        for example in [*block["examples"], *block["error_cases"]]:
            reference = state["feature_by_id"][example["event_id"]]
            for field in [
                "outside_role_hours", "device_seen_before", "role_resource_allowed",
                "location_change_60m", "user_events_1h", "user_volume_1h_mb",
            ]:
                self.assertAlmostEqual(float(example[field]), float(reference[field]), places=8)

    def test_drift_contains_only_the_required_frozen_comparison(self):
        rows = snapshot()["summary"]["drift"]["comparison"]
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["system"] for row in rows}, {"ai_assisted", "static_baseline"})
        self.assertEqual(
            {row["scenario"] for row in rows}, {evaluation.PRE_SCENARIO, evaluation.DRIFT_SCENARIO}
        )

    def test_six_output_contract_documentation_and_validation(self):
        expected = {
            "data/access_events.csv", "results/decisions.csv", "results/results.json",
            "results/plots/baseline-vs-ai.png",
            "results/plots/drift-intervention-rate.png",
            "results/plots/risk-score-distribution.png",
        }
        actual = {
            path.relative_to(ROOT).as_posix()
            for directory in [DATA_DIR, RESULTS_DIR] for path in directory.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual, expected)
        self.assertEqual(snapshot()["summary"]["dataset"]["status"], "PASS")
        self.assertEqual(
            snapshot()["summary"]["dataset"]["checks"]["event_count"],
            len(snapshot()["events"]),
        )
        for name in ["README.md"]:
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn(".pplx.md", text)
            self.assertNotIn("amazonaws.com", text)
        self.assertNotIn("--verify", (ROOT / "README.md").read_text(encoding="utf-8"))


class TimestampBatchTests(unittest.TestCase):
    def event(self, event_id, timestamp, **changes):
        event = {
            "event_id": event_id, "timestamp": timestamp, "user_id": "USR-TEST",
            "role": "software_engineer", "department": "Engineering",
            "resource_id": "RES-003", "resource_sensitivity": "internal",
            "ground_truth": "normal", "anomaly_type": "none", "device_id": "DEVICE-OLD",
            "device_managed": "true", "location": "Athens", "network_zone": "athens_office",
            "action": "read", "outcome": "success", "data_volume_mb": "2.0",
        }
        event.update(changes)
        return event

    def compute(self, events):
        return features.compute_features(events, {"RES-003": {"resource_group": "common"}})

    def tied_events(self):
        return [
            self.event("prior", "2026-08-03T08:59:00Z", outcome="denied"),
            self.event("tie_a", "2026-08-03T09:00:00Z", device_id="DEVICE-NEW",
                       location="London", outcome="denied", data_volume_mb="30"),
            self.event("tie_b", "2026-08-03T09:00:00Z", device_id="DEVICE-NEW",
                       location="London", data_volume_mb="70"),
        ]

    def test_simultaneous_requests_do_not_observe_each_other(self):
        rows = {row["event_id"]: row for row in self.compute(
            self.tied_events() + [self.event("peer", "2026-08-03T09:00:00Z", user_id="USR-PEER")]
        )}
        expected = {
            "user_events_1h": 1, "user_volume_1h_mb": 2.0, "recent_failures_24h": 1,
            "device_seen_before": 0, "device_prior_count": 0, "user_resource_prior_count": 1,
            "location_seen_before": 0, "minutes_since_previous_event": 1.0, "location_change_60m": 1,
        }
        for event_id in ["tie_a", "tie_b"]:
            for field, value in expected.items():
                self.assertEqual(rows[event_id][field], value, field)
        for event_id in ["tie_a", "tie_b", "peer"]:
            self.assertEqual(rows[event_id]["role_resource_prior_count"], 1)

    def test_later_requests_observe_the_whole_completed_batch(self):
        after = self.compute(self.tied_events() + [
            self.event("after", "2026-08-03T09:02:00Z", device_id="DEVICE-NEW", location="London")
        ])[-1]
        expected = {
            "user_events_1h": 3, "user_volume_1h_mb": 102.0, "recent_failures_24h": 2,
            "device_prior_count": 2, "user_resource_prior_count": 3,
            "role_resource_prior_count": 3, "minutes_since_previous_event": 2.0,
            "location_change_60m": 0,
        }
        for field, value in expected.items():
            self.assertEqual(after[field], value, field)

    def test_ids_do_not_choose_a_last_simultaneous_location(self):
        events = [
            self.event("prior", "2026-08-03T08:59:00Z"),
            self.event("tie_a", "2026-08-03T09:00:00Z", location="Athens"),
            self.event("tie_b", "2026-08-03T09:00:00Z", location="London"),
            self.event("after", "2026-08-03T09:01:00Z", location="London"),
            self.event("new_location", "2026-08-03T09:02:00Z", location="Paris"),
        ]
        original = {row["event_id"]: row for row in self.compute(events)}
        swaps = {"tie_a": "tie_b", "tie_b": "tie_a"}
        renamed = [
            dict(event, event_id=swaps.get(event["event_id"], event["event_id"]))
            for event in reversed(events)
        ]
        for row in self.compute(renamed):
            reference = original[swaps.get(row["event_id"], row["event_id"])]
            for field in MODEL_FEATURES:
                self.assertEqual(row[field], reference[field], field)
        self.assertEqual(original["after"]["location_change_60m"], 0)
        self.assertEqual(original["new_location"]["location_change_60m"], 1)

    def test_history_window_boundaries(self):
        rows = self.compute([
            self.event("old_failure", "2026-08-03T09:59:59Z", outcome="denied"),
            self.event("boundary_failure", "2026-08-03T10:00:00Z", outcome="denied"),
            self.event("old_hour", "2026-08-04T08:59:59Z", data_volume_mb="100"),
            self.event("boundary_hour", "2026-08-04T09:00:00Z", data_volume_mb="7"),
            self.event("current_a", "2026-08-04T10:00:00Z", outcome="denied"),
            self.event("current_b", "2026-08-04T10:00:00Z"),
        ])
        for row in rows[-2:]:
            self.assertEqual(row["user_events_1h"], 1)
            self.assertEqual(row["user_volume_1h_mb"], 7.0)
            self.assertEqual(row["recent_failures_24h"], 1)


class DriftRegressionTests(unittest.TestCase):
    def test_rollout_selection_and_changes_ignore_labels(self):
        state = snapshot()
        cohort = evaluation.select_rollout_users(state["events"])
        flipped = [
            dict(event, ground_truth="normal" if event["ground_truth"] == "anomalous" else "anomalous",
                 anomaly_type="none", anomaly_campaign_id="")
            for event in reversed(state["events"])
        ]
        self.assertEqual(cohort, evaluation.select_rollout_users(flipped))
        flipped_test = [
            event for event in flipped
            if features.assign_split(dataset.parse_utc(event["timestamp"])) == "test"
        ]
        changed = {row["event_id"]: row for row in evaluation.build_drift_events(flipped_test, cohort)}
        labels = {"ground_truth", "anomaly_type", "anomaly_campaign_id"}
        for row in state["drift_events"]:
            for field in row.keys() - labels:
                self.assertEqual(row[field], changed[row["event_id"]][field], field)

    def test_rollout_devices_are_fresh_and_begin_with_zero_history(self):
        state = snapshot()
        old_devices = {row["device_id"] for row in state["events"]}
        rollout = [row for row in state["drift_events"] if row["device_id"].startswith("DRIFT-ROLLOUT-")]
        self.assertTrue(old_devices.isdisjoint({row["device_id"] for row in rollout}))
        self.assertEqual(
            {row["user_id"] for row in rollout},
            set(state["summary"]["drift"]["rollout"]["user_ids"]),
        )
        first = {}
        for row in sorted(rollout, key=lambda row: row["timestamp"]):
            first.setdefault(row["device_id"], row["timestamp"])
        for row in rollout:
            self.assertEqual(row["device_managed"], "true")
            if row["timestamp"] == first[row["device_id"]]:
                feature = state["combined_by_id"][row["event_id"]]
                self.assertEqual(feature["device_seen_before"], 0)
                self.assertEqual(feature["device_prior_count"], 0)
        self.assertTrue(any(row["ground_truth"] == "normal" for row in rollout))
        self.assertTrue(any(row["ground_truth"] == "anomalous" for row in rollout))

    def test_first_seen_attacks_remain_fresh_and_unmanaged(self):
        state = snapshot()
        attacks = [
            row for row in state["drift_events"]
            if row["anomaly_type"] == "first_seen_device_sensitive_action"
        ]
        self.assertGreater(len(attacks), 0)
        old_devices = {row["device_id"] for row in state["events"]}
        for row in attacks:
            self.assertNotIn(row["device_id"], old_devices)
            self.assertEqual(row["device_managed"], "false")
            self.assertIn(row["resource_sensitivity"], {"confidential", "privileged"})
            feature = state["combined_by_id"][row["event_id"]]
            self.assertEqual(feature["device_seen_before"], 0)
            self.assertEqual(feature["device_prior_count"], 0)

    def test_replay_preserves_security_attributes_and_paired_labels(self):
        state = snapshot()
        original = {row["event_id"]: row for row in state["events"]}
        changed_fields = {"event_id", "timestamp", "device_id", "anomaly_campaign_id"}
        for row in state["changed"]:
            source = original[row["source_event_id"]]
            for field in set(dataset.EVENT_FIELDS) - changed_fields:
                self.assertEqual(row[field], source[field], field)
            self.assertEqual(
                dataset.parse_utc(row["timestamp"]) - dataset.parse_utc(source["timestamp"]),
                timedelta(days=evaluation.REPLAY_OFFSET_DAYS),
            )
            expected_campaign = (
                f"drift_{source['anomaly_campaign_id']}" if source["anomaly_campaign_id"] else ""
            )
            self.assertEqual(row["anomaly_campaign_id"], expected_campaign)

    def test_contextual_attack_invariants_survive_replay(self):
        state = snapshot()
        categories = defaultdict(list)
        for row in state["drift_events"]:
            categories[row["anomaly_type"]].append(state["combined_by_id"][row["event_id"]])
        self.assertTrue(all(row["outside_role_hours"] == 1 for row in categories["off_hours_privileged_access"]))
        self.assertTrue(all(row["role_resource_allowed"] == 0 for row in categories["role_resource_mismatch"]))
        travel = categories["impossible_travel"]
        self.assertGreater(sum(row["location_change_60m"] == 1 for row in travel) / len(travel), 0.9)
        bulk = categories["bulk_download_spike"]
        self.assertGreater(sum(row["user_events_1h"] >= 5 for row in bulk) / len(bulk), 0.5)

    def test_drift_scores_decisions_metrics_and_frozen_parameters(self):
        state = snapshot()
        model = state["model"]
        coefficients = model["pipeline"].named_steps["classifier"].coef_.copy()
        thresholds = dict(model["thresholds"])
        rows = state["changed"]
        frame = pd.DataFrame([state["combined_by_id"][row["event_id"]] for row in rows])
        expected = risk_model.score_requests(model, frame)
        saved = np.asarray([float(row["risk_score"]) for row in rows])
        np.testing.assert_allclose(saved, expected, rtol=0, atol=1e-9)
        for row in rows:
            result = policy.decide(
                float(row["risk_score"]), policy.context_for_event(row, state["resources"]),
                thresholds["challenge_threshold"], thresholds["block_threshold"],
            )
            self.assertEqual(row["final_decision"], result["final_decision"])
        numeric_records = [dict(row, risk_score=float(row["risk_score"])) for row in rows]
        recomputed = evaluation.metric_row("ai_assisted", evaluation.DRIFT_SCENARIO, numeric_records)
        reported = next(
            row for row in state["summary"]["drift"]["comparison"]
            if row["system"] == "ai_assisted" and row["scenario"] == evaluation.DRIFT_SCENARIO
        )
        self.assertEqual(recomputed, reported)
        self.assertEqual(model["thresholds"], thresholds)
        np.testing.assert_array_equal(model["pipeline"].named_steps["classifier"].coef_, coefficients)

    def test_drift_replay_effect_is_measured(self):
        state = snapshot()
        effect = state["summary"]["drift"]["replay_effect"]
        self.assertEqual(effect["pairs"], len(state["changed"]))
        self.assertEqual(effect["rule_relevant_fields_changed"], {})
        self.assertEqual(effect["baseline_decisions_changed"], 0)
        self.assertEqual(effect["baseline_reason_codes_changed"], 0)
        self.assertEqual(effect["fields_changed"]["timestamp"], len(state["changed"]))
        self.assertEqual(effect["fields_changed"]["event_id"], len(state["changed"]))
        self.assertGreater(effect["fields_changed"]["device_id"], 0)
        self.assertEqual(
            effect["fields_changed"]["device_id"],
            state["summary"]["drift"]["rollout"]["managed_requests_changed"]
            + sum(
                1 for row in state["changed"]
                if row["device_id"].startswith("DRIFT-UNMANAGED-")
            ),
        )
        rows = [
            row for row in state["summary"]["drift"]["comparison"]
            if row["system"] == "static_baseline"
        ]
        self.assertEqual(
            {key: value for key, value in rows[0].items() if key != "scenario"},
            {key: value for key, value in rows[1].items() if key != "scenario"},
        )
        ai_rows = [
            row for row in state["summary"]["drift"]["comparison"]
            if row["system"] == "ai_assisted"
        ]
        self.assertNotEqual(ai_rows[0]["confusion_matrix"], ai_rows[1]["confusion_matrix"])


class RegularizationSelectionTests(unittest.TestCase):
    @staticmethod
    def constant_pipeline():
        pipeline = Mock()
        pipeline.predict_proba.side_effect = lambda rows: np.full((len(rows), 2), 0.5)
        return pipeline

    def test_recorded_selection_matches_training_windows_and_final_classifier(self):
        state = snapshot()
        selection = state["model"]["metadata"]["regularization_selection"]
        self.assertEqual(selection, state["summary"]["model"]["regularization_selection"])
        self.assertEqual(selection["candidates"], list(risk_model.C_CANDIDATES))
        self.assertFalse(selection["calibration_days_used"])
        self.assertFalse(selection["evaluation_days_used"])
        self.assertEqual(selection["fold_weighting"], "equal")
        start = pd.Timestamp(risk_model.START_DATE, tz="UTC")
        train = state["frame"][state["frame"]["split"] == "train"]
        timestamps = pd.to_datetime(train["timestamp"], utc=True)
        for fold, days in zip(selection["folds"], (risk_model.TRAIN_DAYS - 6,
                                                  risk_model.TRAIN_DAYS - 4,
                                                  risk_model.TRAIN_DAYS - 2)):
            boundary = start + pd.Timedelta(days=days)
            stop = boundary + pd.Timedelta(days=2)
            fitting = train[timestamps < boundary]
            validation = train[(timestamps >= boundary) & (timestamps < stop)]
            self.assertEqual(fold["train_start"], start.date().isoformat())
            self.assertEqual(fold["train_end"], (boundary - pd.Timedelta(days=1)).date().isoformat())
            self.assertEqual(fold["validation_start"], boundary.date().isoformat())
            self.assertEqual(fold["validation_end"], (stop - pd.Timedelta(days=1)).date().isoformat())
            self.assertEqual(fold["training_events"], len(fitting))
            self.assertEqual(fold["validation_events"], len(validation))
            self.assertEqual(fold["positive_training_events"], int((fitting["ground_truth"] == "anomalous").sum()))
            self.assertEqual(fold["positive_validation_events"], int((validation["ground_truth"] == "anomalous").sum()))
        for candidate in selection["candidate_scores"]:
            self.assertEqual(len(candidate["fold_average_precision"]), 3)
            self.assertEqual(candidate["mean_average_precision"],
                             float(np.mean(candidate["fold_average_precision"])))
        expected = min(selection["candidate_scores"],
                       key=lambda row: (-row["mean_average_precision"], row["C"]))
        self.assertEqual(selection["selected_C"], expected["C"])
        self.assertEqual(selection["selected_mean_average_precision"], expected["mean_average_precision"])
        self.assertEqual(state["model"]["pipeline"].named_steps["classifier"].C, expected["C"])

    def test_selection_ignores_calibration_and_evaluation_records(self):
        state = snapshot()
        changed = state["frame"].copy()
        mask = changed["split"] != "train"
        changed.loc[mask, "ground_truth"] = changed.loc[mask, "ground_truth"].map(
            {"normal": "anomalous", "anomalous": "normal"}
        )
        changed.loc[mask, risk_model.NUMERIC_FEATURES] = 1_000_000
        changed.loc[mask, risk_model.CATEGORICAL_FEATURES] = "EXCLUDED_FROM_SELECTION"
        selection = risk_model.select_regularization(changed)
        self.assertEqual(selection, state["model"]["metadata"]["regularization_selection"])

    def test_each_fold_fits_fresh_preprocessing_without_probability_calibration(self):
        frame = snapshot()["frame"]
        train = frame[frame["split"] == "train"]
        timestamps = pd.to_datetime(train["timestamp"], utc=True)
        start = pd.Timestamp(risk_model.START_DATE, tz="UTC")
        pipelines = []

        def make_pipeline(c):
            pipeline = self.constant_pipeline()
            pipelines.append((c, pipeline))
            return pipeline

        with patch.object(risk_model, "_build_pipeline", side_effect=make_pipeline) as factory, \
             patch.object(risk_model, "IsotonicRegression") as calibrator:
            risk_model.select_regularization(frame)
        self.assertEqual(factory.call_count, len(risk_model.C_CANDIDATES) * 3)
        self.assertEqual(len({id(pipeline) for _, pipeline in pipelines}), len(pipelines))
        calibrator.assert_not_called()
        for position, (c, pipeline) in enumerate(pipelines):
            self.assertEqual(c, risk_model.C_CANDIDATES[position // 3])
            days = (risk_model.TRAIN_DAYS - 6, risk_model.TRAIN_DAYS - 4,
                    risk_model.TRAIN_DAYS - 2)[position % 3]
            boundary = start + pd.Timedelta(days=days)
            stop = boundary + pd.Timedelta(days=2)
            fitting = pipeline.fit.call_args.args[0]
            validation = pipeline.predict_proba.call_args.args[0]
            self.assertEqual(list(fitting.columns), MODEL_FEATURES)
            self.assertEqual(list(validation.columns), MODEL_FEATURES)
            self.assertEqual(list(fitting.index), list(train[timestamps < boundary].index))
            self.assertEqual(list(validation.index),
                             list(train[(timestamps >= boundary) & (timestamps < stop)].index))
            self.assertTrue(set(fitting.index).isdisjoint(validation.index))

    def test_selection_uses_unrounded_means_and_exact_tie_break(self):
        frame = snapshot()["frame"]
        scores = [0.8] * 3 + [0.80000001] * 3 + [0.8] * 12
        with patch.object(risk_model, "_build_pipeline",
                          side_effect=lambda c: self.constant_pipeline()), \
             patch.object(risk_model, "average_precision_score", side_effect=scores):
            selection = risk_model.select_regularization(frame)
        self.assertEqual(selection["selected_C"], 0.3)
        with patch.object(risk_model, "_build_pipeline",
                          side_effect=lambda c: self.constant_pipeline()), \
             patch.object(risk_model, "average_precision_score", return_value=0.8):
            tied = risk_model.select_regularization(frame)
        self.assertEqual(tied["selected_C"], min(risk_model.C_CANDIDATES))

    def test_selection_rejects_invalid_dates_and_single_class_or_empty_windows(self):
        train = snapshot()["frame"]
        train = train[train["split"] == "train"].copy()
        start = pd.Timestamp(risk_model.START_DATE, tz="UTC")
        timestamps = pd.to_datetime(train["timestamp"], utc=True)
        missing_window = train[
            ~((timestamps >= start + pd.Timedelta(days=risk_model.TRAIN_DAYS - 4))
              & (timestamps < start + pd.Timedelta(days=risk_model.TRAIN_DAYS - 2)))
        ]
        with self.assertRaisesRegex(ValueError, "both classes"):
            risk_model.select_regularization(missing_window)
        single_class = train.copy()
        single_class.loc[timestamps >= start + pd.Timedelta(days=risk_model.TRAIN_DAYS - 2),
                         "ground_truth"] = "normal"
        with self.assertRaisesRegex(ValueError, "both classes"):
            risk_model.select_regularization(single_class)
        invalid_dates = train.copy()
        invalid_dates.loc[invalid_dates.index[0], "timestamp"] = (
            start + pd.Timedelta(days=risk_model.TRAIN_DAYS)
        ).isoformat()
        with self.assertRaisesRegex(ValueError, "configured training period"):
            risk_model.select_regularization(invalid_dates)
        with patch.object(risk_model, "TRAIN_DAYS", 6):
            with self.assertRaisesRegex(ValueError, "at least seven"):
                risk_model.select_regularization(train)


if __name__ == "__main__":
    unittest.main()
