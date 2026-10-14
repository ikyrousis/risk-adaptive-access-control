"""One static baseline and score-plus-context allow/challenge/block controls."""

from __future__ import annotations

import math

from config import ROLE_RESOURCE_GROUPS
from src.dataset import parse_utc
from src.features import HISTORY_DEFINITION

MANDATORY_BLOCK_CODES = (
    "BLOCK_NON_IT_IDENTITY_CHANGE",
    "BLOCK_PRIVILEGED_ROLE_MISMATCH",
    "BLOCK_UNMANAGED_PRIVILEGED_ACCESS",
)
VOLUME_THRESHOLD_CANDIDATES = list(range(50, 1001, 25))


def evaluate_request_controls(event: dict[str, str], resource_group: str) -> tuple[str, list[str]]:
    """Evaluate the per-request controls used inside the single baseline."""
    reasons: list[str] = []
    allowed_groups = ROLE_RESOURCE_GROUPS[event["role"]]
    role_mismatch = resource_group not in allowed_groups
    unmanaged = event["device_managed"] != "true"
    privileged = event["resource_sensitivity"] == "privileged"
    confidential = event["resource_sensitivity"] == "confidential"
    hour = parse_utc(event["timestamp"]).hour
    off_hours = hour < 6 or hour >= 22
    non_it_identity_change = (
        event["department"] != "IT Operations"
        and event["resource_id"] == "RES-025"
        and event["action"] in {"create", "modify"}
    )

    if non_it_identity_change:
        reasons.append("BLOCK_NON_IT_IDENTITY_CHANGE")
    if role_mismatch and privileged:
        reasons.append("BLOCK_PRIVILEGED_ROLE_MISMATCH")
    if unmanaged and privileged:
        reasons.append("BLOCK_UNMANAGED_PRIVILEGED_ACCESS")

    if reasons:
        return "block", reasons

    if off_hours and privileged:
        reasons.append("CHALLENGE_OFF_HOURS_PRIVILEGED")
    if unmanaged and confidential:
        reasons.append("CHALLENGE_UNMANAGED_CONFIDENTIAL")
    if role_mismatch and confidential:
        reasons.append("CHALLENGE_CONFIDENTIAL_ROLE_MISMATCH")
    if (
        event["network_zone"] == "foreign_public_network"
        and event["resource_sensitivity"] in {"confidential", "privileged"}
    ):
        reasons.append("CHALLENGE_FOREIGN_SENSITIVE_ACCESS")
    if float(event["data_volume_mb"]) > 500:
        reasons.append("CHALLENGE_LARGE_TRANSFER")

    if reasons:
        return "challenge", reasons
    return "allow", ["ALLOW_NO_STATIC_RULE_TRIGGERED"]


def mandatory_block_codes(event: dict[str, str], resource_group: str) -> list[str]:
    """Return the non-negotiable block rules that fire for an event."""
    decision, reasons = evaluate_request_controls(event, resource_group)
    if decision != "block":
        return []
    return [code for code in reasons if code in MANDATORY_BLOCK_CODES]


def evaluate_event(
    event: dict[str, str],
    resource_group: str,
    history: dict[str, float],
    volume_threshold_mb: float,
) -> tuple[str, list[str]]:
    """Return the selected static decision.

    ``history`` holds strictly earlier timestamp aggregates for the same user:
    ``location_change_60m`` (0/1) and ``user_volume_1h_mb`` (requested MB in
    the previous hour, excluding the current and simultaneous requests).
    """
    decision, reasons = evaluate_request_controls(event, resource_group)
    if decision == "block":
        return decision, reasons
    reasons = [] if decision == "allow" else list(reasons)
    role_mismatch = resource_group not in ROLE_RESOURCE_GROUPS[event["role"]]
    if int(history["location_change_60m"]) == 1:
        reasons.append("CHALLENGE_RAPID_LOCATION_CHANGE")
    if float(history["user_volume_1h_mb"]) >= volume_threshold_mb:
        reasons.append("CHALLENGE_SESSION_VOLUME")
    if role_mismatch and event["resource_sensitivity"] in {"public", "internal"}:
        reasons.append("CHALLENGE_ROLE_MISMATCH_ANY")
    if reasons:
        return "challenge", reasons
    return "allow", ["ALLOW_NO_STATIC_RULE_TRIGGERED"]


def f1_score_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def select_volume_threshold(events, resources, history) -> dict[str, float]:
    """Choose the cumulative-volume threshold on calibration days only."""
    calibration = [e for e in events if history[e["event_id"]]["split"] == "calibration"]
    if not calibration:
        raise ValueError("Static threshold selection requires calibration events.")
    best = None
    for threshold in VOLUME_THRESHOLD_CANDIDATES:
        tp = fp = fn = 0
        for event in calibration:
            decision, _ = evaluate_event(
                event,
                resources[event["resource_id"]]["resource_group"],
                history[event["event_id"]],
                threshold,
            )
            flagged = decision != "allow"
            actual = event["ground_truth"] == "anomalous"
            tp += flagged and actual
            fp += flagged and not actual
            fn += (not flagged) and actual
        f1 = f1_score_counts(tp, fp, fn)
        # Highest F1 wins; ties prefer the higher threshold.
        candidate = (round(f1, 10), threshold)
        if best is None or candidate > best:
            best = candidate
    return {
        "volume_threshold_mb": float(best[1]),
        "calibration_f1": round(best[0], 4),
        "calibration_events": len(calibration),
    }


def baseline_description(selection: dict) -> dict:
    """Store concise policy/selection evidence inside the single results JSON."""
    threshold = selection["volume_threshold_mb"]
    return {
        "name": "history-aware static baseline",
        "block_rules": list(MANDATORY_BLOCK_CODES),
        "challenge_rules": [
            "CHALLENGE_OFF_HOURS_PRIVILEGED (before 06:00 or from 22:00 UTC)",
            "CHALLENGE_UNMANAGED_CONFIDENTIAL",
            "CHALLENGE_CONFIDENTIAL_ROLE_MISMATCH",
            "CHALLENGE_FOREIGN_SENSITIVE_ACCESS",
            "CHALLENGE_LARGE_TRANSFER (declared size above 500 MB)",
            "CHALLENGE_RAPID_LOCATION_CHANGE (location absent from the latest prior user batch within 60 minutes)",
            f"CHALLENGE_SESSION_VOLUME (prior one-hour user volume of at least {threshold:.0f} MB)",
            "CHALLENGE_ROLE_MISMATCH_ANY (public or internal resource)",
        ],
        "precedence": "block rules first, then challenge rules, otherwise allow",
        "history_definition": HISTORY_DEFINITION,
        "threshold_selection": {
            "method": "maximise calibration-day intervention F1; ties prefer the higher threshold",
            "candidates_mb": [VOLUME_THRESHOLD_CANDIDATES[0], VOLUME_THRESHOLD_CANDIDATES[-1], 25],
            "evaluation_days_used": False,
            **selection,
        },
    }


CONTROL_ALLOW = "standard access logging"
CONTROL_MFA = "phishing-resistant MFA challenge"
CONTROL_MFA_MANAGER = "phishing-resistant MFA plus manager approval"
CONTROL_BLOCK = "request denied and security review opened"
DECISION_SOURCES = ("model", "contextual_override", "mandatory_rule", "fail_safe")


def model_band(score, challenge_threshold, block_threshold) -> str:
    if score >= block_threshold:
        return "block"
    if score >= challenge_threshold:
        return "challenge"
    return "allow"


def valid_score(score) -> bool:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and 0 <= value <= 100


def decide(score, context, challenge_threshold, block_threshold) -> dict:
    """Mandatory blocks take precedence; invalid scores otherwise challenge."""
    mandatory = list(context.get("mandatory_block_codes") or [])
    sensitivity = context["resource_sensitivity"]
    managed = bool(context["device_managed"])
    if not valid_score(score):
        return {
            "model_decision": "invalid",
            "final_decision": "block" if mandatory else "challenge",
            "step_up_control": CONTROL_BLOCK if mandatory else CONTROL_MFA,
            "decision_source": "mandatory_rule" if mandatory else "fail_safe",
            "override_codes": (
                ["FAILSAFE_INVALID_SCORE", "MANDATORY_STATIC_BLOCK", *mandatory]
                if mandatory else ["FAILSAFE_INVALID_SCORE"]
            ),
        }
    band = model_band(float(score), challenge_threshold, block_threshold)
    if mandatory:
        return {
            "model_decision": band, "final_decision": "block",
            "step_up_control": CONTROL_BLOCK,
            "decision_source": "model" if band == "block" else "mandatory_rule",
            "override_codes": ["MANDATORY_STATIC_BLOCK", *mandatory],
        }
    if band == "block":
        return {
            "model_decision": band, "final_decision": "block",
            "step_up_control": CONTROL_BLOCK, "decision_source": "model",
            "override_codes": [],
        }
    if band == "challenge":
        privileged = sensitivity == "privileged"
        return {
            "model_decision": band, "final_decision": "challenge",
            "step_up_control": CONTROL_MFA_MANAGER if privileged else CONTROL_MFA,
            "decision_source": "contextual_override" if privileged else "model",
            "override_codes": ["OVERRIDE_PRIVILEGED_MANAGER_APPROVAL"] if privileged else [],
        }
    if not managed and sensitivity in {"confidential", "privileged"}:
        return {
            "model_decision": band, "final_decision": "challenge",
            "step_up_control": CONTROL_MFA, "decision_source": "contextual_override",
            "override_codes": ["OVERRIDE_UNMANAGED_SENSITIVE_ACCESS"],
        }
    return {
        "model_decision": band, "final_decision": "allow",
        "step_up_control": CONTROL_ALLOW, "decision_source": "model", "override_codes": [],
    }


def context_for_event(event, resources) -> dict:
    group = resources[event["resource_id"]]["resource_group"]
    return {
        "mandatory_block_codes": mandatory_block_codes(event, group),
        "resource_sensitivity": event["resource_sensitivity"],
        "device_managed": event["device_managed"] == "true",
    }


def reason_codes(row) -> list[str]:
    """Human-readable indicators; fitted-model contributions are separate."""
    indicators = [
        (row["device_seen_before"] == 0, "NEW_DEVICE"),
        (row["device_managed_numeric"] == 0, "UNMANAGED_DEVICE"),
        (row["outside_role_hours"] == 1, "OUTSIDE_ROLE_HOURS"),
        (row["role_resource_allowed"] == 0, "ROLE_RESOURCE_RARITY"),
        (row["location_change_60m"] == 1, "RAPID_LOCATION_CHANGE"),
        (row["location_seen_before"] == 0, "NEW_LOCATION"),
        (row["user_events_1h"] >= 12, "HIGH_SESSION_FREQUENCY"),
        (row["user_volume_1h_mb"] >= 500, "HIGH_SESSION_VOLUME"),
        (row["large_transfer"] == 1, "LARGE_TRANSFER"),
        (row["privileged_action"] == 1, "PRIVILEGED_ACTION"),
        (row["recent_failures_24h"] >= 2, "RECENT_FAILURES"),
    ]
    return [name for matched, name in indicators if matched] or ["ELEVATED_COMBINED_PATTERN"]
