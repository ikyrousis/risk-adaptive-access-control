"""Strictly prior-timestamp behavioural features, computed entirely in memory."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import timedelta
from itertools import groupby

from config import (
    CALIBRATION_DAYS,
    DEPARTMENT_SPECS,
    MODEL_FEATURES,
    ROLE_RESOURCE_GROUPS,
    SENSITIVITY_RANK,
    START_DATE,
    TRAIN_DAYS,
)
from src.dataset import parse_utc

IDENTITY_FIELDS = [
    "event_id",
    "timestamp",
    "user_id",
    "role",
    "department",
    "resource_id",
    "resource_sensitivity",
    "ground_truth",
    "anomaly_type",
    "split",
]
FEATURE_FIELDS = list(dict.fromkeys(IDENTITY_FIELDS + MODEL_FEATURES))
HISTORY_DEFINITION = (
    "Only records with strictly earlier UTC timestamps are observed. Equal-time "
    "requests are scored as a batch before history is updated. The latest prior "
    "user timestamp retains all locations, without inventing an order within it."
)


def assign_split(timestamp) -> str:
    day_offset = (timestamp.date() - START_DATE).days
    if day_offset < TRAIN_DAYS:
        return "train"
    if day_offset < TRAIN_DAYS + CALIBRATION_DAYS:
        return "calibration"
    return "test"


def compute_features(events: list[dict[str, str]], resources: dict[str, dict[str, str]]):
    """Compute features from strictly earlier timestamps, not event-ID order.

    Every request in a timestamp batch sees the same pre-batch history. If a
    user's latest prior batch has multiple locations, a rapid change is flagged
    only when the current location is absent from that set. There is no evidence
    that one simultaneous location was the user's last location.
    """
    ordered = sorted(
        ((parse_utc(event["timestamp"]), event) for event in events),
        key=lambda item: (item[0], item[1]["event_id"]),
    )
    seen_devices = defaultdict(set)
    device_counts = defaultdict(int)
    user_resource_counts = defaultdict(int)
    role_resource_counts = defaultdict(int)
    seen_locations = defaultdict(set)
    previous_user_event = {}
    failures_by_user = defaultdict(deque)
    recent_events_by_user = defaultdict(deque)

    output = []
    for timestamp, batch_items in groupby(ordered, key=lambda item: item[0]):
        batch = [event for _, event in batch_items]
        for event in batch:
            user_id = event["user_id"]
            role = event["role"]
            resource_id = event["resource_id"]
            resource_group = resources[resource_id]["resource_group"]
            spec = DEPARTMENT_SPECS[event["department"]]
            hour_decimal = timestamp.hour + timestamp.minute / 60.0

            failure_window = failures_by_user[user_id]
            while failure_window and timestamp - failure_window[0] > timedelta(hours=24):
                failure_window.popleft()

            one_hour_window = recent_events_by_user[user_id]
            while one_hour_window and timestamp - one_hour_window[0][0] > timedelta(hours=1):
                one_hour_window.popleft()

            previous = previous_user_event.get(user_id)
            minutes_since_previous = (
                min((timestamp - previous[0]).total_seconds() / 60.0, 1440.0)
                if previous
                else 1440.0
            )
            location_change_60m = int(
                bool(previous)
                and event["location"] not in previous[1]
                and 0 < minutes_since_previous <= 60
            )
            # Stable summation also avoids dependence on tie ordering in history.
            recent_volume = math.fsum(item[1] for item in one_hour_window)
            action = event["action"]

            row = {
                "event_id": event["event_id"],
                "timestamp": event["timestamp"],
                "user_id": user_id,
                "role": role,
                "department": event["department"],
                "resource_id": resource_id,
                "resource_sensitivity": event["resource_sensitivity"],
                "ground_truth": event["ground_truth"],
                "anomaly_type": event["anomaly_type"],
                "split": assign_split(timestamp),
                "hour_sin": math.sin(2 * math.pi * hour_decimal / 24),
                "hour_cos": math.cos(2 * math.pi * hour_decimal / 24),
                "is_weekend": int(timestamp.weekday() >= 5),
                "outside_role_hours": int(
                    timestamp.hour < spec["work_hours"][0]
                    or timestamp.hour >= spec["work_hours"][1]
                ),
                "sensitivity_rank": SENSITIVITY_RANK[event["resource_sensitivity"]],
                "device_managed_numeric": int(event["device_managed"] == "true"),
                "device_seen_before": int(event["device_id"] in seen_devices[user_id]),
                "device_prior_count": device_counts[(user_id, event["device_id"])],
                "user_resource_prior_count": user_resource_counts[(user_id, resource_id)],
                "role_resource_prior_count": role_resource_counts[(role, resource_id)],
                "role_resource_allowed": int(
                    resource_group in ROLE_RESOURCE_GROUPS[event["role"]]
                ),
                "location_seen_before": int(event["location"] in seen_locations[user_id]),
                "location_change_60m": location_change_60m,
                "minutes_since_previous_event": minutes_since_previous,
                "recent_failures_24h": len(failure_window),
                "user_events_1h": len(one_hour_window),
                "user_volume_1h_mb": recent_volume,
                "large_transfer": int(float(event["data_volume_mb"]) > 100),
                "privileged_action": int(
                    event["resource_sensitivity"] == "privileged"
                    and action in {"approve", "execute", "configure", "create", "modify", "wipe"}
                ),
                "network_zone": event["network_zone"],
                "action": action,
            }
            output.append(row)

        # Commit after every request at this timestamp has been scored.
        batch_locations = defaultdict(set)
        for event in batch:
            user_id = event["user_id"]
            seen_devices[user_id].add(event["device_id"])
            device_counts[(user_id, event["device_id"])] += 1
            user_resource_counts[(user_id, event["resource_id"])] += 1
            role_resource_counts[(event["role"], event["resource_id"])] += 1
            seen_locations[user_id].add(event["location"])
            batch_locations[user_id].add(event["location"])
            if event["outcome"] == "denied":
                failures_by_user[user_id].append(timestamp)
            recent_events_by_user[user_id].append(
                (timestamp, float(event["data_volume_mb"]))
            )
        for user_id, locations in batch_locations.items():
            previous_user_event[user_id] = (timestamp, frozenset(locations))

    return output
