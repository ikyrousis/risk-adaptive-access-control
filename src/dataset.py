"""Generate and validate the synthetic enterprise dataset."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta, timezone

from config import (
    ANOMALY_TARGETS, CALIBRATION_DAYS, DEPARTMENT_SPECS, NORMAL_NETWORKS,
    NUMBER_OF_DAYS, PERSONAL_DEVICE_REQUESTS_PER_USER,
    PERSONAL_DEVICE_USERS_PER_DEPARTMENT, RESOURCE_SPECS, ROLE_RESOURCE_GROUPS,
    SEED, START_DATE, TRAIN_DAYS, TRAVEL_LOCATIONS,
)

EVENT_FIELDS = [
    "event_id", "timestamp", "user_id", "role", "department", "location",
    "network_zone", "device_id", "device_managed", "resource_id",
    "resource_name", "resource_sensitivity", "action", "outcome",
    "data_volume_mb", "ground_truth", "anomaly_type", "anomaly_campaign_id",
]
MANDATORY_FIELDS = {
    "timestamp", "user_id", "role", "department", "location", "network_zone",
    "device_id", "resource_id", "action", "outcome", "ground_truth",
}


def seeded_random(stage_offset: int) -> random.Random:
    return random.Random(SEED + stage_offset)


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def string_records(records):
    """Keep event values consistent with the documented CSV representation."""
    return [{field: str(record[field]) for field in EVENT_FIELDS} for record in records]


def generate_enterprise() -> dict:
    """Return users, assigned devices and resources."""
    rng = seeded_random(1)
    users, devices = [], []
    user_number = 1
    for department, spec in DEPARTMENT_SPECS.items():
        for _ in range(spec["count"]):
            user_id = f"USR-{user_number:03d}"
            role = rng.choices(spec["roles"], weights=spec["role_weights"], k=1)[0]
            remote_ratio = round(rng.uniform(0.15, 0.55), 2)
            primary_device = f"DEV-{user_number:03d}-A"
            has_mobile = rng.random() < 0.68
            users.append({
                "user_id": user_id, "role": role, "department": department,
                "home_office": spec["home_office"],
                "remote_work_ratio": remote_ratio,
                "primary_device_id": primary_device,
            })
            devices.append({
                "device_id": primary_device, "user_id": user_id,
                "device_type": "managed_laptop", "managed": "true", "assigned": "true",
            })
            if has_mobile:
                devices.append({
                    "device_id": f"DEV-{user_number:03d}-M", "user_id": user_id,
                    "device_type": "managed_mobile", "managed": "true", "assigned": "true",
                })
            user_number += 1
    resources = [
        {
            "resource_id": resource_id, "resource_name": name,
            "resource_group": group, "sensitivity": sensitivity,
            "allowed_actions": "|".join(actions),
        }
        for resource_id, name, group, sensitivity, actions in RESOURCE_SPECS
    ]
    return {"users": users, "devices": devices, "resources": resources}


def random_work_timestamp(rng, day, start_hour, end_hour):
    hour = rng.randrange(start_hour, end_hour)
    minute = rng.randrange(60)
    second = rng.randrange(60)
    return datetime.combine(day, time(hour, minute, second), tzinfo=timezone.utc)


def generate_normal_events(enterprise) -> list[dict[str, str]]:
    """Generate routine activity and benign exceptions before the BYOD overlay."""
    rng = seeded_random(2)
    users, resources, devices = (
        enterprise["users"], enterprise["resources"], enterprise["devices"]
    )
    devices_by_user = defaultdict(list)
    for device in devices:
        devices_by_user[device["user_id"]].append(device)
    events = []
    event_number = 1
    for day_offset in range(NUMBER_OF_DAYS):
        day = START_DATE + timedelta(days=day_offset)
        is_weekend = day.weekday() >= 5
        for user in users:
            spec = DEPARTMENT_SPECS[user["department"]]
            low, high = spec["weekend_events"] if is_weekend else spec["weekday_events"]
            event_count = rng.randint(low, high)
            if event_count == 0:
                continue
            allowed_groups = ROLE_RESOURCE_GROUPS[user["role"]]
            eligible_resources = [
                resource for resource in resources
                if resource["resource_group"] in allowed_groups
            ]
            group_weights = []
            for resource in eligible_resources:
                group = resource["resource_group"]
                group_weights.append(
                    4.0 if group == "common" else
                    1.0 if group == "management" else
                    1.5 if group == "privileged" else 3.0
                )
            for _ in range(event_count):
                timestamp = random_work_timestamp(rng, day, *spec["work_hours"])
                resource = rng.choices(eligible_resources, weights=group_weights, k=1)[0]
                action = rng.choice(resource["allowed_actions"].split("|"))
                remote = rng.random() < float(user["remote_work_ratio"])
                location = user["home_office"]
                network_zone = (
                    rng.choice(["corporate_vpn", "home_gr"]) if remote
                    else NORMAL_NETWORKS[user["home_office"]][0]
                )
                user_devices = devices_by_user[user["user_id"]]
                device = (
                    rng.choices(user_devices, weights=[0.82, 0.18], k=1)[0]
                    if len(user_devices) > 1 and action in {"read", "send", "write"}
                    else user_devices[0]
                )
                if action in {"download", "export", "clone"}:
                    volume = round(rng.lognormvariate(1.8, 0.65), 2)
                elif action in {"upload", "push"}:
                    volume = round(rng.lognormvariate(1.1, 0.55), 2)
                else:
                    volume = round(rng.uniform(0.01, 1.2), 2)
                events.append({
                    "event_id": f"EVT-{event_number:06d}",
                    "timestamp": iso_utc(timestamp), "user_id": user["user_id"],
                    "role": user["role"], "department": user["department"],
                    "location": location, "network_zone": network_zone,
                    "device_id": device["device_id"], "device_managed": device["managed"],
                    "resource_id": resource["resource_id"],
                    "resource_name": resource["resource_name"],
                    "resource_sensitivity": resource["sensitivity"], "action": action,
                    "outcome": "denied" if rng.random() < 0.008 else "success",
                    "data_volume_mb": volume, "ground_truth": "normal",
                    "anomaly_type": "none", "anomaly_campaign_id": "",
                })
                event_number += 1

    # Plausible foreign travel days.
    events_by_user_day = defaultdict(list)
    for event in events:
        events_by_user_day[(event["user_id"], event["timestamp"][:10])].append(event)
    travel_groups = [
        key for key, rows in events_by_user_day.items()
        if rows[0]["department"] in {"Engineering", "Sales"} and len(rows) >= 10
    ]
    for group_index, key in enumerate(rng.sample(travel_groups, 10)):
        city = TRAVEL_LOCATIONS[group_index % len(TRAVEL_LOCATIONS)]
        for event in events_by_user_day[key]:
            event["location"] = city
            event["network_zone"] = "foreign_public_network"

    # Approved large individual transfers overlap suspicious campaign sizes.
    transfers = [
        event for event in events if event["action"] in {"download", "export", "clone"}
    ]
    for event in rng.sample(transfers, 60):
        event["data_volume_mb"] = round(rng.uniform(80.0, 420.0), 2)

    # Managed laptop rotation is benign device novelty.
    rotations = [
        event for event in events
        if event["resource_sensitivity"] in {"confidential", "privileged"}
    ]
    rng.shuffle(rotations)
    rotated_users = set()
    for event in rotations:
        if event["user_id"] in rotated_users:
            continue
        event["device_id"] = f"ROLLOUT-{event['user_id']}"
        event["device_managed"] = "true"
        rotated_users.add(event["user_id"])
        if len(rotated_users) == 50:
            break

    # Approved Engineering-Sales project access is absent from the static mapping.
    projects = [
        event for event in events if event["department"] in {"Engineering", "Sales"}
    ]
    for event in rng.sample(projects, 70):
        resource_id = "RES-021" if event["department"] == "Engineering" else "RES-007"
        target = next(resource for resource in resources if resource["resource_id"] == resource_id)
        event.update({
            "resource_id": target["resource_id"], "resource_name": target["resource_name"],
            "resource_sensitivity": target["sensitivity"], "action": "read",
            "data_volume_mb": round(rng.uniform(0.05, 1.0), 2),
        })

    # Approved overnight IT maintenance.
    it_sources = [event for event in events if event["department"] == "IT Operations"]
    production = next(resource for resource in resources if resource["resource_id"] == "RES-026")
    for _ in range(30):
        source = rng.choice(it_sources)
        day = START_DATE + timedelta(days=rng.randrange(NUMBER_OF_DAYS))
        maintenance = dict(source)
        maintenance.update({
            "event_id": f"EVT-{event_number:06d}",
            "timestamp": iso_utc(datetime.combine(
                day, time(rng.randrange(1, 5), rng.randrange(60), rng.randrange(60)),
                tzinfo=timezone.utc,
            )),
            "resource_id": production["resource_id"],
            "resource_name": production["resource_name"],
            "resource_sensitivity": production["sensitivity"],
            "action": "read", "outcome": "success",
            "data_volume_mb": round(rng.uniform(0.05, 1.0), 2),
        })
        events.append(maintenance)
        event_number += 1
    events.sort(key=lambda event: (event["timestamp"], event["event_id"]))
    return string_records(events)


def anomaly_event(source, number, category, campaign, **changes):
    event = dict(source)
    event.update(changes)
    event.update({
        "event_id": f"ANM-{number:06d}", "ground_truth": "anomalous",
        "anomaly_type": category, "anomaly_campaign_id": campaign,
    })
    return event


def generate_anomalies(normal_events, enterprise) -> list[dict[str, str]]:
    """Inject six fixed anomaly recipes using an independent seeded stream."""
    rng = seeded_random(3)
    users, resources = enterprise["users"], enterprise["resources"]
    events_by_user = defaultdict(list)
    for event in normal_events:
        events_by_user[event["user_id"]].append(event)
    anomalies = []
    number = 1

    candidates = [event for event in normal_events if event["network_zone"] != "corporate_vpn"]
    for index, source in enumerate(
        rng.sample(candidates, ANOMALY_TARGETS["impossible_travel"]), 1
    ):
        location = rng.choice([city for city in TRAVEL_LOCATIONS if city != source["location"]])
        shifted = parse_utc(source["timestamp"]) + timedelta(minutes=rng.randint(10, 35))
        anomalies.append(anomaly_event(
            source, number, "impossible_travel", f"IMP-{index:03d}",
            timestamp=iso_utc(shifted), location=location,
            network_zone="foreign_public_network", outcome="success",
        ))
        number += 1

    off_hours_users = [
        user for user in users
        if user["department"] in {"Finance", "Human Resources", "IT Operations"}
    ]
    for index in range(1, ANOMALY_TARGETS["off_hours_privileged_access"] + 1):
        user = rng.choice(off_hours_users)
        resource = rng.choice([
            resource for resource in resources
            if resource["sensitivity"] == "privileged"
            and resource["resource_group"] in ROLE_RESOURCE_GROUPS[user["role"]]
        ])
        day = START_DATE + timedelta(days=rng.randrange(NUMBER_OF_DAYS))
        timestamp = datetime.combine(
            day, time(rng.randrange(5), rng.randrange(60), rng.randrange(60)),
            tzinfo=timezone.utc,
        )
        source = rng.choice(events_by_user[user["user_id"]])
        anomalies.append(anomaly_event(
            source, number, "off_hours_privileged_access", f"OHP-{index:03d}",
            timestamp=iso_utc(timestamp), resource_id=resource["resource_id"],
            resource_name=resource["resource_name"],
            resource_sensitivity=resource["sensitivity"],
            action=rng.choice(resource["allowed_actions"].split("|")),
            data_volume_mb=round(rng.uniform(0.05, 4.0), 2), outcome="success",
        ))
        number += 1

    for index in range(1, ANOMALY_TARGETS["first_seen_device_sensitive_action"] + 1):
        user = rng.choice(users)
        resource = rng.choice([
            resource for resource in resources
            if resource["sensitivity"] in {"confidential", "privileged"}
            and resource["resource_group"] in ROLE_RESOURCE_GROUPS[user["role"]]
        ])
        source = rng.choice(events_by_user[user["user_id"]])
        anomalies.append(anomaly_event(
            source, number, "first_seen_device_sensitive_action", f"FSD-{index:03d}",
            device_id=f"UNMANAGED-{index:04d}", device_managed="false",
            resource_id=resource["resource_id"], resource_name=resource["resource_name"],
            resource_sensitivity=resource["sensitivity"],
            action=rng.choice(resource["allowed_actions"].split("|")),
            network_zone=rng.choice(["home_gr", "corporate_vpn", "guest_network"]),
            outcome="success",
        ))
        number += 1

    for index in range(1, ANOMALY_TARGETS["role_resource_mismatch"] + 1):
        user = rng.choice(users)
        disallowed = [
            resource for resource in resources
            if resource["resource_group"] not in ROLE_RESOURCE_GROUPS[user["role"]]
        ]
        resource = rng.choice(disallowed)
        source = rng.choice(events_by_user[user["user_id"]])
        anomalies.append(anomaly_event(
            source, number, "role_resource_mismatch", f"RRM-{index:03d}",
            resource_id=resource["resource_id"], resource_name=resource["resource_name"],
            resource_sensitivity=resource["sensitivity"],
            action=rng.choice(resource["allowed_actions"].split("|")), outcome="success",
        ))
        number += 1

    remaining = ANOMALY_TARGETS["bulk_download_spike"]
    campaign_index = 1
    while remaining > 0:
        campaign_size = min(remaining, rng.randint(9, 16))
        user = rng.choice(users)
        source = rng.choice(events_by_user[user["user_id"]])
        start = parse_utc(source["timestamp"])
        campaign = f"BULK-{campaign_index:03d}"
        transfer_resources = [
            resource for resource in resources
            if {"download", "export", "clone"} & set(resource["allowed_actions"].split("|"))
            and resource["resource_group"] in ROLE_RESOURCE_GROUPS[user["role"]]
        ]
        for offset in range(campaign_size):
            resource = rng.choice(transfer_resources)
            actions = sorted(
                {"download", "export", "clone"} & set(resource["allowed_actions"].split("|"))
            )
            anomalies.append(anomaly_event(
                source, number, "bulk_download_spike", campaign,
                timestamp=iso_utc(start + timedelta(seconds=offset * rng.randint(8, 25))),
                resource_id=resource["resource_id"], resource_name=resource["resource_name"],
                resource_sensitivity=resource["sensitivity"], action=rng.choice(actions),
                data_volume_mb=round(rng.uniform(35.0, 140.0), 2), outcome="success",
            ))
            number += 1
        remaining -= campaign_size
        campaign_index += 1

    non_it_users = [user for user in users if user["department"] != "IT Operations"]
    identity = next(resource for resource in resources if resource["resource_id"] == "RES-025")
    pair_count = ANOMALY_TARGETS["privilege_escalation_sequence"] // 2
    for index in range(1, pair_count + 1):
        user = rng.choice(non_it_users)
        source = rng.choice(events_by_user[user["user_id"]])
        start = parse_utc(source["timestamp"]) + timedelta(minutes=2)
        for sequence_offset, action in enumerate(["create", "modify"]):
            anomalies.append(anomaly_event(
                source, number, "privilege_escalation_sequence", f"PES-{index:03d}",
                timestamp=iso_utc(start + timedelta(seconds=sequence_offset * 45)),
                resource_id=identity["resource_id"], resource_name=identity["resource_name"],
                resource_sensitivity=identity["sensitivity"], action=action,
                data_volume_mb=round(rng.uniform(0.02, 0.3), 2),
                outcome="denied" if sequence_offset == 0 else "success",
            ))
            number += 1
    return string_records(anomalies)


def add_personal_device_activity(normal_events, enterprise):
    """Convert legitimate requests, changing only device identity and trust.

    Each selected user has one stable personal device and both request kinds
    in every chronological partition. Extra requests are sampled without
    replacement. No model scores or policy decisions influence selection.
    Apply this after anomaly construction so attack records remain unchanged.
    """
    rng = seeded_random(4)
    cohort = []
    for department in DEPARTMENT_SPECS:
        users = sorted(
            (user for user in enterprise["users"] if user["department"] == department),
            key=lambda user: user["user_id"],
        )
        if len(users) < PERSONAL_DEVICE_USERS_PER_DEPARTMENT:
            raise ValueError(f"Insufficient personal-device users in {department}.")
        cohort.extend(rng.sample(users, PERSONAL_DEVICE_USERS_PER_DEPARTMENT))

    resources = {row["resource_id"]: row for row in enterprise["resources"]}
    events_by_user = defaultdict(list)
    for event in sorted(normal_events, key=lambda row: (row["timestamp"], row["event_id"])):
        events_by_user[event["user_id"]].append(event)
    train_end = START_DATE + timedelta(days=TRAIN_DAYS)
    calibration_end = train_end + timedelta(days=CALIBRATION_DAYS)
    selected = []
    for user in sorted(cohort, key=lambda row: row["user_id"]):
        pools = defaultdict(list)
        start_hour, end_hour = DEPARTMENT_SPECS[user["department"]]["work_hours"]
        for event in events_by_user[user["user_id"]]:
            timestamp = parse_utc(event["timestamp"])
            group = resources[event["resource_id"]]["resource_group"]
            if (
                event["ground_truth"] != "normal"
                or event["device_managed"] != "true"
                or event["device_id"].startswith("ROLLOUT-")
                or not start_hour <= timestamp.hour < end_hour
                or group not in ROLE_RESOURCE_GROUPS[user["role"]]
            ):
                continue
            if event["resource_sensitivity"] in {"public", "internal"}:
                kind = "public_or_internal"
            elif event["resource_sensitivity"] == "confidential" and event["action"] == "read":
                kind = "confidential_read"
            else:
                continue
            phase = int(timestamp.date() >= train_end) + int(timestamp.date() >= calibration_end)
            pools[(kind, phase)].append(event)
        for kind, count in PERSONAL_DEVICE_REQUESTS_PER_USER.items():
            buckets = [pools[(kind, phase)] for phase in range(3)]
            candidates = [event for bucket in buckets for event in bucket]
            if count < 3 or any(not bucket for bucket in buckets) or len(candidates) < count:
                raise ValueError(f"Insufficient {kind} requests for {user['user_id']}.")
            # Reserve one per phase, then fill the user/kind quota from all phases.
            chosen = [rng.choice(bucket) for bucket in buckets]
            chosen_ids = {event["event_id"] for event in chosen}
            remaining = [event for event in candidates if event["event_id"] not in chosen_ids]
            selected.extend(chosen + rng.sample(remaining, count - len(chosen)))

    for event in selected:
        event["device_id"] = f"PERSONAL-{event['user_id']}"
        event["device_managed"] = "false"


def generate_dataset():
    """Return the final events and entity definitions, with no intermediate I/O."""
    enterprise = generate_enterprise()
    normal = generate_normal_events(enterprise)
    anomalous = generate_anomalies(normal, enterprise)
    add_personal_device_activity(normal, enterprise)
    events = sorted(normal + anomalous, key=lambda event: (event["timestamp"], event["event_id"]))
    for event in events:
        digest = hashlib.sha256(f"{SEED}:{event['event_id']}".encode("utf-8")).hexdigest()[:16]
        event["event_id"] = f"evt_{digest}"
    return events, enterprise


def validate_dataset(events, enterprise) -> dict:
    """Check assignment dimensions and anomaly meaning before training."""
    users, resources = enterprise["users"], enterprise["resources"]
    failures = []
    if not events:
        raise ValueError("Dataset validation failed: no events.")
    if not MANDATORY_FIELDS.issubset(events[0]):
        raise ValueError("Dataset validation failed: mandatory fields missing.")
    if len(users) < 50 or len({user["department"] for user in users}) < 4:
        failures.append("At least 50 users and four departments are required.")
    if len(resources) < 25:
        failures.append("At least 25 resources are required.")
    if len(events) < 12000:
        failures.append("At least 12,000 events are required.")
    dates = {parse_utc(event["timestamp"]).date() for event in events}
    if len(dates) != NUMBER_OF_DAYS or len(dates) < 14:
        failures.append("The configured history must cover at least 14 dates.")
    anomalies = [event for event in events if event["ground_truth"] == "anomalous"]
    rate = len(anomalies) / len(events)
    if not 0.03 <= rate <= 0.10:
        failures.append("The reference anomaly rate must be between 3% and 10%.")
    categories = Counter(event["anomaly_type"] for event in anomalies)
    if len(categories) < 5:
        failures.append("At least five anomaly categories are required.")
    if len({event["event_id"] for event in events}) != len(events):
        failures.append("Event IDs are not unique.")
    if any(not event.get(field) for event in events for field in MANDATORY_FIELDS):
        failures.append("A mandatory event value is missing.")
    personal = [event for event in events if event["device_id"].startswith("PERSONAL-")]
    personal_users = {event["user_id"] for event in personal}
    personal_mix = Counter(
        "public_or_internal" if event["resource_sensitivity"] in {"public", "internal"}
        else "confidential_read" if (
            event["resource_sensitivity"] == "confidential" and event["action"] == "read"
        ) else "invalid"
        for event in personal
    )
    expected_personal_users = len(DEPARTMENT_SPECS) * PERSONAL_DEVICE_USERS_PER_DEPARTMENT
    expected_personal_mix = {
        kind: expected_personal_users * count
        for kind, count in PERSONAL_DEVICE_REQUESTS_PER_USER.items()
    }
    if (
        len(personal_users) != expected_personal_users
        or dict(personal_mix) != expected_personal_mix
        or any(
            len({event["user_id"] for event in personal if event["department"] == department})
            != PERSONAL_DEVICE_USERS_PER_DEPARTMENT
            for department in DEPARTMENT_SPECS
        )
        or any(
            event["ground_truth"] != "normal" or event["device_managed"] != "false"
            or event["device_id"] != f"PERSONAL-{event['user_id']}"
            for event in personal
        )
    ):
        failures.append("Personal-device activity does not match its documented construction.")
    resources_by_id = {resource["resource_id"]: resource for resource in resources}
    users_by_id = {user["user_id"]: user for user in users}
    events_by_user = defaultdict(list)
    for event in events:
        events_by_user[event["user_id"]].append(event)
        user = users_by_id.get(event["user_id"])
        resource = resources_by_id.get(event["resource_id"])
        if (
            not user or not resource
            or event["role"] != user["role"]
            or event["department"] != user["department"]
            or event["resource_name"] != resource["resource_name"]
            or event["resource_sensitivity"] != resource["sensitivity"]
            or event["action"] not in resource["allowed_actions"].split("|")
            or float(event["data_volume_mb"]) < 0
        ):
            failures.append(f"Invalid entity reference or request fields: {event['event_id']}.")
            break
    for event in anomalies:
        if event["anomaly_type"] == "role_resource_mismatch":
            group = resources_by_id[event["resource_id"]]["resource_group"]
            if group in ROLE_RESOURCE_GROUPS[event["role"]]:
                failures.append("A role-resource mismatch is not genuine.")
                break
        if event["anomaly_type"] == "impossible_travel":
            timestamp = parse_utc(event["timestamp"])
            if not any(
                previous["ground_truth"] == "normal"
                and previous["location"] != event["location"]
                and timedelta(0) < timestamp - parse_utc(previous["timestamp"]) <= timedelta(minutes=35)
                for previous in events_by_user[event["user_id"]]
            ):
                failures.append("An impossible-travel event lacks a valid predecessor.")
                break
    if failures:
        raise ValueError("Dataset validation failed: " + " ".join(failures))
    return {
        "status": "PASS",
        "checks": {
            "user_count": len(users),
            "department_count": len({user["department"] for user in users}),
            "role_count": len({user["role"] for user in users}),
            "resource_count": len(resources),
            "assigned_device_count": len(enterprise["devices"]),
            "event_count": len(events), "covered_dates": len(dates),
            "anomaly_count": len(anomalies), "anomaly_rate": round(rate, 6),
            "anomaly_categories": dict(sorted(categories.items())),
            "unique_event_ids": len({event["event_id"] for event in events}),
            "personal_device_activity": {
                "events": len(personal),
                "users": len(personal_users),
                "public_or_internal_events": personal_mix["public_or_internal"],
                "confidential_read_events": personal_mix["confidential_read"],
            },
        },
    }
