"""Scenario and model configuration for the self-contained experiment."""

from __future__ import annotations

from datetime import date
from pathlib import Path

SEED = 20260905
START_DATE = date(2026, 8, 3)  # Monday
NUMBER_OF_DAYS = 14

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DEPARTMENT_SPECS = {
    "Engineering": {
        "count": 18,
        "roles": ["software_engineer", "senior_engineer", "engineering_manager"],
        "role_weights": [0.67, 0.22, 0.11],
        "home_office": "Athens",
        "weekday_events": (19, 27),
        "weekend_events": (1, 4),
        "work_hours": (8, 19),
    },
    "Finance": {
        "count": 14,
        "roles": ["financial_analyst", "accountant", "finance_manager"],
        "role_weights": [0.43, 0.43, 0.14],
        "home_office": "Athens",
        "weekday_events": (18, 25),
        "weekend_events": (0, 2),
        "work_hours": (8, 18),
    },
    "Human Resources": {
        "count": 10,
        "roles": ["hr_specialist", "recruiter", "hr_manager"],
        "role_weights": [0.5, 0.3, 0.2],
        "home_office": "Thessaloniki",
        "weekday_events": (16, 23),
        "weekend_events": (0, 2),
        "work_hours": (8, 18),
    },
    "Sales": {
        "count": 12,
        "roles": ["account_executive", "sales_operations", "sales_manager"],
        "role_weights": [0.67, 0.17, 0.16],
        "home_office": "Thessaloniki",
        "weekday_events": (18, 26),
        "weekend_events": (1, 4),
        "work_hours": (7, 20),
    },
    "IT Operations": {
        "count": 6,
        "roles": ["system_administrator", "security_analyst", "it_manager"],
        "role_weights": [0.5, 0.33, 0.17],
        "home_office": "Athens",
        "weekday_events": (21, 29),
        "weekend_events": (5, 10),
        "work_hours": (7, 21),
    },
}

ROLE_RESOURCE_GROUPS = {
    "software_engineer": ["common", "engineering"],
    "senior_engineer": ["common", "engineering"],
    "engineering_manager": ["common", "engineering", "management"],
    "financial_analyst": ["common", "finance"],
    "accountant": ["common", "finance"],
    "finance_manager": ["common", "finance", "management"],
    "hr_specialist": ["common", "hr"],
    "recruiter": ["common", "hr"],
    "hr_manager": ["common", "hr", "management"],
    "account_executive": ["common", "sales"],
    "sales_operations": ["common", "sales"],
    "sales_manager": ["common", "sales", "management"],
    "system_administrator": ["common", "it", "privileged"],
    "security_analyst": ["common", "it", "privileged"],
    "it_manager": ["common", "it", "privileged", "management"],
}

RESOURCE_SPECS = [
    ("RES-001", "Corporate Website", "common", "public", ["read"]),
    ("RES-002", "Intranet", "common", "internal", ["read"]),
    ("RES-003", "Email", "common", "internal", ["read", "send"]),
    ("RES-004", "Calendar", "common", "internal", ["read", "write"]),
    ("RES-005", "Chat", "common", "internal", ["read", "send"]),
    ("RES-006", "Document Portal", "common", "confidential", ["read", "upload", "download"]),
    ("RES-007", "Source Repository", "engineering", "confidential", ["read", "clone", "push"]),
    ("RES-008", "CI Pipeline", "engineering", "confidential", ["read", "execute", "configure"]),
    ("RES-009", "Issue Tracker", "engineering", "internal", ["read", "write"]),
    ("RES-010", "Artifact Registry", "engineering", "confidential", ["read", "upload", "download"]),
    ("RES-011", "Engineering Wiki", "engineering", "internal", ["read", "write"]),
    ("RES-012", "General Ledger", "finance", "privileged", ["read", "write", "approve"]),
    ("RES-013", "Expense Platform", "finance", "confidential", ["read", "submit", "approve"]),
    ("RES-014", "Payroll Reports", "finance", "privileged", ["read", "download"]),
    ("RES-015", "Billing System", "finance", "confidential", ["read", "write", "approve"]),
    ("RES-016", "Treasury Dashboard", "finance", "privileged", ["read", "execute"]),
    ("RES-017", "HR Information System", "hr", "privileged", ["read", "write"]),
    ("RES-018", "Applicant Tracking", "hr", "confidential", ["read", "write", "download"]),
    ("RES-019", "Benefits Portal", "hr", "confidential", ["read", "write"]),
    ("RES-020", "Performance Records", "hr", "privileged", ["read", "write", "download"]),
    ("RES-021", "CRM", "sales", "confidential", ["read", "write", "export"]),
    ("RES-022", "Sales Analytics", "sales", "confidential", ["read", "download"]),
    ("RES-023", "Contract Repository", "sales", "confidential", ["read", "upload", "download"]),
    ("RES-024", "Pricing Tool", "sales", "internal", ["read", "write"]),
    ("RES-025", "Identity Administration", "privileged", "privileged", ["read", "create", "modify"]),
    ("RES-026", "Production Console", "privileged", "privileged", ["read", "execute", "configure"]),
    ("RES-027", "Security Information System", "it", "privileged", ["read", "investigate", "export"]),
    ("RES-028", "Device Management", "it", "privileged", ["read", "enrol", "wipe"]),
    ("RES-029", "Network Monitoring", "it", "confidential", ["read", "investigate"]),
    ("RES-030", "Executive Dashboard", "management", "confidential", ["read", "download"]),
]

NORMAL_NETWORKS = {
    "Athens": ["athens_office", "corporate_vpn", "home_gr"],
    "Thessaloniki": ["thessaloniki_office", "corporate_vpn", "home_gr"],
}

TRAVEL_LOCATIONS = ["London", "Singapore", "New York", "São Paulo"]

ANOMALY_TARGETS = {
    "impossible_travel": 130,
    "off_hours_privileged_access": 130,
    "first_seen_device_sensitive_action": 130,
    "role_resource_mismatch": 130,
    "bulk_download_spike": 180,
    "privilege_escalation_sequence": 130,
}

# Synthetic personal-device activity
PERSONAL_DEVICE_USERS_PER_DEPARTMENT = 2
PERSONAL_DEVICE_REQUESTS_PER_USER = {
    "public_or_internal": 10,
    "confidential_read": 4,
}

SENSITIVITY_RANK = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "privileged": 3,
}

TRAIN_DAYS = 9
CALIBRATION_DAYS = 2
TEST_DAYS = 3

MODEL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "outside_role_hours",
    "sensitivity_rank",
    "device_managed_numeric",
    "device_seen_before",
    "device_prior_count",
    "user_resource_prior_count",
    "role_resource_prior_count",
    "role_resource_allowed",
    "location_seen_before",
    "location_change_60m",
    "minutes_since_previous_event",
    "recent_failures_24h",
    "user_events_1h",
    "user_volume_1h_mb",
    "large_transfer",
    "privileged_action",
    "network_zone",
    "action",
    "resource_sensitivity",
    "department",
]
