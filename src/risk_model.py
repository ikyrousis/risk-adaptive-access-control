"""Fit and calibrate one supervised risk model held in memory."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODEL_FEATURES, SEED, START_DATE, TRAIN_DAYS

C_CANDIDATES = (0.1, 0.3, 0.7, 1.0, 3.0, 10.0)
CATEGORICAL_FEATURES = [
    "network_zone",
    "action",
    "resource_sensitivity",
    "department",
]
NUMERIC_FEATURES = [f for f in MODEL_FEATURES if f not in CATEGORICAL_FEATURES]


def choose_thresholds(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Select calibration thresholds by an explicit operational cost function.

    The returned tie counts record how much of the grid the cost function can
    actually discriminate: pairs sharing a decision signature partition the
    calibration set identically and are therefore operationally equivalent.
    """
    grid = []
    for challenge in range(10, 61, 5):
        for block in range(max(challenge + 10, 50), 96, 5):
            decisions = np.where(scores >= block, "block", np.where(scores >= challenge, "challenge", "allow"))
            cost = 0.0
            for actual, decision in zip(y_true, decisions):
                if actual == 1 and decision == "allow":
                    cost += 10.0
                elif actual == 1 and decision == "challenge":
                    cost += 2.0
                elif actual == 0 and decision == "challenge":
                    cost += 1.0
                elif actual == 0 and decision == "block":
                    cost += 6.0
            grid.append({
                "challenge": float(challenge), "block": float(block), "cost": cost,
                "signature": decisions.astype("S9").tobytes(),
            })
    best = min(grid, key=lambda row: (row["cost"], row["challenge"], row["block"]))
    cost_ties = [row for row in grid if row["cost"] == best["cost"]]
    partition_ties = [row for row in cost_ties if row["signature"] == best["signature"]]
    return {
        "challenge_threshold": best["challenge"],
        "block_threshold": best["block"],
        "calibration_operational_cost": best["cost"],
        "grid_pairs_evaluated": len(grid),
        "pairs_at_minimum_cost": len(cost_ties),
        "pairs_selecting_the_identical_partition": len(partition_ties),
        "identical_partition_ranges": {
            "challenge": [
                min(row["challenge"] for row in partition_ties),
                max(row["challenge"] for row in partition_ties),
            ],
            "block": [
                min(row["block"] for row in partition_ties),
                max(row["block"] for row in partition_ties),
            ],
        },
        "tie_break": "lowest cost, then lowest challenge threshold, then lowest block threshold",
    }


def _build_pipeline(c: float) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=SEED,
                    C=c,
                ),
            ),
        ]
    )


def select_regularization(frame: pd.DataFrame) -> dict:
    """Compare C values on three training-only, expanding calendar windows."""
    if TRAIN_DAYS < 7:
        raise ValueError("Regularisation selection requires at least seven training days.")
    train = frame[frame["split"] == "train"].copy()
    timestamps = pd.to_datetime(train["timestamp"], utc=True)
    start = pd.Timestamp(START_DATE, tz="UTC")
    end = start + pd.Timedelta(days=TRAIN_DAYS)
    if train.empty or timestamps.isna().any() or ((timestamps < start) | (timestamps >= end)).any():
        raise ValueError("Training timestamps must lie within the configured training period.")

    blocks, folds = [], []
    for fold, training_days in enumerate((TRAIN_DAYS - 6, TRAIN_DAYS - 4, TRAIN_DAYS - 2), 1):
        training_end = start + pd.Timedelta(days=training_days)
        validation_end = training_end + pd.Timedelta(days=2)
        fitting = train[timestamps < training_end]
        validation = train[(timestamps >= training_end) & (timestamps < validation_end)]
        y_fitting = (fitting["ground_truth"] == "anomalous").astype(int)
        y_validation = (validation["ground_truth"] == "anomalous").astype(int)
        if fitting.empty or validation.empty or y_fitting.nunique() != 2 or y_validation.nunique() != 2:
            raise ValueError(f"Fold {fold} requires non-empty fitting and validation blocks with both classes.")
        blocks.append((fitting, validation, y_fitting, y_validation))
        folds.append({
            "fold": fold,
            "train_start": start.date().isoformat(),
            "train_end": (training_end - pd.Timedelta(days=1)).date().isoformat(),
            "validation_start": training_end.date().isoformat(),
            "validation_end": (validation_end - pd.Timedelta(days=1)).date().isoformat(),
            "training_events": int(len(fitting)),
            "validation_events": int(len(validation)),
            "positive_training_events": int(y_fitting.sum()),
            "positive_validation_events": int(y_validation.sum()),
        })

    candidate_scores = []
    for c in C_CANDIDATES:
        fold_scores = []
        for fitting, validation, y_fitting, y_validation in blocks:
            pipeline = _build_pipeline(c)
            pipeline.fit(fitting[MODEL_FEATURES], y_fitting)
            raw = pipeline.predict_proba(validation[MODEL_FEATURES])[:, 1]
            fold_scores.append(float(average_precision_score(y_validation, raw)))
        candidate_scores.append({
            "C": float(c),
            "fold_average_precision": fold_scores,
            "mean_average_precision": float(np.mean(fold_scores)),
        })
    selected = min(candidate_scores, key=lambda row: (-row["mean_average_precision"], row["C"]))
    return {
        "method": "training-only expanding-window validation",
        "metric": "average_precision",
        "score_source": "uncalibrated model probabilities",
        "candidates": list(C_CANDIDATES),
        "fold_weighting": "equal",
        "date_convention": "inclusive UTC calendar dates",
        "tie_break": "smaller C on exactly equal unrounded mean average precision",
        "calibration_days_used": False,
        "evaluation_days_used": False,
        "folds": folds,
        "candidate_scores": candidate_scores,
        "selected_C": selected["C"],
        "selected_mean_average_precision": selected["mean_average_precision"],
    }


def train_model(frame: pd.DataFrame) -> dict:
    """Select C within training days, then fit, calibrate and choose thresholds."""
    train = frame[frame["split"] == "train"].copy()
    calibration = frame[frame["split"] == "calibration"].copy()
    test = frame[frame["split"] == "test"].copy()

    y_train = (train["ground_truth"] == "anomalous").astype(int)
    y_calibration = (calibration["ground_truth"] == "anomalous").astype(int)

    selection = select_regularization(train)
    pipeline = _build_pipeline(selection["selected_C"])
    pipeline.fit(train[MODEL_FEATURES], y_train)

    raw_calibration = pipeline.predict_proba(calibration[MODEL_FEATURES])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_calibration, y_calibration)
    calibration_scores = calibrator.predict(raw_calibration) * 100
    thresholds = choose_thresholds(y_calibration.to_numpy(), calibration_scores)

    model = {
        "pipeline": pipeline,
        "calibrator": calibrator,
        "feature_columns": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "thresholds": thresholds,
    }
    model["metadata"] = {
        "approach": "supervised calibrated logistic regression",
        "training_events": int(len(train)),
        "calibration_events": int(len(calibration)),
        "evaluation_events": int(len(test)),
        "distinct_calibration_scores": int(len(np.unique(calibration_scores))),
        "positive_training_events": int(y_train.sum()),
        "positive_calibration_events": int(y_calibration.sum()),
        "regularization_selection": selection,
        "threshold_selection": {
            "method": "minimum operational cost on chronological calibration set",
            "false_negative_allow_cost": 10,
            "true_anomaly_challenge_cost": 2,
            "normal_challenge_cost": 1,
            "normal_block_cost": 6,
            **thresholds,
        },
        "excluded_from_features": [
            "ground_truth",
            "anomaly_type",
            "anomaly_campaign_id",
            "event_id",
            "user_id",
            "timestamp",
        ],
    }
    return model


def score_requests(model: dict, frame: pd.DataFrame) -> np.ndarray:
    """Return the rounded 0-100 score used by decisions and reported metrics."""
    raw = model["pipeline"].predict_proba(frame[MODEL_FEATURES])[:, 1]
    return np.round(model["calibrator"].predict(raw) * 100, 2)


def positive_contributions(model: dict, frame: pd.DataFrame) -> list[str]:
    """Explain raw logistic score direction, not additive calibrated risk points."""
    preprocessing = model["pipeline"].named_steps["preprocess"]
    classifier = model["pipeline"].named_steps["classifier"]
    transformed = preprocessing.transform(frame[MODEL_FEATURES])
    names = preprocessing.get_feature_names_out()
    coefficients = classifier.coef_[0]
    output = []
    for values in transformed:
        contributions = values * coefficients
        contributors = []
        for position in np.argsort(contributions)[::-1]:
            if contributions[position] <= 0 or len(contributors) >= 4:
                break
            name = names[position].replace("numeric__", "").replace("categorical__", "")
            contributors.append(f"{name}:{contributions[position]:.3f}")
        output.append("|".join(contributors) or "intercept_or_negative_evidence")
    return output
