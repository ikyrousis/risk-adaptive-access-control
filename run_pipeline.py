"""Rebuild the experiment, export six files, then run the acceptance tests."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys

sys.dont_write_bytecode = True

from config import DATA_DIR, RESULTS_DIR, ROOT
from src.dataset import EVENT_FIELDS
from src.evaluation import DECISION_FIELDS, create_plots, run_experiment


def write_csv(path, rows, fields):
    if len(fields) != len(set(fields)):
        raise ValueError("Duplicate CSV column names.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if sys.argv[1:]:
        raise SystemExit("This script accepts no arguments. Run: python run_pipeline.py")

    print("Generating data, engineering prior-only features and evaluating frozen policies.", flush=True)
    events, decisions, summary = run_experiment()
    dataset_path = DATA_DIR / "access_events.csv"
    write_csv(dataset_path, events, EVENT_FIELDS)
    summary["dataset"]["checks"]["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    write_csv(
        RESULTS_DIR / "decisions.csv",
        [dict(record, risk_score=f"{record['risk_score']:.2f}") for record in decisions],
        DECISION_FIELDS,
    )
    with (RESULTS_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    create_plots(summary, decisions, RESULTS_DIR / "plots")
    metrics = summary["evaluation"]
    print(
        f"Dataset: {len(events):,} events; evaluation: {metrics['evaluation_set']['events']:,} requests.\n"
        f"AI F1={metrics['ai_assisted']['f1']:.4f}; "
        f"baseline F1={metrics['static_baseline']['f1']:.4f}; "
        f"AI PR-AUC={metrics['ai_assisted']['pr_auc']:.4f}.\n"
        "Wrote access_events.csv, decisions.csv, results.json and three plots.",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "-B", "-m", "pytest", str(ROOT / "tests")],
        cwd=ROOT, check=True,
    )
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
