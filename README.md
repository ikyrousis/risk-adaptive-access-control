# Risk-Adaptive Access Control

This project generates synthetic enterprise access logs and compares a static baseline with calibrated risk scoring and contextual decisions. Each evaluated request receives an allow, challenge or block outcome, with its model risk score and explanation recorded for inspection. It exports evaluation results before and after a simulated device rollout without enforcing live access controls.

## Setup

Use Python 3.12 or newer and run commands from root. Create a fresh environment and install the pinned runtime and test dependencies.

```bash
python -m venv .venv
```

Activate the environment using the appropriate command:

| Platform | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Git Bash with Windows Python | `source .venv/Scripts/activate` |

```bash
python -m pip install -r requirements.txt
```

No external dataset, account, API key or server is required. Use the same interpreter for installation and execution.

## Entry points and outputs

| Entry point | Role | Persistent outputs |
|---|---|---|
| `python run_pipeline.py` | Generate, validate, train, calibrate, evaluate and test. | Two CSVs, one JSON and three plots. |
| `python -m pytest` | Test source and existing outputs. | None. |

## Pipeline CLI

Each run regenerates data, refits the model and overwrites the six output files. Run the complete pipeline without flags:

```bash
python run_pipeline.py
```

Completion includes export, plotting and tests, ending with `Pipeline completed successfully.` 

## Configuration

Edit settings in the source files below. Keep the seed and package versions unchanged for reproduction.

| Location | Settings |
|---|---|
| `config.py` | Seed `20260905`, start date `2026-08-03`, 14 days, train/calibration/test split 9/2/3, organisation, resources, permissions, networks, anomaly targets and model inputs. |
| `src/dataset.py` | Sampling distributions and construction routines. |
| `src/risk_model.py` | Preprocessing, classifier, regularisation grid, calibration and score thresholds. |
| `src/policy.py` | Static rules, volume-threshold search and contextual controls. |
| `src/evaluation.py` | Rollout fraction, replay offset, metrics and plots. |

Additional generator parameters as follows:

| Parameter | Value |
|---|---|
| Role weights | Engineering 0.67/0.22/0.11, Finance 0.43/0.43/0.14, Human Resources 0.50/0.30/0.20, Sales 0.67/0.17/0.16, IT Operations 0.50/0.33/0.17. |
| Per-resource weights | Common 4, departmental 3, management 1, privileged 1.5. |
| Transfer payloads | `lognormvariate(1.8, 0.65)` for download/export/clone, `(1.1, 0.55)` for upload/push. |
| Benign payloads | Large transfers 80–420 MB, project reads and overnight maintenance 0.05–1.00 MB. |
| Travel eligibility | Engineering/Sales user-days with at least 10 requests, cycling through London, Singapore, New York and São Paulo. |
| Anomaly payloads | Off-hours privileged 0.05–4.00 MB, privilege-escalation 0.02–0.30 MB. |
| Bulk timing | `campaign_start + i * randint(8, 25)` seconds, with zero-based index `i` and a fresh draw per event. Consecutive gaps are not necessarily 8–25 seconds. |

Seed offsets are `+1` entities, `+2` routine/exceptions, `+3` anomalies, `+4` personal devices and `+11` rollout. The final bulk campaign is capped by the remaining quota and may contain fewer than nine events.

Regularisation compares `C = 0.1, 0.3, 0.7, 1, 3, 10` on three expanding windows within the nine training days: fit on days 1–3, 1–5 and 1–7, then validate on days 4–5, 6–7 and 8–9 respectively. Selection maximises equally weighted mean raw-score average precision, with smaller `C` breaking exact ties, before refitting on all training days.

## Tests

All six outputs must exist before running tests separately. The pipeline runs them automatically after export.

```bash
python -m pytest
```

Pytest runs 50 tests covering generation, leakage, regularisation selection, policies, drift and output/documentation contracts. Some tests regenerate data or fit models in memory without replacing saved evidence, while Python and pytest caches are separate from experiment outputs.

## Generated files

These are the only persistent experiment outputs.

| File | Contents |
|---|---|
| `data/access_events.csv` | Labelled dataset, 18 columns and 14,772 rows under the supplied configuration. |
| `results/decisions.csv` | Decisions and explanations, 31 columns and 1,748 original plus 1,748 replay rows. |
| `results/results.json` | Validation, metadata, thresholds, metrics, examples and drift comparison. |
| `results/plots/baseline-vs-ai.png` | Original-holdout precision, recall and F1. |
| `results/plots/risk-score-distribution.png` | Original-holdout scores by ground-truth label. |
| `results/plots/drift-intervention-rate.png` | Before/after intervention rates for both systems. |

Plots use the Matplotlib `Agg` backend. CSV Boolean values are `true`/`false`, and the JSON records the dataset's SHA-256.

### Combined decisions

Decision rows retain the report's 18 event fields and add the following 13. Code/contribution lists use pipe delimiters.

| Field | Meaning |
|---|---|
| `scenario` | `pre_drift_test` or `new_device_rollout`. |
| `source_event_id` | Original ID pairing original/replay requests. |
| `risk_score` | Calibrated 0–100 score, rounded to two decimals. |
| `baseline_decision` | Static `allow`, `challenge` or `block`. |
| `baseline_reason_codes` | Triggered static-rule codes. |
| `model_decision` | Initial score-band decision before controls. |
| `final_decision` | Final AI-assisted `allow`, `challenge` or `block`. |
| `decision_source` | `model`, `contextual_override`, `mandatory_rule` or `fail_safe`. |
| `override_codes` | Applied control codes, or `NONE`. |
| `step_up_control` | Experimental logging, MFA, approval or denial instruction. |
| `reason_codes` | Context indicators and control reasons. |
| `top_model_contributors` | Up to four positive raw logistic `name:value` contributions, or `intercept_or_negative_evidence`. |
| `explanation` | Score, band, final decision, source, indicators and contributions. |

### Consolidated results

The seven top-level objects are described below.

| Object | Contents |
|---|---|
| `run` | Seed, start date and duration. |
| `dataset` | Validation, counts, dates, anomalies, identifiers, hash and personal-device counts. |
| `split` | Normal/anomalous counts for `train`, `calibration` and `test`. |
| `baseline` | Rules, precedence, history and threshold-selection evidence. |
| `model` | Approach, regularisation-selection evidence, support, features, exclusions and score-threshold costs/results. |
| `evaluation` | Metrics, decisions, overrides, department results and explanations. |
| `drift` | Cohort, replay settings, comparisons and adaptation metadata. |

`model.regularization_selection` records the candidate grid, inclusive UTC fold dates, event/anomaly counts, per-fold and mean average precision, tie-break and selected `C`. Calibration and evaluation records are excluded from this selection.

Evaluation metrics use four decimals, while regularisation-selection scores retain full precision. Rates/bounds use proportions, `pr_auc` is average precision, and confusion matrices use `[[TN, FP], [FN, TP]]`, with challenge/block treated as positive interventions.

`personal_device_activity` counts events, users, public/internal requests and confidential reads. Department rows include support, `ai_`/`baseline_` metrics and selected nominal 95% Wilson bounds.

### In-memory fields

User records add `home_office` (city), `remote_work_ratio` (probability) and `primary_device_id` (laptop). Devices add `device_type` (`managed_laptop`/`managed_mobile`), `managed` and `assigned` (Boolean text).

Resources add `resource_group` (permission group), `sensitivity` (label) and `allowed_actions` (pipe-delimited operations). Feature rows retain event metadata and `split` (`train`/`calibration`/`test`), but only `MODEL_FEATURES` enters the classifier.

## Project layout

```text
.
├── README.md
├── config.py
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── features.py
│   ├── risk_model.py
│   ├── policy.py
│   └── evaluation.py
├── tests/
│   └── test_pipeline.py
├── data/
│   └── access_events.csv
└── results/
    ├── decisions.csv
    ├── results.json
    └── plots/
        ├── baseline-vs-ai.png
        ├── risk-score-distribution.png
        └── drift-intervention-rate.png
```
