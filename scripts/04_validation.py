"""
04_validation.py

v1 had no way to check whether the pipeline actually finds fraud, because
CMS DE-SynPUF is synthetic and has no real fraud labels. This script closes
that gap using the known injected-fraud rows from generate_synthetic_data.py
(billing_velocity, upcoding, physician_ring, duplicate_billing) as ground
truth, purely for validating the METHOD -- these labels are never seen by
the models, only used here afterwards to score them.

This does NOT validate fraud detection on real data. It validates that the
pipeline's logic (features + models + agreement threshold) can recover
known synthetic patterns, which is the most defensible thing you can claim
without real labeled cases.

Run:
    python 04_validation.py
Input : claim_anomaly_results.csv (must include is_injected_fraud, fraud_pattern)
Output: validation_report.csv + printed precision/recall summary
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

INPUT_FILE = "../data/claim_anomaly_results.csv"
OUTPUT_FILE = "../data/validation_report.csv"


def main():
    df = pd.read_csv(INPUT_FILE)
    if "is_injected_fraud" not in df.columns:
        print("No is_injected_fraud column found -- this script only runs against "
              "the synthetic validation dataset, not real claims.")
        return

    y_true = df["is_injected_fraud"]

    print("=== Per-model performance vs. injected fraud labels ===")
    rows = []
    for model in ["isolation_forest", "cblof", "ecod", "ocsvm"]:
        y_pred = df[f"{model}_is_fraud"]
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append({"model": model, "precision": p, "recall": r, "f1": f1})
        print(f"{model:>18}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")

    print("\n=== Ensemble (agreement >= N models) vs. injected fraud labels ===")
    for thresh in [1, 2, 3, 4]:
        y_pred = (df["model_agreement_count"] >= thresh).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append({"model": f"ensemble_agree>={thresh}", "precision": p, "recall": r, "f1": f1})
        print(f"agreement>={thresh}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}  "
              f"(flags {int(y_pred.sum())} claims)")

    print("\n=== Recall by fraud pattern (agreement >= 3) ===")
    df["flagged"] = (df["model_agreement_count"] >= 3).astype(int)
    pattern_recall = (
        df[df["is_injected_fraud"] == 1]
        .groupby("fraud_pattern")["flagged"]
        .mean()
    )
    print(pattern_recall)

    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
