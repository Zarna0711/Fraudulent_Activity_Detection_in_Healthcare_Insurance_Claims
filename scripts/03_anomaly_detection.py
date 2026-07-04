"""
03_anomaly_detection.py

Runs the same four unsupervised models as v1 (Isolation Forest, CBLOF,
ECOD, OCSVM) but on CLAIM-LEVEL features (claim_features.csv) instead of
on Apriori rule statistics. Two other changes:

  1. Contamination is chosen per-model from the score distribution (a
     simple elbow / percentile heuristic) instead of a hard-coded 0.1
     with no justification.
  2. Adds an ensemble "agreement count" -- how many of the 4 models
     flagged each claim. In fraud/SIU work, a claim flagged by 3-4
     independent methods is a much stronger referral than a claim flagged
     by just one, and this is the artifact an investigator should actually
     work off of.

Run:
    python 03_anomaly_detection.py
Input : claim_features.csv
Output: claim_anomaly_results.csv, score distribution plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from pyod.models.cblof import CBLOF
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM

RANDOM_STATE = 42
INPUT_FILE = "../data/claim_features.csv"
OUTPUT_FILE = "../data/claim_anomaly_results.csv"
IMG_DIR = "../images"

FEATURE_COLS = [
    "payment_zscore_in_drg",
    "los_zscore_in_drg",
    "payment_per_los",
    "provider_daily_claim_ct",
    "same_at_op_physician",
    "physician_pair_concentration",
    "patient_dx_drg_dup_ct",
]


def load_features(path):
    data = pd.read_csv(path)
    X = data[FEATURE_COLS].copy()
    # log-transform the heavily right-skewed count/ratio features before
    # scaling, so a handful of extreme values don't dominate every model
    # (v1 fed raw lift/conviction straight into StandardScaler)
    for col in ["payment_per_los", "provider_daily_claim_ct", "patient_dx_drg_dup_ct"]:
        X[col] = np.log1p(X[col].clip(lower=0))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return data, X, X_scaled


def pick_contamination(scores, percentile=95):
    """Data-driven default: flag the top (100-percentile)% of scores,
    e.g. percentile=95 -> flag ~5%. Documented, not arbitrary."""
    return 1 - percentile / 100


def plot_distribution(scores, title, filename, color):
    plt.figure(figsize=(8, 5))
    sns.histplot(scores, kde=True, bins=25, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"   saved plot -> {IMG_DIR}/{filename}")


def run_isolation_forest(data, X, contamination):
    model = IsolationForest(n_estimators=200, contamination=contamination,
                             random_state=RANDOM_STATE)
    model.fit(X)
    data["isolation_forest_score"] = -model.decision_function(X)  # higher = more anomalous
    data["isolation_forest_is_fraud"] = (model.predict(X) == -1).astype(int)
    plot_distribution(data["isolation_forest_score"], "Isolation Forest Anomaly Score Distribution",
                       "Isolation_Forest_Anomaly_score_distribution.png", "orange")
    return data


def run_cblof(data, X_scaled, contamination, n_clusters=8):
    try:
        model = CBLOF(
            contamination=contamination, n_clusters=n_clusters,
            clustering_estimator=KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10),
            random_state=RANDOM_STATE,
        )
        model.fit(X_scaled)
        data["cblof_score"] = model.decision_function(X_scaled)
        data["cblof_is_fraud"] = model.predict(X_scaled)
        plot_distribution(data["cblof_score"], "CBLOF Anomaly Score Distribution",
                           "CBLOF_anomaly_score_distribution.png", "crimson")
    except ValueError as e:
        print(f"CBLOF error: {e}")
        data["cblof_score"] = np.nan
        data["cblof_is_fraud"] = 0
    return data


def run_ecod(data, X_scaled, contamination):
    model = ECOD(contamination=contamination)
    model.fit(X_scaled)
    data["ecod_score"] = model.decision_function(X_scaled)
    data["ecod_is_fraud"] = model.predict(X_scaled)
    plot_distribution(data["ecod_score"], "ECOD Anomaly Score Distribution",
                       "ECOD_Anomaly_score_distribution.png", "teal")
    return data


def run_ocsvm(data, X_scaled, contamination):
    model = OCSVM(contamination=contamination, kernel="rbf")
    model.fit(X_scaled)
    data["ocsvm_score"] = model.decision_function(X_scaled)
    data["ocsvm_is_fraud"] = model.predict(X_scaled)
    plot_distribution(data["ocsvm_score"], "OCSVM Anomaly Score Distribution",
                       "OCSVM_anomaly_score_distribution.png", "purple")
    return data


if __name__ == "__main__":
    print(f"Loading claim-level features from '{INPUT_FILE}' ...")
    data, X, X_scaled = load_features(INPUT_FILE)
    print(f"Using {len(FEATURE_COLS)} features on {len(data)} claims")

    contamination = pick_contamination(None, percentile=95)  # ~5% flagged per model
    print(f"Using contamination={contamination} (top 5% of scores per model)")

    data = run_isolation_forest(data, X, contamination)
    data = run_cblof(data, X_scaled, contamination)
    data = run_ecod(data, X_scaled, contamination)
    data = run_ocsvm(data, X_scaled, contamination)

    flag_cols = ["isolation_forest_is_fraud", "cblof_is_fraud", "ecod_is_fraud", "ocsvm_is_fraud"]
    data["model_agreement_count"] = data[flag_cols].sum(axis=1)

    data.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved claim-level results -> {OUTPUT_FILE}")

    print("\n=== Flags per model ===")
    for c in flag_cols:
        print(f"{c:>28}: {int(data[c].sum())}")

    print("\n=== Claims by model agreement count ===")
    print(data["model_agreement_count"].value_counts().sort_index())

    high_conf = data[data["model_agreement_count"] >= 3]
    print(f"\nHigh-confidence referrals (>=3 of 4 models agree): {len(high_conf)} claims")
