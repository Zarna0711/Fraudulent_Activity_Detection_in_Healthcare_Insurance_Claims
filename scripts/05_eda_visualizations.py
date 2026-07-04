"""
05_eda_visualizations.py

Same EDA charts as v1 (claim amount distribution, physician overlap) plus
two new ones enabled by the redesign: model agreement distribution, and
precision/recall by ensemble threshold. Reads directly from the claim-level
outputs instead of hardcoded manuscript numbers, so charts stay correct if
the underlying data changes.

Run:
    python 05_eda_visualizations.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_DIR = "../images"


def plot_claim_amount_pie():
    df = pd.read_csv("../data/claims_export.csv")
    bins = [-1, 5000, 10000, 15000, float("inf")]
    labels = ["Under 5000", "5001-10000", "10001-15000", "Over 15000"]
    df["bucket"] = pd.cut(df["CLM_PMT_AMT"], bins=bins, labels=labels)
    counts = df["bucket"].value_counts().reindex(labels)

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140,
            colors=["#b0bec5", "#cfd8dc", "#78909c", "#90a4ae"])
    plt.title("Claim Distribution by Payment Amount Category")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/pie_chart_claim_amount_category.png", dpi=150)
    plt.close()
    print("saved pie_chart_claim_amount_category.png")


def plot_physicians_pie():
    df = pd.read_csv("../data/claims_export.csv")
    at_set = set(df["AT_PHYSN_NPI"].dropna().astype(str)) - {"Unknown", ""}
    op_set = set(df["OP_PHYSN_NPI"].dropna().astype(str)) - {"Unknown", ""}
    common = at_set & op_set
    unique_at = at_set - op_set
    unique_op = op_set - at_set

    sizes = [len(common), len(unique_at), len(unique_op)]
    labels = ["Common Physicians", "Unique Attending Physicians", "Unique Operating Physicians"]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140,
            colors=["#c7d9cc", "#a3c4dc", "#f3d1dc"])
    plt.title("Distribution of Unique and Common Physicians")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/unique_and_common_physicians.png", dpi=150)
    plt.close()
    print("saved unique_and_common_physicians.png")


def plot_model_agreement():
    df = pd.read_csv("../data/claim_anomaly_results.csv")
    counts = df["model_agreement_count"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values, color="#5c6bc0", edgecolor="black")
    plt.xlabel("Number of models agreeing a claim is anomalous (of 4)")
    plt.ylabel("Number of claims")
    plt.title("Claim Anomaly Model Agreement Distribution")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/model_agreement_distribution.png", dpi=150)
    plt.close()
    print("saved model_agreement_distribution.png")


def plot_validation_curve():
    try:
        df = pd.read_csv("../data/validation_report.csv")
    except FileNotFoundError:
        print("validation_report.csv not found -- run 04_validation.py first (synthetic data only)")
        return
    ens = df[df["model"].str.startswith("ensemble_agree")]
    if ens.empty:
        return

    plt.figure(figsize=(8, 5))
    x = range(len(ens))
    plt.plot(x, ens["precision"], marker="o", label="Precision")
    plt.plot(x, ens["recall"], marker="o", label="Recall")
    plt.plot(x, ens["f1"], marker="o", label="F1")
    plt.xticks(x, [m.split(">=")[1] for m in ens["model"]])
    plt.xlabel("Minimum models agreeing")
    plt.ylabel("Score")
    plt.title("Precision / Recall vs. Ensemble Agreement Threshold\n(validated on synthetic injected-fraud labels)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/validation_precision_recall.png", dpi=150)
    plt.close()
    print("saved validation_precision_recall.png")


if __name__ == "__main__":
    plot_claim_amount_pie()
    plot_physicians_pie()
    plot_model_agreement()
    plot_validation_curve()
    print("\nEDA visualization pass complete.")
