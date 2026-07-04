"""
01_feature_engineering.py

Builds CLAIM-LEVEL features for anomaly detection -- this is the core fix
vs. v1, which ran anomaly detection on Apriori RULE statistics (75 rows)
instead of on the claims/providers themselves. Anomaly scores here map
directly back to a DESYNPUF_ID / CLM_ID / PRVDR_NUM an investigator can
pull and review.

Features (all peer-normalized where it matters):
  - payment_zscore_in_drg    : how unusual CLM_PMT_AMT is vs. its DRG peers
  - los_zscore_in_drg        : how unusual length-of-stay is vs. its DRG peers
  - payment_per_los          : payment efficiency
  - provider_daily_claim_ct  : claims billed by this claim's provider on the
                                same day (billing-velocity signal)
  - same_at_op_physician     : 1 if attending == operating (NOT inherently
                                suspicious on its own -- kept as context,
                                not as the sole signal like v1)
  - physician_pair_freq      : how often this exact AT/OP physician pair
                                co-occurs, relative to each physician's
                                overall claim volume (catches self-referral
                                rings without flagging every solo
                                practitioner)
  - patient_dx_drg_dup_ct    : count of claims for the same patient with the
                                same diagnosis+DRG within a 5-day window
                                (duplicate-billing signal)

Run:
    python 01_feature_engineering.py
Input : claims_export.csv   (from 01_preprocessing.sql export, or the
                              synthetic generator)
Output: claim_features.csv
"""

import pandas as pd
import numpy as np

INPUT_FILE = "../data/claims_export.csv"
OUTPUT_FILE = "../data/claim_features.csv"


def load(path):
    df = pd.read_csv(path, parse_dates=["CLM_FROM_DT", "CLM_THRU_DT"],
                      dtype={"AT_PHYSN_NPI": str, "OP_PHYSN_NPI": str, "OT_PHYSN_NPI": str,
                             "CLM_DRG_CD": str, "PRVDR_NUM": str, "DESYNPUF_ID": str,
                             "CLM_ID": str, "ICD9_DGNS_CD_1": str, "ADMTNG_ICD9_DGNS_CD": str,
                             "ICD9_PRCDR_CD_1": str})
    return df


def add_drg_peer_features(df):
    grp = df.groupby("CLM_DRG_CD")
    df["drg_payment_mean"] = grp["CLM_PMT_AMT"].transform("mean")
    df["drg_payment_std"] = grp["CLM_PMT_AMT"].transform("std").replace(0, np.nan)
    df["payment_zscore_in_drg"] = (
        (df["CLM_PMT_AMT"] - df["drg_payment_mean"]) / df["drg_payment_std"]
    ).fillna(0)

    df["drg_los_mean"] = grp["CLM_UTLZTN_DAY_CNT"].transform("mean")
    df["drg_los_std"] = grp["CLM_UTLZTN_DAY_CNT"].transform("std").replace(0, np.nan)
    df["los_zscore_in_drg"] = (
        (df["CLM_UTLZTN_DAY_CNT"] - df["drg_los_mean"]) / df["drg_los_std"]
    ).fillna(0)

    df["payment_per_los"] = df["CLM_PMT_AMT"] / df["CLM_UTLZTN_DAY_CNT"].clip(lower=1)
    return df


def add_provider_velocity(df):
    daily = (
        df.groupby(["PRVDR_NUM", "CLM_FROM_DT"])["CLM_ID"]
        .transform("count")
    )
    df["provider_daily_claim_ct"] = daily
    return df


def add_physician_pair_features(df):
    df["AT_PHYSN_NPI"] = df["AT_PHYSN_NPI"].fillna("Unknown").astype(str)
    df["OP_PHYSN_NPI"] = df["OP_PHYSN_NPI"].fillna("Unknown").astype(str)

    df["same_at_op_physician"] = (df["AT_PHYSN_NPI"] == df["OP_PHYSN_NPI"]).astype(int)

    pair_counts = df.groupby(["AT_PHYSN_NPI", "OP_PHYSN_NPI"])["CLM_ID"].transform("count")
    at_totals = df.groupby("AT_PHYSN_NPI")["CLM_ID"].transform("count")
    op_totals = df.groupby("OP_PHYSN_NPI")["CLM_ID"].transform("count")

    # Fraction of each physician's volume concentrated in this one pairing --
    # near 1.0 means "this physician (almost) only ever bills with this other
    # physician," which is the self-referral-ring signal. Excludes the
    # trivial same_at_op_physician=1 rows (solo billing), which are a
    # separate, non-suspicious pattern.
    df["physician_pair_concentration"] = np.where(
        df["same_at_op_physician"] == 1,
        0.0,
        pair_counts / np.minimum(at_totals, op_totals),
    )
    return df


def add_duplicate_billing_features(df):
    # Fill blanks in the grouping keys first -- pandas groupby silently DROPS
    # rows where any key is NaN, which would make dup_counts shorter than df
    # and crash the assignment below. Real DE-SynPUF data can have blank
    # ICD9_DGNS_CD_1 values that the SQL 'Unknown' cleanup didn't cover.
    df["ICD9_DGNS_CD_1"] = df["ICD9_DGNS_CD_1"].fillna("Unknown").astype(str)
    df["CLM_DRG_CD"] = df["CLM_DRG_CD"].fillna("Unknown").astype(str)
    df["DESYNPUF_ID"] = df["DESYNPUF_ID"].fillna("Unknown").astype(str)

    df = df.sort_values(["DESYNPUF_ID", "ICD9_DGNS_CD_1", "CLM_DRG_CD", "CLM_FROM_DT"])
    dup_counts = []
    window = pd.Timedelta(days=5)
    for _, g in df.groupby(["DESYNPUF_ID", "ICD9_DGNS_CD_1", "CLM_DRG_CD"], dropna=False):
        dates = g["CLM_FROM_DT"].values
        counts = []
        for d in dates:
            if pd.isna(d):
                counts.append(0)
                continue
            counts.append(int(((dates >= d - window.to_timedelta64()) &
                                (dates <= d + window.to_timedelta64())).sum() - 1))
        dup_counts.extend(counts)
    df["patient_dx_drg_dup_ct"] = dup_counts
    return df


def main():
    df = load(INPUT_FILE)
    print(f"Loaded {len(df)} claims")

    df = add_drg_peer_features(df)
    df = add_provider_velocity(df)
    df = add_physician_pair_features(df)
    df = add_duplicate_billing_features(df)

    feature_cols = [
        "payment_zscore_in_drg",
        "los_zscore_in_drg",
        "payment_per_los",
        "provider_daily_claim_ct",
        "same_at_op_physician",
        "physician_pair_concentration",
        "patient_dx_drg_dup_ct",
    ]

    keep_cols = [
        "DESYNPUF_ID", "CLM_ID", "PRVDR_NUM", "AT_PHYSN_NPI", "OP_PHYSN_NPI",
        "CLM_DRG_CD", "CLM_PMT_AMT", "CLM_UTLZTN_DAY_CNT",
    ] + feature_cols

    # keep injected-fraud labels ONLY if present (synthetic validation run) --
    # never used as a model input, only for scoring afterwards.
    if "is_injected_fraud" in df.columns:
        keep_cols += ["is_injected_fraud", "fraud_pattern"]

    out = df[keep_cols].copy()
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved claim-level features -> {OUTPUT_FILE}")
    print(f"Feature columns: {feature_cols}")


if __name__ == "__main__":
    main()
