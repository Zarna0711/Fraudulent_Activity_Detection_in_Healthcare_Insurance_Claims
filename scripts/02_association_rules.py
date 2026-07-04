"""
02_association_rules.py

Replaces the v1 WEKA-text-parsing step. Two changes that matter:

  1. Mines rules natively in Python (mlxtend) instead of scraping WEKA's
     console output with a regex -- reproducible, no fragile text parsing,
     and the min-support/min-confidence thresholds are explicit and logged.

  2. The itemset now includes DRG, diagnosis, procedure, and a payment
     bucket -- not just the two physician NPI fields. v1's rules were
     entirely physician-pair co-occurrences (e.g. "OP physician X ==> AT
     physician X"), which mostly reflects normal solo-practitioner billing,
     not fraud. A rule like {DRG=207, dx=D2041, payment_bucket=very_high}
     is a much more defensible fraud signal than a bare physician identity.

Run:
    python 02_association_rules.py
Input : claim_features.csv (needs CLM_DRG_CD; pulls dx/procedure/payment
        bucket back in from claims_export.csv)
Output: association_rules.csv
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

CLAIMS_FILE = "../data/claims_export.csv"
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5
OUTPUT_FILE = "../data/association_rules.csv"


def build_transactions(df):
    df = df.copy()
    df["payment_bucket"] = pd.qcut(
        df["CLM_PMT_AMT"], q=4, labels=["pmt_low", "pmt_med", "pmt_high", "pmt_very_high"]
    )
    df["los_bucket"] = pd.cut(
        df["CLM_UTLZTN_DAY_CNT"], bins=[0, 2, 5, 10, 1000],
        labels=["los_1-2", "los_3-5", "los_6-10", "los_10+"]
    )

    items_per_claim = []
    for _, row in df.iterrows():
        items = [
            f"DRG={row['CLM_DRG_CD']}",
            f"DX={row['ICD9_DGNS_CD_1']}",
            f"PRC={row['ICD9_PRCDR_CD_1']}",
            str(row["payment_bucket"]),
            str(row["los_bucket"]),
            f"PHYS_PAIR={row['AT_PHYSN_NPI']}_{row['OP_PHYSN_NPI']}"
            if row["AT_PHYSN_NPI"] != row["OP_PHYSN_NPI"] else "SOLO_PHYSICIAN",
        ]
        items_per_claim.append(items)
    return items_per_claim, df


def main():
    df = pd.read_csv(CLAIMS_FILE)
    transactions, df_annotated = build_transactions(df)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)

    print(f"Mining rules across {len(onehot)} claims, {onehot.shape[1]} distinct items "
          f"(min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE})")

    frequent_itemsets = apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)
    if frequent_itemsets.empty:
        print("No frequent itemsets found at this support threshold -- lower MIN_SUPPORT.")
        return

    rules = association_rules(frequent_itemsets, metric="confidence",
                               min_threshold=MIN_CONFIDENCE)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))

    keep = ["antecedents", "consequents", "antecedent support", "consequent support",
            "support", "confidence", "lift", "leverage", "conviction"]
    rules = rules[keep]
    rules.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(rules)} rules -> {OUTPUT_FILE}")
    print(rules.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
