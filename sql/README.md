# Healthcare Insurance Fraud Detection (v2)

Claim-level, ensemble-validated fraud-pattern detection on CMS DE-SynPUF
inpatient claims. A ground-up redesign of an earlier rule-level pipeline —
see `docs/METHODOLOGY.md` for exactly what changed and why. Built and run
end-to-end against real CMS DE-SynPUF Inpatient Claims (Sample 2) data.

## Pipeline

| Step | File | Description |
|---|---|---|
| 0 | `sql/01_preprocessing.sql` | SQL Server cleaning: filters on `CLM_ID` (not `DESYNPUF_ID`, a bug in the original pipeline that silently dropped a patient's entire claim history over one bad row), TRY_CAST-safe numeric handling, corrected payment-bucket boundaries |
| 0b | `data/generate_synthetic_data.py` | Generates a synthetic claims dataset with 4 known injected fraud patterns, used to validate the pipeline logic before trusting it on real data |
| 1 | `scripts/01_feature_engineering.py` | Builds 7 claim-level, peer-normalized features (payment/LOS z-scores within DRG, provider billing velocity, physician-pair concentration, duplicate billing) |
| 2 | `scripts/02_association_rules.py` | Mines association rules natively in Python (mlxtend) over a rich itemset (DRG, diagnosis, procedure, payment/LOS buckets, physician pairing) — descriptive/exploratory, not fed into the anomaly models |
| 3 | `scripts/03_anomaly_detection.py` | Runs Isolation Forest, CBLOF, ECOD, OCSVM on claim-level features; adds `model_agreement_count` ensemble score |
| 4 | `scripts/04_validation.py` | Precision/recall of each model and ensemble threshold against injected fraud labels (synthetic data only — real data has no ground truth) |
| 5 | `scripts/05_eda_visualizations.py` | Charts: claim amount distribution, physician overlap, model agreement |

## Quickstart (synthetic demo data, runs end-to-end immediately)

```bash
pip install -r requirements.txt
cd data && python generate_synthetic_data.py && cd ..
cd scripts
python 01_feature_engineering.py
python 02_association_rules.py
python 03_anomaly_detection.py
python 04_validation.py
python 05_eda_visualizations.py
```

## Using real data

1. Run `sql/01_preprocessing.sql` against the real SQL Server DE-SynPUF table
2. Export the final `SELECT` in Step 5 (right-click results grid → Save
   Results As → CSV) to `data/claims_export.csv`
3. **Add a header row manually if your SQL client exports without one** —
   SSMS's "Save Results As" can omit the header; the script needs:
   `DESYNPUF_ID,CLM_ID,PRVDR_NUM,CLM_FROM_DT,CLM_THRU_DT,CLM_PMT_AMT,NCH_PRMRY_PYR_CLM_PD_AMT,AT_PHYSN_NPI,OP_PHYSN_NPI,OT_PHYSN_NPI,CLM_UTLZTN_DAY_CNT,CLM_DRG_CD,ADMTNG_ICD9_DGNS_CD,ICD9_DGNS_CD_1,ICD9_PRCDR_CD_1`
4. Run `scripts/01`–`03`, `05`. Skip `04_validation.py` (no real labels).

## Results on real data (65,807 cleaned claims)

- **32 association rules** mined, e.g. diagnosis code V5789 (orthopedic
  aftercare) → 10+ day stay, lift 4.8 — a real, clinically-plausible pattern,
  not a physician-identity artifact
- **1,421 claims** flagged as anomalous by all 4 models; **2,393 claims**
  at the ≥3-of-4 agreement threshold — this is the working referral list
- Strongest individually-verified cases: extreme payment outliers relative
  to DRG peers (z-scores of 7-11), and physician pairs where one billing
  relationship accounts for 100% of a physician's claim volume
  (`physician_pair_concentration = 1.0`)

## Key result (synthetic validation, done first to check the method)

At ≥3-of-4 model agreement on injected fraud patterns: 57% precision, 81%
recall. Per-pattern recall: billing velocity, duplicate billing, and
upcoding all 100%; physician self-referral rings only 45% — a known,
documented weak point, not hidden.

## Honest limitations

- DE-SynPUF has no real fraud labels — flagged claims are a **referral
  prioritization list**, not a fraud determination
- `CLM_DRG_CD = '000'`/unassigned claims distort peer-group payment
  z-scores (compared against a mostly-$0 "peer group" that isn't a real
  DRG) — found during real-data review, not yet excluded in the feature
  script; treat any DRG-000 flagged claim's z-score with caution
- Physician-ring detection needs a graph-based feature to improve recall
  (currently 45% on synthetic validation)
- Synthetic validation checks pipeline *mechanics*, not real-world
  generalization — real fraud is likely subtler than the injected patterns
