/* =====================================================================
   HEALTHCARE INSURANCE FRAUD DETECTION - DATA PREPROCESSING (v2)
   Source: CMS 2008-2010 DE-SynPUF Inpatient Claims (Sample 2)

   Fixes vs. v1:
     - Row-level filtering on CLM_ID instead of DESYNPUF_ID (v1 dropped a
       patient's ENTIRE claim history if any single claim had a negative
       payment or blank physician field -- that's a bug, not a policy).
     - Keeps diagnosis / procedure / DRG / length-of-stay / provider
       columns all the way through, because the rule-mining and anomaly
       steps need them (v1 quietly dropped everything but the physician
       NPIs before Apriori, which is why the v1 "fraud rules" were really
       just physician-pair identities).
     - Adds provider-day claim-volume and DRG-peer payment stats needed
       for claim-level anomaly features (see 01_feature_engineering.py).
   ===================================================================== */

USE HAP780;

/* ---------------------------------------------------------------------
   STEP 0: Sanity checks
   --------------------------------------------------------------------- */
SELECT COUNT(*) AS TotalRows FROM [HAP780].[dbo].[project];

SELECT
    SUM(CASE WHEN CLM_PMT_AMT < 0 THEN 1 ELSE 0 END)              AS Negative_CLM_PMT_AMT,
    SUM(CASE WHEN NCH_PRMRY_PYR_CLM_PD_AMT < 0 THEN 1 ELSE 0 END) AS Negative_NCH_PRMRY_PYR_CLM_PD_AMT
FROM [HAP780].[dbo].[project];

/* ---------------------------------------------------------------------
   STEP 1: Keep the columns needed downstream (adds LOS/DRG/provider
   volume context that v1 dropped before rule mining)
   --------------------------------------------------------------------- */
SELECT
      [DESYNPUF_ID]
    , [CLM_ID]
    , [SEGMENT]
    , [CLM_FROM_DT]
    , [CLM_THRU_DT]
    , [PRVDR_NUM]
    , [CLM_PMT_AMT]
    , [NCH_PRMRY_PYR_CLM_PD_AMT]
    , [AT_PHYSN_NPI]
    , [OP_PHYSN_NPI]
    , [OT_PHYSN_NPI]
    , [CLM_ADMSN_DT]
    , [ADMTNG_ICD9_DGNS_CD]
    , [CLM_PASS_THRU_PER_DIEM_AMT]
    , [NCH_BENE_IP_DDCTBL_AMT]
    , [NCH_BENE_PTA_COINSRNC_LBLTY_AM]
    , [NCH_BENE_BLOOD_DDCTBL_LBLTY_AM]
    , [CLM_UTLZTN_DAY_CNT]
    , [NCH_BENE_DSCHRG_DT]
    , [CLM_DRG_CD]
    , [ICD9_DGNS_CD_1]
    , [ICD9_PRCDR_CD_1]
INTO [HAP780].[dbo].[project_clean]
FROM [HAP780].[dbo].[project];

/* ---------------------------------------------------------------------
   STEP 2 (FIXED): Drop only the individual claims with negative payment,
   not every claim belonging to that patient.
   --------------------------------------------------------------------- */
SELECT *
INTO [dbo].[project_v2]
FROM [dbo].[project_clean]
WHERE CLM_ID NOT IN (
    SELECT CLM_ID FROM [dbo].[project_clean]
    WHERE CLM_PMT_AMT < 0 OR NCH_PRMRY_PYR_CLM_PD_AMT < 0
);

/* ---------------------------------------------------------------------
   STEP 3 (FIXED): Drop only claims where all three physician fields are
   blank -- again filtered on CLM_ID, not DESYNPUF_ID.
   --------------------------------------------------------------------- */
DELETE FROM [dbo].[project_v2]
WHERE (AT_PHYSN_NPI IS NULL OR AT_PHYSN_NPI = '')
  AND (OP_PHYSN_NPI IS NULL OR OP_PHYSN_NPI = '')
  AND (OT_PHYSN_NPI IS NULL OR OT_PHYSN_NPI = '');

/* ---------------------------------------------------------------------
   STEP 4: Standardize blanks to 'Unknown' for nominal fields only
   (leave numeric fields NULL-able so Python can impute/flag properly
   instead of the string 'Unknown' silently entering numeric features)
   --------------------------------------------------------------------- */
UPDATE [dbo].[project_v2]
SET AT_PHYSN_NPI          = ISNULL(NULLIF(AT_PHYSN_NPI, ''), 'Unknown'),
    OP_PHYSN_NPI          = ISNULL(NULLIF(OP_PHYSN_NPI, ''), 'Unknown'),
    OT_PHYSN_NPI          = ISNULL(NULLIF(OT_PHYSN_NPI, ''), 'Unknown'),
    ADMTNG_ICD9_DGNS_CD   = ISNULL(NULLIF(ADMTNG_ICD9_DGNS_CD, ''), 'Unknown'),
    CLM_DRG_CD            = ISNULL(NULLIF(CLM_DRG_CD, ''), 'Unknown'),
    ICD9_PRCDR_CD_1       = ISNULL(NULLIF(ICD9_PRCDR_CD_1, ''), 'Unknown');

/* ---------------------------------------------------------------------
   STEP 5 (NEW): Export -- this is what feeds 01_feature_engineering.py.
   Includes claim date, payment, LOS, DRG, diagnosis, procedure, provider,
   and all three physician fields, so feature engineering happens in
   Python where it's testable, not silently inside the rule-mining step.
   --------------------------------------------------------------------- */
SELECT
      DESYNPUF_ID
    , CLM_ID
    , PRVDR_NUM
    , CLM_FROM_DT
    , CLM_THRU_DT
    , CLM_PMT_AMT
    , NCH_PRMRY_PYR_CLM_PD_AMT
    , AT_PHYSN_NPI
    , OP_PHYSN_NPI
    , OT_PHYSN_NPI
    , CLM_UTLZTN_DAY_CNT
    , CLM_DRG_CD
    , ADMTNG_ICD9_DGNS_CD
    , ICD9_DGNS_CD_1
    , ICD9_PRCDR_CD_1
FROM [dbo].[project_v2];
-- Export this result set to CSV as data/claims_export.csv

/* =====================================================================
   EDA QUERIES (unchanged logic, kept for the charts)
   ===================================================================== */
SELECT
    CASE
        WHEN CLM_PMT_AMT < 5000 THEN 'Under 5000'
        WHEN CLM_PMT_AMT BETWEEN 5001 AND 10000 THEN '5001-10000'
        WHEN CLM_PMT_AMT BETWEEN 10001 AND 15000 THEN '10001-15000'
        ELSE 'Over 15000'
    END AS Payment_Range,
    COUNT(*) AS Count
FROM [HAP780].[dbo].[project]
GROUP BY
    CASE
        WHEN CLM_PMT_AMT < 5000 THEN 'Under 5000'
        WHEN CLM_PMT_AMT BETWEEN 5001 AND 10000 THEN '5001-10000'
        WHEN CLM_PMT_AMT BETWEEN 10001 AND 15000 THEN '10001-15000'
        ELSE 'Over 15000'
    END;

SELECT COUNT(DISTINCT DESYNPUF_ID) AS patient_frequency, ICD9_DGNS_CD_1
INTO dbo.diagnosisFREQUENCY
FROM dbo.project
GROUP BY ICD9_DGNS_CD_1
ORDER BY patient_frequency DESC;

SELECT PRVDR_NUM, COUNT(DISTINCT CLM_ID) AS claim_count
INTO dbo.claimcountProvider
FROM dbo.project
GROUP BY PRVDR_NUM
ORDER BY claim_count DESC;
