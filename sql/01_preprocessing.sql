/* =====================================================================
   HEALTHCARE INSURANCE FRAUD DETECTION - DATA PREPROCESSING (v2, final)
   Source: CMS 2008-2010 DE-SynPUF Inpatient Claims (Sample 2)

   ===================================================================== */

USE HAP780;

DROP TABLE IF EXISTS dbo.project_clean;
DROP TABLE IF EXISTS dbo.project_v2;
DROP TABLE IF EXISTS dbo.diagnosisFREQUENCY;
DROP TABLE IF EXISTS dbo.claimcountProvider;

/* ---------------------------------------------------------------------
   STEP 0: Sanity checks
   --------------------------------------------------------------------- */
SELECT COUNT(*) AS TotalRows FROM [HAP780].[dbo].[project];

SELECT
    SUM(CASE WHEN TRY_CAST(CLM_PMT_AMT AS FLOAT) < 0 THEN 1 ELSE 0 END)              AS Negative_CLM_PMT_AMT,
    SUM(CASE WHEN TRY_CAST(NCH_PRMRY_PYR_CLM_PD_AMT AS FLOAT) < 0 THEN 1 ELSE 0 END) AS Negative_NCH_PRMRY_PYR_CLM_PD_AMT
FROM [HAP780].[dbo].[project];

/* ---------------------------------------------------------------------
   STEP 0b: Data-quality check
   --------------------------------------------------------------------- */
SELECT
    SUM(CASE WHEN CLM_PMT_AMT IS NOT NULL AND CLM_PMT_AMT <> ''
              AND TRY_CAST(CLM_PMT_AMT AS FLOAT) IS NULL THEN 1 ELSE 0 END) AS Bad_CLM_PMT_AMT,
    SUM(CASE WHEN NCH_PRMRY_PYR_CLM_PD_AMT IS NOT NULL AND NCH_PRMRY_PYR_CLM_PD_AMT <> ''
              AND TRY_CAST(NCH_PRMRY_PYR_CLM_PD_AMT AS FLOAT) IS NULL THEN 1 ELSE 0 END) AS Bad_NCH_PRMRY_PYR_CLM_PD_AMT,
    SUM(CASE WHEN CLM_UTLZTN_DAY_CNT IS NOT NULL AND CLM_UTLZTN_DAY_CNT <> ''
              AND TRY_CAST(CLM_UTLZTN_DAY_CNT AS FLOAT) IS NULL THEN 1 ELSE 0 END) AS Bad_CLM_UTLZTN_DAY_CNT
FROM [HAP780].[dbo].[project];

/* ---------------------------------------------------------------------
   STEP 1: Keep the columns needed
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
   STEP 2: Drop only the individual claims with negative payment
   --------------------------------------------------------------------- */
SELECT *
INTO [dbo].[project_v2]
FROM [dbo].[project_clean]
WHERE CLM_ID NOT IN (
    SELECT CLM_ID FROM [dbo].[project_clean]
    WHERE TRY_CAST(CLM_PMT_AMT AS FLOAT) < 0
       OR TRY_CAST(NCH_PRMRY_PYR_CLM_PD_AMT AS FLOAT) < 0
);

/* ---------------------------------------------------------------------
   STEP 3: Drop only claims where all three physician fields are blank
   --------------------------------------------------------------------- */
DELETE FROM [dbo].[project_v2]
WHERE (AT_PHYSN_NPI IS NULL OR AT_PHYSN_NPI = '')
  AND (OP_PHYSN_NPI IS NULL OR OP_PHYSN_NPI = '')
  AND (OT_PHYSN_NPI IS NULL OR OT_PHYSN_NPI = '');

/* ---------------------------------------------------------------------
   STEP 4: Standardize blanks to 'Unknown' for nominal fields only
   --------------------------------------------------------------------- */
UPDATE [dbo].[project_v2]
SET AT_PHYSN_NPI          = ISNULL(NULLIF(AT_PHYSN_NPI, ''), 'Unknown'),
    OP_PHYSN_NPI          = ISNULL(NULLIF(OP_PHYSN_NPI, ''), 'Unknown'),
    OT_PHYSN_NPI          = ISNULL(NULLIF(OT_PHYSN_NPI, ''), 'Unknown'),
    ADMTNG_ICD9_DGNS_CD   = ISNULL(NULLIF(ADMTNG_ICD9_DGNS_CD, ''), 'Unknown'),
    CLM_DRG_CD            = ISNULL(NULLIF(CLM_DRG_CD, ''), 'Unknown'),
    ICD9_PRCDR_CD_1       = ISNULL(NULLIF(ICD9_PRCDR_CD_1, ''), 'Unknown');

/* ---------------------------------------------------------------------
   STEP 5: Export 
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
   EDA QUERIES 
   ===================================================================== */

SELECT
    CASE
        WHEN TRY_CAST(CLM_PMT_AMT AS FLOAT) < 5000 THEN 'Under 5000'
        WHEN TRY_CAST(CLM_PMT_AMT AS FLOAT) BETWEEN 5000 AND 10000 THEN '5000-10000'
        WHEN TRY_CAST(CLM_PMT_AMT AS FLOAT) BETWEEN 10001 AND 15000 THEN '10001-15000'
        ELSE 'Over 15000'
    END AS Payment_Range,
    COUNT(*) AS Count
FROM [HAP780].[dbo].[project_v2]
GROUP BY
    CASE
        WHEN TRY_CAST(CLM_PMT_AMT AS FLOAT) < 5000 THEN 'Under 5000'
        WHEN TRY_CAST(CLM_PMT_AMT AS FLOAT) BETWEEN 5000 AND 10000 THEN '5000-10000'
        WHEN TRY_CAST(CLM_PMT_AMT AS FLOAT) BETWEEN 10001 AND 15000 THEN '10001-15000'
        ELSE 'Over 15000'
    END;

SELECT COUNT(DISTINCT DESYNPUF_ID) AS patient_frequency, ICD9_DGNS_CD_1
INTO dbo.diagnosisFREQUENCY
FROM dbo.project_v2
GROUP BY ICD9_DGNS_CD_1
ORDER BY patient_frequency DESC;

SELECT PRVDR_NUM, COUNT(DISTINCT CLM_ID) AS claim_count
INTO dbo.claimcountProvider
FROM dbo.project_v2
GROUP BY PRVDR_NUM
ORDER BY claim_count DESC;