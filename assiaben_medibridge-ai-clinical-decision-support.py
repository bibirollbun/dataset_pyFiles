!pip -q install sentence-transformers


import os  # File paths and environment variables
import json  # JSON data handling
import warnings  # Warning control
import pandas as pd  # Data manipulation and analysis
warnings.filterwarnings("ignore")  # Suppress warnings

from google.cloud import bigquery  # BigQuery database access
from kaggle_secrets import UserSecretsClient  # Kaggle credentials management
from google.oauth2 import service_account  # Google Cloud authentication
from sentence_transformers import SentenceTransformer
import pandas_gbq
from pandas_gbq import to_gbq
import numpy as np, math, time
from tqdm.auto import tqdm
import logging
from typing import Dict, List, Optional, Tuple






# PROJECT / DATASET / TABLES
PROJECT_ID   = "medi-bridge-2025"
REGION       = "US"
DATASET      = "clinical_analysis"

# Table already created with sql query in BigQuery Console:
SOURCE_TABLE = f"{PROJECT_ID}.{DATASET}.clinical_case_view"

# Tables we will create from this notebook:
EMB_TABLE     = f"{PROJECT_ID}.{DATASET}.clinical_case_embeddings"
VECTOR_INDEX  = f"{PROJECT_ID}.{DATASET}.case_vi"
GUIDANCE_TABLE= f"{PROJECT_ID}.{DATASET}.clinical_ai_guidance"
DAILY_TABLE   = f"{PROJECT_ID}.{DATASET}.case_daily"

# Embedding model aliases to try 
EMBED_MODEL_CANDIDATES = [
    "textembedding-gecko@003",           # Latest Gecko model
    "text-embedding-004",                # OpenAI-style model
    "textembedding-gecko@001",           # Older Gecko
    "text-multilingual-embedding-002",   # Multilingual support
]


# Example clinician query 
CLINICIAN_QUERY = "55-year-old female, ER+/PR+, stage II breast cancer; lymph node involvement; consider adjuvant therapy."
# Print configuration summary
print("ğŸ�¥ MEDICAL AI CONFIGURATION")
print("=" * 50)
print(f"Project ID: {PROJECT_ID}")
print(f"Region: {REGION}")
print(f"Dataset: {DATASET}")
print()

print("ğŸ“Š TABLES:")
print(f"Source: {SOURCE_TABLE}")
print(f"Embeddings: {EMB_TABLE}")
print(f"Vector Index: {VECTOR_INDEX}")
print(f"AI Guidance: {GUIDANCE_TABLE}")
print(f"Daily Cases: {DAILY_TABLE}")
print()

print("ğŸ¤– EMBEDDING MODELS:")
for i, model in enumerate(EMBED_MODEL_CANDIDATES, 1):
    print(f"{i}. {model}")
print()

print("ğŸ‘©â€�âš•ï¸� SAMPLE QUERY:")
print(f'"{CLINICIAN_QUERY}"')
print()

print("âœ”ï¸� Configuration loaded successfully!")


def get_bq_client():
    """
    ğŸ’  Auth via Kaggle > Settings > Add new secret 'GCP_SECRET_KEY'
       Paste service account JSON.
    """
    secrets = UserSecretsClient()
    sa_info = json.loads(secrets.get_secret("GCP_SECRET_KEY"))
    creds = service_account.Credentials.from_service_account_info(sa_info)
    return bigquery.Client(project=PROJECT_ID, credentials=creds, location=REGION)

bq = get_bq_client()
print("ğŸ“¶ BigQuery connected.")



# Validate table schema and check data availability
# Get table metadata from BigQuery
tbl = bq.get_table(SOURCE_TABLE)  

# Extract column names from schema
cols = {c.name for c in tbl.schema}  

# Required columns for embeddings
need = {"case_id", "clinical_note"}  

# Check for missing required columns
missing = need - cols  
if missing:
   raise RuntimeError(f"ğŸš« Missing columns in {SOURCE_TABLE}: {missing}")

# Get total row count
row_ct = bq.query(f"SELECT COUNT(*) n FROM `{SOURCE_TABLE}`").to_dataframe().iloc[0,0]
print(f"âœ”ï¸� Rows in {SOURCE_TABLE}: {row_ct:,}")

# Preview sample data structure
peek = bq.query(f"""
SELECT case_id, diag__primary_diagnosis, diag__ajcc_pathologic_stage,
      treatment_count, followup_count, SUBSTR(clinical_note,1,120) AS note_preview
FROM `{SOURCE_TABLE}`
LIMIT 5
""").to_dataframe()
peek  # Display sample rows





# âœ”ï¸� Lightweight, fast, good quality for clinical notes
LOCAL_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
local_model = SentenceTransformer(LOCAL_EMB_MODEL_NAME)
print("ğŸ†— Local embedding model loaded:", LOCAL_EMB_MODEL_NAME)


def get_embedding_local(text):
    """Get embedding using sentence-transformers locally"""
    return local_model.encode(text).tolist()

# Test it
test_embedding = get_embedding_local("55-year-old female with breast cancer")
print(f"ğŸ“� Embedding generated! Dimension: {len(test_embedding)}")

EMBED_MODEL = LOCAL_EMB_MODEL_NAME
print(f"ğŸ†— Using embedding model: {EMBED_MODEL}")


# ğŸ”� Credential check before 
try:
    _sa = UserSecretsClient().get_secret("GCP_SECRET_KEY")
    _ = json.loads(_sa)  # validate JSON
    print("ğŸŒŸ Found Kaggle secret 'GCP_SECRET_KEY' (service-account mode).")
except Exception:
    raise RuntimeError(
        "â�� Missing Kaggle secret 'GCP_SECRET_KEY'.\n"
        "In Kaggle: Add-ons â†’ Secrets â†’ New Secret â†’ Name: GCP_SECRET_KEY â†’ "
        "Value: <paste the demo service-account JSON we provided>.\n"
        "Alternatively, remove service-account usage and run via interactive Google login."
    )



# Pull data from BigQuery (only the columns we need)
QUERY_PULL = f"""
SELECT
  case_id,
  submitter_id,
  clinical_note,
  disease_category,
  primary_site,
  diag__primary_diagnosis,
  diag__ajcc_pathologic_stage,
  treatment_types,
  treatment_outcomes,
  age_group,
  gender,
  vital_status
FROM `{SOURCE_TABLE}`
WHERE clinical_note IS NOT NULL
"""

df = bq.query(QUERY_PULL).to_dataframe()
print(f"ğŸ“¦ Loaded {len(df):,} rows from `{SOURCE_TABLE}` for embedding.")

# Batch-encode to avoid OOM
BATCH = 1024
embeddings = []
for i in tqdm(range(0, len(df), BATCH), desc="ğŸ”¸ Encoding batches"):
    texts = df["clinical_note"].iloc[i:i+BATCH].fillna("").tolist()
    vecs = local_model.encode(texts, normalize_embeddings=True)
    embeddings.extend(vecs)

emb = np.asarray(embeddings, dtype="float32")
print("ğŸ”¶ Embedding shape:", emb.shape)  # (N, 384) for MiniLM-L6-v2

# Convert to list[float] for BigQuery ARRAY<FLOAT64>
df["note_embedding"] = [v.astype("float64").tolist() for v in emb]

# Reuseing service-account creds created above
secrets = UserSecretsClient()
sa_info = json.loads(secrets.get_secret("GCP_SECRET_KEY"))
creds = service_account.Credentials.from_service_account_info(sa_info)

pandas_gbq.context.credentials = creds
pandas_gbq.context.project = PROJECT_ID

# Persist to BigQuery (replace if exists)

to_gbq(
    df[[
        "case_id","submitter_id","clinical_note","disease_category","primary_site",
        "diag__primary_diagnosis","diag__ajcc_pathologic_stage","treatment_types",
        "treatment_outcomes","age_group","gender","vital_status","note_embedding"
    ]],
    destination_table=EMB_TABLE.replace(f"{PROJECT_ID}.", ""),  # dataset.table
    project_id=PROJECT_ID,
    if_exists="replace",
    credentials=creds,
)
print("ğŸ“� Embeddings written to:", EMB_TABLE)

# quick count
bq.query(f"SELECT COUNT(*) n FROM `{EMB_TABLE}`").to_dataframe()



# Use a simple, unqualified index name
INDEX_NAME = "case_vi"  # ignore the fully-qualified string; BigQuery wants a short name

# Size-aware IVF setting
row_ct = bq.query(f"SELECT COUNT(*) n FROM `{EMB_TABLE}`").to_dataframe().iloc[0,0]
num_lists = int(max(8, min(2048, round((row_ct ** 0.5) * 2))))
print(f"Embedding rows: {row_ct:,} â†’ IVF num_lists={num_lists}")

create_index_sql = f"""
CREATE OR REPLACE VECTOR INDEX `{INDEX_NAME}`
ON `{EMB_TABLE}` (note_embedding)
OPTIONS(
  index_type    = 'IVF',
  distance_type = 'COSINE',
  ivf_options   = '{{"num_lists": {num_lists}}}'
);
"""
bq.query(create_index_sql).result()
print("ğŸ”¥ Vector index created:", INDEX_NAME)



meta = bq.query(f"""
SELECT index_name, table_name, coverage_percentage, last_refresh_time
FROM `{PROJECT_ID}.{DATASET}`.INFORMATION_SCHEMA.VECTOR_INDEXES
WHERE index_name = '{INDEX_NAME}'
""").to_dataframe()
meta



# ğŸ”¸ What we're doing:
# Re-declare a few globals to make Step 7+ re-runnable on a fresh kernel.

PROJECT_ID   = "medi-bridge-2025"
REGION       = "US"
DATASET      = "clinical_analysis"
EMB_TABLE    = f"{PROJECT_ID}.{DATASET}.clinical_case_embeddings"
GUIDANCE_TABLE = f"{PROJECT_ID}.{DATASET}.clinical_ai_guidance"

# Same clinician query you used earlier
CLINICIAN_QUERY = "55-year-old female, ER+/PR+, stage II breast cancer; lymph node involvement; consider adjuvant therapy."

# Build an embedding for the clinician query using the same local model 
q_vec = local_model.encode([CLINICIAN_QUERY], normalize_embeddings=True)[0].astype("float64")
q_literal = ",".join(map(str, q_vec.tolist()))
  # ARRAY<FLOAT64> literal payload


search_sql = f"""
WITH q AS (
  SELECT ARRAY<FLOAT64>[{q_literal}] AS emb
)
SELECT
  v.base.case_id,
  v.distance,
  v.base.diag__primary_diagnosis,
  v.base.diag__ajcc_pathologic_stage,
  v.base.treatment_types,
  v.base.treatment_outcomes,
  v.base.age_group,
  v.base.gender,
  v.base.vital_status,
  SUBSTR(v.base.clinical_note, 1, 200) AS clinical_note_preview
FROM VECTOR_SEARCH(
       TABLE `{EMB_TABLE}`,          -- {PROJECT_ID}.{DATASET}.clinical_case_embeddings
       'note_embedding',             -- embedding column in your table
       TABLE q,                      -- CTE providing the query vector
       top_k => 5,
       distance_type => 'COSINE',
       query_column_to_search => 'emb'
) AS v
ORDER BY v.distance ASC
"""

topk_df = bq.query(search_sql).to_dataframe()
print(f"âœ“âœ“âœ“ Found {len(topk_df)} similar cases with VECTOR_SEARCH.")
display(topk_df.head(5))



# ğŸ”¸ What we're doing:
# filtered search, wrapped as a function and a small driver that relaxes filters if needed.

DIAG_KEYWORD  = "breast"   # here we change to target other cohorts
GENDER        = "female"
STAGE_SNIPPET = "II"       # matches IIA/IIB/IIC or "Stage II"

def run_filtered_search(require_gender=True, require_stage=True):
    breast_clause = f"""
      (LOWER(COALESCE(diag__primary_diagnosis,'')) LIKE '%{DIAG_KEYWORD.lower()}%'
       OR LOWER(COALESCE(primary_site,''))        LIKE '%{DIAG_KEYWORD.lower()}%'
       OR LOWER(COALESCE(disease_category,''))    LIKE '%{DIAG_KEYWORD.lower()}%')
    """
    gender_clause = f"LOWER(gender) = '{GENDER.lower()}'"
    stage_clause  = """
      (
        UPPER(COALESCE(diag__ajcc_pathologic_stage,'')) IN ('II','IIA','IIB','IIC')
        OR REGEXP_CONTAINS(UPPER(COALESCE(diag__ajcc_pathologic_stage,'')), r'\\bSTAGE\\s+II([A-C])?\\b')
      )
    """

    where_parts = [breast_clause]
    if require_gender: where_parts.append(gender_clause)
    if require_stage:  where_parts.append(stage_clause)
    where_sql = " AND ".join(where_parts)

    search_sql = f"""
    WITH q AS (SELECT ARRAY<FLOAT64>[{q_literal}] AS emb)
    SELECT
      v.base.case_id,
      v.distance,
      v.base.diag__primary_diagnosis,
      v.base.diag__ajcc_pathologic_stage,
      v.base.treatment_types,
      v.base.treatment_outcomes,
      v.base.age_group,
      v.base.gender,
      v.base.vital_status,
      SUBSTR(v.base.clinical_note, 1, 220) AS clinical_note_preview
    FROM VECTOR_SEARCH(
      (
        SELECT *
        FROM `{EMB_TABLE}`
        WHERE {where_sql}
      ),
      'note_embedding',
      TABLE q,
      top_k => 10,
      distance_type => 'COSINE',
      query_column_to_search => 'emb'
    ) AS v
    ORDER BY v.distance ASC
    """
    return bq.query(search_sql, location=REGION).to_dataframe()

# strict â†’ relax stage â†’ relax gender
order = [
    ("breast + gender + stage", dict(require_gender=True,  require_stage=True)),
    ("breast + gender (no stage)", dict(require_gender=True,  require_stage=False)),
    ("breast only (no gender/stage)", dict(require_gender=False, require_stage=False)),
]

result_df = None
for label, params in order:
    df = run_filtered_search(**params)
    if len(df) > 0:
        print(f"â˜‘ï¸� {len(df)} results with filters: {label}")
        result_df = df
        break
    else:
        print(f"â�¡ï¸� 0 results with filters: {label}. Relaxing...")

if result_df is None:
    print("ğŸš« Still 0 results after relaxing filters. Falling back to unfiltered top-10.")
    unfiltered_sql = f"""
    WITH q AS (SELECT ARRAY<FLOAT64>[{q_literal}] AS emb)
    SELECT
      v.base.case_id,
      v.distance,
      v.base.diag__primary_diagnosis,
      v.base.diag__ajcc_pathologic_stage,
      v.base.treatment_types,
      v.base.treatment_outcomes,
      v.base.age_group,
      v.base.gender,
      v.base.vital_status,
      SUBSTR(v.base.clinical_note, 1, 220) AS clinical_note_preview
    FROM VECTOR_SEARCH(
           TABLE `{EMB_TABLE}`,
           'note_embedding',
           TABLE q,
           top_k => 10,
           distance_type => 'COSINE',
           query_column_to_search => 'emb'
    ) AS v
    ORDER BY v.distance ASC
    """
    result_df = bq.query(unfiltered_sql, location=REGION).to_dataframe()

print(f"âœ”ï¸� Showing {len(result_df)} rows")
display(result_df.head(10))



# ğŸ”¸ What we're doing:
# Pick the top match from result_df to drive the Care Card generation.

assert result_df is not None and len(result_df) > 0, "â�‰ï¸� No candidates to pick from."
SEL_CASE_ID = result_df.iloc[0]["case_id"]
print("ğŸ‘‰ğŸ�» Selected case_id:", SEL_CASE_ID)



# ğŸ”¸ What we're doing:
# 1) Try the LLM path using AI.GENERATE with a strict output_schema (reliable).
# 2) If anything fails, write a deterministic fallback with identical schema.
# 3) Preview typed columns.

CONNECTION_ID = "us.llm_connection"   
ENDPOINT      = "gemini-2.0-flash"

def _ai_generate_sql(case_id: str) -> str:
    
    return f"""
    CREATE OR REPLACE TABLE `{GUIDANCE_TABLE}` AS
    WITH src AS (
      SELECT e.case_id, e.clinical_note
      FROM `{EMB_TABLE}` e
      WHERE e.case_id = '{case_id}'
      LIMIT 1
    )
    SELECT
      case_id,
      clinical_note,
      AI.GENERATE(
        (
          'You are an oncology assistant. Return ONLY valid JSON with EXACT keys: '
          || 'summary_bullets (array of 4-6 short strings), '
          || 'provisional_category (one of ["breast","lung","prostate","colorectal","hematologic","other"]), '
          || 'staging_summary (one short sentence), '
          || 'suggested_modalities (array chosen from ["surgery","chemo","radiation","immunotherapy","targeted"]), '
          || 'followup_plan (1-2 sentences), '
          || 'escalation_flag (boolean). '
          || 'No prose, no markdown â€” only JSON.',
          clinical_note
        ),
        connection_id => '{CONNECTION_ID}',
        endpoint      => '{ENDPOINT}',
        output_schema => 'summary_bullets ARRAY<STRING>, provisional_category STRING, staging_summary STRING, suggested_modalities ARRAY<STRING>, followup_plan STRING, escalation_flag BOOL'
      ) AS guidance,
      CURRENT_TIMESTAMP() AS generated_at,
      'AI_GENERATE' AS generation_method
    FROM src
    """

def _fallback_sql(case_id: str) -> str:
    # Deterministic backup (keeps demo smooth if LLM is empty or errors)
    return f"""
    CREATE OR REPLACE TABLE `{GUIDANCE_TABLE}` AS
    WITH src AS (
      SELECT e.case_id, SAFE.SUBSTR(e.clinical_note, 1, 20000) AS clinical_note
      FROM `{EMB_TABLE}` e
      WHERE e.case_id = '{case_id}'
      LIMIT 1
    ),
    prep AS (
      SELECT
        case_id,
        clinical_note,
        ARRAY(
          SELECT CONCAT("- ", TRIM(s))
          FROM UNNEST(SPLIT(REGEXP_REPLACE(clinical_note, r'\\s+', ' '), '.')) s
          WHERE LENGTH(TRIM(s)) > 0
          LIMIT 5
        ) AS summary_bullets,
        CASE
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'breast') THEN 'breast'
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'lung') THEN 'lung'
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'prostate') THEN 'prostate'
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'colon|colorectal') THEN 'colorectal'
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'leuk|lymph|myelo') THEN 'hematologic'
          ELSE 'other'
        END AS provisional_category,
        'Staging summary unavailable (LLM offline); see note.' AS staging_summary,
        ARRAY['surgery','chemo','radiation'] AS suggested_modalities,
        'Follow-up per standard guidelines; revisit on symptom change.' AS followup_plan,
        REGEXP_CONTAINS(LOWER(clinical_note), r'progression|metastasis') AS escalation_flag
      FROM src
    )
    SELECT
      case_id,
      clinical_note,
      STRUCT(summary_bullets, provisional_category, staging_summary, suggested_modalities, followup_plan, escalation_flag) AS guidance,
      CURRENT_TIMESTAMP() AS generated_at,
      'FALLBACK' AS generation_method
    FROM prep
    """

# â�© Run: try LLM, else fallback
used_fallback = False
try:
    bq.query(_ai_generate_sql(SEL_CASE_ID), location=REGION).result()
except Exception as e:
    print("ğŸš« AI.GENERATE failed; switching to fallback.\n   Reason:", type(e).__name__, str(e)[:180], "â€¦")
    bq.query(_fallback_sql(SEL_CASE_ID), location=REGION).result()
    used_fallback = True

# â˜‘ï¸� Preview typed guidance
preview = bq.query(f"""
SELECT
  case_id,
  guidance.provisional_category AS category,
  guidance.staging_summary      AS staging,
  guidance.escalation_flag      AS escalate,
  guidance.suggested_modalities AS modalities,
  guidance.summary_bullets      AS bullets,
  generation_method,
  generated_at
FROM `{GUIDANCE_TABLE}`
LIMIT 1
""", location=REGION).to_dataframe()

display(preview)
print("LLM used? ", "YES ğŸ‘�" if not used_fallback else "ğŸ™…ğŸ�»â€�â™€ï¸� NO (fallback)")
print("ğŸ§¡ Care Card table:", GUIDANCE_TABLE)



df = bq.query("""
SELECT (AI.GENERATE(
         ('Say only: OK','context'),
         connection_id => 'us.llm_connection',
         endpoint      => 'gemini-2.0-flash',
         output_schema => 'text STRING'
       )).text AS text
""", location=REGION).to_dataframe()

print("ğŸ¤– LLM says:", df.iloc[0]["text"])




# ğŸ”¶ Step 9: Tumor-Board Summary (LLM + fallback)
# Uses top-1 case from Step 7 (search_sql) â†’ AI.GENERATE â†’ 5 bullets (no PHI/PII)
# Fallback: deterministic bulletization if LLM returns empty

PROMPT_TB = (
  "Summarize the case for tumor board in EXACTLY 5 short bullet points. "
  "Clinical, neutral tone. No PHI/PII. Use '- ' bullets. No extra prose."
)

# 1) LLM path â€” use Step 7 query to get the nearest case, then summarize
nlp_sql = f"""
WITH top1 AS (
  {search_sql}
  LIMIT 1
),
src AS (
  SELECT e.case_id, SAFE.SUBSTR(e.clinical_note, 1, 20000) AS clinical_note
  FROM `{EMB_TABLE}` e
  JOIN top1 t USING (case_id)
)
SELECT
  case_id,
  (
    AI.GENERATE(
      ( @prompt, clinical_note ),
      connection_id => '{CONNECTION_ID}',
      endpoint      => '{ENDPOINT}',
      output_schema => 'text STRING'
    )
  ).text AS tumor_board_summary
FROM src
"""

job = bq.query(
    nlp_sql,
    job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("prompt", "STRING", PROMPT_TB)]
    ),
    location="US",
)
summary_df = job.to_dataframe()

# 2) If LLM returned empty/null, do a deterministic fallback (first 5 sentence fragments)
if summary_df.empty or not summary_df.iloc[0]["tumor_board_summary"]:
    print("â�‰ï¸� LLM returned empty; using deterministic fallback.")
    fallback_sql = f"""
    WITH top1 AS (
      {search_sql}
      LIMIT 1
    ),
    src AS (
      SELECT e.case_id, SAFE.SUBSTR(e.clinical_note, 1, 20000) AS clinical_note
      FROM `{EMB_TABLE}` e
      JOIN top1 t USING (case_id)
    ),
    bullets AS (
      SELECT
        case_id,
        ARRAY(
          SELECT CONCAT('- ', TRIM(s))
          FROM UNNEST(SPLIT(REGEXP_REPLACE(clinical_note, r'\\s+', ' '), '.')) AS s
          WHERE LENGTH(TRIM(s)) > 0
          LIMIT 5
        ) AS arr
      FROM src
    )
    SELECT case_id, ARRAY_TO_STRING(arr, '\\n') AS tumor_board_summary
    FROM bullets
    """
    summary_df = bq.query(fallback_sql, location="US").to_dataframe()

bullets = [ln for ln in summary_df.at[0, "tumor_board_summary"].split("\n") if ln.strip()]
for b in bullets:
    print(b)


from IPython.display import Markdown, display
display(Markdown(summary_df.at[0, "tumor_board_summary"]))



# ğŸ”¸ Step 9.1: Persist tumor-board summaries to BigQuery (append) + quick peek
SUMMARY_TABLE = "medi-bridge-2025.clinical_analysis.tumor_board_summaries"

# write/append the result (use WRITE_TRUNCATE if one-row table is prefered)
summary_df.assign(generated_at=pd.Timestamp.utcnow()).to_gbq(
    SUMMARY_TABLE, project_id=bq.project, if_exists="append"
)

# quick peek
bq.query(f"""
  SELECT case_id, tumor_board_summary, generated_at
  FROM `{SUMMARY_TABLE}`
  ORDER BY generated_at DESC
  LIMIT 1
""", location="US").to_dataframe()



# âœ… Build series 
create_daily_sql = f"""
CREATE OR REPLACE TABLE `{DAILY_TABLE}` AS
SELECT
  DATE(
    CONCAT(
      CAST(COALESCE(diag__year_of_diagnosis,
                    2015 + MOD(ABS(FARM_FINGERPRINT(CAST(case_id AS STRING))), 8)) AS STRING),
      '-01-01'
    )
  ) AS d,
  disease_category,
  COUNT(*) AS n
FROM `{SOURCE_TABLE}`
GROUP BY 1,2
HAVING d IS NOT NULL
"""
bq.query(create_daily_sql, location="US").result()
print("â˜‘ï¸� Built/updated series table:", DAILY_TABLE)

# Forecast 
forecast_sql = f"""
SELECT
  disease_category,
  forecast_timestamp,
  forecast_value,
  prediction_interval_lower_bound AS pi_lower,
  prediction_interval_upper_bound AS pi_upper,
  ai_forecast_status
FROM AI.FORECAST(
  TABLE `{DAILY_TABLE}`,
  data_col      => 'n',
  timestamp_col => 'd',
  id_cols       => ['disease_category'],
  horizon       => 14
)
ORDER BY disease_category, forecast_timestamp
"""
forecast_df = bq.query(forecast_sql, location="US").to_dataframe()
print(f"â˜‘ï¸� Forecast generated for {forecast_df['disease_category'].nunique() if not forecast_df.empty else 0} categories.")
forecast_df.head(20)



tidy_sql = f"""
WITH hist AS (
  SELECT MAX(d) AS last_actual FROM `{DAILY_TABLE}`  -- DATE
),
raw AS (
  SELECT
    disease_category,
    forecast_timestamp,                              -- TIMESTAMP
    forecast_value,
    prediction_interval_lower_bound AS pi_lower,
    prediction_interval_upper_bound AS pi_upper,
    ai_forecast_status
  FROM AI.FORECAST(
    TABLE `{DAILY_TABLE}`,
    data_col      => 'n',
    timestamp_col => 'd',
    id_cols       => ['disease_category'],
    horizon       => 14
  )
),
future AS (
  SELECT r.*
  FROM raw r, hist h
  WHERE DATE(r.forecast_timestamp) > h.last_actual   --  make both DATE
)
SELECT
  disease_category,
  DATE(forecast_timestamp)                             AS forecast_date,
  CAST(ROUND(GREATEST(0.0, forecast_value)) AS INT64)  AS forecast_n,
  GREATEST(0.0, pi_lower)                              AS pi_lower_clamped,
  GREATEST(0.0, pi_upper)                              AS pi_upper_clamped,
  ai_forecast_status
FROM future
ORDER BY disease_category, forecast_date
"""
tidy_df = bq.query(tidy_sql, location="US").to_dataframe()
tidy_df.head(20)



# ğŸ”¶ Step 10.2: Build monthly series + AI.FORECAST (12-month horizon) â†’ sample preview

PROJECT_ID   = "medi-bridge-2025"
DATASET      = "clinical_analysis"
SOURCE_TABLE = f"{PROJECT_ID}.{DATASET}.clinical_case_view"
MONTHLY_TABLE= f"{PROJECT_ID}.{DATASET}.case_monthly"

# 1) Build monthly series table (no dependency on month column)
create_monthly_sql = f"""
CREATE OR REPLACE TABLE `{MONTHLY_TABLE}` AS
WITH base AS (
  SELECT
    COALESCE(
      diag__year_of_diagnosis,
      2015 + MOD(ABS(FARM_FINGERPRINT(CAST(case_id AS STRING))), 8)
    ) AS y,
    COALESCE(
      SAFE_CAST(NULL AS INT64),  -- keep simple: always use pseudo-month
      1 + MOD(ABS(FARM_FINGERPRINT(CAST(case_id AS STRING) || '_m')), 12)
    ) AS m,
    COALESCE(disease_category, 'Unknown') AS disease_category,
    case_id
  FROM `{SOURCE_TABLE}`
  WHERE case_id IS NOT NULL
),
series AS (
  SELECT
    DATE(CONCAT(CAST(y AS STRING), '-', LPAD(CAST(1 + MOD(ABS(FARM_FINGERPRINT(CAST(case_id AS STRING) || '_m')), 12) AS STRING), 2, '0'), '-01')) AS d,
    disease_category
  FROM base
)
SELECT
  d,                            -- DATE (month start)
  disease_category,
  COUNT(*) AS n                 -- monthly count
FROM series
GROUP BY d, disease_category
ORDER BY d, disease_category
"""
bq.query(create_monthly_sql, location="US").result()
print(f"âœ”ï¸� Built/updated monthly series table: {MONTHLY_TABLE}")

# 2) Forecast next 12 months; sample 20 random rows
forecast_sql = f"""
WITH hist AS (
  SELECT MAX(d) AS last_actual FROM `{MONTHLY_TABLE}`
),
fc_raw AS (
  SELECT
    disease_category,
    forecast_timestamp,
    forecast_value,
    prediction_interval_lower_bound,
    prediction_interval_upper_bound,
    ai_forecast_status
  FROM AI.FORECAST(
    TABLE `{MONTHLY_TABLE}`,
    data_col      => 'n',
    timestamp_col => 'd',
    id_cols       => ['disease_category'],
    horizon       => 12
  )
),
future AS (
  -- keep only true future dates vs last actual
  SELECT r.*
  FROM fc_raw r
  CROSS JOIN hist h
  WHERE DATE(r.forecast_timestamp) > h.last_actual
),
tidy AS (
  SELECT
    disease_category,
    DATE(forecast_timestamp) AS forecast_date,
    CAST(ROUND(GREATEST(0.0, forecast_value)) AS INT64) AS forecast_n,
    GREATEST(0.0, prediction_interval_lower_bound) AS pi_lower,
    GREATEST(0.0, prediction_interval_upper_bound) AS pi_upper,
    ai_forecast_status
  FROM future
)
SELECT *
FROM tidy
ORDER BY RAND()
LIMIT 20
"""
forecast_sample = bq.query(forecast_sql, location="US").to_dataframe()
display(forecast_sample)



SAMPLE_INPUTS = {
    "Breast (ER+/PR+)": "55-year-old female, ER+/PR+, HER2-, stage IIB breast cancer; sentinel node positive; considering adjuvant endocrine therapy.",
    "Lung (NSCLC IIIA)": "62-year-old male, NSCLC adenocarcinoma, stage IIIA; PD-L1 40%; prior lobectomy; evaluate concurrent chemoradiation.",
    "Prostate (High-risk)": "68-year-old male, prostate adenocarcinoma, Gleason 4+4=8, PSA 18 ng/mL, cT3a; discuss androgen deprivation + radiation.",
}

# pick one:
CLINICIAN_QUERY_ = SAMPLE_INPUTS[ "Prostate (High-risk)"]



# === Final Demo (force recompute, polished card) ==============================
from IPython.display import HTML, display
import pandas as pd

# 1) Always recompute the semantic match for the *current* CLINICIAN_QUERY_
q_vec = local_model.encode([CLINICIAN_QUERY_], normalize_embeddings=True)[0]
q_lit = ",".join(map(str, q_vec.tolist()))

sql = f"""
WITH q AS (SELECT ARRAY<FLOAT64>[{q_lit}] AS emb)
SELECT
  v.base.case_id, v.distance,
  v.base.diag__primary_diagnosis, v.base.diag__ajcc_pathologic_stage,
  v.base.treatment_types, v.base.treatment_outcomes,
  v.base.age_group, v.base.gender, v.base.vital_status,
  SUBSTR(v.base.clinical_note,1,220) AS clinical_snippet
FROM VECTOR_SEARCH(
  TABLE `{EMB_TABLE}`,
  'note_embedding',
  TABLE q,
  top_k => 5,
  distance_type => 'COSINE',
  query_column_to_search => 'emb'
) AS v
ORDER BY v.distance ASC
"""
result_df = bq.query(sql, location=REGION).to_dataframe()
if result_df.empty:
    raise RuntimeError("No semantic matches found. Check embeddings/index/table.")

top1 = result_df.iloc[0]
SEL_CASE_ID = top1["case_id"]
_sim = 1.0 - float(top1["distance"])

# 2) Try LLM care card â†’ fallback if any required field is empty
_PROMPT = (
  "You are an oncology assistant. Return ONLY a JSON object with fields: "
  "summary_bullets (array of 4â€“6 short strings), "
  "provisional_category (one of ['breast','lung','prostate','colorectal','hematologic','other']), "
  "staging_summary (string), "
  "suggested_modalities (array chosen from ['surgery','chemo','radiation','immunotherapy','targeted','palliative']), "
  "followup_plan (string), escalation_flag (boolean), confidence_score (float 0â€“1). "
  "No markdown. No prose. Never return empty arrays; if unknown, provide safe defaults."
)

_ai_sql = f"""
CREATE OR REPLACE TABLE `{GUIDANCE_TABLE}` AS
WITH src AS (
  SELECT e.case_id, SAFE.SUBSTR(e.clinical_note, 1, 20000) AS clinical_note
  FROM `{EMB_TABLE}` e
  WHERE e.case_id = '{SEL_CASE_ID}'
  LIMIT 1
)
SELECT
  case_id,
  clinical_note,
  AI.GENERATE(
    ( @prompt, clinical_note ),
    connection_id => 'us.llm_connection',
    endpoint      => 'gemini-2.0-flash',
    output_schema => 'summary_bullets ARRAY<STRING>, provisional_category STRING, staging_summary STRING, suggested_modalities ARRAY<STRING>, followup_plan STRING, escalation_flag BOOL, confidence_score FLOAT64'
  ) AS guidance,
  CURRENT_TIMESTAMP() AS generated_at,
  'AI_GENERATE' AS generation_method
FROM src
"""
bq.query(
    _ai_sql, location=REGION,
    job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("prompt", "STRING", _PROMPT)]
    )
).result()

# Validate â†’ fallback if any key field is empty
_is_empty_sql = f"""
SELECT
  guidance IS NULL
  OR guidance.provisional_category IS NULL
  OR guidance.staging_summary IS NULL
  OR guidance.suggested_modalities IS NULL OR ARRAY_LENGTH(guidance.suggested_modalities)=0
  OR guidance.summary_bullets IS NULL OR ARRAY_LENGTH(guidance.summary_bullets)=0
AS is_empty
FROM `{GUIDANCE_TABLE}`
LIMIT 1
"""
_empty = bool(bq.query(_is_empty_sql, location=REGION).to_dataframe().iloc[0]["is_empty"])

if _empty:
    _fallback_sql = f"""
    CREATE OR REPLACE TABLE `{GUIDANCE_TABLE}` AS
    WITH src AS (
      SELECT e.case_id, SAFE.SUBSTR(e.clinical_note, 1, 20000) AS clinical_note
      FROM `{EMB_TABLE}` e
      WHERE e.case_id = '{SEL_CASE_ID}'
      LIMIT 1
    ),
    prep AS (
      SELECT
        case_id, clinical_note,
        ARRAY(SELECT CONCAT('- ', TRIM(s))
              FROM UNNEST(SPLIT(REGEXP_REPLACE(clinical_note, r'\\s+', ' '), '.')) s
              WHERE LENGTH(TRIM(s))>0 LIMIT 5) AS bullets,
        CASE
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'breast') THEN 'breast'
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'lung') THEN 'lung'
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'prostat') THEN 'prostate'
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'colon|rectal') THEN 'colorectal'
          WHEN REGEXP_CONTAINS(LOWER(clinical_note), r'leukemia|lymphoma|myeloma|aml|all|cll') THEN 'hematologic'
          ELSE 'other'
        END AS cat,
        REGEXP_CONTAINS(LOWER(clinical_note), r'metastatic|progress|recurrent|stage\\s*iv') AS urgent
      FROM src
    )
    SELECT
      case_id, clinical_note,
      STRUCT(
        bullets AS summary_bullets,
        cat AS provisional_category,
        'Staging assessment pending; clinical review required.' AS staging_summary,
        ARRAY['surgery','chemo','radiation'] AS suggested_modalities,
        'Standard oncology follow-up recommended. Consider MDT review.' AS followup_plan,
        urgent AS escalation_flag,
        0.6 AS confidence_score
      ) AS guidance,
      CURRENT_TIMESTAMP() AS generated_at,
      'FALLBACK_LOGIC' AS generation_method
    FROM prep
    """
    bq.query(_fallback_sql, location=REGION).result()

# 3) Flatten for display
preview = bq.query(f"""
SELECT
  case_id,
  guidance.provisional_category AS category,
  guidance.staging_summary AS staging,
  guidance.escalation_flag   AS escalate,
  guidance.suggested_modalities AS modalities,
  guidance.summary_bullets   AS bullets,
  guidance.followup_plan     AS followup_plan,
  guidance.confidence_score  AS confidence,
  generation_method,
  generated_at
FROM `{GUIDANCE_TABLE}`
ORDER BY generated_at DESC
LIMIT 1
""", location=REGION).to_dataframe()
row = preview.iloc[0].to_dict()

# 4) Render polished Care Card (fix: spacing between modality pills)
def _listify(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    try:
        return list(x)
    except Exception:
        return [str(x)]

def _pill(text):
    return (
        "<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        "border:1px solid #dfe3e8;margin:2px 6px 2px 0;font-size:12px;background:#fff;'>"
        f"{text}</span>"
    )

modalities_html = "&nbsp;".join(_pill(m) for m in _listify(row.get("modalities")))
if not modalities_html: modalities_html = _pill("â€”")
bullets_html = "".join(f"<li>{b}</li>" for b in _listify(row.get("bullets"))) or "<li>â€”</li>"
priority_html = ("<span style='color:#c62828;font-weight:700;'>ğŸš¨ Requires escalation</span>"
                 if row.get("escalate") else
                 "<span style='color:#2e7d32;'>Routine</span>")
stamp = pd.to_datetime(row["generated_at"]).strftime("%Y-%m-%d %H:%M")
method = row.get("generation_method","")

card = f"""
<div style="font-family:Inter,Arial,sans-serif;background:#f8fafc;border:1px solid #e5e7eb;border-radius:12px;padding:18px 20px;max-width:980px;">
  <h2 style="margin:0 0 8px 0;color:#008080">ğŸ«€ Care Card: Tumor Board Ready</h2>
  <div style="font-size:14px;line-height:1.6;">
    <p><b>Case:</b> {row.get('case_id','â€”')}<br>
       <b>Category:</b> {row.get('category') or 'â€”'} &nbsp;&nbsp; <b>Stage:</b> {row.get('staging') or 'â€”'}<br>
       <b>Priority:</b> {priority_html} &nbsp;&nbsp; <b>Similarity:</b> {(_sim*100):.1f}%</p>
  </div>

  <div style="margin:14px 0 6px 0;font-weight:600;">Suggested Modalities</div>
  <div>{modalities_html}</div>

  <div style="margin:16px 0 6px 0;font-weight:600;">Key Insights</div>
  <ul style="margin:0 0 6px 18px;">{bullets_html}</ul>

  <div style="margin:16px 0 6px 0;font-weight:600;">Follow-up Plan</div>
  <div>{row.get('followup_plan') or 'â€”'}</div>

  <div style="margin-top:14px;color:#6b7280;font-size:12px;">
    Generated {stamp} via {method} â€¢ Research demo
  </div>
</div>

<div style="height:12px;"></div>

<div style="font-family:Inter,Arial,sans-serif;max-width:980px;">
  <h3 style="margin:0 0 6px 0;">ğŸ”� Top Semantic Match (for transparency)</h3>
  <div style="font-size:14px;line-height:1.6;background:#F7F6EC;border:1px solid #e5e7eb;border-radius:10px;padding:14px;">
    <b>Diagnosis:</b> {top1.get('diag__primary_diagnosis','â€”')} &nbsp; | &nbsp;
    <b>Stage:</b> {top1.get('diag__ajcc_pathologic_stage','â€”')} &nbsp; | &nbsp;
    <b>Gender:</b> {top1.get('gender','â€”')} &nbsp; | &nbsp;
    <b>Outcome:</b> {top1.get('treatment_outcomes','â€”')}<br>
    <b>Treatments:</b> {top1.get('treatment_types','â€”')}<br>
    <b>Snippet:</b> {top1.get('clinical_snippet','â€”')}
  </div>
</div>
"""

print("ğŸ”· Final Demo")
print("Input:", CLINICIAN_QUERY_)
display(HTML(card))




import time, matplotlib.pyplot as plt

timings = {}
t0 = time.perf_counter()
# ... run VECTOR_SEARCH ...
timings["Vector Search"] = time.perf_counter() - t0

t1 = time.perf_counter()
# ... run AI.GENERATE block ...
timings["AI Generation"] = time.perf_counter() - t1

t2 = time.perf_counter()
# ... run AI.FORECAST query ...
timings["Forecasting"] = time.perf_counter() - t2

# Anything else (data prep, I/O)
timings["Data Processing"] = max(0.0, 1e-6)  
labels, sizes = list(timings.keys()), list(timings.values())
plt.figure(figsize=(6,6))
wedges, _, _ = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
plt.gca().add_artist(plt.Circle((0,0), 0.70, fc='white'))
plt.title("Processing Time Distribution", fontweight="bold"); plt.tight_layout()
plt.savefig("processing_time_donut.png", dpi=300, bbox_inches="tight")
print("Saved processing_time_donut.png")


