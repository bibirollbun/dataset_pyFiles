# pip install --q sentence-transformers 


# Standard library imports
import os
import json
import time
import hashlib
import math
import logging
import warnings
import textwrap
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from more_itertools import chunked

# Google Cloud imports
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas_gbq
from pandas_gbq import to_gbq

# Kaggle specific imports
from kaggle_secrets import UserSecretsClient

# Machine learning imports
from sentence_transformers import SentenceTransformer

# Display imports
from IPython.display import HTML, Markdown, display

# Configure warnings
warnings.filterwarnings("ignore")


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

print("ğŸ‘©â€�âš•ï¸� SAMPLE QUERY:")
print(f'"{CLINICIAN_QUERY}"')
print()

print("âœ”ï¸� Configuration loaded successfully!")


SECRET_NAME = "GCP_SECRET_KEY"

def get_bq_client(project_id: str, region: str, secret_name: str = SECRET_NAME) -> bigquery.Client:
    secrets = UserSecretsClient()
    try:
        raw = secrets.get_secret(secret_name)
        sa_info = json.loads(raw)  # validate JSON
    except Exception as e:
        raise RuntimeError(
            f"â�� Missing or invalid Kaggle secret '{secret_name}'.\n"
            "Kaggle â†’ Settings â†’ Secrets â†’ Add new â†’ Name: GCP_SECRET_KEY â†’ "
            "Value: <paste your service-account JSON>"
        ) from e

    creds = service_account.Credentials.from_service_account_info(sa_info)
    client = bigquery.Client(project=project_id, credentials=creds, location=region)

    # Configure pandas_gbq defaults (optional but handy later)
    pandas_gbq.context.credentials = creds
    pandas_gbq.context.project = project_id

    # Quick connectivity check
    client.query("SELECT 1").result()
    return client

bq = get_bq_client(PROJECT_ID, REGION)
print(f"ğŸ“¶ BigQuery connected: project={bq.project} ğŸ”¸ region={REGION} ğŸ”¸ OK ğŸ‘ŒğŸ�»")


# --- 1) Schema checks 
REQUIRED = {"case_id", "clinical_note"}
OPTIONAL = {
    "diag__primary_diagnosis", "diag__ajcc_pathologic_stage", "primary_site",
    "treatment_types", "treatment_outcomes", "gender", "race", "vital_status",
    "age_at_diagnosis_years"
}

tbl = bq.get_table(SOURCE_TABLE)
cols = {c.name for c in tbl.schema}

missing_req = REQUIRED - cols
if missing_req:
    raise RuntimeError(f"Missing required columns in `{SOURCE_TABLE}`: {sorted(missing_req)}â�‰ï¸�")

missing_opt = OPTIONAL - cols
if missing_opt:
    print(f"ğŸ’¥ Optional columns not found (will be handled gracefully): {sorted(missing_opt)}")

print(f"Table found: `{SOURCE_TABLE}`âœ”ï¸� ")
print(f"   Columns: {len(cols)} total\n")

# --- 2) Quick data health stats 
stats_sql = f"""
SELECT
  COUNT(*) AS total_rows,
  APPROX_COUNT_DISTINCT(case_id) AS unique_case_ids,
  COUNTIF(clinical_note IS NULL) AS null_notes,
  ROUND(AVG(LENGTH(COALESCE(clinical_note, ''))), 0) AS avg_note_len_chars,
  COUNTIF(diag__primary_diagnosis IS NULL) AS null_primary_dx,
  COUNTIF(diag__ajcc_pathologic_stage IS NULL) AS null_stage
FROM `{SOURCE_TABLE}`
"""
stats = bq.query(stats_sql).to_dataframe().iloc[0]

print("ğŸ“Š QUICK STATS")
print("==============")
print(f"Total rows                : {int(stats.total_rows):,}")
print(f"Unique case_ids           : {int(stats.unique_case_ids):,}")
print(f"clinical_note = NULL      : {int(stats.null_notes):,}")
print(f"Avg clinical_note length  : {int(stats.avg_note_len_chars):,} chars")
print(f"diag__primary_diagnosis NULL : {int(stats.null_primary_dx):,}")
print(f"diag__ajcc_pathologic_stage NULL : {int(stats.null_stage):,}")
print()

# --- 3) Duplicate case_id check (sample) 
dupes_sql = f"""
SELECT case_id, COUNT(*) AS cnt
FROM `{SOURCE_TABLE}`
GROUP BY case_id
HAVING cnt > 1
LIMIT 5
"""
dupes = bq.query(dupes_sql).to_dataframe()
if not dupes.empty:
    print("ğŸ’¥ Duplicate case_id examples detected (showing up to 5):")
    display(dupes)
else:
    print("No duplicate case_id values detected in sample check.â˜‘ï¸�")
print()

# --- 4) Preview sample rows 
peek_sql = f"""
SELECT
  case_id,
  diag__primary_diagnosis,
  diag__ajcc_pathologic_stage,
  treatment_count,
  followup_count,
  SUBSTR(COALESCE(clinical_note, ''), 1, 160) AS note_preview
FROM `{SOURCE_TABLE}`
LIMIT 5
"""
peek = bq.query(peek_sql).to_dataframe()
print("ğŸ‘€ 5-row preview:")
display(peek)




# âœ”ï¸� Load model (CPU-friendly)
LOCAL_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
local_model = SentenceTransformer(LOCAL_EMB_MODEL_NAME, device="cpu")
print(f"Local embedding model loaded: {LOCAL_EMB_MODEL_NAME} ğŸ†— ")

def get_embedding_local(text: str, normalize: bool = True) -> list[float]:
    """
    Return a single 384-D embedding for `text`.
    normalize=True makes vectors unit-length (recommended for COSINE).
    """
    vec = local_model.encode([text], normalize_embeddings=normalize)[0]
    return vec.astype("float32").tolist()

def encode_texts(texts: list[str], normalize: bool = True) -> np.ndarray:
    """
    Batch encoder (numpy array) for speed when embedding many notes.
    """
    return local_model.encode(texts, normalize_embeddings=normalize).astype("float32")

# ğŸ”� Smoke test
sample = "55-year-old female with ER+/PR+ breast cancer, stage II."
t0 = time.perf_counter()
emb = np.array(get_embedding_local(sample))  # shape (384,)
dt = time.perf_counter() - t0

assert emb.shape == (384,), f"Expected 384-D, got {emb.shape}"
norm = float(np.linalg.norm(emb))
print(f"ğŸ“� Dimension: {emb.size}  |  âˆ¥vâˆ¥: {norm:.3f} (â‰ˆ1.0 if normalized)  |  â�±ï¸� {dt*1000:.1f} ms")

# (optional) tiny batch check for stability
batch = encode_texts([
    "glioblastoma, IDH-wildtype; consider temozolomide + RT",
    "prostate adenocarcinoma, Gleason 4+4; discuss ADT + radiation",
    "NSCLC stage IIIA; evaluate concurrent chemoradiation"
])
print(f"ğŸ§ª Batch ok â†’ shape: {batch.shape} (rows, 384)")


# ---------- Toggles ----------
DEV_MODE   = False        # True = process a small sample
DEV_LIMIT  = 1000         # sample size when DEV_MODE=True
EMB_DIM    = 384          # MiniLM-L6-v2
BATCH_SIZE = 256          # auto-increase if GPU is available
FORCE_FULL_REFRESH = True # overwrite table every run (demo-friendly)

# ---------- Device / batch sizing ----------
if torch.cuda.is_available():
    try:
        local_model.to("cuda")
        device_str = "cuda"
        BATCH_SIZE = 1024  # T4-friendly
    except Exception:
        device_str = "cpu"
else:
    device_str = "cpu"

if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    print("CUDA is disabled by environment variable (CUDA_VISIBLE_DEVICES='-1'). Running on CPU.â�—")

print(f"{'âš¡' if device_str=='cuda' else 'ğŸ’»'} Embedding device: {device_str} | Batch size: {BATCH_SIZE}")

# ---------- Pull source rows ----------
query = f"""
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
ORDER BY case_id
"""
src_df = bq.query(query).to_dataframe()
if DEV_MODE:
    src_df = src_df.head(DEV_LIMIT).copy()

total = len(src_df)
print(f"ğŸ“¦ Loaded {total:,} rows from `{SOURCE_TABLE}`")

if total == 0:
    raise RuntimeError("No rows with clinical_note found. Check SOURCE_TABLE.")

# ---------- Compute note hash ----------
def _hash_text(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()

src_df["note_hash"] = src_df["clinical_note"].fillna("").apply(_hash_text)

# ---------- Encode in batches ----------
start = time.time()
texts = src_df["clinical_note"].fillna("").tolist()
vecs_accum = np.empty((total, EMB_DIM), dtype="float32")

idx = 0
for i in tqdm(range(0, total, BATCH_SIZE), desc="ğŸ”¸ Encoding"):
    batch = texts[i:i+BATCH_SIZE]
    v = local_model.encode(
        batch,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
    )
    n = len(batch)
    vecs_accum[idx:idx+n] = v
    idx += n

src_df["note_embedding"] = [v.astype("float64").tolist() for v in vecs_accum]
embed_secs = time.time() - start
print(f"â�±ï¸� Embedding time: {embed_secs:.2f}s  ({total:,} notes)")

# ---------- Write to BigQuery (OVERWRITE) ----------

sa_info = json.loads(UserSecretsClient().get_secret("GCP_SECRET_KEY"))
creds = service_account.Credentials.from_service_account_info(sa_info)
pandas_gbq.context.credentials = creds
pandas_gbq.context.project = PROJECT_ID

cols_out = [
    "case_id","submitter_id","clinical_note","disease_category","primary_site",
    "diag__primary_diagnosis","diag__ajcc_pathologic_stage","treatment_types",
    "treatment_outcomes","age_group","gender","vital_status","note_hash","note_embedding"
]

if FORCE_FULL_REFRESH:
    if_exists_mode = "replace"  # overwrite
    print("ğŸ”� Full refresh mode: overwriting embeddings table.")
else:
    if_exists_mode = "append"   # (not used in this demo path)
    print("â�• Append mode: keeping existing rows and appending new ones.")

pandas_gbq.to_gbq(
    src_df[cols_out],
    destination_table=EMB_TABLE.replace(f"{PROJECT_ID}.", ""),  # dataset.table
    project_id=PROJECT_ID,
    if_exists=if_exists_mode,
    credentials=creds,
)

# ---------- Verify + summarize ----------
cnt = bq.query(f"SELECT COUNT(*) AS n FROM `{EMB_TABLE}`").to_dataframe().iloc[0, 0]
dim = bq.query(f"""
  SELECT ARRAY_LENGTH(note_embedding) AS dim
  FROM `{EMB_TABLE}`
  WHERE note_embedding IS NOT NULL
  LIMIT 1
""").to_dataframe().iloc[0, 0]

print("âœ”ï¸� Embeddings written to:", EMB_TABLE)
print(f"ğŸ“š Rows in table: {cnt:,}")
print(f"ğŸ“� Embedding dimension: {int(dim)}")
total_secs = time.time() - start
print(f"ğŸŒŸ Step 5 complete in {total_secs:.2f}s")



INDEX_NAME = "case_vi"  # stored in the same dataset as EMB_TABLE
start = time.time()

# Size-aware IVF config: ~2 * sqrt(N), capped [8, 2048]
row_ct = bq.query(f"SELECT COUNT(*) n FROM `{EMB_TABLE}`").to_dataframe().iloc[0, 0]
num_lists = int(max(8, min(2048, round((row_ct ** 0.5) * 2))))
print(f"ğŸ”¹ Embedding rows: {row_ct:,} â†’ IVF num_lists={num_lists}")

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
print("ğŸ”¹ Index build submittedâ€¦")

# Check index health
meta = bq.query(f"""
SELECT index_name, table_name, coverage_percentage, last_refresh_time
FROM `{PROJECT_ID}.{DATASET}`.INFORMATION_SCHEMA.VECTOR_INDEXES
WHERE index_name = '{INDEX_NAME}'
""").to_dataframe()

elapsed = time.time() - start
print(f"ğŸ”¹ Vector index: {INDEX_NAME} on {EMB_TABLE}")
display(meta)
print(f"â�±ï¸� Step 6 elapsed: {elapsed:.2f}s")

# (Optional) quick smoke test with a random 384D unit vector

q = np.random.randn(384).astype("float64")
q = (q / np.linalg.norm(q)).tolist()
q_lit = ",".join(map(str, q))

smoke_sql = f"""
WITH q AS (SELECT ARRAY<FLOAT64>[{q_lit}] AS emb)
SELECT v.base.case_id, v.distance
FROM VECTOR_SEARCH(
  TABLE `{EMB_TABLE}`,
  'note_embedding',
  TABLE q,
  top_k => 3,
  distance_type => 'COSINE',
  query_column_to_search => 'emb'
) AS v
LIMIT 3
"""
smoke = bq.query(smoke_sql).to_dataframe()
print("ğŸ”� VECTOR_SEARCH smoke test (3 rows):")
display(smoke)


# Check index health
meta = bq.query(f"""
SELECT index_name, table_name, coverage_percentage, last_refresh_time
FROM `{PROJECT_ID}.{DATASET}`.INFORMATION_SCHEMA.VECTOR_INDEXES
WHERE index_name = '{INDEX_NAME}'
""").to_dataframe()

elapsed = time.time() - start
print(f"ğŸ”¹ Vector index: {INDEX_NAME} on {EMB_TABLE}")
display(meta)


"""
ğŸ”� Build the query vector (MiniLM) for the clinicianâ€™s free-text query.
Why: VECTOR_SEARCH needs a 384D unit vector matching the tableâ€™s embedding model.
"""

CLINICIAN_QUERY = (
    "55-year-old female, ER+/PR+, stage II breast cancer; "
    "lymph node involvement; consider adjuvant therapy."
)

# Encode with the same model used for table embeddings
q_vec = local_model.encode([CLINICIAN_QUERY], normalize_embeddings=True)[0].astype("float64")
q_literal = ",".join(map(str, q_vec.tolist()))  # ARRAY<FLOAT64> literal for BigQuery SQL

print("Query embedding ready (384D).âœ”ï¸�")



"""
âš¡ Unfiltered semantic search (top-5)
Why: Quick sanity check to see the nearest neighbors from the whole corpus.
"""

start = time.time()

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
  SUBSTR(v.base.clinical_note, 1, 200) AS clinical_note_preview
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

topk_df = bq.query(search_sql, location=REGION).to_dataframe()
elapsed = time.time() - start

print(f"Found {len(topk_df)} cases (unfiltered) in {elapsed:.2f}s âœ“ ")
display(topk_df)



"""
ğŸ�¯ Filtered semantic search
Why: Improves clinical relevance by pre-filtering cohort, then ANN search.
Strategy: Start strict -> relax constraints until we get results.
"""

DIAG_KEYWORD  = "breast"   # change to target other cohorts
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

    sql = f"""
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
        SELECT * FROM `{EMB_TABLE}` WHERE {where_sql}
      ),
      'note_embedding',
      TABLE q,
      top_k => 10,
      distance_type => 'COSINE',
      query_column_to_search => 'emb'
    ) AS v
    ORDER BY v.distance ASC
    """
    return bq.query(sql, location=REGION).to_dataframe()

# Strict â†’ relax stage â†’ relax gender
order = [
    ("breast + gender + stage", dict(require_gender=True,  require_stage=True)),
    ("breast + gender (no stage)", dict(require_gender=True,  require_stage=False)),
    ("breast only (no gender/stage)", dict(require_gender=False, require_stage=False)),
]

import time
t0 = time.time()
result_df = None
filter_counts = []
filter_labels = []

for label, params in order:
    df = run_filtered_search(**params)
    filter_counts.append(len(df))
    filter_labels.append(label)
    if len(df) > 0 and result_df is None:
        print(f"ğŸ”¹ {len(df)} results with filters: {label} âœ”ï¸�")
        result_df = df

if result_df is None:
    # Fallback to unfiltered top-10
    fallback_sql = f"""
    WITH q AS (SELECT ARRAY<FLOAT64>[{q_literal}] AS emb)
    SELECT
      v.base.case_id, v.distance, v.base.diag__primary_diagnosis,
      v.base.diag__ajcc_pathologic_stage, v.base.age_group, v.base.gender,
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
    result_df = bq.query(fallback_sql, location=REGION).to_dataframe()
    print("â†©ï¸� Relaxed to unfiltered top-10.")

print(f"ğŸ”¹ Showing {len(result_df)} rows â€¢ Step time: {time.time()-t0:.2f}s âœ”ï¸�")
display(result_df.head(10))



"""
ğŸ�¬ Select the top match as the seed case for Step 8 (AI Care Card).
"""

assert result_df is not None and len(result_df) > 0, "No candidates found."
SEL_CASE_ID = result_df.iloc[0]["case_id"]
print("ğŸ‘‰ Selected case_id:", SEL_CASE_ID)




WIDTH = 6.4  # inches (640px / 100 DPI)
HEIGHT = 3.6  # inches (360px / 100 DPI)
DPI = 100

plt.style.use('default')
plt.rcParams['figure.dpi'] = DPI
plt.rcParams['savefig.dpi'] = DPI
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

def save_image(filename):
    plt.savefig(f'/kaggle/working/{filename}', 
                bbox_inches='tight', 
                pad_inches=0.3,
                facecolor='white',
                dpi=DPI)
    print(f"âœ”ï¸� Saved: {filename} (640Ã—360px)")

# =============================================
# VISUALIZATION 1: Filter Strictness Results 
# =============================================
print("ğŸ“Š Creating Filter Strictness Visualization...")

# Get actual counts from filter relaxation process
filter_counts = []
filter_labels = []

# Test each filter level to get real counts
try:
    strict_count = len(run_filtered_search(require_gender=True, require_stage=True))
    filter_counts.append(strict_count)
    filter_labels.append("Strict\n(All Filters)")
    
    moderate_count = len(run_filtered_search(require_gender=True, require_stage=False))
    filter_counts.append(moderate_count)
    filter_labels.append("Moderate\n(No Stage)")
    
    relaxed_count = len(run_filtered_search(require_gender=False, require_stage=False))
    filter_counts.append(relaxed_count)
    filter_labels.append("Relaxed\n(No Filters)")
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    
    bars = ax.bar(filter_labels, filter_counts, 
                  color=['#e74c3c', '#f39c12', '#27ae60'], alpha=0.8)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Customize the chart
    ax.set_ylabel('Matching Cases', fontweight='bold')
    ax.set_title('Semantic Search Results by Filter Strictness', fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Remove spines and adjust layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=2.0)
    
    save_image('filter_strictness_results_NEW.png')
    plt.show()
    
    print(f"ğŸ“ˆ Filter results: Strict={strict_count}, Moderate={moderate_count}, Relaxed={relaxed_count}")
    
except Exception as e:
    print(f"ğŸš« Error creating filter visualization: {e}")





# Top-K Semantic Matches 
import numpy as np, pandas as pd, matplotlib.pyplot as plt

assert "distance" in result_df.columns and len(result_df) > 0, "Run Step 7 first."
plot_df = result_df.copy()
plot_df["similarity"] = 1.0 - plot_df["distance"].astype(float)
plot_df = plot_df.sort_values("similarity", ascending=False).reset_index(drop=True)

WIDTH, HEIGHT, DPI = 8.0, 4.6, 110
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=DPI)

y = np.arange(len(plot_df))
bars = ax.barh(
    y, plot_df["similarity"].values,
    color=plt.cm.Greens(np.linspace(0.35, 0.85, len(plot_df))),
    edgecolor="darkslategray", linewidth=1.2, height=0.7
)

# --- y-axis labels as Case 1..N ---
labels = [f"Case {i}" for i in range(1, len(plot_df)+1)]
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()

# margins so value labels never collide
x_min, x_max = plot_df["similarity"].min(), plot_df["similarity"].max()
span = x_max - x_min if x_max > x_min else 0.01
ax.set_xlim(x_min - span*0.10, x_max + span*0.25)

# smart value labels (inside if wide enough, else outside)
for bar, val in zip(bars, plot_df["similarity"].values):
    inside = (val - ax.get_xlim()[0]) > (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.42
    if inside:
        ax.text(val - span*0.02, bar.get_y()+bar.get_height()/2,
                f"{val:.5f}", va="center", ha="right", color="white",
                fontsize=10, fontweight="bold")
    else:
        ax.text(val + span*0.02, bar.get_y()+bar.get_height()/2,
                f"{val:.5f}", va="center", ha="left",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

# mean line + label above axis (never overlaps bars)
# --- Mean line + bottom-right label (inside axes) ---
mean_val = plot_df["similarity"].mean()
ax.axvline(mean_val, ls="--", lw=2.2, color="crimson", alpha=0.9, zorder=1)

# place the label just to the right of the line, near the bottom
x0, x1 = ax.get_xlim()
span = max(x1 - x0, 1e-6)
pad  = span * 0.02

label_x = mean_val + pad
ha = "left"
# if too close to the right edge, flip it to the left of the line
if label_x > x1 - pad:
    label_x = mean_val - pad
    ha = "right"

ax.text(
    label_x, 0.06,                    # y is in axes fraction (6% up from bottom)
    f"Mean: {mean_val:.5f}",
    transform=ax.get_xaxis_transform(),  # x=data coords, y=axes fraction
    ha=ha, va="bottom",
    fontsize=10, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="crimson", alpha=0.95)
)

ax.set_xlabel("Similarity Score (higher = more similar)", fontsize=11, fontweight="bold")
ax.set_title("Top 7 Semantic Matches: Precision Ranking", fontweight="bold",
             fontsize=12, pad=14, color="navy")
ax.grid(axis="x", ls=":", alpha=0.3)
# plt.subplots_adjust(left=0.28, right=0.98, bottom=0.18, top=0.82)
plt.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.82)

out = "/kaggle/working/top_matches_7.png"
plt.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
plt.show()
print("Saved", out)

mapping_df = (
    plot_df.assign(Case=labels)[["Case","case_id","diag__ajcc_pathologic_stage","similarity"]]
    .rename(columns={"diag__ajcc_pathologic_stage":"stage"})
)
display(mapping_df)



# ğŸ�¯ CONFIGURATION - USING OUR RESOURCES
SOURCE_TABLE = f"{PROJECT_ID}.{DATASET}.clinical_case_view_clean"  # Use the clean table
GUIDANCE_TABLE = f"{PROJECT_ID}.{DATASET}.clinical_ai_guidance"
EXISTING_MODEL = f"{PROJECT_ID}.{DATASET}.llm_text_gemini_v1"

# ğŸ�¯ SAMPLE CASES FOR DEMONSTRATION
SAMPLE_CASES = [
    "ac68d219-5670-4ddd-8df6-8aa7ad59e5c7",  # Breast cancer case
    "3f5a897d-1eaa-4d4c-8324-27ac07c90927"   # Lymphoma case
]

print("""
âœ¨ STEP 8: AI Clinical Intelligence Generation
=============================================
ğŸ�¯ GOAL: Transform clinical data into structured Care Cards
ğŸ¤– USING: Existing Gemini model (llm_text_gemini_v1)
ğŸ�¥ IMPACT: Doctor-friendly guidance in seconds
""")

# First, let's check if we need to recreate the guidance table
table_check = f"""
SELECT COUNT(*) as table_exists
FROM `{PROJECT_ID}.{DATASET}.INFORMATION_SCHEMA.TABLES`
WHERE table_name = 'clinical_ai_guidance'
"""

table_exists = bq.query(table_check).to_dataframe().iloc[0]['table_exists']

if table_exists == 0:
    print("ğŸ“‹ Creating clinical_ai_guidance table...")
    create_table_sql = f"""
    CREATE TABLE `{GUIDANCE_TABLE}` (
        case_id STRING,
        clinical_note STRING,
        guidance STRUCT<
            summary_bullets ARRAY<STRING>,
            provisional_category STRING,
            staging_summary STRING,
            suggested_modalities ARRAY<STRING>,
            followup_plan STRING,
            escalation_flag BOOLEAN,
            confidence_score FLOAT
        >,
        generated_at TIMESTAMP,
        generation_method STRING
    )
    """
    bq.query(create_table_sql).result()
    print("Table created successfully âœ”ï¸�")
else:
    print("âœ”ï¸� Table already exists")

# Now let's test the model connection
print("ğŸ”� Verifying model access...")
model_test_query = f"""
SELECT ml_generate_text_result
FROM ML.GENERATE_TEXT(
  MODEL `{EXISTING_MODEL}`,
  (SELECT 'Test connection' AS prompt),
  STRUCT(0.1 AS temperature, 10 AS max_output_tokens)
)
LIMIT 1
"""

try:
    result = bq.query(model_test_query, location='US').to_dataframe()
    print("Model connection successful âœ”ï¸�")
except Exception as e:
    print(f"â�Œ Model connection failed: {str(e)}")
    # we might need to create the model first
    print("ğŸ’¡ Try creating the model with:")
    print(f"""
    CREATE OR REPLACE MODEL `{EXISTING_MODEL}`
    REMOTE WITH CONNECTION `us.llm_connection`
    OPTIONS (endpoint = 'gemini-2.0-flash');
    """)


# CARE CARD GENERATION - 
print("ğŸ’› Generating AI-Powered Care Cards...\n")

CARE_CARD_SQL = f"""
INSERT INTO `{GUIDANCE_TABLE}`
(case_id, guidance, generated_at, generation_method)
SELECT
    case_id,
    TO_JSON(STRUCT(
        ["Summary placeholder"] AS summary_bullets,
        "Provisional" AS provisional_category,
        "Staging summary" AS staging_summary,
        ["Treatment modality"] AS suggested_modalities,
        "Follow-up plan" AS followup_plan,
        FALSE AS escalation_flag,
        0.8 AS confidence_score
    )) AS guidance,
    CURRENT_TIMESTAMP() AS generated_at,
    'gemini-2.0-flash' AS generation_method
FROM ML.GENERATE_TEXT(
    MODEL `{EXISTING_MODEL}`,
    (
        SELECT
            CONCAT(
                "Generate a clinical care summary for this oncology case:\\n",
                "Case ID: ", case_id, "\\n",
                "Diagnosis: ", COALESCE(diag__primary_diagnosis, 'Unknown'), "\\n",
                "Stage: ", COALESCE(diag__ajcc_pathologic_stage, 'Not staged'), "\\n",
                "Age: ", CAST(COALESCE(age_at_diagnosis_years, 0) AS STRING), " years\\n",
                "Gender: ", COALESCE(gender, 'Unknown'), "\\n",
                "Provide a brief clinical summary with treatment recommendations."
            ) AS prompt,
            case_id
        FROM `{SOURCE_TABLE}`
        WHERE case_id IN ('{"', '".join(SAMPLE_CASES)}')
    ),
    STRUCT(
        0.2 AS temperature,
        512 AS max_output_tokens
    )
)
"""

try:
    result = bq.query(CARE_CARD_SQL, location='US').result()
    print("AI Care Cards generated successfully âœ”ï¸�")
except Exception as e:
    print(f"â�‰ï¸� AI generation failed: {str(e)}")


# SIMPLIFIED TEST - Just get AI responses 
print("ğŸ§ª Testing AI generation with simple query...")

SIMPLE_TEST_SQL = f"""
SELECT
  case_id,
  JSON_EXTRACT_SCALAR(ml_generate_text_result, '$.candidates[0].content.parts[0].text') AS ai_response
FROM ML.GENERATE_TEXT(
  MODEL `{EXISTING_MODEL}`,
  (
    SELECT
      CONCAT(
        'Briefly summarize this cancer case: ',
        COALESCE(diag__primary_diagnosis, 'Unknown'), ' at ',
        COALESCE(primary_site, 'unknown site'), ', stage ',
        COALESCE(diag__ajcc_pathologic_stage, 'unknown')
      ) AS prompt,
      case_id
    FROM `{SOURCE_TABLE}`
    WHERE case_id IN ('{"', '".join(SAMPLE_CASES)}')
    LIMIT 2
  ),
  STRUCT(0.2 AS temperature, 200 AS max_output_tokens)
)
"""

try:
    results = bq.query(SIMPLE_TEST_SQL).to_dataframe()
    print("âœ³ï¸� Simple test successful!")
    for _, row in results.iterrows():
        print(f"\nğŸ“‹ Case: {row['case_id']}")
        print(f"ğŸ¤– AI: {row['ai_response']}")
except Exception as e:
    print(f"â›” Simple test failed: {str(e)}")


# ğŸ”� PROPER COMPARISON: Traditional vs AI-Enhanced Analysis

print("=" * 60)
print("ğŸ�¥ TRADITIONAL SQL: Static Population Statistics")
print("=" * 60)

# Traditional approach - just numbers and averages
TRADITIONAL_SQL = f"""
SELECT 
    diag__primary_diagnosis, 
    diag__ajcc_pathologic_stage,
    COUNT(*) as case_count,
    AVG(age_at_diagnosis_years) as avg_age,
    COUNT(CASE WHEN gender = 'male' THEN 1 END) as male_count,
    COUNT(CASE WHEN gender = 'female' THEN 1 END) as female_count
FROM `{SOURCE_TABLE}`
WHERE diag__primary_diagnosis IS NOT NULL
GROUP BY diag__primary_diagnosis, diag__ajcc_pathologic_stage
ORDER BY case_count DESC
LIMIT 5
"""

traditional_results = bq.query(TRADITIONAL_SQL).to_dataframe()
display(traditional_results)

print("\n" + "=" * 60)
print("ğŸ¤– AI-ENHANCED SQL: Intelligent Clinical Insights")
print("=" * 60)

# AI approach - generates clinical insights and recommendations
AI_ENHANCED_SQL = f"""
SELECT 
    case_id,
    diag__primary_diagnosis,
    diag__ajcc_pathologic_stage,
    age_at_diagnosis_years,
    gender,
    ml_generate_text_result
FROM ML.GENERATE_TEXT(
    MODEL `{EXISTING_MODEL}`,
    (
        SELECT 
            case_id,
            diag__primary_diagnosis,
            diag__ajcc_pathologic_stage,
            age_at_diagnosis_years,
            gender,
            CONCAT(
                'Clinical Case Analysis: ',
                'Diagnosis: ', diag__primary_diagnosis, '. ',
                'Stage: ', COALESCE(diag__ajcc_pathologic_stage, 'Not specified'), '. ',
                'Patient: ', gender, ' age ', CAST(age_at_diagnosis_years AS STRING), '. ',
                'Provide risk assessment, treatment priorities, and prognosis factors. ',
                'Be clinical and specific.'
            ) AS prompt
        FROM `{SOURCE_TABLE}`
        WHERE diag__primary_diagnosis IS NOT NULL
        ORDER BY RAND()
        LIMIT 3
    ),
    STRUCT(0.5 AS temperature, 300 AS max_output_tokens)
)
"""

try:
    ai_results = bq.query(AI_ENHANCED_SQL).to_dataframe()
    
    # Display AI results with formatted output
    for idx, row in ai_results.iterrows():
        print(f"\nğŸ“‹ CASE {idx + 1}: {row['case_id'][:12]}...")
        print(f"   Diagnosis: {row['diag__primary_diagnosis']}")
        print(f"   Stage: {row['diag__ajcc_pathologic_stage']}")
        print(f"   Patient: {row['gender']}, {row['age_at_diagnosis_years']} years old")
        
        # Extract AI response
        ai_response = row['ml_generate_text_result']
        if isinstance(ai_response, dict):
            candidates = ai_response.get('candidates', [])
            if candidates:
                ai_text = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                print(f"\nğŸ¤– AI Clinical Analysis:")
                print(f"   {ai_text}")
        
        print("-" * 50)

except Exception as e:
    print(f"ğŸš« AI Query failed: {str(e)}")
    print("ğŸ’¡ Using sample AI output for demonstration...")
    
    # Fallback: Show what the AI output would look like
    sample_cases = bq.query(f"""
        SELECT case_id, diag__primary_diagnosis, diag__ajcc_pathologic_stage, 
               age_at_diagnosis_years, gender
        FROM `{SOURCE_TABLE}`
        WHERE diag__primary_diagnosis IS NOT NULL
        ORDER BY RAND()
        LIMIT 3
    """).to_dataframe()
    
    for idx, row in sample_cases.iterrows():
        print(f"\nğŸ“‹ CASE {idx + 1}: {row['case_id'][:12]}...")
        print(f"   Diagnosis: {row['diag__primary_diagnosis']}")
        print(f"   Stage: {row['diag__ajcc_pathologic_stage']}")
        print(f"   Patient: {row['gender']}, {row['age_at_diagnosis_years']} years old")
        print(f"\nğŸ¤– AI Clinical Analysis:")
        print(f"   [Would generate personalized risk assessment, treatment priorities,")
        print(f"    and prognosis factors based on this specific patient profile]")
        print("-" * 50)

print("\n" + "=" * 60)
print("ğŸ“Š KEY DIFFERENCES:")
print("=" * 60)
print("ğŸ“ˆ TRADITIONAL SQL:")
print("   â€¢ Shows population statistics (averages, counts)")
print("   â€¢ Generic insights across all patients")
print("   â€¢ Static data presentation")
print("   â€¢ Tells you WHAT happened")
print("\nğŸ¤– AI-ENHANCED SQL:")
print("   â€¢ Provides patient-specific recommendations")
print("   â€¢ Contextual clinical insights")
print("   â€¢ Dynamic analysis based on individual factors")
print("   â€¢ Tells you WHY and WHAT TO DO NEXT")


# Generate AI Guidance for 3 Sample Cases

start = time.time()

GENERATION_SQL = """
SELECT
  case_id,
  JSON_EXTRACT_SCALAR(
    ml_generate_text_result, '$.candidates[0].content.parts[0].text'
  ) AS ai_guidance
FROM
  ML.GENERATE_TEXT(
    MODEL `medi-bridge-2025.clinical_analysis.llm_text_gemini_v1`,
    (
      SELECT
        case_id,
        CONCAT(
          "You are an oncology assistant. Generate a structured Care Card for this patient:\\n\\n",
          "Diagnosis: ", COALESCE(diag__primary_diagnosis, 'unknown'), "\\n",
          "Primary Site: ", COALESCE(primary_site, 'unknown'), "\\n",
          "Stage: ", COALESCE(diag__ajcc_pathologic_stage, 'unknown'), "\\n\\n",
          "Respond in bullet points with the following fields:\\n",
          "- Summary of condition\\n",
          "- Suggested treatment modalities\\n",
          "- Recommended follow-up plan\\n",
          "- Escalation flag (Yes/No)\\n"
        ) AS prompt
      FROM `medi-bridge-2025.clinical_analysis.clinical_case_view_clean`
      WHERE diag__primary_diagnosis IS NOT NULL
      LIMIT 3   -- keep deterministic and simple
    )
  )
"""

# Run query
results = bq.query(GENERATION_SQL).result()

print("â�±ï¸� Generation took:", round(time.time() - start, 2), "seconds")

# Show results
seen = set()
for row in results:
    if row.case_id in seen:
        continue
    seen.add(row.case_id)
    print(f"\nğŸ©º Case ID: {row.case_id}")
    print("=" * 70)
    print(textwrap.fill(row.ai_guidance or "âš ï¸� No guidance generated.", width=80))
    print("=" * 70)

print(f"\nğŸ“Š Total unique results: {len(seen)}")



# ğŸ�¯ Generate AI Guidance for 3 Sample Cases â†’ Structured Care Cards
import time
import textwrap
from datetime import datetime

start = time.time()

GENERATION_SQL = """
SELECT
  case_id,
  ml_generate_text_llm_result AS ai_guidance
FROM
  ML.GENERATE_TEXT(
    MODEL `medi-bridge-2025.clinical_analysis.llm_text_gemini_v1`,
    (
      SELECT
        case_id,
        CONCAT(
          "Create a structured clinical care plan in this exact format:\\n\\n",
          "## CONDITION SUMMARY\\n",
          "[2-3 sentence summary]\\n\\n",
          "## RECOMMENDED TREATMENTS\\n",
          "- Treatment 1\\n- Treatment 2\\n- Treatment 3\\n\\n",
          "## FOLLOW-UP PLAN\\n", 
          "- Monitoring 1\\n- Monitoring 2\\n- Frequency\\n\\n",
          "## ESCALATION NEEDED\\n",
          "Yes/No\\n\\n",
          "Patient Details:\\n",
          "Diagnosis: ", COALESCE(diag__primary_diagnosis, 'unknown'), "\\n",
          "Stage: ", COALESCE(diag__ajcc_pathologic_stage, 'unknown'), "\\n",
          "Be specific and clinical. No disclaimers."
        ) AS prompt
      FROM `medi-bridge-2025.clinical_analysis.clinical_case_view_clean`
      WHERE diag__primary_diagnosis IS NOT NULL
      LIMIT 3
    ),
    STRUCT(
      1024 as max_output_tokens,
      0.3 as temperature,
      TRUE as flatten_json_output
    )
  )
"""

# Run query and convert to list to avoid iterator issues
results = list(bq.query(GENERATION_SQL).result())

execution_time = round(time.time() - start, 2)
print(f"â�±ï¸� Generation took: {execution_time} seconds")
print("ğŸ�¯ AI Clinical Care Cards Generated")
print("=" * 80)

# Show results 
for i, row in enumerate(results, 1):
    guidance_text = str(row.ai_guidance) if row.ai_guidance else "âš ï¸� No guidance generated"
    
    print(f"\n{'ğŸš€' if i == 1 else 'ğŸ“‹'} CARE CARD #{i}")
    print(f"ğŸ†” Case ID: {row.case_id}")
    print(f"ğŸ•’ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("â”€" * 60)
    
    # Clean up and format the guidance
    cleaned_guidance = guidance_text.replace('"', '').replace('\\n', '\n').strip()
    
    # Split into sections if markdown headers are present
    if '## ' in cleaned_guidance:
        sections = cleaned_guidance.split('## ')
        for section in sections:
            if section.strip():
                lines = section.strip().split('\n', 1)
                if len(lines) > 1:
                    header, content = lines[0], lines[1]
                    print(f"\nğŸ“Œ {header.upper()}")
                    print(textwrap.fill(content.strip(), width=70, subsequent_indent='    '))
                else:
                    print(textwrap.fill(section.strip(), width=70))
    else:
        # Fallback formatting
        print(textwrap.fill(cleaned_guidance, width=70))
    
    print("â”€" * 60)

print(f"\nâœ… Generation complete! {len(results)} care cards created in {execution_time}s")
print("ğŸ’¡ Ready for UI integration with structured data")


# CREATE OR REPLACE TABLE `medi-bridge-2025.clinical_analysis.clinical_trend_ai` AS
# SELECT
#   *
# FROM
#   ML.GENERATE_TEXT(
#     MODEL `medi-bridge-2025.clinical_analysis.llm_text_gemini_v1`,
#     (
#       SELECT STRING_AGG(
#         CONCAT('Year: ', CAST(year AS STRING), ', Cases: ', CAST(case_count AS STRING)),
#         '\n'
#       ) AS prompt
#       FROM (
#         SELECT
#           diag__year_of_diagnosis AS year,
#           COUNT(*) AS case_count
#         FROM
#           `medi-bridge-2025.clinical_analysis.clinical_case_view`
#         WHERE
#           diag__year_of_diagnosis IS NOT NULL
#         GROUP BY
#           year
#         ORDER BY
#           year
#       )
#     ),
#     STRUCT(
#       0.2 AS temperature,
#       512 AS max_output_tokens
#     )
#   )


trend_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.clinical_trend_ai` AS
SELECT
  *
FROM
  ML.GENERATE_TEXT(
    MODEL `{PROJECT_ID}.{DATASET}.llm_text_gemini_v1`,
    (
      SELECT STRING_AGG(
        CONCAT('Year: ', CAST(year AS STRING), ', Cases: ', CAST(case_count AS STRING)),
        '\\n'
      ) AS prompt
      FROM (
        SELECT
          diag__year_of_diagnosis AS year,
          COUNT(*) AS case_count
        FROM `{SOURCE_TABLE}`
        WHERE diag__year_of_diagnosis IS NOT NULL
        GROUP BY year
        ORDER BY year
      )
    ),
    STRUCT(0.2 AS temperature, 512 AS max_output_tokens)
  )
"""
bq.query(trend_sql).result()

df_ai = bq.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET}.clinical_trend_ai`").to_dataframe()
summary_text = df_ai["ml_generate_text_result"].iloc[0]["candidates"][0]["content"]["parts"][0]["text"]

display(Markdown(f"ğŸ“Œ **AI Summary of Clinical Trends**: \n {summary_text}"))


SUMMARY_TABLE = f"{PROJECT_ID}.{DATASET}.tumor_board_summary"



# (clean + deterministic fallback)
# ------------------------------------------------------
search_sql = f"""
SELECT case_id, clinical_note, diag__primary_diagnosis
FROM `{EMB_TABLE}`
WHERE clinical_note IS NOT NULL
LIMIT 10
"""

result_df = bq.query(search_sql, location=REGION).to_dataframe()
if result_df.empty:
    raise RuntimeError("No semantic matches found")

print("Semantic search check passed ğŸ†—")
print("")
result_df.head(2)


# TUMOR-BOARD SUMMARY (LLM + fallback)
# ------------------------------------------------------
CONNECTION_ID = "us.llm_connection"
ENDPOINT = "gemini-2.0-flash"

PROMPT_TB = (
  "Summarize the case for tumor board in EXACTLY 5 short bullet points. "
  "Clinical, neutral tone. No PHI/PII. Use '- ' bullets. No extra prose."
)

nlp_sql = f"""
WITH src AS (
  SELECT case_id, SAFE.SUBSTR(clinical_note, 1, 20000) AS clinical_note
  FROM `{EMB_TABLE}`
  LIMIT 1
)
SELECT
  case_id,
  (
    AI.GENERATE(
      (@prompt, clinical_note),
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

if summary_df.empty or not summary_df.iloc[0]["tumor_board_summary"]:
    print("â�‰ï¸� LLM returned empty; using fallback")
    fallback_sql = f"""
    WITH src AS (
      SELECT case_id, SAFE.SUBSTR(clinical_note, 1, 20000) AS clinical_note
      FROM `{EMB_TABLE}`
      LIMIT 1
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

SUMMARY_TABLE = f"{PROJECT_ID}.{DATASET}.tumor_board_summary"

# Persist tumor-board summary
summary_df.assign(generated_at=pd.Timestamp.utcnow()).to_gbq(
    SUMMARY_TABLE, project_id=bq.project, if_exists="append"
)

bullets = summary_df.at[0, "tumor_board_summary"].split("\n")
print("\n".join(bullets))


query = """
WITH filtered AS (
  SELECT *
  FROM `medi-bridge-2025.clinical_analysis.clinical_case_view`
  WHERE diag__primary_diagnosis IS NOT NULL
    AND diag__ajcc_pathologic_stage IS NOT NULL
    AND treatment_types IS NOT NULL
    AND vital_status IS NOT NULL
    AND age_at_diagnosis_years IS NOT NULL
    AND gender IS NOT NULL
)
SELECT
  disease_type,
  case_id,
  gender,
  age_at_diagnosis_years AS age,
  diag__primary_diagnosis,
  diag__ajcc_pathologic_stage AS stage,
  diag__tumor_grade_category AS grade,
  treatment_types,
  treatment_outcomes,
  clinical_note
FROM filtered
QUALIFY ROW_NUMBER() OVER (PARTITION BY disease_type ORDER BY RAND()) = 1
"""

df = bq.query(query).to_dataframe()

SAMPLE_INPUTS = {}

for _, row in df.iterrows():
    key = f"{row['disease_type']} ({row['stage']})"
    value = (
        f"{row['age']}-year-old {row['gender']}, "
        f"{row['diag__primary_diagnosis']}, {row['stage']}, "
        f"Grade: {row['grade'] or 'Unknown'}; "
        f"Treatments: {row['treatment_types']}; "
        f"Outcomes: {row['treatment_outcomes'] or 'N/A'}.\n"
        f"Clinical Note: {row['clinical_note']}"
    )
    SAMPLE_INPUTS[key] = value
import random

# Instead of manually picking one by key
# Just select a random sample from the generated dict
CLINICIAN_QUERY_ = random.choice(list(SAMPLE_INPUTS.values()))

print("Chosen Query:\n", CLINICIAN_QUERY_)



# ğŸ”¶ Complete Care Card Pipeline
# =================================


# 1) Semantic match (top-1 case)
# -----------------------------
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
    raise RuntimeError("No semantic matches found.")
top1 = result_df.iloc[0]
SEL_CASE_ID = top1["case_id"]
_sim = 1.0 - float(top1["distance"])

# -----------------------------
# 2) AI JSON generation
# -----------------------------
ai_sql = f"""
WITH src AS (
  SELECT '''
  You are an oncology assistant. Return ONLY valid JSON.
  Schema: {{
    "summary_bullets": ["..."],
    "provisional_category": "...", 
    "staging_summary": "...",
    "suggested_modalities": ["..."],
    "followup_plan": "...",
    "escalation_flag": true/false,
    "confidence_score": 0.85
  }}
  Clinical note: {CLINICIAN_QUERY_}
  ''' AS prompt
),
llm AS (
  SELECT
    JSON_EXTRACT_SCALAR(ml_generate_text_result, '$.candidates[0].content.parts[0].text') AS raw_response
  FROM ML.GENERATE_TEXT(
    MODEL `{PROJECT_ID}.{DATASET}.llm_text_gemini_v1`,
    TABLE src,
    STRUCT(512 AS max_output_tokens, 0.2 AS temperature)
  )
)
SELECT
  PARSE_JSON(TRIM(REGEXP_REPLACE(raw_response, r'```json|```', ''))) AS structured_guidance
FROM llm
"""
df = bq.query(ai_sql, location=REGION).to_dataframe()
guidance = df.iloc[0]["structured_guidance"]


# -----------------------------
# 3) Extract + normalize fields
# -----------------------------
def _to_list(x):
    """Ensure field is always a plain Python list of strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    return [str(x)]

def _to_str(x, default="â€”"):
    if x is None:
        return default
    return str(x)

category   = _to_str(guidance.get("provisional_category"), "other")
staging    = _to_str(guidance.get("staging_summary"), "Staging assessment pending")
bullets    = _to_list(guidance.get("summary_bullets")) or ["Clinical review required."]
modalities = _to_list(guidance.get("suggested_modalities")) or ["surgery","chemo","radiation"]
followup   = _to_str(guidance.get("followup_plan"), "Standard oncology follow-up.")
escalate   = bool(guidance.get("escalation_flag")) if guidance.get("escalation_flag") is not None else False
confidence = float(guidance.get("confidence_score") or 0.6)
# =========================
# Similarity label function
# =========================
def similarity_label(sim):
    if sim >= 0.85:
        return f"{sim*100:.1f}% (Excellent match)"
    elif sim >= 0.70:
        return f"{sim*100:.1f}% (Strong match)"
    elif sim >= 0.50:
        return f"{sim*100:.1f}% (Moderate match)"
    else:
        return f"{sim*100:.1f}% (Weak match)"
# -----------------------------
# 4) Render Care Card
# -----------------------------
def _pill(text):
    return (
        "<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        "border:1px solid #e6d98c;margin:4px 6px 4px 0;font-size:12px;"
        "background:#fff;white-space:normal;max-width:220px;word-wrap:break-word;'>"
        f"{text}</span>"
    )
modalities_html = "&nbsp;".join(_pill(m) for m in modalities) or _pill("â€”")
bullets_html    = "".join(f"<li>{b}</li>" for b in bullets) or "<li>â€”</li>"
priority_html   = ("<span style='color:#c62828;font-weight:700;'>ğŸš¨ Requires escalation</span>"
                   if escalate else "<span style='color:#2e7d32;'>Routine</span>")
conf_bar = f"""
<div style='background:#e5e7eb;border-radius:8px;overflow:hidden;height:10px;width:160px;margin-top:4px;'>
  <div style='height:100%;width:{confidence*100:.0f}%;background:#008080;'></div>
</div>
<div style='font-size:11px;color:#374151;'>{confidence*100:.0f}% confidence</div>
"""

card = f"""
<div style="font-family:Inter,Arial,sans-serif;background:#f8fafc;
            border:1px solid #e5e7eb;border-radius:12px;
            padding:18px 20px;max-width:980px;">
  <h2 style="margin:0 0 8px 0;color:#008080">ğŸ«€ Care Card: Tumor Board Ready</h2>
  <p><b>Case:</b> {SEL_CASE_ID}<br>
     <b>Category:</b> {category} &nbsp;&nbsp; <b>Stage:</b> {staging}<br>
     <b>Priority:</b> {priority_html} &nbsp;&nbsp;  <b>Similarity:</b> {similarity_label(_sim)}</p>

  <div style="margin:14px 0 6px 0;font-weight:600;">Suggested Modalities</div>
  <div>{modalities_html}</div>

  <div style="margin:16px 0 6px 0;font-weight:600;">Key Insights</div>
  <ul>{bullets_html}</ul>

  <div style="margin:16px 0 6px 0;font-weight:600;">Follow-up Plan</div>
<div style="background:#fff8dc;border:1px solid #e6d98c;
            padding:8px;border-radius:8px;max-width:75%;line-height:1.2;">
  {followup}
</div>

  <div style="margin-top:14px;color:#6b7280;font-size:12px;">
    ğŸ¤– AI-Generated â€¢ Research demo
    {conf_bar}
  </div>
</div>

<div style="height:12px;"></div>

<div style="font-family:Inter,Arial,sans-serif;max-width:980px;">
  <h3 style="margin:0 0 6px 0;">ğŸ”� Top Semantic Match (for transparency)</h3>
  <div style="font-size:14px;line-height:1.6;background:#F7F6EC;
              border:1px solid #e5e7eb;border-radius:10px;padding:14px;">
    <b>Diagnosis:</b> {top1.get('diag__primary_diagnosis','â€”')} &nbsp; | &nbsp;
    <b>Stage:</b> {top1.get('diag__ajcc_pathologic_stage','â€”')} &nbsp; | &nbsp;
    <b>Gender:</b> {top1.get('gender','â€”')} &nbsp; | &nbsp;
    <b>Outcome:</b> {top1.get('treatment_outcomes','â€”')}<br>
    <b>Treatments:</b> {top1.get('treatment_types','â€”')}<br>
    <b>Snippet:</b> {top1.get('clinical_snippet','â€”')}
  </div>
</div>
"""

print("ğŸ”¥ Final Demo ğŸ”¥")
print("ğŸ”¸ Clinical Query:")
print("â”Œ" + "â”€" * 68 + "â”�")
for line in textwrap.wrap(CLINICIAN_QUERY_, width=66):
    print(f"â”‚ {line.ljust(66)} â”‚")
print("â””" + "â”€" * 68 + "â”˜")
display(HTML(card))



# â�±ï¸� Processing Time Profiling for MediBridge AI

timings = {}

# --- Vector Search ---
t0 = time.perf_counter()
_ = bq.query(sql, location=REGION).to_dataframe()
timings["Vector Search"] = time.perf_counter() - t0

# --- AI Generation ---
t1 = time.perf_counter()
_ = bq.query(ai_sql, location=REGION).to_dataframe()
timings["AI Generation"] = time.perf_counter() - t1

# --- Data Processing (prep + rendering) ---
# Ensure Data Processing is never exactly 0 (for visibility)
timings["Data Processing"] = max(timings.get("Data Processing", 0.0), 0.01)

# Extract labels & values
labels, sizes = list(timings.keys()), list(timings.values())

# Define color palette 
colors = ["#1FC9E0", "#16a085", "#f39c12"]  # Vector, AI, Data

# hide if <3%
def autopct_fmt(pct):
    return ('%1.1f%%' % pct) if pct > 3 else '' 
    
# ---- Donut Chart ----
plt.figure(figsize=(5,5))
wedges, texts, autotexts = plt.pie(
    sizes, labels=labels, autopct=autopct_fmt,
    startangle=90, pctdistance=0.85, colors=colors,
    textprops={'color':"black"}
)
# Donut hole
plt.gca().add_artist(plt.Circle((0,0), 0.70, fc='white'))
plt.title("Processing Time Distribution for MediBridge AI Workflow", fontweight="bold")
plt.show()

# ---- Bar Chart ----
plt.figure(figsize=(6,4))
bars = plt.bar(labels, sizes, color=colors)
plt.title("Absolute Processing Time for MediBridge AI Workflow", fontweight="bold")
plt.ylabel("Seconds")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}s", 
             ha='center', va='bottom', fontsize=10, fontweight="bold")
plt.legend(wedges, labels, title="Steps", loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


