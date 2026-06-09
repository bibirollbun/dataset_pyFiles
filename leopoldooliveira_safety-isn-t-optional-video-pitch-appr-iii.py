from IPython.display import YouTubeVideo
YouTubeVideo('X6ny5kjq4VM', width=800, height=500)



#------------------------------MANDATORY VARIABLES-----------------------------#
# GOOGLE CLOUD PROJECT AND DATASET
GOOGLE_CLOUD_PROJECT = "kaggle-genai-469221"
GOOGLE_CLOUD_DATASET = "safety_analytics"

# GOOGLE CLOUD STORAGE BUCKET
CLOUD_STORAGE_BUCKET = "bigquery_kaggle"

# VERTEX AI CONNECTION
VERTEX_AI_CONNECTION = 'projects/kaggle-genai-469221/locations/us/connections/safetyanalytics_google'

#------------------------------OPTIONAL VARIABLES------------------------------#
# MODEL CONFIGURATION
GOOGLE_CLOUD_MODEL = "safety_classifier_model"
GOOGLE_MODEL_TEMPERATURE = 0.2


# === Standard library ===
import os
import io
import time
from functools import reduce
import requests   # For making HTTP requests

# === Data manipulation & analysis ===
import pandas as pd
import numpy as np

# === Visualization ===
import matplotlib.pyplot as plt
import seaborn as sns

# === Google Cloud / Colab integration ===
from google.colab import auth
from google.cloud import bigquery, storage  # BigQuery + Cloud Storage clients

# === Kaggle integration ===
import kagglehub   # Kaggle utility functions (datasets, outputs, etc.)



# -------------------------------------------------------------------
# Download OSH indicators directly from ILOSTAT
# -------------------------------------------------------------------

BASE = "https://rplumber.ilo.org/data/indicator/"
COMMON = "&sex=SEX_T&classif1=MIG_STATUS_TOTAL&latestyear=TRUE&type=label&format=.csv"

INDICATORS = {
    "non_fatal_occupation_per_100_000": "SDG_N881_SEX_MIG_RT_A",
    "fatal_occupation_per_100_000":     "SDG_F881_SEX_MIG_RT_A",
    "inspectors_per_100_000":           "LAI_INDE_NOC_RT_A",  # usually per 10,000 employees
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://ilostat.ilo.org/"
}

def fetch_indicator(code: str, max_retries: int = 3, backoff: float = 1.5) -> pd.DataFrame:
    """
    Fetch a single indicator as CSV and return a tidy DataFrame with columns:
    - 'country'
    - '<metric column>' (renamed later by caller)
    Implements simple retries to handle transient 403/5xx responses.
    """
    url = f"{BASE}?id={code}{COMMON}"
    last_err = None

    with requests.Session() as s:
        s.headers.update(HEADERS)
        for attempt in range(1, max_retries + 1):
            try:
                resp = s.get(url, timeout=60)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                # Standardize column names we need
                if "ref_area.label" not in df.columns or "obs_value" not in df.columns:
                    raise ValueError(f"Unexpected schema for indicator {code}. Columns: {list(df.columns)}")
                return df.rename(columns={"ref_area.label": "country"})[["country", "obs_value"]]
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)
                else:
                    raise last_err

dfs = []
for colname, code in INDICATORS.items():
    df_i = fetch_indicator(code)
    df_i = df_i.rename(columns={"obs_value": colname})
    dfs.append(df_i)

# Merge all indicators by country
df_safety = reduce(lambda left, right: pd.merge(left, right, on="country", how="outer"), dfs)

# Convert to numeric and clean
for col in ["non_fatal_occupation_per_100_000",
            "fatal_occupation_per_100_000",
            "inspectors_per_100_000"]:
    df_safety[col] = pd.to_numeric(df_safety[col], errors="coerce")

# Normalize inspectors from per 10,000 to per 100,000 to match the other metrics
if "inspectors_per_100_000" in df_safety.columns:
    df_safety["inspectors_per_100_000"] = df_safety["inspectors_per_100_000"] * 10

# Optional: basic sanity filters (keep rows with at least one metric present)
df_safety = df_safety.dropna(how="all", subset=[
    "non_fatal_occupation_per_100_000",
    "fatal_occupation_per_100_000",
    "inspectors_per_100_000"
])

# Preview
print("Preview:")
display(df_safety.head())

# Optional: save to CSV (uncomment if you want a local/Drive copy)
# df_safety.to_csv("/content/df_safety.csv", index=False)

"""
This DataFrame contains occupational safety and health (OSH) indicators aligned to SDG 8.8.1:
- non_fatal_occupation_per_100_000: Non-fatal occupational injuries per 100K workers
- fatal_occupation_per_100_000: Fatal occupational injuries per 100K workers
- inspectors_per_100_000: Labor inspectors per 100K employees (normalized here to 100K)
"""



df_safety.info()


# --- Data Analysis & Visualization ---

# Columns of interest (define at the start for clarity)
COL_FATAL = 'fatal_occupation_per_100_000'
COL_INSPECTORS = 'inspectors_per_100_000'
COL_COUNTRY = 'country'

# 1. Data Preparation

# Filter rows with non-null values in relevant columns
df_filtered = df_safety.dropna(subset=[COL_FATAL, COL_INSPECTORS]).copy()

# 2. Select Top 30 Countries with Most Fatalities

# Sort descending by fatalities and select top 30
df_top_30_fatalities = df_filtered.sort_values(by=COL_FATAL, ascending=False).head(30).copy()

# 3. Multi-Axis Chart

# Set figure size
fig, ax1 = plt.subplots(figsize=(18, 10))

# --- Primary Axis (ax1) ---
# Bar plot for fatalities
sns.barplot(
    x=COL_COUNTRY,
    y=COL_FATAL,
    data=df_top_30_fatalities,
    ax=ax1,
    color='skyblue',
    label='Fatalities per 100K'
)

# Add value labels on bars
for p in ax1.patches:
    ax1.text(
        p.get_x() + p.get_width() / 2.,
        p.get_height(),
        f'{p.get_height():.2f}',
        ha='center', va='bottom',
        fontsize=10
    )

# --- Secondary Axis (ax2) ---
# Line plot for inspectors
ax2 = ax1.twinx()
ax2.plot(
    df_top_30_fatalities[COL_COUNTRY],
    df_top_30_fatalities[COL_INSPECTORS],
    marker='o',
    linestyle='-',
    color='red',
    label='Inspectors per 10K'
)

# Add value labels on line points
for i, insp_value in enumerate(df_top_30_fatalities[COL_INSPECTORS]):
    ax2.text(
        i,
        insp_value,
        f'{insp_value:.2f}',
        ha='center', va='bottom',
        color='red', fontsize=10
    )

# 4. Chart Configuration

# Axis labels and title
ax1.set_xlabel('Country', fontsize=14, fontweight='bold')
ax1.set_ylabel('Fatalities per 100K Workers', fontsize=12)
ax2.set_ylabel('Inspectors per 10K Employees', color='red', fontsize=12)
plt.title('Top 30 Countries: Fatalities vs. Inspectors', fontsize=16, fontweight='bold')

# Rotate x-axis labels
ax1.tick_params(axis='x', rotation=90)

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Optimize layout
plt.tight_layout()

# Show chart
plt.show()


# --- Data Analysis & Visualization (Lowest Fatality Rates) ---

# Define columns of interest (using variables improves maintainability)
COL_FATAL = 'fatal_occupation_per_100_000'
COL_INSPECTORS = 'inspectors_per_100_000'
COL_COUNTRY = 'country'

# 1. Data Preparation

# Filter rows with non-null values in key columns
df_filtered = df_safety.dropna(subset=[COL_FATAL, COL_INSPECTORS]).copy()

# 2. Select 30 Countries with the Lowest Fatality Rates

# Sort ascending and take the first 30
df_bottom_30_fatalities = df_filtered.sort_values(by=COL_FATAL, ascending=True).head(30).copy()

# 3. Multi-Axis Chart

# Set figure size
fig, ax1 = plt.subplots(figsize=(18, 10))

# --- Primary Axis (ax1) ---
# Bar plot for fatalities
sns.barplot(
    x=COL_COUNTRY,
    y=COL_FATAL,
    data=df_bottom_30_fatalities,
    ax=ax1,
    color='skyblue',
    label='Fatalities per 100K'
)

# Add value labels to bars
for p in ax1.patches:
    ax1.text(
        p.get_x() + p.get_width() / 2.,
        p.get_height(),
        f'{p.get_height():.2f}',
        ha='center', va='bottom',
        fontsize=9
    )

# --- Secondary Axis (ax2) ---
# Line plot for inspectors
ax2 = ax1.twinx()
ax2.plot(
    df_bottom_30_fatalities[COL_COUNTRY],
    df_bottom_30_fatalities[COL_INSPECTORS],
    marker='o',
    linestyle='-',
    color='red',
    label='Inspectors per 10K'
)

# Add value labels to line points
for i, insp_value in enumerate(df_bottom_30_fatalities[COL_INSPECTORS]):
    ax2.text(
        i,
        insp_value,
        f'{insp_value:.2f}',
        ha='center', va='bottom',
        color='red', fontsize=9
    )

# 4. Chart Configuration

# Axis labels and title
ax1.set_xlabel('Country', fontsize=14, fontweight='bold')
ax1.set_ylabel('Fatalities per 100K Workers', fontsize=12)
ax2.set_ylabel('Inspectors per 10K Employees', color='red', fontsize=12)
plt.title('Top 30 Countries with Lowest Fatality Rates vs. Inspectors', fontsize=16, fontweight='bold')

# Rotate x-axis labels
ax1.tick_params(axis='x', rotation=90)

# Remove secondary axis grid
ax2.grid(False)

# Combine and show legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Adjust layout
plt.tight_layout()

# Show chart
plt.show()



# === Download Kaggle public dataset ===

# Dataset path identifier from Kaggle
PATH_DATABASE = "ihmstefanini/industrial-safety-and-health-analytics-database"

print("Downloading Kaggle dataset...")
# Download the dataset locally using kagglehub
path = kagglehub.dataset_download(PATH_DATABASE)

# Confirm where the dataset has been stored
print(f"Dataset successfully downloaded to: {path}")

# === Load CSV into a DataFrame ===

# Relative path of the target CSV inside the downloaded folder
NAME_CSV = "/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv"

# Load the CSV into a pandas DataFrame
df_train = pd.read_csv(path + NAME_CSV)



df_train.head()


# basic pre processing
df_train['source'] = "Text" # all registers from text

df_train = df_train.rename(columns={'Unnamed: 0': 'index_'})

df_train.head()


# Authenticate user for Google Cloud
auth.authenticate_user()

# Initialize BigQuery client
client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)



# --- Upload Data to BigQuery ---

# Full table reference: project.dataset.table
table_id = f"{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.safety_accidents"

# Load job configuration
# WRITE_TRUNCATE = overwrite table each run (keeps table in sync with DataFrame)
job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
)

# Submit load job (upload pandas DataFrame to BigQuery)
job = client.load_table_from_dataframe(
    df_train, table_id, job_config=job_config
)

# Block until job is complete
job.result()

# Confirm completion and number of rows inserted
print(f"Data successfully loaded: {job.output_rows} rows into {table_id}")



# --- Query Execution with Generative AI in BigQuery ---

# 1. Define SQL query
# - Creates or replaces a target table
# - Uses AI.GENERATE_TABLE on column 'Description'
sql_query = f"""
CREATE OR REPLACE TABLE `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.tb_bronze_safety_accidents` AS
WITH gen_table AS (
  SELECT *
  FROM `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.safety_accidents`
)
SELECT *
FROM AI.GENERATE_TABLE(
  MODEL `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.{GOOGLE_CLOUD_MODEL}`,
  (
    -- Subquery must output a column named 'prompt'
    SELECT CONCAT(
      'Classify the incident into:', '\\n',
      '- incident_type (choose one): ["crushing_pinch","slip_fall_same_level","fall_from_height","struck_by_caught_between","chemical_thermal_exposure","projection_flying_object","mechanical_breakdown_failure","electrocution_electrical_arc","ergonomic_overexertion","animal_insect_bite","vehicle_mobile_equipment","tool_hand_tool_injury","other"]', '\\n',
      '- root_cause (choose one): ["human_error","procedural_failure","equipment_failure","environmental_hazard","inadequate_ppe","lack_of_communication","unstable_material_structure","other"]', '\\n',
      '- is_ppe_used (BOOL): true/false/null (use "null" if not mentioned)', '\\n',
      '- is_ppe_ok (BOOL): true/false/null (use "null" if not mentioned)', '\\n',
      'Also provide a short "summary". Respond only with valid JSON.', '\\n',
      'Text: ', Description, '\\n',
      'Record ID: ', index_
    ) AS prompt
    FROM gen_table
  ),
  STRUCT(
    "is_ppe_ok STRING, is_ppe_used STRING, incident_type STRING, root_cause STRING, accident_category STRING, summary STRING, id STRING" AS output_schema,
    {GOOGLE_MODEL_TEMPERATURE} AS temperature
  )
);
"""

# 2. Execute query
print("Submitting BigQuery job...")
job = client.query(sql_query)

# 3. Wait for completion
job.result()

# 4. Confirmation
print("Job completed successfully.")
print("Table tb_bronze_safety_accidents created or replaced.")


# --- Data Join & Enrichment in BigQuery ---

# 1. SQL query definition
# - Joins raw accidents table (a) with AI-enriched classifications (ia)
# - Creates a clean/enriched table for downstream use
sql_query = f"""
CREATE OR REPLACE TABLE `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.tb_silver_incidents_enriched` AS
SELECT
    -- Original fields
    a.index_,
    a.Data,
    a.Countries,
    a.Local,
    a.`Industry Sector` AS industry_sector,   -- renamed to snake_case
    a.Genre,
    a.`Employee or Third Party` AS employee_thirdparty,   -- renamed for clarity
    a.Description,
    a.source,

    -- AI-generated fields
    ia.accident_category,
    ia.incident_type,
    ia.is_ppe_ok,
    ia.is_ppe_used,
    ia.summary,
    ia.full_response,
    ia.prompt

FROM `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.safety_accidents` AS a
LEFT JOIN `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.tb_bronze_safety_accidents` AS ia
ON CAST(a.index_ AS STRING) = ia.id;   -- join key normalized as string
"""

# 2. Execute query
print("Submitting BigQuery job to build enriched dataset...")
job = client.query(sql_query)

# 3. Wait until it finishes
job.result()

# 4. Confirmation
print("Job completed successfully.")
print("Table tb_silver_incidents_enriched created/replaced with joined data.")




# --- Visualization & Validation of Enriched Data ---

# 1. SQL query: fetch a small sample for inspection
sql_query = f"""
SELECT *
FROM `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.tb_silver_incidents_enriched`
LIMIT 5;
"""

# 2. Execute query and return as pandas DataFrame
print("Fetching sample data from enriched table...")
df = client.query(sql_query).to_dataframe()

# 3. Display sample for quick validation
print("Data fetched successfully. Preview of 5 rows:")
display(df)


# --- Download latest dataset version from Kaggle ---

# Fetch dataset and store local path
path = kagglehub.dataset_download("leopoldooliveira/workspace-accidents-videos")

# Confirm download location
print("Dataset files downloaded to:", path)



# --- Upload dataset files to Cloud Storage ---

# Initialize storage client and target bucket
storage_client = storage.Client()
bucket = storage_client.bucket(CLOUD_STORAGE_BUCKET)

def upload_files(local_dir: str, bucket):
    """
    Uploads all files from a local directory to the given Cloud Storage bucket.
    Existing files with the same name will be overwritten.
    """
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_file = os.path.join(root, filename)

            # Relative path inside the bucket (preserves folder structure)
            blob_path = os.path.relpath(local_file, local_dir)
            blob = bucket.blob(blob_path)

            # Upload file ("create or replace")
            blob.upload_from_filename(local_file)
            print(f"Uploaded: {local_file} -> gs://{bucket.name}/{blob_path}")

# Execute upload
upload_files(path, bucket)



# --- Generate list of Cloud Storage URIs ---

# Container for URIs
uris = []

def create_list_uris(bucket_name: str):
    """List all object URIs from a Cloud Storage bucket and store them in 'uris'."""
    storage_client = storage.Client()

    # Fetch all objects (blobs) from the bucket
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        uri = f"gs://{bucket_name}/{blob.name}"
        uris.append(uri)

# Populate the list
create_list_uris(CLOUD_STORAGE_BUCKET)

# Print URIs
print("URIs found in bucket:")
print(uris)



# --- Create/Replace External Table in BigQuery ---

# 1. Define SQL query
# - Creates an external table linked to GCS video files
# - Uses uris[] list of Cloud Storage paths
sql_query = f"""
CREATE OR REPLACE EXTERNAL TABLE `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.videos`
WITH CONNECTION `{VERTEX_AI_CONNECTION}`
OPTIONS(
  object_metadata = 'SIMPLE',
  uris = {uris}
  -- max_staleness and metadata_cache_mode can be enabled if needed
);
"""

# 2. Execute query
print("Submitting BigQuery job to create external table...")
job = client.query(sql_query)

# 3. Wait until finished
job.result()

# 4. Confirmation
print("Job completed successfully.")
print("External table 'videos' created or replaced.")



# --- Query External Video Table ---

# 1. SQL query: select all records from the external table
sql_query = f"SELECT * FROM `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.videos`"

# 2. Execute query and load into pandas DataFrame
print("Fetching data from external video table...")
df_external_table = client.query(sql_query).to_dataframe()

# 3. Display sample for validation
print("Data fetched successfully. Preview of first rows:")
display(df_external_table.head())



# --- Video Processing with AI in BigQuery ---

# SQL query:
# - Creates or replaces tb_bronze_safety_videos
# - Uses AI.GENERATE_TABLE to classify accident videos stored in GCS
sql_query_videos = rf"""
CREATE OR REPLACE TABLE `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.tb_bronze_safety_videos` AS
WITH vids AS (
  SELECT
    uri AS gcs_uri,
    ref,
    GENERATE_UUID() AS video_id   -- Unique identifier for joins
  FROM `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.videos`
)
SELECT *
FROM AI.GENERATE_TABLE(
  MODEL `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.{GOOGLE_CLOUD_MODEL}`,
  (
    SELECT
      -- Prompt must be a STRUCT with fields 'text' and 'media'
      STRUCT(
        CONCAT(
          'Classify the incident into:', '\n',
          '- incident_type (choose one): ["crushing_pinch","slip_fall_same_level","fall_from_height","struck_by_caught_between","chemical_thermal_exposure","projection_flying_object","mechanical_breakdown_failure","electrocution_electrical_arc","ergonomic_overexertion","animal_insect_bite","vehicle_mobile_equipment","tool_hand_tool_injury","other"]', '\n',
          '- root_cause (choose one): ["human_error","procedural_failure","equipment_failure","environmental_hazard","inadequate_ppe","lack_of_communication","unstable_material_structure","other"]', '\n',
          '- industry_sector (choose one): ["mining","metals","other","logistics"]', '\n',
          '- is_ppe_used (BOOL): true/false/null (use "null" if not explicit)', '\n',
          '- is_ppe_ok (BOOL): true/false/null (use "null" if not explicit)', '\n',
          '- local = accident location (e.g., logistics, manufacturing, assembly line)', '\n',
          '- genre = Male/Female (use "null" if not explicit)', '\n',
          'Provide a short summary. Respond ONLY with valid JSON.', '\n',
          'Record ID: ', video_id
        ) AS text,  -- Text field of the prompt
        OBJ.GET_ACCESS_URL(OBJ.FETCH_METADATA(ref), 'r') AS media  -- Media field of the prompt
      ) AS prompt,
      video_id,
      gcs_uri
    FROM vids
  ),
  STRUCT(
    -- Define expected JSON schema from model output
    "is_ppe_ok STRING, is_ppe_used STRING, incident_type STRING, root_cause STRING, accident_category STRING, summary STRING, id STRING, local STRING, industry_sector STRING, genre STRING" AS output_schema,
    2048 AS max_output_tokens,
    {GOOGLE_MODEL_TEMPERATURE} AS temperature
  )
);
"""

# Execute BigQuery job
print("Submitting job to build tb_bronze_safety_videos...")
job = client.query(sql_query_videos)

# Wait until finished
job.result()

# Confirmation
print("âœ… Table tb_bronze_safety_videos created or replaced successfully.")





# --- Video Processing for Silver Table in BigQuery ---

# SQL query:
# - Creates tb_silver_safety_videos as a standardized/enriched table
# - Aligns schema with text-based incidents for easier unification
sql_query_videos = f"""
CREATE OR REPLACE TABLE `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.tb_silver_safety_videos` AS
SELECT
    id AS index_,                -- Rename 'id' to 'index_' for consistency
    NULL AS data,                -- Not applicable, set to NULL
    NULL AS countries,           -- Not applicable, set to NULL
    local,                       -- Incident location
    industry_sector,             -- Industry sector (fixed typo from 'industry_setor')
    genre,                       -- Person's gender
    NULL AS employee_thirdparty, -- Not applicable, set to NULL
    summary AS description,      -- Use AI-generated summary as description
    "Video" AS source,           -- Mark data source as 'Video'
    accident_category,           -- AI-predicted accident category
    incident_type,               -- AI-predicted incident type
    is_ppe_ok,                   -- AI-predicted PPE condition
    is_ppe_used,                 -- AI-predicted PPE usage
    summary,                     -- AI-generated summary
    full_response,               -- Full AI model output
    prompt.media AS prompt       -- Original media URL used in AI prompt
FROM `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.tb_bronze_safety_videos`
"""

# Execute BigQuery job
print("Submitting job to build tb_silver_safety_videos...")
job = client.query(sql_query_videos)

# Wait until finished
job.result()

# Confirmation
print("âœ… Table tb_silver_safety_videos created or replaced successfully.")



# --- Creation of "Gold Layer" Consolidated View in BigQuery ---

# SQL query:
# - Creates view vw_gold_accidents
# - Unifies silver video and silver text accident tables
# - Ensures aligned schema for downstream analysis
sql_query_videos = f"""
CREATE OR REPLACE VIEW `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.vw_gold_accidents` AS

-- Video-based accidents
SELECT
    index_,
    CAST(data AS STRING) AS date_,             -- Normalize date as string
    CAST(countries AS STRING) AS countries,    -- Normalize countries as string
    local,
    industry_sector,                           -- Fixed typo (was industry_setor)
    genre,
    CAST(employee_thirdparty AS STRING) AS employee_thirdparty,
    description,
    source,
    accident_category,
    incident_type,
    is_ppe_ok,
    is_ppe_used,
    summary,
    full_response
FROM `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.tb_silver_safety_videos`

UNION ALL

-- Text-based accidents
SELECT
    CAST(index_ AS STRING) AS index_,
    CAST(Data AS STRING) AS date_,             -- Normalize column name for consistency
    Countries AS countries,
    Local AS local,
    industry_sector,
    genre,
    employee_thirdparty,
    description,
    source,
    accident_category,
    incident_type,
    is_ppe_ok,
    is_ppe_used,
    summary,
    full_response
FROM `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.tb_silver_incidents_enriched`;
"""

# Execute BigQuery job
print("Submitting job to build vw_gold_accidents view...")
job = client.query(sql_query_videos)

# Wait until finished
job.result()

# Confirmation
print("âœ… View vw_gold_accidents created or replaced successfully.")



# --- Final Validation of Consolidated View ---

# 1) BigQuery client (uses the active project)
client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)

# 2) SQL: fetch all rows from the consolidated view
sql_query = f"SELECT * FROM `{GOOGLE_CLOUD_PROJECT}.{GOOGLE_CLOUD_DATASET}.vw_gold_accidents`"

# 3) Execute and load into a DataFrame
print("Fetching data from vw_gold_accidents...")
df_final_view = client.query(sql_query).to_dataframe()

# 4) Preview a few rows for a quick structure check
print("Data fetched successfully. Preview:")
display(df_final_view.head())




