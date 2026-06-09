from IPython.display import Image
display(Image("/kaggle/input/qualitynexus/QualityNexus_Overview.jpg", width=700))


from IPython.display import Image
display(Image("/kaggle/input/qualitynexus/QualityNexus_Architecture.png", width=700))


from IPython.display import Image
display(Image("/kaggle/input/qualitynexus/QualityNexus_BigQueryAssistant.png", width=700))


!pip install pdfminer.six         # For pdf processing
!pip install yt-dlp webvtt-py     # For youtube download and transcript
       
import sys, subprocess, importlib
import os, time, json, warnings, math
import pandas as pd,re
import plotly.graph_objects as go
import plotly.io as pio           # Kaggle-safe Plotly rendering
import io, pathlib
import yt_dlp        
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import html as _html
import json, textwrap             # For BigQuery Assistant

from google.cloud import bigquery
from google.cloud import storage
from google.oauth2 import service_account
from kaggle_secrets import UserSecretsClient
from IPython.display import display, HTML, Markdown, Image, IFrame, FileLink
from pdfminer.high_level import extract_text
from pathlib import Path


# ===================== Define environment parameters =====================
# Set project and dataset parameter
user_secrets = UserSecretsClient()
project = user_secrets.get_secret("GCP_PROJECT_ID")
gcp_key_json = user_secrets.get_secret("GCP_SA_KEY")
location = 'US'
DATASET = "oss_quality"
SKIP_GCS = True   # Set True if data already available in the Google Cloud Storage or BigQuery, set False to load data and run once

# Set Google Cloud Storage Parameter
DATASET   = "oss_quality"
BUCKET = "kagglempqa"             #bucket in Google Cloud
PREFIX = "document/k8s"           #Dcoument folder inside the bucket
IMAGE_PREFIX = "kpi/images/k8s"   #Image folder inside the bucket
MAX_CHARS = 200_000               # safety cap per doc

# Set plot image folder
PLOT_DIR = pathlib.Path("./plots")
PLOT_DIR.mkdir(exist_ok=True)

# Set BigQuery Connection named 'genai' , and default model enpoint.
BQ_GENAI_CONNECTION  = f"projects/{project}/locations/{location.lower()}/connections/genai"
GEN_TEXT_ENDPOINT = "gemini-2.0-flash"
GEN_AUDIO_ENDPOINT = "gemini-1.5-pro" 
GEN_EMBEDDING_ENDPOINT  = "text-embedding-004"

# Silence warning
warnings.filterwarnings("ignore", message="BigQuery Storage module not found")

# Build a Credentials object directly from the JSON
creds = service_account.Credentials.from_service_account_info(json.loads(gcp_key_json))

# Write the key to a temporary file in the notebook's environment
key_file_path = 'gcp_key.json'
try:
    with open(key_file_path, 'w') as f:
        f.write(gcp_key_json)
    
    # Authenticate the gcloud tool using the key file
    !gcloud auth activate-service-account --key-file={key_file_path} 
    
    # Configure the gcloud tool to use your project
    !gcloud config set project {project} 
    
finally:
    # Securely delete the key file immediately after use
    if os.path.exists(key_file_path):
        os.remove(key_file_path)

# Enable the Vertex AI and BigQuery Connection APIs. Run only once Or Enable using the Cloud Interface.
!gcloud services enable aiplatform.googleapis.com bigqueryconnection.googleapis.com 

# This command creates the connection resource. Run only once
#!bq mk --connection --location={location} --connection_type=CLOUD_RESOURCE genai

# Initiate BigQuery client.
client = bigquery.Client(project=project, location=location,credentials=creds)
client

# Ensure dataset exists
ds = bigquery.Dataset(f"{project}.{DATASET}")
ds.location = location
client.create_dataset(ds, exists_ok=True)


# ===================== Define Helper Functions =====================
# Helper to execute query by the bigquery client
def _bq(sql, params=None):
    cfg = bigquery.QueryJobConfig(query_parameters=params or [])
    job = client.query(sql, job_config=cfg, location=location)
    # Simple wait loop
    while True:
        j = client.get_job(job.job_id, location=location)
        if j.done():
            if j.error_result:
                raise RuntimeError(j.error_result)
            return j
        time.sleep(1)
if "bq" not in globals():
    bq = _bq  # use shim

# Helper to execute query return the dataframe

def qdf(sql: str):
    job = client.query(sql, location=location)
    return job.result().to_dataframe(bqstorage_client=bqstorage_client)

# Helper to show the table with left-aligned approach
def show_left(df: pd.DataFrame, pre_wrap: bool = True, only_text_cols: bool = False):
    """Display a DataFrame with left-aligned cells & headers.
       Set only_text_cols=True to left-align just object/string columns."""
    subset = None
    if only_text_cols:
        subset = list(df.select_dtypes(include=["object", "string"]).columns)

    styler = df.style.hide(axis="index")

    props = {"text-align": "left"}
    if pre_wrap:
        props["white-space"] = "pre-wrap"

    if subset:
        styler = styler.set_properties(subset=subset, **props)
    else:
        styler = styler.set_properties(**props)

    # Make sure headers are left-aligned too
    styler = styler.set_table_styles([
        {"selector": "th", "props": [("text-align", "left")]},
        {"selector": "td", "props": [("text-align", "left")]}
    ])
    display(styler)

def _table_exists(fqn: str) -> bool:
    proj, ds, tbl = fqn.split(".")
    sql = f"""
      SELECT 1
      FROM `{proj}.{ds}.INFORMATION_SCHEMA.TABLES`
      WHERE table_name = @t
      LIMIT 1
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("t","STRING", tbl)])
    return client.query(sql, job_config=cfg, location=location).result().to_dataframe().shape[0] == 1



#  ===================== Dump githubarchive.month ===================== 
# (2019-01..2019-12) â†’ BigQuery table (type, payload, created_at) 
# !!!! RUN ONLY ONCE !!!

# ---- Config ----
TABLE       = "gh_month_2019_k8s"
target_repo = "kubernetes/kubernetes"
start_month = "2019-01"          # inclusive YYYY-MM
end_month   = "2019-12"          # inclusive YYYY-MM
# ---------------

# Param values used by wildcard sharded tables
start_suffix = start_month.replace("-", "")   # "201901"
end_suffix   = end_month.replace("-", "")     # "201912"

# Does the destination table already exist?
table_id = f"{project}.{DATASET}.{TABLE}"

# If SKIP_GCS is set OR table already exists, skip the CREATE â€¦ AS SELECT
try:
    SKIP_GCS  # just to see if it's defined
except NameError:
    SKIP_GCS = False  # default

if SKIP_GCS or _table_exists(table_id):
    reasons = []
    if SKIP_GCS:
        reasons.append("SKIP_GCS=True")
    if _table_exists(table_id):
        reasons.append("table already exists")
    print(f"â�­ï¸�  Skipping BigQuery load ({' and '.join(reasons)}) â†’ {table_id}")

    # Optional: quick row count if table exists
    if _table_exists(table_id):
        row_count_df = client.query(
            f"SELECT COUNT(*) AS total_rows FROM `{table_id}`",
            location=location
        ).result().to_dataframe()
        display(row_count_df)
else:
    create_sql = """
    CREATE OR REPLACE TABLE `{project}.{dataset}.{table}` AS
    SELECT
      type,
      payload,               -- keep as RECORD to preserve structure
      TIMESTAMP(created_at) AS created_at
    FROM `githubarchive.month.*`
    WHERE _TABLE_SUFFIX BETWEEN @start_suffix AND @end_suffix
      AND LOWER(repo.name) = LOWER(@target_repo);
    """.format(project=project, dataset=DATASET, table=TABLE)

    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_suffix", "STRING", start_suffix),
            bigquery.ScalarQueryParameter("end_suffix",   "STRING", end_suffix),
            bigquery.ScalarQueryParameter("target_repo",  "STRING", target_repo),
        ]
    )

    job = client.query(create_sql, job_config=job_cfg, location=location)

    # Poll until done (avoids some notebook async quirks)
    while True:
        j = client.get_job(job.job_id, location=location)
        if j.done():
            if j.error_result:
                raise RuntimeError(j.error_result)
            break
        time.sleep(1)

    # Optional: row count
    row_count_df = client.query(
        f"SELECT COUNT(*) AS total_rows FROM `{table_id}`",
        location=location
    ).result().to_dataframe()
    display(row_count_df)


#  ===================== Dump Kubernetes-related Stack Overflow ===================== 
#  Questions in 2019 â†’ BigQuery table 
# !!!! RUN ONLY ONCE !!! 

# --- Config ---
TABLE        = "so_2019_kube_questions"         # destination table
start_date   = "2019-01-01"                      # inclusive
end_date     = "2019-12-31"                      # inclusive
kube_tag_regex = r'^(kube|k8s|helm).*|^(minikube|kubectl|kubelet)$'
# -------------

# Does the destination table already exist?
table_id = f"{project}.{DATASET}.{TABLE}"


# If SKIP_GCS is set OR table already exists, skip the CREATE â€¦ AS SELECT
try:
    SKIP_GCS  # just to see if it's defined
except NameError:
    SKIP_GCS = False  # default

if SKIP_GCS or _table_exists(table_id):
    reasons = []
    if SKIP_GCS:
        reasons.append("SKIP_GCS=True")
    if _table_exists(table_id):
        reasons.append("table already exists")
    print(f"â�­ï¸�  Skipping BigQuery load ({' and '.join(reasons)}) â†’ {table_id}")

    # Optional: quick row count if table exists
    if _table_exists(table_id):
        row_count_df = client.query(
            f"SELECT COUNT(*) AS total_rows FROM `{table_id}`",
            location=location
        ).result().to_dataframe()
        display(row_count_df)
else:
    # Materialize filtered questions into a table
    create_sql = f"""
    CREATE OR REPLACE TABLE `{project}.{DATASET}.{TABLE}` AS
    WITH filtered AS (
      SELECT
        id,
        creation_date,
        tags,
        title,
        body,
        score,
        view_count,
        accepted_answer_id,
        answer_count,
        comment_count,
        owner_user_id
      FROM `bigquery-public-data.stackoverflow.posts_questions`
      WHERE DATE(creation_date) BETWEEN @start_date AND @end_date
    ),
    expanded AS (
      SELECT f.*, t
      FROM filtered f,
      UNNEST(
        IFNULL(
          CASE
            WHEN REGEXP_CONTAINS(LOWER(f.tags), r'<')
              THEN REGEXP_EXTRACT_ALL(LOWER(f.tags), r'<([^>]+)>')
            ELSE SPLIT(LOWER(f.tags), '|')
          END,
          []
        )
      ) AS t
    )
    SELECT DISTINCT
      id,
      creation_date,
      tags,
      title,
      body,
      score,
      view_count,
      accepted_answer_id,
      answer_count,
      comment_count,
      owner_user_id
    FROM expanded
    WHERE REGEXP_CONTAINS(t, @kube_tag_regex);
    """
    
    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date",     "DATE",   start_date),
            bigquery.ScalarQueryParameter("end_date",       "DATE",   end_date),
            bigquery.ScalarQueryParameter("kube_tag_regex", "STRING", kube_tag_regex),
        ]
    )
    
    job = client.query(create_sql, job_config=job_cfg, location=location)
    
    # Poll until done (avoid .result() if your env had async quirks)
    while True:
        j = client.get_job(job.job_id, location=location)
        if j.done():
            if j.error_result:
                raise RuntimeError(j.error_result)
            break
        time.sleep(1)
    
    # Optional: quick row count
    client.query(
        f"SELECT COUNT(*) AS total_rows FROM `{project}.{DATASET}.{TABLE}`",
        location=location
    ).result().to_dataframe()



# =====================  Build Monthly features table ===================== 
sql_A0 = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.k8s_quality_monthly_2019` AS
WITH gh AS (
  SELECT
    FORMAT_DATE('%Y-%m', DATE(created_at)) AS ym,
    COUNT(*)                                           AS gh_events,
    COUNTIF(type = 'WorkflowRunEvent')                 AS ci_runs,
    COUNTIF(type = 'IssuesEvent')                      AS issues,
    COUNTIF(type = 'ReleaseEvent')                     AS releases
  FROM `{project}.{DATASET}.gh_month_2019_k8s`
  GROUP BY ym
),
so AS (
  SELECT
    FORMAT_DATE('%Y-%m', DATE(creation_date)) AS ym,
    COUNT(*)                                   AS so_questions
  FROM `{project}.{DATASET}.so_2019_kube_questions`
  GROUP BY ym
)
SELECT
  'kubernetes/kubernetes' AS product,
  ym,
  IFNULL(gh.gh_events, 0)    AS gh_events,
  IFNULL(gh.ci_runs, 0)      AS ci_runs,
  IFNULL(gh.issues, 0)       AS issues,
  IFNULL(gh.releases, 0)     AS releases,
  IFNULL(so.so_questions, 0) AS so_questions
FROM gh
FULL OUTER JOIN so USING (ym)
ORDER BY ym;
"""
bq(sql_A0)
print("âœ… 1.1 Done â†’ oss_quality.k8s_quality_monthly_2019")



# =====================  Load monthly metrics ===================== 
a0_df = client.query(
    f"""
    SELECT ym, gh_events, ci_runs, issues, releases, so_questions
    FROM `{project}.{DATASET}.k8s_quality_monthly_2019`
    ORDER BY ym
    """,
    location=location
).result().to_dataframe()

# Time axis + cleaning
a0_df["dt"] = pd.to_datetime(a0_df["ym"] + "-01")
a0_df = a0_df.sort_values("dt").fillna(0)
max_rel = int(a0_df["releases"].max()) if len(a0_df) else 1


# Force iframe renderer (helps in Kaggle Commit view)
pio.renderers.default = "iframe"

fig = go.Figure()

# Left axis: CI / Issues / SO
for col, label in [
    ("ci_runs", "CI Runs"),
    ("issues", "Issues"),
    ("so_questions", "StackOverflow Questions"),
]:
    fig.add_trace(go.Scatter(
        x=a0_df["dt"], y=a0_df[col],
        mode="lines+markers", name=label, yaxis="y1",
        hovertemplate=f"%{{x|%Y-%m}}<br>{label}: %{{y}}<extra></extra>",
    ))

# Right axis: GitHub Events
fig.add_trace(go.Scatter(
    x=a0_df["dt"], y=a0_df["gh_events"],
    mode="lines+markers", name="GitHub Events", yaxis="y2",
    hovertemplate="%{x|%Y-%m}<br>GitHub Events: %{y}<extra></extra>",
))

# Bars: Releases (keep bars, hide their axis visuals)
fig.add_trace(go.Bar(
    x=a0_df["dt"], y=a0_df["releases"],
    name="Releases", yaxis="y3",
    opacity=0.9, text=a0_df["releases"], textposition="auto",
    hovertemplate="%{x|%Y-%m}<br>Releases: %{y}<extra></extra>",
))

fig.update_layout(
    template="plotly_white",
    height=430,
    title=dict(
        text="Kubernetes Monthly Metrics (2019) â€” Triple Axis ",
        x=0.01, xanchor="left", y=0.98, yanchor="top"
    ),
    # Legend on its own row above title
    legend=dict(
        orientation="h", x=0, xanchor="left",
        y=1.14, yanchor="top",
        bgcolor="rgba(255,255,255,0.6)"
    ),
    # Plot area with a small right gutter
    xaxis=dict(title="Month", domain=[0.0, 0.90]),
    yaxis=dict(
        title="CI / Issues / SO Questions",
        rangemode="tozero",
        gridcolor="rgba(0,0,0,0.1)",
        showline=True,
    ),
    yaxis2=dict(
        title="GitHub Events",
        overlaying="y",
        side="right",
        anchor="free",
        position=0.94,
        rangemode="tozero",
        showgrid=False,
        showline=True,
    ),
    # Hide Releases axis (keep it only for scaling bars)
    yaxis3=dict(
        overlaying="y", side="right", anchor="free", position=0.98,
        showgrid=False, showline=False, showticklabels=False,
        ticks="", title_text=None, zeroline=False
    ),
    barmode="overlay",
    bargap=0.25,
    margin=dict(t=120, l=70, r=90, b=50),
)

display(Markdown("### âœ… **Preview : Github Event Monly Metrics**"))
fig.show(renderer="iframe")
#Reference table - Optional
display(a0_df[["ym","releases","ci_runs","issues","so_questions","gh_events"]].style.hide(axis="index"))



# =====================  Analysis: train ARIMA+ and forecast Github events (3 months horizon) ===================== 
sql_m_ghe = f"""
CREATE OR REPLACE MODEL `{project}.{DATASET}.k8s_ghevents_ts`
OPTIONS(
  MODEL_TYPE='ARIMA_PLUS',
  TIME_SERIES_TIMESTAMP_COL='dt',
  TIME_SERIES_DATA_COL='val',
  AUTO_ARIMA=TRUE
) AS
SELECT DATE(CONCAT(ym,'-01')) AS dt, gh_events AS val
FROM `{project}.{DATASET}.k8s_quality_monthly_2019`
ORDER BY dt;
"""
bq(sql_m_ghe)

sql_fc_ghe = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.k8s_forecast_gh_events_qnext` AS
SELECT * FROM ML.FORECAST(
  MODEL `{project}.{DATASET}.k8s_ghevents_ts`,
  STRUCT(3 AS horizon, 0.8 AS confidence_level)
);
"""
bq(sql_fc_ghe)
print("âœ… 1.2 done â†’ forecast GH events")

# Display smooth join from history â†’ forecast with confidence band

# History (monthly)
hist = client.query(
    f"""
    SELECT DATE(CONCAT(ym,'-01')) AS dt, gh_events
    FROM `{project}.{DATASET}.k8s_quality_monthly_2019`
    ORDER BY dt
    """,
    location=location
).result().to_dataframe()

# Forecast (next quarter)
fc = client.query(
    f"""
    SELECT
      forecast_timestamp AS dt,
      forecast_value     AS yhat,
      prediction_interval_lower_bound AS yhat_lo,
      prediction_interval_upper_bound AS yhat_hi
    FROM `{project}.{DATASET}.k8s_forecast_gh_events_qnext`
    ORDER BY dt
    """,
    location=location
).result().to_dataframe()

# Fix dtype/tz mismatches 
hist["dt"] = pd.to_datetime(hist["dt"], utc=True).dt.tz_convert(None)
fc["dt"]   = pd.to_datetime(fc["dt"],   utc=True).dt.tz_convert(None)

# Combined series to visually connect lines
combined = pd.concat(
    [
        hist.rename(columns={"gh_events": "value"})[["dt", "value"]],
        fc.rename(columns={"yhat": "value"})[["dt", "value"]],
    ],
    ignore_index=True,
).drop_duplicates(subset=["dt"], keep="last").sort_values("dt")

# Plot
pio.renderers.default = "iframe"  # helps in Kaggle Commit view
fig = go.Figure()

# Thin combined connector (smooth join)
fig.add_trace(go.Scatter(
    x=combined["dt"], y=combined["value"],
    mode="lines", line=dict(width=1), opacity=0.35,
    name="Combined", hoverinfo="skip", showlegend=False
))

# History
fig.add_trace(go.Scatter(
    x=hist["dt"], y=hist["gh_events"],
    mode="lines+markers", name="History"
))

# Forecast
fig.add_trace(go.Scatter(
    x=fc["dt"], y=fc["yhat"],
    mode="lines+markers", name="Forecast"
))

# Confidence band
fig.add_trace(go.Scatter(
    x=pd.concat([fc["dt"], fc["dt"][::-1]]),
    y=pd.concat([fc["yhat_hi"], fc["yhat_lo"][::-1]]),
    fill="toself", opacity=0.20, line=dict(width=0),
    name="Confidence", hoverinfo="skip"
))

fig.update_layout(
    template="plotly_white",
    height=430,
    title=dict(
        text="GitHub Events â€” Next Quarter Forecast (Continuous)",
        x=0.01, xanchor="left", y=0.98, yanchor="top"
    ),
    legend=dict(
        orientation="h", x=0, xanchor="left",
        y=1.14, yanchor="top",
        bgcolor="rgba(255,255,255,0.6)"
    ),
    xaxis_title="Month",
    yaxis_title="Events",
    margin=dict(t=120, l=60, r=30, b=50),
)

display(Markdown("### âœ… **Github Event Forecast** â€” created by ML.FORECAST"))
fig.show(renderer="iframe")

#  Show the forecast table (Optional)
display(fc)


# =====================  Forecast "issues" for the next quarter ===================== 
sql_forecast_issues = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.k8s_forecast_issues_qnext` AS
SELECT
  forecast_timestamp,
  forecast_value,
  prediction_interval_lower_bound,
  prediction_interval_upper_bound
FROM AI.FORECAST(
  (
    SELECT
      DATE(CONCAT(ym, '-01')) AS ts,
      CAST(issues AS FLOAT64) AS y
    FROM `{project}.{DATASET}.k8s_quality_monthly_2019`
    WHERE issues IS NOT NULL
    ORDER BY ts
  ),
  data_col => 'y',
  timestamp_col => 'ts',
  horizon => 3
);
"""
bq(sql_forecast_issues)
print("âœ… 1.3 Done. Issue Forecast created â†’", f"{project}.{DATASET}.k8s_forecast_issues_qnext")

# Display issues history + forecast
import pandas as pd, plotly.graph_objects as go

hist_issues = client.query(
    f"""
    SELECT DATE(CONCAT(ym,'-01')) AS dt, CAST(issues AS FLOAT64) AS val
    FROM `{project}.{DATASET}.k8s_quality_monthly_2019`
    ORDER BY dt
    """, location=location
).result().to_dataframe()

fc_issues = client.query(
    f"""
    SELECT
      forecast_timestamp AS dt,
      forecast_value     AS yhat,
      prediction_interval_lower_bound AS yhat_lo,
      prediction_interval_upper_bound AS yhat_hi
    FROM `{project}.{DATASET}.k8s_forecast_issues_qnext`
    ORDER BY dt
    """, location=location
).result().to_dataframe()

hist_issues["dt"] = pd.to_datetime(hist_issues["dt"]).dt.tz_localize(None)
fc_issues["dt"]   = pd.to_datetime(fc_issues["dt"]).dt.tz_localize(None)

combined_issues = pd.concat(
    [
        hist_issues.rename(columns={"val":"value"})[["dt","value"]],
        fc_issues.rename(columns={"yhat":"value"})[["dt","value"]],
    ],
    ignore_index=True
).drop_duplicates(subset=["dt"], keep="last").sort_values("dt")

# Plot with non-overlapping legend + iframe renderer
pio.renderers.default = "iframe"

fig = go.Figure()

# Confidence ribbon (behind lines)
fig.add_trace(go.Scatter(
    x=fc_issues["dt"], y=fc_issues["yhat_hi"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
))
fig.add_trace(go.Scatter(
    x=fc_issues["dt"], y=fc_issues["yhat_lo"],
    mode="lines", line=dict(width=0), fill="tonexty",
    name="Forecast 80% PI"
))

# History
fig.add_trace(go.Scatter(
    x=hist_issues["dt"], y=hist_issues["val"],
    mode="markers", name="Issues (history)"
))

# Combined line (history + forecast)
fig.add_trace(go.Scatter(
    x=combined_issues["dt"], y=combined_issues["value"],
    mode="lines+markers", name="Issues (history + forecast)"
))

fig.update_layout(
    template="plotly_white",
    height=430,
    title=dict(
        text="GitHub Issues â€” Next Quarter Forecast (AI.FORECAST)",
        x=0.01, xanchor="left", y=0.98, yanchor="top"
    ),
    xaxis_title="Month",
    yaxis_title="Issues",
    legend=dict(
        orientation="h",
        x=0, xanchor="left",
        y=1.14, yanchor="top",
        bgcolor="rgba(255,255,255,0.6)"
    ),
    margin=dict(t=120, l=60, r=30, b=50),
)

display(Markdown("### âœ… **GitHub Issue Forecast** â€” created by AI.FORECAST"))
fig.show(renderer="iframe")

#  Show the forecast table (Optional)
display(fc_issues)


# =====================  Create AI-generated quarterly executive insights based on statistical metrics from Github and Stackflow ===================== 
sql_A1_q = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.k8s_quarterly_gen_statistical_2019` AS
WITH m AS (
  SELECT DATE(CONCAT(ym,'-01')) AS dt, gh_events, ci_runs, issues, releases, so_questions
  FROM `{project}.{DATASET}.k8s_quality_monthly_2019`
),
q AS (
  SELECT
    'kubernetes/kubernetes' AS product,
    CONCAT(CAST(EXTRACT(YEAR FROM dt) AS STRING), '-Q', CAST(EXTRACT(QUARTER FROM dt) AS STRING)) AS qtr,
    SUM(gh_events)   AS gh_events,
    SUM(ci_runs)     AS ci_runs,
    SUM(issues)      AS issues,
    SUM(releases)    AS releases,
    SUM(so_questions) AS so_questions
  FROM m
  GROUP BY product, qtr
)
SELECT
  product,
  qtr,
  AI.GENERATE(
    CONCAT(
      'You are an engineering program lead. Produce a concise quarterly executive update for ', product, '. ',
      'Period: ', qtr, '. Inputs: ',
      'GitHub events=', gh_events, ', CI runs=', ci_runs, ', releases=', releases, ', issues=', issues,
      ', StackOverflow questions=', so_questions, '. ',
      'Output exactly two parts on one line each: ',
      'Summary: <<=100 words about the quarter>. ',
      'Recommendation: <one actionable step>.'
    ),
    connection_id => '{BQ_GENAI_CONNECTION }',
    endpoint      => '{GEN_TEXT_ENDPOINT}'
  ) AS insight
FROM q
ORDER BY qtr;
"""
bq(sql_A1_q)
print("âœ… 1.4 quarterly created â†’ oss_quality.exec_insights_k8s_2019_qtr")


# Display: Show Summary & Recommendation nicely
pd.set_option("display.max_colwidth", None)

q = f"""
SELECT qtr, CAST(insight.result AS STRING) AS txt
FROM `{project}.{DATASET}.k8s_quarterly_gen_statistical_2019`
ORDER BY qtr
"""
raw = client.query(q, location=location).result().to_dataframe()

def split_summary_reco(text: str):
    parts = re.split(r'(?i)\brecommendation\s*:\s*', text or "", maxsplit=1)
    return pd.Series({
        "summary": (parts[0].replace("Summary:", "").strip() if parts else None),
        "recommendation": (parts[1].strip() if len(parts)==2 else None)
    })

present = pd.concat([raw[["qtr"]], raw["txt"].apply(split_summary_reco)], axis=1)

display(Markdown("### âœ… **Quaterly Statistical Insights** - created by AI.GENERATE"))
display(show_left(present))


#   =====================  Create AI-generated quarterly executive insights based on textual payload from github events and stackflow questions ===================== 
sql_quarterly_gen = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.k8s_quarterly_gen_sentimental_2019` AS
-- PullRequestEvent (merged) â†’ collect PR titles per quarter
WITH prs_base AS (
  SELECT
    CONCAT(CAST(EXTRACT(YEAR FROM created_at) AS STRING), '-Q', CAST(EXTRACT(QUARTER FROM created_at) AS STRING)) AS qtr,
    JSON_VALUE(payload, '$.pull_request.title') AS title,
    created_at
  FROM `{project}.{DATASET}.gh_month_2019_k8s`
  WHERE type = 'PullRequestEvent'
    AND JSON_VALUE(payload, '$.pull_request.merged') = 'true'
),
prs_dedup AS (
  SELECT
    qtr, title, created_at,
    ROW_NUMBER() OVER (PARTITION BY qtr, title ORDER BY created_at DESC) AS rn
  FROM prs_base
  WHERE title IS NOT NULL
),
prs_titles AS (
  SELECT
    qtr,
    ARRAY_AGG(title ORDER BY created_at DESC LIMIT 15) AS pr_titles
  FROM prs_dedup
  WHERE rn = 1
  GROUP BY qtr
),

-- IssuesEvent (opened) â†’ collect issue titles by most commented
issues_base AS (
  SELECT
    CONCAT(CAST(EXTRACT(YEAR FROM created_at) AS STRING), '-Q', CAST(EXTRACT(QUARTER FROM created_at) AS STRING)) AS qtr,
    JSON_VALUE(payload, '$.issue.title') AS title,
    CAST(JSON_VALUE(payload, '$.issue.comments') AS INT64) AS comments
  FROM `{project}.{DATASET}.gh_month_2019_k8s`
  WHERE type = 'IssuesEvent'
    AND JSON_VALUE(payload, '$.action') = 'opened'
),
issues_dedup AS (
  SELECT
    qtr, title, IFNULL(comments, 0) AS comments,
    ROW_NUMBER() OVER (PARTITION BY qtr, title ORDER BY IFNULL(comments,0) DESC) AS rn
  FROM issues_base
  WHERE title IS NOT NULL
),
issues_titles AS (
  SELECT
    qtr,
    ARRAY_AGG(title ORDER BY comments DESC LIMIT 15) AS issue_titles
  FROM issues_dedup
  WHERE rn = 1
  GROUP BY qtr
),

-- StackOverflow question titles per quarter
so_base AS (
  SELECT
    CONCAT(CAST(EXTRACT(YEAR FROM creation_date) AS STRING), '-Q', CAST(EXTRACT(QUARTER FROM creation_date) AS STRING)) AS qtr,
    title, score, view_count
  FROM `{project}.{DATASET}.so_2019_kube_questions`
  WHERE title IS NOT NULL
),
so_dedup AS (
  SELECT
    qtr, title, score, view_count,
    ROW_NUMBER() OVER (PARTITION BY qtr, title ORDER BY score DESC, view_count DESC) AS rn
  FROM so_base
),
soq AS (
  SELECT
    qtr,
    ARRAY_AGG(title ORDER BY score DESC, view_count DESC LIMIT 15) AS so_titles
  FROM so_dedup
  WHERE rn = 1
  GROUP BY qtr
),

-- Ensure all 2019 quarters appear
qall AS (
  SELECT CONCAT('2019-','Q', q) AS qtr FROM UNNEST([1,2,3,4]) AS q
),

joined AS (
  SELECT
    qall.qtr,
    IFNULL(p.pr_titles,    []) AS pr_titles,
    IFNULL(i.issue_titles, []) AS issue_titles,
    IFNULL(s.so_titles,    []) AS so_titles
  FROM qall
  LEFT JOIN prs_titles    p USING (qtr)
  LEFT JOIN issues_titles i USING (qtr)
  LEFT JOIN soq           s USING (qtr)
)

SELECT
  qtr,
  AI.GENERATE(
    CONCAT(
      'Provide a concise quarterly summary for Kubernetes. ',
      'Focus on features (PRs merged), major issues, and key StackOverflow questions. ',
      'Return <=150 words. ',
      'PRs: ', ARRAY_TO_STRING(pr_titles, ' | '), '. ',
      'Issues: ', ARRAY_TO_STRING(issue_titles, ' | '), '. ',
      'SO: ', ARRAY_TO_STRING(so_titles, ' | ')
    ),
    connection_id => '{BQ_GENAI_CONNECTION}',
    endpoint      => '{GEN_TEXT_ENDPOINT}'
  ) AS summary
FROM joined
ORDER BY qtr;
"""
bq(sql_quarterly_gen)
print("âœ… 1.6 done â†’ k8s_quarterly_gen_2019")

# Display: quarterly summaries
pd.set_option("display.max_colwidth", None)

qsum = client.query(
    f"SELECT qtr, CAST(summary.result AS STRING) AS summary_text "
    f"FROM `{project}.{DATASET}.k8s_quarterly_gen_sentimental_2019` ORDER BY qtr",
    location=location
).result().to_dataframe()

display(Markdown("### âœ… **Quaterly Sentimental Insights** - created by AI.GENERATE"))
show_left(qsum)


# =====================  Build a 2019 yearly summary from the two quarterly tables: ===================== 
#   - k8s_quarterly_gen_statistical_2019   (AI over stats)
#   - k8s_quarterly_gen_sentimental_2019   (AI over titles/text)
#
# Creates: {DATASET}.yearly_summary_2019 (year, yearly_text)

conn_id  = BQ_GENAI_CONNECTION
endpoint = GEN_TEXT_ENDPOINT

sql_yearly = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.yearly_summary_2019` AS
WITH stat_q AS (
  SELECT COALESCE(
           STRING_AGG(CONCAT(qtr, ': ', CAST(insight.result AS STRING)), '\\n\\n' ORDER BY qtr),
           ''
         ) AS text
  FROM `{project}.{DATASET}.k8s_quarterly_gen_statistical_2019`
),
sent_q AS (
  SELECT COALESCE(
           STRING_AGG(CONCAT(qtr, ': ', CAST(summary.result AS STRING)), '\\n\\n' ORDER BY qtr),
           ''
         ) AS text
  FROM `{project}.{DATASET}.k8s_quarterly_gen_sentimental_2019`
)
SELECT
  '2019' AS year,
  CAST(
    AI.GENERATE(
      CONCAT(
        'You are a VP of Engineering writing a concise 2019 year-in-review for Kubernetes. ',
        'Use BOTH sources provided below.\\n\\n',
        'A) Statistical exec insights per quarter:\\n', (SELECT text FROM stat_q), '\\n\\n',
        'B) Textual/semantic summaries per quarter:\\n', (SELECT text FROM sent_q), '\\n\\n',
        'Task: First, write a <=200-word yearly summary highlighting themes and quality signals without repeating raw numbers. ',
        'Then, provide exactly 3 bullet-like action items focused on testing, CI, and release quality. ',
        'Be factual and non-redundant.'
      ),
      connection_id => '{conn_id}',
      endpoint      => '{endpoint}'
    ).result AS STRING
  ) AS yearly_text;
"""
bq(sql_yearly)
print("âœ… Built:", f"{project}.{DATASET}.yearly_summary_2019")



#  Display the yearly text
df = client.query(
    f"SELECT year, yearly_text FROM `{project}.{DATASET}.yearly_summary_2019`",
    location=location
).result().to_dataframe()

display(Markdown("### âœ… **Yearly Summary** converging statistical and generative analysis - created by AI.GENERATE"))
display(show_left(df))


# =====================  Build a unified â€œtest-focus signalâ€� text per GitHub event (2019, k8s repo) ===================== 
sql_test_texts = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.k8s_test_texts_2019` AS
SELECT
  GENERATE_UUID() AS id,
  TIMESTAMP(created_at) AS created_at,
  type,
  -- Synthesize a test-focused text from payload:
  CASE
    WHEN type = 'PushEvent' THEN CONCAT(
      'type=PushEvent; ref=', COALESCE(JSON_VALUE(payload, '$.ref'), ''),
      '; commit_messages=[', (
        SELECT IFNULL(STRING_AGG(DISTINCT JSON_VALUE(c, '$.message'), ' | '), '')
        FROM UNNEST(IFNULL(JSON_QUERY_ARRAY(payload, '$.commits'), [])) AS c
      ), ']'
    )
    WHEN type = 'PullRequestEvent' THEN CONCAT(
      'type=PullRequestEvent; action=', COALESCE(JSON_VALUE(payload,'$.action'), ''),
      '; merged=', COALESCE(JSON_VALUE(payload,'$.pull_request.merged'), ''),
      '; title=', COALESCE(JSON_VALUE(payload,'$.pull_request.title'), ''),
      '; body=', COALESCE(JSON_VALUE(payload,'$.pull_request.body'), '')
    )
    WHEN type = 'ReleaseEvent' THEN CONCAT(
      'type=ReleaseEvent; action=', COALESCE(JSON_VALUE(payload,'$.action'), ''),
      '; tag=', COALESCE(JSON_VALUE(payload,'$.release.tag_name'), ''),
      '; name=', COALESCE(JSON_VALUE(payload,'$.release.name'), ''),
      '; notes=', COALESCE(JSON_VALUE(payload,'$.release.body'), '')
    )
    WHEN type = 'IssuesEvent' THEN CONCAT(
      'type=IssuesEvent; action=', COALESCE(JSON_VALUE(payload,'$.action'), ''),
      '; title=', COALESCE(JSON_VALUE(payload,'$.issue.title'), ''),
      '; body=', COALESCE(JSON_VALUE(payload,'$.issue.body'), '')
    )
    ELSE CONCAT('type=', type, '; payload_snippet=', SUBSTR(payload,1,1000))
  END AS text
FROM `{project}.{DATASET}.gh_month_2019_k8s`;
"""
bq(sql_test_texts)
print("âœ… 2.1 table created â†’ oss_quality.k8s_test_texts_2019")


# =====================  Create embedding model, embed test texts ===================== 
# 1) Remote embedding model
CONN_SHORT = f"{location.lower()}.genai"

sql_embed_model = f"""
CREATE OR REPLACE MODEL `{project}.{DATASET}.test_embed_model`
REMOTE WITH CONNECTION `{CONN_SHORT}`
OPTIONS (endpoint = '{GEN_EMBEDDING_ENDPOINT}');
"""
bq(sql_embed_model)
print("âœ… Embedding model ready â†’", f"{project}.{DATASET}.test_embed_model")

# 2) Embed GitHub test signals (TVF with `content`; alias the returned column)
EMBED_TBL = f"{project}.{DATASET}.k8s_test_texts_embed"
SRC_TBL   = f"{project}.{DATASET}.k8s_test_texts_2019"

if _table_exists(EMBED_TBL) and SKIP_GCS:
    print(f"â�­ï¸�  Skipping embed build because {EMBED_TBL} already exists and SKIP_GCS=True")
else:
    # Ensure source exists
    if not _table_exists(SRC_TBL):
        raise RuntimeError(f"Source table not found: {SRC_TBL}")

    sql_embed_gh = f"""
    CREATE OR REPLACE TABLE `{EMBED_TBL}` AS
    WITH src AS (
      SELECT id, created_at, type, CAST(text AS STRING) AS text
      FROM `{SRC_TBL}`
      WHERE text IS NOT NULL AND LENGTH(text) > 0
    ),
    prep AS (
      SELECT id, created_at, type, SUBSTR(text, 1, 4000) AS content  -- TVF requires `content`
      FROM src
    )
    SELECT
      id,
      created_at,
      type,
      content AS text,
      ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
      MODEL `{project}.{DATASET}.test_embed_model`,
      (SELECT id, created_at, type, content FROM prep)
    );
    """
    # Note: Embedding can take 8 minutes depending on row count
    bq(sql_embed_gh)
    print("âœ… Embedded texts â†’", EMBED_TBL)


# ===================== Check embedding dimensions ===================== 
dim_stats = client.query(
    f"""
    SELECT
      COUNT(*) AS row_count,
      ARRAY_LENGTH(embedding) AS dim
    FROM `{project}.{DATASET}.k8s_test_texts_embed`
    WHERE embedding IS NOT NULL
    GROUP BY dim
    ORDER BY row_count DESC
    """,
    location=location
).result().to_dataframe()

display(Markdown("### âœ… **Embedding dimension** of Github events in BigQuery - created by ML.GENERATE_EMBEDDING"))
display(dim_stats)


# ============================= Ingest PDF from GCS towards BigQuery ===================
# List PDFs under gs://BUCKET/PREFIX
st = storage.Client(project=project, credentials=creds)
blobs = st.list_blobs(BUCKET, prefix=PREFIX)

rows = []
for b in blobs:
    if not b.name.lower().endswith(".pdf"):
        continue
    # fetch bytes in memory
    pdf_bytes = b.download_as_bytes()      # requires storage.objects.get
    # extract text
    try:
        text = extract_text(io.BytesIO(pdf_bytes)) or ""
    except Exception:
        text = ""
    text = text.replace("\x00", " ").strip()
    if not text:
        # keep a tiny placeholder so rows arenâ€™t dropped
        text = "(empty or non-text PDF content)"
    rows.append({
        "file_name": pathlib.Path(b.name).name,
        "gcs_uri": f"gs://{BUCKET}/{b.name}",
        "text": text[:MAX_CHARS]
    })

df = pd.DataFrame(rows)
print(f"PDFs found: {len(df)} | total chars (capped per file): {df['text'].str.len().sum():,}")

# Load to BigQuery table
ds = bigquery.Dataset(f"{project}.{DATASET}")
ds.location = location
client.create_dataset(ds, exists_ok=True)

table_id = f"{project}.{DATASET}.k8s_doc_texts"
job_cfg = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",
    schema=[
        bigquery.SchemaField("file_name", "STRING"),
        bigquery.SchemaField("gcs_uri",   "STRING"),
        bigquery.SchemaField("text",      "STRING"),
    ],
)
client.load_table_from_dataframe(df, table_id, job_config=job_cfg, location=location).result()
print("âœ… Loaded:", table_id)


#=====================  Preview the first loaded pdf summary in BigQuery dataset ===================== 
preview = client.query(
    f"SELECT file_name, CAST(gen.result AS STRING) AS summary FROM `{project}.{DATASET}.k8s_doc_summaries` LIMIT 2",
    location=location
).result().to_dataframe()

display(Markdown("### âœ… **Sample PDF summary in BigQuery**"))
display(preview.style.hide(axis='index').set_properties(**{'text-align':'left','white-space':'pre-wrap'})
        .set_table_styles([{'selector':'th','props':[('text-align','left')]}]))



# ===================== Detect source table + correct text expression  ===================== 
def table_exists(fqn: str) -> bool:
    proj, ds, tbl = fqn.split(".")
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("t", "STRING", tbl)]
    )
    sql = f"""
      SELECT 1
      FROM `{proj}.{ds}.INFORMATION_SCHEMA.TABLES`
      WHERE table_name = @t
      LIMIT 1
    """
    return client.query(sql, job_config=cfg, location=location).result().to_dataframe().shape[0] == 1

def list_columns(fqn: str):
    proj, ds, tbl = fqn.split(".")
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("t", "STRING", tbl)]
    )
    sql = f"""
      SELECT LOWER(column_name) AS c, data_type
      FROM `{proj}.{ds}.INFORMATION_SCHEMA.COLUMNS`
      WHERE table_name = @t
    """
    df = client.query(sql, job_config=cfg, location=location).result().to_dataframe()
    return {row["c"]: row["data_type"] for _, row in df.iterrows()}

SRC_SUM  = f"{project}.{DATASET}.k8s_doc_summaries"  # preferred
SRC_TEXT = f"{project}.{DATASET}.k8s_doc_texts"      # fallback

source_table = None
text_expr    = None  # SQL expression that yields STRING (e.g., "summary", "gen.result", "text")

for candidate in (SRC_SUM, SRC_TEXT):
    if not table_exists(candidate):
        continue
    cols = list_columns(candidate)
    # Prefer typical names; handle STRUCT 'gen' specially
    if "summary" in cols:
        text_expr = "summary"
    elif "text" in cols:
        text_expr = "text"
    elif "gen" in cols:
        # AI.GENERATE output STRUCT â€” use the STRING field 'result'
        text_expr = "gen.result"
    elif "result" in cols:  # sometimes the string result is flattened
        text_expr = "result"
    elif "content" in cols:
        text_expr = "content"
    elif "comment_text" in cols:
        text_expr = "comment_text"
    else:
        text_expr = None

    if text_expr:
        source_table = candidate
        break

assert source_table and text_expr, (
    "No usable text column found. Checked "
    f"{SRC_SUM} and {SRC_TEXT} for one of: summary/text/gen(.result)/result/content/comment_text."
)
print(f"ğŸ”� Using source: {source_table} column/expression: {text_expr}")

# Create the embeddings table
sql_embed = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.k8s_doc_embed` AS
WITH src AS (
  SELECT
    file_name,
    CAST({text_expr} AS STRING) AS content
  FROM `{source_table}`
  WHERE CAST({text_expr} AS STRING) IS NOT NULL
    AND CAST({text_expr} AS STRING) != ''
)
SELECT
  s.file_name,
  s.content AS text,
  (
    SELECT ml_generate_embedding_result
    FROM ML.GENERATE_EMBEDDING(
      MODEL `{project}.{DATASET}.test_embed_model`,
      (SELECT s.content AS content)   -- TVF requires a column literally named 'content'
    )
  ) AS embedding
FROM src AS s;
"""
bq(sql_embed)
print("âœ… Built:", f"{project}.{DATASET}.k8s_doc_embed")

#  Quick checks 5 embedded pdf summary dimension and content
dim_df = client.query(
    f"""
    SELECT COUNT(*) AS row_count, ARRAY_LENGTH(embedding) AS dim
    FROM `{project}.{DATASET}.k8s_doc_embed`
    GROUP BY dim
    ORDER BY row_count DESC
    """,
    location=location
).result().to_dataframe()
display(Markdown("### âœ… **Preview embedding** of PDFs summaries - created by ML.GENERATE_EMBEDDING "))
display(show_left(dim_df))

preview_df = client.query(
    f"""
    SELECT file_name, SUBSTR(text, 1, 200) AS text_head,
           ARRAY_LENGTH(embedding) AS dim
    FROM `{project}.{DATASET}.k8s_doc_embed`
    LIMIT 5
    """, location=location
).result().to_dataframe()

display(show_left(preview_df))



# ===========RAG Troubleshooter: VECTOR_SEARCH with similarity [0,1] over DOC + Github embeddings ================

def k8s_rag_troubleshoot(
    issue: str,
    top_docs: int = 6,
    top_gh: int = 6,
    max_ctx: int = 900,
    *,
    tbl_doc: str = None,
    tbl_gh: str = None,
    emb_model: str = None,
    connection_id: str = None,
    endpoint: str = None,
):
    """
    Run a DOC + GitHub RAG search in BigQuery and generate troubleshooting guidance.

    Returns: (ctx_df, guidance_text)
    """
    import re
    from google.cloud import bigquery

    # Use globals if not provided
    _tbl_doc   = tbl_doc   or f"{project}.{DATASET}.k8s_doc_embed"
    _tbl_gh    = tbl_gh    or f"{project}.{DATASET}.k8s_test_texts_embed"
    _emb_model = emb_model or f"{project}.{DATASET}.test_embed_model"
    _conn_id   = connection_id or BQ_GENAI_CONNECTION
    _endpoint  = endpoint or GEN_TEXT_ENDPOINT

    # 1) Build context (no temp table; just a UNION query)
    sql_ctx = f"""
    WITH
      q AS (
        SELECT ml_generate_embedding_result AS emb
        FROM ML.GENERATE_EMBEDDING(
          MODEL `{_emb_model}`,
          (SELECT @q AS content)
        )
      ),
      docs AS (
        SELECT
          'doc' AS source,
          vs.base.file_name AS id,
          CAST(NULL AS STRING) AS type,
          CAST(NULL AS TIMESTAMP) AS created_at,
          SUBSTR(vs.base.text, 1, 4000) AS context,
          1 - (vs.distance / 2.0) AS similarity
        FROM VECTOR_SEARCH(
          (SELECT file_name, text, embedding FROM `{_tbl_doc}`),
          'embedding',
          (SELECT emb FROM q),
          'emb',
          top_k => @kdoc,
          distance_type => 'COSINE',
          options => '{{"use_brute_force": true}}'
        ) AS vs
      ),
      gh AS (
        SELECT
          'github' AS source,
          vs.base.id AS id,
          vs.base.type AS type,
          vs.base.created_at AS created_at,
          SUBSTR(vs.base.text, 1, 4000) AS context,
          1 - (vs.distance / 2.0) AS similarity
        FROM VECTOR_SEARCH(
          (SELECT id, type, created_at, text, embedding FROM `{_tbl_gh}`),
          'embedding',
          (SELECT emb FROM q),
          'emb',
          top_k => @kgh,
          distance_type => 'COSINE',
          options => '{{"use_brute_force": true}}'
        ) AS vs
      )
    SELECT source, id, type, created_at, context, similarity
    FROM (
      SELECT * FROM docs
      UNION ALL
      SELECT * FROM gh
    )
    ORDER BY similarity DESC
    """

    params = [
        bigquery.ScalarQueryParameter("q",    "STRING", re.sub(r"\s+"," ", issue)[:160]),
        bigquery.ScalarQueryParameter("kdoc", "INT64",  int(top_docs)),
        bigquery.ScalarQueryParameter("kgh",  "INT64",  int(top_gh)),
    ]

    ctx_df = client.query(
        sql_ctx,
        job_config=bigquery.QueryJobConfig(query_parameters=params),
        location=location
    ).result().to_dataframe()

    # 2) Build compact grounding
    def _sig(r):
        head = f"{r.get('source','')} {(r.get('type') or '')} {(str(r.get('created_at')) if r.get('created_at') else '')}".strip()
        txt  = (r.get('context') or '')[:max_ctx]
        return head + "\n" + txt

    signals = "\n---\n".join(_sig(r) for _, r in ctx_df.iterrows())

    # 3) Generate guidance
    prompt = (
        "You are a Kubernetes SRE.\n"
        f"Issue: {issue}\n"
        "Using ONLY the context below, provide:\n"
        " - Key insight(s)\n"
        " - Probable causes\n"
        " - Diagnostics (kubectl, logs, events)\n"
        " - Remediations / config fixes\n"
        "Be specific, keep the reply in 300 words, no guesses beyond the context, be version aware.\n\n"
        "Context:\n" + signals
    )

    sql_gen = (
        "SELECT CAST(\n"
        "  AI.GENERATE(\n"
        "    @p,\n"
        f"    connection_id => '{_conn_id}',\n"
        f"    endpoint      => '{_endpoint}'\n"
        "  ).result AS STRING\n"
        ") AS guidance"
    )

    res_df = client.query(
        sql_gen,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("p","STRING", prompt)]
        ),
        location=location
    ).result().to_dataframe()

    guidance_text = (res_df.iloc[0, 0] if len(res_df) else "(no output)")

    # Optional: quick view
    try:
        from IPython.display import display
        print("ğŸ§© Top matches (concise):")
        display(ctx_df[["source","id","type","created_at","similarity"]]
                .head(top_docs+top_gh).style.hide(axis="index"))
    except Exception:
        pass

    return ctx_df, guidance_text


#=================== TRY the RAG Troubleshoot - VECTOR_SEARCH =============
QUESTION = "Pods fail to mount CSI volumes after upgrade to v1.17"
ctx, guidance = k8s_rag_troubleshoot(
    QUESTION,
    top_docs=6,
    top_gh=6,
)

display(Markdown(f"### âœ… **Troubleshooting guidance** , based on GitHub + PDFs using VECTOR_SEARCH. \n\n **Question**: {QUESTION}. \n\n **Analysis**:"))
print(guidance)


# =====================  Download audio-only from YouTube to the Kaggle workspace ===================== 
# --- Config ---
VIDEO_IDS = ["Zv2fxIdj85s", "omh9eNeD5rw"]  # <-- your IDs
URLS = [f"https://www.youtube.com/watch?v={v}" for v in VIDEO_IDS]
AUDIO_DIR = Path("/kaggle/working/yt_audio")
BUCKET = "kagglempqa"         # <-- change if needed
PREFIX = "youtube/audio/"     # gs://BUCKET/PREFIX*.m4a

# --- GCS pre-check ---
try:
    st = storage.Client(project=project, credentials=creds if "creds" in globals() else None)
except Exception:
    st = storage.Client(project=project)

existing = list(st.list_blobs(BUCKET, prefix=PREFIX, max_results=3))
if SKIP_GCS and len(existing) > 0:
    print(f"â�­ï¸�  Skipping download & upload â€” found {len(existing)} file(s) under gs://{BUCKET}/{PREFIX}")
    for b in existing:
        print("â€¢", f"gs://{BUCKET}/{b.name}")
else:
    # Ensure local folder exists
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Download from YouTube as M4A ----
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"])
        import yt_dlp  # noqa: F401

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in URLS:
            try:
                ydl.extract_info(u, download=True)
            except Exception as e:
                print("âš ï¸�", u, e)

    files = list(AUDIO_DIR.glob("*.m4a"))
    print(f"ğŸ�§ Downloaded {len(files)} audio file(s):", [f.name for f in files])

    # ---- Upload to GCS ----
    bucket = st.bucket(BUCKET)
    uploaded = 0
    for fp in files:
        blob = bucket.blob(PREFIX + fp.name)
        blob.upload_from_filename(str(fp), content_type="audio/mp4")
        uploaded += 1
        print("uploaded:", f"gs://{BUCKET}/{PREFIX}{fp.name}")
    print("âœ… Uploaded", uploaded)


# =====================  Create an Object Table over the audio in BigQuery (Multimodal) ===================== 
sql_obj = f"""
CREATE OR REPLACE EXTERNAL TABLE `{project}.{DATASET}.yt_audio_ot`
WITH CONNECTION `{location.lower()}.genai`
OPTIONS (
  object_metadata = 'SIMPLE',
  uris = ['gs://{BUCKET}/{PREFIX}*.m4a']
);
"""
bq(sql_obj)
print("âœ… Object table:", f"{project}.{DATASET}.yt_audio_ot")


# =====================  Inspect dataset yt_audio_ot, ensure it contain essential information - OPTIONAL =====================      
warnings.filterwarnings("ignore",
    message="Unable to determine type for field 'access_url'",
    module="google.cloud.bigquery._pandas_helpers")

# 1) Ensure the object table exists
exists = client.query(
    f"""
    SELECT table_name
    FROM `{project}.{DATASET}.INFORMATION_SCHEMA.TABLES`
    WHERE table_name = 'yt_audio_ot'
    """,
    location=location
).result().to_dataframe()
assert not exists.empty, f"Missing table: {project}.{DATASET}.yt_audio_ot"

        
# 2) Show schema - Optional
schema = client.query(
    f"""
    SELECT column_name, data_type, is_nullable
    FROM `{project}.{DATASET}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'yt_audio_ot'
    ORDER BY ordinal_position
    """,
    location=location
).result().to_dataframe()
#print("Schema:")
#display(show_left(schema))

# 3) Row counts / content types - Optional
counts = client.query(
    f"""
    SELECT
      COUNT(*) AS rows_total,
      COUNTIF(STARTS_WITH(content_type,'audio/')) AS rows_audio,
      COUNTIF(STARTS_WITH(content_type,'video/')) AS rows_video,
      COUNTIF(content_type IS NULL)               AS rows_ct_null
    FROM `{project}.{DATASET}.yt_audio_ot`
    """,
    location=location
).result().to_dataframe()
print("Row counts:")
display(counts)

types = client.query(
    f"""
    SELECT content_type, COUNT(*) AS n
    FROM `{project}.{DATASET}.yt_audio_ot`
    GROUP BY content_type
    ORDER BY n DESC NULLS LAST
    LIMIT 20
    """,
    location=location
).result().to_dataframe()
print("Top content types:")
display(show_left(types))

# 4) Sample a few audio rows (file name, uri, ref fields) - Optional
sample = client.query(
    f"""
    SELECT
      REGEXP_EXTRACT(uri, r'/([^/]+)$') AS file_name,
      content_type,
      uri,
      ref.uri     AS ref_uri,
      ref.version AS ref_version
    FROM `{project}.{DATASET}.yt_audio_ot`
    WHERE STARTS_WITH(content_type,'audio/')
    LIMIT 5
    """,
    location=location
).result().to_dataframe()
display(Markdown("### âœ… **Preview ObjectTable and ObjectRef** of multimedia - created by AI.GENERATE_TABLE"))
print("Sample audio rows:")
display(show_left(sample))

# 5) Test ObjectRef â†’ signed access URL (requires bucket access for the BigQuery connection SA)
try:
    url_df = client.query(
        f"""
        SELECT OBJ.GET_ACCESS_URL(ref, 'r') AS access_url
        FROM `{project}.{DATASET}.yt_audio_ot`
        WHERE STARTS_WITH(content_type,'audio/')
        LIMIT 3
        """,
        location=location
    ).result().to_dataframe(dtypes={"access_url": pd.StringDtype()})
    print("OBJ.GET_ACCESS_URL(ref,'r') test (first 3):")
    display(show_left(url_df))
except Exception as e:
    print("âš ï¸�  OBJ.GET_ACCESS_URL(ref,'r') failed â€” grant Storage Object Viewer to your BigQuery *connection* service account on the bucket.")
    print("Error:", e)


# =====================  Build embedding of the transcript per audio ===================== 

# 1) Set Remote Models
bq(f"""
CREATE OR REPLACE MODEL `{project}.{DATASET}.gen_text_model`
  REMOTE WITH CONNECTION `{location.lower()}.genai`
  OPTIONS (endpoint = '{GEN_TEXT_ENDPOINT}');
""")
print("âœ… Remote gen text model set to:", GEN_TEXT_ENDPOINT)

bq(f"""
CREATE OR REPLACE MODEL `{project}.{DATASET}.test_embed_model`
  REMOTE WITH CONNECTION `{location.lower()}.genai`
  OPTIONS (endpoint = '{GEN_EMBEDDING_ENDPOINT}');
""")
print("âœ… Remote embedding model set to:", GEN_EMBEDDING_ENDPOINT)

# 2) Transcribe each audio 
sql_transcripts = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.yt_audio_transcripts` AS
SELECT
  file_name,
  transcript
FROM AI.GENERATE_TABLE(
  MODEL `{project}.{DATASET}.gen_text_model`,
  (
    SELECT
      (
        'You are a precise transcriber.',
        'Transcribe the audio verbatim in English.',
        'Return exactly one column named "transcript".',
        OBJ.GET_ACCESS_URL(ref, 'r')
      ) AS prompt,
      REGEXP_EXTRACT(uri, r'/([^/]+)$') AS file_name
    FROM `{project}.{DATASET}.yt_audio_ot`
    WHERE STARTS_WITH(content_type,'audio/')
  ),
  STRUCT('transcript STRING' AS output_schema)  -- âœ… must be a STRUCT
);
"""
bq(sql_transcripts)
print("âœ… Built:", f"{project}.{DATASET}.yt_audio_transcripts")


# =====================  Chunk full transcript and embed (chunks + doc-level centroid) ===================== 
CHUNK_CHARS = 4000
# 1) Chunks
bq(f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.yt_audio_transcript_chunks` AS
WITH base AS (
  SELECT file_name, transcript, LENGTH(transcript) AS n
  FROM `{project}.{DATASET}.yt_audio_transcripts`
  WHERE transcript IS NOT NULL AND transcript != ''
),
parts AS (
  SELECT
    file_name,
    pos AS chunk_index,
    SUBSTR(transcript, pos*{CHUNK_CHARS}+1, {CHUNK_CHARS}) AS text
  FROM base, UNNEST(GENERATE_ARRAY(0, CAST(CEIL(n / {CHUNK_CHARS}) AS INT64) - 1)) AS pos
)
SELECT * FROM parts WHERE text IS NOT NULL AND text != '';
""")
print("âœ… Built:", f"{project}.{DATASET}.yt_audio_transcript_chunks")

# 2) Chunk embeddings
bq(f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.yt_audio_transcript_embed_chunks` AS
SELECT
  c.file_name,
  c.chunk_index,
  c.text,
  (
    SELECT ml_generate_embedding_result
    FROM ML.GENERATE_EMBEDDING(
      MODEL `{project}.{DATASET}.test_embed_model`,
      (SELECT c.text AS content)
    )
  ) AS embedding
FROM `{project}.{DATASET}.yt_audio_transcript_chunks` AS c;
""")
print("âœ… Built:", f"{project}.{DATASET}.yt_audio_transcript_embed_chunks")

# 3) Doc-level centroid + concatenated full text
bq(f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.yt_audio_transcript_embed` AS
WITH unn AS (
  SELECT file_name, pos, AVG(val) AS val
  FROM (
    SELECT file_name, val, pos
    FROM `{project}.{DATASET}.yt_audio_transcript_embed_chunks`,
         UNNEST(embedding) AS val WITH OFFSET pos
  )
  GROUP BY file_name, pos
),
agg AS (
  SELECT file_name, ARRAY_AGG(val ORDER BY pos) AS embedding
  FROM unn
  GROUP BY file_name
),
fulltext AS (
  SELECT file_name, STRING_AGG(text, '\\n\\n' ORDER BY chunk_index) AS text
  FROM `{project}.{DATASET}.yt_audio_transcript_embed_chunks`
  GROUP BY file_name
)
SELECT a.file_name, f.text, a.embedding
FROM agg a
JOIN fulltext f USING (file_name);
""")
print("âœ… Built:", f"{project}.{DATASET}.yt_audio_transcript_embed")

# 4) Quick preview
df = client.query(
    f"""
    SELECT file_name,
           ARRAY_LENGTH(embedding) AS dim,
           LENGTH(text) AS n_chars,
           SUBSTR(text,1,140) AS text_head
    FROM `{project}.{DATASET}.yt_audio_transcript_embed`
    ORDER BY file_name
    LIMIT 5
    """,
    location=location
).result().to_dataframe()

# Show proposed tests
display(Markdown("### âœ… **Preview embedding dimension** of audio transcripts - by ML.GENERATE_EMBEDDING"))
show_left(df)


# ===================== Test Advisor based on GitHub + PDF + Meeting ===================== 
# Set embeddings table for RAG retrieval , including Github, pdf document, YouTube transcript
TBL_GH     = f"{project}.{DATASET}.k8s_test_texts_embed"          # GitHub events/texts + embeddings
TBL_DOC    = f"{project}.{DATASET}.k8s_doc_embed"                 # PDF doc summaries + embeddings
TBL_AUDIO  = f"{project}.{DATASET}.yt_audio_transcript_embed"     # YouTube transcripts + embeddings  # <-- NEW
EMB_MODEL  = f"{project}.{DATASET}.test_embed_model"              # Remote embedding model FQN

# Helper functions
def _table_exists(fqn: str) -> bool:
    # Utility: checks INFORMATION_SCHEMA to see if a table exists
    proj, ds, tbl = fqn.split(".")
    sql = f"""
      SELECT 1
      FROM `{proj}.{ds}.INFORMATION_SCHEMA.TABLES`
      WHERE table_name = @t
      LIMIT 1
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("t","STRING", tbl)])
    return client.query(sql, job_config=cfg, location=location).result().to_dataframe().shape[0] == 1

def _build_regex(q: str) -> str:
    # Create a permissive regex from the query:
    #   - keep alphanum/dot/dash tokens (>= 3 chars)
    #   - seed with common K8s terms to improve recall
    toks = re.findall(r"[A-Za-z0-9\.\-]{3,}", q.lower())
    toks += ["kubernetes","k8s","ingress","egress","network","cni","upgrade","1\\.17","release","controller","api","storage","csi"]
    toks = sorted(set(toks))
    return "(" + "|".join(map(re.escape, toks)) + ")"

# Test case proposal function
def propose_tests(
    query: str,
    top_gh: int = 6,           # how many GitHub rows to return post-distance
    top_docs: int = 6,         # how many Doc rows to return post-distance
    top_audio: int = 6,        # how many Audio rows to return post-distance
    prefilter_gh: int = 400,   # prefilter cap before distance (regex-pruned)
    prefilter_docs: int = 200, # prefilter cap for docs
    prefilter_audio: int = 200,# prefilter cap for audio
    max_ctx_chars: int = 900   # truncate each context to keep prompts tight
):
    # Check which sources are available
    HAS_GH     = _table_exists(TBL_GH)
    HAS_DOC    = _table_exists(TBL_DOC)
    HAS_AUDIO  = _table_exists(TBL_AUDIO)

    # Require at least one source
    if not (HAS_GH or HAS_DOC or HAS_AUDIO):
        raise RuntimeError(
            "No embeddings table found. Create at least one of:\n"
            f"  {TBL_GH}\n  {TBL_DOC}\n  {TBL_AUDIO}"
        )

    # Build regex and param bundle for the query job
    rx = _build_regex(query)
    k_total = int(top_gh) + int(top_docs) + int(top_audio)

    params = [
        bigquery.ScalarQueryParameter("q",        "STRING", query),
        bigquery.ScalarQueryParameter("rx",       "STRING", rx),
        bigquery.ScalarQueryParameter("kgh",      "INT64",  top_gh),
        bigquery.ScalarQueryParameter("kdocs",    "INT64",  top_docs),
        bigquery.ScalarQueryParameter("kaudio",   "INT64",  top_audio),
        bigquery.ScalarQueryParameter("pf_gh",    "INT64",  prefilter_gh),
        bigquery.ScalarQueryParameter("pf_docs",  "INT64",  prefilter_docs),
        bigquery.ScalarQueryParameter("pf_audio", "INT64",  prefilter_audio),
        bigquery.ScalarQueryParameter("k_total",  "INT64",  k_total),
    ]

    # Manual cosine distance (no VECTOR_DISTANCE):
    # dist = 1 - (dot(a,b) / (||a|| * ||b||))
    # We compute dot and norms via UNNEST joins and aggregates.
    cos_dist_expr = (
        "(\n"
        "  1 - SAFE_DIVIDE(\n"
        "        (\n"
        "          SELECT SUM(v1 * v2)\n"
        "          FROM UNNEST(embedding) AS v1 WITH OFFSET p1\n"
        "          JOIN UNNEST((SELECT emb FROM q)) AS v2 WITH OFFSET p2\n"
        "          ON p1 = p2\n"
        "        ),\n"
        "        NULLIF(\n"
        "          (\n"
        "            SQRT((SELECT SUM(x*x) FROM UNNEST(embedding) AS x))\n"
        "            *\n"
        "            SQRT((SELECT SUM(y*y) FROM UNNEST((SELECT emb FROM q)) AS y))\n"
        "          ), 0)\n"
        "      )\n"
        ") AS dist"
    )

    ctes = []
    #  Query embedding via TVF:
    #  'q' CTE returns a single embedding array 'emb' for the user query
    ctes.append(
        "q AS (\n"
        f"  SELECT ml_generate_embedding_result AS emb\n"
        f"  FROM ML.GENERATE_EMBEDDING(MODEL `{EMB_MODEL}`, (SELECT @q AS content))\n"
        ")"
    )

    unions = []

    # GitHub source
    if HAS_GH:
        # Pre-filter GH rows by regex and cap to pf_gh to reduce compute
        ctes.append(
            "gh_pref AS (\n"
            f"  SELECT CAST(id AS STRING) AS id,\n"
            f"         CAST(type AS STRING) AS type,\n"
            f"         CAST(created_at AS TIMESTAMP) AS created_at,\n"
            f"         text, embedding\n"
            f"  FROM `{TBL_GH}`\n"
            f"  WHERE embedding IS NOT NULL\n"
            f"    AND REGEXP_CONTAINS(LOWER(text), @rx)\n"
            f"  LIMIT @pf_gh\n"
            ")"
        )
        # Score the prefetched GH rows with cosine distance and keep top_k
        ctes.append(
            "gh AS (\n"
            "  SELECT 'github' AS source,\n"
            "         id, type, created_at,\n"
            "         text AS context,\n"
            f"         {cos_dist_expr}\n"
            "  FROM gh_pref\n"
            "  ORDER BY dist ASC\n"
            "  LIMIT @kgh\n"
            ")"
        )
        unions.append("SELECT * FROM gh")

    #  Docs (PDF) source
    if HAS_DOC:
        # Pre-filter Docs by regex and cap to pf_docs
        ctes.append(
            "docs_pref AS (\n"
            f"  SELECT CAST(file_name AS STRING) AS id,\n"
            f"         text, embedding\n"
            f"  FROM `{TBL_DOC}`\n"
            f"  WHERE embedding IS NOT NULL\n"
            f"    AND REGEXP_CONTAINS(LOWER(text), @rx)\n"
            f"  LIMIT @pf_docs\n"
            ")"
        )
        # Score the prefetched Docs rows and keep top_k
        ctes.append(
            "docs AS (\n"
            "  SELECT 'doc' AS source,\n"
            "         id,\n"
            "         CAST(NULL AS STRING) AS type,\n"
            "         CAST(NULL AS TIMESTAMP) AS created_at,\n"
            "         text AS context,\n"
            f"         {cos_dist_expr}\n"
            "  FROM docs_pref\n"
            "  ORDER BY dist ASC\n"
            "  LIMIT @kdocs\n"
            ")"
        )
        unions.append("SELECT * FROM docs")

    # YouTube transcripts source 
    if HAS_AUDIO:
        # Pre-filter audio transcripts by regex and cap to pf_audio
        ctes.append(
            "audio_pref AS (\n"
            f"  SELECT CAST(file_name AS STRING) AS id,\n"
            f"         text, embedding\n"
            f"  FROM `{TBL_AUDIO}`\n"
            f"  WHERE embedding IS NOT NULL\n"
            f"    AND REGEXP_CONTAINS(LOWER(text), @rx)\n"
            f"  LIMIT @pf_audio\n"
            ")"
        )
        # Score the prefetched Audio rows and keep top_k
        ctes.append(
            "audio AS (\n"
            "  SELECT 'audio' AS source,\n"
            "         id,\n"
            "         CAST(NULL AS STRING) AS type,\n"
            "         CAST(NULL AS TIMESTAMP) AS created_at,\n"
            "         text AS context,\n"
            f"         {cos_dist_expr}\n"
            "  FROM audio_pref\n"
            "  ORDER BY dist ASC\n"
            "  LIMIT @kaudio\n"
            ")"
        )
        unions.append("SELECT * FROM audio")

    # Assemble the CTEs and union the enabled sources
    ctes_sql   = ",\n".join(ctes)
    unions_sql = " UNION ALL ".join(unions) if unions else "SELECT NULL WHERE FALSE"

    # Final retrieval: order by distance across sources and cap by @k_total
    sql_ctx = (
        "WITH\n" + ctes_sql + "\n"
        "SELECT * FROM (\n" + unions_sql + "\n"
        ")\nORDER BY dist ASC\n"
        "LIMIT @k_total"
    )

    # Execute retrieval query and materialize as DataFrame
    ctx_df = client.query(
        sql_ctx,
        job_config=bigquery.QueryJobConfig(query_parameters=params),
        location=location
    ).result().to_dataframe()

    # Compact grounding text: header (source/type/time) + truncated context
    def row_to_sig(r):
        head = f"{r.get('source','')} {(r.get('type') or '')} {(str(r.get('created_at')) if r.get('created_at') else '')}".strip()
        txt  = (r.get('context') or '')[:max_ctx_chars]
        return head + "\n" + txt

    signals = "\n---\n".join(row_to_sig(r) for _, r in ctx_df.iterrows())

    # Strict JSON schema hint for the generator (keeps output parseable)
    schema_hint = '{"area": "string", "rationale": "string", "test_cases": ["string", "string"]}.'
    prompt = (
        "You are a senior SDET. Based on the following project signals "
        "(GitHub events, Kubernetes PDFs, and YouTube transcripts), propose focused test areas and two concrete test cases per area.\n"
        "Return STRICT JSON (array of objects): " + schema_hint + "\n"
        "Keep items concise and actionable.\n\n"
        'User request: "' + query + '"\n\nSignals:\n' + signals + "\n"
    )

    # Call AI.GENERATE from BigQuery to produce JSON text as a single cell
    sql_gen = (
        "SELECT CAST(\n"
        "  AI.GENERATE(\n"
        "    @p,\n"
        f"    connection_id => '{BQ_GENAI_CONNECTION}',\n"
        f"    endpoint      => '{GEN_TEXT_ENDPOINT}'\n"
        "  ).result AS STRING\n"
        ") AS json_text"
    )
    gen_df = client.query(
        sql_gen,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("p","STRING", prompt)]
        ),
        location=location
    ).result().to_dataframe()

    raw = gen_df.iloc[0,0] if len(gen_df) else "[]"

    # Robust JSON parse:
    #   - try direct json.loads
    #   - fallback: extract first top-level array via regex
    def parse_json_maybe(text: str):
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.S)
            if m:
                try: return json.loads(m.group(1))
                except Exception: pass
            return None

    items = parse_json_maybe(raw) or []
    rows = []
    if isinstance(items, dict):
        # Some models return {"items":[...]} â€” handle that shape
        items = items.get("items", [])
    if isinstance(items, list) and items:
        # Normalize to tidy rows
        for it in items:
            tests = it.get("test_cases") or [None, None]
            rows.append({
                "area": it.get("area"),
                "rationale": it.get("rationale"),
                "test_case_1": tests[0] if len(tests) > 0 else None,
                "test_case_2": tests[1] if len(tests) > 1 else None,
            })
    else:
        # If parsing fails, return the raw LLM output for inspection
        rows.append({"area": "LLM output (raw)", "rationale": raw, "test_case_1": None, "test_case_2": None})

    out_df = pd.DataFrame(rows)
    return ctx_df, out_df


#==============  Example run Test Focus Advisor ===========. 
qtxt = "Focus testing for CSIPersistentVolumeSource based on changes in Kubernetes 1.17"

# Propose Test focus based on top 5 rank search from embedded GitHub events , k8s documents and meeting transcripts
ctx_df, tests_df = propose_tests(qtxt, top_gh=5, top_docs=5)  # Change the query string and top ranking as needed.

# Show TOP MATCHES â€” IDs only
display(Markdown("### âœ…**Top matches of the question** from GitHub events, Product document, and Meetings"))
cols = [c for c in ["source", "id", "type", "created_at"] if c in ctx_df.columns]
display(
    ctx_df.loc[:, cols]
          .drop_duplicates()
          .reset_index(drop=True)
          .style.hide(axis="index")
          .set_properties(**{"text-align": "left"})
)

# Show proposed tests
display(Markdown(f"### âœ… **Proposed test areas & cases** - created by AI.GENERATE. \n\n **Question**: {qtxt}. \n\n **Analysis**:"))
show_left(tests_df)


# ================== QualityNexus: DORA-style KPI Dashboard (Kubernetes 2019) ==================
# Builds 4 proxies from GitHub events and renders a single HTML dashboard.
YEAR = 2019
SRC = f"{project}.{DATASET}.gh_month_2019_k8s"
DASH_PATH = "/kaggle/working/qualitynexus_kpi_dashboard.html"

def q(sql: str) -> pd.DataFrame:
    return client.query(sql, location=location).result().to_dataframe()

# --- 1) Lead time for changes (PR opened â†’ merged), monthly p50 hours ----------
sql_leadtime_month = f"""
WITH opened AS (
  SELECT CAST(JSON_VALUE(payload,'$.pull_request.number') AS INT64) pr, MIN(created_at) opened_at
  FROM `{SRC}`
  WHERE type='PullRequestEvent' AND JSON_VALUE(payload,'$.action')='opened'
    AND EXTRACT(YEAR FROM created_at)={YEAR}
  GROUP BY pr
),
merged AS (
  SELECT CAST(JSON_VALUE(payload,'$.pull_request.number') AS INT64) pr, MIN(created_at) merged_at
  FROM `{SRC}`
  WHERE type='PullRequestEvent' AND JSON_VALUE(payload,'$.action')='closed'
    AND JSON_VALUE(payload,'$.pull_request.merged')='true'
    AND EXTRACT(YEAR FROM created_at)={YEAR}
  GROUP BY pr
),
lt AS (
  SELECT DATE_TRUNC(m.merged_at, MONTH) month,
         TIMESTAMP_DIFF(m.merged_at, o.opened_at, HOUR) lead_time_h
  FROM merged m JOIN opened o USING(pr)
  WHERE m.merged_at >= o.opened_at
)
SELECT month,
       APPROX_QUANTILES(lead_time_h,100)[OFFSET(50)] AS p50_lead_hours,
       COUNT(*) AS merged_prs
FROM lt
GROUP BY month
ORDER BY month
"""
lead_df = q(sql_leadtime_month)
lead_df["month"] = pd.to_datetime(lead_df["month"])

# --- 2) Deployment frequency: releases per month --------------------------------
sql_releases_month = f"""
SELECT DATE_TRUNC(created_at, MONTH) AS month, COUNT(*) AS releases
FROM `{SRC}`
WHERE type='ReleaseEvent'
  AND JSON_VALUE(payload,'$.action')='published'
  AND EXTRACT(YEAR FROM created_at)={YEAR}
GROUP BY month
ORDER BY month
"""
rel_df = q(sql_releases_month)
rel_df["month"] = pd.to_datetime(rel_df["month"])

# --- 3) Change fail rate (proxy via revert commits) per month -------------------
sql_failrate_month = f"""
WITH merged AS (
  SELECT DATE_TRUNC(created_at, MONTH) month, COUNT(*) merged_prs
  FROM `{SRC}`
  WHERE type='PullRequestEvent'
    AND JSON_VALUE(payload,'$.action')='closed'
    AND JSON_VALUE(payload,'$.pull_request.merged')='true'
    AND EXTRACT(YEAR FROM created_at)={YEAR}
  GROUP BY month
),
reverts AS (
  SELECT DATE_TRUNC(created_at, MONTH) month, COUNT(*) revert_commits
  FROM `{SRC}`,
       UNNEST(JSON_EXTRACT_ARRAY(payload, '$.commits')) c
  WHERE type='PushEvent'
    AND EXTRACT(YEAR FROM created_at)={YEAR}
    AND REGEXP_CONTAINS(LOWER(JSON_VALUE(c,'$.message')), r'\\brevert\\b')
  GROUP BY month
)
SELECT m.month,
       m.merged_prs,
       IFNULL(r.revert_commits,0) AS revert_commits,
       SAFE_DIVIDE(IFNULL(r.revert_commits,0), NULLIF(m.merged_prs,0)) AS fail_rate
FROM merged m
LEFT JOIN reverts r USING (month)
ORDER BY month
"""
fail_df = q(sql_failrate_month)
fail_df["month"] = pd.to_datetime(fail_df["month"])

# --- 4) MTTR proxy: median hours to next release <= 7 days, per month ----------
sql_mttr_month = f"""
WITH rel AS (
  SELECT created_at ts
  FROM `{SRC}`
  WHERE type='ReleaseEvent'
    AND JSON_VALUE(payload,'$.action')='published'
    AND EXTRACT(YEAR FROM created_at)={YEAR}
),
seq AS (
  SELECT ts,
         LEAD(ts) OVER(ORDER BY ts) AS next_ts,
         TIMESTAMP_DIFF(LEAD(ts) OVER(ORDER BY ts), ts, HOUR) AS hrs_to_next
  FROM rel
),
hotfix AS (
  SELECT DATE_TRUNC(ts, MONTH) AS month, hrs_to_next
  FROM seq
  WHERE hrs_to_next IS NOT NULL AND hrs_to_next <= 24*7
)
SELECT month,
       APPROX_QUANTILES(hrs_to_next,100)[OFFSET(50)] AS mttr_p50_hours,
       COUNT(*) AS hotfix_pairs
FROM hotfix
GROUP BY month
ORDER BY month
"""
mttr_df = q(sql_mttr_month)
mttr_df["month"] = pd.to_datetime(mttr_df["month"])


# Normalize month column in each DF to naive (no timezone) month-start
for d in (lead_df, rel_df, fail_df,mttr_df):
    d["month"] = (
        pd.to_datetime(d["month"], utc=True)   # ensure tz-aware uniformly
          .dt.tz_localize(None)                # drop timezone => naive
          .dt.to_period("M").dt.to_timestamp() # month-start
    )

# --- Merge month scaffolding so empty months still render -----------------------
months = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-01", freq="MS")
scaf = pd.DataFrame({"month": months})
lead_df = scaf.merge(lead_df, on="month", how="left")
rel_df  = scaf.merge(rel_df,  on="month", how="left")
fail_df = scaf.merge(fail_df, on="month", how="left")
mttr_df = scaf.merge(mttr_df, on="month", how="left")

# === Build charts ===============================================================
def fig_lead():
    f = go.Figure()
    f.add_trace(go.Scatter(
        x=lead_df["month"], y=lead_df["p50_lead_hours"],
        mode="lines+markers", name="Lead time p50 (hrs)",
        hovertemplate="%{x|%b %Y}<br>p50: %{y:.0f}h<extra></extra>"
    ))
    f.update_layout(
        title="Lead Time for Changes (median hours)",
        xaxis_title="Month", yaxis_title="Hours",
        template="plotly_white", height=320, margin=dict(l=60,r=20,t=60,b=50)
    )
    return f

def fig_releases():
    f = go.Figure()
    f.add_trace(go.Bar(
        x=rel_df["month"], y=rel_df["releases"],
        name="Releases / month",
        hovertemplate="%{x|%b %Y}<br>Releases: %{y}<extra></extra>"
    ))
    f.update_layout(
        title="Deployment Frequency (releases per month)",
        xaxis_title="Month", yaxis_title="Count",
        template="plotly_white", height=320, margin=dict(l=60,r=20,t=60,b=50)
    )
    return f

def fig_failrate():
    f = go.Figure()
    f.add_trace(go.Scatter(
        x=fail_df["month"], y=(fail_df["fail_rate"]*100.0),
        mode="lines+markers", name="Revert / Merge %",
        hovertemplate="%{x|%b %Y}<br>Fail rate: %{y:.2f}%<extra></extra>"
    ))
    f.update_layout(
        title="Change Fail Rate (proxy: revert commits / merged PRs)",
        xaxis_title="Month", yaxis_title="Percent",
        template="plotly_white", height=320, margin=dict(l=60,r=20,t=60,b=50)
    )
    return f

def fig_mttr():
    f = go.Figure()
    f.add_trace(go.Scatter(
        x=mttr_df["month"], y=mttr_df["mttr_p50_hours"],
        mode="lines+markers", name="MTTR p50 (hrs)",
        hovertemplate="%{x|%b %Y}<br>p50: %{y:.0f}h<extra></extra>"
    ))
    f.update_layout(
        title="Failed Deployment Recovery Time (proxy MTTR, hotfixâ‰¤7d)",
        xaxis_title="Month", yaxis_title="Hours",
        template="plotly_white", height=320, margin=dict(l=60,r=20,t=60,b=50)
    )
    return f

f1, f2, f3, f4 = fig_lead(), fig_releases(), fig_failrate(), fig_mttr()

# --- Headline cards (year aggregates) ------------------------------------------
# Lead time overall p50 from per-PR distribution (approx via monthly weighted median is complex),
# so we show simple year medians of month p50s, which is still useful at-a-glance.
headline = {
    "Lead p50 (hrs)": f"{lead_df['p50_lead_hours'].median():.0f}" if lead_df['p50_lead_hours'].notna().any() else "â€”",
    "Releases": int(rel_df['releases'].sum()) if rel_df['releases'].notna().any() else 0,
    "Fail rate avg (%)": f"{(fail_df['fail_rate'].mean()*100.0):.2f}" if fail_df['fail_rate'].notna().any() else "â€”",
    "MTTR p50 (hrs)": f"{mttr_df['mttr_p50_hours'].median():.0f}" if mttr_df['mttr_p50_hours'].notna().any() else "â€”",
}

# === Compose one HTML page with all charts =====================================
def to_div(fig):
    return fig.to_html(include_plotlyjs='cdn', full_html=False, config={"displayModeBar": True})

html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>QualityNexus KPI Dashboard {YEAR}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 16px; }}
  h1 {{ margin: 0 0 10px 0; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .cards {{ display:flex; gap:16px; margin:12px 0 20px 0; flex-wrap: wrap; }}
  .card  {{ border:1px solid #eee; border-radius:12px; padding:12px 16px; min-width:180px; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
  .k  {{ font-size:12px; color:#666; }}
  .v  {{ font-size:24px; font-weight:600; margin-top:4px; }}
</style>
</head>
<body>
  <h1>QualityNexus KPI Dashboard â€” {YEAR}</h1>
  <div class="cards">
    <div class="card"><div class="k">Lead time (median hrs)</div><div class="v">{headline['Lead p50 (hrs)']}</div></div>
    <div class="card"><div class="k">Releases (total)</div><div class="v">{headline['Releases']}</div></div>
    <div class="card"><div class="k">Change fail rate (avg %)</div><div class="v">{headline['Fail rate avg (%)']}</div></div>
    <div class="card"><div class="k">MTTR (median hrs)</div><div class="v">{headline['MTTR p50 (hrs)']}</div></div>
  </div>
  <div class="grid">
    <div>{to_div(f1)}</div>
    <div>{to_div(f2)}</div>
    <div>{to_div(f3)}</div>
    <div>{to_div(f4)}</div>
  </div>
  <p class="k">Notes: These are open-source proxies for DORA metrics (e.g., revertsâ‰ˆchange fail rate; hotfixâ‰¤7dâ‰ˆMTTR).</p>
</body>
</html>
"""

with open(DASH_PATH, "w", encoding="utf-8") as f:
    f.write(html)

# Simple card HTML from your dict (label -> value)
def _card(label, value):
    return f"""
    <div style="flex:1 1 220px; padding:14px 16px; border-radius:12px; background:#fff;
                box-shadow:0 4px 12px rgba(0,0,0,.06)">
      <div style="font-size:12px; color:#6b7280">{label}</div>
      <div style="font-size:28px; font-weight:600">{value}</div>
    </div>"""

cards = "\n".join(_card(k, v) for k, v in headline.items())

cards_html = f"""
<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
            margin:16px 16px 4px 16px">
  <div style="display:flex; gap:12px; flex-wrap:wrap">
    {cards}
  </div>
</div>
"""


# #==================Display the cards and dashboard inline in the notebook ==================
display(Markdown("### âœ… **DORA-Style KPI Dashboard**"))

# card
display(HTML(cards_html))


# Build ONE self-contained HTML string with plotly.js inlined
first = pio.to_html(f1, full_html=False, include_plotlyjs="inline")
rest  = "".join(pio.to_html(ff, full_html=False, include_plotlyjs=False) for ff in [f2, f3, f4])
dashboard_html = first + rest

# (Optional) also write to file for download
with open(DASH_PATH, "w", encoding="utf-8") as f:
    f.write(dashboard_html)

# Show via iframe srcdoc so it renders in both Edit and Commit views
iframe = f"""
<iframe
  srcdoc="{_html.escape(dashboard_html)}"
  style="width:100%; height:1100px; border:0;"
  loading="lazy">
</iframe>
"""
display(HTML(iframe))



# ================== Save KPI charts as PNGs using Matplotlib (works in Kaggle Commit) ==================
IMG_DIR = Path("/kaggle/working/kpi_images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

def _style(ax, title, ylab):
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for label in ax.get_xticklabels():
        label.set_rotation(0)

# 1) Lead time
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.plot(lead_df["month"], lead_df["p50_lead_hours"], marker="o")
_style(ax, "Lead Time for Changes (median hours)", "Hours")
fig.tight_layout()
fig.savefig(IMG_DIR / "lead_time.png")
plt.close(fig)

# 2) Releases per month
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.bar(rel_df["month"], rel_df["releases"], width=20)  # ~month bar
_style(ax, "Deployment Frequency (releases per month)", "Count")
fig.tight_layout()
fig.savefig(IMG_DIR / "releases.png")
plt.close(fig)

# 3) Change fail rate
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.plot(fail_df["month"], fail_df["fail_rate"] * 100.0, marker="o")
_style(ax, "Change Fail Rate (reverts / merged PRs)", "Percent")
fig.tight_layout()
fig.savefig(IMG_DIR / "fail_rate.png")
plt.close(fig)

# 4) MTTR proxy
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.plot(mttr_df["month"], mttr_df["mttr_p50_hours"], marker="o")
_style(ax, "Failed Deployment Recovery Time (proxy MTTR, hotfixâ‰¤7d)", "Hours")
fig.tight_layout()
fig.savefig(IMG_DIR / "mttr.png")
plt.close(fig)

# (Optional) Headline cards as a simple image
fig, ax = plt.subplots(figsize=(10, 2.4), dpi=150)
ax.axis("off")
keys = list(headline.keys())
vals = [str(headline[k]) for k in keys]
for i, (k, v) in enumerate(zip(keys, vals)):
    x0 = 0.02 + i * 0.24
    ax.add_patch(plt.Rectangle((x0 - 0.01, 0.05), 0.22, 0.9, fill=False, lw=1, ec="#e5e7eb", transform=ax.transAxes))
    ax.text(x0, 0.65, k, fontsize=10, color="#6b7280", transform=ax.transAxes)
    ax.text(x0, 0.20, v, fontsize=18, fontweight="bold", transform=ax.transAxes)
fig.tight_layout()
fig.savefig(IMG_DIR / "headline_cards.png", bbox_inches="tight")
plt.close(fig)

list(IMG_DIR.glob("*.png"))


# ================== Upload images to GCS and create Object Table in BigQuery , followed by Inspection==================
import mimetypes
from google.api_core.exceptions import NotFound, Forbidden
# ---- set these ----
BUCKET       = "kagglempqa"
IMAGE_PREFIX = "kpi/images/k8s"   # final path: gs://BUCKET/kpi/images/k8s/<file>.png
LOCAL_ROOT   = Path("/kaggle/working")  # where your PNGs are saved
OVERWRITE    = True

# 1) Find PNGs
pngs = sorted({p.resolve() for p in LOCAL_ROOT.rglob("*.png")})
if not pngs:
    print(f"âš ï¸�  No PNGs found under {LOCAL_ROOT}")
else:
    print(f"Found {len(pngs)} PNG(s). First 5:")
    for p in pngs[:5]: print(" â€¢", p)

    # 2) GCS client
    try:
        st = storage.Client(project=project, credentials=creds)  # Kaggle env
    except NameError:
        st = storage.Client(project=project)

    try:
        bucket = st.bucket(BUCKET)
        bucket.reload()
    except NotFound:
        raise RuntimeError(f"Bucket not found: gs://{BUCKET}")
    except Forbidden:
        raise RuntimeError(f"Forbidden: no access to gs://{BUCKET}")

    # 3) Upload
    if not IMAGE_PREFIX.endswith("/"): IMAGE_PREFIX += "/"
    uploaded = 0
    for fp in pngs:
        dest = IMAGE_PREFIX + fp.name
        blob = bucket.blob(dest)
        if not OVERWRITE:
            try:
                blob.reload(); print("â�­ï¸�  exists:", f"gs://{BUCKET}/{dest}"); continue
            except NotFound:
                pass
        ctype = mimetypes.guess_type(fp.name)[0] or "image/png"
        blob.upload_from_filename(str(fp), content_type=ctype)
        uploaded += 1
        print("uploaded:", f"gs://{BUCKET}/{dest}")

    print(f"âœ… Uploaded {uploaded} image(s) to gs://{BUCKET}/{IMAGE_PREFIX}")

    # 4) Quick listing to verify
    objs = list(st.list_blobs(BUCKET, prefix=IMAGE_PREFIX))
    print(f"ğŸ“¦ Objects now under gs://{BUCKET}/{IMAGE_PREFIX} : {len(objs)}")
    for b in objs[:10]:
        print(f" â€¢ {b.name} ({b.size} bytes)")


# Create an Object Table over the images (for ObjectRef)
sql_ot = f"""
CREATE OR REPLACE EXTERNAL TABLE `{project}.{DATASET}.kpi_img_ot`
WITH CONNECTION `{location.lower()}.genai`
OPTIONS (
  object_metadata = 'SIMPLE',
  uris = ['gs://{BUCKET}/{IMAGE_PREFIX}*.png']
);
"""
bq(sql_ot)
print("âœ… Object table created:", f"{project}.{DATASET}.kpi_img_ot")


# Inspect object table of image
warnings.filterwarnings("ignore",
    message="Unable to determine type for field 'access_url'",
    module="google.cloud.bigquery._pandas_helpers")

display(Markdown("### âœ… **Preview ObjectTable and ObjectRef** of images - created by AI.GENERATE_TABLE"))
# 0) Assert the table exists
exists = client.query(
    f"""
    SELECT 1
    FROM `{project}.{DATASET}.INFORMATION_SCHEMA.TABLES`
    WHERE table_name = 'kpi_img_ot'
    LIMIT 1
    """,
    location=location
).result().to_dataframe()
#assert not exists.empty, f"Missing table: {project}.{DATASET}.kpi_img_ot"

# 1) Show schema (column names & types) - Optional
schema = client.query(
    f"""
    SELECT column_name, data_type, is_nullable
    FROM `{project}.{DATASET}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'kpi_img_ot'
    ORDER BY ordinal_position
    """,
    location=location
).result().to_dataframe()
#print("Schema:")
#display(schema.style.hide(axis="index").set_properties(**{"text-align":"left"}))

# 2) Row counts / image content types
counts = client.query(
    f"""
    SELECT
      COUNT(*) AS rows_total,
      COUNTIF(STARTS_WITH(content_type, 'image/')) AS rows_image,
      COUNTIF(content_type IS NULL) AS rows_ct_null
    FROM `{project}.{DATASET}.kpi_img_ot`
    """,
    location=location
).result().to_dataframe()
print("\nRow counts:")
display(counts)

types = client.query(
    f"""
    SELECT content_type, COUNT(*) AS n
    FROM `{project}.{DATASET}.kpi_img_ot`
    GROUP BY content_type
    ORDER BY n DESC NULLS LAST
    """,
    location=location
).result().to_dataframe()
print("\nTop content types:")
display(types.style.hide(axis="index").set_properties(**{"text-align":"left"}))

# 3) Sample a few image rows (file name, uri, ref fields, size)
sample = client.query(
    f"""
    SELECT
      _FILE_NAME AS file_name,
      content_type,
      size AS size_bytes,
      uri,
      ref.uri     AS ref_uri,
      ref.version AS ref_version
    FROM `{project}.{DATASET}.kpi_img_ot`
    WHERE STARTS_WITH(content_type,'image/')
    ORDER BY _FILE_NAME
    LIMIT 5
    """,
    location=location
).result().to_dataframe()
print("\nSample image rows:")
display(sample.style.hide(axis="index").set_properties(**{"text-align":"left"}))

# 4) Test ObjectRef â†’ signed access URL 
try:
    url_df = client.query(
        f"""
        SELECT OBJ.GET_ACCESS_URL(ref, 'r') AS access_url
        FROM `{project}.{DATASET}.kpi_img_ot`
        WHERE STARTS_WITH(content_type,'image/')
        LIMIT 3
        """,
        location=location
    ).result().to_dataframe(dtypes={"access_url": pd.StringDtype()})
    print("\nOBJ.GET_ACCESS_URL(ref,'r') test (first 3):")
    display(show_left(url_df))
except Exception as e:
    print("âš ï¸�  OBJ.GET_ACCESS_URL(ref,'r') failed â€”")
    print("   Grant **Storage Object Viewer** (roles/storage.objectViewer) to your BigQuery *connection* service account on the bucket.")
    print("Error:", e)


# ================== Multimodal image â†’ structured insights (no stats table involved) ==================
# Ensure the remote text model exists (safe to re-run)
bq(f"""
CREATE OR REPLACE MODEL `{project}.{DATASET}.gen_text_model`
  REMOTE WITH CONNECTION `{location.lower()}.genai`
  OPTIONS (endpoint = '{GEN_TEXT_ENDPOINT}');
""")
print("âœ… Remote model ready â†’", f"{project}.{DATASET}.gen_text_model")

# Multimodal KPI images â†’ (insight, action) using only literal settings

# Build image insights as a single 'summary' string (Insight + Action bullets)
# Multimodal KPI images â†’ (insight, action) using only literal settings
sql_insights = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.kpi_img_insights` AS
SELECT
  file_name,
  summary
FROM AI.GENERATE_TABLE(
  MODEL `{project}.{DATASET}.gen_text_model`,
  (
    SELECT
      (
        'You are a release quality analyst.',
        'Look at the KPI chart image and produce ONE field named "summary".',
        'Return EXACTLY two bullets on separate lines:',
        '- Insight: A few concise sentences about what the chart shows, including highlights and trends.(<=150 words)',
        '- Action: A few imperative next steps (<=50 words).',
        'No extra prose. If uncertain, still suggest a concrete Action.',
        OBJ.GET_ACCESS_URL(ref, 'r')
      ) AS prompt,
      REGEXP_EXTRACT(uri, r'/([^/]+)$') AS file_name
    FROM `{project}.{DATASET}.kpi_img_ot`
    WHERE STARTS_WITH(content_type,'image/')
  ),
  STRUCT('summary STRING' AS output_schema)  -- single merged field
);
"""
bq(sql_insights)
print("âœ… Built:", f"{project}.{DATASET}.kpi_img_insights")


# converged summary based on the images in the kpi_img_insight, so we have a horizon view of the KPI
#  sanity check
rowcheck = client.query(
    f"SELECT COUNT(*) AS n FROM `{project}.{DATASET}.kpi_img_insights`",
    location=location
).result().to_dataframe()
#assert int(rowcheck.iloc[0,0]) > 0, "kpi_img_insights is empty â€” run the image insight step first."

# Build a converged, year-wide horizon summary from all chart insights
sql_conv_simple = f"""
CREATE OR REPLACE TABLE `{project}.{DATASET}.kpi_img_converged` AS
WITH s AS (
  SELECT file_name, summary
  FROM `{project}.{DATASET}.kpi_img_insights`
  WHERE summary IS NOT NULL AND summary != ''
)
SELECT CAST(
  AI.GENERATE(
    CONCAT(
      'You are a VP of Engineering. Combine the KPI image summaries into one concise report with 4 sections: ',
      'Executive Headline (<=2 sentences); Key Observations (3 bullets); Risks (2 bullets); Recommended Actions (3 bullets). ',
      'Keep <=180 words.',
      CHR(10), CHR(10), 'Per-chart summaries:', CHR(10),
      ARRAY_TO_STRING(
        ARRAY_AGG(CONCAT('- ', file_name, ': ', summary) ORDER BY file_name),
        CHR(10)
      )
    ),
    connection_id => '{BQ_GENAI_CONNECTION}',
    endpoint      => '{GEN_TEXT_ENDPOINT}'
  ).result AS STRING
) AS report
FROM s;
"""
bq(sql_conv_simple)
print("âœ… Built:", f"{project}.{DATASET}.kpi_img_converged")


# Preview the image-driven insights
preview = client.query(
    f"""
    SELECT file_name, summary
    FROM `{project}.{DATASET}.kpi_img_insights`
    ORDER BY file_name
    """,
    location=location
).result().to_dataframe()
display(Markdown("### âœ… **Summary of each DORA-style KPI** based on dashboard images"))
show_left(preview)


# Preview the converged horizon view
preview = client.query(
    f"SELECT report FROM `{project}.{DATASET}.kpi_img_converged`",
    location=location
).result().to_dataframe()
display(Markdown("### âœ… **Comprehensive Executive insights** based on all dashboards"))
show_left(preview)


from IPython.display import Image
display(Image("/kaggle/input/qualitynexus/QualityNexus_BigQueryAssistant.png", width=500))


# ===================== BigQueryAssistant Helpers Function ==========================
# -------- Optional: lightweight logger (creates the table on first use) -------
LOG_TBL = f"{project}.{DATASET}.assistant_query_log"

def _ensure_log_table():
    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{LOG_TBL}` (
      dataset    STRING,
      table_name STRING,
      question   STRING,
      sql_text   STRING,
      summary    STRING,
      ts         TIMESTAMP
    )
    PARTITION BY DATE(ts)
    CLUSTER BY dataset, table_name
    """
    client.query(ddl, location=location).result()

def _log_success(dataset_name: str,
                 table_name: str,
                 question: str,
                 sql_text: str,
                 summary: str):
    _ensure_log_table()
    ins = f"""
      INSERT INTO `{LOG_TBL}` (dataset, table_name, question, sql_text, summary, ts)
      VALUES (@ds, @tbl, @q, @sql, @sum, CURRENT_TIMESTAMP())
    """
    qp = [
        bigquery.ScalarQueryParameter("ds",  "STRING", dataset_name),
        bigquery.ScalarQueryParameter("tbl", "STRING", table_name),
        bigquery.ScalarQueryParameter("q",   "STRING", question),
        bigquery.ScalarQueryParameter("sql", "STRING", (sql_text or "")[:900000]),
        bigquery.ScalarQueryParameter("sum", "STRING", (summary or "")[:900000]),
    ]
    client.query(ins, job_config=bigquery.QueryJobConfig(query_parameters=qp), location=location).result()

# -------------------------- Helpers ------------------------------------------
def _split_fqn(name_or_fqn: str):
    # returns (proj, dataset, table)
    parts = name_or_fqn.split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return project, parts[0], parts[1]
    else:
        return project, DATASET, parts[0]

def _table_exists_fqn(fqn: str) -> bool:
    proj, ds, tbl = _split_fqn(fqn)
    sql = f"""
      SELECT 1
      FROM `{proj}.{ds}.INFORMATION_SCHEMA.TABLES`
      WHERE table_name = @t
      LIMIT 1
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("t","STRING", tbl)])
    df = client.query(sql, job_config=cfg, location=location).result().to_dataframe()
    return not df.empty

def _get_schema_text(fqn: str) -> str:
    proj, ds, tbl = _split_fqn(fqn)
    sql = f"""
      SELECT column_name, data_type
      FROM `{proj}.{ds}.INFORMATION_SCHEMA.COLUMNS`
      WHERE table_name=@t
      ORDER BY ordinal_position
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("t","STRING", tbl)])
    df = client.query(sql, job_config=cfg, location=location).result().to_dataframe()
    return "\n".join([f"- {r.column_name}: {r.data_type}" for _, r in df.iterrows()])

def _ai_generate(prompt: str, connection_id: str, endpoint: str) -> str:
    # Call BigQuery AI.GENERATE and return the text result (STRING)
    q = (
        "SELECT CAST(\n"
        "  AI.GENERATE(\n"
        "    @p,\n"
        f"    connection_id => '{connection_id}',\n"
        f"    endpoint      => '{endpoint}'\n"
        "  ).result AS STRING\n"
        ") AS txt"
    )
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("p","STRING", prompt)])
    df = client.query(q, job_config=cfg, location=location).result().to_dataframe()
    return (df.iloc[0,0] if len(df) else "").strip()

def _extract_year(question: str):
    m = re.search(r"(20\d{2})", question)
    return int(m.group(1)) if m else None

def _safe_head_text(df: pd.DataFrame, limit_chars=4000) -> str:
    # Compact CSV head as prompt context
    if df is None or df.empty:
        return ""
    csv = df.to_csv(index=False)
    if len(csv) > limit_chars:
        csv = csv[:limit_chars] + "\n... (truncated)"
    return csv

# -------------------------- KPI templates (GitHub events) ---------------------
def _sql_lead_time(fqn: str, year: int|None):
    year_clause = f"AND EXTRACT(YEAR FROM created_at)={year}" if year else ""
    return f"""
WITH opened AS (
  SELECT CAST(JSON_VALUE(payload,'$.pull_request.number') AS INT64) pr, MIN(created_at) opened_at
  FROM `{fqn}`
  WHERE type='PullRequestEvent' AND JSON_VALUE(payload,'$.action')='opened' {year_clause}
  GROUP BY pr
),
merged AS (
  SELECT CAST(JSON_VALUE(payload,'$.pull_request.number') AS INT64) pr, MIN(created_at) merged_at
  FROM `{fqn}`
  WHERE type='PullRequestEvent' AND JSON_VALUE(payload,'$.action')='closed'
    AND JSON_VALUE(payload,'$.pull_request.merged')='true' {year_clause}
  GROUP BY pr
),
lt AS (
  SELECT DATE_TRUNC(m.merged_at, MONTH) month,
         TIMESTAMP_DIFF(m.merged_at, o.opened_at, HOUR) lead_time_h
  FROM merged m JOIN opened o USING(pr)
  WHERE m.merged_at >= o.opened_at
)
SELECT month,
       APPROX_QUANTILES(lead_time_h,100)[OFFSET(50)] AS p50_lead_hours,
       COUNT(*) AS merged_prs
FROM lt
GROUP BY month
ORDER BY month
""".strip()

def _sql_deploy_freq(fqn: str, year: int|None):
    year_clause = f"AND EXTRACT(YEAR FROM created_at)={year}" if year else ""
    return f"""
SELECT DATE_TRUNC(created_at, MONTH) AS month, COUNT(*) AS releases
FROM `{fqn}`
WHERE type='ReleaseEvent'
  AND JSON_VALUE(payload,'$.action')='published'
  {year_clause}
GROUP BY month
ORDER BY month
""".strip()

def _sql_issues_opened(fqn: str, year: int|None):
    year_clause = f"AND EXTRACT(YEAR FROM created_at)={year}" if year else ""
    return f"""
SELECT DATE_TRUNC(created_at, MONTH) AS month, COUNT(*) AS issues_opened
FROM `{fqn}`
WHERE type='IssuesEvent'
  AND JSON_VALUE(payload,'$.action')='opened'
  {year_clause}
GROUP BY month
ORDER BY month
""".strip()

def _sql_fail_rate_proxy(fqn: str, year: int|None):
    year_clause = f"AND EXTRACT(YEAR FROM created_at)={year}" if year else ""
    return f"""
WITH merged AS (
  SELECT DATE_TRUNC(created_at, MONTH) month, COUNT(*) merged_prs
  FROM `{fqn}`
  WHERE type='PullRequestEvent'
    AND JSON_VALUE(payload,'$.action')='closed'
    AND JSON_VALUE(payload,'$.pull_request.merged')='true'
    {year_clause}
  GROUP BY month
),
reverts AS (
  SELECT DATE_TRUNC(created_at, MONTH) month, COUNT(*) revert_commits
  FROM `{fqn}`,
       UNNEST(JSON_EXTRACT_ARRAY(payload, '$.commits')) c
  WHERE type='PushEvent'
    {year_clause}
    AND REGEXP_CONTAINS(LOWER(JSON_VALUE(c,'$.message')), r'\\brevert\\b')
  GROUP BY month
)
SELECT m.month,
       m.merged_prs,
       IFNULL(r.revert_commits,0) AS revert_commits,
       SAFE_DIVIDE(IFNULL(r.revert_commits,0), NULLIF(m.merged_prs,0)) AS fail_rate
FROM merged m
LEFT JOIN reverts r USING (month)
ORDER BY month
""".strip()

def _sql_mttr_proxy(fqn: str, year: int|None):
    year_clause = f"AND EXTRACT(YEAR FROM created_at)={year}" if year else ""
    return f"""
WITH rel AS (
  SELECT created_at ts
  FROM `{fqn}`
  WHERE type='ReleaseEvent'
    AND JSON_VALUE(payload,'$.action')='published'
    {year_clause}
),
seq AS (
  SELECT ts,
         LEAD(ts) OVER(ORDER BY ts) AS next_ts,
         TIMESTAMP_DIFF(LEAD(ts) OVER(ORDER BY ts), ts, HOUR) AS hrs_to_next
  FROM rel
),
hotfix AS (
  SELECT DATE_TRUNC(ts, MONTH) AS month, hrs_to_next
  FROM seq
  WHERE hrs_to_next IS NOT NULL AND hrs_to_next <= 24*7
)
SELECT month,
       APPROX_QUANTILES(hrs_to_next,100)[OFFSET(50)] AS mttr_p50_hours,
       COUNT(*) AS hotfix_pairs
FROM hotfix
GROUP BY month
ORDER BY month
""".strip()

def _maybe_template_sql(fqn: str, question: str) -> tuple[str|None, str]:
    ql = question.lower()
    year = _extract_year(ql)
    if any(k in ql for k in ["lead time", "leadtime", "cycle time"]):
        return _sql_lead_time(fqn, year), "lead_time"
    if any(k in ql for k in ["deployment frequency", "deploy frequency", "releases per month", "release frequency"]):
        return _sql_deploy_freq(fqn, year), "deployment_frequency"
    if any(k in ql for k in ["issues", "bug count", "issue volume"]):
        return _sql_issues_opened(fqn, year), "issues_opened"
    if any(k in ql for k in ["fail rate", "failure rate", "change fail"]):
        return _sql_fail_rate_proxy(fqn, year), "fail_rate_proxy"
    if any(k in ql for k in ["mttr", "recovery", "time to restore"]):
        return _sql_mttr_proxy(fqn, year), "mttr_proxy"
    return None, "llm_fallback"

def show_bq_assistant(question: str, res: dict, preview_rows: int = 5):
    """Pretty-print a BigQueryAssistant() result dict.
    Expects keys: 'sql_text' (str), 'df' (pd.DataFrame), 'summary' (str).
    """
    display(Markdown(f"### âœ… **Question processed by BigQuery Assistant**\n**Question:** {question}"))

    # SQL
    sql_text = res.get("sql")
    if sql_text:
        display(Markdown("**Generated SQL:**"))
        display(Markdown(f"```sql\n{sql_text}\n```"))

    # Table
    df = res.get("df")
    display(Markdown(f"### **Generated table (preview):** with top {preview_rows}"))
    if isinstance(df, pd.DataFrame) and not df.empty:
        display(df.head(preview_rows))
    else:
        display(Markdown("_No rows returned._"))

    # Summary
    summary = res.get("summary")
    if summary:
        display(Markdown("**Summary:**"))
        display(Markdown(str(summary)))


# ================== Main function of BigQueryAssistant ==================
def BigQueryAssistant(table_name: str,
                      question: str,
                      *,
                      gen_connection: str = BQ_GENAI_CONNECTION,
                      gen_endpoint: str   = GEN_TEXT_ENDPOINT,
                      do_log: bool = True):
    """
    Natural-language â†’ BigQuery SQL â†’ execute â†’ LLM summary.
    Returns: dict(sql=..., df=DataFrame, summary=..., used_template=True/False, mode=...).
    """
    # Resolve table FQN and basic checks
    proj, ds, tbl = _split_fqn(table_name)
    fqn = f"{proj}.{ds}.{tbl}"
    if not _table_exists_fqn(fqn):
        raise RuntimeError(f"Table not found: `{fqn}` (check project/dataset/table and region)")

    # 1) Try a known KPI template first
    sql_text, mode = _maybe_template_sql(fqn, question)

    # 2) If no template matched, ask LLM to synthesize SQL, using schema as context
    if sql_text is None:
        schema_txt = _get_schema_text(fqn)
        prompt = textwrap.dedent(f"""
        You are a BigQuery expert. Given the table `{fqn}` and the user request below,
        write a complete StandardSQL SELECT query that answers the question.

        - Output ONLY the SQL (no markdown, no backticks, no commentary).
        - Prefer GROUP BY, DATE_TRUNC, SAFE_DIVIDE, APPROX_QUANTILES where helpful.
        - Keep it runnable as a single query (no scripting).
        - If the question suggests a time window (e.g. 2019), filter accordingly.

        Table schema:
        {schema_txt}

        User request: {question}
        """).strip()
        sql_text = _ai_generate(prompt, gen_connection, gen_endpoint)
        # guardrails: strip accidental code fences if present
        sql_text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql_text.strip(), flags=re.I|re.S)
        mode = "llm_sql"

    # 3) Execute SQL
    try:
        df = client.query(sql_text, location=location).result().to_dataframe()
    except Exception as e:
        raise RuntimeError(f"Query failed.\n--- SQL ---\n{sql_text}\n--- ERROR ---\n{e}")

    # 4) Summarize results with AI.GENERATE
    head_txt = _safe_head_text(df, limit_chars=5000)
    sum_prompt = textwrap.dedent(f"""
    You are an engineering program lead. Summarize the table below in 3 crisp bullets and 1 actionable recommendation.
    - Be specific and numeric where possible.
    - â‰¤120 words total.
    - Audience is release/quality leadership.

    User request: {question}

    Data (CSV head):
    {head_txt}
    """).strip()
    summary = _ai_generate(sum_prompt, gen_connection, gen_endpoint)

    # 5) Log success
    if do_log:
        try:
            _log_success(ds, tbl, question, sql_text, summary)
        except Exception:
            pass  # non-fatal

    return {
        "sql": sql_text,
        "df": df,
        "summary": summary,
        "used_template": (mode != "llm_sql"),
        "mode": mode,
    }

# ===================== End BigQueryAssistant ==================================


# =====================  BigQueryAssistant Test Examples ==================================
# Example 1: GitHub events KPI
QUESTION = "Summarize Lead Time for Changes KPI for 2019"
res = BigQueryAssistant(f"{project}.{DATASET}.gh_month_2019_k8s", QUESTION)
show_bq_assistant(QUESTION, res)


# Example2: Natural-language insight over Stack Overflow
QUESTION = "Find top 10 question themes and summarize the biggest pain points in 2019 with one action."
res = BigQueryAssistant(f"{project}.{DATASET}.so_2019_kube_questions", QUESTION)
show_bq_assistant(QUESTION, res, 5)


# Example3: Stackflow Forecast
QUESTION = "Forecast StackOverflow questions for the next quarter and summarize the trend."
res = BigQueryAssistant(f"{project}.{DATASET}.k8s_quality_monthly_2019", QUESTION)
show_bq_assistant(QUESTION, res, 5)

