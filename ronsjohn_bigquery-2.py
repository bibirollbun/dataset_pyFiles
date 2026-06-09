# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os, json, pandas as pd, numpy as np
from pathlib import Path


pd.set_option("display.max_colwidth", None)  # show full strings
# (optional) widen the console table
pd.set_option("display.width", 0)
pd.set_option("display.max_columns", None)

from IPython.display import display


# =========================
# Modes
# =========================
RUN_LIVE = False        # Judges: leave False. Set True only if you will run live against your GCP.


# =========================
# Configuration
# =========================
PROJECT_ID = "gen-lang-client-0774611257"         # Only needed if RUN_LIVE=True (e.g., "my-gcp-project")
BQ_LOCATION = "US"      # Keep EU for consistency with your dataset & connection
DATASET = "news_ai_us"     # BigQuery dataset name (created if missing, in LIVE mode)

# Models/endpoints
# GEMINI_MODEL = "gemini-2.0-flash-001"          # For summarizing / final answer
GEMINI_MODEL = "gemini-2.5-flash-lite"
# EMBEDDING_ENDPOINT = "text-multilingual-embedding-002"  # For ML.GENERATE_EMBEDDING
EMBEDDING_ENDPOINT = "gemini-embedding-001"
# BQ_CONNECTION = "us.us_llm_conn"                  # BigQuery connection to Vertex AI (pre-created)

# REPRO fixtures path (attach your dataset in Kaggle "Add data")
FIXTURES_PATH = "/kaggle/input/repro-files-1"
# FIXTURES_PATH = Path("/kaggle/working/fixtures") # temp for testing
FIXTURES_OUT = Path("/kaggle/working/fixtures")
FIXTURES_OUT.mkdir(parents=True, exist_ok=True)

# Limits (keep costs low)
URL_LIMIT = 100           # how many URLs to fetch from GDELT (LIVE)
EXTRACT_LIMIT = 100       # how many URLs to send to Gemini (LIVE)
TOP_K = 20                # neighbors for vector search
print('Done.')



def running_in_kaggle():
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""

def load_fixture(name):
    p = Path(FIXTURES_PATH)/name
    assert p.exists(), f"Fixture not found: {p}"
    print(pd.read_parquet(p))
    return pd.read_parquet(p)

def df_head(df, n=7):
    # pretty head
    display(df.head(n))


print("Mode:", "LIVE" if RUN_LIVE else "REPRO (fixtures)")



def load_fixture(name: str) -> pd.DataFrame:
    p = Path(FIXTURES_PATH) / name
    if not p.exists():
        raise FileNotFoundError(f"Fixture missing: {p}")
    return pd.read_parquet(p)

def save_fixture(df: pd.DataFrame, name: str):
    p = FIXTURES_OUT / name
    df.to_parquet(p, index=False)
    print("Saved fixture:", p)

def bq_to_df(table: str, cols="*", where: str | None=None, limit: int | None=None) -> pd.DataFrame:
    q = f"SELECT {cols} FROM `{PROJECT_ID}.{DATASET}.{table}`"
    if where:
        q += f" WHERE {where}"
    if limit:
        q += f" LIMIT {limit}"
    return bq.query(q).result().to_dataframe()



if RUN_LIVE:
    from google.cloud import bigquery
    assert PROJECT_ID, "Set PROJECT_ID to your GCP project id"
    bq = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)
    print("BigQuery client ready:", PROJECT_ID, BQ_LOCATION)

    # Ensure dataset exists
    ds_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET}")
    ds_ref.location = BQ_LOCATION
    try:
        bq.get_dataset(ds_ref)
        print("Dataset exists:", f"{PROJECT_ID}.{DATASET}")
    except Exception:
        bq.create_dataset(ds_ref, exists_ok=True)
        print("Created dataset:", f"{PROJECT_ID}.{DATASET}")
else:
    print("Skipping BigQuery client (REPRO mode).")



if RUN_LIVE:
    sql = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.url_window` AS
    SELECT MIN(GLOBALEVENTID) AS GLOBALEVENTID, SOURCEURL AS url
    FROM `gdelt-bq.gdeltv2.events_partitioned`
    WHERE _PARTITIONTIME >= TIMESTAMP('2025-08-23 00:00:00+00')
      AND _PARTITIONTIME <  TIMESTAMP('2025-08-24 00:00:00+00')
      AND SOURCEURL IS NOT NULL
    GROUP BY SOURCEURL
    LIMIT {URL_LIMIT};
    """
    bq.query(sql).result()
    # url_df = bq.query(f"SELECT GLOBALEVENTID, url FROM `{PROJECT_ID}.{DATASET}.url_window`").result().to_dataframe()
    # save_fixture(url_df, "url_window.parquet")  # Save for REPRO
    try:
        url_df = bq_to_df("url_window")
        save_fixture(url_df, "url_window.parquet")
    except Exception as e:
        print("Could not save url_window fixture:", e)
else:
    url_df = load_fixture("url_window.parquet")
    print("REPRO url_window:", len(url_df))

print("URLs:", len(url_df))
df_head(url_df)



try:
    from kaggle_secrets import UserSecretsClient
    GEMINI_API_KEY = UserSecretsClient().get_secret("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# assert GEMINI_API_KEY, "Provide GEMINI_API_KEY via Kaggle Secrets or env var"


if bool(GEMINI_API_KEY and RUN_LIVE) :  
    print('Yes')
else:
    print('No')


%%time

import os, json, time, random, concurrent.futures as cf
from google import genai
from google.genai import types

USE_GROUNDING = False            # toggle if you want Google Search grounding
MAX_WORKERS   = int(os.getenv("GEMINI_MAX_WORKERS", 8))   # 3–6 is usually safe
MAX_RETRIES   = 1

if bool(GEMINI_API_KEY) and RUN_LIVE:

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Tool (optional)
    grounding_tool = types.Tool(google_search=types.GoogleSearch()) if USE_GROUNDING else None

    # Structured output schema
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "url": types.Schema(type=types.Type.STRING),
            "title": types.Schema(type=types.Type.STRING),
            "summary": types.Schema(type=types.Type.STRING),
            "keywords": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "entities": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "relationships": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "themes": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "sentiment": types.Schema(type=types.Type.STRING),
        },
        required=["url", "title", "summary", "themes"],
    )

    GEMINI_MODEL = GEMINI_MODEL if 'GEMINI_MODEL' in globals() else "gemini-2.0-flash-001"

    # figure out the event id column name robustly
    gid_col = "GLOBALEVENTID" if "GLOBALEVENTID" in url_df.columns else \
              "global_event_id" if "global_event_id" in url_df.columns else None
    if gid_col is None:
        raise ValueError("Could not find a global event id column in url_df")

    def build_prompt(u: str) -> str:
        return f"""
Read the article at the following url
url: {u}

Extract the following structured information from the article. Always respond in English.

**Title:** (mandatory)

**Summary:** (5–7 sentences, mandatory)

**Keywords:** (5–10 relevant keywords; avoid generic words like "news", "article")

**Key Entities:** (organizations, governments, countries, locations, people, prominent figures; one per line; 'None' if none)

**Sentiment:** (one word: Positive, Negative, Neutral, or Mixed)

**Relationships:** (each as [Entity1, Relationship, Entity2]; Relationship 1–2 words max; ensure Entity1 and Entity2 present)

**Themes:** (themes/subjects/topics; one per line; e.g., women's rights, protest, immigration, crime, finance, banking)
""".strip()

    def extract_one(u: str, gid):
        """
        Call Gemini for a single URL with retries + backoff.
        Returns a dict with parsed JSON fields + url + GLOBALEVENTID.
        On final failure, returns a minimal record with _error.
        """
        prompt = build_prompt(u)
        tools = [grounding_tool] if grounding_tool else None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,   # you can also pass [types.Part(text=prompt), types.Part.from_uri(uri=u)]
                    config=types.GenerateContentConfig(
                        tools=tools,
                        response_mime_type="application/json",
                        response_schema=schema,
                        max_output_tokens=768,
                        temperature=0.2,
                    ),
                )
                data = {}
                try:
                    data = json.loads(resp.text or "{}")
                except json.JSONDecodeError:
                    data = {}

                data["url"] = data.get("url") or u
                data["GLOBALEVENTID"] = gid
                return data

            except Exception as e:
                if attempt == MAX_RETRIES:
                    return {"url": u, "GLOBALEVENTID": gid, "_error": str(e)}
                # exponential backoff with jitter
                time.sleep((2 ** attempt) + random.random())

    # Build the worklist
    pairs = list(url_df[["url", gid_col]].head(EXTRACT_LIMIT).itertuples(index=False, name=None))

    rows = []
    errors = 0
    print(f"Submitting {len(pairs)} Gemini requests with max_workers={MAX_WORKERS} ...")
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_to_item = {ex.submit(extract_one, u, gid): (u, gid) for (u, gid) in pairs}
        for fut in cf.as_completed(fut_to_item):
            rec = fut.result()
            if "_error" in rec:
                errors += 1
                # keep the failed record (still useful for audit)
            rows.append(rec)

    gemini_df = pd.DataFrame(rows)
    print(f"Gemini completed: {len(gemini_df)} rows; errors: {errors}")

    # Optional quick clean-up: drop rows with missing mandatory fields
    must_have = ["title", "summary", "themes"]
    if all(col in gemini_df.columns for col in must_have):
        before = len(gemini_df)
        gemini_df = gemini_df.dropna(subset=[c for c in must_have]).reset_index(drop=True)
        print(f"Filtered incomplete rows: {before - len(gemini_df)} removed, {len(gemini_df)} remain.")

    # Save for REPRO
    save_fixture(gemini_df, "gemini_url_extractions.parquet")

    print("Extracted with Gemini:", len(gemini_df))

else:
    print("Loading fixture gemini_url_extractions.parquet.")
    gemini_df = load_fixture("gemini_url_extractions.parquet")

df_head(gemini_df)

#3min 6s for 1000 records
#49.4 sec for 200 records
#1min 26 sec for 400 records
#57 sec for 300 records


if RUN_LIVE:
    from google.cloud import bigquery
    table_id = f"{PROJECT_ID}.{DATASET}.raw_extractions"
    job = bq.load_table_from_dataframe(gemini_df, table_id, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
    job.result()
    print("Loaded structured extractions to:", table_id)
else:
    print("REPRO mode: using in-memory DataFrame for downstream steps.")



if RUN_LIVE:
    # Articles
    bq.query(f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.articles` AS
    SELECT GLOBALEVENTID, url, title, summary FROM `{PROJECT_ID}.{DATASET}.raw_extractions`;
    """).result()

    print(f"Normalized table {PROJECT_ID}.{DATASET}.articles created in BigQuery.")
else:
    print("REPRO mode: keeping normalized info in DataFrames or fixtures.")


if RUN_LIVE:
    # Keywords
    bq.query(f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.article_keywords` AS
    SELECT GLOBALEVENTID, url, kw AS keyword
    FROM `{PROJECT_ID}.{DATASET}.raw_extractions`, UNNEST(keywords) kw;
    """).result()

    print(f"Normalized table {PROJECT_ID}.{DATASET}.article_keywords created in BigQuery.")
else:
    print("REPRO mode: keeping normalized info in DataFrames or fixtures.")


if RUN_LIVE:
    # Entities
    bq.query(f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.article_entities` AS
    SELECT GLOBALEVENTID, url, ent AS entity
    FROM `{PROJECT_ID}.{DATASET}.raw_extractions`, UNNEST(entities) ent;
    """).result()

    print(f"Normalized table {PROJECT_ID}.{DATASET}.article_entities created in BigQuery.")
else:
    print("REPRO mode: keeping normalized info in DataFrames or fixtures.")


if RUN_LIVE:
    # Themes
    bq.query(f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.article_themes` AS
    SELECT GLOBALEVENTID, url, th AS theme
    FROM `{PROJECT_ID}.{DATASET}.raw_extractions`, UNNEST(themes) th;
    """).result()

    print(f"Normalized table {PROJECT_ID}.{DATASET}.article_themes created in BigQuery.")
else:
    print("REPRO mode: keeping normalized info in DataFrames or fixtures.")


if RUN_LIVE:
    # Relationships (assuming strings like "[E1, Rel, E2]"; parse as needed)
    bq.query(f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.article_relationships` AS
    WITH rels AS (
      SELECT GLOBALEVENTID, url, rel FROM `{PROJECT_ID}.{DATASET}.raw_extractions`, UNNEST(relationships) rel
    )
    SELECT GLOBALEVENTID, url, rel AS relation
    FROM rels;
    """).result()

    print(f"Normalized table {PROJECT_ID}.{DATASET}.article_relationships created in BigQuery.")
else:
    print("REPRO mode: keeping normalized info in DataFrames or fixtures.")


if RUN_LIVE:
    # 2A) Save raw Gemini extraction (from your live Gemini step)
    # try:
    #     if "gemini_df" in globals():
    #         save_fixture(gemini_df, "gemini_url_extractions.parquet")
    # except Exception as e:
    #     print("Could not save gemini_url_extractions fixture:", e)

    # 2B) Save normalized tables pulled from BigQuery
    try:
        save_fixture(bq_to_df("articles"),             "articles.parquet")
        save_fixture(bq_to_df("article_keywords"),     "article_keywords.parquet")
        save_fixture(bq_to_df("article_entities"),     "article_entities.parquet")
        save_fixture(bq_to_df("article_themes"),       "article_themes.parquet")
    except Exception as e:
        print("Could not save normalized article fixtures:", e)

    # 2C) Save relationships if created in BQ
    try:
        save_fixture(bq_to_df("article_relationships"), "article_relationships.parquet")
    except Exception as e:
        print("Could not save article_relationships fixture:", e)


if not RUN_LIVE:
    # Try fully normalized fixtures first
    try:
        articles_df          = load_fixture("articles.parquet")
        article_keywords_df  = load_fixture("article_keywords.parquet")
        article_entities_df  = load_fixture("article_entities.parquet")
        article_themes_df    = load_fixture("article_themes.parquet")
        # Relationships (either pre-normalized or raw array form)
        try:
            article_relationships_df = load_fixture("article_relationships.parquet")
        except FileNotFoundError:
            print('Cannot find file article_relationships.parquet')
        print("REPRO number of articles:", len(articles_df))
    except FileNotFoundError:
        print('Cannot find file.')





if RUN_LIVE:
    BQ_CONNECTION   = f"projects/{PROJECT_ID}/locations/us/connections/us_llm_conn"
    
    sql = f"""
    CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET}.gemini_model`
    REMOTE WITH CONNECTION `{BQ_CONNECTION}`
    OPTIONS (ENDPOINT = '{GEMINI_MODEL}');
    """
    
    job = bq.query(sql)
    job.result()
    print("Created remote model:", f"{PROJECT_ID}.{DATASET}.gemini_model")


if RUN_LIVE:
    BQ_CONNECTION   = f"projects/{PROJECT_ID}/locations/us/connections/us_llm_conn"
    
    sql = f"""
    CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET}.gemini_model_2`
    REMOTE WITH CONNECTION `{BQ_CONNECTION}`
    OPTIONS (ENDPOINT = 'gemini-2.5-flash');
    """
    
    job = bq.query(sql)
    job.result()
    print("Created remote model:", f"{PROJECT_ID}.{DATASET}.gemini_model_2")


if RUN_LIVE:
    # These statements require that the BigQuery Connection (BQ_CONNECTION) already exists and points to Vertex AI.
    create_embed_model = f"""
    CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET}.embed_model`
    REMOTE WITH CONNECTION `{BQ_CONNECTION}`
    OPTIONS (ENDPOINT = '{EMBEDDING_ENDPOINT}');
    """
    # create_text_model = f"""
    # CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET}.gemini_model`
    # REMOTE WITH CONNECTION `{BQ_CONNECTION}`
    # OPTIONS (ENDPOINT = '{GEMINI_MODEL}');
    # """

    try:
        bq.query(create_embed_model).result()
        print("Created/confirmed embed_model")
        # bq.query(create_text_model).result()
        # print("Created/confirmed gemini_model")
    except Exception as e:
        print("Could not create remote models. Ensure BigQuery connection exists:", BQ_CONNECTION)
        print("Error:", e)
        print("You can still proceed in REPRO mode, or pre-create models in your GCP project.")



# %%time

# # This is the fastest so far.

# if RUN_LIVE:
#     # Build a combined text field (title + summary + keywords + entities)
#     sql_emb = f"""
#             CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.article_emb` AS
#             SELECT
#               url,
#               ml_generate_embedding_result AS emb
#             FROM ML.GENERATE_EMBEDDING(
#               MODEL `{PROJECT_ID}.{DATASET}.embed_model`,
#               (
#                 SELECT
#                   CONCAT(
#                     title, '\\n', summary, '\\n',
#                     IFNULL((SELECT STRING_AGG(keyword, ' ') FROM `{PROJECT_ID}.{DATASET}.article_keywords` k WHERE k.url = articles.url AND k.GLOBALEVENTID = articles.GLOBALEVENTID), ''),
#                     ' ',
#                     IFNULL((SELECT STRING_AGG(entity, ' ') FROM `{PROJECT_ID}.{DATASET}.article_entities` e WHERE e.url = articles.url AND e.GLOBALEVENTID = articles.GLOBALEVENTID), ''),
#                     ' ',
#                     IFNULL((SELECT STRING_AGG(relation, ' ') FROM `{PROJECT_ID}.{DATASET}.article_relationships` e WHERE e.url = articles.url AND e.GLOBALEVENTID = articles.GLOBALEVENTID), ''),
#                     ' ',
#                     IFNULL((SELECT STRING_AGG(theme, ' ') FROM `{PROJECT_ID}.{DATASET}.article_themes` e WHERE e.url = articles.url AND e.GLOBALEVENTID = articles.GLOBALEVENTID), '')
#                   ) AS content,
#                   url
#                 FROM `{PROJECT_ID}.{DATASET}.articles` articles
#                 WHERE summary IS NOT NULL
#                 Limit 1
#               )
#             );
#             """
#     print('Embedding...')
#     bq.query(sql_emb).result()
#     print("Embedded articles → article_emb")
#     try:
#         emb_df_live = bq_to_df("article_emb")
#         save_fixture(emb_df_live, "article_emb.parquet")
#     except Exception as e:
#         print("Could not save article_emb fixture:", e)

    
# else:
#     print("REPRO mode: load precomputed embeddings if provided in fixtures.")
#     try:
#         emb_df = load_fixture("article_emb.parquet")  # columns: url, emb (list<float>)
#         print("REPRO article_emb:", len(emb_df))
#     except FileNotFoundError:
#         emb_df = pd.DataFrame()
#         print("No article_emb fixture found; skip vector demo or attach the fixture.")

# #16 min for 200 rec
# #25min 18s for 300 records


# %%time
# if not RUN_LIVE:
#     sql_emb = f"""
#     CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.article_emb` AS
#     WITH base_articles AS (
#       SELECT url, title, summary, GLOBALEVENTID
#       FROM `{PROJECT_ID}.{DATASET}.articles`
#       WHERE summary IS NOT NULL
#       LIMIT 1000
#     ),
#     kw AS (
#       SELECT k.GLOBALEVENTID, k.url, STRING_AGG(k.keyword, ' ') AS kw
#       FROM `{PROJECT_ID}.{DATASET}.article_keywords` k
#       JOIN base_articles a USING (GLOBALEVENTID)
#       GROUP BY k.url, k.GLOBALEVENTID
#     ),
#     ent AS (
#       SELECT e.GLOBALEVENTID, e.url, STRING_AGG(e.entity, ' ') AS ent
#       FROM `{PROJECT_ID}.{DATASET}.article_entities` e
#       JOIN base_articles a USING (GLOBALEVENTID)
#       GROUP BY e.url, e.GLOBALEVENTID
#     ),
#     txt AS (
#       SELECT
#         a.url,
#         CONCAT(
#           a.title, '\\n', a.summary, '\\n',
#           IFNULL(kw.kw, ''), ' ',
#           IFNULL(ent.ent, '')
#         ) AS content
#       FROM base_articles a
#       LEFT JOIN kw
#         ON kw.url = a.url AND kw.GLOBALEVENTID = a.GLOBALEVENTID
#       LEFT JOIN ent
#         ON ent.url = a.url AND ent.GLOBALEVENTID = a.GLOBALEVENTID
#     )
#     SELECT
#       url,
#       ml_generate_embedding_result AS emb
#     FROM ML.GENERATE_EMBEDDING(
#       MODEL `{PROJECT_ID}.{DATASET}.embed_model`,
#       (SELECT content, url FROM txt)
#     );
#     """
#     print("Embedding...")
#     bq.query(sql_emb).result()
#     print("Embedded articles → article_emb")

#     try:
#         emb_df_live = bq_to_df("article_emb")
#         save_fixture(emb_df_live, "article_emb.parquet")
#     except Exception as e:
#         print("Could not save article_emb fixture:", e)
# else:
#     print("REPRO mode: load precomputed embeddings if provided in fixtures.")
#     try:
#         emb_df = load_fixture("article_emb.parquet")  # columns: url, emb (list<float>)
#         print("REPRO article_emb:", len(emb_df))
#     except FileNotFoundError:
#         emb_df = pd.DataFrame()
#         print("No article_emb fixture found; skip vector demo or attach the fixture.")
# #4.62 sec for 1 recs
# #34min 39 se for 400 rec


%%time
# This will build the table with the text for embedding
if RUN_LIVE:
    # 1) Build the table with the text to embed (content + url)
    sql_text = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.article_text` AS
    WITH base_articles AS (
      SELECT url, title, summary, GLOBALEVENTID
      FROM `{PROJECT_ID}.{DATASET}.articles`
      WHERE summary IS NOT NULL
      LIMIT 1000
    ),
    kw AS (
      SELECT k.url, k.GLOBALEVENTID, STRING_AGG(k.keyword, ' ') AS kw
      FROM `{PROJECT_ID}.{DATASET}.article_keywords` k
      JOIN base_articles a
        ON a.url = k.url AND a.GLOBALEVENTID = k.GLOBALEVENTID
      GROUP BY k.url, k.GLOBALEVENTID
    ),
    ent AS (
      SELECT e.url, e.GLOBALEVENTID, STRING_AGG(e.entity, ' ') AS ent
      FROM `{PROJECT_ID}.{DATASET}.article_entities` e
      JOIN base_articles a
        ON a.url = e.url AND a.GLOBALEVENTID = e.GLOBALEVENTID
      GROUP BY e.url, e.GLOBALEVENTID
    ),
    rel AS (
      SELECT r.url, r.GLOBALEVENTID, STRING_AGG(r.relation, ' ') AS rel
      FROM `{PROJECT_ID}.{DATASET}.article_relationships` r
      JOIN base_articles a
        ON a.url = r.url AND a.GLOBALEVENTID = r.GLOBALEVENTID
      GROUP BY r.url, r.GLOBALEVENTID
    ),
    th AS (
      SELECT t.url, t.GLOBALEVENTID, STRING_AGG(t.theme, ' ') AS th
      FROM `{PROJECT_ID}.{DATASET}.article_themes` t
      JOIN base_articles a
        ON a.url = t.url AND a.GLOBALEVENTID = t.GLOBALEVENTID
      GROUP BY t.url, t.GLOBALEVENTID
    )
    SELECT
      a.GLOBALEVENTID,
      a.url,
      CONCAT(
        a.title, '\\n', a.summary, '\\n',
        IFNULL(kw.kw, ''), ' ',
        IFNULL(ent.ent, ''), ' ',
        IFNULL(rel.rel, ''), ' ',
        IFNULL(th.th, '')
      ) AS content
    FROM base_articles a
    LEFT JOIN kw  ON kw.url  = a.url AND kw.GLOBALEVENTID  = a.GLOBALEVENTID
    LEFT JOIN ent ON ent.url = a.url AND ent.GLOBALEVENTID = a.GLOBALEVENTID
    LEFT JOIN rel ON rel.url = a.url AND rel.GLOBALEVENTID = a.GLOBALEVENTID
    LEFT JOIN th  ON th.url  = a.url AND th.GLOBALEVENTID  = a.GLOBALEVENTID;
    """
    bq.query(sql_text).result()
    print("Built table:", f"{PROJECT_ID}.{DATASET}.article_text")


%%time
if RUN_LIVE:
    # 2) Embed using the table (no subquery)
    sql_emb = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.article_emb` AS
    SELECT
      GLOBALEVENTID,
      url,
      ml_generate_embedding_result AS emb
    FROM ML.GENERATE_EMBEDDING(
      MODEL `{PROJECT_ID}.{DATASET}.embed_model`,
      TABLE `{PROJECT_ID}.{DATASET}.article_text`,
      STRUCT('SEMANTIC_SIMILARITY' as task_type)
    );
    """
    print("Embedding...")
    bq.query(sql_emb).result()
    print("Embedded articles → article_emb")

    # (optional) Save fixture
    try:
        emb_df_live = bq_to_df("article_emb")
        save_fixture(emb_df_live, "article_emb.parquet")
    except Exception as e:
        print("Could not save article_emb fixture:", e)

# 5.248 s per record

else:
    article_emb_df    = load_fixture("article_emb.parquet")


# 1 = 2


# pd.read_parquet('/kaggle/working/fixtures/article_emb.parquet')


if RUN_LIVE:
    # # Your existing embedding code...
    # bq.query(sql_emb).result()
    # print("Embedded articles → article_emb")
    
    # Check row count before creating index
    count_sql = f"SELECT COUNT(*) as row_count FROM `{PROJECT_ID}.{DATASET}.article_emb`"
    count_result = bq.query(count_sql).result()
    row_count = list(count_result)[0].row_count
    
    if row_count >= 5000:
        sql_idx = f"""
        CREATE VECTOR INDEX `{PROJECT_ID}.{DATASET}.idx_article_emb`
        ON `{PROJECT_ID}.{DATASET}.article_emb`(emb)
        OPTIONS(index_type='IVF', distance_type='COSINE');
        """
        bq.query(sql_idx).result()
        print("Created vector index idx_article_emb")
    else:
        print(f"Dataset has {row_count} rows (minimum 5000 required for IVF index). Use VECTOR_SEARCH function directly.")
    


# # Checking the columns returned
# user_query = "central bank rate hike impact"
# if RUN_LIVE:
#     # First, let's see what columns VECTOR_SEARCH actually returns
#     sql_check = f"""
#     SELECT *
#     FROM VECTOR_SEARCH(
#            TABLE `{PROJECT_ID}.{DATASET}.article_emb`, 'emb',
#            (SELECT ml_generate_embedding_result FROM ML.GENERATE_EMBEDDING(
#               MODEL `{PROJECT_ID}.{DATASET}.embed_model`,
#               (SELECT @user_query AS content)
#            )),
#            top_k => 3, distance_type => 'COSINE') AS s
#     """
    
#     check_df = bq.query(sql_check, job_config=bigquery.QueryJobConfig(
#         query_parameters=[bigquery.ScalarQueryParameter("user_query", "STRING", user_query)]
#     )).result().to_dataframe()
    
#     print("VECTOR_SEARCH returns these columns:")
#     print(check_df.columns.tolist())
#     print(check_df.head())


# user_query = "venezuela"
# if RUN_LIVE:
#     sql_vs = f"""
#     WITH q AS (
#       SELECT ml_generate_embedding_result AS v
#       FROM ML.GENERATE_EMBEDDING(
#         MODEL `{PROJECT_ID}.{DATASET}.embed_model`,
#         (SELECT @user_query AS content),
#         STRUCT('SEMANTIC_SIMILARITY' as task_type, 32 AS output_dimensionality)
#       )
#     )
#     SELECT a.url, a.title, s.distance
#     FROM VECTOR_SEARCH(
#            TABLE `{PROJECT_ID}.{DATASET}.article_emb`, 'emb',
#            (SELECT v FROM q),
#            top_k => {TOP_K}, distance_type => 'COSINE') AS s
#     JOIN `{PROJECT_ID}.{DATASET}.articles` a
#       ON a.url = s.base.url  -- Access url from the base struct
#     ORDER BY s.distance;
#     """
#     vs_df = bq.query(sql_vs, job_config=bigquery.QueryJobConfig(
#         query_parameters=[bigquery.ScalarQueryParameter("user_query", "STRING", user_query)]
#     )).result().to_dataframe()

#     # --- Save fixture (LIVE) ---
#     try:
#         vs_to_save = vs_df.copy()
#         vs_to_save.insert(0, "query", user_query)
#         vs_to_save["top_k"] = TOP_K
#         if "save_fixture" in globals():
#             save_fixture(vs_to_save, "vector_search_demo_1.parquet")
#             print("Saved fixture: vector_search_demo_1.parquet")
#         else:
#             print("Could not save vector_search_demo_1 fixture:")
#     except Exception as e:
#         print("Error: Could not save vector_search_demo_1 fixture:", e)    

# else:
#     vs_df = load_fixture("vector_search_demo_1.parquet")
# print("Vector search 1 hits:", len(vs_df))
# df_head(vs_df, 10)


# user_query = "are there any news on stocks?"
# if RUN_LIVE:
#     sql_vs = f"""
#     WITH q AS (
#       SELECT ml_generate_embedding_result AS v
#       FROM ML.GENERATE_EMBEDDING(
#         MODEL `{PROJECT_ID}.{DATASET}.embed_model`,
#         (SELECT @user_query AS content),
#         STRUCT('SEMANTIC_SIMILARITY' as task_type, 32 AS output_dimensionality)
#       )
#     )
#     SELECT a.url, a.title, s.distance
#     FROM VECTOR_SEARCH(
#            TABLE `{PROJECT_ID}.{DATASET}.article_emb`, 'emb',
#            (SELECT v FROM q),
#            top_k => {TOP_K}, distance_type => 'COSINE') AS s
#     JOIN `{PROJECT_ID}.{DATASET}.articles` a
#       ON a.url = s.base.url  -- Access url from the base struct
#     ORDER BY s.distance;
#     """
#     vs_df = bq.query(sql_vs, job_config=bigquery.QueryJobConfig(
#         query_parameters=[bigquery.ScalarQueryParameter("user_query", "STRING", user_query)]
#     )).result().to_dataframe()
    

#     # --- Save fixture (LIVE) ---
#     try:
#         vs_to_save = vs_df.copy()
#         vs_to_save.insert(0, "query", user_query)
#         vs_to_save["top_k"] = TOP_K
#         if "save_fixture" in globals():
#             save_fixture(vs_to_save, "vector_search_demo_2.parquet")
#             print("Saved fixture: vector_search_demo_2.parquet")
#         else:
#             print("Could not save vector_search_demo_2 fixture:")
#     except Exception as e:
#         print("Error: Could not save vector_search_demo_2 fixture:", e)    
# else:
#     vs_df = load_fixture("vector_search_demo_2.parquet")
# print("Vector search 2 hits:", len(vs_df))
# df_head(vs_df, 10)


user_query = "any news on Trump?"
TOP_K = 10

if RUN_LIVE:
    sql_vs = f"""
    -- 1) Expand the user query with Gemini (structured output)
    WITH expand AS (
      SELECT intent, theme, subject, related_keywords, expanded_questions, final_query
      FROM AI.GENERATE_TABLE(
        MODEL `{PROJECT_ID}.{DATASET}.gemini_model_2`,
        (
          SELECT CONCAT(
            'You are a query expansion assistant. Analyze the user input and return strictly JSON that matches the schema.',
            'Answer the following:',
            '1) intent: What is the user trying to accomplish?',
            '2) theme: High-level topic domain (1–3 words).',
            '3) subject: The concrete topic/entities referenced (short phrase).',
            '5) expanded_questions: 3–5 natural-language questions implied by the query.',
            '6) final_query: One concise, expanded search query (15–30 words) that best represents all of the above. Avoid stopwords and fluff.',
            'Return only valid JSON.',
            'Original query: "', @user_query, '"'
          ) AS prompt
        ),
        STRUCT(
          "intent STRING, theme STRING, subject STRING, related_keywords ARRAY<STRING>, expanded_questions ARRAY<STRING>, final_query STRING" AS output_schema,
          1024 AS max_output_tokens,
          0.3 AS temperature
        )
      )
    ),

    -- 2) Embed the final query
    q AS (
      SELECT ml_generate_embedding_result AS v
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{PROJECT_ID}.{DATASET}.embed_model`,
        (SELECT final_query AS content FROM expand),
        STRUCT('SEMANTIC_SIMILARITY' as task_type, 32 AS output_dimensionality)
      )
    )

    -- 3) Vector search using the final query embedding
    SELECT
      a.url,
      a.title,
      s.distance,
      e.intent,
      e.theme,
      e.subject,
      e.related_keywords,
      e.expanded_questions,
      e.final_query
    FROM VECTOR_SEARCH(
           TABLE `{PROJECT_ID}.{DATASET}.article_emb`, 'emb',
           (SELECT v FROM q),
           top_k => {TOP_K}, distance_type => 'COSINE'
         ) AS s
    JOIN `{PROJECT_ID}.{DATASET}.articles` a
      ON a.url = s.base.url
    CROSS JOIN expand e
    ORDER BY s.distance;
    """

    vs_df = bq.query(
        sql_vs,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("user_query", "STRING", user_query)]
        ),
    ).result().to_dataframe()
    
    # --- Save fixture (LIVE) ---
    try:
        vs_to_save = vs_df.copy()
        vs_to_save.insert(0, "query", user_query)
        vs_to_save["top_k"] = TOP_K
        if "save_fixture" in globals():
            save_fixture(vs_to_save, "vector_search_demo_1.parquet")
            print("Saved fixture: vector_search_demo_1.parquet")
        else:
            print("Could not save vector_search_demo_1 fixture:")
    except Exception as e:
        print("Error: Could not save vector_search_demo_1 fixture:", e)    

else:
    # REPRO fallback (keeps your behavior)
    vs_df = load_fixture("vector_search_demo_1.parquet")

print("Vector search 1 hits:", len(vs_df))
df_head(vs_df, 10)

# Optional: print the final query the model constructed
try:
    fq = vs_df["final_query"].iloc[0]
    print("\nFinal expanded query used for embedding:\n", fq)
except Exception:
    pass



display(vs_df)


# # Optional Clustering

# DO_CLUSTER = True

# if RUN_LIVE and DO_CLUSTER:
#     # Train KMeans on embeddings (small K to keep it quick)
#     bq.query(f"""
#     CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET}.km_articles`
#     OPTIONS(MODEL_TYPE='KMEANS', NUM_CLUSTERS=30) AS
#     SELECT emb FROM `{PROJECT_ID}.{DATASET}.article_emb`;
#     """).result()

#     # Assign cluster_id
#     bq.query(f"""
#     CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.article_clusters` AS
#     SELECT e.url, p.centroid_id AS cluster_id
#     FROM ML.PREDICT(MODEL `{PROJECT_ID}.{DATASET}.km_articles`,
#                     (SELECT emb FROM `{PROJECT_ID}.{DATASET}.article_emb`)) p
#     JOIN `{PROJECT_ID}.{DATASET}.article_emb` e USING(emb);
#     """).result()

#     # Find dominant cluster among retrieved results
#     dom_cluster_sql = f"""
#     WITH hits AS (
#       SELECT url FROM UNNEST(@urls) url
#     ),
#     joined AS (
#       SELECT c.cluster_id, COUNT(*) AS cnt
#       FROM `{PROJECT_ID}.{DATASET}.article_clusters` c
#       JOIN hits h USING (url)
#       GROUP BY cluster_id
#       ORDER BY cnt DESC
#       LIMIT 1
#     )
#     SELECT cluster_id FROM joined
#     """
#     urls_param = vs_df["url"].tolist()
#     result = bq.query(dom_cluster_sql, job_config=bigquery.QueryJobConfig(
#         query_parameters=[bigquery.ArrayQueryParameter("urls", "STRING", urls_param)]
#     )).result().to_dataframe()
#     winning_cluster = int(result.iloc[0]["cluster_id"]) if len(result) else None
#     print("Winning cluster:", winning_cluster)
# else:
#     winning_cluster = None
#     print("Clustering skipped or REPRO mode.")



# if RUN_LIVE:
#     # Build a context from the retrieved set (restricted to winning_cluster if available)
#     if winning_cluster is not None:
#         filter_sql = f"""
#         SELECT a.summary FROM `{PROJECT_ID}.{DATASET}.articles` a
#         JOIN `{PROJECT_ID}.{DATASET}.article_clusters` c USING(url)
#         WHERE c.cluster_id = @cid AND url IN UNNEST(@urls)
#         """
#         summaries_df = bq.query(filter_sql, job_config=bigquery.QueryJobConfig(
#             query_parameters=[
#                 bigquery.ScalarQueryParameter("cid", "INT64", winning_cluster),
#                 bigquery.ArrayQueryParameter("urls", "STRING", vs_df["url"].tolist())
#             ]
#         )).result().to_dataframe()
#     else:
#         filter_sql = f"SELECT summary FROM `{PROJECT_ID}.{DATASET}.articles` WHERE url IN UNNEST(@urls)"
#         summaries_df = bq.query(filter_sql, job_config=bigquery.QueryJobConfig(
#             query_parameters=[bigquery.ArrayQueryParameter("urls", "STRING", vs_df["url"].tolist())]
#         )).result().to_dataframe()

#     bundle = "\n\n".join(summaries_df["summary"].tolist())[:12000]  # keep prompt small

#     # Use AI.GENERATE_TABLE for a structured single-row answer
#     answer_sql = f"""
#     CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.answer_card` AS
#     SELECT *
#     FROM AI.GENERATE_TABLE(
#       MODEL `{PROJECT_ID}.{DATASET}.gemini_model`,
#       ( SELECT STRUCT(
#             'Answer the user query using the summaries below. Provide a brief overview, key themes, and 3 bullet insights. Keep it under 180 words.' AS prompt
#           ) AS input,
#           CONCAT('Query: ', @user_query, '\\n\\nSummaries:\\n', @ctx) AS context ),
#       STRUCT("answer STRING" AS output_schema, 384 AS max_output_tokens)
#     );
#     """
#     bq.query(answer_sql, job_config=bigquery.QueryJobConfig(
#         query_parameters=[
#             bigquery.ScalarQueryParameter("user_query", "STRING", user_query),
#             bigquery.ScalarQueryParameter("ctx", "STRING", bundle),
#         ]
#     )).result()

#     answer_df = bq.query(f"SELECT answer FROM `{PROJECT_ID}.{DATASET}.answer_card`").result().to_dataframe()
#     display(answer_df)
# else:
#     # REPRO mode: just render a synthetic answer from fixtures or print a placeholder
#     print("REPRO mode: final answer would be generated here by AI.GENERATE_TABLE over retrieved summaries.")




















