# !pip -q install --upgrade google-cloud-bigquery pandas-gbq db-dtypes
# !pip -q install sentence-transformers faiss-cpu plotly folium geopy pyproj
# !pip -q install sentence-transformers faiss-cpu


import os, time, re, numpy as np, pandas as pd
import plotly.express as px
from google.cloud import bigquery
from sentence_transformers import SentenceTransformer
import faiss
from kaggle_secrets import UserSecretsClient
from google.oauth2 import service_account
import pandas_gbq, json
from google.oauth2 import service_account
from math import radians, sin, cos, asin, sqrt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---- project config (edit based on your project in BQ platform) ----

YOUR_PROJECT = "kag-bike-469910"    # GCP project id
YOUR_DATASET = "citibike_ai"        # working dataset
REGION       = "US"                 # BigQuery region for this project/dataset

bq = bigquery.Client(project=YOUR_PROJECT)
print("Config OK:", YOUR_PROJECT, YOUR_DATASET, REGION)


client = bigquery.Client()

# 1) List tables

dataset_id = "bigquery-public-data.new_york_citibike"
tables = client.list_tables(dataset_id)
print("Tables in dataset:")
for t in tables:
    print(f"- {t.table_id}")

# 2) Peek at trips schema

table_ref = client.dataset("new_york_citibike", project="bigquery-public-data").table("citibike_trips")
table = client.get_table(table_ref)
print("\nSchema (citibike_trips):")
for s in table.schema:
    print(s.name, s.field_type, s.mode)


client = bigquery.Client()

def run_query(sql: str) -> pd.DataFrame:
    job = client.query(sql)
    return job.result().to_dataframe(create_bqstorage_client=False)


# A) Total rows

run_query("""
SELECT COUNT(*) AS total_rows
FROM `bigquery-public-data.new_york_citibike.citibike_trips`
""")


# B) Min/Max starttime

run_query("""
SELECT
  MIN(starttime) AS min_starttime,
  MAX(starttime) AS max_starttime
FROM `bigquery-public-data.new_york_citibike.citibike_trips`
""")


# C) Rows by year

run_query("""
SELECT
  EXTRACT(YEAR FROM starttime) AS year,
  COUNT(*) AS n
FROM `bigquery-public-data.new_york_citibike.citibike_trips`
GROUP BY year
ORDER BY year DESC
""")


# D) Rows by month for a chosen year (2017)

run_query("""
SELECT
  EXTRACT(YEAR FROM starttime) AS yr,
  EXTRACT(MONTH FROM starttime) AS mo,
  COUNT(*) AS n
FROM `bigquery-public-data.new_york_citibike.citibike_trips`
WHERE EXTRACT(YEAR FROM starttime) = 2017
GROUP BY yr, mo
ORDER BY yr DESC, mo DESC
""")


sql_trips = """
SELECT
  tripduration AS trip_seconds,
  starttime,
  stoptime,
  start_station_id,
  start_station_name,
  start_station_latitude,
  start_station_longitude,
  end_station_id,
  end_station_name,
  end_station_latitude,
  end_station_longitude,
  usertype,
  birth_year,
  gender,
  customer_plan
FROM `bigquery-public-data.new_york_citibike.citibike_trips`
WHERE EXTRACT(YEAR FROM starttime) = 2017
  AND EXTRACT(MONTH FROM starttime) BETWEEN 6 AND 8
  AND start_station_id IS NOT NULL
  AND end_station_id IS NOT NULL
LIMIT 100000
"""
trips_df = run_query(sql_trips)
print("Trips rows:", len(trips_df))
trips_df.head()


stations_df = run_query("""
SELECT
  station_id,
  name AS station_name,
  latitude AS lat,
  longitude AS lon
FROM `bigquery-public-data.new_york_citibike.citibike_stations`
WHERE station_id IS NOT NULL
""")
print("Stations rows:", len(stations_df))
stations_df.head()





sql_trips = """
SELECT
  tripduration AS trip_seconds,
  starttime,
  stoptime,
  start_station_id, start_station_name,
  start_station_latitude, start_station_longitude,
  end_station_id,   end_station_name,
  end_station_latitude,   end_station_longitude,
  usertype, birth_year, gender, customer_plan
FROM `bigquery-public-data.new_york_citibike.citibike_trips`
WHERE starttime IS NOT NULL
  AND EXTRACT(YEAR FROM starttime)=2017
  AND EXTRACT(MONTH FROM starttime) BETWEEN 6 AND 10
  AND start_station_id IS NOT NULL
  AND end_station_id   IS NOT NULL
LIMIT 250000
"""
trips_df = run_query(sql_trips)
print("Trips rows:", len(trips_df))
trips_df.head()


stations_df = run_query("""
SELECT
  station_id,
  name AS station_name,
  latitude AS lat,
  longitude AS lon
FROM `bigquery-public-data.new_york_citibike.citibike_stations`
WHERE station_id IS NOT NULL
""")
print("Stations:", len(stations_df))


sql_station_feats = """
WITH base AS (
  SELECT
    start_station_id AS station_id,
    start_station_name AS station_name,
    start_station_latitude  AS lat,
    start_station_longitude AS lon,
    tripduration AS trip_seconds,
    EXTRACT(HOUR  FROM starttime) AS hr,
    EXTRACT(DAYOFWEEK FROM starttime) AS dow  -- 1=Sunday ... 7=Saturday
  FROM `bigquery-public-data.new_york_citibike.citibike_trips`
  WHERE starttime IS NOT NULL
    AND EXTRACT(YEAR FROM starttime)=2017
    AND EXTRACT(MONTH FROM starttime) BETWEEN 6 AND 10
    AND start_station_id IS NOT NULL
    AND end_station_id   IS NOT NULL
),
agg AS (
  SELECT
    station_id, ANY_VALUE(station_name) AS station_name,
    ANY_VALUE(lat) AS lat, ANY_VALUE(lon) AS lon,
    COUNT(*) AS trips,
    AVG(trip_seconds) AS avg_secs,
    SUM(CASE WHEN dow BETWEEN 2 AND 6 THEN 1 ELSE 0 END) AS wk_trips,
    SUM(CASE WHEN dow IN (1,7) THEN 1 ELSE 0 END) AS wknd_trips,
    SUM(CASE WHEN hr BETWEEN 7 AND 10 THEN 1 ELSE 0 END) AS morning_trips,
    SUM(CASE WHEN hr BETWEEN 17 AND 20 THEN 1 ELSE 0 END) AS evening_trips,
    SUM(CASE WHEN trip_seconds <= 10*60 THEN 1 ELSE 0 END) AS short_trips
  FROM base
  GROUP BY station_id
)
SELECT
  station_id, station_name, lat, lon, trips, avg_secs,
  SAFE_DIVIDE(wknd_trips, NULLIF(wk_trips+wknd_trips,0)) AS wknd_ratio,
  SAFE_DIVIDE(morning_trips, NULLIF(trips,0)) AS morning_ratio,
  SAFE_DIVIDE(evening_trips, NULLIF(trips,0)) AS evening_ratio,
  SAFE_DIVIDE(short_trips,   NULLIF(trips,0)) AS short_ratio
FROM agg
"""
feat_df = run_query(sql_station_feats)
feat_df.fillna(0.0, inplace=True)
print("Stations with features:", len(feat_df))
feat_df.head()


def tag_row(r):
    tags = []
    if r['wknd_ratio'] >= 0.40: tags.append("weekend-friendly")
    if r['morning_ratio'] >= 0.20: tags.append("commuter-morning")
    if r['evening_ratio'] >= 0.20: tags.append("after-work crowd")
    if r['short_ratio'] >= 0.40: tags.append("short-trip hub")
    if r['avg_secs'] <= 12*60: tags.append("quick rides")
    if r['trips'] >= 10000: tags.append("high activity")
    return list(dict.fromkeys(tags))

def blurb_row(r, tags):
    base = f"{r['station_name']} sees {int(r['trips']):,} rides in this season."
    bits = []
    if "weekend-friendly" in tags: bits.append("Popular on weekends")
    if "commuter-morning" in tags: bits.append("busy in morning commute")
    if "after-work crowd" in tags: bits.append("active after work hours")
    if "short-trip hub" in tags: bits.append("often used for short city hops")
    if "quick rides" in tags: bits.append("typical rides are quick")
    if "high activity" in tags: bits.append("overall a high-traffic station")
    detail = "; ".join(bits) + "." if bits else ""
    return f"{base} {detail}".strip()

feat_df["tags"] = feat_df.apply(tag_row, axis=1)
feat_df["blurb"] = feat_df.apply(lambda r: blurb_row(r, r["tags"]), axis=1)

enrichment_df = feat_df[["station_id","station_name","lat","lon","tags","blurb"]].copy()
enrichment_df.head(10)


top_viz = enrichment_df.sort_values("station_id").dropna(subset=["lat","lon"]).head(1000)
fig = px.scatter_mapbox(
    top_viz, lat="lat", lon="lon",
    hover_name="station_name",
    hover_data={"blurb":True, "tags":True, "lat":False, "lon":False},
    zoom=11, height=600
)
fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
fig.show()



model = SentenceTransformer("/kaggle/input/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2")

texts = enrichment_df["blurb"].fillna("").tolist()
emb = model.encode(texts, normalize_embeddings=True)
emb = np.asarray(emb, dtype="float32")

index = faiss.IndexFlatIP(emb.shape[1])  # cosine via inner product on normalized vectors
index.add(emb)

def search(query:str, top_k:int=10):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, top_k)
    res = enrichment_df.iloc[I[0]].copy()
    res["score"] = D[0]
    return res[["station_id","station_name","tags","blurb","score"]]

search("family friendly weekend scenic")



def plot_semantic_map(enrichment_df: pd.DataFrame,
                      top_df: pd.DataFrame,
                      zoom: int = 11,
                      score_col: str | None = None):
    """
    enrichment_df: must have columns [station_id, station_name, lat, lon, tags, blurb]
    top_df: must have at least [station_id]; may include a score column (e.g., 'score' or 'distance')
    score_col: name of score column in top_df to display in hover (optional)
    """
    # Ensure unique IDs & join any score for hover
    top_ids = pd.unique(top_df["station_id"]).tolist()
    base = enrichment_df.dropna(subset=["lat","lon"]).copy()

    focus = base[base["station_id"].isin(top_ids)].copy()
    if score_col and score_col in top_df.columns:
        focus = focus.merge(top_df[["station_id", score_col]], on="station_id", how="left")

    background = base[~base["station_id"].isin(top_ids)].copy()

    # Background (all stations, faint)
    fig = px.scatter_mapbox(
        background, lat="lat", lon="lon",
        hover_name="station_name",
        opacity=0.35, zoom=zoom, height=650
    )
    fig.update_traces(marker={"size":5})

    # Focus (top-K stations, bigger markers with extra hover)
    hover_data = {"blurb": True, "tags": True, "lat": False, "lon": False}
    if score_col and score_col in focus.columns:
        hover_data[score_col] = True

    fig2 = px.scatter_mapbox(
        focus, lat="lat", lon="lon",
        hover_name="station_name",
        hover_data=hover_data,
        zoom=zoom, height=650
    )
    fig2.update_traces(marker={"size":12})

    # Overlay focus layer
    for tr in fig2.data:
        fig.add_trace(tr)

    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    fig.show()


fig



YOUR_PROJECT = "kag-bike-469910"
YOUR_DATASET = "citibike_ai"   # make sure this dataset is in Location = US :))

secrets = UserSecretsClient()
sa_info = json.loads(secrets.get_secret("GCP_SA_KEY"))
creds = service_account.Credentials.from_service_account_info(sa_info)

bq = bigquery.Client(project=YOUR_PROJECT, credentials=creds)
pandas_gbq.context.credentials = creds
pandas_gbq.context.project = YOUR_PROJECT

# sanity write test to check everything is ok
import pandas as pd
pandas_gbq.to_gbq(pd.DataFrame({"ok":[1,2,3]}),
                  f"{YOUR_DATASET}.hello_world",
                  project_id=YOUR_PROJECT, if_exists="replace")
print("Ready:", YOUR_PROJECT, YOUR_DATASET)


YOUR_PROJECT = "kag-bike-469910"
YOUR_DATASET = "citibike_ai"
REGION = "US"

secrets = UserSecretsClient()
creds = service_account.Credentials.from_service_account_info(
    json.loads(secrets.get_secret("GCP_SA_KEY"))
)

bq = bigquery.Client(project=YOUR_PROJECT, credentials=creds)
pandas_gbq.context.credentials = creds
pandas_gbq.context.project = YOUR_PROJECT

def run_query(sql: str) -> pd.DataFrame:
    job = bq.query(sql)
    
    return job.result().to_dataframe(create_bqstorage_client=False)

print("✅ Auth ready:", YOUR_PROJECT, YOUR_DATASET)


sql_station_feats = """
WITH base AS (
  SELECT
    start_station_id AS station_id,
    start_station_name AS station_name,
    start_station_latitude  AS lat,
    start_station_longitude AS lon,
    tripduration AS trip_seconds,
    EXTRACT(HOUR FROM starttime) AS hr,
    EXTRACT(DAYOFWEEK FROM starttime) AS dow  -- 1=Sun ... 7=Sat
  FROM `bigquery-public-data.new_york_citibike.citibike_trips`
  WHERE starttime IS NOT NULL
    AND EXTRACT(YEAR FROM starttime) = 2017
    AND EXTRACT(MONTH FROM starttime) BETWEEN 6 AND 10
    AND start_station_id IS NOT NULL
    AND end_station_id   IS NOT NULL
),
agg AS (
  SELECT
    station_id, ANY_VALUE(station_name) AS station_name,
    ANY_VALUE(lat) AS lat, ANY_VALUE(lon) AS lon,
    COUNT(*) AS trips,
    AVG(trip_seconds) AS avg_secs,
    SUM(CASE WHEN dow BETWEEN 2 AND 6 THEN 1 ELSE 0 END) AS wk_trips,
    SUM(CASE WHEN dow IN (1,7) THEN 1 ELSE 0 END) AS wknd_trips,
    SUM(CASE WHEN hr BETWEEN 7 AND 10 THEN 1 ELSE 0 END) AS morning_trips,
    SUM(CASE WHEN hr BETWEEN 17 AND 20 THEN 1 ELSE 0 END) AS evening_trips,
    SUM(CASE WHEN trip_seconds <= 10*60 THEN 1 ELSE 0 END) AS short_trips
  FROM base
  GROUP BY station_id
)
SELECT
  station_id, station_name, lat, lon, trips, avg_secs,
  SAFE_DIVIDE(wknd_trips, NULLIF(wk_trips+wknd_trips,0)) AS wknd_ratio,
  SAFE_DIVIDE(morning_trips, NULLIF(trips,0)) AS morning_ratio,
  SAFE_DIVIDE(evening_trips, NULLIF(trips,0)) AS evening_ratio,
  SAFE_DIVIDE(short_trips,   NULLIF(trips,0)) AS short_ratio
FROM agg
"""
feat_df = run_query(sql_station_feats)
feat_df = feat_df.fillna(0.0)
print("Stations w/ features:", len(feat_df))
feat_df.head()


def tag_row(r):
    t = []
    if r['wknd_ratio']   >= 0.40: t.append("weekend-friendly")
    if r['morning_ratio']>= 0.20: t.append("commuter-morning")
    if r['evening_ratio']>= 0.20: t.append("after-work crowd")
    if r['short_ratio']  >= 0.40: t.append("short-trip hub")
    if r['avg_secs']     <= 12*60: t.append("quick rides")
    if r['trips']        >= 10000: t.append("high activity")
    return list(dict.fromkeys(t))

def blurb_row(r, tags):
    base = f"{r['station_name']} sees {int(r['trips']):,} rides in this season."
    bits = []
    if "weekend-friendly" in tags: bits.append("Popular on weekends")
    if "commuter-morning" in tags: bits.append("busy during morning commute")
    if "after-work crowd" in tags: bits.append("active after work hours")
    if "short-trip hub" in tags: bits.append("often used for short city hops")
    if "quick rides" in tags: bits.append("typical rides are quick")
    if "high activity" in tags: bits.append("overall a high-traffic station")
    return f"{base} {'; '.join(bits)+'.' if bits else ''}".strip()

feat_df["tags"]  = feat_df.apply(tag_row, axis=1)
feat_df["blurb"] = feat_df.apply(lambda r: blurb_row(r, r["tags"]), axis=1)

enrichment_df = feat_df[["station_id","station_name","lat","lon","tags","blurb"]].copy()
enrichment_df.head(10)


from sentence_transformers import SentenceTransformer
import numpy as np
import pandas_gbq

model = SentenceTransformer("/kaggle/input/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2")

texts = enrichment_df["blurb"].fillna("").tolist()
emb = model.encode(texts, normalize_embeddings=True)
emb = np.asarray(emb, dtype="float32")

# Attach vectors

enrichment_df["embedding"] = [v.astype("float64").tolist() for v in emb]  # FLOAT64 array for BQ

# Write both blurbs and embeddings to your dataset
pandas_gbq.to_gbq(
    enrichment_df.astype({"station_id":"int64"}),
    destination_table=f"{YOUR_DATASET}.station_embeddings",
    project_id=YOUR_PROJECT,
    if_exists="replace"
)
print("✅ Wrote table:", f"{YOUR_PROJECT}.{YOUR_DATASET}.station_embeddings",
      f"rows={len(enrichment_df)} dim={emb.shape[1]}")


query_text = "family friendly scenic waterfront parks for a relaxed weekend"
qv = model.encode([query_text], normalize_embeddings=True)[0].astype("float64").tolist()
qv_literal = ",".join(map(str, qv))

sql_bq_search = f"""
WITH query AS (SELECT [ {qv_literal} ] AS qv)
SELECT
  se.station_id,
  se.station_name,
  se.tags,
  se.blurb,
  COSINE_DISTANCE(se.embedding, (SELECT qv FROM query)) AS distance
FROM `{YOUR_PROJECT}.{YOUR_DATASET}.station_embeddings` AS se
ORDER BY distance ASC
LIMIT 20
"""
bq_semantic = run_query(sql_bq_search)
bq_semantic.head()


plot_semantic_map(enrichment_df, bq_semantic, score_col="distance")


query_text = "family friendly scenic waterfront parks for a relaxed weekend"
qv = model.encode([query_text], normalize_embeddings=True)[0].astype("float64").tolist()
qv_literal = ",".join(map(str, qv))

sql_bq_search_no_index = f"""
WITH q AS (SELECT [ {qv_literal} ] AS qv)
SELECT
  vs.base.station_id,
  vs.base.station_name,
  vs.base.tags,
  vs.base.blurb,
  vs.distance
FROM q,
UNNEST((
  SELECT ARRAY(
    SELECT AS STRUCT *
    FROM VECTOR_SEARCH(
      TABLE `{YOUR_PROJECT}.{YOUR_DATASET}.station_embeddings`,
      'embedding',
      (SELECT qv FROM q),
      top_k => 20
    )
  )
)) AS vs
ORDER BY vs.distance ASC
"""
bq_semantic = run_query(sql_bq_search_no_index)
bq_semantic.head()


plot_semantic_map(enrichment_df, bq_semantic, score_col="distance")


sql_pairs = """
WITH slice AS (
  SELECT
    start_station_id, start_station_name,
    end_station_id,   end_station_name,
    tripduration
  FROM `bigquery-public-data.new_york_citibike.citibike_trips`
  WHERE starttime IS NOT NULL
    AND EXTRACT(YEAR  FROM starttime) BETWEEN 2016 AND 2017
    AND EXTRACT(MONTH FROM starttime) BETWEEN 5 AND 10
    AND start_station_id IS NOT NULL AND end_station_id IS NOT NULL
),
pairs AS (
  SELECT
    start_station_id, end_station_id,
    ANY_VALUE(start_station_name) AS start_name,
    ANY_VALUE(end_station_name)   AS end_name,
    COUNT(*) AS trip_count,
    AVG(tripduration) AS avg_secs
  FROM slice
  GROUP BY 1,2
)
SELECT * FROM pairs
WHERE trip_count >= 3   -- keep light dedup but keep many rows
"""
route_pairs_df = run_query(sql_pairs)
print("Route pairs rows:", len(route_pairs_df))
route_pairs_df.head()


# Make a short blurb per route

def route_blurb(r):
    mins = int(round(r["avg_secs"]/60.0))
    return (f"Popular ride from {r['start_name']} to {r['end_name']} "
            f"with {int(r['trip_count'])} trips; average time {mins} minutes.")

route_pairs_df["blurb"] = route_pairs_df.apply(route_blurb, axis=1)

# Embed with the SAME model used for stations

route_vecs = model.encode(route_pairs_df["blurb"].tolist(), normalize_embeddings=True)

# Prepare table to write (vectors must be FLOAT64 arrays for BigQuery)
import numpy as np, pandas_gbq
route_pairs_df["embedding"] = [v.astype("float64").tolist() for v in route_vecs.astype("float32")]

# Optional stable id for joins
route_pairs_df["route_id"] = (
    route_pairs_df["start_station_id"].astype(str) + "-" + route_pairs_df["end_station_id"].astype(str)
)

# Write the table
pandas_gbq.to_gbq(
    route_pairs_df[[
        "route_id",
        "start_station_id","end_station_id",
        "start_name","end_name",
        "trip_count","avg_secs","blurb","embedding"
    ]],
    f"{YOUR_DATASET}.route_embeddings",
    project_id=YOUR_PROJECT,
    if_exists="replace"
)
print("✅ wrote:", f"{YOUR_PROJECT}.{YOUR_DATASET}.route_embeddings")





bq.query(f"""
CREATE OR REPLACE VECTOR INDEX `{YOUR_PROJECT}.{YOUR_DATASET}.route_embeddings_idx`
ON `{YOUR_PROJECT}.{YOUR_DATASET}.route_embeddings` (embedding)
STORING(route_id, start_station_id, end_station_id, start_name, end_name, trip_count, avg_secs, blurb)
OPTIONS(
  index_type   = 'IVF',
  distance_type = 'COSINE'
  -- You can tune for larger tables:
  -- , ivf_options = '{{"num_lists": 64}}'
);
""").result()
print("✅ index DDL submitted (IVF)")


run_query(f"""
SELECT index_name, index_status, coverage_percentage, last_refresh_time
FROM `{YOUR_PROJECT}.{YOUR_DATASET}`.INFORMATION_SCHEMA.VECTOR_INDEXES
""")





q = "family friendly scenic waterfront parks for a relaxed weekend"
qv = model.encode([q], normalize_embeddings=True)[0].astype("float64").tolist()
qv_literal = ",".join(map(str, qv))

sql_vec_search = f"""
WITH q AS (SELECT [ {qv_literal} ] AS embedding)
SELECT
  base.route_id,
  base.start_station_id, base.end_station_id,
  base.start_name, base.end_name,
  base.trip_count, base.avg_secs,
  base.blurb,
  distance
FROM
  VECTOR_SEARCH(
    TABLE `{YOUR_PROJECT}.{YOUR_DATASET}.route_embeddings`,
    'embedding',
    TABLE q,
    top_k => 20,
    distance_type => 'COSINE'
  )
ORDER BY distance ASC
"""
routes_semantic = run_query(sql_vec_search)
routes_semantic.head()


print("routes_semantic rows:", len(routes_semantic))
routes_semantic.head()


# Build the set of station IDs we need (start+end) and fetch their coords from trips


for col in ["start_station_id", "end_station_id"]:
    routes_semantic[col] = pd.to_numeric(routes_semantic[col], errors="coerce").astype("Int64")

station_ids_needed = pd.unique(
    pd.concat([routes_semantic["start_station_id"], routes_semantic["end_station_id"]], ignore_index=True)
).dropna().astype("int64").tolist()

print("Unique station IDs needed:", len(station_ids_needed))

# Query coords for those IDs from the same time window you aggregated
ids_param = bigquery.ArrayQueryParameter("ids", "INT64", station_ids_needed)
job_config = bigquery.QueryJobConfig(query_parameters=[ids_param])

sql_coords = """
WITH slice AS (
  SELECT
    start_station_id, start_station_name,
    start_station_latitude  AS slat,
    start_station_longitude AS slon,
    end_station_id,   end_station_name,
    end_station_latitude    AS elat,
    end_station_longitude   AS elon
  FROM `bigquery-public-data.new_york_citibike.citibike_trips`
  WHERE starttime IS NOT NULL
    AND EXTRACT(YEAR  FROM starttime) BETWEEN 2016 AND 2017
    AND EXTRACT(MONTH FROM starttime) BETWEEN 5 AND 10
    AND start_station_id IS NOT NULL AND end_station_id IS NOT NULL
    AND (start_station_id IN UNNEST(@ids) OR end_station_id IN UNNEST(@ids))
),
starts AS (
  SELECT
    start_station_id AS station_id,
    ANY_VALUE(start_station_name) AS station_name,
    AVG(slat) AS lat, AVG(slon) AS lon  -- average in case of slight drift
  FROM slice
  GROUP BY station_id
),
ends AS (
  SELECT
    end_station_id AS station_id,
    ANY_VALUE(end_station_name) AS station_name,
    AVG(elat) AS lat, AVG(elon) AS lon
  FROM slice
  GROUP BY station_id
)
SELECT station_id, ANY_VALUE(station_name) AS station_name,
       ANY_VALUE(lat) AS lat, ANY_VALUE(lon) AS lon
FROM (SELECT * FROM starts UNION ALL SELECT * FROM ends)
GROUP BY station_id
"""
station_geo = bq.query(sql_coords, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
# Make dtypes align
station_geo["station_id"] = pd.to_numeric(station_geo["station_id"], errors="coerce").astype("Int64")

print("Fetched coords for:", len(station_geo))
station_geo.head()


# Merge coords to routes + plot

sxy = station_geo.rename(columns={"station_id":"start_station_id","lat":"slat","lon":"slon"})
exy = station_geo.rename(columns={"station_id":"end_station_id","lat":"elat","lon":"elon"})

plot_df = (
    routes_semantic
      .merge(sxy[["start_station_id","slat","slon"]], on="start_station_id", how="left")
      .merge(exy[["end_station_id","elat","elon"]],   on="end_station_id",   how="left")
)

print(
    "Routes after merge:", len(plot_df),
    "| with both endpoints:", plot_df.dropna(subset=["slat","slon","elat","elon"]).shape[0]
)
plot_df.head()


# Map (center on NYC so you always see something)


# Points (start + end)
pts = pd.concat([
    plot_df.rename(columns={"start_name":"name","slat":"lat","slon":"lon"})[["name","lat","lon"]].assign(kind="start"),
    plot_df.rename(columns={"end_name":"name","elat":"lat","elon":"lon"})[["name","lat","lon"]].assign(kind="end"),
], ignore_index=True).dropna(subset=["lat","lon"])

# If still empty, bail early with a message
if pts.empty:
    raise RuntimeError("No coordinates found after merging. Check that your route period matches the coord query period.")

fig = px.scatter_mapbox(
    pts, lat="lat", lon="lon", hover_name="name",
    zoom=11, height=650, center={"lat": 40.73, "lon": -73.98}
)
fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))

# Lines
for _, r in plot_df.dropna(subset=["slat","slon","elat","elon"]).iterrows():
    fig.add_scattermapbox(
        lat=[r["slat"], r["elat"]],
        lon=[r["slon"], r["elon"]],
        mode="lines",
        hoverinfo="skip"
    )
fig.show()





# ---------- knobs ----------
QUERY_TEXT    = "family friendly scenic waterfront parks for a relaxed weekend"
TOP_K         = 25
YEAR_START    = 2016
YEAR_END      = 2017
MONTHS        = list(range(5, 11))         # May–Oct



# Search top-K routes in BigQuery using VECTOR_SEARCH

qv = model.encode([QUERY_TEXT], normalize_embeddings=True)[0].astype("float64").tolist()
qv_literal = ",".join(map(str, qv))

sql_vec_search = f"""
WITH q AS (SELECT [ {qv_literal} ] AS embedding)
SELECT
  base.route_id,
  base.start_station_id, base.end_station_id,
  base.start_name, base.end_name,
  base.trip_count, base.avg_secs,
  base.blurb,
  distance
FROM
  VECTOR_SEARCH(
    TABLE `{YOUR_PROJECT}.{YOUR_DATASET}.route_embeddings`,
    'embedding',
    TABLE q,
    top_k => {TOP_K},
    distance_type => 'COSINE'
  )
ORDER BY distance ASC
"""
routes_semantic = run_query(sql_vec_search)
if routes_semantic.empty:
    raise RuntimeError("No routes returned. Did route_embeddings get created?")

# Fetch coordinates for just these station IDs from historical trips
ids_needed = pd.unique(
    pd.concat([routes_semantic["start_station_id"], routes_semantic["end_station_id"]], ignore_index=True)
).astype("int64").tolist()

ids_param = bigquery.ArrayQueryParameter("ids", "INT64", ids_needed)
months_param = bigquery.ArrayQueryParameter("months", "INT64", MONTHS)
job_config = bigquery.QueryJobConfig(query_parameters=[ids_param, months_param])

sql_coords = f"""
WITH slice AS (
  SELECT
    start_station_id, start_station_name,
    start_station_latitude  AS slat, start_station_longitude AS slon,
    end_station_id,   end_station_name,
    end_station_latitude    AS elat, end_station_longitude   AS elon
  FROM `bigquery-public-data.new_york_citibike.citibike_trips`
  WHERE starttime IS NOT NULL
    AND EXTRACT(YEAR  FROM starttime) BETWEEN {YEAR_START} AND {YEAR_END}
    AND EXTRACT(MONTH FROM starttime) IN UNNEST(@months)
    AND start_station_id IS NOT NULL AND end_station_id IS NOT NULL
    AND (start_station_id IN UNNEST(@ids) OR end_station_id IN UNNEST(@ids))
),
starts AS (
  SELECT start_station_id AS station_id,
         ANY_VALUE(start_station_name) AS station_name,
         AVG(slat) AS lat, AVG(slon) AS lon
  FROM slice GROUP BY station_id
),
ends AS (
  SELECT end_station_id AS station_id,
         ANY_VALUE(end_station_name) AS station_name,
         AVG(elat) AS lat, AVG(elon) AS lon
  FROM slice GROUP BY station_id
)
SELECT station_id, ANY_VALUE(station_name) AS station_name,
       ANY_VALUE(lat) AS lat, ANY_VALUE(lon) AS lon
FROM (SELECT * FROM starts UNION ALL SELECT * FROM ends)
GROUP BY station_id
"""
station_geo = bq.query(sql_coords, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)

# Merge coords + compute straight-line distance (km)

for c in ["start_station_id","end_station_id"]: routes_semantic[c] = pd.to_numeric(routes_semantic[c], errors="coerce").astype("Int64")
station_geo["station_id"] = pd.to_numeric(station_geo["station_id"], errors="coerce").astype("Int64")

sxy = station_geo.rename(columns={"station_id":"start_station_id","lat":"slat","lon":"slon"})
exy = station_geo.rename(columns={"station_id":"end_station_id","lat":"elat","lon":"elon"})
plot_df = routes_semantic.merge(sxy[["start_station_id","slat","slon"]], on="start_station_id", how="left") \
                         .merge(exy[["end_station_id","elat","elon"]], on="end_station_id",   how="left")

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
    dlat = lat2-lat1; dlon = lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

good = plot_df.dropna(subset=["slat","slon","elat","elon"]).copy()
good["km_straight"] = haversine_km(good["slat"],good["slon"],good["elat"],good["elon"])
good["mins_avg"]    = (good["avg_secs"] / 60).round(1)
good["similarity"]  = (1 - good["distance"]).round(4)

# KPIs

kpi = {
    "Top-K routes": len(good),
    "Median straight-line km": round(good["km_straight"].median(), 2) if len(good) else None,
    "Median avg ride mins":    round(good["mins_avg"].median(), 1)    if len(good) else None,
    "Best similarity":         round(good["similarity"].max(), 4)     if len(good) else None,
}
display(pd.DataFrame([kpi]))

# Bar: Top origins by count within the semantic results

bar_df = good.groupby(["start_name"], as_index=False)["trip_count"].sum().sort_values("trip_count", ascending=False).head(10)
fig_bar = px.bar(bar_df, x="start_name", y="trip_count", title="Top origin stations (within semantic Top-K)")
fig_bar.update_layout(xaxis_title="", yaxis_title="Trips", margin=dict(l=0,r=0,t=40,b=0))
fig_bar.show()

# Map: draw routes

pts = pd.concat([
    good.rename(columns={"start_name":"name","slat":"lat","slon":"lon"})[["name","lat","lon"]].assign(kind="start"),
    good.rename(columns={"end_name":"name","elat":"lat","elon":"lon"})[["name","lat","lon"]].assign(kind="end"),
], ignore_index=True).dropna(subset=["lat","lon"])

fig_map = px.scatter_mapbox(
    pts, lat="lat", lon="lon", hover_name="name",
    zoom=11, height=650, center={"lat":40.73, "lon":-73.98},
)
fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))

for _, r in good.iterrows():
    fig_map.add_scattermapbox(
        lat=[r["slat"], r["elat"]],
        lon=[r["slon"], r["elon"]],
        mode="lines",
        hoverinfo="skip"
    )
fig_map.show()

# Results table (pretty)

show_cols = ["start_name","end_name","mins_avg","km_straight","trip_count","similarity","distance","blurb"]
display(good[show_cols].sort_values("distance").reset_index(drop=True).head(20))





assert "routes_semantic" in globals() and not routes_semantic.empty, "Run the semantic route search first."

def vibe_from_names(a: str, b: str):
    text = f"{a} {b}".lower()
    vibe = []
    if re.search(r"river|riverside|pier|water|bay|harbor|hudson|east\s*river", text):
        vibe.append("waterfront")
    if re.search(r"park|green|square|plaza|garden|central park|battery park|prospect park", text):
        vibe.append("parks")
    if re.search(r"museum|art|gallery|met|moma|whitney", text):
        vibe.append("culture")
    if re.search(r"bridge|brooklyn bridge|manhattan bridge|williamsburg", text):
        vibe.append("bridge views")
    if re.search(r"school|family|playground", text):
        vibe.append("family-friendly")
    return vibe

def make_line(r):
    mins = int(round(r["avg_secs"]/60.0))
    vibe = vibe_from_names(r["start_name"], r["end_name"])
    labels = []
    if mins <= 10: labels.append("short")
    if r["trip_count"] >= 100: labels.append("popular")
    if "parks" in vibe: labels.append("green")
    if "waterfront" in vibe: labels.append("waterfront")
    if "bridge views" in vibe: labels.append("scenic")
    txt = ", ".join(dict.fromkeys(labels))  # de-dupe, keep order
    add = f" ({txt})" if txt else ""
    return f"{r['start_name']} → {r['end_name']}: ~{mins} min; {int(r['trip_count'])} historic trips{add}."

route_summaries = routes_semantic.copy()
route_summaries["summary"] = route_summaries.apply(make_line, axis=1)
route_summaries["source"]  = "rule-based"

# Save for the write-up/demo
pandas_gbq.to_gbq(
    route_summaries[["route_id","start_name","end_name","trip_count","avg_secs","summary","source"]],
    f"{YOUR_DATASET}.route_summaries",
    project_id=YOUR_PROJECT,
    if_exists="replace"
)
print(f"✅ Generated {len(route_summaries)} summaries (rule-based) → saved to {YOUR_PROJECT}.{YOUR_DATASET}.route_summaries")
display(route_summaries.head(10))


!pip -q install transformers accelerate sentencepiece torch --extra-index-url https://download.pytorch.org/whl/cpu


# === Optional: refine lines with a local model (no Vertex) ===
# Install once per session (CPU is fine for short batches)


tok   = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
gen   = pipeline("text2text-generation", model=model, tokenizer=tok, max_length=64)

def refine(s: str):
    prompt = f"Rewrite as one friendly sentence under 25 words for tourists, highlight vibe: {s}"
    return gen(prompt, num_return_sequences=1)[0]["generated_text"]

route_summaries_llm = route_summaries.copy()
route_summaries_llm["summary_llm"] = route_summaries_llm["summary"].apply(refine)
route_summaries_llm["source"] = "flan-t5-base"

import pandas_gbq
pandas_gbq.to_gbq(
    route_summaries_llm[["route_id","start_name","end_name","trip_count","avg_secs","summary_llm","source"]],
    f"{YOUR_DATASET}.route_summaries_llm",
    project_id=YOUR_PROJECT,
    if_exists="replace"
)
print(f"✅ Refined {len(route_summaries_llm)} summaries with FLAN-T5 → saved to {YOUR_PROJECT}.{YOUR_DATASET}.route_summaries_llm")
display(route_summaries_llm.head(10))





TOP_K = 25
routes_semantic.head(TOP_K).to_csv("semantic_routes_topk.csv", index=False)
print("Saved -> semantic_routes_topk.csv")


def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0088
    dlat = radians(b_lat - a_lat); dlon = radians(b_lon - a_lon)
    lat1, lat2 = radians(a_lat), radians(b_lat)
    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2*R*asin(sqrt(h))

# Coordinates from trips

sql_coords = """
WITH slice AS (
  SELECT
    start_station_id, start_station_name,
    start_station_latitude  AS slat,
    start_station_longitude AS slon,
    end_station_id, end_station_name,
    end_station_latitude    AS elat,
    end_station_longitude   AS elon
  FROM `bigquery-public-data.new_york_citibike.citibike_trips`
  WHERE starttime IS NOT NULL
    AND EXTRACT(YEAR  FROM starttime) BETWEEN 2016 AND 2017
    AND EXTRACT(MONTH FROM starttime) BETWEEN 5 AND 10
    AND start_station_id IS NOT NULL AND end_station_id IS NOT NULL
),
starts AS (
  SELECT start_station_id AS station_id,
         ANY_VALUE(start_station_name) AS station_name,
         MAX(slat) AS lat, MAX(slon) AS lon
  FROM slice GROUP BY start_station_id
),
ends AS (
  SELECT end_station_id AS station_id,
         ANY_VALUE(end_station_name) AS station_name,
         MAX(elat) AS lat, MAX(elon) AS lon
  FROM slice GROUP BY end_station_id
),
unioned AS (
  SELECT * FROM starts
  UNION ALL
  SELECT * FROM ends
)
SELECT station_id,
       ANY_VALUE(station_name) AS station_name,
       MAX(lat) AS lat, MAX(lon) AS lon
FROM unioned
GROUP BY station_id
"""
coords_df = run_query(sql_coords)
coords_df["station_id"] = pd.to_numeric(coords_df["station_id"], errors="coerce").astype("Int64")

# Harmonize key dtypes

for col in ["start_station_id", "end_station_id"]:
    routes_semantic[col] = pd.to_numeric(routes_semantic[col], errors="coerce").astype("Int64")

# Join coords to routes

sxy = coords_df.rename(columns={"station_id":"start_station_id","lat":"slat","lon":"slon"})
exy = coords_df.rename(columns={"station_id":"end_station_id","lat":"elat","lon":"elon"})

plot_df = (routes_semantic
           .merge(sxy[["start_station_id","slat","slon"]], on="start_station_id", how="left")
           .merge(exy[["end_station_id","elat","elon"]],   on="end_station_id",   how="left"))
miss_before = len(plot_df)
plot_df = plot_df.dropna(subset=["slat","slon","elat","elon"]).copy()
print(f"Joined {miss_before} → {len(plot_df)} rows with valid coords.")

# Straight-line distance, KPIs, export

plot_df["km_straight"] = plot_df.apply(lambda r: haversine_km(r.slat, r.slon, r.elat, r.elon), axis=1)

TOP_K = 25
top = plot_df.head(TOP_K).copy()
if top.empty:
    raise ValueError("No rows after join; check that routes_semantic has start/end station IDs.")

kpi_df = pd.DataFrame([{
    "top_k": TOP_K,
    "median_straight_km": round(top["km_straight"].median(), 2),
    "median_avg_mins": round((top["avg_secs"]/60).median(), 1),
    "best_similarity": round(1 - top["distance"].min(), 4)
}])

top.to_csv("semantic_routes_topk.csv", index=False)
kpi_df.to_csv("semantic_routes_kpi.csv", index=False)
print("Saved -> semantic_routes_topk.csv, semantic_routes_kpi.csv")
kpi_df


print(routes_semantic.columns.tolist())
print(routes_semantic[["start_station_id","end_station_id"]].head())

