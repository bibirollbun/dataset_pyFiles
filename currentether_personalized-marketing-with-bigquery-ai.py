%pip install --upgrade bigframes google-cloud-automl google-cloud-translate google-ai-generativelanguage tensorflow 


from google.cloud import bigquery
import pandas_gbq
from IPython.display import display, HTML
import pathlib

# ---------- CONFIG ----------
PROJECT    = "impressive-mile-469303-k4"
DATASET    = "BQAI_KAGGLE"  # US region
CONNECTION = "impressive-mile-469303-k4.US.kaggle_bqai"
MODEL      = f"{PROJECT}.{DATASET}.GEMINI_HTML"
LOCATION   = "US"

bq = bigquery.Client(project=PROJECT)


sql_step1 = """
-- 0) Workspace
CREATE SCHEMA IF NOT EXISTS `{PROJECT}.{DATASET}` OPTIONS(location="US");

-- 1) Remote model â†’ Gemini
CREATE OR REPLACE MODEL `{MODEL}`
REMOTE WITH CONNECTION `{CONNECTION}`
OPTIONS (endpoint = 'gemini-2.5-flash');

-- 2) Customer features from theLook
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.customer_features` AS
WITH orders AS (
  SELECT o.order_id, o.user_id, o.created_at, o.status
  FROM `bigquery-public-data.thelook_ecommerce.orders` o
  WHERE status NOT IN ('Cancelled','Returned')
),
items AS (
  SELECT order_id, product_id, sale_price, created_at AS item_time
  FROM `bigquery-public-data.thelook_ecommerce.order_items`
),
prods AS (
  SELECT id AS product_id, brand, category, department
  FROM `bigquery-public-data.thelook_ecommerce.products`
),
j AS (
  SELECT
    o.user_id, o.order_id, i.sale_price, i.item_time,
    p.brand, p.category, p.department
  FROM orders o
  JOIN items  i USING (order_id)
  JOIN prods  p USING (product_id)
),
agg AS (
  SELECT
    user_id,
    COUNT(DISTINCT order_id)            AS orders_cnt,
    COUNT(*)                            AS items_cnt,
    ROUND(SUM(sale_price),2)            AS revenue,
    MAX(item_time)                      AS last_purchase_ts,
    APPROX_TOP_COUNT(category, 3)       AS top_cats,
    APPROX_TOP_COUNT(brand, 3)          AS top_brands
  FROM j
  GROUP BY user_id
)
SELECT
  user_id, orders_cnt, items_cnt, revenue, last_purchase_ts,
  DATE_DIFF(CURRENT_DATE(), DATE(last_purchase_ts), DAY) AS recency_days,
  NTILE(5) OVER (ORDER BY DATE_DIFF(CURRENT_DATE(), DATE(last_purchase_ts), DAY)) AS r_score,
  NTILE(5) OVER (ORDER BY orders_cnt DESC) AS f_score,
  NTILE(5) OVER (ORDER BY revenue DESC)    AS m_score,
  top_cats[SAFE_OFFSET(0)].value  AS fav_category,
  top_brands[SAFE_OFFSET(0)].value AS fav_brand
FROM agg;
""".format(PROJECT=PROJECT, DATASET=DATASET, MODEL=MODEL, CONNECTION=CONNECTION)

job = bq.query(sql_step1, location=LOCATION)
job.result()
print("Step 1 complete: dataset, model, and customer_features created.")


N_CUSTOMERS = 10   # how many customers to generate for (cost control)
N_RENDER    = 8     # how many to preview inline


sql_prompts = """
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.email_prompts_html` AS
SELECT
  user_id,
  CONCAT(
    -- ==== OUTPUT CONTRACT (strict) ====
    'Return ONLY compact JSON exactly like {{"subject":"...","html_b64":"..."}}. ',
    'No markdown, no code fences, no extra prose, no extra keys. ',
    'Escape quotes per JSON. Both keys must always be present (use empty strings if unsure). ',
    'The value of "html_b64" must be BASE64 of a UTF-8 HTML fragment.',

    -- ==== CONTENT GUARDRAILS (privacy-safe) ====
    'Do NOT mention exact order counts, spend totals, or specific dates. ',
    'Do NOT say "we know". Keep personalization to general tastes only (category/brand). ',
    'Avoid repetition, filler, or awkward phrasing.',

    -- ==== TONE & STYLE ====
    'Tone: warm, conversational, human; short sentences; friendly and helpful. ',
    'No shouty ALL CAPS. Use at most one tasteful emoji in the subject (optional).',

    -- ==== STRUCTURE RULES (for consistent HTML emails) ====
    'Subject: max 7 words, natural and catchy. ',
    'HTML fragment: single column, mobile-friendly (max-width 560px), inline CSS only, ',
    'no <html> or <body> wrapper; include a heading, 2 short paragraphs, and one prominent CTA button. ',
    'CTA links to a relative slug like "/', REPLACE(LOWER(COALESCE(fav_category, 'shop')), ' ', '-'), '".',

    -- ==== PERSONALIZATION INPUTS (soft) ====
    'Customer focus: category=', COALESCE(fav_category, 'New Arrivals'),
    '; brand=', COALESCE(fav_brand, 'our favorite brands'), '. ',

    -- ==== COPY HINTS (make it read like a person) ====
    'Open with appreciation. ',
    'Suggest new items in the focus category/brand. ',
    'Close with a friendly nudge to browse.'
  ) AS prompt
FROM `{PROJECT}.{DATASET}.customer_features`
WHERE orders_cnt >= 2
LIMIT {N_CUSTOMERS};
""".format(PROJECT=PROJECT, DATASET=DATASET, N_CUSTOMERS=N_CUSTOMERS)


bq.query(sql_prompts, location=LOCATION).result()
print("Step 2 complete: Prompts table created.")


import pandas_gbq

pandas_gbq.read_gbq(
    "SELECT * FROM `impressive-mile-469303-k4.BQAI_KAGGLE.email_prompts_html` LIMIT 5",
    project_id=PROJECT
)


sql_gen = f"""
-- 1) Generate (leave parsing to us)
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.gen_raw_json` AS
SELECT
  p.user_id,
  TO_JSON_STRING(ml_generate_text_result) AS result_json
FROM ML.GENERATE_TEXT(
  MODEL `{MODEL}`,
  TABLE `{PROJECT}.{DATASET}.email_prompts_html`,
  STRUCT(
    0.7 AS temperature,
    4096 AS max_output_tokens
  )
) AS gen
JOIN `{PROJECT}.{DATASET}.email_prompts_html` p
ON TRUE;

-- 2) Extract text â†’ strip fences â†’ pull JSON â†’ get fields
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.generated_html_emails` AS
WITH text_part AS (
  SELECT
    user_id,
    JSON_VALUE(result_json, '$.candidates[0].content.parts[0].text') AS body
  FROM `{PROJECT}.{DATASET}.gen_raw_json`
),
strip_fences AS (
  SELECT
    user_id,
    REGEXP_REPLACE(body, r'(?is)^\\s*```(?:json)?\\s*|\\s*```\\s*$', '') AS body_clean
  FROM text_part
),
grab_object AS (
  SELECT
    user_id,
    REGEXP_EXTRACT(body_clean, r'(?s)\\{{.*\\}}') AS json_str,
    body_clean
  FROM strip_fences
),
parsed AS (
  SELECT
    user_id,
    body_clean,
    SAFE.PARSE_JSON(json_str) AS j
  FROM grab_object
),
fields AS (
  SELECT
    user_id,
    body_clean,
    JSON_VALUE(j, '$.subject')  AS subject,
    JSON_VALUE(j, '$.html_b64') AS html_b64_raw
  FROM parsed
),
prep AS (
  SELECT
    user_id,
    body_clean,
    subject,
    REGEXP_REPLACE(COALESCE(html_b64_raw,''), r'[^A-Za-z0-9+/=]', '')  AS b64_std,
    REGEXP_REPLACE(COALESCE(html_b64_raw,''), r'[^A-Za-z0-9\\-_]', '') AS b64_url
  FROM fields
),
padded AS (
  SELECT
    user_id,
    body_clean,
    subject,
    CONCAT(b64_std, REPEAT('=', MOD(4 - MOD(LENGTH(b64_std), 4), 4))) AS std_pad,
    CONCAT(REPLACE(REPLACE(b64_url, '-', '+'), '_', '/'),
           REPEAT('=', MOD(4 - MOD(LENGTH(b64_url), 4), 4)))          AS url_pad
  FROM prep
),
decoded AS (
  SELECT
    user_id,
    body_clean,
    subject,
    SAFE.FROM_BASE64(std_pad) AS std_bytes,
    SAFE.FROM_BASE64(url_pad) AS url_bytes
  FROM padded
),
utf8 AS (
  SELECT
    user_id,
    body_clean,
    subject,
    SAFE_CAST(std_bytes AS STRING) AS html_std,
    SAFE_CAST(url_bytes AS STRING) AS html_url
  FROM decoded
)
SELECT
  user_id,
  subject,
  COALESCE(
    html_std,
    html_url,
    CASE WHEN REGEXP_CONTAINS(body_clean, r'(?i)<\\s*[a-z][^>]*>') THEN body_clean END
  ) AS html
FROM utf8
WHERE subject IS NOT NULL;
"""

bq.query(sql_gen, location=LOCATION).result()
print("Step 3 completed: Generated and parsed HTML emails.")


df = pandas_gbq.read_gbq(f"""
  SELECT user_id, subject, html
  FROM `{PROJECT}.{DATASET}.generated_html_emails`
  WHERE subject IS NOT NULL AND html IS NOT NULL
  LIMIT {N_RENDER}
""", project_id=PROJECT)

for _, row in df.iterrows():
    display(HTML(f"""
    <div style="border:1px solid #ddd;margin:16px 0;padding:12px;">
      <div style="font:600 16px/1.2 system-ui,Arial;margin-bottom:8px;">
        {row['subject']}
      </div>
      <div style="border:1px dashed #ccc;padding:12px;max-width:560px;">
        {row['html']}
      </div>
    </div>
    """))

# Save standalone HTMLs
outdir = pathlib.Path("/kaggle/working/html_emails")
outdir.mkdir(parents=True, exist_ok=True)
for _, row in df.iterrows():
    with open(outdir / f"email_{row['user_id']}.html", "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n" + row["html"])
print(f"Saved {len(df)} files to {outdir}")


from graphviz import Digraph

# Create flow diagram
dot = Digraph(comment="Promo Email Generation Flow", format="png")
dot.attr(rankdir="LR", size="8")

# Nodes
dot.node("A", "1. Workspace Setup\n(BigQuery Schema + Remote Gemini Model)", shape="box", style="rounded,filled", fillcolor="lightblue")
dot.node("B", "2. Feature Engineering\nRFM + Fav Category/Brand", shape="box", style="rounded,filled", fillcolor="lightyellow")
dot.node("C", "3. Prompt Table\n(JSON contract: subject + html_b64)", shape="box", style="rounded,filled", fillcolor="lightpink")
dot.node("D", "4. Text Generation\nML.GENERATE_TEXT", shape="box", style="rounded,filled", fillcolor="lightgreen")
dot.node("E", "5. Parsing & Decoding\nExtract subject + decode html_b64", shape="box", style="rounded,filled", fillcolor="wheat")
dot.node("F", "6. Visualization\nRender HTML in Kaggle Notebook", shape="box", style="rounded,filled", fillcolor="lightgrey")

# Edges
dot.edges(["AB", "BC", "CD", "DE", "EF"])

# Save & render
flow_path = "/mnt/data/promo_email_flowchart"
dot.render(flow_path, format="png", cleanup=True)

flow_path + ".png"



from google.cloud import bigquery
bq = bigquery.Client(project=PROJECT)

sql = f"""
-- Create (or replace) the text-embedding model
CREATE OR REPLACE MODEL `impressive-mile-469303-k4.BQAI_KAGGLE.EMB_TXT`
REMOTE WITH CONNECTION `projects/impressive-mile-469303-k4/locations/us/connections/kaggle_bqai`
OPTIONS (endpoint = 'text-embedding-004');

-- Materialize product embeddings as FLOAT64 vectors
CREATE OR REPLACE TABLE `impressive-mile-469303-k4.BQAI_KAGGLE.product_embeddings` AS
WITH gen AS (
  SELECT
    id,
    brand,
    category,
    department,
    name,
    ml_generate_embedding_result AS emb_strs
  FROM ML.GENERATE_EMBEDDING(
    MODEL `impressive-mile-469303-k4.BQAI_KAGGLE.EMB_TXT`,
    (
      SELECT
        CONCAT(brand, ' ', category, ' ', name) AS content,
        CAST(id AS STRING) AS id,
        brand, category, department, name
      FROM `bigquery-public-data.thelook_ecommerce.products`
    )
  )
)
SELECT
  CAST(id AS INT64) AS product_id,
  brand,
  category,
  department,
  name,
  ARRAY(SELECT CAST(x AS FLOAT64) FROM UNNEST(emb_strs) AS x) AS embedding
from gen
;

"""
bq.query(sql, location=LOCATION).result()
print("S2.1 updated: EMB_TXT model created and product_embeddings materialized with FLOAT64 vectors.")


sql = f"""
-- Distinct purchases per user (ignore cancelled/returned)
CREATE OR REPLACE TABLE `impressive-mile-469303-k4.BQAI_KAGGLE.user_purchases` AS
SELECT DISTINCT o.user_id, oi.product_id
FROM `bigquery-public-data.thelook_ecommerce.orders` o
JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi USING(order_id)
WHERE o.status NOT IN ('Cancelled','Returned');

-- Embeddings for purchased products
CREATE OR REPLACE TABLE `impressive-mile-469303-k4.BQAI_KAGGLE.user_item_embeddings` AS
SELECT
  up.user_id,
  prod.id AS product_id,
  -- cast the ARRAY<STRING> into ARRAY<FLOAT64>
  ARRAY(SELECT CAST(x AS FLOAT64) FROM UNNEST(gen.ml_generate_embedding_result) AS x) AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `impressive-mile-469303-k4.BQAI_KAGGLE.EMB_TXT`,
  (
    SELECT
      CONCAT(p.brand, ' ', p.category, ' ', p.name) AS content,
      p.id
    FROM `bigquery-public-data.thelook_ecommerce.products` p
  ),
  STRUCT(TRUE AS flatten_json_output)
) AS gen
JOIN `bigquery-public-data.thelook_ecommerce.products` prod
  ON gen.id = prod.id
JOIN `impressive-mile-469303-k4.BQAI_KAGGLE.user_purchases` up
  ON up.product_id = prod.id;

-- Average embedding per user
CREATE OR REPLACE TABLE `impressive-mile-469303-k4.BQAI_KAGGLE.user_embeddings` AS
WITH exploded AS (
  SELECT
    user_id,
    pos,
    val
  FROM `impressive-mile-469303-k4.BQAI_KAGGLE.user_item_embeddings`,
  UNNEST(embedding) AS val WITH OFFSET AS pos
),
dim_avg AS (
  SELECT
    user_id,
    pos,
    AVG(val) AS mean_val
  FROM exploded
  GROUP BY user_id, pos
)
SELECT
  user_id,
  ARRAY_AGG(mean_val ORDER BY pos) AS user_emb
FROM dim_avg
GROUP BY user_id;
"""
bq.query(sql, location=LOCATION).result()
print("S2.2 done: user_purchases, user_item_embeddings, user_embeddings created.")



sql = f"""
-- S2.3 â€” User â†’ Product recommendations via cosine similarity (no VECTOR_SEARCH)

DECLARE TOP_K INT64 DEFAULT 5;

CREATE OR REPLACE TABLE `impressive-mile-469303-k4.BQAI_KAGGLE.user_product_recos` AS
WITH
u AS (
  SELECT user_id, user_emb
  FROM `impressive-mile-469303-k4.BQAI_KAGGLE.user_embeddings`
  WHERE user_emb IS NOT NULL AND ARRAY_LENGTH(user_emb) > 0
  limit 10
),
p AS (
  SELECT product_id, name, brand, category, department, embedding
  FROM `impressive-mile-469303-k4.BQAI_KAGGLE.product_embeddings`
  WHERE embedding IS NOT NULL AND ARRAY_LENGTH(embedding) > 0
),
already_bought AS (
  SELECT DISTINCT o.user_id, oi.product_id
  FROM `bigquery-public-data.thelook_ecommerce.orders` AS o
  JOIN `bigquery-public-data.thelook_ecommerce.order_items` AS oi
    USING (order_id)
  WHERE o.status NOT IN ('Cancelled','Returned')
),
pairwise AS (
  -- compute dot, norms, cosine per (user, product)
  SELECT
    u.user_id,
    p.product_id,
    p.name,
    p.brand,
    p.category,
    p.department,
    SAFE_DIVIDE(SUM(ue * pe),
                (SQRT(SUM(ue * ue)) * SQRT(SUM(pe * pe)))) AS cosine_sim
  FROM u
  CROSS JOIN p
  JOIN UNNEST(u.user_emb) AS ue WITH OFFSET pos_u
  JOIN UNNEST(p.embedding) AS pe WITH OFFSET pos_p
    ON pos_u = pos_p
  GROUP BY
    u.user_id, p.product_id, p.name, p.brand, p.category, p.department
),
ranked AS (
  SELECT
    pw.*,
    ROW_NUMBER() OVER (
      PARTITION BY pw.user_id
      ORDER BY pw.cosine_sim DESC, pw.product_id
    ) AS rn
  FROM pairwise AS pw
  LEFT JOIN already_bought AS ab
    ON pw.user_id = ab.user_id
   AND pw.product_id = ab.product_id
  WHERE ab.product_id IS NULL
    AND pw.cosine_sim IS NOT NULL
)

SELECT
  user_id,
  product_id,
  name,
  brand,
  category,
  department,
  cosine_sim,
  rn
FROM ranked
WHERE rn <= TOP_K
ORDER BY user_id, rn;
"""

bq.query(sql, location=LOCATION).result()
print("S2.3 done: User â†’ Product recommendations via cosine similarity (no VECTOR_SEARCH).")


sql = f"""
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.user_recs_top3` AS
SELECT
  ups.user_id,
  ARRAY_AGG(STRUCT(pe.name AS name, pe.category AS category, pe.brand AS brand)
            ORDER BY ups.cosine_sim DESC LIMIT 3) AS recs
FROM `{PROJECT}.{DATASET}.user_product_recos` ups
JOIN `{PROJECT}.{DATASET}.product_embeddings` pe
  ON pe.product_id = ups.product_id
GROUP BY ups.user_id;
"""
bq.query(sql, location=LOCATION).result()
print("S2.4 done: user_recs_top3 created.")


sql = f"""
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.email_prompts_embed` AS
SELECT
  u.user_id,
  CONCAT(
    'Return ONLY compact JSON: {{"subject":"...","html_b64":"..."}}. ',
    'No code fences, no markdown, no extra keys or prose. ',
    'Escape quotes per JSON; both keys must be present. ',
    'html_b64 must be BASE64 of a UTF-8 HTML fragment. ',
    'Tone: warm, conversational, natural; short sentences. ',
    'Do NOT mention exact purchases, spend, or dates. ',
    'HTML: single-column (max-width 560px), inline CSS, heading + 2 short paragraphs + one CTA button. ',
    '\\nRecommended items:\\n- ',
    COALESCE(r.recs[SAFE_OFFSET(0)].name, 'Fresh arrivals'),
    '\\n- ', COALESCE(r.recs[SAFE_OFFSET(1)].name, 'Best sellers'),
    '\\n- ', COALESCE(r.recs[SAFE_OFFSET(2)].name, 'Editor picks'),
    '\\nCTA should link to "/recommended".'
  ) AS prompt
FROM `{PROJECT}.{DATASET}.user_product_recos` u
LEFT JOIN `{PROJECT}.{DATASET}.user_recs_top3` r
  ON r.user_id = u.user_id;
"""
bq.query(sql, location=LOCATION).result()
print("S2.5 done: email_prompts_embed created.")


sql = f"""
-- Generate raw JSON via ML.GENERATE_TEXT
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.gen_raw_json_embed` AS
SELECT
  p.user_id,
  TO_JSON_STRING(ml_generate_text_result) AS result_json
FROM ML.GENERATE_TEXT(
  MODEL `{PROJECT}.{DATASET}.GEMINI_HTML`,
  TABLE `{PROJECT}.{DATASET}.email_prompts_embed`,
  STRUCT(
    0.7 AS temperature,
    4096 AS max_output_tokens
  )
) AS gen
JOIN `{PROJECT}.{DATASET}.email_prompts_embed` p
ON TRUE;

-- Parse â†’ decode Base64 (standard + URL-safe) â†’ raw HTML fallback
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.generated_html_emails_embed` AS
WITH text_part AS (
  SELECT
    user_id,
    JSON_VALUE(result_json, '$.candidates[0].content.parts[0].text') AS body
  FROM `{PROJECT}.{DATASET}.gen_raw_json_embed`
),
strip_fences AS (
  SELECT
    user_id,
    REGEXP_REPLACE(body, r'(?is)^\\s*```(?:json)?\\s*|\\s*```\\s*$', '') AS body_clean
  FROM text_part
),
grab_object AS (
  SELECT
    user_id,
    REGEXP_EXTRACT(body_clean, r'(?s)\\{{.*\\}}') AS json_str,
    body_clean
  FROM strip_fences
),
parsed AS (
  SELECT
    user_id,
    body_clean,
    SAFE.PARSE_JSON(json_str) AS j
  FROM grab_object
),
fields AS (
  SELECT
    user_id,
    body_clean,
    JSON_VALUE(j, '$.subject')  AS subject,
    JSON_VALUE(j, '$.html_b64') AS html_b64_raw
  FROM parsed
),
prep AS (
  SELECT
    user_id, body_clean, subject,
    REGEXP_REPLACE(COALESCE(html_b64_raw,''), r'[^A-Za-z0-9+/=]', '')  AS b64_std,
    REGEXP_REPLACE(COALESCE(html_b64_raw,''), r'[^A-Za-z0-9\\-_]', '') AS b64_url
  FROM fields
),
padded AS (
  SELECT
    user_id, body_clean, subject,
    CONCAT(b64_std, REPEAT('=', MOD(4 - MOD(LENGTH(b64_std), 4), 4))) AS std_pad,
    CONCAT(REPLACE(REPLACE(b64_url, '-', '+'), '_', '/'),
           REPEAT('=', MOD(4 - MOD(LENGTH(b64_url), 4), 4)))          AS url_pad
  FROM prep
),
decoded AS (
  SELECT
    user_id, body_clean, subject,
    SAFE.FROM_BASE64(std_pad) AS std_bytes,
    SAFE.FROM_BASE64(url_pad) AS url_bytes
  FROM padded
),
utf8 AS (
  SELECT
    user_id, body_clean, subject,
    SAFE_CAST(std_bytes AS STRING) AS html_std,
    SAFE_CAST(url_bytes AS STRING) AS html_url
  FROM decoded
)
SELECT
  user_id,
  subject,
  COALESCE(
    html_std,
    html_url,
    CASE WHEN REGEXP_CONTAINS(body_clean, r'(?i)<\\s*[a-z][^>]*>') THEN body_clean END
  ) AS html
FROM utf8
WHERE subject IS NOT NULL;
"""
bq.query(sql, location=LOCATION).result()
print("S2.6 done: generated_html_emails_embed created.")


import pandas_gbq
from IPython.display import HTML, display

N_RENDER = 8
df = pandas_gbq.read_gbq(f"""
  SELECT user_id, subject, html
  FROM `{PROJECT}.{DATASET}.generated_html_emails_embed`
  WHERE subject IS NOT NULL AND html IS NOT NULL
  LIMIT {N_RENDER}
""", project_id=PROJECT)

for _, row in df.iterrows():
    display(HTML(f"""
    <div style="border:1px solid #ddd;margin:16px 0;padding:12px;">
      <div style="font:600 16px/1.2 system-ui,Arial;margin-bottom:8px;">
        {row['subject']}
      </div>
      <div style="border:1px dashed #ccc;padding:12px;max-width:560px;">
        {row['html']}
      </div>
    </div>
    """))
print(f"Rendered {len(df)} emails.")

