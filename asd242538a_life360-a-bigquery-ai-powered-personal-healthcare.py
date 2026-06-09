#sql
CREATE OR REPLACE TABLE `Life360DataV1.ocr_docs_vec` AS
SELECT
  uri,
  text,
  ml_generate_embedding_result AS emb
FROM ML.GENERATE_EMBEDDING(
  MODEL `Life360DataV1.gecko_embedding`,
  (SELECT text AS content, uri, text FROM `Life360DataV1.ocr_docs_raw`)
);

emb column is an ARRAY<FLOAT64>, required by BigQuery VECTOR_SEARCH.


#sql
-- Step 1: Create a query embedding
WITH qvec AS (
  SELECT ml_generate_embedding_result AS vec
  FROM ML.GENERATE_EMBEDDING(
    MODEL `Life360DataV1.gecko_embedding`,
    (SELECT 'type 2 diabetes low GI diet added sugars limit sodium control' AS content)
  )
),

-- Step 2: Search guideline embeddings
guides AS (
  SELECT ge.base.text AS snippet, ge.distance, 'guide' AS src
  FROM VECTOR_SEARCH(
    TABLE `Life360DataV1.guides_embeddings_norm`,
    'emb',
    (SELECT vec FROM qvec),
    top_k => 3
  ) AS ge
),

-- Step 3: Search OCR embeddings
ocr AS (
  SELECT ov.base.text AS snippet, ov.distance, 'ocr' AS src
  FROM VECTOR_SEARCH(
    TABLE `Life360DataV1.ocr_docs_vec`,
    'emb',
    (SELECT vec FROM qvec),
    top_k => 2
  ) AS ov
),

-- Step 4: Aggregate retrieved context
ctx AS (
  SELECT STRING_AGG(snippet, '\n- ') AS snippets
  FROM (SELECT * FROM guides UNION ALL SELECT * FROM ocr)
),

-- Step 5: Prompt LLM
prompts AS (
  SELECT
    FORMAT("""You are a nutrition coach for type-2 diabetes.

Relevant context:
- %s

Task:
1) Decide if the food is suitable for a T2D patient (Yes/No).
2) Provide evidence-based reasoning (GI, added sugars, sodium, carbs).
3) Suggest 2 alternatives or usage tips if not ideal.
""", ctx.snippets) AS prompt,
    0.2 AS temperature,
    512 AS max_output_tokens
  FROM ctx
)

-- Step 6: Generate answer
SELECT JSON_VALUE(ml_generate_text_result,'$.candidates[0].content.parts[0].text') AS answer
FROM ML.GENERATE_TEXT(
  MODEL `Life360DataV1.gen_text`,
  TABLE prompts
);



# If running in Colab/local, uncomment the next two lines:
# !pip -q install google-cloud-bigquery google-cloud-storage google-cloud-aiplatform google-auth google-auth-oauthlib

import os, json, datetime, textwrap
PROJECT_ID = "life360-472313"         # <--- CHANGE ME
BQ_REGION  = "us-central1"
DOCAI_LOC  = "us"                      # OCR processors are in "us"
DATASET    = "Life360DataV1"

print(PROJECT_ID, BQ_REGION, DOCAI_LOC, DATASET)




# AUTH OPTION A (Colab/Local): will open a browser to login
# from google.colab import auth
# auth.authenticate_user()

# AUTH OPTION B (Kaggle): upload a service-account JSON (as a Kaggle dataset â€œlife360-gcp-keyâ€�)
# then set GOOGLE_APPLICATION_CREDENTIALS to that local path.
# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/kaggle/input/life360-gcp-key/key.json"
# print("Credentials:", os.environ["GOOGLE_APPLICATION_CREDENTIALS"])



from google.cloud import bigquery

bq = bigquery.Client(project=PROJECT_ID, location=BQ_REGION)
print("BigQuery client ready")



# Create dataset if not exists
from google.cloud.exceptions import Conflict

dataset_id = f"{PROJECT_ID}.{DATASET}"
ds = bigquery.Dataset(dataset_id)
ds.location = BQ_REGION
try:
    bq.create_dataset(ds)
    print("Created dataset:", dataset_id)
except Conflict:
    print("Dataset exists:", dataset_id)




#sql
-- Create a BigQuery "remote" model for embeddings (Vertex AI text-embedding-004)
CREATE OR REPLACE MODEL `life360-472313.Life360DataV1.gecko_embedding`
REMOTE WITH CONNECTION `us-central1.vertex-conn`
OPTIONS (endpoint = 'text-embedding-004');



#sql
-- Create a BigQuery "remote" model for text generation (Gemini Flash 2.0)
CREATE OR REPLACE MODEL `life360-472313.Life360DataV1.gen_text`
REMOTE WITH CONNECTION `us-central1.vertex-conn`
OPTIONS (endpoint = 'gemini-2.0-flash-001');



#sql 
-- Minimal demo: guidelines knowledge base
CREATE OR REPLACE TABLE `life360-472313.Life360DataV1.guides_base` AS
SELECT * FROM UNNEST([
  STRUCT(
    'WHO/ACSM activity' AS source,
    'Adults should accumulate at least 150 minutes per week of moderate-intensity aerobic physical activity.' AS text
  ),
  STRUCT(
    'ADA low-GI diet' AS source,
    'Prefer low-glycemic index carbohydrates such as oats, legumes, and whole grains. Distribute carbohydrates evenly across meals.' AS text
  ),
  STRUCT(
    'Added sugars' AS source,
    'Limit added sugars intake; choose whole foods and check nutrition labels for total and added sugars.' AS text
  )
]);



#sql
-- Minimal demo: one OCR row (as if extracted from Document AI)
CREATE OR REPLACE TABLE `life360-472313.Life360DataV1.ocr_docs_raw` AS
SELECT * FROM UNNEST([
  STRUCT(
    'gs://life360-ocr-us-central1/processed/xxx/nutrition_label-0.json' AS uri,
    '''Nutrition Facts
Serving Size: 1 bar (50g)
Calories: 210
Total Fat 8g
Saturated Fat 3g
Cholesterol 0mg
Sodium 120mg
Total Carbohydrate 29g
Dietary Fiber 4g
Total Sugars 12g (Incl. 10g Added Sugars)
Protein 5g''' AS text
  )
]);



#sql
-- Guidelines â†’ embeddings
CREATE OR REPLACE TABLE `life360-472313.Life360DataV1.guides_embeddings` AS
WITH emb AS (
  SELECT
    source,
    text,
    ml_generate_embedding_result AS emb
  FROM ML.GENERATE_EMBEDDING(
    MODEL `life360-472313.Life360DataV1.gecko_embedding`,
    TABLE `life360-472313.Life360DataV1.guides_base`
  )
)
SELECT source, text, emb FROM emb;



#sql 
-- OCR â†’ embeddings  (NOTE: input column must be named 'content')
CREATE OR REPLACE TABLE `life360-472313.Life360DataV1.ocr_docs_vec` AS
WITH cleaned AS (
  SELECT uri, text AS content
  FROM `life360-472313.Life360DataV1.ocr_docs_raw`
  WHERE text IS NOT NULL AND text != ''
),
emb AS (
  SELECT
    uri,
    content AS text,
    ml_generate_embedding_result AS emb
  FROM ML.GENERATE_EMBEDDING(
    MODEL `life360-472313.Life360DataV1.gecko_embedding`,
    TABLE cleaned
  )
)
SELECT uri, text, emb FROM emb;



#sql
-- Quick sanity check
SELECT 'guides_embeddings' AS tbl, COUNT(*) AS n, ARRAY_LENGTH(emb) AS dim
FROM `life360-472313.Life360DataV1.guides_embeddings` LIMIT 1
UNION ALL
SELECT 'ocr_docs_vec', COUNT(*), ARRAY_LENGTH(emb)
FROM `life360-472313.Life360DataV1.ocr_docs_vec` LIMIT 1;



#sql
-- One-shot RAG: retrieve top context from guides + ocr, then generate with Gemini
WITH qvec AS (
  SELECT ml_generate_embedding_result AS vec
  FROM ML.GENERATE_EMBEDDING(
    MODEL `life360-472313.Life360DataV1.gecko_embedding`,
    (SELECT 'type 2 diabetes low GI diet added sugars limit sodium control' AS content)
  )
),
guides AS (
  SELECT ge.base.text AS snippet, ge.distance, 'guide' AS src
  FROM VECTOR_SEARCH(
    TABLE `life360-472313.Life360DataV1.guides_embeddings`,
    'emb',
    (SELECT vec FROM qvec),
    top_k => 3
  ) AS ge
),
ocr AS (
  SELECT ov.base.text AS snippet, ov.distance, 'ocr' AS src
  FROM VECTOR_SEARCH(
    TABLE `life360-472313.Life360DataV1.ocr_docs_vec`,
    'emb',
    (SELECT vec FROM qvec),
    top_k => 2
  ) AS ov
),
ctx AS (
  SELECT STRING_AGG(snippet, '\n- ') AS snippets
  FROM (SELECT * FROM guides UNION ALL SELECT * FROM ocr)
),
prompts AS (
  SELECT
    FORMAT("""You are a nutrition coach for type-2 diabetes.

Relevant context:
- %s

Task:
1) Decide if the food is suitable for a T2D patient (Yes/No).
2) Give evidence-based reasons (GI/added sugars/sodium/carbs).
3) Provide 2 practical substitutes or usage tips if not ideal.
Return plain text (150â€“250 words).""", ctx.snippets) AS prompt,
    0.2 AS temperature,
    512 AS max_output_tokens,
    1   AS candidate_count
  FROM ctx
)
SELECT JSON_VALUE(ml_generate_text_result,'$.candidates[0].content.parts[0].text') AS answer
FROM ML.GENERATE_TEXT(
  MODEL `life360-472313.Life360DataV1.gen_text`,
  TABLE prompts
);



#sql
-- Log table
CREATE TABLE IF NOT EXISTS `life360-472313.Life360DataV1.qa_log`(
  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  question STRING,
  answer   STRING
);



#SQL
-- Stored procedure: sp_rag_answer_with_log(question)
CREATE OR REPLACE PROCEDURE `life360-472313.Life360DataV1.sp_rag_answer_with_log`(question STRING)
BEGIN
  DECLARE answer STRING DEFAULT "";

  SET answer = (
    WITH qvec AS (
      SELECT ml_generate_embedding_result AS vec
      FROM ML.GENERATE_EMBEDDING(
        MODEL `life360-472313.Life360DataV1.gecko_embedding`,
        (SELECT question AS content)
      )
    ),
    guides AS (
      SELECT ge.base.text AS snippet, ge.distance, 'guide' AS src
      FROM VECTOR_SEARCH(
        TABLE `life360-472313.Life360DataV1.guides_embeddings`,
        'emb',
        (SELECT vec FROM qvec),
        top_k => 3
      ) AS ge
    ),
    ocr AS (
      SELECT ov.base.text AS snippet, ov.distance, 'ocr' AS src
      FROM VECTOR_SEARCH(
        TABLE `life360-472313.Life360DataV1.ocr_docs_vec`,
        'emb',
        (SELECT vec FROM qvec),
        top_k => 2
      ) AS ov
    ),
    ctx AS (
      SELECT STRING_AGG(snippet, '\n- ') AS snippets
      FROM (SELECT * FROM guides UNION ALL SELECT * FROM ocr)
    ),
    prompt AS (
      SELECT
        FORMAT("""You are a health & nutrition assistant.
Question: %s

Relevant context:
- %s

Answer clearly with specific, actionable advice.
If the context is insufficient, say so first, then provide general guidance.""", question, ctx.snippets) AS prompt,
        0.2 AS temperature,
        768 AS max_output_tokens,
        1   AS candidate_count
      FROM ctx
    ),
    gen AS (
      SELECT JSON_VALUE(ml_generate_text_result,'$.candidates[0].content.parts[0].text') AS txt
      FROM ML.GENERATE_TEXT(
        MODEL `life360-472313.Life360DataV1.gen_text`,
        TABLE prompt
      )
    )
    SELECT COALESCE(txt, 'Not found in context.') FROM gen LIMIT 1
  );

  INSERT INTO `life360-472313.Life360DataV1.qa_log`(question, answer)
  VALUES (question, answer);

  SELECT question, answer;
END;



#SQL
-- Try it
CALL `life360-472313.Life360DataV1.sp_rag_answer_with_log`('Suggest two low-GI substitutes based on the label.');



#!/usr/bin/env bash
set -euo pipefail

############################################
#            REQUIRED CONFIG (EDIT)        #
############################################
PROJECT_ID="life360-472313"
BQ_REGION="us-central1"                # BigQuery region
DOCAI_LOC="us"                         # Document AI OCR region (OCR processor is us)
PROCESSOR_ID="REPLACE_WITH_YOUR_PROCESSOR_ID"  # â†� Change to your processor ID

# GCS
BUCKET="life360-ocr-us-central1"
IN_DIR="gs://${BUCKET}/incoming/"
OUT_DIR="gs://${BUCKET}/processed/"

# Local sample files (can remove/modify)
FILES=("nutrition_label.png" "case_report.pdf")

# BigQuery
BQ_DATASET="Life360DataV1"
BQ_TABLE_RAW="${BQ_DATASET}.ocr_docs_raw"
VEC_TABLE="${BQ_DATASET}.ocr_docs_vec"
EMB_MODEL="${BQ_DATASET}.gecko_embedding"

############################################
#               Tools / Self-check         #
############################################
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }; }
need gcloud
need jq

gcloud config set project "${PROJECT_ID}" >/dev/null

if [[ -z "${PROCESSOR_ID}" || "${PROCESSOR_ID}" == REPLACE_* ]]; then
  echo "Please set PROCESSOR_ID at the top of the script to your Document AI OCR processor ID"; exit 1
fi

############################################
#     1) Ensure bucket, upload input files #
############################################
echo "== Ensure/Create GCS bucket =="
gcloud storage buckets create "gs://${BUCKET}" --location="${BQ_REGION}" 2>/dev/null || true

echo "== Upload input files to ${IN_DIR} =="
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] && gcloud storage cp "$f" "${IN_DIR}" || echo "WARN: Local file $f not found (can ignore)"
done

echo "== List pending input files =="
gcloud storage ls "${IN_DIR}**" || true
CNT=$(gcloud storage ls "${IN_DIR}**" 2>/dev/null | wc -l | tr -d ' ')
if [[ "${CNT}" -eq 0 ]]; then
  echo "ERROR: Input directory ${IN_DIR} is empty, upload PDF/PNG/JPG/TIF before running."; exit 1
fi

############################################
#     2) Trigger batchProcess & poll       #
############################################
echo "== Get access token =="
TOKEN="$(gcloud auth print-access-token)"

echo "== Trigger Document AI batchProcess =="
REQ=$(cat <<EOF
{
  "inputDocuments": { "gcsPrefix": { "gcsUriPrefix": "${IN_DIR}" } },
  "documentOutputConfig": { "gcsOutputConfig": { "gcsUri": "${OUT_DIR}" } },
  "processOptions": { "ocrConfig": { "enableNativePdfParsing": true } },
  "skipHumanReview": true
}
EOF
)

RESP="$(curl -s -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "x-goog-user-project: ${PROJECT_ID}" \
  -H "Content-Type: application/json; charset=utf-8" \
  "https://${DOCAI_LOC}-documentai.googleapis.com/v1/projects/${PROJECT_ID}/locations/${DOCAI_LOC}/processors/${PROCESSOR_ID}:batchProcess" \
  -d "${REQ}")"

OP_NAME="$(echo "${RESP}" | jq -r '.name // empty')"
if [[ -z "${OP_NAME}" ]]; then
  echo "ERROR: Did not get operation name:"; echo "${RESP}" | jq .; exit 1
fi
echo "Operation: ${OP_NAME}"

echo "== Polling OCR job (max 600s) =="
DEADLINE=$(( $(date +%s) + 600 ))
while true; do
  OP_JSON="$(curl -s -X GET \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "x-goog-user-project: ${PROJECT_ID}" \
    "https://${DOCAI_LOC}-documentai.googleapis.com/v1/${OP_NAME}")"
  DONE="$(echo "${OP_JSON}" | jq -r '.done // false')"
  STATE="$(echo "${OP_JSON}" | jq -r '.metadata.state // empty')"
  ERR="$(echo "${OP_JSON}" | jq -r '.error.message // empty')"
  echo "done=${DONE}  state=${STATE}"

  if [[ "${DONE}" == "true" ]]; then
    [[ -n "${ERR}" && "${ERR}" != "null" ]] && { echo "ERROR: ${ERR}"; echo "${OP_JSON}" | jq .; exit 1; }
    break
  fi
  (( $(date +%s) > DEADLINE )) && { echo "ERROR: Polling timeout"; echo "${OP_JSON}" | jq .; exit 1; }
  sleep 5
done
echo "== OCR Finished =="

############################################
#   3) Select latest op output directory   #
############################################
echo "== Select latest op dir under processed/ =="
LATEST="$(gcloud storage ls "${OUT_DIR}" | grep '/$' | sort | tail -n1)"
[[ -z "${LATEST}" ]] && { echo "ERROR: No output directory found: ${OUT_DIR}"; exit 1; }
echo "Latest output dir: ${LATEST}"

############################################
#  4) Parse *-0.json, extract text â†’ NDJSON #
############################################
echo "== Iterate JSON, extract text (*-0.json only) =="

list_json_files() {
  gcloud storage ls "${LATEST}**" | grep '\-0\.json$' || true
}

mapfile -t JSON_FILES < <(list_json_files)

if [[ ${#JSON_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No *-0.json found under ${LATEST}"
  exit 1
fi

TMP="/tmp/ocr_text.ndjson"
> "${TMP}"

JQ_EXTRACT='
  .document.text // .text // (
    ((.document.pages // .pages) // [])
    | map(
        [
          (.paragraphs // [])[]? | .layout.textAnchor.content? // empty,
          (.blocks     // [])[]? | .layout.textAnchor.content? // empty,
          (.tokens     // [])[]? | .layout.textAnchor.content? // empty
        ]
      )
    | add? // []
    | map(select(. != null and . != ""))
    | join(" ")
  )
'

LINES_WRITTEN=0

for j in "${JSON_FILES[@]}"; do
  echo "processing: $j"

  # 1) Validate JSON
  if ! gcloud storage cat "$j" | jq -e . >/dev/null 2>&1; then
    echo "WARN: Invalid JSON, skipping: $j"
    continue
  fi

  # 2) Read file once
  RAW="$(gcloud storage cat "$j")"

  # 3) Extract text
  TEXT="$(printf '%s' "${RAW}" | jq -r "${JQ_EXTRACT}" 2>/dev/null || echo "")"
  echo "DEBUG: text_len=$(printf '%s' "${TEXT}" | wc -c | tr -d ' ')"

  # 4) Extract source URI (fallback for schema variants)
  URI="$(printf '%s' "${RAW}" | jq -r '.document.uri // .inputConfig.gcsSource.uri // empty' 2>/dev/null || echo "")"
  [[ -z "${URI}" ]] && URI="unknown"

  # 5) Only write if text is non-empty
  if [[ -n "${TEXT}" ]]; then
    printf '%s\n' "$(jq -c --arg uri "${URI}" --arg text "${TEXT}" '{uri:$uri, text:$text}')" >> "${TMP}"
    LINES_WRITTEN=$((LINES_WRITTEN+1))
  else
    echo "WARN: text empty, skipping: $j"
  fi
done

echo "DEBUG: lines_written=${LINES_WRITTEN}"
echo "DEBUG: tmp_file_size=$(wc -c < "${TMP}" | tr -d ' ') bytes"
echo "DEBUG: first_line=$(head -n1 "${TMP}" || true)"

if [[ "${LINES_WRITTEN}" -eq 0 ]]; then
  echo "ERROR: NDJSON has no valid lines, stop upload; check DEBUG/WARN above."
  exit 1
fi

NDJSON_URI="${OUT_DIR}ocr_text.ndjson"
echo "== Upload NDJSON (${LINES_WRITTEN} lines) =="
gcloud storage rm -f "${NDJSON_URI}" >/dev/null 2>&1 || true
gcloud storage cp "${TMP}" "${NDJSON_URI}"
echo "NDJSON uploaded to: ${NDJSON_URI}"

############################################
#        5) BigQuery load (raw table)      #
############################################
echo "== Load into BigQuery: ${BQ_TABLE_RAW} =="

bq --location="${BQ_REGION}" load \
  --source_format=NEWLINE_DELIMITED_JSON \
  --schema=uri:STRING,text:STRING \
  "${PROJECT_ID}:${BQ_TABLE_RAW}" \
  "${NDJSON_URI}"

echo "== Preview first 5 rows =="
bq --location="${BQ_REGION}" query --use_legacy_sql=false "
SELECT uri, SUBSTR(text,1,200) AS snippet
FROM \`${PROJECT_ID}.${BQ_TABLE_RAW}\`
LIMIT 5;"

echo "ğŸ�‰ DONE"


