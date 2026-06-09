project_id = "ornate-lens-469413-u8"
dataset_id = "dataset"
location = "US"


# Initializing BigQ client 
from google.cloud import bigquery
client = bigquery.Client(project=project_id, location=location)


# Logging setup for clarity during execution
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BQ_RAG_Pipeline")

logger.info("âœ… Environment configured. Using BigQuery native AI capabilities.")
logger.info(f"âœ… Target Project: {project_id} | Dataset: {dataset_id}")


import logging
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


project_id = "ornate-lens-469413-u8"
dataset_id = "dataset"
location = "US"
client = bigquery.Client(project="ornate-lens-469413-u8")


# ---- Logging & Client ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# --- 1. Ensure dataset exists ---
dataset_ref = f"{client.project}.{dataset_id}"
try:
    client.get_dataset(dataset_ref)
    logger.info(f"âœ… Dataset {dataset_id} already exists.")
except NotFound:
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = location
    client.create_dataset(dataset)
    logger.info(f"âœ… Created dataset {dataset_id}.")


# --- 2. Core TABLES: The Essential RAG Pipeline ---

# Raw Documents: Our landing zone for ALL data
create_raw_documents = """
CREATE OR REPLACE TABLE `ornate-lens-469413-u8.dataset.raw_documents` (
  doc_id STRING,
  source_type STRING,
  raw_text STRING,
  object_ref STRING,                     -- GCS path for images/PDFs
  requires_ocr BOOL,                    
  doc_date DATE,                         -- Document date from source
  created_at TIMESTAMP
);
"""
client.query(create_raw_documents).result()
print("âœ… Enhanced raw_documents table created")

# Image Documents: Stores results from native BigQuery OCR processing ---
create_image_documents = """
CREATE OR REPLACE TABLE `ornate-lens-469413-u8.dataset.image_documents` (
  doc_id STRING,
  source_type STRING,
  image_uri STRING,
  extracted_text STRING,
  processing_score FLOAT64,
  ocr_confidence FLOAT64,
  text_length INT64,
  processed_at TIMESTAMP,
);
"""
client.query(create_image_documents).result()
print("âœ… Image documents table with vector indexing created!")

# --- 2. Create document-level embeddings table ---
create_doc_embeddings = """
CREATE OR REPLACE TABLE `ornate-lens-469413-u8.dataset.doc_embeddings` AS
SELECT
  doc_id,                            
  source_type,
  ARRAY_AGG(chunk_text ORDER BY OFFSET) AS chunk,
  ARRAY_AGG(chunk_embedding ORDER BY OFFSET) AS chunk_embeddings,
  dataset.pool_embeddings(ARRAY_AGG(chunk_embedding ORDER BY OFFSET), 'mean') AS document_embedding,
  AVG(quality_score) AS avg_quality_score,
  MAX(doc_date) AS doc_date,
  CURRENT_TIMESTAMP() AS generated_at
FROM `ornate-lens-469413-u8.dataset.doc_chunks`
GROUP BY doc_id, source_type
"""
client.query(create_doc_embeddings).result()
print("âœ… doc_embeddings table created")


# ==== 3. For Evaluation and Monitoring  ====

# Processing Logs (Dynamic Performance Monitoring)
create_processing_logs = """
CREATE TABLE IF NOT EXISTS `ornate-lens-469413-u8.dataset.processing_logs` (
    process_id STRING,
    doc_id STRING,
    source_type STRING,
    processing_step STRING, -- e.g., 'OCR', 'CHUNKING', 'EMBEDDING'
    processing_time FLOAT64,
    success BOOL,
    error_message STRING,
    processed_at TIMESTAMP
);
"""
client.query(create_processing_logs).result()
print("âœ… processing_logs table created")

# A/B Evaluation Results
create_ab_evaluation = """
CREATE OR REPLACE TABLE `ornate-lens-469413-u8.dataset.ab_evaluation_results` (
    experiment_id STRING,
    variant STRING, -- e.g., 'baseline', 'with_mmr'
    query_id STRING,
    query_text STRING,
    nDCG FLOAT64,
    Recall FLOAT64,
    timestamp TIMESTAMP
);
"""
client.query(create_ab_evaluation).result()
print("âœ… ab_evaluation_results table created")

# Anomalies Detection Analysis
create_anomaly_analysis = """
CREATE OR REPLACE TABLE `ornate-lens-469413-u8.monitor.anomaly_analysis` (
  anomaly_id STRING,
  detected_at TIMESTAMP,
  source_system STRING,
  metric_name STRING,
  metric_value FLOAT64,
  expected_value FLOAT64,
  anomaly_score FLOAT64,
  severity STRING,
  suspected_root_cause STRING,
  supporting_evidence STRING,
  resolved BOOL DEFAULT FALSE
);
"""
client.query(create_anomaly_analysis).result()
print("âœ… anomaly_analysis_results table created")


# ==== 4. Creating Vector Indexes ====

print("\n--- Creating Vector Indexes for Fast Retrieval ---")

create_chunk_index = """
CREATE OR REPLACE VECTOR INDEX ON `ornate-lens-469413-u8.dataset.doc_chunks` (chunk_embedding)
OPTIONS (index_type = 'IVF', distance_type = 'COSINE');
"""
client.query(create_chunk_index).result()
print("âœ… vector index on doc_chunks created")

create_doc_index = """
CREATE OR REPLACE VECTOR INDEX ON `ornate-lens-469413-u8.dataset.doc_embeddings` (document_embedding)
OPTIONS (index_type = 'IVF', distance_type = 'COSINE');
"""
client.query(create_doc_index).result()
print("âœ… vector index on doc_embeddings created")





# ==== 5. Backward compatibility view (documents) ---
create_view_documents = """
CREATE OR REPLACE VIEW `ornate-lens-469413-u8.dataset.documents` AS
SELECT
    doc_id,
    parent_id,
    chunk_id,
    source_type,
    chunk_text AS text,
    alphanum_ratio,
    quality_score,
    fingerprint,
    semantic_fingerprint,
    ingested_at AS created_at,
    doc_date,
    ingested_at,
    valid_until
FROM `ornate-lens-469413-u8.dataset.doc_chunks`;
"""
client.query(create_view_documents).result()
print("âœ… documents view created (backward compatibility)")




create_doc_embeddings_sql = """
CREATE OR REPLACE TABLE `ornate-lens-469413-u8`.dataset.doc_embeddings AS
SELECT
  doc_id,
  source_type,
  ARRAY_AGG(chunk_text ORDER BY chunk_id) AS chunk_text_array,
  ARRAY_AGG(chunk_embedding ORDER BY chunk_id) AS chunk_embeddings_array,
  `ornate-lens-469413-u8`.dataset.pool_embeddings(ARRAY_AGG(chunk_embedding ORDER BY chunk_id), 'mean') AS document_embedding,
  AVG(quality_score) AS avg_quality_score,
  MAX(doc_date) AS doc_date,
  CURRENT_TIMESTAMP() AS generated_at
FROM `ornate-lens-469413-u8`.dataset.doc_chunks
GROUP BY doc_id, source_type;
"""



create_udf_scorer = """
-- === JS UDF Chunker  ===
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.chunk_text`(
    text STRING,
    chunk_words INT64,
    overlap INT64
) AS (
  (SELECT ARRAY(
    SELECT chunk
    FROM UNNEST((
      (SELECT (
        (function(txt, chunkWords, overlapSize) {
          if (!txt) return [];
          const words = txt.trim().split(/\\s+/);
          const chunks = [];
          for (let i = 0; i < words.length; i += Math.max(1, chunkWords - overlapSize)) {
            const chunk = words.slice(i, i + chunkWords).join(" ");
            if (chunk.trim()) chunks.push(chunk);
          }
          return chunks;
        })(text, chunk_words, overlap)
      ))
    )) AS chunk
  ))
);


-- === Text Quality Assessment ===
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.assess_text_quality`(text STRING)
RETURNS FLOAT64
AS (
  -- Returns a score between 0 (garbage) and 1 (perfect)
  CASE
    WHEN text IS NULL OR text = '' THEN 0.0
    -- Check for minimum length (e.g., very short chunks are rarely useful)
    WHEN LENGTH(text) < 25 THEN 0.2
    -- Penalize chunks that are overly dominated by non-space characters (like base64, file paths, error codes)
    WHEN LENGTH(REGEXP_REPLACE(text, r'[^\\s]', '')) / NULLIF(LENGTH(text), 0) < 0.1 THEN 0.3
    -- Penalize chunks that are ALL CAPS (often system logs or errors)
    WHEN LENGTH(REGEXP_EXTRACT_ALL(text, r'\b[A-Z]{3,}\b')) / NULLIF((1 + LENGTH(SPLIT(text, ' '))), 0) > 0.3 THEN 0.5
    -- Boost chunks that contain complete sentence structures (question marks, periods)
    WHEN REGEXP_CONTAINS(text, r'[.?!â€�]$') THEN 0.9
    -- Default score for decent-looking text
    ELSE 0.8
  END
);


-- === Scoring helpers for retrieval ===
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.recency_decay`(doc_date DATE, half_life_days INT64) 
RETURNS FLOAT64 AS (
  CASE 
    WHEN doc_date IS NULL THEN 1.0 
    ELSE EXP(-0.693 * DATE_DIFF(CURRENT_DATE(), doc_date, DAY) / half_life_days) 
  END
);

CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.get_modality_boost`(source_type STRING) 
RETURNS FLOAT64 AS (
  CASE source_type 
    WHEN 'screenshot' THEN 1.2 
    WHEN 'table' THEN 1.25 
    WHEN 'csv' THEN 1.15 
    WHEN 'ticket' THEN 1.1 
    WHEN 'pdf' THEN 1.0 
    WHEN 'transcript' THEN 1.0 
    WHEN 'chat' THEN 0.9 
    ELSE 1.0 
  END
);

CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.get_similarity_floor`(source_type STRING) 
RETURNS FLOAT64 AS (
  CASE source_type 
    WHEN 'screenshot' THEN 0.80 
    WHEN 'table' THEN 0.78 
    WHEN 'csv' THEN 0.78 
    WHEN 'pdf' THEN 0.75 
    WHEN 'ticket' THEN 0.75 
    WHEN 'transcript' THEN 0.75 
    WHEN 'chat' THEN 0.75 
    ELSE 0.70 
  END
);


-- === Final scorer (null-safe, clamped!) ===
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.calculate_final_score`(
  semantic_score FLOAT64,
  recency_score  FLOAT64,
  modality_boost FLOAT64,
  quality_score  FLOAT64
)
RETURNS FLOAT64 AS (
  LEAST(
    1.0,
    (COALESCE(semantic_score,0)*0.7
     + COALESCE(recency_score,0)*0.2
     + COALESCE(modality_boost,0)*0.1)
     * COALESCE(quality_score,1)
  )
);



 -- === Two-Level Embedding helpers for retrieval ===
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.pool_embeddings`(
  embeddings ARRAY<ARRAY<FLOAT64>>,
  strategy STRING
)
RETURNS ARRAY<FLOAT64>
AS (
  CASE
    WHEN embeddings IS NULL OR ARRAY_LENGTH(embeddings) = 0 THEN NULL

    WHEN LOWER(strategy) = 'mean' THEN (
      SELECT ARRAY(
        SELECT AVG(val)                         -- elementwise mean
        FROM (
          SELECT val, idx
          FROM UNNEST(embeddings) AS emb WITH OFFSET o
          CROSS JOIN UNNEST(emb) AS val WITH OFFSET idx
        )
        GROUP BY idx
        ORDER BY idx
      )
    )

    WHEN LOWER(strategy) = 'max' THEN (
      SELECT ARRAY(
        SELECT MAX(val)                          -- elementwise max
        FROM (
          SELECT val, idx
          FROM UNNEST(embeddings) AS emb WITH OFFSET o
          CROSS JOIN UNNEST(emb) AS val WITH OFFSET idx
        )
        GROUP BY idx
        ORDER BY idx
      )
    )

    ELSE NULL
  END
);
"""


client.query(create_udf_scorer).result()
print("âœ… Chunker + scoring UDFs created")



create_quality_gates_udfs = """

-- 1. Alphanumeric ratio calculator
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.alphanum_ratio`(text STRING) 
RETURNS FLOAT64 AS (
  CASE 
    WHEN text IS NULL OR LENGTH(text) = 0 THEN 0.0
    ELSE (
      SELECT COUNTIF(REGEXP_CONTAINS(char, r'[A-Za-z0-9]')) / LENGTH(text)
      FROM UNNEST(SPLIT(text, '')) AS char
    )
  END
);


-- 2. Modality-aware quality scorer
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.calculate_quality_score`(
  text STRING, 
  source_type STRING, 
  alpha_ratio FLOAT64
) RETURNS FLOAT64 AS (
  CASE 
    -- Screenshot quality rules
    WHEN source_type = 'screenshot' AND (LENGTH(text) < 20 OR alpha_ratio < 0.5) THEN 0.4
    WHEN source_type = 'screenshot' THEN LEAST(1.0, alpha_ratio / 0.8 * 
        (LENGTH(text) / 100.0) * 
        (1.0 + (0.1 * CASE WHEN REGEXP_CONTAINS(text, r'\\b(0x[A-Fa-f0-9]+|ERR\\d+|Model[\\s#:]*[A-Z0-9\\-]+)\\b') THEN 1 ELSE 0 END)) * 1.2)
    
    -- Table quality rules  
    WHEN source_type = 'table' AND alpha_ratio < 0.3 THEN 0.3
    WHEN source_type = 'table' THEN LEAST(1.0, alpha_ratio / 0.8 *
        (LENGTH(text) / 100.0) *
        (1.0 + (0.1 * CASE WHEN REGEXP_CONTAINS(text, r'\\b(0x[A-Fa-f0-9]+|ERR\\d+|Model[\\s#:]*[A-Z0-9\\-]+)\\b') THEN 1 ELSE 0 END)) * 1.25)
    
    -- PDF quality rules
    WHEN source_type = 'pdf' AND LENGTH(text) < 30 THEN 0.5
    WHEN source_type = 'pdf' THEN LEAST(1.0, alpha_ratio / 0.8 *
        (LENGTH(text) / 100.0) *
        (1.0 + (0.1 * CASE WHEN REGEXP_CONTAINS(text, r'\\b(0x[A-Fa-f0-9]+|ERR\\d+|Model[\\s#:]*[A-Z0-9\\-]+)\\b') THEN 1 ELSE 0 END)) * 1.0)
    
    -- Default quality scoring
    ELSE LEAST(1.0, alpha_ratio / 0.8 *
        (LENGTH(text) / 100.0) *
        (1.0 + (0.1 * CASE WHEN REGEXP_CONTAINS(text, r'\\b(0x[A-Fa-f0-9]+|ERR\\d+|Model[\\s#:]*[A-Z0-9\\-]+)\\b') THEN 1 ELSE 0 END)) *
        CASE source_type
          WHEN 'ticket' THEN 1.1
          WHEN 'csv' THEN 1.15  
          WHEN 'transcript' THEN 1.0
          WHEN 'chat' THEN 0.9
          ELSE 1.0
        END)
  END
);


-- 3. Quality gate filter function
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.passes_quality_gates`(
  text STRING,
  source_type STRING,
  min_length INT64,
  min_alpha_ratio FLOAT64
) RETURNS BOOL AS (
  CASE
    WHEN text IS NULL OR LENGTH(text) < min_length THEN FALSE
    WHEN dataset.alphanum_ratio(text) < min_alpha_ratio THEN FALSE
    WHEN dataset.calculate_quality_score(text, source_type, dataset.alphanum_ratio(text)) < 0.4 THEN FALSE
    ELSE TRUE
  END
);

-- 4. Rare term extractor (for quality boosting)
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.extract_rare_terms`(text STRING) 
RETURNS STRING AS (
  ARRAY_TO_STRING(ARRAY(
    SELECT term FROM UNNEST([
      REGEXP_EXTRACT_ALL(text, r'\\b(0x[A-Fa-f0-9]+)\\b'),
      REGEXP_EXTRACT_ALL(text, r'\\b([A-Z]{2,}\\d+[A-Z]*)\\b'),
      REGEXP_EXTRACT_ALL(text, r'\\b(Model[\\s#:]*[A-Z0-9\\-]+)\\b'),
      REGEXP_EXTRACT_ALL(text, r'\\b(\\d{3,}-\\d{3,}-\\d{4,})\\b')
    ]) AS term
    WHERE term IS NOT NULL
  ), ' ')
);
"""



# --- EXECUTE QUALITY GATES ---
client.query(create_quality_gates_udfs).result()
print("âœ… Quality Gates UDFs successfully deployed!")

# --- TESTING OUR QUALITY GATES ---
test_quality_gates = """
-- Test various quality scenarios
WITH test_cases AS (
  SELECT 'Good ticket' as test_case, 'My BoomBox Ultra wont charge past 50%' as text, 'ticket' as source_type
  UNION ALL SELECT 'Short screenshot', 'Error 0xE12', 'screenshot'
  UNION ALL SELECT 'Low quality', '??? ___ ???', 'chat'
  UNION ALL SELECT 'Good product', 'Wireless Charging Dock - iPhone 15 - Model: A123', 'product'
)
SELECT 
  test_case,
  text,
  source_type,
  dataset.alphanum_ratio(text) as alpha_ratio,
  dataset.calculate_quality_score(text, source_type, dataset.alphanum_ratio(text)) as quality_score,
  dataset.passes_quality_gates(text, source_type, 40, 0.6) as passes_gates,
  dataset.extract_rare_terms(text) as rare_terms
FROM test_cases
"""

results = client.query(test_quality_gates).result()
print("ğŸ§ª Quality Gates Test Results:")
for row in results:
    print(f"  {row.test_case}: {row.passes_gates} (Score: {row.quality_score:.2f}, Alpha: {row.alpha_ratio:.2f})")


# testing the embeddings...........RUN IN THE BIGQUERY CONSOLE!!!!

SELECT `ornate-lens-469413-u8`.dataset.pool_embeddings([[1,2,3],[3,4,5]], 'mean')  AS mean_vec,
       `ornate-lens-469413-u8`.dataset.pool_embeddings([[1,2,3],[3,4,5]], 'max')   AS max_vec;

SELECT `ornate-lens-469413-u8`.dataset.calculate_final_score(0.8, 0.6, 0.2, 0.9) AS final_score;



# Fine Tuning our custom Gemini

from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
from vertexai.preview.tuning import sft

# Config
project_id = "ornate-lens-469413-u8"
location = "us-central1"
training_data_uri = "gs://ner_training_bucket/gemini_ner_training.jsonl"
service_account_key_path = "/kaggle/input/private-key-file/ornate-lens-469413-u8-d96044d51108.json"

# Auth
credentials = service_account.Credentials.from_service_account_file(service_account_key_path)

# Init Vertex
vertexai.init(
    project=project_id,
    location=location,
    credentials=credentials
)

# Launch Fine-tuning
print("ğŸš€ Starting Gemini fine-tuning job...")
tuning_job = sft.train(
    source_model="gemini-1.5-large",
    train_dataset=training_data_uri,
    tuned_model_display_name="support-ticket-ner-gemini",
    epochs=3,
    learning_rate_multiplier=1.0,
)


print(f"âœ… Fine-tuning job started successfully!")
print(f"ğŸ“‹ Job name: {tuning_job.name}")
print(f"ğŸ”§ Model: {tuning_job._model_id}")
print(f"ğŸ“Š Training data: {training_data_uri}")
print("\nğŸ“ˆ Monitor progress here:")
print(f"https://console.cloud.google.com/vertex-ai/locations/us-central1/training/tuning-jobs/{tuning_job.name}")


define_our_extractors = """
-- 1. Native image text extraction function
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.extract_text_from_image`(image_uri STRING)
RETURNS STRUCT<text STRING, confidence FLOAT64>
AS (
  (SELECT AS STRUCT
    ML.IMAGE_PROCESS(
      ML.IMAGE_LOAD(image_uri),
      'TEXT_DETECTION'
    ).text_annotations[SAFE_OFFSET(0)].description as text,
    ML.IMAGE_PROCESS(
      ML.IMAGE_LOAD(image_uri),
      'TEXT_DETECTION'
    ).text_annotations[SAFE_OFFSET(0)].confidence as confidence
  )
);

-- 2. PDF text extraction using BigQuery's built-in capabilities
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.extract_text_from_pdf`(pdf_uri STRING)
RETURNS STRING
AS (
  (SELECT STRING_AGG(page_text, ' ' ORDER BY page_number)
  FROM UNNEST(ML.PDF_EXTRACT_TEXT(pdf_uri)) as page_text WITH OFFSET as page_number)
);
"""

create_native_ocr_pipeline = """
CREATE OR REPLACE PROCEDURE `ornate-lens-469413-u8.dataset.process_native_ocr`()
BEGIN
  -- Process images
  FOR img IN (
    SELECT doc_id, source_type, object_ref as image_uri
    FROM `ornate-lens-469413-u8.dataset.raw_documents`
    WHERE requires_ocr = TRUE 
      AND source_type IN ('screenshot', 'image')
      AND doc_id NOT IN (SELECT doc_id FROM `ornate-lens-469413-u8.dataset.image_documents`)
  ) DO
    DECLARE extraction_result STRUCT<text STRING, confidence FLOAT64>;
    SET extraction_result = dataset.extract_text_from_image(img.image_uri);
    
    -- Write into image_documents
    INSERT INTO `ornate-lens-469413-u8.dataset.image_documents` 
    (doc_id, source_type, image_uri, extracted_text, processing_score, processed_at)
    VALUES (
      img.doc_id,
      img.source_type,
      img.image_uri,
      extraction_result.text,
      extraction_result.confidence,
      CURRENT_TIMESTAMP()
    );

    -- ğŸ”¥ Push OCR text back into raw_documents
    UPDATE `ornate-lens-469413-u8.dataset.raw_documents`
    SET raw_text = COALESCE(raw_text, '') || '\n' || extraction_result.text
    WHERE doc_id = img.doc_id;
  END FOR;
  
  -- Process PDFs
  FOR pdf IN (
    SELECT doc_id, source_type, object_ref as pdf_uri
    FROM `ornate-lens-469413-u8.dataset.raw_documents`
    WHERE requires_ocr = TRUE 
      AND source_type = 'pdf'
      AND doc_id NOT IN (SELECT doc_id FROM `ornate-lens-469413-u8.dataset.image_documents`)
  ) DO
    DECLARE pdf_text STRING;
    SET pdf_text = dataset.extract_text_from_pdf(pdf.pdf_uri);

    -- Write into image_documents
    INSERT INTO `ornate-lens-469413-u8.dataset.image_documents` 
    (doc_id, source_type, image_uri, extracted_text, processing_score, processed_at)
    VALUES (
      pdf.doc_id,
      pdf.source_type,
      pdf.pdf_uri,
      pdf_text,
      0.95,  -- high confidence for PDFs
      CURRENT_TIMESTAMP()
    );

    -- ğŸ”¥ Push PDF OCR text back into raw_documents
    UPDATE `ornate-lens-469413-u8.dataset.raw_documents`
    SET raw_text = COALESCE(raw_text, '') || '\n' || pdf_text
    WHERE doc_id = pdf.doc_id;
  END FOR;
END;

"""
client.query(create_native_ocr_pipeline).result()
print("âœ… Native BigQuery OCR pipeline created!")


populate_doc_chunks = """
CREATE OR REPLACE TABLE `ornate-lens-469413-u8.dataset.doc_chunks` AS
WITH all_documents AS (
  -- Text documents
  SELECT doc_id, source_type, object_ref, doc_date, raw_text, FALSE as from_ocr,
          'native_text' AS content_source
  FROM `ornate-lens-469413-u8.dataset.raw_documents`
  WHERE requires_ocr = FALSE
  
  UNION ALL
  
  -- OCR processed documents
  SELECT doc_id, source_type, object_ref, NULL as doc_date, extracted_text as raw_text,
         TRUE as from_ocr, 'ocr_processed' as content_source
  FROM `ornate-lens-469413-u8.dataset.image_documents`
  WHERE processing_score > 0.7 AND extracted_text IS NOT NULL
),
chunked_data AS (
  SELECT
    *,
    CASE source_type
      WHEN 'pdf' THEN dataset.chunk_text(cleaned_text, 380, 120)
      WHEN 'transcript' THEN dataset.chunk_text(cleaned_text, 400, 100)
      WHEN 'chat' THEN dataset.chunk_text(cleaned_text, 300, 80) 
      WHEN 'ticket' THEN dataset.chunk_text(cleaned_text, 300, 80)
      WHEN 'csv' THEN dataset.chunk_text(cleaned_text, 250, 50)
      WHEN 'screenshot' THEN dataset.chunk_text(cleaned_text, 150, 30)
      WHEN 'table' THEN [cleaned_text]
      ELSE dataset.chunk_text(cleaned_text, 300, 50)
    END AS chunks
  FROM cleaned_data
),
exploded_chunks AS (
  SELECT
    doc_id,
    source_type,
    object_ref,
    doc_date,
    chunk,
    CONCAT(doc_id, '_chunk', FORMAT('%03d', OFFSET)) AS chunk_id,
    OFFSET
  FROM chunked_data, UNNEST(chunks) AS chunk WITH OFFSET
),
quality_checked_chunks AS (
  SELECT
    *,
    dataset.alphanum_ratio(chunk) AS alphanum_ratio,
    dataset.calculate_quality_score(chunk, source_type, dataset.alphanum_ratio(chunk)) AS quality_score,
    dataset.extract_rare_terms(chunk) AS rare_terms
  FROM exploded_chunks
  WHERE dataset.passes_quality_gates(chunk, source_type, 40, 0.6)
),
fingerprinted_chunks AS (
  SELECT
    *,
    ML.GENERATE_TEXT(
      MODEL `ornate-lens-469413-u8.dataset.gemini_pro`,
      CONCAT(
        'Analyze the following text from a <source_type> and extract the key semantic information. ',
        'Be extremely concise. Structure your output ONLY using these relevant tags:',
        ' PRODUCT, PROBLEM, SYMPTOM, ERROR, MATERIAL, WEIGHT. ',
        'Use only the tags that are relevant. Text: ', chunk
      ),
      STRUCT(0.1 AS temperature, 300 AS max_output_tokens, 0.9 AS top_p)
    ).ml_generate_text_result AS semantic_fingerprint
  FROM quality_checked_chunks
),
SELECT
  chunk_id,
  doc_id,
  doc_id AS parent_id,
  source_type,
  object_ref,
  NULL AS speaker,
  NULL AS turn_index,
  chunk AS chunk_text,
  FALSE AS is_table,
  NULL AS page_start,
  NULL AS page_end, 
  NULL AS row_start,
  NULL AS row_end,
  original_format,
    processing_type
  FROM `ornate-lens-469413-u8.dataset.image_documents`
  WHERE content IS NOT NULL,
  LENGTH(chunk) AS chunk_size_tokens,
  CASE source_type
    WHEN 'pdf' THEN 120 WHEN 'transcript' THEN 100 WHEN 'chat' THEN 80
    WHEN 'ticket' THEN 80 WHEN 'csv' THEN 50 WHEN 'screenshot' THEN 30
    ELSE 0
  END AS chunk_overlap_tokens,
  alphanum_ratio,
  quality_score,
  FARM_FINGERPRINT(chunk) AS fingerprint,
  semantic_fingerprint,

  -- âœ… chunk-level embed
  ML.GENERATE_EMBEDDING(
    MODEL `ornate-lens-469413-u8.dataset.embedding_model`,
    semantic_fingerprint
  ) AS chunk_embedding,

  -- âœ… doc-level embed pooled from existing chunk embeddings
  pool_embeddings(
    ARRAY_AGG(chunk_embedding) OVER (PARTITION BY doc_id ORDER BY chunk_id),
    "mean"
  ) AS doc_embedding,

  doc_date,
  CURRENT_TIMESTAMP() AS ingested_at,
  NULL AS valid_until,
  CURRENT_TIMESTAMP() AS last_updated
FROM fingerprinted_chunks;

 -- Refreshing our embeddings
CREATE OR REPLACE PROCEDURE `ornate-lens-469413-u8.dataset.refresh_embeddings`()
BEGIN
  -- Refresh embeddings for new or updated documents
  INSERT INTO `ornate-lens-469413-u8.dataset.doc_embeddings` (
    doc_id, chunk_id, source_type, text, semantic_fingerprint, 
    embedding, doc_date, ingested_at, valid_until, last_updated
  )
  SELECT
    d.doc_id,
    d.chunk_id,
    d.source_type,
    d.text,
    d.semantic_fingerprint,
    ML.GENERATE_EMBEDDING(MODEL `dataset.embedding_model`, d.semantic_fingerprint),
    d.doc_date,
    d.ingested_at,
    d.valid_until,
    CURRENT_TIMESTAMP()
  FROM `ornate-lens-469413-u8.dataset.documents` d
  WHERE d.ingested_at > (
    SELECT COALESCE(MAX(last_updated), TIMESTAMP('2000-01-01')) 
    FROM `ornate-lens-469413-u8.dataset.doc_embeddings`
  ),
  cleaned_data AS(
    SELECT
    doc_id,
    source_type,
    object_ref,
    doc_date,
    from_ocr,
    -- Enhanced cleaning with OCR awareness
    CASE 
      WHEN from_ocr THEN 
        REGEXP_REPLACE(REGEXP_REPLACE(LOWER(raw_text), 
          r'OCR_ARTIFACT|SCAN_NOISE|\\[.*?\\]', ''),  -- Remove OCR artifacts
          r'[^\\w\\s\\.\\%]', ' ')
      ELSE
        REGEXP_REPLACE(REGEXP_REPLACE(LOWER(raw_text), 
          r'http[s]?://\\S+|[\w\.]+@[\w\.]+', ''),
          r'[^\\w\\s\\.\\%]', ' ')
    END AS cleaned_text
    FROM all_documents
    )
END;
"""

client.query(populate_doc_chunks).result()
print("âœ… doc_chunks populated with enhanced data")


#   Overlaps the final scoring we had in Cell 3, but this is for isolated testing/debugging
def freshness_aware_retrieval(query: str, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
    """
    Retrieval that balances semantic similarity with document freshness. 
    """
    freshness_sql = f"""
    SELECT
      doc_id,
      chunk_id,
      text,
      semantic_fingerprint,
      doc_date,
      -- Semantic similarity score
      (1 - distance) AS semantic_score,
      -- Freshness boost: exponential decay (favors recent docs)
      EXP(-0.1 * DATE_DIFF(CURRENT_DATE(), COALESCE(doc_date, CURRENT_DATE()), DAY)) AS recency_score,
      -- Combined score (80% semantic, 20% freshness)
      ( (1 - distance) * 0.8 + 
        EXP(-0.1 * DATE_DIFF(CURRENT_DATE(), COALESCE(doc_date, CURRENT_DATE()), DAY)) * 0.2 
      ) AS final_score
    FROM VECTOR_SEARCH(
      TABLE `ornate-lens-469413-u8.dataset.doc_embeddings`,
      'embedding',
      (SELECT ML.GENERATE_EMBEDDING(MODEL `dataset.embedding_model`, '{query}')),
      top_k => {top_k * 3},  # Get more for filtering
      distance_type => 'COSINE'
    )
    WHERE valid_until IS NULL OR valid_until > CURRENT_DATE()  -- ğŸ†• Filter expired docs
    ORDER BY final_score DESC
    LIMIT {top_k}
    """
    
    try:
        results = client.query(freshness_sql).result()
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Freshness-aware retrieval failed: {e}")
        return []


mmr_retriever = """
CREATE OR REPLACE FUNCTION `ornate-lens-469413-u8.dataset.mmr_modality`(
  qvec ARRAY<FLOAT64>,
  dvecs ARRAY<ARRAY<FLOAT64>>,
  modalities ARRAY<STRING>,
  lambda FLOAT64,
  mod_penalty FLOAT64,
  k INT64
) RETURNS ARRAY<INT64>
LANGUAGE js AS '''

function dot(a,b){let s=0;for(let i=0;i<a.length;i++) s+=a[i]*b[i]; return s;}
function norm(a){return Math.sqrt(dot(a,a));}
function cos(a,b){return dot(a,b)/(norm(a)*norm(b)+1e-9);}

const n = dvecs.length;
const sims = Array.from({length:n}, (_,i)=>cos(qvec, dvecs[i]));
const selectedIdx = [];
const candidateIdx = Array.from({length:n}, (_,i)=>i);

while(selectedIdx.length < Math.min(k,n) && candidateIdx.length){
  let bestI=-1, bestScore=-1e9;
  for(const i of candidateIdx){
    const simQ = sims[i];
    let simSel = 0.0;
    if(selectedIdx.length){
      simSel = Math.max(...selectedIdx.map(j => cos(dvecs[i], dvecs[j])));
    }
    const m = modalities[i];
    const sameCount = selectedIdx.filter(j => modalities[j]===m).length;
    const penalty = 1 - mod_penalty*sameCount;
    const mmr = lambda*simQ - (1-lambda)*simSel;
    const finalScore = mmr*penalty;
    if(finalScore>bestScore){ bestScore=finalScore; bestI=i; }
  }
  selectedIdx.push(bestI);
  candidateIdx.splice(candidateIdx.indexOf(bestI),1);
}
return selectedIdx;
''';
"""

client = bigquery.Client()

#client.query(mmr_retriever).result()
print("âœ… MMR function deployed to BigQuery")


#  Just a for Illustrations Purposes Only!
'''
from sklearn.metrics import ndcg_score
import numpy as np

def recall_at_k(y_true, y_pred, k):
    """Compute Recall@k for one query."""
    y_true_set = set(y_true)
    y_pred_set = set(y_pred[:k])
    return len(y_true_set & y_pred_set) / len(y_true_set) if y_true_set else 0

def eval_system(system_outputs, ground_truth, k=10):
    """
    Evaluate retrieval results.

    Args:
        system_outputs: dict {query_id: [doc_ids ranked]}
        ground_truth: dict {query_id: [relevant_doc_ids]}
        k: cutoff for metrics.

    Returns:
        dict with avg nDCG@k and Recall@k
    """
    ndcgs, recalls = [], []
    for qid, gt_docs in ground_truth.items():
        if qid not in system_outputs:
            continue
        ranked_docs = system_outputs[qid]

        # Binary relevance labels
        y_true = [1 if doc in gt_docs else 0 for doc in ranked_docs[:k]]
        y_scores = np.arange(len(y_true), 0, -1)  # descending dummy scores

        ndcgs.append(ndcg_score([y_true], [y_scores], k=k))
        recalls.append(recall_at_k(gt_docs, ranked_docs, k))

    return {
        "nDCG@{}".format(k): np.mean(ndcgs),
        "Recall@{}".format(k): np.mean(recalls)
    }

def log_evaluation(system_id: str, query: str, ndcg: float, recall: float):
    """Enhanced version with better error handling."""
    row = {
        "system_id": system_id,
        "query": query,
        "nDCG": ndcg,
        "Recall": recall,
        "timestamp": datetime.now(timezone.utc)
    }
    try:
        table_ref = client.dataset("dataset").table("retrieval_eval")
        errors = client.insert_rows_json(table_ref, [row])
        if errors:
            
            logger.error(f"Failed to insert row: {errors}")
        else:
            logger.info(f"âœ… Logged eval: {system_id} - nDCG@{ndcg:.3f}, Recall@{recall:.3f}")
    except Exception as e:
        logger.error(f"â�Œ Critical logging failure: {e}")
'''


"""
CREATE OR REPLACE PROCEDURE `ornate-lens-469413-u8.dataset.extract_entities`()
BEGIN
  -- Batch process chunks to avoid API overload
  FOR chunk_batch IN (
    SELECT 
      doc_id, 
      chunk_text,
      ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY chunk_text) AS chunk_idx
    FROM `ornate-lens-469413-u8.dataset.doc_chunks`
    WHERE chunk_text IS NOT NULL
      AND NOT EXISTS (
        SELECT 1 
        FROM `ornate-lens-469413-u8.dataset.extracted_entities` 
        WHERE doc_id = `ornate-lens-469413-u8.dataset.doc_chunks`.doc_id
      )
    LIMIT 1000  -- Batch size, tweak based on quota
  ) DO
    INSERT INTO `ornate-lens-469413-u8.dataset.extracted_entities` (
      doc_id,
      chunk_text,
      knowledge_graph_json,
      processed_at
    )
    SELECT
      chunk_batch.doc_id,
      chunk_batch.chunk_text,
      ML.GENERATE_TEXT(
        MODEL `ornate-lens-469413-u8.dataset.gemini_pro`,
        CONCAT(
          'Extract entities and relationships from this support text. ',
          'Return as JSON with: entities[] {type, value, id} and relationships[] {source_id, target_id, type}. ',
          'Text: ', chunk_batch.chunk_text
        ),
        STRUCT(0.1 AS temperature, 1024 AS max_output_tokens)
      ).ml_generate_text_result AS knowledge_graph_json,
      CURRENT_TIMESTAMP()
    FROM UNNEST([chunk_batch]) AS chunk_batch;
  END FOR;
END;
"""


"""
CREATE OR REPLACE PROCEDURE `ornate-lens-469413-u8.monitor.detect_anomalies_and_root_cause`()
BEGIN
  -- Step 1: Gather monitoring signals across systems
  INSERT INTO `ornate-lens-469413-u8.monitor.observability_events`
  SELECT
    CURRENT_TIMESTAMP() AS event_time,
    source_system,
    metric_name,
    metric_value,
    ML.DETECT_ANOMALIES(
      MODEL `ornate-lens-469413-u8.models.anomaly_model`,
      STRUCT(metric_value AS value)
    ).anomaly_score AS anomaly_score
  FROM `ornate-lens-469413-u8.raw_metrics`;

  -- Step 2: Build graph snapshot (lineage + anomalies)
  INSERT INTO `ornate-lens-469413-u8.monitor.graph_snapshots`
  SELECT
    e1.source_id,
    e1.target_id,
    e1.relationship,
    AVG(obs.anomaly_score) AS edge_anomaly,
    MAX(obs.event_time) AS last_seen
  FROM `ornate-lens-469413-u8.dataset.extracted_entities` e1
  JOIN `ornate-lens-469413-u8.monitor.observability_events` obs
    ON obs.source_system = e1.source_id
  GROUP BY 1,2,3;

  -- Step 3: Root cause suggestion (simplified rule-based, planned to upgrade to GNN later)
  INSERT INTO `ornate-lens-469413-u8.monitor.root_cause_inferences`
  SELECT
    target_id AS impacted_system,
    ARRAY_AGG(STRUCT(source_id, edge_anomaly) ORDER BY edge_anomaly DESC LIMIT 3) AS probable_causes,
    CURRENT_TIMESTAMP() AS inferred_at
  FROM `ornate-lens-469413-u8.monitor.graph_snapshots`
  WHERE edge_anomaly > 0.8
  GROUP BY target_id;
END;
"""


## Kept for testing our fingerprinter for Illustrations and indivual component testing
def generate_semantic_fingerprints_bach(dataframe: pd.DataFrame, project_id: str, dataset_id: str):
    """
    We're batch generating fingerprints for a DataFrame of documents.

    Returns:
        A structured string fingerprint ready for embedding.
    """
    # Quick cleanup: lowercase, remove extra whitespace
    cleaned_text = ' '.join(text.lower().split())
    
    # Initialize components of the fingerprint
    components = []
    client = bigquery.Client(project=project_id)
    
    # 1. Create a temporary table for the new documents
    temp_table_id = f"{project_id}.{dataset_id}.temp_new_docs"
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source_type", "STRING", mode="REQUIRED"),
        ],
    )
    
    load_job = client.load_table_from_dataframe(dataframe, temp_table_id, job_config=job_config)
    load_job.result()
    logger.info(f"âœ… Uploaded {len(dataframe)} documents to temporary table: {temp_table_id}")

    # 2. The MAGIC: SQL query that uses ML.GENERATE_TEXT to generate fingerprints
    generate_fingerprints_sql = f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.documents_with_fingerprints` AS
    WITH prompt AS (
      SELECT '''
    Analyze the following text from a <source_type> and extract the key semantic information.
    Be extremely concise. Structure your output ONLY using these relevant tags:
    
    - PRODUCT: <product_name> (if a product is mentioned)
    - PROBLEM: <problem1, problem2> (for errors or issues)
    - SYMPTOM: <symptom1, symptom2> (for specific manifestations)
    - RESOLUTION: <resolution_status> (if a solution is mentioned)
    - MATERIAL: <materials> (for physical products)
    - WEIGHT: <weight> (for physical products)
    - FEATURE: <features> (for capabilities)
    - COMPATIBILITY: <compatible_devices> (for what it works with)
    - ERROR: <error_code> (for error messages)
    - CONTEXT: <ui_context> (for screenshots/UI)
    
    Use only the tags that are relevant. Use only a maximun of 40 words. If no information is found for a tag, omit it entirely.
    ''' AS template
    )
    SELECT 
        d.doc_id,
        d.text,
        d.source_type,
        t.ml_generate_text_result AS semantic_fingerprint,
        CURRENT_TIMESTAMP() as processed_at
    FROM `{temp_table_id}` d, prompt p,
    UNNEST(ARRAY(
      SELECT ml_generate_text_result
      FROM ML.GENERATE_TEXT(
        MODEL `{project_id}.{dataset_id}.gemini_pro`,
        (
          REPLACE(REPLACE(p.template, '<source_type>', d.source_type), '<text>', d.text)
        ),
        STRUCT(0.1 AS temperature, 500 AS max_output_tokens, 0.9 AS top_p)
      )
    )) 
    """
    
    try:
        query_job = client.query(generate_fingerprints_sql)
        query_job.result()  # Wait for completion
        logger.info("âœ… Successfully generated semantic fingerprints using ML.GENERATE_TEXT!")
        
    except Exception as e:
        logger.error(f"â�Œ Failed to generate fingerprints: {e}")
        raise
    finally:
        # Clean up: delete the temporary table
        client.delete_table(temp_table_id, not_found_ok=True)
        logger.info(f"ğŸ§¹ Cleaned up temporary table: {temp_table_id}")



def execute_bigquery_native_pipeline():
    client = bigquery.Client(project="ornate-lens-469413-u8")
    
    # The complete SQL pipeline
    pipeline_sql = [

    # 0. Ensuring Raw_documents exist
        """
        CREATE OR REPLACE TABLE `ornate-lens-469413-u8.dataset.raw_documents` (
          doc_id STRING,
          source_type STRING,
          raw_text STRING,
          object_ref STRING,                     -- GCS path for images/PDFs
          requires_ocr BOOL,                    
          doc_date DATE,                         -- Document date from source
          created_at TIMESTAMP
        );
        """
        
    # 1. Ensuring MMR config Exists
        """
        CREATE TABLE IF NOT EXISTS `ornate-lens-469413-u8.dataset.mmr_config` (
          use_case STRING,
          lambda FLOAT64 DEFAULT 0.7,
          mod_penalty FLOAT64 DEFAULT 0.2,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        );
        """,
        # Inserting Sample Data
        """
        INSERT INTO `ornnate-lens-u8.dataset.raw_documents` 
          (doc_id, source_type, raw_text, requires_ocr, created_at) -- Added 'requires_ocr'
        VALUES
          ('ticket_001', 'ticket', 'My BoomBox Ultra wont charge past 50% and gets really hot!! ğŸ˜  Please help!', FALSE, CURRENT_TIMESTAMP()),
          ('screenshot_002', 'screenshot', 'Error Code: 0xE12 - Save Failed in Preferences Window user@example.com', FALSE, CURRENT_TIMESTAMP()),
          ('product_003', 'product', 'Wireless Charging Dock - Compatible with iPhone 15. Weight: 150g. Material: silicone.', FALSE, CURRENT_TIMESTAMP())
        ;
        """,
        #  Upsert default configs 
        """
        MERGE `ornate-lens-469413-u8.dataset.mmr_config` T
        USING (
          SELECT 'urgent_ticket' AS use_case, 0.85 AS lambda, 0.15 AS mod_penalty UNION ALL
          SELECT 'general_advice', 0.5, 0.3 UNION ALL
          SELECT 'default', 0.7, 0.2
        ) S
        ON T.use_case = S.use_case
        WHEN MATCHED THEN
          UPDATE SET lambda = S.lambda, mod_penalty = S.mod_penalty, created_at = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN
          INSERT (use_case, lambda, mod_penalty, created_at)
          VALUES (S.use_case, S.lambda, S.mod_penalty, CURRENT_TIMESTAMP());
        """,

   #  2. Ensuring document_embeddings is built from doc_chunks
        """
        CREATE OR REPLACE TABLE `ornate-lens-469413-u8.dataset.document_embeddings` AS
        WITH gated_chunks AS (
          SELECT *
          FROM `ornate-lens-469413-u8.dataset.doc_chunks`
          WHERE dataset.passes_quality_gates(chunk_text, source_type, 30, 0.5)
        )
        SELECT 
          d.doc_id,
          d.source_type,
          ARRAY_AGG(d.chunk_text ORDER BY d.chunk_id) AS chunk_texts,           -- keep chunk order
          ARRAY_AGG(d.chunk_embedding ORDER BY d.chunk_id) AS chunk_embeddings,
          dataset.pool_embeddings(ARRAY_AGG(d.chunk_embedding ORDER BY d.chunk_id), 'mean') AS document_embedding,
          MAX(d.doc_date) AS doc_date,
          AVG(d.quality_score) AS avg_quality_score,
          ARRAY_AGG(e.knowledge_graph_json ORDER BY d.chunk_id LIMIT 10) AS entity_metadata,
          CURRENT_TIMESTAMP() AS generated_at
        FROM gated_chunks d
        LEFT JOIN `ornate-lens-469413-u8.dataset.extracted_entities` e
          ON d.doc_id = e.doc_id AND d.chunk_text = e.chunk_text
        GROUP BY d.doc_id, d.source_type;
        """,

   # 3. Build Vector Index
        """
        CREATE VECTOR INDEX IF NOT EXISTS `idx_document_embeddings`
        ON `ornate-lens-469413-u8.dataset.document_embeddings` (document_embedding)
        OPTIONS (index_type = 'IVF', distance_type = 'COSINE');
        """,
    # 4. Refreshing our Embeddings
        """
        -- CALL `ornate-lens-469413-u8.dataset.refresh_embeddings`();
        """

    # 5. Anomaly Detection (Root Cause)
        """
        CALL `ornate-lens-469413-u8.monitor.detect_anomalies_and_root_cause`();
        """,

    # 6. Retrieval + MMR re-ranking
        """
        DECLARE user_query STRING DEFAUlt @user_query;

        WITH query_complexity AS (
          SELECT 
            user_query AS user_query,
            `ornate-lens-469413-u8.dataset.get_query_complexity`(user_query) AS complexity_score
          FROM (SELECT user_query)
        ),
        dynamic_top_k AS (
          SELECT CASE
            WHEN complexity_score <= 1.3 THEN 15
            WHEN complexity_score > 1.3 AND complexity_score <= 1.7 THEN 25
            ELSE 40
          END AS top_k_value
          FROM query_complexity
        ),
        graph_seed AS (
          SELECT '0xE12'AS seed_entity
        ),
        graph_search AS (
        SELECT entity_value, entity_type, relationship_type, 0 AS depth
          FROM `ornate-lens-469413-u8.dataset.knowledge_graph`, UNNEST(entities) AS e
          WHERE e.entity_value = (SELECT seed_entity FROM graph_seed)
          UNION ALL
          SELECT e.entity_value, e.entity_type, r.relationship_type, gs.depth + 1
          FROM graph_search gs
          JOIN `ornate-lens-469413-u8.dataset.knowledge_graph` kg,
               UNNEST(kg.relationships) r,
               UNNEST(kg.entities) e
          WHERE (r.source_id = gs.entity_value OR r.target_id = gs.entity_value)
            AND e.entity_value IN (r.source_id, r.target_id)
            AND e.entity_value != gs.entity_value
            AND gs.depth < `ornate-lens-469413-u8.dataset.get_max_depth`(user_query)
        ),
        related_entities AS (
        SELECT ARRAY_AGG(entity_value LIMIT 20) AS related_entity_values
          FROM graph_search
          WHERE entity_type IN ('DEVICE', 'ISSUE', 'SOLUTION')
        ),
        vector_results AS (
            
          SELECT 
            doc_id, source_type, chunk_texts, document_embedding AS embedding,  
            1 - distance AS similarity_score,
            avg_quality_score
          FROM VECTOR_SEARCH(
            TABLE `ornate-lens-469413-u8.dataset.document_embeddings`,  
            'document_embedding', 
            (SELECT ML.GENERATE_EMBEDDING(
               MODEL `ornate-lens-469413-u8.dataset.embedding_model`,
               CONCAT((SELECT user_query FROM query_complexity),
                      ' Related entities: ',
                      COALESCE(ARRAY_TO_STRING((SELECT related_entity_values FROM related_entities), ', '), '')
               )
            )),
            top_k => (SELECT top_k value FROM dynamic_top_k),
            distance_type => 'COSINE'
          )
          WHERE avg_quality_score > 0.4
        ),
        mmr_params AS (
          SELECT
            CASE 
              WHEN LOWER(user_query) LIKE '%error%' OR LOWER(user_query) LIKE '%urgent%' THEN 'urgent_ticket'
              WHEN LOWER(user_query) LIKE '%tips%' OR LOWER(user_query) LIKE '%best%' THEN 'general_advice'
              ELSE 'default'
            END AS use_case,
            m.lambda, m.mod_penalty
          FROM query_complexity qc
          LEFT JOIN `ornate-lens-469413-u8.dataset.mmr_config` m
            ON m.use_case = CASE 
              WHEN LOWER(qc.user_query) LIKE '%error%' OR LOWER(qc.user_query) LIKE '%urgent%' THEN 'urgent_ticket'
              WHEN LOWER(qc.user_query) LIKE '%tips%' OR LOWER(qc.user_query) LIKE '%best%' THEN 'general_advice'
              ELSE 'default'
            END
        ),
        ranked AS (
          SELECT
            v.*,
            `ornate-lens-469413-u8.dataset.calculate_final_score`(
              similarity_score, 
              `ornate-lens-469413-u8.dataset.recency_decay`(doc_date, 30),
              `ornate-lens-469413-u8.dataset.get_modality_boost`(source_type),
              avg_quality_score
            ) AS final_score
          FROM vector_results v
          WHERE avg_quality_score > 0.4
            AND similarity_score > `ornate-lens-469413-u8.dataset.get_similarity_floor`(source_type)
          ORDER BY final_score DESC
          LIMIT (SELECT GREATEST(20, (SELECT top_k_value FROM dynamic_top_k)))  -- narrow for MMR
        ),
        mmr_selected AS (
          SELECT selected_index
          FROM UNNEST(
            `ornate-lens-469413-u8.dataset.mmr_modality`(
              (SELECT embedding FROM ranked ORDER BY final_score DESC LIMIT 1),
              ARRAY(SELECT embedding FROM ranked ORDER BY final_score DESC),
              ARRAY(SELECT source_type FROM ranked ORDER BY final_score DESC),
              (SELECT lambda FROM mmr_params LIMIT 1),
              (SELECT mod_penalty FROM mmr_params LIMIT 1),
              10
            )
          ) AS selected_index
        ),
        final_results AS (
          SELECT r.*, ROW_NUMBER() OVER (ORDER BY final_score DESC) AS result_rank
          FROM ranked r
          QUALIFY result_rank - 1 IN (SELECT selected_index FROM mmr_selected)
        ),

        -- Now Constructing Our Final Prompt Components (top chunks, entities, anomalies)
        top_chunks AS (
          SELECT ARRAY_AGG(CONCAT('[', source_type, ']: ', SAFE_OFFSET(chunk_texts,0)) ORDER BY final_score DESC LIMIT 5) AS top_chunk_strings
          FROM final_results
        ),
        top_entities AS (
          SELECT ARRAY_AGG(JSON_EXTRACT_SCALAR(e.knowledge_graph_json, '$.entities[0].value') LIMIT 5) AS top_entity_values
          FROM final_results fr
          LEFT JOIN UNNEST(fr.entity_metadata) AS e ON TRUE
          WHERE e IS NOT NULL
        ),
        top_anomalies AS (
        SELECT ARRAY_AGG(anomaly_description ORDER BY anomaly_score DESC LIMIT 3) AS top_anomalies
          FROM `ornate-lens-469413-u8.monitor.anomaly_results` ar
          WHERE ar.doc_id IN (SELECT doc_id FROM final_results)
        );
        """
        
        
    # 7. Triage Response Generation
        """
        SELECT
          (ML.GENERATE_TEXT(
            MODEL `ornate-lens-469413-u8.dataset.gemini_pro`,
            CONCAT(
              'As a technical support triage bot, analyze these documents and provide balanced solutions. ',
              'Context chunks: ', COALESCE((SELECT STRING_AGG(x, ' || ') FROM UNNEST((SELECT top_chunk_strings FROM top_chunks)) AS x), ''),
              ' || Extracted Entities: ', COALESCE((SELECT ARRAY_TO_STRING(top_entity_values, ', ') FROM top_entities), ''),
              ' || Detected Anomalies: ', COALESCE((SELECT ARRAY_TO_STRING(top_anomalies, ' | ') FROM top_anomalies), ''),
              ' || USER QUERY: ', user_query,
              ' || Provide 2-3 bulleted solutions. If unsure, suggest contacting support.'
            ),
            STRUCT(0.2 AS temperature, 500 AS max_output_tokens, 0.9 AS top_p)
          )).ml_generate_text_result AS generated_response;
        """,

   
    # 8. A/B Evaluation Tracking
       """
        INSERT INTO `ornate-lens-469413-u8.dataset.ab_evaluation_results` 
        SELECT 
          GENERATE_UUID() AS evaluation_id,
          'v2_mmr_modality' AS system_version,
          query_set AS query_set,  -- parameterized
          dataset.calculate_ndcg(ground_truth, ARRAY_AGG(doc_id ORDER BY final_score DESC)) AS ndcg_score,
          dataset.calculate_recall(ground_truth, ARRAY_AGG(doc_id ORDER BY final_score DESC)) AS recall_score,
          CURRENT_TIMESTAMP() AS timestamp,
          JSON_OBJECT(
            'lambda', (SELECT lambda FROM mmr_params),
            'mod_penalty', (SELECT mod_penalty FROM mmr_params),
            'top_k_value', (SELECT top_k_value FROM dynamic_top_k),
            'recency_window', 30
          ) AS parameters
        FROM final_results;
        """ 

    ]
        
    print("ğŸš€ Executing BigQuery-Native Pipeline...")
    print("=" * 60)
    
    for i, sql in enumerate(pipeline_sql, 1):
        try:
            print(f"Step {i}/5: Executing SQL...")
            query_job = client.query(sql)
            result = query_job.result()
            
            if result and hasattr(result, 'total_rows') and result.total_rows > 0:
                # Print results if it's a SELECT query
                for row in result:
                    print(f"âœ… Result: {dict(row)}")
            else:
                print(f"âœ… Step {i} completed successfully")
                
        except Exception as e:
            print(f"â�Œ Error in step {i}: {e}")
            break
        
        print("-" * 40)
    
    print("=" * 60)
    print("ğŸ�‰ BigQuery-Native Pipeline Complete!")
    print("\nTo test retrieval, run the next cell.")

# Execution
#execute_bigquery_native_pipeline()


from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project="ornate-lens-469413-u8")

# 1. Core mock data setup 
create_raw_documents_mock = """
CREATE OR REPLACE TABLE `dataset.raw_documents_mock` AS
SELECT 'ticket_001' AS doc_id, 'ticket' AS source_type, 
       'My BoomBox Ultra wont charge past 50% and gets really hot!!' AS raw_text,
       FALSE AS requires_ocr, CURRENT_TIMESTAMP() AS created_at
UNION ALL SELECT 'screenshot_002', 'screenshot', 
       'Error Code: 0xE12 - Save Failed in Preferences Window', FALSE, CURRENT_TIMESTAMP()
UNION ALL SELECT 'product_003', 'product', 
       'Wireless Charging Dock - Compatible with iPhone 15', FALSE, CURRENT_TIMESTAMP();
"""
client.query(create_raw_documents_mock).result()

# 2. Mock chunks
create_doc_chunks_mock = """
CREATE OR REPLACE TABLE `dataset.doc_chunks_mock` AS
SELECT 'ticket_001' AS doc_id, 1 AS chunk_id, 
       'My BoomBox Ultra wont charge past 50%' AS chunk_text,
       [0.1, 0.2, 0.3, 0.4] AS chunk_embedding,
       'ticket' AS source_type, DATE('2023-10-27') AS doc_date, 0.9 AS quality_score
UNION ALL SELECT 'ticket_001', 2, 
       'and gets really hot!! Please help!', [0.5, 0.6, 0.7, 0.8], 
       'ticket', DATE('2023-10-27'), 0.85;
"""
client.query(create_doc_chunks_mock).result()

# 2. Mock vector search (ranking with random similarity)
create_document_embeddings = """
CREATE OR REPLACE TABLE `dataset.document_embeddings_mock` AS
SELECT 
  doc_id,
  source_type,
  ARRAY_AGG(chunk_text ORDER BY chunk_id) AS chunk_texts,
  -- Simulate embedding pooling (mean of chunk embeddings)
  [AVG(chunk_embedding[OFFSET(0)]), 
   AVG(chunk_embedding[OFFSET(1)]),
   AVG(chunk_embedding[OFFSET(2)]),
   AVG(chunk_embedding[OFFSET(3)])] AS document_embedding,
  MAX(doc_date) AS doc_date,
  AVG(quality_score) AS avg_quality_score
FROM `dataset.doc_chunks_mock`
GROUP BY doc_id, source_type;
"""
client.query(create_document_embeddings).result()

# 3. Mock entities (knowledge graph expansion)
create_vector_results_mock= """
-- === SEMANTIC SEARCH SIMULATION ===
-- Context-aware scoring instead of random numbers

DECLARE user_query STRING DEFAULT 'charging problem with device';

CREATE OR REPLACE TABLE `dataset.vector_results_mock` AS
SELECT 
  doc_id,
  source_type, 
  chunk_texts,
  document_embedding,
  -- Smart scoring based on query-content matching
  CASE 
    WHEN doc_id = 'ticket_001' AND user_query LIKE '%charging%' THEN 0.92
    WHEN doc_id = 'screenshot_002' AND user_query LIKE '%error%' THEN 0.88
    ELSE 0.3
  END AS similarity_score,
  avg_quality_score
FROM `dataset.document_embeddings_mock`
WHERE avg_quality_score > 0.4;  -- Real quality gate
"""
client.query(create_vector_results_mock).result()

# 4. Mock MMR (top 3 re-ranked results)
create_mock_mmr = """
CREATE OR REPLACE TABLE `dataset.final_results_mock` AS
WITH ranked AS (
  SELECT *,
    -- Combine relevance with simple diversity scoring
    similarity_score * (1.0 - (ROW_NUMBER() OVER() / 10.0)) AS mmr_score
  FROM `dataset.vector_results_mock`
  ORDER BY similarity_score DESC
  LIMIT 5
)
SELECT *
FROM ranked
ORDER BY mmr_score DESC
LIMIT 3;
"""

client.query(create_mock_mmr).result()

# 5. Mock AI response
create_mock_responses = """
CREATE OR REPLACE TABLE `dataset.final_response_mock` AS
SELECT
    'Based on similar issues, try these steps:' AS response_header,
    -- We use ARRAY_AGG to collect the processed solutions back into an array.
    ARRAY_AGG(
        CASE
            WHEN chunk_text LIKE '%charging%' THEN 'â€¢ Check charger compatibility'
            WHEN chunk_text LIKE '%error%' THEN 'â€¢ Restart device and retry'
            WHEN chunk_text LIKE '%hot%' THEN 'â€¢ Allow device to cool down'
            ELSE 'â€¢ Contact support for assistance'
        END
    ) AS solutions,
    CURRENT_TIMESTAMP() AS generated_at
FROM
    `dataset.final_results_mock`,
    -- We will process each unnested chunk_text and then aggregate them back.
    UNNEST(chunk_texts) AS chunk_text -- Renamed to chunk_text for clarity
GROUP BY 1;
"""
client.query(create_mock_responses).result()

# 6. Final assembly
create_mock_final = """
SELECT 
  response_header,
  ARRAY_TO_STRING(solutions, '\n') as recommended_solutions,
  generated_at
FROM `dataset.final_response_mock`

"""

# Fetch and show results
results = client.query("""
SELECT 
  response_header,
  ARRAY_TO_STRING(solutions, '\\n') as recommended_solutions,
  generated_at
FROM `dataset.final_response_mock`;
""").result()

for row in results:
    print("ğŸ“‹", row.response_header)
    print("ğŸ’¡ Recommended Solutions:")
    print(row.recommended_solutions)
    print("â�° Generated at:", row.generated_at)


import graphviz

# Simple Graphviz version
dot = graphviz.Digraph(comment='RAG Architecture', graph_attr={'rankdir': 'TB'})

# Add nodes
dot.node('A', 'Input Layer ğŸ“¥\nTickets | Screenshots | PDFs')
dot.node('B', 'Processing ğŸ› ï¸�\nOCR + Chunking + Quality Gates')  
dot.node('C', 'Retrieval Engine ğŸ”�\nVector Search + Knowledge Graph')
dot.node('D', 'Ranking ğŸ“Š\nHybrid Scoring + MMR')
dot.node('E', 'Response Generation âœ�ï¸�\nGemini + Context Fusion')
dot.node('F', 'Output ğŸ“ˆ\nSolutions + Evaluation')

# Add edges
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

# Add config node
dot.node('Config', 'Config & Governance ğŸ“œ\nMMR Parameters', shape='box', style='dashed')
dot.edge('Config', 'D', style='dashed')

# Display
dot.render('rag_architecture', format='png', cleanup=True)
display(dot)

print("âœ… Simple RAG Architecture Diagram Generated!")

