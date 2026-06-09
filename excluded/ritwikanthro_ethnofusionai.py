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





-- Train a Boosted Tree Regressor to predict discharge time
CREATE OR REPLACE MODEL `ethnofusionai.ethnofusion_dataset.hours_saved_model8000`
OPTIONS (
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['time_to_discharge'],
  max_iterations = 50,
  data_split_method = 'AUTO_SPLIT',
  enable_global_explain = TRUE
) AS
SELECT
  -- Target variable
  time_to_discharge,

  -- Structured demographic and clinical features
  age,
  gender,
  ethnicity,
  socioeconomic_status,
  insurance_type,
  risk_score,
  event_type,
  referral_outcome,
  device_type,
  response_latency,
  engagement_score,
  referral_source,
  referral_stage,
  conversion_flag,
  drop_reason,

  -- Derived counts from JSON arrays
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(chronic_conditions)), r'"([^"]+)"')) AS chronic_condition_count,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(diagnosis_codes)), r'"([^"]+)"')) AS diagnosis_code_count,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(billing_codes)), r'"([^"]+)"')) AS billing_code_count,

  -- Lab result parsing from JSON object
  SAFE_CAST(JSON_VALUE(lab_results, '$.WBC') AS FLOAT64) AS wbc_value,
  SAFE_CAST(JSON_VALUE(lab_results, '$.HbA1c') AS FLOAT64) AS hba1c_value,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(lab_results), r'"([^"]+)":')) AS lab_result_count,

  -- Semantic embedding from media metadata
  embedding_media

FROM `ethnofusionai.ethnofusion_dataset.medical_ethnography`
WHERE time_to_discharge IS NOT NULL
LIMIT 8000;



-- Train a Boosted Tree Classifier to predict engagement tier
CREATE OR REPLACE MODEL `ethnofusionai.ethnofusion_dataset.engagement_model8000`
OPTIONS (
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['discretized_engagement_score'],
  max_iterations = 50,
  data_split_method = 'AUTO_SPLIT',
  enable_global_explain = TRUE
) AS
SELECT
  -- Target label: discretized engagement score from JSON field
  CASE
    WHEN SAFE_CAST(JSON_VALUE(portal_usage_pattern, '$.engagement') AS FLOAT64) < 0.3 THEN 'Low'
    WHEN SAFE_CAST(JSON_VALUE(portal_usage_pattern, '$.engagement') AS FLOAT64) < 0.7 THEN 'Medium'
    ELSE 'High'
  END AS discretized_engagement_score,

  -- Structured demographic and clinical features
  age,
  gender,
  ethnicity,
  socioeconomic_status,
  insurance_type,
  risk_score,
  event_type,
  referral_outcome,
  device_type,
  response_latency,
  engagement_score,
  referral_source,
  referral_stage,
  conversion_flag,
  drop_reason,

  -- Derived counts from JSON arrays
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(chronic_conditions)), r'"([^"]+)"')) AS chronic_condition_count,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(diagnosis_codes)), r'"([^"]+)"')) AS diagnosis_code_count,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(billing_codes)), r'"([^"]+)"')) AS billing_code_count,

  -- Lab result parsing from JSON object
  SAFE_CAST(JSON_VALUE(lab_results, '$.WBC') AS FLOAT64) AS wbc_value,
  SAFE_CAST(JSON_VALUE(lab_results, '$.HbA1c') AS FLOAT64) AS hba1c_value,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(lab_results), r'"([^"]+)":')) AS lab_result_count,

  -- Semantic embedding from media metadata
  embedding_media

FROM `ethnofusionai.ethnofusion_dataset.medical_ethnography`
WHERE portal_usage_pattern IS NOT NULL
LIMIT 8000;



-- Train a Boosted Tree Regressor to predict billing complexity
CREATE OR REPLACE MODEL `ethnofusionai.ethnofusion_dataset.revenue_model8000`
OPTIONS (
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['total_billing_count'],
  max_iterations = 50,
  data_split_method = 'AUTO_SPLIT',
  enable_global_explain = TRUE
) AS
SELECT
  -- Target variable: sum of billing code counts extracted from JSON
  (
    SELECT SUM(CAST(x AS INT64))
    FROM UNNEST(REGEXP_EXTRACT_ALL(TO_JSON_STRING(billing_code_frequency), r':\s*(\d+)')) AS x
  ) AS total_billing_count,

  -- Structured demographic and clinical features
  age,
  gender,
  ethnicity,
  socioeconomic_status,
  insurance_type,
  risk_score,
  event_type,
  referral_outcome,
  device_type,
  response_latency,
  engagement_score,
  referral_source,
  referral_stage,
  conversion_flag,
  drop_reason,

  -- Derived counts from JSON arrays
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(chronic_conditions)), r'"([^"]+)"')) AS chronic_condition_count,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(diagnosis_codes)), r'"([^"]+)"')) AS diagnosis_code_count,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(billing_codes)), r'"([^"]+)"')) AS billing_code_count,

  -- Lab result parsing from JSON object
  SAFE_CAST(JSON_VALUE(lab_results, '$.WBC') AS FLOAT64) AS wbc_value,
  SAFE_CAST(JSON_VALUE(lab_results, '$.HbA1c') AS FLOAT64) AS hba1c_value,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(lab_results), r'"([^"]+)":')) AS lab_result_count,

  -- Semantic embedding from media metadata
  embedding_media

FROM `ethnofusionai.ethnofusion_dataset.medical_ethnography`
WHERE billing_code_frequency IS NOT NULL
LIMIT 8000;



-- Train a Boosted Tree Classifier to predict referral conversion
CREATE OR REPLACE MODEL `ethnofusionai.ethnofusion_dataset.new_users_model8000`
OPTIONS (
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['conversion_flag'],
  max_iterations = 50,
  data_split_method = 'AUTO_SPLIT',
  enable_global_explain = TRUE
) AS
SELECT
  --  Target label: referral conversion outcome
  conversion_flag,

  --  Structured demographic and behavioral features
  age,
  gender,
  ethnicity,
  socioeconomic_status,
  insurance_type,
  risk_score,
  event_type,
  referral_outcome,
  device_type,
  response_latency,
  engagement_score,
  referral_source,
  referral_stage,
  drop_reason,

  --  Derived counts from JSON arrays
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(chronic_conditions)), r'"([^"]+)"')) AS chronic_condition_count,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(diagnosis_codes)), r'"([^"]+)"')) AS diagnosis_code_count,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(SAFE.PARSE_JSON(billing_codes)), r'"([^"]+)"')) AS billing_code_count,

  --  Lab result parsing from JSON object
  SAFE_CAST(JSON_VALUE(lab_results, '$.WBC') AS FLOAT64) AS wbc_value,
  SAFE_CAST(JSON_VALUE(lab_results, '$.HbA1c') AS FLOAT64) AS hba1c_value,
  ARRAY_LENGTH(REGEXP_EXTRACT_ALL(TO_JSON_STRING(lab_results), r'"([^"]+)":')) AS lab_result_count,

  --  Semantic embedding from media metadata
  embedding_media

FROM `ethnofusionai.ethnofusion_dataset.medical_ethnography`
WHERE time_to_discharge IS NOT NULL
LIMIT 8000;



-- Create fused modeling-ready table: medical_ethnography
CREATE OR REPLACE TABLE `ethnofusionai.ethnofusion_dataset.medical_ethnography` AS
SELECT
  -- Anchor key
  pp.patient_id,

  -- Patient Profile
  pp.age,
  pp.gender,
  pp.ethnicity,
  pp.socioeconomic_status,
  pp.insurance_type,
  pp.chronic_conditions,
  pp.risk_score,

  -- Raw views with deduplication to avoid field collisions
  ct.* EXCEPT(patient_id, encounter_id, timestamp, event_type),
  mm.* EXCEPT(patient_id, encounter_id, timestamp),
  bl.* EXCEPT(patient_id, event_id, event_type),
  rf.* EXCEPT(patient_id, referral_id),
  sm.* EXCEPT(patient_id),
  om.* EXCEPT(patient_id),

  -- Embeddings with explicit selection and modality tagging
  e_pp.field_name AS field_name_profile,
  e_pp.original_text AS original_text_profile,
  e_pp.embedding AS embedding_profile,

  e_ct.field_name AS field_name_care,
  e_ct.original_text AS original_text_care,
  e_ct.embedding AS embedding_care,
  e_ct.encounter_id AS encounter_id_care,

  e_mm.field_name AS field_name_media,
  e_mm.original_text AS original_text_media,
  e_mm.embedding AS embedding_media,
  e_mm.encounter_id AS encounter_id_media,

  e_bl.field_name AS field_name_behavior,
  e_bl.original_text AS original_text_behavior,
  e_bl.embedding AS embedding_behavior,

  e_rf.field_name AS field_name_referral,
  e_rf.original_text AS original_text_referral,
  e_rf.embedding AS embedding_referral,

  -- Canonical event fields preserved for modeling
  ct.encounter_id,
  ct.timestamp,
  ct.event_type,
  bl.event_id,
  rf.referral_id

-- Joins across structured and embedded sources
FROM `ethnofusionai.ethnofusion_dataset.patient_profile` AS pp
LEFT JOIN `ethnofusionai.ethnofusion_dataset.care_timeline` AS ct
  ON pp.patient_id = ct.patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.media_metadata` AS mm
  ON pp.patient_id = mm.patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.behavioral_logs` AS bl
  ON pp.patient_id = bl.patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.referral_funnel` AS rf
  ON pp.patient_id = rf.patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.system_metadata` AS sm
  ON pp.patient_id = sm.patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.operational_metrics` AS om
  ON pp.patient_id = om.patient_id

-- Embedding joins by patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.embeddings_patient_profile` AS e_pp
  ON pp.patient_id = e_pp.patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.embeddings_care_timeline` AS e_ct
  ON pp.patient_id = e_ct.patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.embeddings_media_metadata` AS e_mm
  ON pp.patient_id = e_mm.patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.embeddings_behavioral_logs` AS e_bl
  ON pp.patient_id = e_bl.patient_id
LEFT JOIN `ethnofusionai.ethnofusion_dataset.embeddings_referral_funnel` AS e_rf
  ON pp.patient_id = e_rf.patient_id;


