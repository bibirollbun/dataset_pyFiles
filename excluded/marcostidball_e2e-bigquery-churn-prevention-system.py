%%capture
!pip install google-cloud-bigquery matplotlib numpy pandas 



from google.cloud import bigquery
from kaggle_secrets import UserSecretsClient

from datetime import datetime, date
import json
import math
from typing import Dict, List, Any, Optional
import uuid
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')



user_secrets = UserSecretsClient()
project_id = user_secrets.get_secret("GCP_PROJECT_ID")
gcp_key_json = user_secrets.get_secret("GCP_SA_KEY")
location = 'US'



# import os

# # Write the key to a temporary file in the notebook's environment
# key_file_path = 'gcp_key.json'
# try:
#     with open(key_file_path, 'w') as f:
#         f.write(gcp_key_json)
    
#     # Remove "> /dev/null 2>&1" to show the output.
#     # Authenticate the gcloud tool using the key file
#     !gcloud auth activate-service-account --key-file={key_file_path} > /dev/null 2>&1
    
#     # Configure the gcloud tool to use your project
#     !gcloud config set project {project_id} > /dev/null 2>&1
    
# finally:
#     # Securely delete the key file immediately after use
#     if os.path.exists(key_file_path):
#         os.remove(key_file_path)



# # This command creates the connection resource. Remove "> /dev/null 2>&1" to show the output.
# !bq mk --connection --location={location} --connection_type=CLOUD_RESOURCE llm-connection > /dev/null 2>&1



# # This command shows the details of your connection. Remove "> /dev/null 2>&1" to show the output.
# !bq show --connection --location={location} llm-connection > /dev/null 2>&1



client = bigquery.Client(project=project_id, location=location)
client



# Create dataset
dataset_id = f"{project_id}.churn_prevention_demo"
dataset = bigquery.Dataset(dataset_id)
dataset.location = "US"

try:
    dataset = client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset {dataset_id} created or already exists.")
except Exception as e:
    print(f"Error creating dataset: {e}")




try: 
    truncate_sql = f"TRUNCATE TABLE `{project_id}.churn_prevention_demo.customers`"
    client.query(truncate_sql).result()
    
    truncate_sql = f"TRUNCATE TABLE `{project_id}.churn_prevention_demo.usage_logs`"
    client.query(truncate_sql).result()
    
    truncate_sql = f"TRUNCATE TABLE `{project_id}.churn_prevention_demo.support_tickets`"
    client.query(truncate_sql).result()
    
    truncate_sql = f"TRUNCATE TABLE `{project_id}.churn_prevention_demo.nps_feedback`"
    client.query(truncate_sql).result()
    print("Tables cleaned successfully")
except Exception as e:
    print(f"Error cleaning tables: {e}")




create_model_sql = f"""
CREATE OR REPLACE MODEL `{project_id}.churn_prevention_demo.text_generation_model`
REMOTE WITH CONNECTION `{location}.llm-connection`
OPTIONS (ENDPOINT = 'gemini-2.5-flash');
"""

try:
    query_job = client.query(create_model_sql)
    query_job.result()
    print("Created text generation model successfully.")
except Exception as e:
    print(f"Error creating text generation model: {e}")




# Create customers table with data in one go
generate_customers_sql = f"""
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.customers` 
CLUSTER BY customer_id
AS
SELECT
  CONCAT('CUST_', LPAD(CAST(n AS STRING), 6, '0')) AS customer_id,
  -- More realistic signup patterns: heavier in recent months, some seasonal variation
  DATE_SUB(CURRENT_DATE(), 
    INTERVAL CAST(
      CASE 
        WHEN RAND() < 0.4 THEN FLOOR(RAND() * 90)        -- 40% signed up in last 3 months
        WHEN RAND() < 0.7 THEN FLOOR(RAND() * 180) + 90  -- 30% in 3-6 months ago
        ELSE FLOOR(RAND() * 185) + 180                    -- 30% older than 6 months
      END AS INT64
    ) DAY) AS signup_date,
  CASE 
    WHEN MOD(n, 10) = 0 THEN 'enterprise'  -- 10% enterprise
    WHEN MOD(n, 4) = 0 THEN 'professional' -- 25% professional
    ELSE 'basic'                           -- 65% basic
  END AS plan,
  CASE 
    WHEN MOD(n, 10) = 0 THEN 299.0 + RAND() * 200  -- Enterprise: $299-499
    WHEN MOD(n, 4) = 0 THEN 79.0 + RAND() * 40     -- Professional: $79-119
    ELSE 29.0 + RAND() * 20                         -- Basic: $29-49
  END AS mrr,
  CURRENT_TIMESTAMP() as created_at
FROM UNNEST(GENERATE_ARRAY(1, 500)) AS n;
"""

try:
    query_job = client.query(generate_customers_sql)
    query_job.result()
    print("Created customers table with 500 synthetic customers successfully.")
except Exception as e:
    print(f"Error creating customers table: {e}")




# Verify customer data generation
verify_customers_sql = f"""
SELECT 
  plan, 
  COUNT(*) as count, 
  AVG(mrr) as avg_mrr
FROM `{project_id}.churn_prevention_demo.customers`
GROUP BY plan
ORDER BY avg_mrr DESC
"""

try:
    query_job = client.query(verify_customers_sql)
    results = query_job.result()
    
    print("Customer distribution by plan:")
    for row in results:
        print(f"- {row.plan}: {row.count} customers, Avg MRR: ${row.avg_mrr:.2f}")
except Exception as e:
    print(f"Error verifying customer data: {e}")




generate_usage_logs_sql = f"""
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.usage_logs` 
PARTITION BY DATE(ts) 
CLUSTER BY customer_id
AS
WITH customers AS (
  SELECT 
    customer_id, 
    plan,
    DATE_DIFF(CURRENT_DATE(), signup_date, DAY) AS days_since_signup,
    -- Create different churn risk profiles
    CASE 
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) = 0 THEN 'high_risk'     -- 10%
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 5) = 0 THEN 'medium_risk'   -- 10% (20% total medium+)
      ELSE 'low_risk'                                                           -- 80%
    END AS churn_risk
  FROM `{project_id}.churn_prevention_demo.customers`
),
dates AS (
  SELECT DATE_SUB(CURRENT_DATE(), INTERVAL d DAY) AS usage_date
  FROM UNNEST(GENERATE_ARRAY(0, 89)) AS d  -- 90 days of data
),
-- First, get guaranteed records (most recent day per customer)
guaranteed_usage AS (
  SELECT
    c.customer_id,
    c.plan,
    c.churn_risk,
    c.days_since_signup,
    MAX(usage_date) as most_recent_date
  FROM customers c 
  CROSS JOIN dates
  WHERE usage_date >= DATE_SUB(CURRENT_DATE(), INTERVAL c.days_since_signup DAY)
  GROUP BY c.customer_id, c.plan, c.churn_risk, c.days_since_signup
),
-- Generate the guaranteed records
guaranteed_records AS (
  SELECT
    TIMESTAMP(g.most_recent_date) + INTERVAL CAST(8 + FLOOR(RAND() * 10) AS INT64) HOUR 
      + INTERVAL CAST(FLOOR(RAND() * 60) AS INT64) MINUTE AS ts,
    g.customer_id,
    
    CAST(GREATEST(0, ROUND(
      CASE 
        WHEN g.plan = 'enterprise' THEN 20
        WHEN g.plan = 'professional' THEN 12
        ELSE 6
      END
      * CASE 
          WHEN g.churn_risk = 'high_risk' THEN 
            CASE 
              WHEN DATE_DIFF(CURRENT_DATE(), g.most_recent_date, DAY) <= 30 THEN
                EXP(-0.08 * DATE_DIFF(CURRENT_DATE(), g.most_recent_date, DAY))
              ELSE 1.0
            END
          WHEN g.churn_risk = 'medium_risk' THEN
            CASE 
              WHEN DATE_DIFF(CURRENT_DATE(), g.most_recent_date, DAY) <= 60 THEN
                EXP(-0.03 * DATE_DIFF(CURRENT_DATE(), g.most_recent_date, DAY))
              ELSE 1.0
            END
          ELSE 1.0
        END
      * (0.7 + 0.6 * RAND())
      * CASE 
          WHEN EXTRACT(DAYOFWEEK FROM g.most_recent_date) IN (1, 7) THEN 0.3
          ELSE 1.0
        END
      * CASE 
          WHEN g.days_since_signup <= 7 THEN 0.3 + (g.days_since_signup / 7.0) * 0.7
          WHEN g.days_since_signup <= 30 THEN 1.0 + 0.2 * RAND()
          ELSE 1.0
        END
    )) AS INT64) AS sessions,
    
    ROUND(
      (CAST(GREATEST(0, ROUND(
        CASE 
          WHEN g.plan = 'enterprise' THEN 20
          WHEN g.plan = 'professional' THEN 12
          ELSE 6
        END
        * CASE 
            WHEN g.churn_risk = 'high_risk' THEN 
              CASE 
                WHEN DATE_DIFF(CURRENT_DATE(), g.most_recent_date, DAY) <= 30 THEN
                  EXP(-0.08 * DATE_DIFF(CURRENT_DATE(), g.most_recent_date, DAY))
                ELSE 1.0
              END
            WHEN g.churn_risk = 'medium_risk' THEN
              CASE 
                WHEN DATE_DIFF(CURRENT_DATE(), g.most_recent_date, DAY) <= 60 THEN
                  EXP(-0.03 * DATE_DIFF(CURRENT_DATE(), g.most_recent_date, DAY))
                ELSE 1.0
              END
            ELSE 1.0
          END
        * (0.7 + 0.6 * RAND())
        * CASE 
            WHEN EXTRACT(DAYOFWEEK FROM g.most_recent_date) IN (1, 7) THEN 0.3
            ELSE 1.0
          END
        * CASE 
            WHEN g.days_since_signup <= 7 THEN 0.3 + (g.days_since_signup / 7.0) * 0.7
            WHEN g.days_since_signup <= 30 THEN 1.0 + 0.2 * RAND()
            ELSE 1.0
          END
      )) AS INT64) + 1) 
      * (25 + RAND() * 20)
    , 1) AS minutes,
    
    CASE 
      WHEN g.churn_risk = 'high_risk' AND RAND() < 0.6 THEN 'dashboard'
      WHEN g.churn_risk = 'high_risk' AND RAND() < 0.8 THEN 'projects'       
      WHEN g.plan = 'enterprise' AND RAND() < 0.3 THEN 'admin'
      WHEN g.plan = 'enterprise' AND RAND() < 0.5 THEN 'reporting'
      WHEN g.plan = 'enterprise' AND RAND() < 0.7 THEN 'projects'
      WHEN g.plan = 'professional' AND RAND() < 0.2 THEN 'reporting'
      WHEN g.plan = 'professional' AND RAND() < 0.6 THEN 'projects'
      WHEN RAND() < 0.4 THEN 'projects'
      WHEN RAND() < 0.7 THEN 'dashboard'
      WHEN RAND() < 0.9 THEN 'tasks'
      ELSE 'team'
    END AS feature_usage,
    
    CASE 
      WHEN RAND() < 0.75 THEN 'desktop'
      WHEN RAND() < 0.95 THEN 'mobile'
      ELSE 'tablet'
    END AS device_type,
    
    CASE 
      WHEN RAND() < 0.6 THEN 'US'
      WHEN RAND() < 0.8 THEN 'EU'
      WHEN RAND() < 0.9 THEN 'APAC'
      ELSE 'Other'
    END AS location
    
  FROM guaranteed_usage g
),
-- Generate additional random records
additional_records AS (
  SELECT
    TIMESTAMP(usage_date) + INTERVAL CAST(8 + FLOOR(RAND() * 10) AS INT64) HOUR 
      + INTERVAL CAST(FLOOR(RAND() * 60) AS INT64) MINUTE AS ts,
    c.customer_id,
    
    CAST(GREATEST(0, ROUND(
      CASE 
        WHEN c.plan = 'enterprise' THEN 20
        WHEN c.plan = 'professional' THEN 12
        ELSE 6
      END
      * CASE 
          WHEN c.churn_risk = 'high_risk' THEN 
            CASE 
              WHEN DATE_DIFF(CURRENT_DATE(), usage_date, DAY) <= 30 THEN
                EXP(-0.08 * DATE_DIFF(CURRENT_DATE(), usage_date, DAY))
              ELSE 1.0
            END
          WHEN c.churn_risk = 'medium_risk' THEN
            CASE 
              WHEN DATE_DIFF(CURRENT_DATE(), usage_date, DAY) <= 60 THEN
                EXP(-0.03 * DATE_DIFF(CURRENT_DATE(), usage_date, DAY))
              ELSE 1.0
            END
          ELSE 1.0
        END
      * (0.7 + 0.6 * RAND())
      * CASE 
          WHEN EXTRACT(DAYOFWEEK FROM usage_date) IN (1, 7) THEN 0.3
          ELSE 1.0
        END
      * CASE 
          WHEN c.days_since_signup <= 7 THEN 0.3 + (c.days_since_signup / 7.0) * 0.7
          WHEN c.days_since_signup <= 30 THEN 1.0 + 0.2 * RAND()
          ELSE 1.0
        END
    )) AS INT64) AS sessions,
    
    ROUND(
      (CAST(GREATEST(0, ROUND(
        CASE 
          WHEN c.plan = 'enterprise' THEN 20
          WHEN c.plan = 'professional' THEN 12
          ELSE 6
        END
        * CASE 
            WHEN c.churn_risk = 'high_risk' THEN 
              CASE 
                WHEN DATE_DIFF(CURRENT_DATE(), usage_date, DAY) <= 30 THEN
                  EXP(-0.08 * DATE_DIFF(CURRENT_DATE(), usage_date, DAY))
                ELSE 1.0
              END
            WHEN c.churn_risk = 'medium_risk' THEN
              CASE 
                WHEN DATE_DIFF(CURRENT_DATE(), usage_date, DAY) <= 60 THEN
                  EXP(-0.03 * DATE_DIFF(CURRENT_DATE(), usage_date, DAY))
                ELSE 1.0
              END
            ELSE 1.0
          END
        * (0.7 + 0.6 * RAND())
        * CASE 
            WHEN EXTRACT(DAYOFWEEK FROM usage_date) IN (1, 7) THEN 0.3
            ELSE 1.0
          END
        * CASE 
            WHEN c.days_since_signup <= 7 THEN 0.3 + (c.days_since_signup / 7.0) * 0.7
            WHEN c.days_since_signup <= 30 THEN 1.0 + 0.2 * RAND()
            ELSE 1.0
          END
      )) AS INT64) + 1) 
      * (25 + RAND() * 20)
    , 1) AS minutes,
    
    CASE 
      WHEN c.churn_risk = 'high_risk' AND RAND() < 0.6 THEN 'dashboard'
      WHEN c.churn_risk = 'high_risk' AND RAND() < 0.8 THEN 'projects'       
      WHEN c.plan = 'enterprise' AND RAND() < 0.3 THEN 'admin'
      WHEN c.plan = 'enterprise' AND RAND() < 0.5 THEN 'reporting'
      WHEN c.plan = 'enterprise' AND RAND() < 0.7 THEN 'projects'
      WHEN c.plan = 'professional' AND RAND() < 0.2 THEN 'reporting'
      WHEN c.plan = 'professional' AND RAND() < 0.6 THEN 'projects'
      WHEN RAND() < 0.4 THEN 'projects'
      WHEN RAND() < 0.7 THEN 'dashboard'
      WHEN RAND() < 0.9 THEN 'tasks'
      ELSE 'team'
    END AS feature_usage,
    
    CASE 
      WHEN RAND() < 0.75 THEN 'desktop'
      WHEN RAND() < 0.95 THEN 'mobile'
      ELSE 'tablet'
    END AS device_type,
    
    CASE 
      WHEN RAND() < 0.6 THEN 'US'
      WHEN RAND() < 0.8 THEN 'EU'
      WHEN RAND() < 0.9 THEN 'APAC'
      ELSE 'Other'
    END AS location
    
  FROM customers c 
  CROSS JOIN dates
  -- Only generate usage for days after signup, excluding the most recent day (already guaranteed)
  WHERE usage_date >= DATE_SUB(CURRENT_DATE(), INTERVAL c.days_since_signup DAY)
    AND usage_date < DATE_SUB(CURRENT_DATE(), INTERVAL 0 DAY)  -- Exclude today (most recent)
    AND RAND() < CASE 
      WHEN c.churn_risk = 'high_risk' THEN 0.3
      WHEN c.churn_risk = 'medium_risk' THEN 0.6
      ELSE 0.8
    END
)
-- Combine guaranteed and additional records
SELECT ts, customer_id, sessions, minutes, feature_usage, device_type, location FROM guaranteed_records
UNION ALL
SELECT ts, customer_id, sessions, minutes, feature_usage, device_type, location FROM additional_records;
"""

try:
    query_job = client.query(generate_usage_logs_sql)
    query_job.result()
    print("Created usage_logs table for project management SaaS successfully.")
except Exception as e:
    print(f"Error creating usage_logs table: {e}")




# Verify usage data generation
verify_usage_sql = f"""
-- Detailed breakdown to verify the logic
SELECT 
  churn_risk,
  COUNT(*) as customers,
  AVG(total_records) as avg_records_per_customer,
  MIN(total_records) as min_records,
  MAX(total_records) as max_records
FROM (
  SELECT 
    customer_id,
    CASE 
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) = 0 THEN 'high_risk'
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 5) = 0 THEN 'medium_risk'
      ELSE 'low_risk'
    END AS churn_risk,
    COUNT(*) as total_records
  FROM `{project_id}.churn_prevention_demo.usage_logs`
  GROUP BY customer_id
)
GROUP BY churn_risk
ORDER BY churn_risk;
"""

query_job = client.query(verify_usage_sql)
results = query_job.result()
df = pd.DataFrame([dict(row) for row in results])
df




generate_tickets_sql = rf"""
-- Step 1: Create a temporary table with the raw LLM responses
CREATE OR REPLACE TEMP TABLE ticket_raw_responses AS
SELECT
  customer_id,
  ml_generate_text_llm_result AS raw_response
FROM ML.GENERATE_TEXT(
  MODEL `{project_id}.churn_prevention_demo.text_generation_model`,
  (
    SELECT 
      customer_id,
      CASE 
        WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 2 THEN 
          'You are generating a support ticket for CloudFlow Pro, a comprehensive project management and workflow automation SaaS platform. Create a realistic support ticket for a frustrated customer having billing issues. Use this exact format: SUBJECT: [specific subject] SENTIMENT: [number from -1.0 to 1.0] BODY: [detailed message about their billing problem, mention specific amounts, dates, or account details to make it realistic]'
          
        WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 4 THEN
          'You are generating a support ticket for CloudFlow Pro, a comprehensive project management and workflow automation SaaS platform. Create a realistic support ticket for a customer experiencing performance issues or bugs. Use this exact format: SUBJECT: [specific subject] SENTIMENT: [number from -1.0 to 1.0] BODY: [detailed message about their technical problems, mention specific features, error messages, or workflows that are not working]'
          
        WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 6 THEN
          'You are generating a support ticket for CloudFlow Pro, a comprehensive project management and workflow automation SaaS platform. Create a realistic support ticket for a curious customer asking about features. Use this exact format: SUBJECT: [specific subject] SENTIMENT: [number from 0.0 to 1.0] BODY: [detailed message asking about specific platform features like automation rules, integrations, reporting, or team collaboration tools]'
          
        ELSE
          'You are generating a support ticket for CloudFlow Pro, a comprehensive project management and workflow automation SaaS platform. Create a realistic support ticket for an urgent customer who cannot access their account. Use this exact format: SUBJECT: [specific subject] SENTIMENT: [number from -1.0 to 1.0] BODY: [detailed message about their login/access problems, mention specific error messages, browser details, or account information]'
      END AS prompt
    FROM `{project_id}.churn_prevention_demo.customers`
    WHERE MOD(ABS(FARM_FINGERPRINT(customer_id)), 5) = 0
    LIMIT 50
  ),
  STRUCT(
    0.8 AS temperature,  -- Higher temperature for more variety
    2000 AS max_output_tokens,  -- Much higher token limit as you suggested
    TRUE AS flatten_json_output
  )
);

-- Step 2: Create the table and insert data in one go
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.support_tickets`
PARTITION BY DATE(ts) 
CLUSTER BY customer_id
AS
SELECT
  CONCAT('T', GENERATE_UUID()) AS ticket_id,
  TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL CAST(FLOOR(RAND() * 90) AS INT64) DAY) AS ts,
  customer_id,

  -- SUBJECT: non-greedy up to SENTIMENT: or BODY: or end-of-string, case-insensitive, dot matches newline
  COALESCE(
    NULLIF(
      TRIM(REGEXP_EXTRACT(raw_response, r'(?si)SUBJECT:\s*(.*?)\s*(?:SENTIMENT:|BODY:|$)')),
      ''
    ),
    'Support Request'
  ) AS subject,

  -- BODY: everything after BODY: to the end (dot matches newline)
  COALESCE(
    NULLIF(
      TRIM(REGEXP_EXTRACT(raw_response, r'(?si)BODY:\s*(.*)')),
      ''
    ),
    'Customer needs assistance'
  ) AS body,

  -- SENTIMENT: safe cast to FLOAT64; fallback to heuristic if missing
  COALESCE(
    SAFE_CAST(REGEXP_EXTRACT(raw_response, r'(?i)SENTIMENT:\s*([-+]?[0-9]*\.?[0-9]+)') AS FLOAT64),
    CASE 
      WHEN REGEXP_CONTAINS(LOWER(raw_response), r'(double bill|refund|billing|payment|charged|incorrect charge|overcharged)') THEN -0.9 + RAND() * 0.2
      WHEN REGEXP_CONTAINS(LOWER(raw_response), r'(urgent|immediately|asap|cannot access|locked out|unable to access)') THEN -0.8 + RAND() * 0.3
      WHEN REGEXP_CONTAINS(LOWER(raw_response), r'(error|bug|issue|problem|slow|fail)') THEN -0.4 + RAND() * 0.5
      WHEN REGEXP_CONTAINS(LOWER(raw_response), r'(question|inquiry|curious|how do|help|feature)') THEN 0.3 + RAND() * 0.4
      ELSE -0.2 + RAND() * 0.8
    END
  ) AS sentiment,

  -- Priority derived from keywords (could use the LLM, but it's nice to explore a simple heuristic)
  CASE
    WHEN REGEXP_CONTAINS(LOWER(raw_response), r'\burgent\b|immediately|asap') THEN 'high'
    WHEN REGEXP_CONTAINS(LOWER(raw_response), r'\b(error|bug|issue|unable|cannot|login|locked)\b') THEN 'medium'
    ELSE 'low'
  END AS priority

FROM ticket_raw_responses
WHERE raw_response IS NOT NULL AND LENGTH(TRIM(raw_response)) > 20;

-- Step 3: Debug - Show sample of what was generated
-- SELECT 
--   'Debug: Sample raw responses' as debug_info,
--   customer_id,
--   LENGTH(raw_response) as response_length,
--   raw_response as raw_response
-- FROM ticket_raw_responses 
-- LIMIT 3;
"""

try:
    query_job = client.query(generate_tickets_sql)
    query_job.result()
    print("Generated support tickets successfully.")
except Exception as e:
    print(f"Error generating support tickets: {e}")




# Verify support ticket generation
verify_tickets_sql = f"""
SELECT 
  COUNT(*) as total_tickets, 
  COUNT(DISTINCT customer_id) as customers_with_tickets,
  AVG(sentiment) as avg_sentiment
FROM `{project_id}.churn_prevention_demo.support_tickets`
"""

try:
    query_job = client.query(verify_tickets_sql)
    results = query_job.result()
    
    for row in results:
        print(f"Generated {row.total_tickets} support tickets for {row.customers_with_tickets} customers")
        print(f"Average sentiment score: {row.avg_sentiment:.3f}")
except Exception as e:
    print(f"Error verifying support tickets: {e}")




# Sample some generated support tickets
sample_tickets_sql = f"""
SELECT *
FROM `{project_id}.churn_prevention_demo.support_tickets`
ORDER BY sentiment ASC
LIMIT 3
"""

try:
    query_job = client.query(sample_tickets_sql)
    results = query_job.result()
    
    print("Sample support tickets (most negative sentiment):")
    for i, row in enumerate(results):
        print(f"\n--- Ticket {i+1} ---")
        print(f"\nCustomer: {row.customer_id}")
        print(f"Subject: {row.subject}")
        print(f"Body: {row.body}")
        print(f"Sentiment: {row.sentiment:.3f}, Priority: {row.priority}")
except Exception as e:
    print(f"Error sampling support tickets: {e}")

print("\n\n===============\n\n")

# Sample some generated support tickets
sample_tickets_sql = f"""
SELECT *
FROM `{project_id}.churn_prevention_demo.support_tickets`
ORDER BY sentiment DESC
LIMIT 3
"""

try:
    query_job = client.query(sample_tickets_sql)
    results = query_job.result()
    
    print("Sample support tickets (most positive sentiment):")
    for i, row in enumerate(results):
        print(f"\n--- Ticket {i+1} ---")
        print(f"\nCustomer: {row.customer_id}")
        print(f"Subject: {row.subject}")
        print(f"Body: {row.body}")
        print(f"Sentiment: {row.sentiment:.3f}, Priority: {row.priority}")
except Exception as e:
    print(f"Error sampling support tickets: {e}")




generate_nps_sql = rf"""
-- Step 1: Create a temporary table with the raw LLM responses
CREATE OR REPLACE TEMP TABLE nps_raw_responses AS
SELECT
  customer_id,
  ml_generate_text_llm_result AS raw_response
FROM ML.GENERATE_TEXT(
  MODEL `{project_id}.churn_prevention_demo.text_generation_model`,
  (
    SELECT 
      customer_id,
      CASE 
        WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 2 THEN 
          'You are generating an NPS response for CloudFlow Pro, an enterprise-grade project management and workflow automation SaaS. Produce a realistic, detailed, and slightly upset/dissatisfied customer response (score 0-6). Mention specific product problems (e.g., billing errors, performance slowness, missing feature names, dates, amounts, account ids, error messages) and sign with a believable name and company. Use this exact format (SCORE then COMMENT):\n\nSCORE: [0-6]\nCOMMENT: [detailed multi-line comment describing frustrations and concrete examples]'
          
        WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 4 THEN
          'You are generating an NPS response for CloudFlow Pro. Produce a realistic, balanced/neutral customer response (score 7-8) that includes both positives and negatives. Mention concrete features they like and pain points they encounter (examples, dates, small suggestions). Use this exact format:\n\nSCORE: [7-8]\nCOMMENT: [detailed comment with balanced feedback]'
          
        ELSE
          'You are generating an NPS response for CloudFlow Pro. Produce a realistic, enthusiastic promoter response (score 9-10). Include specific features the customer loves (names of features, workflows, integrations), business value, and a short sign-off. Use this exact format:\n\nSCORE: [9-10]\nCOMMENT: [detailed, specific praise and examples]'
      END AS prompt
    FROM `{project_id}.churn_prevention_demo.customers`
    WHERE MOD(ABS(FARM_FINGERPRINT(customer_id)), 7) = 0
    LIMIT 50
  ),
  STRUCT(
    0.85 AS temperature,  -- higher temperature for variety
    2000 AS max_output_tokens,
    TRUE AS flatten_json_output
  )
);

-- Step 2: Create the table and insert data in one go
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.nps_feedback`
PARTITION BY DATE(ts) 
CLUSTER BY customer_id
AS
SELECT
  CONCAT('NPS_', GENERATE_UUID()) AS nps_id,
  TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL CAST(FLOOR(RAND() * 60) AS INT64) DAY) AS ts,
  customer_id,

  -- final score: prefer extracted value, otherwise deterministic fallback based on fingerprint
  COALESCE(
    score_extracted,
    CASE 
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 2 THEN CAST(1 + RAND() * 5 AS INT64)   -- 1..5 (detractor-ish)
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 4 THEN CAST(7 + RAND() * 1 AS INT64)   -- 7..8 (passive)
      ELSE CAST(9 + RAND() * 1 AS INT64)                                                      -- 9..10 (promoter)
    END
  ) AS score,

  -- comment: prefer extracted, fallback to default message
  COALESCE(
    NULLIF(TRIM(comment_extracted), ''),
    'No comment provided'
  ) AS comment,

  -- category derived from the final score
  CASE
    WHEN COALESCE(
      score_extracted,
      CASE 
        WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 2 THEN CAST(1 + RAND() * 5 AS INT64)
        WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 4 THEN CAST(7 + RAND() * 1 AS INT64)
        ELSE CAST(9 + RAND() * 1 AS INT64)
      END
    ) <= 6 THEN 'detractor'
    WHEN COALESCE(
      score_extracted,
      CASE 
        WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 2 THEN CAST(1 + RAND() * 5 AS INT64)
        WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 4 THEN CAST(7 + RAND() * 1 AS INT64)
        ELSE CAST(9 + RAND() * 1 AS INT64)
      END
    ) <= 8 THEN 'passive'
    ELSE 'promoter'
  END AS category

FROM (
  -- inner extract: do the potentially-failing extracts once and reuse them above
  SELECT
    customer_id,
    raw_response,
    SAFE_CAST(REGEXP_EXTRACT(raw_response, r'(?i)SCORE:\s*([0-9]+)') AS INT64) AS score_extracted,
    REGEXP_EXTRACT(raw_response, r'(?si)COMMENT:\s*(.*)') AS comment_extracted
  FROM nps_raw_responses
) t
WHERE raw_response IS NOT NULL;

-- Step 3: Debug - show a few raw responses + what we extracted
SELECT
  'Debug: Sample NPS raw responses' AS debug_info,
  customer_id,
  score_extracted,
  IFNULL(LENGTH(raw_response), 0) AS response_length,
  raw_response
FROM (
  SELECT
    customer_id,
    raw_response,
    SAFE_CAST(REGEXP_EXTRACT(raw_response, r'(?i)SCORE:\s*([0-9]+)') AS INT64) AS score_extracted
  FROM nps_raw_responses
)
LIMIT 5;
"""

try:
    query_job = client.query(generate_nps_sql)
    query_job.result()
    print("Generated NPS feedback successfully.")
except Exception as e:
    print(f"Error generating NPS feedback: {e}")




# Verify NPS feedback generation
verify_nps_sql = f"""
SELECT 
  category, 
  COUNT(*) as count, 
  AVG(score) as avg_score
FROM `{project_id}.churn_prevention_demo.nps_feedback`
GROUP BY category
ORDER BY avg_score
"""

try:
    query_job = client.query(verify_nps_sql)
    results = query_job.result()
    
    print("NPS feedback distribution:")
    for row in results:
        print(f"- {row.category}: {row.count} responses, Avg score: {row.avg_score:.2f}")
except Exception as e:
    print(f"Error verifying NPS feedback: {e}")




# Sample some generated NPS feedback
sample_nps_sql = f"""
SELECT 
  customer_id,
  score,
  comment as comment,
  category
FROM `{project_id}.churn_prevention_demo.nps_feedback`
ORDER BY score ASC
LIMIT 3
"""

try:
    query_job = client.query(sample_nps_sql)
    results = query_job.result()
    
    print("Sample NPS feedback (lowest scores):")
    for row in results:
        print(f"\nCustomer: {row.customer_id}")
        print(f"Score: {row.score} ({row.category})")
        print(f"Comment: {row.comment}")
except Exception as e:
    print(f"Error sampling NPS feedback: {e}")




verification_sql = f"""
SELECT 'customers' as table_name, COUNT(*) as record_count FROM `{project_id}.churn_prevention_demo.customers`
UNION ALL
SELECT 'usage_logs' as table_name, COUNT(*) as record_count FROM `{project_id}.churn_prevention_demo.usage_logs`
UNION ALL
SELECT 'support_tickets' as table_name, COUNT(*) as record_count FROM `{project_id}.churn_prevention_demo.support_tickets`
UNION ALL
SELECT 'nps_feedback' as table_name, COUNT(*) as record_count FROM `{project_id}.churn_prevention_demo.nps_feedback`
"""

try:
    query_job = client.query(verification_sql)
    results = query_job.result()
    
    print("Data generation summary:")
    for row in results:
        print(f"- {row.table_name}: {row.record_count} records")
except Exception as e:
    print(f"Error verifying data: {e}")




partition_sql = f"""
SELECT table_name, partition_id, total_rows
FROM `{project_id}.churn_prevention_demo.INFORMATION_SCHEMA.PARTITIONS`
WHERE table_name = 'usage_logs'
ORDER BY partition_id DESC
LIMIT 5
"""

try:
    query_job = client.query(partition_sql)
    results = query_job.result()
    
    print("Partition information for usage_logs:")
    for row in results:
        print(f"- Partition {row.partition_id}: {row.total_rows} rows")
except Exception as e:
    print(f"Error checking partitioning: {e}")




# Create the customer_features table with comprehensive metrics
customer_features_sql = f"""
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.customer_features` AS
WITH usage_stats AS (
  SELECT 
    customer_id,
    -- Recency features
    DATE_DIFF(CURRENT_DATE(), DATE(MAX(ts)), DAY) AS days_since_last_usage,
    DATE_DIFF(CURRENT_DATE(), DATE(MIN(ts)), DAY) AS customer_age_days,
    
    -- Frequency features (different time windows)
    COALESCE(SUM(CASE WHEN DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) THEN sessions ELSE 0 END), 0) AS sessions_7d,
    COALESCE(SUM(CASE WHEN DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN sessions ELSE 0 END), 0) AS sessions_30d,
    COALESCE(SUM(CASE WHEN DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY) AND DATE(ts) < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN sessions ELSE 0 END), 0) AS sessions_30_60d,
    
    -- Engagement depth
    AVG(CASE WHEN DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN minutes/NULLIF(sessions,0) ELSE NULL END) AS avg_session_duration_30d,
    COUNT(DISTINCT CASE WHEN DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN feature_usage END) AS unique_features_30d,
    
    -- Trend indicators
    COALESCE(
      CASE 
        WHEN IS_NAN(CORR(DATE_DIFF(CURRENT_DATE(), DATE(ts), DAY), sessions)) THEN 0
        ELSE CORR(DATE_DIFF(CURRENT_DATE(), DATE(ts), DAY), sessions)
      END, 
      0
    ) AS usage_trend_correlation
  FROM `{project_id}.churn_prevention_demo.usage_logs`
  GROUP BY customer_id
),
support_stats AS (
  SELECT 
    customer_id,
    COUNT(*) AS support_tickets_total,
    COUNT(CASE WHEN DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN 1 END) AS support_tickets_30d,
    AVG(sentiment) AS avg_sentiment,
    MIN(sentiment) AS min_sentiment
  FROM `{project_id}.churn_prevention_demo.support_tickets`
  GROUP BY customer_id
)
SELECT 
  c.customer_id,
  c.plan,
  c.mrr,
  c.signup_date,
  
  -- Usage features
  COALESCE(u.days_since_last_usage, 999) AS days_since_last_usage,
  COALESCE(u.customer_age_days, 0) AS customer_age_days,
  COALESCE(u.sessions_7d, 0) AS sessions_7d,
  COALESCE(u.sessions_30d, 0) AS sessions_30d,
  COALESCE(u.sessions_30_60d, 0) AS sessions_30_60d,
  
  -- Calculated ratios
  COALESCE(SAFE_DIVIDE(u.sessions_30d, NULLIF(u.sessions_30_60d, 0)), 0) AS usage_ratio_30_vs_60d,
  COALESCE(u.avg_session_duration_30d, 0) AS avg_session_duration_30d,
  COALESCE(u.unique_features_30d, 0) AS unique_features_30d,
  COALESCE(u.usage_trend_correlation, 0) AS usage_trend_correlation,
  
  -- Support features  
  COALESCE(s.support_tickets_total, 0) AS support_tickets_total,
  COALESCE(s.support_tickets_30d, 0) AS support_tickets_30d,
  COALESCE(s.avg_sentiment, 0) AS avg_sentiment,
  COALESCE(s.min_sentiment, 0) AS min_sentiment,
  
  -- Risk flags
  CASE WHEN COALESCE(u.sessions_30d, 0) = 0 THEN 1 ELSE 0 END AS zero_usage_30d,
  CASE WHEN COALESCE(u.sessions_7d, 0) = 0 THEN 1 ELSE 0 END AS zero_usage_7d,
  
  CURRENT_TIMESTAMP() AS feature_created_at
FROM `{project_id}.churn_prevention_demo.customers` c
LEFT JOIN usage_stats u USING(customer_id)
LEFT JOIN support_stats s USING(customer_id);
"""

try:
    query_job = client.query(customer_features_sql)
    query_job.result()
    print("Customer features table created successfully.")
except Exception as e:
    print(f"Error creating customer features table: {e}")




# Verify feature distributions
verify_features_sql = f"""
SELECT 
  plan, 
  COUNT(*) as count, 
  AVG(sessions_30d) as avg_sessions_30d
FROM `{project_id}.churn_prevention_demo.customer_features`
GROUP BY plan
ORDER BY count DESC
"""

try:
    query_job = client.query(verify_features_sql)
    results = query_job.result()
    
    print("Feature verification completed:")
    for row in results:
        print(f"- {row.plan}: {row.count} customers, avg_sessions_30d: {row.avg_sessions_30d:.2f}")
except Exception as e:
    print(f"Error verifying features: {e}")




# Feature quality checks: check for nulls in critical fields and infinite values
feature_quality_sql = f"""
SELECT
  *
FROM `{project_id}.churn_prevention_demo.customer_features`
"""

query_job = client.query(feature_quality_sql)
results = query_job.result()
df = pd.DataFrame([dict(row) for row in results])

print("Number of null values in each column:")
df.isnull().sum()



# Create time series data for forecasting
usage_timeseries_sql = f"""
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.usage_timeseries` AS
SELECT 
  customer_id,
  DATE_TRUNC(DATE(ts), WEEK) AS week_start,
  SUM(sessions) AS weekly_sessions,
  AVG(minutes) AS avg_weekly_minutes,
  COUNT(*) AS days_with_usage,  -- Track data completeness
  -- Add some smoothing for more stable forecasts
  SUM(sessions) + (7 - COUNT(*)) * (SUM(sessions) / COUNT(*)) * 0.1 AS smoothed_weekly_sessions
FROM `{project_id}.churn_prevention_demo.usage_logs`
GROUP BY customer_id, week_start
-- More lenient filter - need at least 2 data points per week instead of 4
HAVING COUNT(*) >= 2 
   AND SUM(sessions) > 0  -- Ensure actual usage occurred
ORDER BY customer_id, week_start;
"""

try:
    query_job = client.query(usage_timeseries_sql)
    query_job.result()
    print("Usage timeseries table created successfully.")
except Exception as e:
    print(f"Error creating usage timeseries table: {e}")




# Generate forecasts using AI.FORECAST and pivot for ML features
usage_forecasts_sql = f"""
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.usage_forecasts` AS

-- Step 1: Check data availability first
WITH data_check AS (
  SELECT 
    customer_id,
    COUNT(*) AS weeks_of_data,
    MIN(week_start) AS first_week,
    MAX(week_start) AS last_week,
    AVG(weekly_sessions) AS avg_sessions
  FROM `{project_id}.churn_prevention_demo.usage_timeseries`
  GROUP BY customer_id
  -- Only include customers with enough historical data for forecasting
  HAVING COUNT(*) >= 6  -- At least 6 weeks of data
     AND AVG(weekly_sessions) >= 1  -- Some meaningful usage
),

-- Step 2: Create forecasts only for customers with sufficient data
raw_forecasts AS (
  SELECT 
    customer_id,
    forecast_timestamp AS forecast_week,
    forecast_value AS predicted_sessions,
    confidence_level,
    prediction_interval_lower_bound,
    prediction_interval_upper_bound
  FROM 
    AI.FORECAST(
      (
        SELECT 
          ut.customer_id, 
          ut.week_start,
          -- Use smoothed data for more stable forecasts
          ut.smoothed_weekly_sessions AS weekly_sessions
        FROM `{project_id}.churn_prevention_demo.usage_timeseries` ut
        INNER JOIN data_check dc USING(customer_id)
        WHERE ut.week_start >= DATE_SUB(CURRENT_DATE(), INTERVAL 16 WEEK)  -- More history
        ORDER BY ut.customer_id, ut.week_start
      ),
      timestamp_col => 'week_start',
      data_col => 'weekly_sessions',
      id_cols => ['customer_id'],
      horizon => 4,
      confidence_level => 0.95,
      model => 'TimesFM 2.0'
    )
),

-- Step 3: Pivot forecasting data
forecast_features AS (
  SELECT 
    customer_id,
    forecast_week,
    GREATEST(predicted_sessions, 0) AS predicted_sessions,  -- Ensure non-negative
    prediction_interval_lower_bound,
    prediction_interval_upper_bound,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY forecast_week) as week_num
  FROM raw_forecasts
)

SELECT 
  customer_id,
  
  -- Individual week forecasts
  MAX(CASE WHEN week_num = 1 THEN predicted_sessions END) AS forecast_week1_sessions,
  MAX(CASE WHEN week_num = 2 THEN predicted_sessions END) AS forecast_week2_sessions,
  MAX(CASE WHEN week_num = 3 THEN predicted_sessions END) AS forecast_week3_sessions,
  MAX(CASE WHEN week_num = 4 THEN predicted_sessions END) AS forecast_week4_sessions,
  
  -- Confidence intervals for first and last week
  MAX(CASE WHEN week_num = 1 THEN prediction_interval_lower_bound END) AS forecast_week1_lower,
  MAX(CASE WHEN week_num = 1 THEN prediction_interval_upper_bound END) AS forecast_week1_upper,
  MAX(CASE WHEN week_num = 4 THEN prediction_interval_lower_bound END) AS forecast_week4_lower,
  MAX(CASE WHEN week_num = 4 THEN prediction_interval_upper_bound END) AS forecast_week4_upper,
  
  -- Aggregate features for ML model
  AVG(CASE WHEN week_num BETWEEN 1 AND 4 THEN predicted_sessions END) AS forecast_avg_sessions,
  MAX(CASE WHEN week_num BETWEEN 1 AND 4 THEN predicted_sessions END) AS forecast_max_sessions,
  MIN(CASE WHEN week_num BETWEEN 1 AND 4 THEN predicted_sessions END) AS forecast_min_sessions,
  
  -- Trend indicators
  SAFE_DIVIDE(
    (MAX(CASE WHEN week_num = 4 THEN predicted_sessions END) - 
     MAX(CASE WHEN week_num = 1 THEN predicted_sessions END)),
    NULLIF(MAX(CASE WHEN week_num = 1 THEN predicted_sessions END), 0)
  ) AS forecast_trend_slope,
   
  -- Uncertainty indicators
  SAFE_DIVIDE(
    (MAX(CASE WHEN week_num = 1 THEN prediction_interval_upper_bound END) - 
     MAX(CASE WHEN week_num = 1 THEN prediction_interval_lower_bound END)),
    NULLIF(MAX(CASE WHEN week_num = 1 THEN predicted_sessions END), 0)
  ) AS forecast_week1_uncertainty,
  
  SAFE_DIVIDE(
    (MAX(CASE WHEN week_num = 4 THEN prediction_interval_upper_bound END) - 
     MAX(CASE WHEN week_num = 4 THEN prediction_interval_lower_bound END)),
    NULLIF(MAX(CASE WHEN week_num = 4 THEN predicted_sessions END), 0)
  ) AS forecast_week4_uncertainty,
  
  -- Risk indicators
  CASE 
    WHEN MAX(CASE WHEN week_num = 4 THEN predicted_sessions END) < 
         MAX(CASE WHEN week_num = 1 THEN predicted_sessions END) * 0.6 THEN TRUE
    ELSE FALSE 
  END AS forecast_steep_decline,
  
  CASE 
    WHEN AVG(CASE WHEN week_num BETWEEN 1 AND 4 THEN predicted_sessions END) < 3 THEN TRUE
    ELSE FALSE 
  END AS forecast_low_usage,
  
  -- Forecast volatility
  STDDEV(CASE WHEN week_num BETWEEN 1 AND 4 THEN predicted_sessions END) AS forecast_volatility,
  
  -- Confidence score (lower uncertainty = higher confidence)
  1.0 / (1.0 + AVG(CASE WHEN week_num BETWEEN 1 AND 4 THEN 
    prediction_interval_upper_bound - prediction_interval_lower_bound END)) AS forecast_confidence
  
FROM forecast_features
GROUP BY customer_id

UNION ALL

-- Include customers without forecasts (too little data) with NULL values
SELECT 
  customer_id,
  NULL AS forecast_week1_sessions,
  NULL AS forecast_week2_sessions,
  NULL AS forecast_week3_sessions,
  NULL AS forecast_week4_sessions,
  NULL AS forecast_week1_lower,
  NULL AS forecast_week1_upper,
  NULL AS forecast_week4_lower,
  NULL AS forecast_week4_upper,
  NULL AS forecast_avg_sessions,
  NULL AS forecast_max_sessions,
  NULL AS forecast_min_sessions,
  NULL AS forecast_trend_slope,
  NULL AS forecast_week1_uncertainty,
  NULL AS forecast_week4_uncertainty,
  FALSE AS forecast_steep_decline,  -- Conservative defaults
  TRUE AS forecast_low_usage,       -- Flag as low usage due to insufficient data
  NULL AS forecast_volatility,
  0.0 AS forecast_confidence        -- Low confidence due to no forecast
FROM `{project_id}.churn_prevention_demo.customers`
WHERE customer_id NOT IN (SELECT customer_id FROM data_check)

ORDER BY customer_id;
"""

try:
    query_job = client.query(usage_forecasts_sql)
    query_job.result()
    print("Enhanced usage forecasts with ML features generated successfully.")
except Exception as e:
    print(f"Error generating enhanced usage forecasts: {e}")



# Convert forecast results to pandas DataFrame for easy viewing
query_job = client.query(f"""
SELECT * 
FROM `{project_id}.churn_prevention_demo.usage_forecasts` 
ORDER BY customer_id 
LIMIT 5
""")

print("Forecast results for 5 customers:")
results = query_job.result()
df = pd.DataFrame([dict(row) for row in results])
df



# Create churn labels based on business rules + statistical approach
churn_labels_hybrid_sql = f"""
DROP TABLE IF EXISTS `{project_id}.churn_prevention_demo.churn_labels_generated`;

CREATE TABLE `{project_id}.churn_prevention_demo.churn_labels_generated` 
CLUSTER BY customer_id
AS
WITH customer_risk AS (
  SELECT 
    customer_id,
    CASE 
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) = 0 THEN 'high_risk'
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 5) = 0 THEN 'medium_risk'
      ELSE 'low_risk'
    END AS synthetic_churn_risk
  FROM `{project_id}.churn_prevention_demo.customers`
),
churn_analysis AS (
  SELECT 
    cf.customer_id,
    cr.synthetic_churn_risk,
    cf.days_since_last_usage,
    cf.sessions_30d,
    cf.sessions_30_60d,
    cf.usage_ratio_30_vs_60d,
    cf.avg_sentiment,
    COALESCE(uf.forecast_steep_decline, FALSE) as forecast_steep_decline,
    COALESCE(uf.forecast_low_usage, FALSE) as forecast_low_usage,
    
    -- Determine primary churn reason FIRST (all of these values are hard-coded and kinda ficticious)
    CASE 
      WHEN cf.days_since_last_usage > 25 THEN 'no_activity'
      WHEN cf.days_since_last_usage > 15 THEN 'reduced_activity'
      WHEN cf.days_since_last_usage > 10 THEN 'declining_activity'
      WHEN cf.sessions_30d = 0 AND cf.sessions_30_60d > 0 THEN 'sudden_stop'
      WHEN cf.usage_ratio_30_vs_60d < 0.4 THEN 'steep_decline'
      WHEN cf.usage_ratio_30_vs_60d < 0.6 THEN 'declining_usage'  
      WHEN cf.avg_sentiment < -0.6 THEN 'very_negative_sentiment'
      WHEN cf.avg_sentiment < -0.3 THEN 'negative_sentiment'
      WHEN COALESCE(uf.forecast_steep_decline, FALSE) = TRUE 
           AND COALESCE(uf.forecast_low_usage, FALSE) = TRUE THEN 'low_forecasted_usage'
      ELSE 'active'
    END AS primary_churn_reason
    
  FROM `{project_id}.churn_prevention_demo.customer_features` cf
  LEFT JOIN `{project_id}.churn_prevention_demo.usage_forecasts` uf USING(customer_id)
  LEFT JOIN customer_risk cr USING(customer_id)
)
SELECT 
  customer_id,
  CASE 
    -- High-risk customers: churn if they have ANY concerning signal
    WHEN synthetic_churn_risk = 'high_risk' AND primary_churn_reason IN (
      'no_activity', 'reduced_activity', 'declining_activity', 'sudden_stop',
      'steep_decline', 'declining_usage', 'very_negative_sentiment', 
      'negative_sentiment', 'low_forecasted_usage'
    ) THEN TRUE
    
    -- Medium-risk customers: churn if they have moderate+ signals
    WHEN synthetic_churn_risk = 'medium_risk' AND primary_churn_reason IN (
      'no_activity', 'reduced_activity', 'sudden_stop', 'steep_decline', 
      'very_negative_sentiment', 'low_forecasted_usage'
    ) THEN TRUE
    
    -- Low-risk customers: churn only on severe signals
    WHEN synthetic_churn_risk = 'low_risk' AND primary_churn_reason IN (
      'no_activity', 'sudden_stop', 'steep_decline', 'very_negative_sentiment'
    ) THEN TRUE
    
    ELSE FALSE
  END AS churned,
  
  -- Churn reason is consistent with churned flag
  CASE 
    WHEN synthetic_churn_risk = 'high_risk' AND primary_churn_reason IN (
      'no_activity', 'reduced_activity', 'declining_activity', 'sudden_stop',
      'steep_decline', 'declining_usage', 'very_negative_sentiment', 
      'negative_sentiment', 'low_forecasted_usage'
    ) THEN primary_churn_reason
    
    WHEN synthetic_churn_risk = 'medium_risk' AND primary_churn_reason IN (
      'no_activity', 'reduced_activity', 'sudden_stop', 'steep_decline', 
      'very_negative_sentiment', 'low_forecasted_usage'
    ) THEN primary_churn_reason
    
    WHEN synthetic_churn_risk = 'low_risk' AND primary_churn_reason IN (
      'no_activity', 'sudden_stop', 'steep_decline', 'very_negative_sentiment'
    ) THEN primary_churn_reason
    
    ELSE 'active'
  END AS churn_reason,
  
  CURRENT_DATE() AS churn_date
  
FROM churn_analysis
"""

try:
    query_job = client.query(churn_labels_hybrid_sql)
    query_job.result()
    print("Churn labels generated successfully.")
except Exception as e:
    print(f"Error generating churn labels: {e}")



# Verify churn label distribution
verify_churn_labels_sql = f"""
WITH risk_breakdown AS (
  SELECT 
    CASE 
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) = 0 THEN 'high_risk'
      WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 5) = 0 THEN 'medium_risk'
      ELSE 'low_risk'
    END AS risk_profile,
    churned,
    churn_reason
  FROM `{project_id}.churn_prevention_demo.churn_labels_generated`
)
SELECT 
  risk_profile,
  churned,
  churn_reason,
  COUNT(*) as count,
  COUNT(*) / SUM(COUNT(*)) OVER() AS percentage
FROM risk_breakdown
GROUP BY risk_profile, churned, churn_reason
ORDER BY risk_profile, churned DESC, count DESC;
"""

query_job = client.query(verify_churn_labels_sql)
results = query_job.result()

print("Churn label distribution:")
df = pd.DataFrame([dict(row) for row in results])
df



# Create train/test split for proper model evaluation
create_train_test_split_sql = f"""
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.model_train_test_split` AS
SELECT 
  customer_id,
  -- Create a train/test split column where 80% is training and 20% is test
  -- using a deterministic hash function to ensure consistent splits
  CASE 
    WHEN MOD(ABS(FARM_FINGERPRINT(customer_id)), 10) < 8 THEN 'TRAIN'
    ELSE 'TEST'
  END AS data_split
FROM `{project_id}.churn_prevention_demo.customers`;
"""

try:
    query_job = client.query(create_train_test_split_sql)
    query_job.result()
    print("Created train/test split successfully.")
except Exception as e:
    print(f"Error creating train/test split: {e}")




# Train churn prediction model with multiple features ONLY on training data
train_churn_model_sql = f"""
CREATE OR REPLACE MODEL `{project_id}.churn_prevention_demo.churn_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['churned'],
  max_iterations=50,
  learn_rate=0.1,
  subsample=0.8,
  early_stop=TRUE,
  enable_global_explain=TRUE
) AS
SELECT 
  -- Customer basic features
  cf.sessions_7d,
  cf.sessions_30d,
  cf.sessions_30_60d,
  cf.usage_ratio_30_vs_60d,
  cf.days_since_last_usage,
  cf.customer_age_days,
  cf.avg_session_duration_30d,
  cf.unique_features_30d,
  cf.usage_trend_correlation,
  cf.support_tickets_30d,
  cf.avg_sentiment,
  cf.min_sentiment,
  LOG(cf.mrr + 1) AS log_mrr,
  
  -- Categorical features
  cf.plan,
  cf.zero_usage_7d,
  cf.zero_usage_30d,
  
  -- Forecast features (with proper null handling)
  uf.forecast_week1_sessions,
  uf.forecast_week4_sessions,
  COALESCE(uf.forecast_avg_sessions, 0) AS forecast_avg_sessions,
  COALESCE(uf.forecast_trend_slope, 0) AS forecast_trend_slope,
  COALESCE(uf.forecast_steep_decline, FALSE) AS forecast_steep_decline,
  COALESCE(uf.forecast_low_usage, FALSE) AS forecast_low_usage,
  COALESCE(uf.forecast_week1_uncertainty, 0) AS forecast_week1_uncertainty,
  COALESCE(uf.forecast_week4_uncertainty, 0) AS forecast_week4_uncertainty,
  COALESCE(uf.forecast_volatility, 0) AS forecast_volatility,
  COALESCE(uf.forecast_confidence, 0) AS forecast_confidence,
  
  -- Derived features
  SAFE_DIVIDE(cf.support_tickets_30d, cf.customer_age_days) as ticket_rate,
  SAFE_DIVIDE(cf.support_tickets_30d, cf.sessions_30d) as support_per_session_ratio,
  
  -- Target variable
  cl.churned
FROM `{project_id}.churn_prevention_demo.customer_features` cf
JOIN `{project_id}.churn_prevention_demo.churn_labels_generated` cl USING(customer_id)
LEFT JOIN `{project_id}.churn_prevention_demo.usage_forecasts` uf USING(customer_id)
JOIN `{project_id}.churn_prevention_demo.model_train_test_split` split USING(customer_id)
WHERE split.data_split = 'TRAIN';
"""

try:
    query_job = client.query(train_churn_model_sql)
    query_job.result()
    print("Churn prediction model trained successfully.")
except Exception as e:
    print(f"Error training churn model: {e}")



# Evaluate model performance on test data (proper out-of-sample evaluation)
evaluate_model_sql = f"""
SELECT *
FROM ML.EVALUATE(
  MODEL `{project_id}.churn_prevention_demo.churn_model`,
  (
    SELECT 
      -- Features (same as in training)
      cf.sessions_7d,
      cf.sessions_30d,
      cf.sessions_30_60d,
      cf.usage_ratio_30_vs_60d,
      cf.days_since_last_usage,
      cf.customer_age_days,
      cf.avg_session_duration_30d,
      cf.unique_features_30d,
      cf.usage_trend_correlation,
      cf.support_tickets_30d,
      cf.avg_sentiment,
      cf.min_sentiment,
      LOG(cf.mrr + 1) AS log_mrr,
      cf.plan,
      cf.zero_usage_7d,
      cf.zero_usage_30d,
      
      -- Forecast features with proper null handling
      uf.forecast_week1_sessions,
      uf.forecast_week4_sessions,
      COALESCE(uf.forecast_avg_sessions, 0) AS forecast_avg_sessions,
      COALESCE(uf.forecast_trend_slope, 0) AS forecast_trend_slope,
      COALESCE(uf.forecast_steep_decline, FALSE) AS forecast_steep_decline,
      COALESCE(uf.forecast_low_usage, FALSE) AS forecast_low_usage,
      COALESCE(uf.forecast_week1_uncertainty, 0) AS forecast_week1_uncertainty,
      COALESCE(uf.forecast_week4_uncertainty, 0) AS forecast_week4_uncertainty,
      COALESCE(uf.forecast_volatility, 0) AS forecast_volatility,
      COALESCE(uf.forecast_confidence, 0) AS forecast_confidence,
      
      -- Derived features
      SAFE_DIVIDE(cf.support_tickets_30d, cf.customer_age_days) as ticket_rate,
      SAFE_DIVIDE(cf.support_tickets_30d, cf.sessions_30d) as support_per_session_ratio,
      
      -- Target variable
      cl.churned
    FROM `{project_id}.churn_prevention_demo.customer_features` cf
    JOIN `{project_id}.churn_prevention_demo.churn_labels_generated` cl USING(customer_id)
    LEFT JOIN `{project_id}.churn_prevention_demo.usage_forecasts` uf USING(customer_id)
    JOIN `{project_id}.churn_prevention_demo.model_train_test_split` split USING(customer_id)
    WHERE split.data_split = 'TEST'
  )
)
"""

query_job = client.query(evaluate_model_sql)
results = query_job.result()

print("Model Evaluation Results on Test Data:")
df = pd.DataFrame([dict(row) for row in results])
df



# Get feature importance
feature_importance_sql = f"""
SELECT *
FROM ML.FEATURE_IMPORTANCE(MODEL `{project_id}.churn_prevention_demo.churn_model`)
ORDER BY importance_weight DESC, importance_gain DESC
LIMIT 15
"""

query_job = client.query(feature_importance_sql)
results = query_job.result()

print("Top 15 Features by Importance:")
df = pd.DataFrame([dict(row) for row in results])
df



# Generate predictions for all customers, marking test data as true "out-of-sample" predictions
generate_predictions_sql = f"""
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.churn_predictions` AS
SELECT 
  customer_id,
  predicted_churned, -- Either TRUE or FALSE
  predicted_churned_probs[OFFSET(0)].prob AS churn_probability,
  predicted_churned_probs[OFFSET(1)].prob AS retention_probability,
  split.data_split AS prediction_type -- Mark whether this was a TRAIN or TEST customer
FROM ML.PREDICT(
  MODEL `{project_id}.churn_prevention_demo.churn_model`,
  (
    SELECT 
      -- IMPORTANT: Include customer_id first
      cf.customer_id,
      
      -- Customer basic features
      cf.sessions_7d,
      cf.sessions_30d,
      cf.sessions_30_60d,
      cf.usage_ratio_30_vs_60d,
      cf.days_since_last_usage,
      cf.customer_age_days,
      cf.avg_session_duration_30d,
      cf.unique_features_30d,
      cf.usage_trend_correlation,
      cf.support_tickets_30d,
      cf.avg_sentiment,
      cf.min_sentiment,
      LOG(cf.mrr + 1) AS log_mrr,
      
      -- Categorical features
      cf.plan,
      cf.zero_usage_7d,
      cf.zero_usage_30d,
      
      -- Forecast features (with proper null handling)
      COALESCE(uf.forecast_week1_sessions, 0) AS forecast_week1_sessions,
      COALESCE(uf.forecast_week4_sessions, 0) AS forecast_week4_sessions,
      COALESCE(uf.forecast_avg_sessions, 0) AS forecast_avg_sessions,
      COALESCE(uf.forecast_trend_slope, 0) AS forecast_trend_slope,
      COALESCE(uf.forecast_steep_decline, FALSE) AS forecast_steep_decline,
      COALESCE(uf.forecast_low_usage, FALSE) AS forecast_low_usage,
      COALESCE(uf.forecast_week1_uncertainty, 0) AS forecast_week1_uncertainty,
      COALESCE(uf.forecast_week4_uncertainty, 0) AS forecast_week4_uncertainty,
      COALESCE(uf.forecast_volatility, 0) AS forecast_volatility,
      COALESCE(uf.forecast_confidence, 0) AS forecast_confidence,
      
      -- Derived features
      SAFE_DIVIDE(cf.support_tickets_30d, cf.customer_age_days) as ticket_rate,
      SAFE_DIVIDE(cf.support_tickets_30d, cf.sessions_30d) as support_per_session_ratio
      
    FROM `{project_id}.churn_prevention_demo.customer_features` cf
    LEFT JOIN `{project_id}.churn_prevention_demo.usage_forecasts` uf USING(customer_id)
    JOIN `{project_id}.churn_prevention_demo.model_train_test_split` split USING(customer_id)
  )
)
JOIN `{project_id}.churn_prevention_demo.model_train_test_split` split USING(customer_id);
"""

try:
    query_job = client.query(generate_predictions_sql)
    query_job.result()
    print("Churn predictions generated successfully.")
except Exception as e:
    print(f"Error generating predictions: {e}")



# Analyze predictions distribution, comparing train vs test results
prediction_distribution_sql = f"""
SELECT 
  prediction_type,
  CASE
    WHEN churn_probability >= 0.8 THEN 'Very High Risk (80-100%)'
    WHEN churn_probability >= 0.6 THEN 'High Risk (60-80%)'
    WHEN churn_probability >= 0.4 THEN 'Medium Risk (40-60%)'
    WHEN churn_probability >= 0.2 THEN 'Low Risk (20-40%)'
    ELSE 'Very Low Risk (0-20%)'
  END AS risk_category,
  COUNT(*) as customer_count,
  AVG(churn_probability) as avg_probability
FROM `{project_id}.churn_prevention_demo.churn_predictions`
GROUP BY prediction_type, risk_category
ORDER BY prediction_type, avg_probability DESC
"""

query_job = client.query(prediction_distribution_sql)
results = query_job.result()

print("Churn prediction distribution:")
df = pd.DataFrame([dict(row) for row in results])
df



high_risk_customers_sql = f"""
SELECT 
  cp.customer_id, 
  cp.churn_probability,
  cf.plan,
  cf.days_since_last_usage,
  cf.sessions_30d,
  cf.sessions_30_60d,
  cf.usage_ratio_30_vs_60d,
  cf.support_tickets_30d,
  cf.avg_sentiment,
  COALESCE(uf.forecast_trend_slope, 0) AS forecast_trend
FROM `{project_id}.churn_prevention_demo.churn_predictions` cp
JOIN `{project_id}.churn_prevention_demo.customer_features` cf USING(customer_id)
LEFT JOIN `{project_id}.churn_prevention_demo.usage_forecasts` uf USING(customer_id)
LEFT JOIN `{project_id}.churn_prevention_demo.churn_labels_generated` cl USING(customer_id)
WHERE cp.churn_probability > 0.7
ORDER BY cp.churn_probability DESC
LIMIT 10
"""

query_job = client.query(high_risk_customers_sql)
results = query_job.result()

print("Top 10 highest churn risk customers:")
df = pd.DataFrame([dict(row) for row in results])
df



# Create remote embedding model
embedding_model_sql = f"""
CREATE OR REPLACE MODEL `{project_id}.churn_prevention_demo.embedding_model`
REMOTE WITH CONNECTION `{location}.llm-connection`
OPTIONS(ENDPOINT = 'gemini-embedding-001');
"""

try:
    query_job = client.query(embedding_model_sql)
    query_job.result()
    print("Embedding model created successfully.")
except Exception as e:
    print(f"Error creating embedding model: {e}")



create_text_embeddings_sql = f"""
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.text_embeddings` AS

-- Support tickets embeddings
SELECT
  CONCAT('ticket_', ticket_id) AS artifact_id,
  'support_ticket' AS artifact_type,
  customer_id,
  CONCAT(subject, '. ', body) AS text_content,
  ts AS created_at,
  sentiment,
  priority,
  ml_generate_embedding_result AS embedding_vector
FROM ML.GENERATE_EMBEDDING(
  MODEL `{project_id}.churn_prevention_demo.embedding_model`,
  (
    SELECT 
      ticket_id,
      customer_id,
      subject,
      body,
      ts,
      sentiment,
      priority,
      CONCAT(subject, '. ', SUBSTR(body, 1, 500)) AS content
    FROM `{project_id}.churn_prevention_demo.support_tickets`
    WHERE LENGTH(body) > 10
  ),
  STRUCT(TRUE AS flatten_json_output)
)

UNION ALL

-- NPS feedback embeddings
SELECT
  CONCAT('nps_', nps_id) AS artifact_id,
  'nps_feedback' AS artifact_type,
  customer_id,
  comment AS text_content,
  ts AS created_at,
  CAST(score AS FLOAT64) / 10.0 AS sentiment,  -- Normalize score to 0-1 range
  CASE 
    WHEN category = 'detractor' THEN 'high'
    WHEN category = 'passive' THEN 'medium' 
    ELSE 'low' 
  END AS priority,
  ml_generate_embedding_result AS embedding_vector
FROM ML.GENERATE_EMBEDDING(
  MODEL `{project_id}.churn_prevention_demo.embedding_model`,
  (
    SELECT 
      nps_id,
      customer_id,
      comment,
      ts,
      score,
      category,
      comment AS content  -- Use the full comment as embedding content
    FROM `{project_id}.churn_prevention_demo.nps_feedback`
    WHERE comment IS NOT NULL AND LENGTH(comment) > 5
  ),
  STRUCT(TRUE AS flatten_json_output)
);
"""

try:
    query_job = client.query(create_text_embeddings_sql)
    query_job.result()
    print("Text embeddings created successfully.")
except Exception as e:
    print(f"Error creating text embeddings: {e}")



# Verify text embeddings
verify_embeddings_sql = f"""
SELECT 
  artifact_type,
  COUNT(*) as count,
  AVG(ARRAY_LENGTH(embedding_vector)) as avg_vector_length
FROM `{project_id}.churn_prevention_demo.text_embeddings`
GROUP BY artifact_type
"""

try:
    query_job = client.query(verify_embeddings_sql)
    results = query_job.result()
    
    print("Text embeddings verification:")
    for row in results:
        print(f"- {row.artifact_type}: {row.count} embeddings, vector dimension: {row.avg_vector_length}")
except Exception as e:
    print(f"Error verifying embeddings: {e}")



# Create customer profile embeddings (we remove any kind of ground truth data)
customer_profile_embeddings_sql = f"""
CREATE OR REPLACE TABLE `{project_id}.churn_prevention_demo.customer_profile_embeddings` AS

-- Step 1: Create a temporary table with all the necessary data, including the 'content' column
WITH profile_data AS (
  SELECT
    cf.customer_id,
    -- This is the `profile_text` column that you want to keep
    CONCAT(
      'Customer profile: Plan=', cf.plan,
      ', MRR=$', CAST(ROUND(cf.mrr) AS STRING),
      ', Age=', CAST(cf.customer_age_days AS STRING), ' days',
      ', Recent sessions=', CAST(cf.sessions_30d AS STRING),
      ', Usage trend=', CASE
        WHEN cf.usage_trend_correlation > 0.1 THEN 'increasing'
        WHEN cf.usage_trend_correlation < -0.1 THEN 'decreasing'
        ELSE 'stable'
      END,
      ', Support sentiment=', CASE
        WHEN cf.avg_sentiment > 0.2 THEN 'positive'
        WHEN cf.avg_sentiment < -0.2 THEN 'negative'
        ELSE 'neutral'
      END,
      ', Churn risk=', CAST(ROUND(cp.churn_probability * 100) AS STRING), '%'
    ) AS profile_text,
    -- This is the 'content' column for embedding generation (same as profile_text)
    CONCAT(
      'Customer profile: Plan=', cf.plan,
      ', MRR=$', CAST(ROUND(cf.mrr) AS STRING),
      ', Recent sessions=', CAST(cf.sessions_30d AS STRING),
      ', Usage trend=', CASE
        WHEN cf.usage_trend_correlation > 0.1 THEN 'increasing'
        WHEN cf.usage_trend_correlation < -0.1 THEN 'decreasing'
        ELSE 'stable'
      END,
      ', Support sentiment=', CASE
        WHEN cf.avg_sentiment > 0.2 THEN 'positive'
        WHEN cf.avg_sentiment < -0.2 THEN 'negative'
        ELSE 'neutral'
      END,
      ', Churn risk=', CAST(ROUND(cp.churn_probability * 100) AS STRING), '%'
    ) AS content,
    cp.churn_probability,
    cp.prediction_type, -- Include whether this was train or test data
    COALESCE(uf.forecast_trend_slope, 0) as usage_forecast_slope
  FROM `{project_id}.churn_prevention_demo.customer_features` cf
  JOIN `{project_id}.churn_prevention_demo.churn_predictions` cp USING(customer_id)
  LEFT JOIN `{project_id}.churn_prevention_demo.usage_forecasts` uf USING(customer_id)
)

-- Step 2: Generate embeddings from the 'profile_data'
SELECT
  p.customer_id,
  p.profile_text,
  ml_generate_embedding_result AS profile_embedding, -- The generated embedding vector
  p.churn_probability,
  p.prediction_type,
  p.usage_forecast_slope
FROM
  ML.GENERATE_EMBEDDING(
    MODEL `{project_id}.churn_prevention_demo.embedding_model`,
    (SELECT * FROM profile_data) -- Pass the temporary table to the function
  ) as embeddings,
  profile_data AS p
WHERE embeddings.content = p.content;
"""

try:
    query_job = client.query(customer_profile_embeddings_sql)
    query_job.result()
    print("Customer profile embeddings created successfully.")
except Exception as e:
    print(f"Error creating customer profile embeddings: {e}")



# Verify customer profile embeddings
verify_profiles_sql = f"""
SELECT 
  prediction_type,
  COUNT(*) as total_customers,
  AVG(ARRAY_LENGTH(profile_embedding)) as embedding_dimensions,
  COUNT(DISTINCT customer_id) as unique_customers
FROM `{project_id}.churn_prevention_demo.customer_profile_embeddings`
GROUP BY prediction_type
"""

try:
    query_job = client.query(verify_profiles_sql)
    results = query_job.result()
    
    for row in results:
        print(f"Customer profile embeddings verification:")
        print(f"- Total profiles: {row.total_customers}")
        print(f"- Unique customers: {row.unique_customers}")
        print(f"- Embedding dimensions: {row.embedding_dimensions}")
except Exception as e:
    print(f"Error verifying customer profile embeddings: {e}")



# Test similarity search for a sample customer
similarity_search_sql = f"""
WITH target_customer AS (
  SELECT customer_id, profile_embedding
  FROM `{project_id}.churn_prevention_demo.customer_profile_embeddings`
  WHERE churn_probability > 0.7
  LIMIT 1
)
SELECT 
  cp.customer_id,
  cp.profile_text,
  cp.churn_probability,
  cp.prediction_type,
  -- Calculate cosine similarity (1 - cosine distance) to get most similar results
  1 - ML.DISTANCE(cp.profile_embedding, tc.profile_embedding, 'COSINE') AS similarity_score
FROM `{project_id}.churn_prevention_demo.customer_profile_embeddings` cp
CROSS JOIN target_customer tc
WHERE cp.customer_id != tc.customer_id
ORDER BY similarity_score DESC
LIMIT 5
"""

try:
    query_job = client.query(similarity_search_sql)
    results = query_job.result()
    
    print("Vector similarity search test:")
    print("Finding similar customers to a high-risk customer:")
    
    # Get the target customer ID
    first_row = next(iter(results))
    target_id = first_row.customer_id
    
    # Reset iterator
    results = query_job.result()
    
    for row in results:
        print(f"\nCustomer: {row.customer_id}")
        print(f"Profile: {row.profile_text}")
        print(f"Churn probability: {row.churn_probability:.2f}")
        print(f"Similarity score: {row.similarity_score:.4f}")
except Exception as e:
    print(f"Error testing similarity search: {e}")



# Create agent audit logs table
audit_logs_sql = f"""
CREATE TABLE IF NOT EXISTS `{project_id}.churn_prevention_demo.agent_audit_logs` (
  audit_id STRING,
  customer_id STRING,
  invoked_at TIMESTAMP,
  recommendation_json JSON,
  agent_trace_json JSON,
  execution_time_seconds FLOAT64,
  success BOOL,
  error_message STRING,
  llm_tokens_used INT64,
  tools_invoked ARRAY<STRING>,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) PARTITION BY DATE(invoked_at) CLUSTER BY customer_id;
"""

try:
    client.query(audit_logs_sql).result()
    print("Agent audit logs table created successfully.")
except Exception as e:
    print(f"Error creating agent audit logs table: {e}")



# Create outreach recommendations table
recommendations_sql = f"""
CREATE TABLE IF NOT EXISTS `{project_id}.churn_prevention_demo.outreach_recommendations` (
  recommendation_id STRING,
  customer_id STRING,
  audit_id STRING,
  churn_probability FLOAT64,
  recommended_action STRING,
  action_parameters JSON,
  confidence_score FLOAT64,
  evidence_sources ARRAY<STRING>,
  status STRING, -- PENDING, APPROVED, EXECUTED, REJECTED
  human_reviewer STRING,
  review_notes STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) PARTITION BY DATE(created_at) CLUSTER BY customer_id;
"""

try:
    client.query(recommendations_sql).result()
    print("Outreach recommendations table created successfully.")
except Exception as e:
    print(f"Error creating outreach recommendations table: {e}")



class BigQueryAgentTools:
    def __init__(self, client, project_id: str, dataset_id: str = "churn_prevention_demo"):
        self.client = client
        self.project_id = project_id
        self.dataset_id = dataset_id
        
    def _execute_query(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Execute BigQuery SQL and return results as list of dicts"""
        try:
            if params:
                job_config = bigquery.QueryJobConfig(query_parameters=params)
                job = self.client.query(query, job_config=job_config)
            else:
                job = self.client.query(query)
                
            results = job.result()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Query execution failed: {e}")
            return []
    
    def get_customer_features(self, customer_id: str) -> Dict[str, Any]:
        """Get customer features"""
        query = f"""
        SELECT 
          customer_id,
          plan,
          mrr,
          days_since_last_usage,
          sessions_7d,
          sessions_30d,
          sessions_30_60d,
          usage_ratio_30_vs_60d,
          avg_session_duration_30d,
          unique_features_30d,
          usage_trend_correlation,
          support_tickets_30d,
          avg_sentiment,
          min_sentiment,
          zero_usage_30d,
          zero_usage_7d,
          customer_age_days
        FROM `{self.project_id}.{self.dataset_id}.customer_features`
        WHERE customer_id = @customer_id
        """
        
        params = [bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)]
        results = self._execute_query(query, params)

        try:
            params = [bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)]
            results = self._execute_query(query, params)
            return results[0]
        except Exception as e:
            print(f"Error returning results: {e}\nResults:{results}")
    
    def get_churn_prediction(self, customer_id: str) -> Dict[str, Any]:
        """Get churn probability for customer"""
        query = f"""
        SELECT 
          customer_id,
          churn_probability,
          retention_probability,
          predicted_churned
        FROM `{self.project_id}.{self.dataset_id}.churn_predictions`
        WHERE customer_id = @customer_id
        """
        
        try:
            params = [bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)]
            results = self._execute_query(query, params)
            return results[0]
        except Exception as e:
            print(f"Error returning results: {e}\nResults:{results}")
    
    def get_usage_forecast(self, customer_id: str) -> Dict[str, Any]:
        """Get usage forecasting data"""
        query = f"""
        SELECT 
          customer_id,
          forecast_avg_sessions,
          forecast_trend_slope,
          forecast_steep_decline,
          forecast_low_usage,
          forecast_week1_sessions,
          forecast_week4_sessions,
          forecast_week1_uncertainty,
          forecast_week4_uncertainty
        FROM `{self.project_id}.{self.dataset_id}.usage_forecasts`
        WHERE customer_id = @customer_id
        """
        
        try:
            params = [bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)]
            results = self._execute_query(query, params)
            return results[0]
        except Exception as e:
            print(f"Error returning results: {e}\nResults:{results}")
    
    def search_similar_customers(self, customer_id: str, k: int = 5) -> Dict[str, Any]:
        """Search for similar customers using vector search"""
        search_query = f"""
        WITH target_profile AS (
          SELECT profile_embedding as target_embedding
          FROM `{self.project_id}.{self.dataset_id}.customer_profile_embeddings`
          WHERE customer_id = @customer_id
        )
        SELECT 
          pe.customer_id,
          pe.profile_text,
          pe.churn_probability,
          pe.prediction_type,  -- Whether this was from train or test dataset
          pe.usage_forecast_slope,
          1 - ML.DISTANCE(pe.profile_embedding, tp.target_embedding, 'COSINE') as similarity_score
        FROM `{self.project_id}.{self.dataset_id}.customer_profile_embeddings` pe
        CROSS JOIN target_profile tp
        WHERE pe.customer_id != @customer_id
        ORDER BY similarity_score DESC
        LIMIT @k
        """
        
        params = [
            bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id),
            bigquery.ScalarQueryParameter("k", "INT64", k)
        ]
        similar_results = self._execute_query(search_query, params)
        
        return {"similar_customers": similar_results}
    
    def search_relevant_artifacts(self, customer_id: str, query_text: str, k: int = 5) -> Dict[str, Any]:
        """Search for relevant support tickets and NPS feedback"""
        search_query = f"""
        WITH query_embedding AS (
          SELECT ml_generate_embedding_result AS query_vector
          FROM ML.GENERATE_EMBEDDING(
            MODEL `{self.project_id}.{self.dataset_id}.embedding_model`,
            (SELECT @query_text AS content),
            STRUCT(TRUE AS flatten_json_output)
          )
        )
        SELECT 
          te.artifact_id,
          te.artifact_type,
          te.text_content,
          te.sentiment,
          te.priority,
          1 - ML.DISTANCE(te.embedding_vector, qe.query_vector, 'COSINE') as relevance_score
        FROM `{self.project_id}.{self.dataset_id}.text_embeddings` te
        CROSS JOIN query_embedding qe
        WHERE te.customer_id = @customer_id
        ORDER BY relevance_score DESC
        LIMIT @k
        """
        
        params = [
            bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id),
            bigquery.ScalarQueryParameter("query_text", "STRING", query_text),
            bigquery.ScalarQueryParameter("k", "INT64", k)
        ]
        results = self._execute_query(search_query, params)
        
        return {"relevant_artifacts": results}

# Initialize the agent tools
bq_agent_tools = BigQueryAgentTools(client, project_id)

print("BigQuery Agent Tools initialized successfully.")



# Test the agent tools with a sample customer
test_customer_sql = f"""
SELECT customer_id 
FROM `{project_id}.churn_prevention_demo.churn_predictions`
WHERE churn_probability < 0.5
LIMIT 1
"""

try:
    query_job = client.query(test_customer_sql)
    results = query_job.result()
    test_customer_id = list(results)[0].customer_id
    
    print(f"Testing agent tools with customer: {test_customer_id}")
    
    # Test get_customer_features
    features = bq_agent_tools.get_customer_features(test_customer_id)
    print(f"Customer features retrieved: {len(features)} fields")
    
    # Test get_churn_prediction
    churn_data = bq_agent_tools.get_churn_prediction(test_customer_id)
    print(f"Churn prediction retrieved: {churn_data}")
    
    # Test get_usage_forecast
    forecast_data = bq_agent_tools.get_usage_forecast(test_customer_id)
    print(f"Usage forecast retrieved: {forecast_data}")
    
    # Test search_similar_customers
    similar_customers = bq_agent_tools.search_similar_customers(test_customer_id, k=3)
    print(f"Similar customers found: {len(similar_customers)}")
    
    # Test search_relevant_artifacts
    artifacts = bq_agent_tools.search_relevant_artifacts(test_customer_id, "billing issues support", k=3)
    print(f"Relevant artifacts found: {len(artifacts['relevant_artifacts'])}")
    
    print("\nAll agent tools tested successfully!")
    
except Exception as e:
    print(f"Error testing agent tools: {e}")



class BigQueryNativeChurnAgent:
    def __init__(self, client, project_id: str, dataset_id: str = "churn_prevention_demo"):
        self.bq_tools = BigQueryAgentTools(client, project_id, dataset_id)
        self.client = client
        self.project_id = project_id
        self.dataset_id = dataset_id
        
        # Use the text generation model created in A-02
        self.llm_model_path = f"`{project_id}.{dataset_id}.text_generation_model`"
    
    def _call_bigquery_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 5000) -> str:
        """Call BigQuery's ML.GENERATE_TEXT for LLM reasoning"""
        llm_query = f"""
        SELECT ml_generate_text_llm_result
        FROM ML.GENERATE_TEXT(
          MODEL {self.llm_model_path},
          (SELECT @prompt AS prompt),
          STRUCT(
            {temperature} AS temperature,
            {max_tokens} AS max_output_tokens,
            TRUE AS flatten_json_output
          )
        )
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("prompt", "STRING", prompt)
            ]
        )
        
        results = list(self.client.query(llm_query, job_config=job_config).result())
        return results[0].ml_generate_text_llm_result if results else ""
    
    def analyze_customer(self, customer_id: str) -> Dict[str, Any]:
        """Main function to analyze customer using BigQuery-native AI"""
        start_time = datetime.utcnow()
        trace_steps = []
        
        try:
            # Step 1: Get customer features
            features = self.bq_tools.get_customer_features(customer_id)
            trace_steps.append({"step": "get_customer_features", "data": features})
            
            if not features:
                return {"error": f"Customer {customer_id} not found", "trace": trace_steps}
            
            # Step 2: Get churn prediction
            churn_data = self.bq_tools.get_churn_prediction(customer_id)
            trace_steps.append({"step": "get_churn_prediction", "data": churn_data})
            
            # Step 3: Get usage forecast/trends
            forecast_data = self.bq_tools.get_usage_forecast(customer_id)
            trace_steps.append({"step": "get_usage_forecast", "data": forecast_data})
            
            # Step 4: Find similar customers for context
            similar_customers = self.bq_tools.search_similar_customers(customer_id, k=3)
            trace_steps.append({"step": "search_similar_customers", "data": similar_customers})
            
            # Step 5: Search for relevant customer artifacts
            query_text = f"customer issues plan {features.get('plan', '')} usage decline support"
            artifacts = self.bq_tools.search_relevant_artifacts(customer_id, query_text, k=3)
            trace_steps.append({"step": "search_relevant_artifacts", "data": artifacts})
            
            # Step 6: Build context for LLM
            context = {
                "customer_profile": features,
                "churn_analysis": churn_data, 
                "usage_forecast": forecast_data,
                "similar_customers": similar_customers.get("similar_customers", [])[:2],
                "relevant_history": artifacts.get("relevant_artifacts", [])[:2]
            }
            
            # Step 7: Generate recommendation using BigQuery ML.GENERATE_TEXT
            system_prompt = self._build_recommendation_prompt(context)
            
            llm_response = self._call_bigquery_llm(system_prompt)
            trace_steps.append({"step": "generate_recommendation", "prompt_length": len(system_prompt), "response": llm_response})
            
            # Step 8: Parse LLM response
            try:
                # Try to extract JSON from response
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                json_str = llm_response[json_start:json_end]
                recommendation = json.loads(json_str)
            except Exception as e:
                print(f"Error parsing recommendation: {e}")
            
            # Add metadata
            recommendation["execution_time_seconds"] = (datetime.utcnow() - start_time).total_seconds()
            recommendation["trace_steps"] = trace_steps
            recommendation["generated_via"] = "BigQuery ML.GENERATE_TEXT"
            
            # Step 9: Store audit log
            audit_id = self._store_audit_log(customer_id, recommendation, context)
            recommendation["audit_id"] = audit_id
            
            return recommendation
            
        except Exception as e:
            error_response = {
                "customer_id": customer_id,
                "error": str(e),
                "execution_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "trace_steps": trace_steps
            }
            self._store_audit_log(customer_id, error_response, {})
            return error_response
    
    def _build_recommendation_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for BigQuery LLM"""
        prompt = """You are an expert customer success AI agent for CloudFlow Pro, a project management and workflow automation SaaS platform. Analyze the provided customer data and recommend the optimal churn prevention strategy.

CUSTOMER ANALYSIS:
""" + json.dumps(context.get("customer_profile", {}), indent=2) + """

CHURN RISK ASSESSMENT:  
""" + json.dumps(context.get("churn_analysis", {}), indent=2) + """

USAGE TRENDS:
""" + json.dumps(context.get("usage_forecast", {}), indent=2) + """

SIMILAR CUSTOMER OUTCOMES:
""" + json.dumps(context.get("similar_customers", []), indent=2) + """

CUSTOMER HISTORY:
""" + json.dumps(context.get("relevant_history", []), indent=2) + """

AVAILABLE ACTIONS:
1. DISCOUNT: Offer 5-20% discount on next billing cycle
2. FREE_TRIAL: Extend trial period by 7-30 days
3. PERSONALIZED_EMAIL: Send targeted email addressing specific concerns  
4. CUSTOMER_SUCCESS_CALL: Schedule human intervention call
5. PRODUCT_DEMO: Offer personalized feature walkthrough

Based on the evidence above, provide your recommendation as valid JSON:

{
    "customer_id": "...",
    "churn_probability": 0.xx,
    "recommended_action": "DISCOUNT|FREE_TRIAL|PERSONALIZED_EMAIL|CUSTOMER_SUCCESS_CALL|PRODUCT_DEMO",
    "action_parameters": {
        "discount_percent": 15,
        "trial_days": 14,
        "email_subject": "...",
        "urgency": "high|medium|low"
    },
    "rationale": "Evidence-based explanation referencing similar customers and usage patterns...",
    "evidence_sources": ["similar_customer_X", "support_ticket_Y"],
    "confidence": 0.xx
}

Respond only with valid JSON:"""
        
        return prompt
    
    def _store_audit_log(self, customer_id: str, recommendation: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Store audit log in BigQuery with proper JSON handling"""
        audit_id = str(uuid.uuid4())
        
        # Helper function to clean data for JSON serialization
        def clean_for_json(obj):
            """Recursively clean data to ensure JSON compatibility"""
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, float):
                # Handle NaN, infinity, and precision issues
                if math.isnan(obj) or math.isinf(obj):
                    return None
                # Round to avoid precision issues that break JSON parsing
                return round(obj, 6)
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            else:
                return obj
        
        # Clean the data before serialization
        clean_recommendation = clean_for_json(recommendation)
        clean_context = clean_for_json(context)
        clean_trace_steps = clean_for_json(recommendation.get("trace_steps", []))
        
        insert_query = f"""
        INSERT INTO `{self.project_id}.{self.dataset_id}.agent_audit_logs` 
        (audit_id, customer_id, invoked_at, recommendation_json, agent_trace_json, 
         execution_time_seconds, success, tools_invoked, created_at)
        VALUES (
            @audit_id, @customer_id, @invoked_at, 
            PARSE_JSON(@recommendation), PARSE_JSON(@trace), @exec_time, @success, @tools, CURRENT_TIMESTAMP()
        )
        """
        
        # Create trace object
        trace_data = {
            "context": clean_context, 
            "steps": clean_trace_steps
        }
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("audit_id", "STRING", audit_id),
                bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id),
                bigquery.ScalarQueryParameter("invoked_at", "TIMESTAMP", datetime.utcnow()),
                bigquery.ScalarQueryParameter("recommendation", "STRING", json.dumps(clean_recommendation, ensure_ascii=False, separators=(',', ':'))),
                bigquery.ScalarQueryParameter("trace", "STRING", json.dumps(trace_data, ensure_ascii=False, separators=(',', ':'))),
                bigquery.ScalarQueryParameter("exec_time", "FLOAT64", recommendation.get("execution_time_seconds", 0)),
                bigquery.ScalarQueryParameter("success", "BOOL", "error" not in recommendation),
                bigquery.ArrayQueryParameter("tools", "STRING", ["get_customer_features", "get_churn_prediction", "search_similar_customers", "ML.GENERATE_TEXT"])
            ]
        )
        
        try:
            self.client.query(insert_query, job_config=job_config).result()
            
            # Also store the recommendation in the outreach_recommendations table for human review
            if "error" not in recommendation and "recommended_action" in recommendation:
                recommendation_id = str(uuid.uuid4())
                
                # Extract evidence sources if available
                evidence_sources = recommendation.get("evidence_sources", [])
                if isinstance(evidence_sources, list):
                    evidence_sources_array = evidence_sources
                else:
                    # Handle case where evidence_sources might not be a list
                    evidence_sources_array = [str(evidence_sources)] if evidence_sources else []
                
                outreach_query = f"""
                INSERT INTO `{self.project_id}.{self.dataset_id}.outreach_recommendations` 
                (recommendation_id, customer_id, audit_id, churn_probability, recommended_action, 
                 action_parameters, confidence_score, evidence_sources, status, created_at)
                VALUES (
                    @recommendation_id, @customer_id, @audit_id, @churn_probability,
                    @action, PARSE_JSON(@parameters), @confidence, @evidence, 
                    'PENDING', CURRENT_TIMESTAMP()
                )
                """
                
                outreach_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("recommendation_id", "STRING", recommendation_id),
                        bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id),
                        bigquery.ScalarQueryParameter("audit_id", "STRING", audit_id),
                        bigquery.ScalarQueryParameter("churn_probability", "FLOAT64", recommendation.get("churn_probability", 0.0)),
                        bigquery.ScalarQueryParameter("action", "STRING", recommendation.get("recommended_action", "UNKNOWN")),
                        bigquery.ScalarQueryParameter("parameters", "STRING", json.dumps(recommendation.get("action_parameters", {}), ensure_ascii=False, separators=(',', ':'))),
                        bigquery.ScalarQueryParameter("confidence", "FLOAT64", recommendation.get("confidence", 0.0)),
                        bigquery.ArrayQueryParameter("evidence", "STRING", evidence_sources_array)
                    ]
                )
                
                try:
                    self.client.query(outreach_query, job_config=outreach_config).result()
                    print(f"Stored recommendation in outreach_recommendations table with ID: {recommendation_id}")
                except Exception as e:
                    print(f"Error storing outreach recommendation: {e}")
            
            return audit_id
        except Exception as e:
            print(f"Error storing audit log: {e}")
            # Fallback: try without JSON parsing (store as strings)
            fallback_query = f"""
            INSERT INTO `{self.project_id}.{self.dataset_id}.agent_audit_logs` 
            (audit_id, customer_id, invoked_at, recommendation_json, agent_trace_json, 
             execution_time_seconds, success, tools_invoked, created_at)
            VALUES (
                @audit_id, @customer_id, @invoked_at, 
                @recommendation_str, @trace_str, @exec_time, @success, @tools, CURRENT_TIMESTAMP()
            )
            """
            
            fallback_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("audit_id", "STRING", audit_id),
                    bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id),
                    bigquery.ScalarQueryParameter("invoked_at", "TIMESTAMP", datetime.utcnow()),
                    bigquery.ScalarQueryParameter("recommendation_str", "STRING", json.dumps(clean_recommendation)),
                    bigquery.ScalarQueryParameter("trace_str", "STRING", json.dumps(trace_data)),
                    bigquery.ScalarQueryParameter("exec_time", "FLOAT64", recommendation.get("execution_time_seconds", 0)),
                    bigquery.ScalarQueryParameter("success", "BOOL", "error" not in recommendation),
                    bigquery.ArrayQueryParameter("tools", "STRING", ["get_customer_features", "get_churn_prediction", "search_similar_customers", "ML.GENERATE_TEXT"])
                ]
            )
            
            self.client.query(fallback_query, job_config=fallback_config).result()
            return audit_id

# Initialize the BigQuery native agent
churn_agent = BigQueryNativeChurnAgent(client, project_id)

print("BigQuery Native Churn Agent initialized successfully.")



# Demo the agent with high-risk customers
demo_customers_sql = f"""
SELECT customer_id, churn_probability
FROM `{project_id}.churn_prevention_demo.churn_predictions`
WHERE churn_probability > 0.7
ORDER BY churn_probability DESC
LIMIT 3
"""

try:
    query_job = client.query(demo_customers_sql)
    results = query_job.result()
    demo_customers = [{"id": row.customer_id, "risk": row.churn_probability} for row in results]
    
    print(f"Running BigQuery AI Agent Demo on {len(demo_customers)} high-risk customers:")
    print("\n\n===============\n\n")
    
    agent_results = []
    for customer in demo_customers:
        customer_id = customer["id"]
        print(f"\nAnalyzing Customer: {customer_id} (Risk: {customer['risk']:.1%})")
        
        # Run agent analysis
        result = churn_agent.analyze_customer(customer_id)
        agent_results.append(result)
        
        if "error" not in result:
            print(f"Recommendation: {result.get('recommended_action', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.1%}")
            print(f"Execution Time: {result.get('execution_time_seconds', 0):.2f}s")
            print(f"Generated via: {result.get('generated_via', 'Unknown')}")
            
            # Show rationale
            rationale = result.get('rationale', '')
            if rationale:
                print(f"Rationale: {rationale}")
            
            # Show action parameters
            params = result.get('action_parameters', {})
            if params:
                print(f"Action Details: {params}")
                
        else:
            print(f"Analysis failed: {result.get('error', 'Unknown error')}")
            print(result)
        
        print("\n\n===============\n\n")
    
    print(f"\nAnalyzed {len(demo_customers)} customers successfully.")
    print(f"All results logged to audit table with comprehensive traces.")
    
except Exception as e:
    print(f"Error running agent demo: {e}")



# Verify audit logs were created
verify_audit_sql = f"""
SELECT 
  audit_id,
  customer_id,
  invoked_at,
  JSON_EXTRACT_SCALAR(recommendation_json, '$.recommended_action') as recommended_action,
  JSON_EXTRACT_SCALAR(recommendation_json, '$.confidence') as confidence,
  execution_time_seconds,
  success
FROM `{project_id}.churn_prevention_demo.agent_audit_logs`
ORDER BY invoked_at DESC
LIMIT 5
"""

try:
    query_job = client.query(verify_audit_sql)
    results = query_job.result()
    
    print("Recent Agent Audit Logs:")
    df = pd.DataFrame([dict(row) for row in results])
    print(df.to_string(index=False))
    
except Exception as e:
    print(f"Error verifying audit logs: {e}")

# Verify outreach recommendations were created
verify_outreach_sql = f"""
SELECT 
  recommendation_id,
  customer_id,
  audit_id,
  churn_probability,
  recommended_action,
  confidence_score,
  status,
  created_at
FROM `{project_id}.churn_prevention_demo.outreach_recommendations`
ORDER BY created_at DESC
LIMIT 5
"""

try:
    query_job = client.query(verify_outreach_sql)
    results = query_job.result()
    
    print("\nRecent Agent Outreach Recommendations:")
    df = pd.DataFrame([dict(row) for row in results])
    if df.empty:
        print("No outreach recommendations found.")
    else:
        print(df.to_string(index=False))
    
except Exception as e:
    print(f"Error verifying outreach recommendations: {e}")



plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12, 6)

print("BigQuery AI Churn Prevention System Demo")
print("="*70)
print("Demonstrating the core capabilities of our BigQuery-powered churn prevention system:")
print("1. Train/Test Split Model Performance Analysis")
print("2. Vector Search with ML.GENERATE_EMBEDDING")
print("3. Agent Recommendations with ML.GENERATE_TEXT")
print("4. Outreach Workflow Integration")
print("="*70)



print("CHURN MODEL PERFORMANCE ANALYSIS")
print("="*50)

# Get model evaluation metrics for train vs test comparison
train_test_metrics_sql = f"""
-- Get evaluation metrics on training data
WITH train_eval AS (
  SELECT 
    'TRAIN' as dataset_type,
    *
  FROM ML.CONFUSION_MATRIX(
    MODEL `{project_id}.churn_prevention_demo.churn_model`,
    (
      SELECT 
        -- Customer basic features
        cf.sessions_7d,
        cf.sessions_30d,
        cf.sessions_30_60d,
        cf.usage_ratio_30_vs_60d,
        cf.days_since_last_usage,
        cf.customer_age_days,
        cf.avg_session_duration_30d,
        cf.unique_features_30d,
        cf.usage_trend_correlation,
        cf.support_tickets_30d,
        cf.avg_sentiment,
        cf.min_sentiment,
        LOG(cf.mrr + 1) AS log_mrr,
        
        -- Categorical features
        cf.plan,
        cf.zero_usage_7d,
        cf.zero_usage_30d,
        
        -- Forecast features (with proper null handling)
        uf.forecast_week1_sessions,
        uf.forecast_week4_sessions,
        COALESCE(uf.forecast_avg_sessions, 0) AS forecast_avg_sessions,
        COALESCE(uf.forecast_trend_slope, 0) AS forecast_trend_slope,
        COALESCE(uf.forecast_steep_decline, FALSE) AS forecast_steep_decline,
        COALESCE(uf.forecast_low_usage, FALSE) AS forecast_low_usage,
        COALESCE(uf.forecast_week1_uncertainty, 0) AS forecast_week1_uncertainty,
        COALESCE(uf.forecast_week4_uncertainty, 0) AS forecast_week4_uncertainty,
        COALESCE(uf.forecast_volatility, 0) AS forecast_volatility,
        COALESCE(uf.forecast_confidence, 0) AS forecast_confidence,
        
        -- Derived features
        SAFE_DIVIDE(cf.support_tickets_30d, cf.customer_age_days) as ticket_rate,
        SAFE_DIVIDE(cf.support_tickets_30d, cf.sessions_30d) as support_per_session_ratio,
        
        -- Target variable
        cl.churned
      FROM `{project_id}.churn_prevention_demo.customer_features` cf
      JOIN `{project_id}.churn_prevention_demo.churn_labels_generated` cl USING(customer_id)
      LEFT JOIN `{project_id}.churn_prevention_demo.usage_forecasts` uf USING(customer_id)
      JOIN `{project_id}.churn_prevention_demo.model_train_test_split` split USING(customer_id)
      WHERE split.data_split = 'TRAIN'
    )
  )
),
-- Get evaluation metrics on test data
test_eval AS (
  SELECT 
    'TEST' as dataset_type,
    *
  FROM ML.CONFUSION_MATRIX(
    MODEL `{project_id}.churn_prevention_demo.churn_model`,
    (
      SELECT 
        -- Customer basic features
        cf.sessions_7d,
        cf.sessions_30d,
        cf.sessions_30_60d,
        cf.usage_ratio_30_vs_60d,
        cf.days_since_last_usage,
        cf.customer_age_days,
        cf.avg_session_duration_30d,
        cf.unique_features_30d,
        cf.usage_trend_correlation,
        cf.support_tickets_30d,
        cf.avg_sentiment,
        cf.min_sentiment,
        LOG(cf.mrr + 1) AS log_mrr,
        
        -- Categorical features
        cf.plan,
        cf.zero_usage_7d,
        cf.zero_usage_30d,
        
        -- Forecast features (with proper null handling)
        uf.forecast_week1_sessions,
        uf.forecast_week4_sessions,
        COALESCE(uf.forecast_avg_sessions, 0) AS forecast_avg_sessions,
        COALESCE(uf.forecast_trend_slope, 0) AS forecast_trend_slope,
        COALESCE(uf.forecast_steep_decline, FALSE) AS forecast_steep_decline,
        COALESCE(uf.forecast_low_usage, FALSE) AS forecast_low_usage,
        COALESCE(uf.forecast_week1_uncertainty, 0) AS forecast_week1_uncertainty,
        COALESCE(uf.forecast_week4_uncertainty, 0) AS forecast_week4_uncertainty,
        COALESCE(uf.forecast_volatility, 0) AS forecast_volatility,
        COALESCE(uf.forecast_confidence, 0) AS forecast_confidence,
        
        -- Derived features
        SAFE_DIVIDE(cf.support_tickets_30d, cf.customer_age_days) as ticket_rate,
        SAFE_DIVIDE(cf.support_tickets_30d, cf.sessions_30d) as support_per_session_ratio,
        
        -- Target variable
        cl.churned
      FROM `{project_id}.churn_prevention_demo.customer_features` cf
      JOIN `{project_id}.churn_prevention_demo.churn_labels_generated` cl USING(customer_id)
      LEFT JOIN `{project_id}.churn_prevention_demo.usage_forecasts` uf USING(customer_id)
      JOIN `{project_id}.churn_prevention_demo.model_train_test_split` split USING(customer_id)
      WHERE split.data_split = 'TEST'
    )
  )
)
-- Combine results
SELECT * FROM train_eval
UNION ALL
SELECT * FROM test_eval
"""

metrics_job = client.query(train_test_metrics_sql)
metrics_results = metrics_job.result()
metrics_df = pd.DataFrame([dict(row) for row in metrics_results])

print("Raw confusion matrix data:")
print(metrics_df)

# Calculate metrics from confusion matrix
def calculate_metrics_from_confusion_matrix(df):
    results = []
    
    for dataset_type in df['dataset_type'].unique():
        subset = df[df['dataset_type'] == dataset_type]
        
        # Extract confusion matrix values
        # True Positives: expected_label=True, predicted=True
        tp = subset[(subset['expected_label'] == True)]['TRUE'].iloc[0] if len(subset[subset['expected_label'] == True]) > 0 else 0
        
        # False Positives: expected_label=False, predicted=True  
        fp = subset[(subset['expected_label'] == False)]['TRUE'].iloc[0] if len(subset[subset['expected_label'] == False]) > 0 else 0
        
        # False Negatives: expected_label=True, predicted=False
        fn = subset[(subset['expected_label'] == True)]['FALSE'].iloc[0] if len(subset[subset['expected_label'] == True]) > 0 else 0
        
        # True Negatives: expected_label=False, predicted=False
        tn = subset[(subset['expected_label'] == False)]['FALSE'].iloc[0] if len(subset[subset['expected_label'] == False]) > 0 else 0
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'dataset_type': dataset_type,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })
    
    return pd.DataFrame(results)

# Calculate the metrics
metrics_summary = calculate_metrics_from_confusion_matrix(metrics_df)

print("\nModel Performance Comparison (Train vs Test):")
print(metrics_summary[['dataset_type', 'precision', 'recall', 'accuracy', 'f1_score']].to_string(index=False))

# Get feature importance for model interpretability
feature_importance_query = f"""
SELECT feature, importance_weight
FROM ML.FEATURE_IMPORTANCE(MODEL `{project_id}.churn_prevention_demo.churn_model`)
ORDER BY importance_weight DESC
LIMIT 10
"""

feature_job = client.query(feature_importance_query)
feature_results = feature_job.result()
importance_df = pd.DataFrame([dict(row) for row in feature_results])

# Visualize train vs test performance and feature importance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Model metrics comparison
metrics = ['precision', 'recall', 'accuracy', 'f1_score']
x = np.arange(len(metrics))
width = 0.35

train_values = metrics_summary[metrics_summary['dataset_type'] == 'TRAIN'][metrics].values[0]
test_values = metrics_summary[metrics_summary['dataset_type'] == 'TEST'][metrics].values[0]

ax1.bar(x - width/2, train_values, width, label='Training Data')
ax1.bar(x + width/2, test_values, width, label='Test Data')
ax1.set_title('Model Performance: Train vs Test')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.set_ylim([0, 1])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Feature importance
ax2.barh(importance_df['feature'][::-1], importance_df['importance_weight'][::-1])
ax2.set_title('Top 10 Feature Importance')
ax2.set_xlabel('Importance Weight')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


print("VECTOR SEARCH CAPABILITIES")
print("="*50)

# Get a test customer to demonstrate with
demo_customer_query = f"""
SELECT customer_id, churn_probability, profile_text, prediction_type
FROM `{project_id}.churn_prevention_demo.customer_profile_embeddings`
WHERE prediction_type = 'TEST'  -- Focus on test data
ORDER BY churn_probability DESC
LIMIT 1
"""

demo_result = client.query(demo_customer_query).result()
demo_customer = list(demo_result)[0]
target_id = demo_customer.customer_id
target_risk = demo_customer.churn_probability
target_profile = demo_customer.profile_text

print(f"Target Test Customer: {target_id}")
print(f"Churn Risk: {target_risk:.1%}")
print(f"Profile: {target_profile}")
print(f"Prediction Type: {demo_customer.prediction_type}")

# Find similar customers using vector search
vector_search_query = f"""
WITH target_profile AS (
  SELECT profile_embedding as target_embedding
  FROM `{project_id}.churn_prevention_demo.customer_profile_embeddings`
  WHERE customer_id = '{target_id}'
)
SELECT 
  pe.customer_id,
  pe.profile_text,
  pe.churn_probability,
  pe.prediction_type,  -- Show whether this is train or test data
  1 - ML.DISTANCE(pe.profile_embedding, tp.target_embedding, 'COSINE') as similarity_score
FROM `{project_id}.churn_prevention_demo.customer_profile_embeddings` pe
CROSS JOIN target_profile tp
WHERE pe.customer_id != '{target_id}'
ORDER BY similarity_score DESC
LIMIT 5
"""

similar_results = client.query(vector_search_query).result()
similar_df = pd.DataFrame([dict(row) for row in similar_results])

print("\nTop 5 Most Similar Customers:")
print(similar_df[['customer_id', 'churn_probability', 'prediction_type', 'similarity_score']].to_string(index=False))

# Visualize similarity by prediction type
plt.figure(figsize=(12, 6))
colors = ['blue' if pt == 'TRAIN' else 'red' for pt in similar_df['prediction_type']]
plt.bar(range(len(similar_df)), similar_df['similarity_score'], color=colors)
plt.title(f'Customer Similarity Scores to {target_id}')
plt.xlabel('Similar Customers (Blue=Train Data, Red=Test Data)')
plt.ylabel('Cosine Similarity Score')
plt.xticks(range(len(similar_df)), similar_df['customer_id'], rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



print("AI AGENT RECOMMENDATION GENERATION")
print("="*50)

# Run the agent on our target customer
print(f"Executing BigQuery AI Agent on test customer {target_id}...")
start_time = datetime.utcnow()
agent_result = churn_agent.analyze_customer(target_id)
execution_time = (datetime.utcnow() - start_time).total_seconds()

if "error" not in agent_result:
    # Display core recommendation
    print(f"\nGenerated Recommendation:")
    print(f"  Customer: {agent_result.get('customer_id')}")
    print(f"  Churn Probability: {agent_result.get('churn_probability', 0):.2f}")
    print(f"  Recommended Action: {agent_result.get('recommended_action', 'Unknown')}")
    print(f"  Confidence Score: {agent_result.get('confidence', 0):.2f}")
    print(f"  Execution Time: {execution_time:.2f}s")
    
    # Action details
    if 'action_parameters' in agent_result:
        print(f"\nAction Parameters:")
        for k, v in agent_result.get('action_parameters', {}).items():
            print(f"  - {k.replace('_', ' ').title()}: {v}")
    
    # Show rationale
    if 'rationale' in agent_result:
        print(f"\nRationale:")
        print(agent_result.get('rationale'))
else:
    print(f"Error: {agent_result.get('error', 'Unknown error')}")



print("OUTREACH WORKFLOW INTEGRATION")
print("="*50)

# Get the pending recommendations
outreach_query = f"""
SELECT 
  recommendation_id,
  customer_id,
  churn_probability,
  recommended_action,
  confidence_score,
  status,
  created_at
FROM `{project_id}.churn_prevention_demo.outreach_recommendations`
ORDER BY created_at DESC
LIMIT 5
"""

outreach_job = client.query(outreach_query)
outreach_results = outreach_job.result()
outreach_df = pd.DataFrame([dict(row) for row in outreach_results])

if outreach_df.empty:
    print("No outreach recommendations found in the system.")
else:
    print("Recent Outreach Recommendations:")
    print(outreach_df.to_string(index=False))
    
    # Visualize recommendation distribution
    if len(outreach_df) > 1:
        plt.figure(figsize=(12, 6))
        recommendation_counts = outreach_df['recommended_action'].value_counts()
        plt.bar(recommendation_counts.index, recommendation_counts.values)
        plt.title('Distribution of Recommended Actions')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()



# Define the survey response
survey_response = """**Team Survey: BigQuery AI - Building the Future of Data**

**Instructions:**

*   This survey is for bonus points.
*   Points are awarded for completeness, not for the content of your answers.
*   We highly encourage everyone to submit one.
*   There are 3 questions in total - please answer all 3.
---

**Team Member Experience:**

1)  **BigQuery AI:** Please list each team member(s) months of experience with BigQuery AI.
    *   Marcos Tidball: 1 month

2)  **Google Cloud:** Please list each team member(s) months of experience with Google Cloud.
    *   Marcos Tidball: 2 months

---

3)  **Feedback:**

We'd love to hear from you and your experience in working with the technology during this hackathon, positive or negative. Please provide any feedback on your experience with BigQuery AI.

I'm primarily a Python and Spark user when it comes to manipulating data, training & evaluating models and orchestrating different systems, so I was really surprised discovering that BigQuery has different built-in AI  functionalities! My only experience BigQuery with it prior to this competition involved using it to transform raw tables into processed data that could be consumed in Python, i.e. basically standard SQL for data processing and analysis.

Here's some positive feedback:
- **Unified Data Processing**: The biggest strength is having everything in one place. Coming from a pandas/scikit-learn/transformers/langchain/etc. background where you have to maintain multiple dependencies, the ability to do feature engineering, ML training, vector embeddings, and LLM inference all within BigQuery is really impressive. It's nice to be able to not worry about imports!

- **No Infrastructure Overhead**: Compared to setting up Jupyter environments, managing Python dependencies, or configuring HuggingFace models, the "just write SQL" approach is refreshing. No pip installs, no version conflicts, no GPU/API management.

- **Easy Cloud Setup**: Even though I'm kinda new to Google Cloud, it was really easy to set up everything necessary to get the project working. Also, especially when it comes to LLMs, it's nice to have everything centralized in just one cloud instead of having to juggle between different APIs.

- **Speed of Prototyping**: Once I got past the SQL learning curve, iterating on the churn model was fast. It felt really good to be able to start the notebook, initialize the BigQuery client and get back into the action without needing to wait a long time for data to be processed.

- **LLM Integration**: The ML.GENERATE_TEXT function is really powerful and easy to use. Coming from LangChain (which I is very verbose and convoluted in my opinion), the direct SQL interface to generate text is much cleaner. It's also great to not have to worry about having enough GPUs or exorbitant API costs!

Here's some friction points:
- **SQL-First Paradigm Shift**: This is the biggest adjustment. As someone coming from Python, using different ML models took some time to get used to. I'm still not quite sure if I had the best approach when it comes to performing train/test splits, for example. It felt a bit less natural for me than doing the same thing in the standard DS Python environment. With that said, I think it's just a matter of getting used to it.

- **Debugging Complexity**: When a complex SQL query fails, debugging it is harder than stepping through Python code. The error messages aren't always clear, and you can't inspect intermediate results as easily. It took me some time especially to understand how to get the LLMs to output text the way that I wanted them to.

- **Lack of Algorithm Flexibility**: This it to be expected, but I can see some situations where certain specific models not available in BigQuery would be necessary, especially when dealing with more advanced ML stuff. With that said, I think that the current models are enough for a vast majority of situations!
"""


import os
output_file_path = '/kaggle/working/survey_answer.txt'
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(survey_response)

