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


import requests
try:
    response = requests.get("https://www.google.com", timeout=5)
    print("Internet connection working")
except:
    print("No internet connection - enable in Settings")


from google.cloud import bigquery
import os

# Set your project ID
PROJECT_ID = 'customeranalytics-472107'

# Initialize BigQuery client
client = bigquery.Client(project=PROJECT_ID)

print(f"Connected to project: {PROJECT_ID}\n{client}")


query = """
-- 1. Check total users in each CTE
WITH user_info AS (
  SELECT id AS user_id, first_name, last_name, email, created_at AS user_created_at
  FROM `customeranalytics-472107.Ecommerce.users`
),

purchase_stats AS (
  SELECT
    user_id,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(sale_price) AS total_spend,
    AVG(sale_price) AS avg_order_value,
    -- Use the most recent activity date across all statuses
    GREATEST(
      COALESCE(MAX(delivered_at), '1900-01-01'),
      COALESCE(MAX(shipped_at), '1900-01-01'),
      COALESCE(MAX(returned_at), '1900-01-01')
    ) AS last_purchase_date
  FROM `customeranalytics-472107.Ecommerce.order_items`
  WHERE 
    status IN ('Complete', 'Shipped', 'Returned')
    AND user_id IN (
      -- Only include customers who have at least 1 Complete order
      SELECT DISTINCT user_id 
      FROM `customeranalytics-472107.Ecommerce.order_items`
      WHERE status = 'Complete'
    )
  GROUP BY user_id
),

engagement_stats AS (
  SELECT
    user_id,
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNT(*) AS total_events,
    MAX(created_at) AS last_engagement_date
  FROM `customeranalytics-472107.Ecommerce.events`
  GROUP BY user_id
),

-- Add data validation step
data_check AS (
  SELECT
    u.user_id,
    u.first_name,
    u.last_name,
    u.email,
    u.user_created_at,
    COALESCE(ps.total_orders, 0) AS total_orders,
    COALESCE(ps.total_spend, 0) AS total_spend,
    ps.last_purchase_date,
    COALESCE(es.total_sessions, 0) AS total_sessions,
    COALESCE(es.total_events, 0) AS total_events,
    es.last_engagement_date
  FROM user_info u
  LEFT JOIN purchase_stats ps ON u.user_id = ps.user_id
  LEFT JOIN engagement_stats es ON u.user_id = es.user_id
),

scored_customers AS (
  SELECT
    *,
    -- Days calculations
    CASE 
      WHEN last_purchase_date IS NULL THEN 999
      ELSE DATE_DIFF(CURRENT_DATE(), DATE(last_purchase_date), DAY)
    END AS days_since_last_purchase,
    
    CASE 
      WHEN last_engagement_date IS NULL THEN 999
      ELSE DATE_DIFF(CURRENT_DATE(), DATE(last_engagement_date), DAY)
    END AS days_since_last_engagement,
    
    DATE_DIFF(CURRENT_DATE(), DATE(user_created_at), DAY) AS customer_since_days,

    -- NTILE scoring (1-5 scale) - higher numbers = better performance
    CASE 
      WHEN total_orders = 0 THEN 1
      ELSE NTILE(5) OVER (ORDER BY total_orders ASC)
    END AS freq_score,
    
    CASE 
      WHEN total_spend = 0 THEN 1
      ELSE NTILE(5) OVER (ORDER BY total_spend ASC)
    END AS monetary_score,
    
    CASE 
      WHEN last_purchase_date IS NULL THEN 1
      ELSE NTILE(5) OVER (ORDER BY last_purchase_date DESC)  -- DESC because recent is better
    END AS recency_score,
    
    CASE 
      WHEN total_sessions = 0 THEN 1
      ELSE NTILE(5) OVER (ORDER BY total_sessions ASC)
    END AS engagement_score,
    
    NTILE(5) OVER (ORDER BY user_created_at ASC) AS lifecycle_score  -- Older customers get higher scores

  FROM data_check
),

bucketed_customers AS (
  SELECT
    *,
    (freq_score + monetary_score + recency_score + engagement_score + lifecycle_score) AS total_score,
    
    -- **IMPROVED BALANCED BOUNDARIES** based on 5-25 point scale
    CASE
      WHEN (freq_score + monetary_score + recency_score + engagement_score + lifecycle_score) >= 21 THEN 'Champions'          -- Top 10-15%
      WHEN (freq_score + monetary_score + recency_score + engagement_score + lifecycle_score) >= 18 THEN 'Loyal'             -- Next 15-20%
      WHEN (freq_score + monetary_score + recency_score + engagement_score + lifecycle_score) >= 14 THEN 'Potential Loyalists' -- Middle 30-35%
      WHEN (freq_score + monetary_score + recency_score + engagement_score + lifecycle_score) >= 10 THEN 'At-Risk'           -- Next 20-25%
      ELSE 'Hibernating'                                                                                                        -- Bottom 15-20%
    END AS customer_segment
  FROM scored_customers
)

SELECT 
  user_id,
  first_name,
  last_name,
  email,
  total_orders,
  total_spend,
  days_since_last_purchase,
  total_sessions,
  days_since_last_engagement,
  customer_since_days,
  total_score,
  customer_segment,
  ml_generate_text_llm_result AS marketing_message
FROM ML.GENERATE_TEXT(
  MODEL `customeranalytics-472107.Ecommerce.gemini_model`,
  (
    SELECT 
      user_id,
      first_name,
      last_name,
      email,
      total_orders,
      total_spend,
      days_since_last_purchase,
      total_sessions,
      days_since_last_engagement,
      customer_since_days,
      total_score,
      customer_segment,
      CONCAT(
        "Write a short, friendly marketing message for a customer named ",
        first_name, " ",
        last_name, ". ",
        "They are in the '", customer_segment, "' segment. ",
        "Total orders: ", CAST(total_orders AS STRING), 
        ", Total spend: $", CAST(total_spend AS STRING), 
        ", Last purchase was ", CAST(days_since_last_purchase AS STRING), " days ago. ",
        "Make the tone engaging and motivating, tell them we are missing you in the starting, greeting from Ecommerce.com. End with a call-to-action."
      ) AS prompt
    FROM bucketed_customers
    WHERE user_id IN (96438, 11983, 52446, 71061, 96190)
  ),
  STRUCT(
    150 AS max_output_tokens,
    0.8 AS temperature,
    TRUE AS flatten_json_output
  )
);

"""
print(f"Query read...")


# Clean and parse the JSON
from kaggle_secrets import UserSecretsClient
from google.cloud import bigquery
from google.oauth2 import service_account
import json

user_secrets = UserSecretsClient()
raw_secret = user_secrets.get_secret("gcp-service-account")

# Clean the JSON string
cleaned_json = raw_secret.strip()  # Remove whitespace
cleaned_json = cleaned_json.replace('\ufeff', '')  # Remove BOM if present
cleaned_json = cleaned_json.replace('\r\n', '\n')  # Normalize line endings

try:
    # Parse cleaned JSON
    service_account_info = json.loads(cleaned_json)
    print("JSON parsed successfully!")
    
    # Create credentials
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = bigquery.Client(credentials=credentials, project='customeranalytics-472107')
    
    # Test connection
    test_result = client.query("SELECT 'Success!' as message").to_dataframe()
    print(f"BigQuery connection working: {test_result.iloc[0]['message']}")
    
except json.JSONDecodeError as e:
    print(f"JSON still invalid: {e}")



# Run query and display results
try:
    df_results = client.query(query).to_dataframe()
    print(f"Analysis complete! Generated {len(df_results)} personalized marketing messages")
    
    # Display results
    for idx, row in df_results.iterrows():
        print(f"\nðŸ“§ {row['customer_segment']} Customer: {row['first_name']} {row['last_name']}")
        print(f"   ðŸ“Š Score: {row['total_score']} | Orders: {row['total_orders']} | Spend: ${row['total_spend']:.2f}")
        print(f"   ðŸ’Œ Message:\n {row['marketing_message']}")
        print("=" * 80)
    
    # Save results
    df_results.to_csv('customer_segmentation_with_ai_messages.csv', index=False)
    print(f"\nResults saved to CSV file!")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure vertexCon connection exists")  
    print("2. Verify gemini_model is created")
    print("3. Check Vertex AI permissions")

