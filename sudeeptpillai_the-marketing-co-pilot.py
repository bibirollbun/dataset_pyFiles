# Import necessary libraries
import pandas as pd
import google.generativeai as genai
import getpass
import json # Important: for parsing the secret string
from kaggle_secrets import UserSecretsClient
from google.oauth2 import service_account
from google.cloud import bigquery

# --- CONFIGURATION (GEMINI API KEY) ---
# This prompts any user running the notebook for their personal Gemini API key.
try:
    GEMINI_API_KEY = getpass.getpass('Please enter your Google AI Studio API Key: ')
    genai.configure(api_key=GEMINI_API_KEY)
    print("âœ… Gemini client configured.")
except Exception as e:
    print(f"â�Œ Could not configure the Gemini API key. Error: {e}")


# --- BIGQUERY AUTHENTICATION (SERVICE ACCOUNT VIA KAGGLE SECRETS) ---
GCP_PROJECT_ID = "kaggle-bigquery-challenge" # Your GCP Project ID

try:
    # Initialize the Kaggle secrets client
    user_secrets = UserSecretsClient()
    
    # Get the JSON key stored as a string from Kaggle Secrets
    sa_key_string = user_secrets.get_secret("GCP_SA_KEY")
    
    # Parse the string into a JSON object (a Python dictionary)
    sa_info = json.loads(sa_key_string)

    # Create credentials from the service account info
    credentials = service_account.Credentials.from_service_account_info(sa_info)

    # Configure the BigQuery client to use these credentials
    bq_client = bigquery.Client(project=GCP_PROJECT_ID, credentials=credentials)
    
    print("âœ… BigQuery Client configured successfully using Service Account.")

except Exception as e:
    print(f"â�Œ Could not configure BigQuery client. Have you created the 'GCP_SA_KEY' secret in Kaggle? Error: {e}")


!pip install pillow
from PIL import Image
import requests
from io import BytesIO


import pandas as pd
import google.generativeai as genai
from google.colab import userdata
from google.cloud import bigquery

# --- CONFIGURATION ---
from google.cloud import bigquery
bigquery_client = bigquery.Client(project='kaggle-bigquery-challenge')  # <-- IMPORTANT: Your GCP Project ID
# -------------------

# Configure the Gemini API client
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")

# Configure the BigQuery client

# print("Clients configured successfully.")
# for m in genai.list_models():
#     print(m.name, "-", m.display_name)



def analyze_image_from_url(image_url):
    """Takes an image URL and uses Gemini Vision to describe it."""
    try:
        response = requests.get(image_url)
        # Make sure the request was successful
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))

        vision_model = genai.GenerativeModel("gemini-2.5-pro")

        print("AI is analyzing the product image...")
        response = vision_model.generate_content(["Describe this product image from a marketing perspective.", img])
        return f"\n--- Image Analysis ---\n{response.text}"
    except Exception as e:
        return f"\n--- Image Analysis ---\nCould not analyze image. Error: {e}"


# A more generic function to run ANY valid SQL query
def execute_bq_query(sql_query):
  """Executes a SQL query against BigQuery and returns a DataFrame."""
  print("Executing AI-generated SQL...")
  try:
    df = bigquery_client.query(sql_query).to_dataframe()
    return df
  except Exception as e:
    print(f"Error executing SQL: {e}")
    return None

# The new core function that turns natural language into SQL
def generate_sql_from_prompt(user_prompt):
    """Uses Gemini to convert a user's question into a BigQuery SQL query."""

    system_prompt = f"""
    You are a GoogleSQL expert. Your task is to write a single, valid BigQuery SQL query based on a user's request.
    You must follow these rules:
    0. THE TABLE NAME IS CASE-SENSITIVE
    1. You have access to the following tables:
      - `kaggle-bigquery-challenge.Sales_transaction.online_transaction` (alias t) has columns: TransactionNo(STRING), Date(DATE), ProductNo(STRING), ProductName(STRING), Price(FLOAT), Quantity(INTEGER), CustomerNo(STRING), Country(STRING).
      - `kaggle-bigquery-challenge.Sales_transaction.product_review` (alias r) has columns: ProductNo(STRING), Review(STRING).
      - `kaggle-bigquery-challenge.Sales_transaction.product_image` (alias i) has columns: ProductName(STRING), ImageUrl(STRING).
    2. You must write a single query. Do not write multiple queries.
    3. Use `SELECT DISTINCT` to avoid duplicate rows in the output **Whenever its is necessary ONLY**.
    4. You must correctly JOIN tables using the `ProductNo` column when a user's question requires information from multiple tables.
    5. You must wrap any string values in the WHERE clause in single quotes (e.g., `WHERE t.ProductName = 'RED RETROSPOT MUG'`).
    6 Your response must be ONLY the SQL query, with no explanation, comments, or markdown.

    **IMPORTANT RULE:** To use  model with `ML.PREDICT`, the input data MUST have the exact same features and transformations used during training.
    The model was trained on `CustomerNo` and a column named `features` which was created with the transformation `ml.TF_IDF(SPLIT(purchase_history, ', ')) OVER()`.

    Therefore, to find the cluster for customer '12345', the query MUST follow this exact structure:
    ```sql
    SELECT
      CENTROID_ID
    FROM
      ML.PREDICT(MODEL `kaggle-bigquery-challenge.Sales_transaction.customer_segment_model`,
        (
          SELECT
            CustomerNo,
            ml.TF_IDF(SPLIT(purchase_history, ', ')) OVER() AS features
          FROM
            `kaggle-bigquery-challenge.Sales_transaction.vw_customer_purchase_history`
          WHERE CustomerNo = '12345'
        )
      )
    
    ```
   **IMPORTANT RULE 2: Answering Questions About Clusters**
    To answer questions about customers within a specific cluster, you must first get ALL predictions for ALL customers, and then JOIN those results with other tables. Use a Common Table Expression (WITH clause) for this.
    
    **EXAMPLE of a complex query to find the top country for customers in cluster 2:**
    ```sql
    WWITH Predictions AS (
      SELECT CustomerNo, CENTROID_ID
      FROM ML.PREDICT(MODEL `kaggle-bigquery-challenge.Sales_transaction.customer_segment_model`,
        (SELECT CustomerNo, ml.TF_IDF(SPLIT(purchase_history, ', ')) OVER() AS features FROM `kaggle-bigquery-challenge.Sales_transaction.vw_customer_purchase_history`)
      )
    )
    SELECT
      t.Country
    FROM `kaggle-bigquery-challenge.Sales_transaction.online_transaction` AS t
    JOIN Predictions AS p ON t.CustomerNo = p.CustomerNo
    WHERE p.CENTROID_ID = 2
    GROUP BY t.Country
    ORDER BY COUNT(t.CustomerNo) DESC
    LIMIT 1
    ```

    
    User's request: "{user_prompt}"

    SQL Query:
    """

    print("Generating SQL from user prompt...")
    response = model.generate_content(system_prompt)

    # Clean up the response to get only the SQL
    sql = response.text.strip().replace("```sql", "").replace("```", "")
    return sql

    # Clean up the response to get only the SQL
    sql = response.text.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()


# --- Let's test our MASTER AGENT ---
user_question = "Which cluster does customer 17490 belong to?"

# 1. The agent converts your question into a complex SQL query
generated_sql = generate_sql_from_prompt(user_question)
print(f"ğŸ¤– Generated SQL:\n{generated_sql}")

# 2. The agent executes its own query against BigQuery
results_df = execute_bq_query(generated_sql)

# 3. The agent presents the structured data results
print("\n--- ğŸ¤– AGENT QUERY RESULT ---")
if results_df is not None:
  print(results_df.to_string())

  # 4. AGENTIC STEP: The agent inspects the results and decides to use its vision tool
  if 'ImageUrl' in results_df.columns and not results_df['ImageUrl'].empty:
      image_url = results_df['ImageUrl'][0]
      if image_url:
          # If an image URL was found, call the vision tool
          image_analysis_result = analyze_image_from_url(image_url)
          print(image_analysis_result)
else:
  print("The query failed to execute.")

