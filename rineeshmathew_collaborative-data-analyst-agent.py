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


import sqlite3
from typing import List, Dict

# --- PART 1: SETUP (Simulating the Kaggle Dataset) ---
# We create a temporary in-memory SQLite database with dummy data
def setup_dummy_database():
    conn = sqlite3.connect(':memory:')
    df = pd.DataFrame({
        'CustomerID': [101, 102, 103, 104, 105],
        'Region': ['North', 'South', 'North', 'East', 'South'],
        'MonthlyCharges': [50.0, 80.0, 55.0, 20.0, 85.0],
        'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
    })
    df.to_sql('customer_churn', conn, index=False)
    return conn

# Initialize the DB
db_connection = setup_dummy_database()

# --- PART 2: THE TOOLS (The Hands) ---

def tool_rag_schema_lookup(user_query: str) -> str:
    """
    RAG Tool: Simulates retrieving relevant metadata from a Vector Store.
    In production, this queries ChromaDB/Pinecone.
    """
    # Simulated retrieved context based on the query "Analyze churn..."
    schema_context = """
    DATA DICTIONARY MATCHES:
    Table: customer_churn
    - Column 'Region' (Text): Geographic location.
    - Column 'MonthlyCharges' (Float): The amount charged to the customer.
    - Column 'Churn' (Text): 'Yes' indicates churn, 'No' indicates retention.
    """
    return schema_context

def tool_execute_sql(sql_query: str) -> str:
    """
    SQL Tool: Executes the query against the database.
    """
    try:
        # Safety: A real agent needs read-only permissions here
        df = pd.read_sql_query(sql_query, db_connection)
        return df.to_markdown(index=False)
    except Exception as e:
        return f"SQL ERROR: {str(e)}"

# --- PART 3: THE AGENT LOGIC (The Specialist) ---

def data_query_agent(sub_task: str, llm_client):
    """
    The Logic Flow:
    1. Look up schema info (Grounding).
    2. Construct a prompt with schema + task.
    3. Generate SQL.
    4. Execute SQL.
    """
    print(f"\nğŸ¤– DATA AGENT: Received task -> '{sub_task}'")
    
    # Step 1: Call RAG Tool to ground the knowledge
    print("   ... Consulting Data Dictionary (RAG) ...")
    schema_context = tool_rag_schema_lookup(sub_task)
    
    # Step 2: Construct the Prompt
    # We inject the schema_context into the system prompt
    prompt = f"""
    You are an expert SQL Data Analyst.
    
    1. CONTEXT (Database Schema):
    {schema_context}
    
    2. TASK:
    {sub_task}
    
    3. INSTRUCTION:
    Write a valid SQL query to answer the task. 
    Return ONLY the SQL query. Do not wrap in markdown or backticks.
    """
    
    # Step 3: LLM generates SQL (Simulated LLM call)
    # In production: response = llm_client.chat.completions.create(...)
    print("   ... Generating SQL ...")
    
    # Hardcoded simulation of what GPT-4o would return given the prompt above
    generated_sql = "SELECT Churn, AVG(MonthlyCharges) as Avg_Cost FROM customer_churn GROUP BY Churn"
    print(f"   ... Generated: {generated_sql}")

    # Step 4: Execute Tool
    print("   ... Executing Query ...")
    raw_results = tool_execute_sql(generated_sql)
    
    return raw_results

# --- PART 4: EXECUTION ---

# The Coordinator (User) sends a specific sub-task
mission = "Calculate the average monthly charges for churned vs non-churned users."

# Run the Agent
final_data = data_query_agent(mission, llm_client=None)

print("\nğŸ“Š AGENT OUTPUT (Raw Data):")
print(final_data)


# !pip install -q -U google-generative-ai

import sqlite3
import pandas as pd
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import textwrap

# 1. AUTHENTICATION
# Retrieve the API key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("âœ… Google AI Studio API Configured Successfully.")
except Exception as e:
    print("â�Œ Error: Could not find 'GOOGLE_API_KEY' in Kaggle Secrets.")
    print("Please go to Add-ons -> Secrets -> Add 'GOOGLE_API_KEY'.")

# 2. MODEL SETUP
# We use Gemini 1.5 Flash (Fast & Cost-efficient for Logic/SQL)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# 3. DATA SETUP (Simulating a Kaggle CSV)
# We create a local SQLite DB to represent the dataset
conn = sqlite3.connect(':memory:')
df_dummy = pd.DataFrame({
    'CustomerID': [101, 102, 103, 104, 105, 106, 107, 108],
    'Region': ['North', 'South', 'North', 'East', 'South', 'West', 'North', 'East'],
    'MonthlyCharges': [50.0, 80.0, 55.0, 20.0, 85.0, 110.0, 45.0, 25.0],
    'Churn': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No']
})
df_dummy.to_sql('customer_churn', conn, index=False)
print("âœ… Dummy Database 'customer_churn' Created.")


def tool_rag_lookup(user_query: str):
    """
    Simulates the RAG retrieval. In a real scenario, this queries a Vector DB.
    Here, it returns the schema string required for SQL generation.
    """
    # Context meant to ground the LLM so it doesn't hallucinate column names
    schema_info = """
    TABLE: customer_churn
    COLUMNS:
    - Region (TEXT): The geographic area (North, South, East, West).
    - MonthlyCharges (REAL): The monthly bill amount in USD.
    - Churn (TEXT): 'Yes' if the customer left, 'No' if they stayed.
    """
    return schema_info

def tool_execute_sql(sql_query: str):
    """
    Executes the SQL against the SQLite database.
    """
    try:
        # Execute and return DataFrame
        return pd.read_sql_query(sql_query, conn)
    except Exception as e:
        return f"SQL_ERROR: {e}"

def clean_sql_output(llm_response: str):
    """
    Helper to strip markdown formatting (```sql ... ```) from Gemini's response.
    """
    clean_text = llm_response.replace("```sql", "").replace("```", "").strip()
    return clean_text


def data_query_agent(sub_task: str):
    print(f"\nğŸ¤– AGENT WORKING ON: '{sub_task}'")
    
    # 1. Retrieve Context (RAG)
    schema_context = tool_rag_lookup(sub_task)
    print(f"   ... Context Retrieved: customer_churn table schema")

    # 2. Construct Prompt
    # We use a 'System' style prompt to define behavior
    prompt = f"""
    You are a SQL Expert for a Data Analysis Agent.
    
    CONTEXT (Database Schema):
    {schema_context}
    
    TASK:
    {sub_task}
    
    INSTRUCTIONS:
    1. Write a standard SQL query (SQLite compatible) to solve the task.
    2. Return ONLY the SQL code. No explanation, no markdown formatting.
    """

    # 3. Call Google AI Studio (Gemini)
    print(f"   ... Calling Gemini API ...")
    response = model.generate_content(prompt)
    
    # 4. Clean Output
    sql_query = clean_sql_output(response.text)
    print(f"   ... Generated SQL: {sql_query}")
    
    # 5. Execute SQL
    print(f"   ... Executing against Database ...")
    result_df = tool_execute_sql(sql_query)
    
    return result_df


# --- SCENARIO ---
# The Coordinator has broken the user request down into this specific sub-task:
mission = "Compare the average monthly charges between customers who churned and those who did not."

# Execute Agent
final_result = data_query_agent(mission)

# Output Results
print("\nğŸ“Š FINAL RAW DATA (Output for Reporting Agent):")
if isinstance(final_result, pd.DataFrame):
    print(final_result.to_markdown(index=False))
else:
    print(final_result)




