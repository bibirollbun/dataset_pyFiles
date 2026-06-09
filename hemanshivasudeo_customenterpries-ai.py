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



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import os
import google.generativeai as genai
import numpy as np
import pandas as pd
from kaggle_secrets import UserSecretsClient # Import the client for reliable secret access

# List files in the input directory (standard Kaggle practice)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# --- BEGIN FINAL CORRECTED AI AGENT CODE ---

# 1. API KEY CONFIGURATION
MODEL = "gemini-2.5-flash"

# Global flag to track if the API is configured correctly
IS_CONFIGURED = False
API_KEY = None 

# --- Retrieval Block ---
try:
    # Attempt to retrieve the secret labeled "GOOGLE_API_KEY"
    user_secrets = UserSecretsClient()
    API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to retrieve secret 'GOOGLE_API_KEY'. Check if the secret exists and is enabled for this notebook. Details: {e}")

# --- Configuration Block ---
if API_KEY:
    try:
        # Attempt to configure the API using the retrieved key
        genai.configure(api_key=API_KEY)
        IS_CONFIGURED = True
        print("INFO: Gemini API configured successfully.")
    except Exception as e:
        # This triggers if the key is retrieved but is invalid/expired
        print(f"ERROR: Could not configure Gemini API. The retrieved key appears to be invalid or expired. Details: {e}")
        IS_CONFIGURED = False
else:
    print("WARNING: GOOGLE_API_KEY is missing. API calls will be skipped.")


# ----------------------------------------------------
# 2. AGENT DEFINITIONS
# ----------------------------------------------------

CUSTOMER_SYSTEM_PROMPT = """
You are a polite Customer Support Agent. Your job is to help customers with orders, refunds, and tracking.
CRITICAL RULE: Do NOT invent order details. Ask the customer for an order number if one is missing.

User Query: """ 


def customer_agent(message: str) -> str:
    """Handles customer-facing queries."""
    if not IS_CONFIGURED:
        return "ERROR: API is not configured. Please resolve the key issue above."

    PROMPT = CUSTOMER_SYSTEM_PROMPT + message
    try:
        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(PROMPT)
        return response.text
    except Exception as e:
        return f"API ERROR: Could not process request. Details: {e}"


ENTERPRISE_SYSTEM_PROMPT = """
You are an Enterprise Internal Support Agent. Your job is to assist internal staff (HR, Sales, Operations) with procedures and guidance.
CRITICAL RULE: DO NOT answer customer questions. If a customer question is asked, tell the user to contact the Customer Support team.

Staff Query: """


def enterprise_agent(message: str) -> str:
    """Handles internal employee queries."""
    if not IS_CONFIGURED:
        return "ERROR: API is not configured. Please resolve the key issue above."

    PROMPT = ENTERPRISE_SYSTEM_PROMPT + message
    try:
        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(PROMPT)
        return response.text
    except Exception as e:
        return f"API ERROR: Could not process request. Details: {e}"


# ----------------------------------------------------
# 3. TEST AND PRINT RESULTS
# ----------------------------------------------------
print(f"\n--- Running AI Agents using Model: {MODEL} ---")

# Test 1: Customer Query
customer_query = "Where is my refund for order #12345?"
print("\n--- CUSTOMER AGENT ---")
print(f"Query: {customer_query}")
print("Reply:\n", customer_agent(customer_query))

# Test 2: Enterprise Query
enterprise_query = "What is the new training schedule for the sales team?"
print("\n--- ENTERPRISE AGENT ---")
print(f"Query: {enterprise_query}")
print("Reply:\n", enterprise_agent(enterprise_query))

# --- END FINAL CORRECTED AI AGENT CODE ---

