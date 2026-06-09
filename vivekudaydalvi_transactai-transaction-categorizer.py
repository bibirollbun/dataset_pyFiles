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


! pip install -q google-genai pandas pydantic streamlit


import pandas as pd
import json
from kaggle_secrets import UserSecretsClient

# 1. Get the key from secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
except Exception as e:
    print("Error: GOOGLE_API_KEY not found in secrets. Please add it.")
    raise e

# 2. Configure the client
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal

client = genai.Client(api_key=GOOGLE_API_KEY)
# Use a capable and fast model
MODEL_NAME = 'gemini-2.5-flash'


# The master list of categories the agent can use
CATEGORY_OPTIONS = Literal[
    "Groceries", "Food & Dining", "Transportation", 
    "Utilities", "Income", "Transfer", "Entertainment", 
    "Shopping", "Other"
]


class TransactionCategory(BaseModel):
    """A financial transaction object for categorization."""
    
    # Use the Literal type to constrain the model's output to only the defined categories
    category: CATEGORY_OPTIONS = Field(
        description="The assigned category for the transaction."
    )
    # Optional: include the merchant name for extra detail
    merchant_name: str = Field(
        description="The clean, common name of the merchant/payer."
    )
    # Optional: include the confidence score to show advanced reasoning
    confidence_score: Literal["High", "Medium", "Low"] = Field(
        description="Confidence in the assigned category."
    )

# The configuration for the API call
output_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=TransactionCategory,
)


# Simulate or load your transaction data
# If you are using an external CSV, replace this with: pd.read_csv('/kaggle/input/your-dataset/transactions.csv')
data = {
    'raw_text': [
        "DEBIT EUR 45.75 at Starbucks #1234/Paris",
        "TRF from John Doe, ref: Salary March 2025",
        "Payment to Netflix.com on 03/15/2025",
        "Charged USD 19.99 for Uber trip to airport",
        "W/D 150.00 ATM Fee Waived, ACME Grocery Store",
        "TRANSFER TO UTILITY COMPANY REF: ELEC BILL",
        "rs 1230 paid to 987654323 via gpay"
    ]
}
df = pd.DataFrame(data)


def categorize_transaction(raw_text: str) -> dict:
    """Uses the Gemini API to categorize a single transaction."""
    
    # Define the core instruction for the model
    prompt = f"""
    Analyze the following raw financial transaction text. 
    Your task is to extract the merchant name and assign the most accurate category 
    from the predefined options.

    Transaction Text: "{raw_text}"
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
            config=output_config
        )
        # The response.text is guaranteed to be valid JSON due to structured output
        return json.loads(response.text)
    
    except Exception as e:
        print(f"An error occurred for transaction '{raw_text}': {e}")
        return {"category": "Error", "merchant_name": "N/A", "confidence_score": "Low"}


# Apply the function to the 'raw_text' column
df['agent_result'] = df['raw_text'].apply(categorize_transaction)

# Expand the dictionary result into separate columns for a clean view
df = pd.concat([df.drop(['agent_result'], axis=1), 
               df['agent_result'].apply(pd.Series)], axis=1)

print("\n--- Final Categorization Results ---")
print(df.to_markdown(index=False))

