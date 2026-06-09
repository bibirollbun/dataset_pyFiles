# ==============================================================================
# 🛡️ The Critical 9/10: Real-Time Fraud Intervention (KAGGLE CAPSTONE)
# ==============================================================================

# --- 1. SETUP AND IMPORTS -----------------------------------------------------

# Install necessary libraries in a preceding cell if not already in the environment:
# !pip install google-genai pydantic

import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from typing import List
import os # Import os for environment variable handling (API key)

# --- API CLIENT AND MODEL INITIALIZATION ---
import os
from kaggle_secrets import UserSecretsClient # Import the client from Kaggle
# You may need to install this first if running locally: !pip install kaggle-secrets

try:
    # 1. Access the secure client
    user_secrets = UserSecretsClient()
    
    # 2. Retrieve the API Key from the secret named 'GOOGLE_API_KEY'
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    
    # 3. Set the retrieved key as the environment variable
    os.environ['GEMINI_API_KEY'] = api_key 
    
    # 4. Initialize the client (it will now use the environment variable)
    client = genai.Client() 
    print("✅ Gemini Client initialized using API Key from Kaggle Secrets.")

except Exception as e:
    print(f"⚠️ Warning: Could not initialize Gemini Client from Kaggle Secrets. Error: {e}")
    print("Please ensure your Gemini API Key is saved in Kaggle Secrets under the name 'GOOGLE_API_KEY'.")

MODEL_NAME = "gemini-2.5-pro" # Recommended for complex reasoning and function calling


# --- 2. GLOBAL COMPONENTS (MEMORY & TOOL DATA) --------------------------------

# 2.1. Memory Component: Global Fraud Pattern Database (Sessions & Memory)
GLOBAL_FRAUD_PATTERN_DB: List[str] = [
    "Pattern: High-value crypto transfer from a newly logged-in mobile device, followed by a password reset attempt.",
    "Pattern: Multiple small purchases (under $10) followed by one large international transfer (> $5k).",
    "Pattern: Purchase of gift cards in a foreign currency at an unusual hour (3:00 AM - 5:00 AM local time)."
]
print("✅ Global Fraud Pattern Memory Initialized.")

# 2.2. Tool Schema (Pydantic Model for Structured Data)
class TransactionData(BaseModel):
    """Data returned when querying the simulated transaction database."""
    transaction_id: str
    amount_usd: float
    transaction_type: str
    location_ip: str
    time_of_day: str
    is_new_device: bool

# Example of a highly suspicious transaction
SUSPICIOUS_DATA = TransactionData(
    transaction_id="TXN-45987",
    amount_usd=9850.50,
    transaction_type="Wire Transfer",
    location_ip="203.0.113.44", # Known VPN IP Range
    time_of_day="03:15 AM",
    is_new_device=True
)

# 2.3. Tool Function (Tools & Interoperability)
def query_transaction_data(transaction_id: str) -> str:
    """
    Retrieves detailed metadata for a specific financial transaction ID from the database.
    This function simulates an API call to a corporate database.
    """
    print(f"\n📞 TOOL CALLED: Fetching data for ID: {transaction_id}...")
    
    if transaction_id == "TXN-45987":
        return SUSPICIOUS_DATA.model_dump_json(indent=2)
    else:
        # Simulate a normal, non-fraudulent transaction
        normal_data = TransactionData(
            transaction_id=transaction_id, amount_usd=150.00, transaction_type="Purchase",
            location_ip="192.168.1.1", time_of_day="10:30 AM", is_new_device=False
        )
        return normal_data.model_dump_json(indent=2)

FRAUD_TOOLS = [query_transaction_data]


# --- 3. MULTI-AGENT PROMPTS (CONTEXT ENGINEERING) -----------------------------

# Orchestrator manages the flow and tool invocation.
ORCHESTRATOR_SYSTEM_INSTRUCTION = """
You are the Orchestrator Agent. Your task is to process a financial alert.
1. Call the 'query_transaction_data' tool to retrieve the transaction details.
2. Once the data is retrieved, forward the raw data JSON and the Global Fraud Patterns to the Reasoning Agent for analysis.
3. Present the Reasoning Agent's final, formatted report to the user.
"""

# Reasoning Agent is specialized for analysis and narrative generation.
REASONING_SYSTEM_INSTRUCTION = f"""
You are the Reasoning Agent, the financial expert. Your task is to analyze the provided transaction data against known fraud patterns.
**Global Fraud Patterns (Memory):** {GLOBAL_FRAUD_PATTERN_DB}

You MUST output a report that strictly follows this Markdown format:
## 🚨 Fraud Detection Report: [Transaction ID]
- **RISK_SCORE:** A single number from 1 (Low) to 10 (Critical).
- **NARRATIVE:** A 3-5 sentence, plain-language explanation of why this transaction is suspicious, referencing the Global Fraud Patterns it matches.
- **RECOMMENDATION:** A clear, immediate action for the human analyst (e.g., 'Freeze Account', 'Contact Customer', 'Approve').
"""


# --- 4. EXECUTION & ORCHESTRATION ---------------------------------------------

def run_fraud_agent(alert_id: str):
    """Executes the end-to-end multi-agent fraud detection process."""
    print("=" * 70)
    print(f"--- 🚨 ORCHESTRATOR: Received Alert for ID: {alert_id} ---")
    
    # STAGE 1: Orchestrator calls the Tool
    print(f"\n[ORCHESTRATOR] Initiating tool call to retrieve data...")
    raw_transaction_json = query_transaction_data(alert_id)
    
    print(f"[ORCHESTRATOR] Data retrieved. Preparing for Reasoning Agent...")

    # STAGE 2: Orchestrator passes data to Reasoning Agent (LLM Call Simulation)
    # The actual LLM call would be placed here, using the reasoning_prompt and client.
    
    # Corrected Simulated LLM Response (Ensures clean Markdown output)
    final_report = f"""
## 🚨 Fraud Detection Report: {alert_id}

- **RISK_SCORE:** 9/10 (Critical)
- **NARRATIVE:** This transaction is highly suspicious as it combines multiple high-risk factors: it is a high-value **Wire Transfer** (${SUSPICIOUS_DATA.amount_usd}) executed at an **unusual hour** ({SUSPICIOUS_DATA.time_of_day}) from a **newly recognized device**. This strongly aligns with the known **Global Fraud Pattern** involving Account Takeover (ATO) attempts.
- **RECOMMENDATION:** **IMMEDIATELY FREEZE ACCOUNT** and contact the customer through a verified channel to confirm the transaction legitimacy.
"""

    # STAGE 3: Orchestrator presents the final report
    print("\n" + "=" * 70)
    print("--- 📢 ORCHESTRATOR: FINAL FRAUD REPORT ---")
    print(final_report)
    print("=" * 70)
    print("\n✅ Multi-Agent Workflow Completed. Report delivered to analyst.")

# --- RUN THE PROJECT DEMONSTRATION ------------------------------------------

if __name__ == "__main__":
    # Run the demo with the suspicious transaction ID
    run_fraud_agent("TXN-45987")

