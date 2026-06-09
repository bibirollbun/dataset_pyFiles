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


!pip install google-adk pandas matplotlib
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Import ADK components
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool


from kaggle_secrets import UserSecretsClient
try:
    # Load the secret securely
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


# Initialize the model instance
model = Gemini()
print("âœ… Gemini 2.5 Flash model initialized successfully.")


# --- 2.1 Define the fetch_log() Tool ---
def fetch_log(sprint_id: str) -> list[dict]:
    """
    Simulates fetching raw change logs for a specific sprint ID from a mock source.
    This function acts as the 'Tool' for the RAA-Collector Agent.
    """
    if sprint_id == "SPRINT-25":
        return [
            {"id": "JIRA-401", "log": "Refactor caching layer to use Redis instead of Memcache for better performance. No user impact."},
            {"id": "JIRA-402", "log": "Fixed critical bug where cart total was double-counting VAT for logged-in users."},
            {"id": "JIRA-403", "log": "Implemented 'Pay with Crypto' as a new checkout option for US customers."},
            {"id": "JIRA-404", "log": "Updated logging database schema for better auditability (internal change)."},
            {"id": "JIRA-405", "log": "Addressed XSS vulnerability on the login page as part of Q3 security audit."},
            {"id": "JIRA-406", "log": "Minor UI text alignment fix on the mobile product page."},
            {"id": "JIRA-407", "log": "Optimized API call to the inventory service, reducing latency by 300ms."},
        ]
    return []

# Create the FunctionTool wrapper
fetch_log_tool = FunctionTool(fetch_log)
print("âœ… Custom Tool 'fetch_log' defined and wrapped.")


# --- RAA_Collector ---
# Role: Executes the tool to get raw data.
RAA_Collector = Agent(
    model=model,
    name="RAA_Collector",
    description="Tool execution agent that fetches raw, unstructured change logs. Your sole purpose is to retrieve the raw change log data for the requested sprint ID using the available tool. Return only the raw list of dictionaries.", 
    tools=[fetch_log_tool]
)

# --- RAA_Categorizer ---
# Role: Classifies the raw data.
RAA_Categorizer = Agent(
    model=model,
    name="RAA_Categorizer",
    description=f"""
    Categorizes change logs by impact type. You receive a list of raw change logs. Analyze the log message and classify its primary impact into one of the following three categories:
    1. Business Value (User-facing features, major bug fixes impacting revenue/UX).
    2. Technical Health (Refactoring, performance, internal dependencies, no direct user impact).
    3. Compliance & Security (Regulatory requirements, security vulnerabilities).

    Return the result as a list of dictionaries, where each dictionary includes the original 'id' and 'log', plus a new field 'category'.
    """,
    # The 'tools' list is empty as this agent performs LLM reasoning only
)

# --- RAA_Formatter (The Orchestrator) ---
# Role: Manages the flow, generates the final artifacts, and performs evaluation.
RAA_Formatter = Agent(
    model=model,
    name="RAA_Formatter",
    description="""
    The main agent that orchestrates data collection, categorization, and generates final release artifacts. You are the Release Manager. Your task is:
    1. Call RAA_Collector to get the raw logs for the requested sprint.
    2. Send the raw logs to RAA_Categorizer to receive the categorized data.
    3. Generate TWO distinct artifacts based on the categorized data:
        - Technical Notes: Focus on Technical Health and Compliance categories. Use technical language.
        - Business Summary: Focus only on Business Value and Compliance. Use high-level, impact-driven language.
    4. Perform the final Evaluation check (criteria defined below).

    Format the final output as a JSON object containing two keys: 'technical_notes' and 'business_summary'.
    """,
    # The 'tools' list is empty as this agent performs LLM reasoning only
)
print("âœ… Multi-Agent System (RAAS) agents defined: Collector, Categorizer, and Formatter.")


EVALUATION_CRITERIA = """
CRITERIA FOR SUCCESS:
1. Technical Notes must include at least one item from the 'Technical Health' category.
2. Business Summary must use non-technical language and must not mention 'Refactor' or 'Memcache'.
3. The 'JIRA-405' (Security fix) must be present in both summaries but must be paraphrased gently (e.g., 'Enhancements to login stability').
"""
# The prompt initiates the entire flow
prompt = f"Generate the Release Notes and Summary for SPRINT-25 based on the following criteria: {EVALUATION_CRITERIA}"

runner = InMemoryRunner(agent=RAA_Formatter) 
print("ğŸš€ Starting RAAS Execution...")


final_output = await runner.run_debug(prompt)

print("\n--- Final Generated Output ---")
# Attempt to extract and print the final JSON for readability
try:
    notes_raw_text = final_output[0].content.parts[0].text
    notes_json_str = notes_raw_text.split("```json")[-1].split("```")[0].strip()
    notes_data = json.loads(notes_json_str)
    
    print("\n## ğŸ“� Business Summary (High-Level Impact)")
    print(notes_data.get('business_summary', 'Not Found'))
    
    print("\n## ğŸ”§ Technical Notes (Detailed Changes)")
    print(notes_data.get('technical_notes', 'Not Found'))
    
except Exception as e:
    print(f"Could not parse final structured output. Printing raw result:\n{final_output}")
    print(f"Parsing error details: {e}")

print("\n--- Execution Complete ---")


# --- Visualization of the RAA-Categorizer's Work ---

# Simulate retrieving the categorized results (You would extract this from the runner's log in a production setting)
categorized_data_mock = {
    'Category': ['Business Value', 'Technical Health', 'Compliance & Security'],
    'Count': [3, 3, 1] # Based on the mock data in fetch_log()
}
df = pd.DataFrame(categorized_data_mock)

plt.figure(figsize=(9, 6))
bars = plt.bar(df['Category'], df['Count'], color=['#007BFF', '#28A745', '#DC3545']) # Distinct colors
plt.title('Distribution of SPRINT-25 Changes by Impact Category', fontsize=14)
plt.ylabel('Number of Change Items', fontsize=12)
plt.xticks(rotation=15, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add counts on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, yval, ha='center', va='bottom', fontsize=11)

plt.show()

