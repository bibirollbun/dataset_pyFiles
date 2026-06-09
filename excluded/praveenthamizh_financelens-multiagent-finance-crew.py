import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


import pandas as pd
from io import StringIO

csv_data = """Date,Description,Type,Amount,Balance
2025-11-01,Opening Balance,Credit,2450.00,2450.00
2025-11-01,Salary - ACME Corp,Credit,3500.00,5950.00
2025-11-02,Rent Payment - Landlord,Debit,1200.00,4750.00
2025-11-02,Starbucks #524,Debit,6.75,4743.25
2025-11-03,Grocery - WholeMart,Debit,84.22,4659.03
2025-11-03,Transfer to Savings,Debit,500.00,4159.03
2025-11-04,Electricity Bill - EGrid,Debit,62.40,4096.63
2025-11-04,Spotify Subscription,Debit,9.99,4086.64
2025-11-05,ATM Withdrawal,Debit,100.00,3986.64
2025-11-05,Refund - Online Store,Credit,25.00,4011.64
2025-11-05,Netflix Subscription,Debit,20.00,3991.64
2025-11-05,Amazon Subscription,Debit,15.00,3971.64
"""

df = pd.read_csv(StringIO(csv_data))
df.head()


# Global Memory State (Simulating a Database)
MEMORY_STATE = {
    "raw_drafts": [],
    "processed_data": [],
    "budgets": {"Dining": 500, "Groceries": 500, "Utilities": 1000, "OTT": 25}
}


def ingest_tool():
    """
    INGEST TOOL: Reads raw CSV data and normalizes it.
    Args:
        no inputs, reads from global csv_data variable
    Returns:
        A dictionary with status and the parsed CSV data as a list of dictionaries.
        Success: {'status': 'success', 'csv_data_dict': [{'transaction_id': 'TXN-0001', 'Date': '2025-11-01', 'Description': 'Opening Balance', 'Category': 'Balance', 'Type': 'Credit', 'Amount': 2450.0, 'Balance': 2450.0}, {'transaction_id': 'TXN-0002', 'Date': '2025-11-01', 'Description': 'Salary - ACME Corp', 'Category': 'Salary', 'Type': 'Credit', 'Amount': 3500.0, 'Balance': 5950.0}]}
        Error: {"status": "error", "message": "Unsupported file format"}
    """
    print(f"   [Tool: Ingest] Reading CSV data...")
    try:
        df = pd.read_csv(StringIO(csv_data))
        df.columns = df.columns.str.lower()
        drafts = df.to_dict(orient='records')
        MEMORY_STATE['raw_drafts'] = drafts
        return {"status": "success", "csv_data_dict": drafts}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_budget_for_all_categories():
    """
    Retrieves budget allocations for all categories from memory.
    Returns:
        dict: A dictionary with category names as keys and budget amounts as values.
        Success: {'Dining': 500, 'Groceries': 300, 'Utilities': 150}
        Error: {"status": "error", "message": "No budgets found"}
    """
    try:
        budgets = MEMORY_STATE.get("budgets", {})
        if not budgets:
            return {"status": "error", "message": "No budgets found"}
        return {"status": "success", "budgets": budgets}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def load_processed_data():
    """
    Loads processed transaction data from memory.
    Returns:
        list: A list of processed transaction dictionaries.
        Success: [{'transaction_id': 'TXN-0001', 'Date': '2025-11-01', 'Description': 'Opening Balance', 'Category': 'Balance', 'Type': 'Credit', 'Amount': 2450.0, 'Balance': 2450.0}, ...]
        Error: {"status": "error", "message": "No processed data found"}
    """
    try:
        data = MEMORY_STATE.get("processed_data", [])
        if not data:
            return {"status": "error", "message": "No processed data found"}
        return data
    except Exception as e:
        return {"status": "error", "message": str(e)}


def remove_duplicates(csv_str: str):
    """
    Removes duplicate transactions based on 'transaction_id'.
    Args:
        csv_str: CSV string of transactions.
    Returns:
        A dictionary with status and the parsed CSV data as a list of dictionaries.
        Success: {'status': 'success', 'csv_data_dict': [{'transaction_id': 'TXN-0001', 'Date': '2025-11-01', 'Description': 'Opening Balance', 'Category': 'Balance', 'Type': 'Credit', 'Amount': 2450.0, 'Balance': 2450.0}, {'transaction_id': 'TXN-0002', 'Date': '2025-11-01', 'Description': 'Salary - ACME Corp', 'Category': 'Salary', 'Type': 'Credit', 'Amount': 3500.0, 'Balance': 5950.0}]}
        Error: {"status": "error", "message": "Unsupported file format"}
    """
    try:
        df = pd.read_csv(StringIO(csv_str))
        df.columns = df.columns.str.lower()
        df.drop(df[df["isDuplicate".lower()]].index, inplace=True)
        drafts = df.to_dict(orient='records')
        MEMORY_STATE['processed_data'] = drafts
        print(drafts)
        return {"status": "success", "csv_data_dict": drafts}
    except Exception as e:
        return {"status": "error", "message": str(e)}    


categorization_agent = Agent(
    name="categorization_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=(
        f"You are a financial transaction categorization agent. "
        f"Before classifying, call the tool named 'ingest_tool' to load transactions into memory."
        f"After ingestion, read transactions from {MEMORY_STATE['raw_drafts']} and classify each transaction into exactly one category from: "
        f"dining, transport, investment, utility, healthcare, entertainment, groceries, shopping, rent, OTT, income, miscellaneous. "
        f"Use only these eight categories. Do not invent categories or tags."
        f"return the full list of transactions with their assigned categories in csv format same as the input."
        f"For each transaction add the following columns: category (one of the eight), confidence (float 0.0-1.0)"
        f"Classify each transaction internally based on your knowledge.\n"
        f"CRITICAL INSTRUCTION: Do NOT call any tool to submit the result. "

    ),
    tools=[
        FunctionTool(ingest_tool)
    ],
    output_key="categorized_transactions",
)


reconciliation_agent = Agent(
    name="reconciliation_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=(
        """You are a strict Financial Auditor.
        Your goal is to ensure data integrity by identifying duplicate transactions and mark them.\n\n
        I duplicate transactions are fonud, mark the first transaction as False and the subsequent duplicates as True.\n\n
        You will receive a CSV string of categorized transactions from: {categorized_transactions}
        For each transaction add a column: isduplicate(boolean True/False)
        finally, call the tool 'remove_duplicates' to filter out the duplicates from the list"""
    ),
    tools=[FunctionTool(remove_duplicates)],
    output_key="reconciled_transactions", 
)


calculation_agent = Agent(
    name="CalculationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a specialized calculator that ONLY responds with Python code. You are forbidden from providing any text, explanations, or conversational responses.
 
     Your task is to take a request for a calculation and translate it into a single block of Python code that calculates the answer.
     
     **RULES:**
    1.  Your output MUST be ONLY a Python code block.
    2.  Do NOT write any text before or after the code block.
    3.  The Python code MUST calculate the result.
    4.  The Python code MUST print the final result to stdout.
    5.  You are PROHIBITED from performing the calculation yourself. Your only job is to generate the code that will perform the calculation.
   
    Failure to follow these rules will result in an error.
       """,
    code_executor=BuiltInCodeExecutor(),  # Use the built-in Code Executor Tool. This gives the agent code execution capabilities
)


budget_insights_agent = Agent(
    name="budget_insights_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=(
        """ 1. You are a helpful Financial Advisor. Your goal is to help the user save money.
            2. Before generating the report, call the tools 'get_budget_for_all_categories' and 'load_processed_data' to load budgets and verified transactions from memory.\n
            3. (CRITICAL): You are strictly prohibited from performing any arithmetic calculations yourself. You must use the calculation_agent tool to generate Python code that calculates the needed amount.
            4. Calculate Total Money Spent Amount using calculate_agent tool by instructing it calculate by writing python code, get the data from load_processed_data tool.
            5. Calculate Amount Spent on each category using calculate_agent tool by instructing it calculate by writing python code, get the data from load_processed_data tool.
            6. Generate a 'Monthly Financial Report' with the following sections:\n\n"
                **ðŸ“Š Monthly Summary**\n"
                    - Total money spent.\n"
                    - List each category with Spent vs Budget (e.g., 'Dining: $200 / $150').\n"
                    - Use emojis for status (âœ… for under budget, ðŸš¨ for over budget).\n\n"
                **ðŸš¨ Budget Breaches**\n"
                    - Highlight any category where the user overspent.\n"
                    - Show the exact overage amount.\n\n"
                **ðŸ’¡ Top 3 Suggested Reductions**\n"
                    - Look at the 'top_merchants_by_spend' from the tool output.\n"
                    - Identify the top 3 specific merchants or descriptions where the user spends the most.\n"
                    - Give a specific 1-sentence tip for each (e.g., 'Starbucks: Consider making coffee at home to save $50')."""
    ),
    tools=[
        AgentTool(calculation_agent), FunctionTool(get_budget_for_all_categories), FunctionTool(load_processed_data)
    ],
    output_key="financial_report",
)



root_agent = SequentialAgent(
    name="BlogPipeline",
    sub_agents=[categorization_agent, reconciliation_agent, budget_insights_agent],
)


runner = InMemoryRunner(agent=root_agent)

response = await runner.run_debug(
        "categorize, reconcile the transactions and generate finance insights."
    )

