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


pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent , SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, AgentTool
from google.genai import types



# Define helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")



retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


#-----DB helper functions--------#
import sqlite3
from datetime import datetime

DB_FILE = "expense.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  description TEXT,
                  amount REAL,
                  category TEXT,
                  split_details TEXT,
                  transaction_type TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS categories
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE,
                  budget REAL)''')
    
    # NEW: Debts table for Splitwise features
    c.execute('''CREATE TABLE IF NOT EXISTS debts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  debtor TEXT,
                  creditor TEXT,
                  amount REAL,
                  description TEXT,
                  timestamp TEXT,
                  status TEXT)''')
    
    c.execute("SELECT count(*) FROM categories")
    if c.fetchone()[0] == 0:
        defaults = [('Dining', 200), ('Groceries', 300), ('Transport', 100), 
                    ('Entertainment', 150), ('Shopping', 200), ('Bills', 500), ('Others', 100)]
        c.executemany("INSERT INTO categories (name, budget) VALUES (?, ?)", defaults)
        
    conn.commit()
    conn.close()

init_db()


#---- DB tool to read data ----#
def read_sql_query_tool(query: str) -> str:
    """
    Executes a READ-ONLY SQL query on the expense database.
    Useful for answering questions 
    
    To answer budget or transaction related questions, either join transactions.category with categories.name 
    and use SUM(amount) to compute spent and remaining or for transactions use table 'transactions' with columns: id, timestamp,
    description, amount, category, split_details, transaction_type. 
    -transaction_type = 'debit' means money going out like (spending).
    -transaction_type = 'credit' means money coming in like (income/refunds/etc.).
    -for example :
    "How much did I spend on food?" or "List recent transactions" or "What is my Dining Budget and how much is left ?".

    To answer debt related questions for example :
    "Whom do I have to pay?" or "Who has to pay me?" or "who settled ?" etc
    query table 'detbs' with columns: id Id, debtor, creditor,amount ,description , timestamp , status with appropriate filters 
    on debtor and creditor and status. where for user ALWAYS use 'Me' in creditor or debtor according to question and for status
    use 'unsettled' for open or unsettled debts/transcations and 'settled' for settled debts/transactions
    
    Args: 
        query: A valid SQL SELECT statement.
    """
    import sqlite3
    try:
        if not query.strip().upper().startswith("SELECT"):
            return "ERROR: Only SELECT queries are allowed."
            
        conn = sqlite3.connect("expense.db")
        c = conn.cursor()
        c.execute(query)
        rows = c.fetchall()
        columns = [description[0] for description in c.description]
        conn.close()
        
        if not rows:
            return "No results found."
            
        result = [dict(zip(columns, row)) for row in rows]
        return str(result)
    except Exception as e:
        return f"ERROR: Query failed. {str(e)}"


#------- DB tool to update Data-------#
def execute_sql_update_tool (query: str, params: dict={}) -> dict:
    conn = sqlite3.connect("expense.db")
    cur = conn.cursor()
    cur.execute(query, params or {})
    conn.commit()
    rows_affected = cur.rowcount
    conn.close()
    return {"rows_affected": rows_affected}



# Category Classifier Agent 

root_agent = Agent(
    name="CategoryClassifier",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An intelligent agent that enriches transaction data. It takes raw, ambiguous descriptions (e.g., 'Paid $15 at The Golden Fleece') and uses Google Search to identify the merchant's business type (e.g., 'Restaurant' vs. 'Clothing Store') to assign a precise budget category.",
    instruction="""
        You are an autonomous Transaction Classifier. 
        
        RULES:
        1. Input: A string like "Uber trip to airport" or "Starbucks $5".
        2. Action: If the merchant is not obvious, use the 'google_search' tool to find their primary business type.
           - Example: If user says "Dunder Mifflin", search for it. If search says "Paper Company", categorize as "Office Supplies" (or 'Shopping').
        3. Output: specific JSON format: {"category": "...", "description": "..."}
        
        Standard Categories: [Groceries, Dining, Transport, Bills, Shopping, Entertainment, Health, Investment, Others]
        """,
    tools=[google_search],
    output_key="category_found"
)


#------transaction saver tool -------------
def save_transaction_tool(description: str, amount: float, category: str, transaction_type: str,split_details: str = "None") -> str:
    """
    Saves a validated transaction to the persistent SQLite database.
    
    Args:
        description: What was bought (e.g., "Starbucks Coffee").
        amount: The cost as a float (e.g., 5.50).
        category: The category determined by the Classifier (e.g., "Dining").
        split_details: (Optional) Text describing who owes what (e.g., "Bob owes $2.75").
        transaction_type: debit if user is paying or lending money (losing money), credit if user is receiving money (earning money).
    Returns:
        A success message with the Transaction ID.
    """
    if amount <= 0:
        return "ERROR: Amount must be positive."
    
    if not category or category == "Unknown":
        return "ERROR: Category is missing. Please categorize before saving."

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Check budget
        c.execute("SELECT budget FROM categories WHERE name = ?", (category,))
        row = c.fetchone()
        budget_msg = ""
        if row:
            budget = row[0]
            c.execute("SELECT SUM(amount) FROM transactions WHERE category = ?", (category,))
            spent = c.fetchone()[0] or 0
            if spent + float(amount) > budget:
                budget_msg = f" WARNING: You have exceeded your {category} budget of ${budget}!"
        
        c.execute("INSERT INTO transactions (timestamp, description, amount, category, split_details, transaction_type) VALUES (?, ?, ?, ?, ?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description, float(amount), category, split_details, transaction_type))
        trans_id = c.lastrowid
        conn.commit()
        conn.close()
        return f"SUCCESS: Transaction #{trans_id} saved. {description} - ${amount} ({category}).{budget_msg}"
    except Exception as e:
            return f"ERROR: Failed to save transaction. {str(e)}"



# Transaction Saver agent 

saver_agent = Agent(
    name="TransactionSaver",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="The final agent in the pipeline. It commits validated data to the database.",
        instruction="""
        You are the Transaction Gatekeeper.
        
        **Your Goal:** Save the transaction using the 'save_transaction_tool'.
        
        **Rules:**
        1. You will receive inputs including Description, Amount, Category, Split Details and transaction_type (either 'debit' or 'credit').
        2. If transaction_type is not provided, default to 'debit' if user is paying or lending money (losing money) in transaction. Use 'credit' when the user clearly receives money (Example : refunds, salary, friend paying you back) .
        3. You MUST call the 'save_transaction_tool' with these exact parameters.
        4. If the tool returns "SUCCESS", confirm this to the user.
        5. If the tool returns "ERROR", report the error back to the Orchestrator/User do NOT try to fake a save.
        
        **Critical:** Do not hallucinate a successful save. Only report success if the tool returns it.
        """,
    tools=[save_transaction_tool]
)



#---- splitwise agent tools------
DB_FILE = "expense.db"
ME_NAME = "Me"

def add_debt_tool(debtor: str, creditor: str, amount: float,
                  description: str, status: str) -> str:
    """
    Low-level tool: record a SINGLE one-to-one debt row.

    Args:
        debtor: Person who owes money.
        creditor: Person who is owed money.
        amount: Amount owed.
        description: What the debt is for.
        status: "unsettled" or "settled".
    """
    import sqlite3
    from datetime import datetime

    if amount <= 0:
        return "ERROR: Amount must be positive."

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        c.execute(
            "INSERT INTO debts (debtor, creditor, amount, description, status, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (debtor.strip(), creditor.strip(), float(amount), description, status, now)
        )
        conn.commit()
        conn.close()
        return f"SUCCESS: Recorded that {debtor} owes {creditor} {amount} for {description} ({status})."
    except Exception as e:
        return f"ERROR: Failed to record debt. {str(e)}"



#---splitwise agent tool -----
def record_group_debts(
    creditors: str,
    debtors: str,
    total_amount: float,
    description: str,
    status: str = "unsettled",
    split_mode: str = "equal",
    fair_shares: str = "",
    paid_shares: str = "",
) -> str:
    """
    High-level tool: expand group expense into one-to-one debts involving Me.

    Args:
        creditors: Comma-separated people who paid (e.g., "Me" or "Me, John").
        debtors: Comma-separated people who did not pay but consumed
                 (e.g., "John, Sarah, Bob" or "Sarah, Bob, Rafael").
        total_amount: Total bill.
        description: What the expense was for.
        status: "unsettled" or "settled".
        split_mode:
            - "equal": equal fair share among all consumers (creditors + debtors).
            - "custom": use fair_shares string.
        fair_shares:
            - For "custom": per-person FAIR share, e.g.
              "Me:205, John:50, Bob:100, Rafael:70, Sarah:75".
        paid_shares:
            - Optional: who actually paid how much, e.g.
              "Me:300, John:200". If omitted:
              - in "equal" mode, assume equal payment among creditors;
              - in "custom" mode, assume each person paid exactly their fair share.
    
    Behavior:
        - Computes net = paid - fair_share for each participant.
        - Builds debts ONLY where ME_NAME is debtor or creditor.
        - Uses add_debt_tool() to insert each edge.
    """
    from collections import defaultdict

    # 1) Parse lists
    creditor_list = [c.strip() for c in creditors.split(",") if c.strip()]
    debtor_list = [d.strip() for d in debtors.split(",") if d.strip()]
    participants = sorted(set(creditor_list + debtor_list))

    if ME_NAME not in participants:
        return "INFO: Me is not part of this transaction; nothing recorded."

    # 2) FAIR SHARE for each participant
    fair = {}

    if split_mode.lower() == "equal":
        per_person = float(total_amount) / len(participants)
        for p in participants:
            fair[p] = per_person
    elif split_mode.lower() == "custom":
        if not fair_shares:
            return "ERROR: fair_shares is required for split_mode='custom'."
        # fair_shares like "Me:205, John:50, Bob:100, Rafael:70, Sarah:75"
        parts = [x.strip() for x in fair_shares.split(",") if x.strip()]
        for part in parts:
            name, val = [x.strip() for x in part.split(":", 1)]
            fair[name] = float(val)
        for p in participants:
            if p not in fair:
                return f"ERROR: No fair share specified for '{p}' in fair_shares."
    else:
        return "ERROR: split_mode must be 'equal' or 'custom'."

    # 3) PAID AMOUNTS
    paid = defaultdict(float)

    if paid_shares:
        # paid_shares like "Me:300, John:200"
        parts = [x.strip() for x in paid_shares.split(",") if x.strip()]
        for part in parts:
            name, val = [x.strip() for x in part.split(":", 1)]
            paid[name] = float(val)
    else:
        if split_mode.lower() == "equal":
            # assume equal payment among creditors
            per_creditor_paid = float(total_amount) / len(creditor_list)
            for c in creditor_list:
                paid[c] = per_creditor_paid
        else:
            # custom mode, assume each pays exactly their fair share
            for p in participants:
                paid[p] = fair.get(p, 0.0)

    # 4) NET position for each participant
    net = {}
    for p in participants:
        net[p] = paid[p] - fair.get(p, 0.0)

    net_me = net.get(ME_NAME, 0.0)
    if abs(net_me) < 1e-6:
        return "INFO: Me is already settled; no debts recorded."

    calls_made = 0

    # Case A: Me is creditor (others owe Me)
    if net_me > 0:
        total_owing = sum(-net[p] for p in participants if net[p] < 0)
        if total_owing <= 0:
            return "INFO: No one owes Me after balancing."

        for p in participants:
            if p == ME_NAME:
                continue
            if net[p] < 0:
                # proportional share of what they owe to Me
                share_to_me = net_me * (-net[p] / total_owing)
                if share_to_me > 0.01:
                    add_debt_tool(
                        debtor=p,
                        creditor=ME_NAME,
                        amount=round(share_to_me, 2),
                        description=description,
                        status=status
                    )
                    calls_made += 1

    # Case B: Me is debtor (I owe others)
    else:
        total_credit = sum(net[p] for p in participants if net[p] > 0)
        if total_credit <= 0:
            return "INFO: Me owes no one after balancing."

        for p in participants:
            if p == ME_NAME:
                continue
            if net[p] > 0:
                share_from_me = (-net_me) * (net[p] / total_credit)
                if share_from_me > 0.01:
                    add_debt_tool(
                        debtor=ME_NAME,
                        creditor=p,
                        amount=round(share_from_me, 2),
                        description=description,
                        status=status
                    )
                    calls_made += 1

    return f"SUCCESS: Recorded {calls_made} debt edges involving Me for '{description}'."



# --- SPLITWISE AGENT ---
splitwise_agent = Agent(
    name="SplitwiseManager",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="Manages debts and shared expenses between people.",
    instruction="""
        You are the Splitwise Manager for a personal expense tracker.
        
        Goal:
        - Track who owes whom, but ONLY for the user ('Me').
        - Always convert group expenses into one-to-one debts where either the debtor or the creditor is 'Me'.
        
        Tools:
        - Use 'record_group_debts' to RECORD new debts from natural-language descriptions.
        - Use 'read_sql_query_tool' to READ existing debts from the 'debts' table.
        
        Recording debts (record_group_debts):
        - This tool handles one-to-one, one-to-many, many-to-one, and many-to-many.
        - It should always be called instead of directly calling SQL inserts.
        - Arguments:
          - creditors: comma-separated people who paid (e.g., "Me", or "Me, John").
          - debtors: comma-separated people who did not pay but consumed (e.g., "Me","Me,Bob, Sarah").
          - total_amount: total bill.
          - description: short label, like "Dinner" or "Cab".
          - status: usually "unsettled" for new debts.
          - split_mode:
              * "equal"  -> equal fair share among all consumers.
              * "custom" -> use fair_shares to specify per-person fair share.
          - fair_shares (optional for "custom"): "Name1:x1, Name2:x2, ..."
          - paid_shares (optional): "Name1:x1, Name2:x2, ..." for who actually paid how much.
        
        Work through these examples:
        
        1) Example 1: "I paid 200/- for John, Sarah and Bob and to be split equally."
           - creditors="Me"
           - debtors="John, Sarah, Bob"
           - total_amount=200
           - split_mode="equal"
           Call:
           record_group_debts(
               creditors="Me",
               debtors="John, Sarah, Bob",
               total_amount=200,
               description="Dinner",
               status="unsettled",
               split_mode="equal"
           )
        
        2) Example 2: "I paid 200 for John, Sarah and Bob where Bob owes 60, Sarah 40 and John 20."
           - You already know each person's FAIR SHARE.
           - creditors="Me"
           - debtors="John, Sarah, Bob"
           - total_amount=200
           - split_mode="custom"
           - fair_shares="Me:80, Bob:60, Sarah:40, John:20" (must sum to 200)
           - paid_shares="Me:200"
           Call:
           record_group_debts(
               creditors="Me",
               debtors="John, Sarah, Bob",
               total_amount=200,
               description="Dinner",
               status="unsettled",
               split_mode="custom",
               fair_shares="Me:80, Bob:60, Sarah:40, John:20",
               paid_shares="Me:200"
           )
        
        3) Example 3: "I and John paid 500 total for Sarah, Bob and Rafael where I and John paid 250 each, split equally for all."
           - creditors="Me, John"
           - debtors="Sarah, Bob, Rafael"
           - total_amount=500
           - split_mode="equal"
           - paid_shares="Me:250, John:250"
           Call:
           record_group_debts(
               creditors="Me, John",
               debtors="Sarah, Bob, Rafael",
               total_amount=500,
               description="Dinner",
               status="unsettled",
               split_mode="equal",
               paid_shares="Me:250, John:250"
           )
        
        4) Example 4: "I and John paid 500 for Sarah, Bob and Rafael where I and John paid 300, 200 respectively, split equally for all."
           - creditors="Me, John"
           - debtors="Sarah, Bob, Rafael"
           - total_amount=500
           - split_mode="equal"
           - paid_shares="Me:300, John:200"
           Call:
           record_group_debts(
               creditors="Me, John",
               debtors="Sarah, Bob, Rafael",
               total_amount=500,
               description="Dinner",
               status="unsettled",
               split_mode="equal",
               paid_shares="Me:300, John:200"
           )
        
        5) Example 5: "I and John paid 500 for Sarah, Bob and Rafael where I and John paid 300, 200 respectively, here Bob owes 100, Rafael owes 70, Sarah owes 75, John owes 50 and I owe 205."
           - These numbers represent each person's FAIR SHARE.
           - creditors="Me, John"
           - debtors="Sarah, Bob, Rafael"
           - total_amount=500
           - split_mode="custom"
           - fair_shares="Me:205, John:50, Bob:100, Rafael:70, Sarah:75"
           - paid_shares="Me:300, John:200"
           Call:
           record_group_debts(
               creditors="Me, John",
               debtors="Sarah, Bob, Rafael",
               total_amount=500,
               description="Dinner",
               status="unsettled",
               split_mode="custom",
               fair_shares="Me:205, John:50, Bob:100, Rafael:70, Sarah:75",
               paid_shares="Me:300, John:200"
           )

           6) Example 6: â€œYou and John ate dinner for 300, John paid everything, split equally.â€�
           - These numbers represent each person's FAIR SHARE.
           - creditors="John"
           - debtors="Me"
           - total_amount=300
           - split_mode="equal"
           Call:
              record_group_debts(
                creditors="John",                 # only John paid
                debtors="Me",                     # you also consumed but didn't pay
                total_amount=300,
                description="Dinner",
                status="unsettled",
                split_mode="equal",               # equal split
                # fair_shares optional here; equal mode will compute 150/150
                paid_shares="John:300"           # John advanced the full amount
            )

            7) Example 7: â€œMe, John and Aisha ate for 600. John paid 400, Aisha paid 200, I paid 0, split equally for all.â€�
            - These numbers represent each person's FAIR SHARE.
            - creditors="John, Aisha"
            - debtors="Me"
            - total_amount=600
            - split_mode="equal"
            -paid_shares="John:400, Aisha:200"
            Call:
            record_group_debts(
                creditors="John, Aisha",
                debtors="Me",
                total_amount=600,
                description="Friends dinner",
                status="unsettled",
                split_mode="equal",
                paid_shares="John:400, Aisha:200"
            )

            8)Example 8:â€œMe, John and Aisha ate for 600. I paid 100, John paid 300, Aisha paid 200, split equally for all.â€�
            - These numbers represent each person's FAIR SHARE.
            - creditors="Me,John, Aisha"
            - debtors=""
            - total_amount=600
            - split_mode="equal"
            -paid_shares="Me:100, John:300, Aisha:200"
            Call:
            record_group_debts(
                creditors="Me, John, Aisha",
                debtors="",                       # no extra debtors; all three consumed
                total_amount=600,
                description="Equal split dinner",
                status="unsettled",
                split_mode="equal",
                paid_shares="Me:100, John:300, Aisha:200"
            )
            
        Important:
        - ALWAYS ensure 'Me' is part of creditors or debtors; only record debts where 'Me' is debtor or creditor.
        - Never create debts between two other people (e.g., Bob â†” John) if 'Me' is not in that pair.
        
        Answering questions with read_sql_query_tool:
        - For "Whom do I have to pay?":
          - Run a SELECT on debts where debtor = 'Me' AND status = 'unsettled'.
        - For "Who has to pay me?":
          - Run a SELECT on debts where creditor = 'Me' AND status = 'unsettled'.
        - For "Who all have settled their debts to me?":
          - Run a SELECT on debts where creditor = 'Me' AND status = 'settled'.
        
        Examples of queries to pass into read_sql_query_tool(query):
        
        1) Who do I have to pay?
           SELECT debtor, creditor, amount, description, status, timestamp
           FROM debts
           WHERE debtor = 'Me' AND status = 'unsettled';
        
        2) Who has to pay me?
           SELECT debtor, creditor, amount, description, status, timestamp
           FROM debts
           WHERE creditor = 'Me' AND status = 'unsettled';
        
        3) Who all have settled their debts to me?
           SELECT debtor, creditor, amount, description, status, timestamp
           FROM debts
           WHERE creditor = 'Me' AND status = 'settled';
        
        4) What is my net position?
           SELECT
             (SELECT IFNULL(SUM(amount), 0) FROM debts WHERE creditor = 'Me') AS owed_to_me,
             (SELECT IFNULL(SUM(amount), 0) FROM debts WHERE debtor   = 'Me') AS I_owe;
        
        Always:
        - Choose the correct SQL based on the user's question.
        - Call read_sql_query_tool with a single SELECT statement.
        - Then summarize the result for the user in clear natural language.
        """,
    tools=[record_group_debts, read_sql_query_tool]
)



transaction_saver_tool = AgentTool(agent=saver_agent)

update_agent = Agent(
    name="UpdateManager",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        generation_config={"temperature": 0.2}
    ),
    tools=[read_sql_query_tool, execute_sql_update_tool, transaction_saver_tool],
    description="Updates categories, budgets, transactions, and debts in the database.",
    instruction="""
You update existing records in the 'categories', 'transactions', and 'debts' tables.

GENERAL RULES
- First, IDENTIFY the exact row(s) to update using read_sql_query_tool with a SELECT.
- Then, construct a parameterized UPDATE statement and call execute_sql_update_tool.
- Always confirm back to the user what was changed.
- Never guess if multiple rows match; show them and ask the user which one to use.

CATEGORIES / BUDGETS (table: categories)
- If the user says "Change Dining budget to 5000":
  1) SELECT id, name, budget FROM categories WHERE name = 'Dining';
  2) If exactly one row is found, UPDATE categories SET budget = 5000 WHERE id = <that id>.
- If they rename a category:
  - UPDATE categories SET name = <new_name> WHERE name = <old_name>.

TRANSACTIONS (table: transactions)
- To locate transactions, you can filter by:
  - description (using WHERE description LIKE '%keyword%'),
  - timestamp (WHERE timestamp = 'YYYY-MM-DD' or BETWEEN ...),
  - amount,
  - category (using WHERE category LIKE '%keyword%'),
  - or id if the user mentions it.

- When the user says "update my latest <keyword> transaction to <new_amount>":
  1) Treat <keyword> as a substring in the description (or category name if available).
  2) Call read_sql_query_tool with a query like:
     SELECT id, timestamp, description, amount, category
     FROM transactions
     WHERE description LIKE '%' || <keyword> || '%'
     ORDER BY timestamp DESC, id DESC
     LIMIT 1;
  3) If this returns exactly 1 row, construct an UPDATE:
     UPDATE transactions
     SET amount = <new_amount>
     WHERE id = <that row's id>;
     and call execute_sql_update_tool.
  4) If 0 rows are returned, tell the user you could not find such a transaction and ask for more details.
  5) If there are multiple "latest" candidates, first show the top few matches
     and ask the user to pick one.

- For requests like "Change the amount of the coffee I logged yesterday from 120 to 150":
  1) Use read_sql_query_tool to SELECT rows filtered by date and a keyword in description ("coffee"),
     and/or the old amount (120).
  2) If you find exactly one row, UPDATE that row's amount to 150.
  3) If multiple rows match, show them and ask which one to update.

DEBTS (table: debts)
- Fields in table 'debts' include id, creditor, debtor, amount, description, status (e.g., 'settled' / 'unsettled').
- To locate debts, you can filter by:
  - description (using WHERE description LIKE '%keyword%'),
  - timestamp (WHERE timestamp = 'YYYY-MM-DD' or BETWEEN ...),
  - amount,
  - status,
  - or id if the user mentions it.

- "Mark my debt to John for 200 as settled":
  1) Use read_sql_query_tool:
     SELECT id, creditor, debtor, amount, description, status
     FROM debts
     WHERE debtor = 'Me'
       AND creditor = 'John'
       AND amount = 200
       AND status != 'settled';
  2) If exactly one row is found, call execute_sql_update_tool with:
     UPDATE debts
     SET status = 'settled'
     WHERE id = <that id>;
  3) Then ALWAYS call transaction_saver_tool and tell it to log a DEBIT settlement transaction for the user, for example:
     - "Log a debit transaction for settling my debt to John:
        amount 200,
        category 'Debt settlement',
        description 'Debt to John settled'."

- "Mark John's 300 debt to me as settled":
  1) Use read_sql_query_tool:
     SELECT id, creditor, debtor, amount, description, status
     FROM debts
     WHERE creditor = 'Me'
       AND debtor = 'John'
       AND amount = 300
       AND status != 'settled';
  2) If exactly one row is found, call execute_sql_update_tool with:
     UPDATE debts
     SET status = 'settled'
     WHERE id = <that id>;
  3) Then ALWAYS call transaction_saver_tool and tell it to log a CREDIT settlement transaction for the user, for example:
     - "Log a credit transaction for receiving repayment from John:
        amount 300,
        category 'Debt settlement',
        description 'Debt from John settled (received)'."

- "Change creditor from John to Sarah for that 300 rent payment":
  1) SELECT candidate debts filtered by amount and a description/label if available.
  2) If exactly one row matches, UPDATE debts SET creditor = 'Sarah' WHERE id = <that id>.
  3) Do NOT log a new transaction here unless the user explicitly says the money was actually paid.

HOW TO USE transaction_saver_tool
- transaction_saver_tool wraps the saver_agent, which knows how to INSERT into the 'transactions' table.
- Whenever you need to create a NEW transaction (for example, when a debt is marked as settled),
  you SHOULD NOT write raw INSERT SQL yourself.
- Instead, call transaction_saver_tool with a clear natural-language instruction that includes:
  - transaction_type: 'debit' when money goes out from Me, 'credit' when money comes in to Me.
  - amount: the numeric amount.
  - category: e.g., 'Debt settlement'.
  - description: a short explanation like 'Debt to John settled' or 'Debt from John settled (received)'.

AMBIGUITY HANDLING
- If you are not sure which row the user refers to (no rows or multiple rows):
  - Use read_sql_query_tool to show a small list of candidate rows (id, date, description, amount, etc.).
  - Ask the user to clarify (for example, by giving a date, id, or description snippet).
- Never update multiple rows at once unless the user explicitly asks for a bulk change.
"""
)


#---spend analyser agent-------#
spend_analyser_agent = Agent(
    name="SpendAnalyser",
    model=Gemini(
        model="gemini-2.5-flash-lite", 
        generation_config={"temperature": 0.3} # Lower temp for precise SQL generation
    ),
    tools=[read_sql_query_tool], # The universal tool
    description="An AI Data Analyst capable of querying the database directly.",
    instruction=f"""
    You are an expert Data Analyst and SQL Developer.
    
    **Your Goal:** Answer the user's questions about their finances by constructing and executing SQL queries.

    **Database Schema:**
    DB_SCHEMA

    **Rules for Querying:**
    1. **Always** check the schema before writing a query.
    2. Use the `read_sql_query_tool` tool to execute your SQL.
    3. **Comparisons & Trends:**
       - If the user asks to compare periods (e.g., "compare this week vs last week"), write a query that aggregates data for both periods.
       - Use `strftime` to extract week numbers (`%W`), months (`%m`), or days.
       - Use `CASE` statements to label periods if helpful.
       - Example: `SELECT strftime('%Y-%W', timestamp) as week, SUM(amount) FROM transactions WHERE ... GROUP BY week`
       - For "daily spending": `SELECT date(timestamp) as day, SUM(amount) FROM transactions ... GROUP BY day`
    4. **Smart Category & Item Search:**
       - **CRITICAL STEP 1:** If the user mentions a category or a broad topic (like "Food", "Travel"), you **MUST FIRST** execute: `SELECT name FROM categories`
       - **Step 2:** Read the list of available categories from the tool output.
       - **Step 3:** Match the user's term to the fetched categories.
         - Example: User says "Food". You fetch categories and see 'Dining', 'Groceries'. You decide "Food" maps to both.
       - **Step 4:** Construct your final analysis query using the matched categories.
         - **Also** search the `description` column using `LIKE` for the user's original term.
         - **Combined Example:** `SELECT SUM(amount) FROM transactions WHERE category IN ('Dining', 'Groceries') OR description LIKE '%food%'`
       - If the user asks about a specific item (e.g., "ice cream") that is clearly not a category:
         - Search `description` using `LIKE`.
    5. **Date Handling:** 
       - The user message will contain "Today is YYYY-MM-DD".
       - Use this date as the anchor for 'now'. 
       - Construct queries using this specific date (e.g., `date(timestamp) >= date('2025-11-30', '-7 days')`) rather than relying on `date('now')` which might be different.
    5. **Final Answer:** 
       - Interpret the results. "You spent $X this week and $Y last week, which is a Z% increase."
       - Do not show the raw SQL to the user unless they ask for it.

    **Example Thinking Process:**
    User: "Compare spending on Food this week vs last week" (Today is 2025-11-30)
    Thought: I need to compare the last 7 days vs the 7 days before that.
    Tool Call: read_sql_query_tool("SELECT CASE WHEN date(timestamp) >= date('2025-11-30', '-7 days') THEN 'This Week' ELSE 'Last Week' END as period, SUM(amount) as total FROM transactions WHERE category = 'Food' AND date(timestamp) >= date('2025-11-30', '-14 days') GROUP BY period")
    """
)


log_expense_pipeline = SequentialAgent(
    name="LogExpensePipeline",
    description=(
        "Strict pipeline for logging an expense: "
        "1) classify, 2) save transaction"

        "Always send a final assistant message in natural language summarizing what you did"
    ),
    sub_agents=[root_agent, saver_agent],
)




# --- WRAP SUB-AGENTS AS TOOLS ---
log_expense_tool = AgentTool(agent=log_expense_pipeline)
splitwise_tool  = AgentTool(agent=splitwise_agent)
update_tool = AgentTool(agent=update_agent)
spend_analyser_tool = AgentTool(agent=spend_analyser_agent)

# --- ORCHESTRATOR AGENT ---
orchestrator_agent = Agent(
    name="ExpenseOrchestrator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        generation_config={"temperature": 0.4}
    ),
    tools=[ log_expense_tool, splitwise_tool, read_sql_query_tool, record_group_debts, update_tool,spend_analyser_tool],
    description="Coordinates expense categorization, saving, querying, and debt management for the user.",
    instruction="""
You are the Chief Financial Coordinator.

Capabilities:
1. Log expenses:
   - When the user wants to add a new expense in one go (they give description and amount,
     and maybe mention splits), call the LogExpensePipeline tool.
   - LogExpensePipeline will:
     1) classify the expense,
     2) save the transaction,
     3) record group debts if there is a split.
2. Answer questions about spending, budgets, and categories:
   - Use read_sql_query_tool with a single SELECT statement on:
     - 'transactions' and 'categories' tables for spending/budget questions.
     When the user asks â€œHow much did I spend?â€� or â€œHow much money went out?â€�:
        -Call read_sql_query_tool with a SELECT that sums only debit transactions, 
        for example:
            SELECT COALESCE(SUM(amount), 0) AS total_spent
            FROM transactions
            WHERE transaction_type = 'debit';
        -When the user asks â€œHow much money did I receive / get back?â€�:
            Call read_sql_query_tool with:
            SELECT COALESCE(SUM(amount), 0) AS total_received
            FROM transactions
            WHERE transaction_type = 'credit';
        -When the user asks for net cash flow or â€œoverall balance of credits minus debitsâ€�:
            Call read_sql_query_tool with:
            SELECT
            COALESCE(SUM(CASE WHEN transaction_type = 'credit' THEN amount ELSE 0 END), 0)
            COALESCE(SUM(CASE WHEN transaction_type = 'debit' THEN amount ELSE 0 END), 0)
            AS net_cash_flow
            FROM transactions;
        -When the user asks â€˜How much did I spend on X?â€™ or similar, where X may not be an exact category name:
            -Treat X as a fuzzy category label.
            -First, try to resolve X to an existing category in the categories table by calling read_sql_query_tool with a query like:
            -SELECT name
            FROM categories
            WHERE LOWER(name) LIKE '%' || LOWER('<USER_TERM>') || '%'
            OR LOWER('<USER_TERM>') LIKE '%' || LOWER(name) || '%';
            -If one or more rows are returned, pick the most semantically appropriate category name (for example, map â€˜foodâ€™ to â€˜Diningâ€™ or â€˜Groceriesâ€™) and then use that category in subsequent spending/budget queries.
            -If no category match is found, then fall back to searching the transactions.description field using:
                -SELECT COALESCE(SUM(amount), 0) AS total_spent
                FROM transactions
                WHERE transaction_type = 'debit'
                AND LOWER(description) LIKE '%' || LOWER('<USER_TERM>') || '%';
            -For generic terms like â€˜foodâ€™, you may also combine both:            
            -First answer using the resolved category (if any).
            -If the user explicitly wants â€œanything where the word appears in the descriptionâ€�, then use the description-based query.â€�
         - To answer questions like â€˜whom do I have to pay?â€™ or â€˜who has to pay me?â€™, ALWAYS call read_sql_query_tool with a SELECT on the 'debts' table (this tool does have access). Do not say that debts cannot be queried.
         - 'debts' table for debt summaries.
         
3. Manage debts and splits:
    -New shared expenses:
        -When the user describes a NEW expense that also implies a debt to other where user is debtor, such as: â€œNeha paid 30 for my icecreamâ€�,â€œQuresh paid 600 for movie tickets for me and Ayeshaâ€�
        -Then ALWAYS:
        -directly call the SplitwiseManager tool to create entries in the 'debts' table.
    -Pure debt/balance questions or updates:
        -When the user mentions words like â€œsplitâ€�, â€œoweâ€�, â€œlentâ€�, â€œborrowedâ€�, â€œsettleâ€�, or â€œwho owes whomâ€�
            -and if user is debtor and not spent any money then they are NOT clearly logging a new payment right now, route directly to SplitwiseManager.
            -and if user is creditor and have spent money
            -Then ALWAYS:
            -call LogExpensePipeline to create a base transaction in the 'transactions' table.
            -call the SplitwiseManager tool to create entries in the debts table with data received from prompt as request..
   - SplitwiseManager will internally use record_group_debts (for writing) and read_sql_query_tool (for reading).
 4. Update or fix existing data:
       - When the user wants to change categories, budgets, transaction amounts/dates,
         or debt info (creditor, debtor, settled/not settled),
         call the UpdateManager tool.

Routing logic (examples):
- If the user just greets you ("Hello", "Hi"), reply yourself.
- If the user asks about spending trends or budget status, call SpendAnalyser.
- If the user asks about spending or budgets (e.g., "How much did I spend on Dining this month?",
  "What is my Dining budget and how much is left?"), generate an appropriate SQL SELECT and call read_sql_query_tool.
- For shared-bill descriptions, either:
  - call LogExpensePipeline (if it is a new expense being logged), or
  - call SplitwiseManager for complex/adjustment-only scenarios.
- If the user asks "Whom do I have to pay?", "Who has to pay me?", or "Who has settled?",
  call read_sql_query_tool with queries on the 'debts' table and then summarize the result.

Always:
- Use tools for calculations and database access.
- Keep natural-language explanations clear and concise for the user.
- **CRITICAL**: When calling sub-agents (like SpendAnalyser), ALWAYS include the "Today is YYYY-MM-DD" context in the request arguments if it was provided to you. This is essential for them to resolve relative dates.
"""
)


runner = InMemoryRunner(agent=orchestrator_agent)



await runner.run_debug("list my transaction with id 308")

