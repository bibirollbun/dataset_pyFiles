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
import time
from typing import Dict, Any,List, Callable
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import logging
from kaggle_secrets import UserSecretsClient
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}") 


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import google_search
print("âœ… ADK components imported successfully.")


#logging.basicConfig(level=logging.INFO, format='%(message)s')
##logger = logging.getLogger(__name__)

#RETRY_CONFIG = types.HttpRetryOptions(
#    attempts=3,
#    exp_base=2,
#    initial_delay=1,
#    http_status_codes=[429, 500, 503],
##)

# Global Statistics Counter
stats = {
   "total_cases": 0,
    "assigned_to_human": 0,
    "resolved_by_main_agent": 0,
    "search_queries_run": 0
}


# This allows the code to run anywhere without specific Kaggle environment dependencies.
class Gemini:
    def __init__(self, model: str, retry_options=None, description=None, instruction=None, tools=None):
        self.model = model
        self.description = description
        self.instruction = instruction
        self.tools = tools or []

class Agent:
    def __init__(self, name: str, model, description: str, instruction: str, tools: List[Callable]):
        self.name = name
        self.model = model
        self.description = description
        self.instruction = instruction
        self.tools = {t.__name__: t for t in tools}

class Runner:
    def __init__(self, app_name: str, agent: Agent, session_service=None):
        self.agent = agent

    def run(self, session_id: str, message: str):
        # Simple parser to extract priority and question from the message string
        # This mimics the LLM's tool calling ability for this demo
        class Response:
            def __init__(self, text): self.text = text

        if self.agent.name == "main_agent":
            # Simulating the LLM parsing the input "PRIORITY=X, QUESTION='Y'"
            try:
                # Extract priority
                p_start = message.find("PRIORITY=") + 9
                p_end = message.find(",", p_start)
                priority = int(message[p_start:p_end].strip())
                
                # Extract question
                q_start = message.find("QUESTION='") + 10
                q_end = message.rfind("'")
                question = message[q_start:q_end]
                
                # Call the tool directly (Simulating Agent behavior)
                result = self.agent.tools["screening_tool"](priority, question)
                return Response(result["result"])
            except Exception as e:
                return Response(f"Error parsing input: {e}")

        elif self.agent.name == "summary_agent":
            # Simulating the LLM calling summary_tool
            stats = self.agent.tools["summary_tool"]()
            
            # Constructing the text response based on the stats
            summary_text = (
                "ğŸ“Š **DASHBOARD SUMMARY**\n"
                f"- Total cases processed: {stats['total_cases']}\n"
                f"- Auto-resolved by agents: {stats['resolved_by_main_agent']}\n"
                f"- Escalated to human support: {stats['assigned_to_human']}\n"
                f"- Resolution rate: {(stats['resolved_by_main_agent'] / stats['total_cases'] * 100):.1f}%" 
                if stats['total_cases'] > 0 else "0%"
            )
            return Response(summary_text)
            
        return Response("Unknown Agent")

class InMemorySessionService:
    pass

class AgentTool:
    def __init__(self, agent): self.agent = agent


def google_search(query: str) -> Dict[str, Any]:
    # Simple deterministic mock logic based on string length
    if len(query) % 2 == 0:
        return {
            "status": "success",
            "answer": f"âœ… Found: {query}\nSource: https://example.com/search?q={query}"
        }
    else:
        return {"status": "not_found", "answer": None}

def format_customer_email(query: str, answer: str) -> str:
    """Formats professional email response."""
    return (
        f"ğŸ“§ FORMATTED EMAIL RESPONSE\n"
        f"==========================\n"
        f"Subject: Your Query Resolution - [{query[:50]}...]\n\n"
        f"Dear Customer,\n\n"
        f"Thank you for contacting support.\n"
        f"We investigated your query and found:\n\n"
        f"ğŸ’¡ {answer}\n\n"
        f"If you need further assistance, please reply to this email.\n\n"
        f"Best regards,\n"
        f"Automated Support Agent\n"
        f"support@company.com"
    )



   def p0_handler(case: Dict[str, Any]) -> str:
    """P0: Always assign to human support - no search needed."""
    stats["assigned_to_human"] += 1
    return (
        "ğŸš¨ PRIORITY P0 - CRITICAL CASE ğŸš¨\n"
        "As case is critical, assigning this case to Customer Support Personnel.\n"
        "Ticket ID: CSP-Assigned"
    )

def p1_handler(case: Dict[str, Any]) -> str:
    """P1: Search immediately, resolve if found, else escalate."""
    query = case.get("question", "No question provided")
    search_result = google_search(query)

    if search_result["status"] == "success" and search_result["answer"]:
        stats["resolved_by_main_agent"] += 1
        email_body = format_customer_email(query, search_result["answer"])
        return f"âœ… P1 AUTO-RESOLVED\n{email_body}"
    else:
        stats["assigned_to_human"] += 1
        return (
            "â�Œ P1: Not able to find answer\n"
            "As answer not found, assigning this ticket to Customer Support Personnel.\n"
            "Ticket ID: CSP-Assigned"
        )

def p2_handler(case: Dict[str, Any]) -> str:
    """P2: Wait 24hrs (simulated), then search."""
    print("â�³ P2: Simulating 24hr wait (0.5sec for demo)...")
    time.sleep(0.5)  # SIMULATED

    query = case.get("question", "No question provided")
    search_result = google_search(query)

    if search_result["status"] == "success" and search_result["answer"]:
        stats["resolved_by_main_agent"] += 1
        email_body = format_customer_email(query, search_result["answer"])
        return f"âœ… P2 AUTO-RESOLVED (after delay)\n{email_body}"
    else:
        stats["assigned_to_human"] += 1
        return (
            "â�Œ P2: Not able to find answer after 24hr wait\n"
            "Assigning this ticket to Customer Support Personnel.\n"
            "Ticket ID: CSP-Assigned"
        )




def screening_tool(priority: int, question: str) -> Dict[str, Any]:
    """Routes cases to P0/P1/P2 handlers based on priority."""
    global stats
    stats["total_cases"] += 1  

    case = {"priority": priority, "question": question}
    
    if priority == 0:
        result = p0_handler(case)
    elif priority == 1:
        result = p1_handler(case)
    elif priority == 2:
        result = p2_handler(case)
    else:
        result = f"âš ï¸� Unknown priority {priority} - Defaulting to human support"
        stats["assigned_to_human"] += 1

    return {"result": result, "priority": priority}

def summary_tool() -> Dict[str, Any]:
    """Returns case handling statistics."""
    return stats.copy()


main_agent = Agent(
    name="main_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",  
    ),
    description="Routes customer support cases by priority to specialized sub-agents",
    instruction="Call screening_tool(priority, question) exactly once per case.",
    tools=[screening_tool],
)


summary_agent = Agent(
    name="summary_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
    ),
    description="Generates summary statistics of case handling",
    instruction="Call summary_tool() and format a professional dashboard summary.",
    tools=[summary_tool],
)


def run_demo():
    session_service = InMemorySessionService()
    main_runner = Runner("app", main_agent, session_service)
    summary_runner = Runner("summary", summary_agent, session_service)

    test_cases = [
        {"priority": 0, "question": "System completely crashed - production down"},
        {"priority": 1, "question": "How to reset admin password?"},
        {"priority": 2, "question": "Upgrade plan next month"},
        {"priority": 1, "question": "Database timeout error XYZ123"},
    ]

    # List to hold data for CSV
    csv_data = []

    print("ğŸ�¯ RUNNING SUPPORT TICKET AGENT")
    print("=" * 70)

    for i, case in enumerate(test_cases, 1):
        print(f"\nğŸ“‹ CASE #{i}: Priority={case['priority']}")
        user_input = f"PRIORITY={case['priority']}, QUESTION='{case['question']}'"
        
        # Run Agent
        response = main_runner.run(f"case_{i}", user_input)
        print(response.text)
        
        # Collect data for CSV
        csv_data.append({
            "Case_ID": i,
            "Priority": case['priority'],
            "Question": case['question'],
            "Agent_Response": response.text.replace("\n", " | ") # Flatten newlines for clean CSV
        })

    # Generate Summary
    print("\n" + "="*70)
    summary_response = summary_runner.run("summary_1", "Generate summary")
    print(summary_response.text)

    # --- CSV WRITING BLOCK ---
    csv_filename = "support_report.csv"
    print(f"\nğŸ’¾ SAVING OUTPUT TO {csv_filename}...")
    
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["Case_ID", "Priority", "Question", "Agent_Response"])
            writer.writeheader()
            writer.writerows(csv_data)
            
            # Optional: Append summary stats at the bottom of CSV
            writer.writerow({}) # Empty row for spacing
            writer.writerow({"Case_ID": "SUMMARY STATS"})
            writer.writerow({"Case_ID": "Total Cases", "Priority": stats['total_cases']})
            writer.writerow({"Case_ID": "Escalated", "Priority": stats['assigned_to_human']})
            writer.writerow({"Case_ID": "Resolved", "Priority": stats['resolved_by_main_agent']})
            
        print("âœ… CSV file created successfully.")
        
    except Exception as e:
        print(f"â�Œ Error writing CSV: {e}")

if __name__ == "__main__":
    run_demo()


