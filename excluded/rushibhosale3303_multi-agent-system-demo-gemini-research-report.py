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


!pip install -q google-genai numpy pandas nest_asyncio


import warnings
warnings.filterwarnings('ignore')


# 1nstall necessary libraries (e.g., google-genai, google-adk if used, or LangGraph)
# !pip install google-genai ...

# 2. Securely load API Key from Kaggle Secrets
import os
from kaggle_secrets import UserSecretsClient
os.environ['GOOGLE_API_KEY'] = UserSecretsClient().get_secret('GEMINI_API_KEY') 
# The Google Search tool is usually enabled automatically if the key is valid.

from google import genai
client = genai.Client()


def format_report(research_data: dict) -> str:
    """Custom tool: Formats the gathered research data into a structured Markdown report."""
    report = "## FINAL RESEARCH REPORT\n\n"
    for agent_name, summary in research_data.items():
        report += f"### {agent_name.replace('Agent', ' Findings')}\n"
        report += summary  # Assumes summary is already formatted text
        report += "\n---\n"
    report += "### Conclusion & Synthesis\n"
    # The Report Agent LLM will add the synthesis after this tool is called, 
    # or the tool can be designed to call the LLM again for the final synthesis.
    return report

# ... (Register this as a tool for the Report Agent)


# --- Setup and Authentication ---
!pip install -q google-genai numpy pandas

import os
import asyncio
import json
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient

# 1. Securely load API Key from Kaggle Secrets
try:
    os.environ['GOOGLE_API_KEY'] = UserSecretsClient().get_secret('GEMINI_API_KEY')
    print("âœ… Gemini API Key loaded from Kaggle Secrets.")
except Exception as e:
    print(f"âš ï¸� Authentication Error: Please add 'GEMINI_API_KEY' to your Kaggle secrets. Details: {e}")

# 2. Initialize Gemini Client
client = genai.Client()
MODEL = 'gemini-2.5-flash'
print(f"âœ… Gemini Client initialized using model: {MODEL}")

# --- Global State Management (Simulating InMemorySessionService) ---
# Sessions & State Management: Used by the Coordinator to track progress.
session_state = {
    "user_query": "",
    "sub_tasks": [],
    "raw_research_results": {},
    "final_report": "",
    "long_term_memory_check": False
}

# Long Term Memory (Simulated): Simple in-memory storage of past reports
long_term_memory = {} # {query_hash: final_report_text}


# --- Custom Tools and Utility Functions ---

# Custom Tool: Report Formatting
def format_report(research_data: dict, topic: str) -> str:
    """
    Custom Tool: Formats the gathered research data into a structured Markdown report.
    This demonstrates a custom tool/function that an LLM-powered agent can call.
    """
    print("ğŸ› ï¸� Custom Tool 'format_report' executed.")
    report = f"# Research Report: {topic}\n\n"
    report += "## Summary of Findings\n\n"

    for agent_name, summary in research_data.items():
        report += f"### {agent_name.replace('Agent', ' Findings')}\n"
        report += f"> {summary}\n\n" # Uses blockquote for clean separation
    
    report += "---\n\n"
    report += "## Synthesis (To be completed by the Report Agent)\n\n"
    report += "The agent will now use the collected findings above to write the final synthesized conclusion."
    
    return report

# Function to get the declaration for the Report Agent
def get_format_report_declaration():
    # This declaration is in the required OpenAPI schema format for the Gemini API
    return types.FunctionDeclaration(
        name="format_report",
        description="Formats the combined research findings from all Data Agents into a structured Markdown report template for final synthesis.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "research_data": types.Schema(
                    type=types.Type.OBJECT,
                    description="A dictionary where keys are agent names (e.g., 'Market Agent') and values are their compacted research summaries."
                ),
                "topic": types.Schema(
                    type=types.Type.STRING,
                    description="The original user query/topic of the report."
                )
            },
            required=["research_data", "topic"]
        )
    )

print("âœ… Custom Tool 'format_report' defined and declared.")


# --- Agent Definitions and Orchestration (Final, Final Corrected Version) ---

import asyncio
import json
from google.genai import types

# =======================================================================
# AGENT EXECUTION FUNCTION (Used by Parallel Agents)
# =======================================================================

async def run_data_agent_task(agent_name: str, client: genai.Client, prompt: str, config: types.GenerateContentConfig) -> tuple[str, str]:
    """
    Runs a single Data Agent asynchronously. It calls the synchronous client method 
    and relies on the concurrency manager (asyncio.gather) to handle parallel execution.
    """
    print(f"  [START] Running {agent_name}...")
    
    # FIX: Use the synchronous client.models.generate_content(...) method
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config
    )
    
    summary = response.text.strip()
    print(f"  [FINISH] {agent_name} completed. Summary length: {len(summary)} chars.")
    
    return agent_name, summary


# =======================================================================
# COORDINATOR AGENT / WORKFLOW ORCHESTRATOR
# (LLM-Powered Agent, handles Sequential and Parallel Flow)
# =======================================================================

async def run_multi_agent_system(user_query: str):
    """The main orchestration function (Coordinator Agent)."""
    
    global session_state
    session_state["user_query"] = user_query
    
    print(f"## 1. Coordinator Agent: Initial Request & Memory Check")
    
    # Long Term Memory Check (Simple Hash Check)
    query_hash = hash(user_query)
    if query_hash in long_term_memory:
        session_state["long_term_memory_check"] = True
        print(f"ğŸ’¾ Found existing report in Long Term Memory! Skipping research.")
        return long_term_memory[query_hash]
        
    session_state["long_term_memory_check"] = False # Reset for this run
    print(f"ğŸ”­ No existing report found. Beginning research workflow.")

    # --- Step 1: LLM-Powered Agent (Coordinator) plans the work ---
    sub_tasks = {
        "Market Agent": f"Latest market size, growth trends, and key competitors for {user_query}",
        "Policy Agent": f"Recent government regulations, incentives, and investment trends affecting {user_query}"
    }
    session_state["sub_tasks"] = sub_tasks
    print(f"  Planning: Broke task into {len(sub_tasks)} parallel research branches.")


    # --- Step 2: Parallel Agents Execution ---
    print(f"\n## 2. Parallel Agents: Running Research Concurrently")
    
    tasks = []
    for agent_name, topic in sub_tasks.items():
        # Configuration for Data Agents
        data_agent_config = types.GenerateContentConfig(
            # Tools: Built-in Google Search enabled
            tools=[{"google_search": {}}], 
            system_instruction=f"""
            You are the specialized **{agent_name}**. Your goal is to perform research on: **{topic}**.
            1. Use the 'google_search' tool to find 3-5 high-quality, relevant search snippets.
            2. **Context Engineering (Compaction):** Synthesize the search results into a concise, factual summary of no more than 150 words.
            3. Return ONLY the final compacted summary text. Do not add any conversational text or formatting outside the summary.
            """
        )
        tasks.append(run_data_agent_task(
            agent_name, 
            client, 
            prompt=f"Research the following: {topic}",
            config=data_agent_config
        ))

    # Execute all research tasks in parallel
    results = await asyncio.gather(*tasks)
    
    # Collect results into session state
    for agent_name, summary in results:
        session_state["raw_research_results"][agent_name] = summary


    # --- Step 3: Sequential Agent Execution (Report Agent) ---
    print(f"\n## 3. Sequential Agent: Report Generation")
    
    # Configuration for Report Agent
    report_agent_config = types.GenerateContentConfig(
        # Tools: Custom Tool 'format_report' provided via its declaration
        tools=[get_format_report_declaration()],
        system_instruction=f"""
        You are the **Report Agent**. Your goal is to synthesize the findings from the Data Agents into a final, professional report.
        1. First, you MUST call the 'format_report' tool with the full research data to structure the report template.
        2. Once the template is provided, write the **Conclusion & Synthesis** section, drawing insights ONLY from the provided research data.
        3. The final output must be the complete, well-formatted Markdown report.
        """
    )
    
    # 3a. Trigger Custom Tool (format_report) - A2A Protocol / Function Calling
    tool_call_prompt = (
        "Take the following research data and format it into a report, then write a synthesis. "
        f"Research Data: {json.dumps(session_state['raw_research_results'])}"
    )
    
    # FIX: Use the synchronous client.models.generate_content(...)
    response = client.models.generate_content( 
        model=MODEL,
        contents=tool_call_prompt,
        config=report_agent_config
    )
    
    # Handle Tool Call Execution
    if response.function_calls:
        fc = response.function_calls[0]
        print(f"  Report Agent is calling Custom Tool: {fc.name}(...)")
        
        # Execute the custom tool locally
        tool_output = format_report(**dict(fc.args), topic=user_query)

        # 3b. Send Tool Output back to LLM to complete synthesis (Sequential Step)
        # FIX: Use the synchronous client.models.generate_content(...)
        final_response = client.models.generate_content( 
            model=MODEL,
            contents=[
                types.Part.from_function_response(
                    name="format_report",
                    response={"report_template": tool_output} # Pass the structured template back
                )
            ]
        )
        final_report = final_response.text
        
    else:
        # Fallback if the agent doesn't use the tool
        print("  Report Agent did not call the custom tool. Proceeding with direct generation.")
        final_response = response
        final_report = final_response.text

    session_state["final_report"] = final_report
    
    # --- Step 4: Long Term Memory Update ---
    long_term_memory[query_hash] = final_report
    print(f"\nğŸ’¾ Final report saved to Long Term Memory.")
    
    return final_report


# Helper to run the async function in a notebook environment
import nest_asyncio
try:
    nest_asyncio.apply()
except Exception:
    print("\nâš ï¸� Note: If you encounter asyncio errors, ensure 'nest_asyncio' is installed.")
    pass 

def generate_report(query: str):
    """
    Runs the multi-agent system asynchronously using a notebook-compatible method.
    """
    # Use the existing event loop if running, or start a new one
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(run_multi_agent_system(query))
        else:
            return asyncio.run(run_multi_agent_system(query))
    except Exception:
        # Final fallback
        return asyncio.run(run_multi_agent_system(query))
        
print("âœ… Agent and Orchestration functions defined. Notebook-compatible execution enabled.")


# ---Execute the Multi-Agent System ---

USER_REQUEST = "Analyze the growth potential of solar power in Southeast Asia for 2026."

print(f"====================================================================")
print(f"DEMO START: Multi-Agent Research System")
print(f"Query: {USER_REQUEST}")
print(f"====================================================================")

final_output = generate_report(USER_REQUEST)

print("\n\n" + "-"*70)
print("âœ… FINAL GENERATED REPORT")
print("-"*70)
print(final_output)

# --- Observability / Debugging ---
# Logging/Tracing: Print the final state variables for verification
print("\n--- Session & State Check ---")
print(f"User Query: {session_state['user_query']}")
print(f"Long Term Memory Check: {session_state['long_term_memory_check']}")
print(f"Raw Research Collected: {list(session_state['raw_research_results'].keys())}")
print(f"Long Term Memory Count: {len(long_term_memory)}")


# --- Demo: Demonstrate Long-Term Memory ---
print("\n--- DEMO: Long Term Memory / Sequential Check ---")
USER_REQUEST_2 = "Analyze the growth potential of solar power in Southeast Asia for 2026." # Same query

print(f"Re-running the exact same query: '{USER_REQUEST_2}'...")
report_from_memory = generate_report(USER_REQUEST_2)

if session_state['long_term_memory_check']:
    print("âœ… Long Term Memory SUCCESS: The system avoided research and returned the saved report.")

