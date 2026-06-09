# --------------------------------------------------------------------------------
# 0. INSTALL & IMPORTS
# --------------------------------------------------------------------------------
import os
import sys
import subprocess
import time
import json
import threading
from datetime import datetime, timedelta

def install_dependencies():
    """
    Ensures required packages are installed.
    If installation fails (e.g. no internet), we just print a warning
    instead of crashing the notebook.
    """
    try:
        import google.generativeai
        import duckduckgo_search
        print("âœ… Dependencies already installed.")
    except ImportError:
        print("ğŸ“¦ Installing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "google-generativeai", "duckduckgo-search"
            ])
            print("âœ… Installation complete.")
        except subprocess.CalledProcessError as e:
            print("âš ï¸� Installation failed. This may be due to internet / env restrictions.")
            print("   You or a teammate can re-run installation later.")
            # Do NOT re-raise, so the notebook continues

install_dependencies()

import google.generativeai as genai

# Try to import duckduckgo_search, but don't crash if it's missing
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    DDGS = None
    print("âš ï¸� duckduckgo_search is not installed in this environment. "
          "search_tool will use a dummy fallback response.")




# --------------------------------------------------------------------------------
# 1. CONFIGURATION: GOOGLE API KEY
# --------------------------------------------------------------------------------

GOOGLE_API_KEY = None

# Try Kaggle Secrets first
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
except Exception:
    # If not running on Kaggle or secret not found, ignore
    pass

# Fallback: environment variable
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("âš ï¸� Warning: GOOGLE_API_KEY not set. "
          "Set it in Kaggle Secrets or as an environment variable before running the workflow.")



# --------------------------------------------------------------------------------
# 2. TOOLS: SEARCH + DATE CALCULATOR
# --------------------------------------------------------------------------------

def search_tool(query: str) -> str:
    """
    Searches the web with a strict timeout to prevent hanging.
    If duckduckgo_search is not available, returns a fallback message.
    """
    if not DUCKDUCKGO_AVAILABLE:
        # Fallback behaviour when the library isn't installed
        return (
            "âš ï¸� Real web search is not available in this environment.\n"
            f"[FALLBACK DATA]: Assuming standard tourist visa requirements for: {query}"
        )

    results_container = {"data": None}

    def run_search():
        try:
            # Using max_results=3 for better coverage
            results = DDGS().text(query, max_results=3)
            results_container["data"] = results
        except Exception as e:
            results_container["error"] = str(e)

    # Run search in a separate thread with 5s timeout
    t = threading.Thread(target=run_search)
    t.start()
    t.join(timeout=5)

    if t.is_alive():
        return (
            "âš ï¸� Search timed out (Kaggle network restriction). \n"
            f"[FALLBACK DATA]: Assuming standard tourist visa requirements for {query}."
        )

    if results_container.get("data"):
        return "\n".join(
            [f"- {r['title']}: {r['body']}" for r in results_container["data"]]
        )

    return "No search results found."




# --------------------------------------------------------------------------------
# 3. AGENT CLASS
# --------------------------------------------------------------------------------

class Agent:
    def __init__(self, name: str, model_name: str = "gemini-2.5-flash", system_instruction: str = ""):
        self.name = name
        self.model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction
        )


    def run(self, input_context: str) -> str:
        """
        Sends the input context to the underlying model and returns text output.
        If an error occurs, returns a string describing the error.
        """
        try:
            response = self.model.generate_content(input_context)
            return response.text
        except Exception as e:
            return f"Agent Error: {str(e)}"



# --------------------------------------------------------------------------------
# 4. INITIALIZE SPECIALIZED AGENTS
# --------------------------------------------------------------------------------

researcher_agent = Agent(
    name="Researcher Agent",
    system_instruction=(
        "You are an expert Immigration Researcher. "
        "Given the user context, output ONLY the search query string that should "
        "be used to find official visa processing information."
    )
)

specialist_agent = Agent(
    name="Visa Specialist Agent",
    system_instruction=(
        "You are a Visa Consultant. "
        "Analyze the search results and extract 'visa_type' and 'processing_days' (int) as JSON.\n"
        "CRITICAL RULES:\n"
        "- If you cannot find specific processing times or visa types, return 'processing_days': -1.\n"
        "- Do NOT guess 0.\n"
        "- Output MUST be valid JSON."
    )
)

concierge_agent = Agent(
    name="Concierge Agent",
    system_instruction=(
        "You are a Travel Concierge.\n"
        "- Write a final friendly response in Markdown.\n"
        "- If processing_days is -1: Tell the user that official info wasn't verified, "
        "but usually it's safer to apply 3-4 weeks early.\n"
        "- If processing_days > 0: Use the provided deadline info and give the user "
        "a clear recommendation of when to apply."
    )
)



# --------------------------------------------------------------------------------
# 5. CORE LOGIC: RUNNING THE MULTI-AGENT WORKFLOW
# --------------------------------------------------------------------------------

def run_agent_workflow(citizenship: str, destination: str, travel_date_str: str):
    """
    Orchestrates the Visa Logic Agent workflow end-to-end.
    Prints intermediate steps and the final Markdown report.
    """
    if not GOOGLE_API_KEY:
        print("â�Œ Error: GOOGLE_API_KEY not found.")
        print("ğŸ‘‰ FIX: Go to 'Add-ons' -> 'Secrets' -> Add 'GOOGLE_API_KEY'.")
        return

    genai.configure(api_key=GOOGLE_API_KEY)

    # Parse Date
    try:
        travel_date = datetime.strptime(travel_date_str, "%Y-%m-%d")
    except ValueError:
        print("âš ï¸� Error: Invalid Date Format. Please use YYYY-MM-DD.")
        return

    print("\n" + "="*60)
    print(f"ğŸš€ STARTING AGENT WORKFLOW: {citizenship} -> {destination}")
    print("="*60)

    # 1. Researcher
    print(f"\nğŸ”� [Researcher Agent] Formulating search strategy...")
    prompt_1 = (
        f"User is a {citizenship} citizen traveling to {destination}. "
        "What should we search to find the official visa processing time?"
    )
    search_query = researcher_agent.run(prompt_1).strip()
    print(f"   ğŸ‘‰ Query: {search_query}")

    # 2. Tool: Web Search
    print(f"\nğŸŒ� [Tool] Searching the web...")
    search_results = search_tool(search_query)

    # Truncate for display if too long
    display_results = (search_results[:200] + '...') if len(search_results) > 200 else search_results
    print(f"   ğŸ‘‰ Found: {display_results}")

    # 3. Specialist Agent
    print(f"\nğŸ§  [Specialist Agent] Analyzing policies...")
    prompt_2 = (
        f"Analyze results regarding {destination} visa for {citizenship} citizens:\n"
        f"{search_results}\n"
        "Extract visa type and processing time (days) as JSON."
    )
    json_response = specialist_agent.run(prompt_2)

    visa_details = {"processing_days": -1}
    try:
        json_clean = json_response.replace("```json", "").replace("```", "").strip()
        visa_details = json.loads(json_clean)
        print(f"   ğŸ‘‰ Extracted: {visa_details}")
    except Exception:
        print(f"   âš ï¸� Parsing Error. Raw output: {json_response}")

    # 4. Timeline / Date Logic
    print(f"\nğŸ“… [Timeline Tool] Calculating application window...")
    deadline_info = "Info unavailable / Manual check required."

    proc_days = visa_details.get("processing_days")
    if proc_days is None:
        proc_days = -1

    if proc_days > 0:
        travel_date_formatted = travel_date.strftime("%Y-%m-%d")
        deadline_info = date_calculator_tool(travel_date_formatted, proc_days)
        print(
            f"   ğŸ‘‰ Logic: Travel Date ({travel_date_formatted}) - "
            f"({proc_days} days processing + 7 days buffer)"
        )
    elif proc_days == -1:
        deadline_info = "âš ï¸� DATA MISSING. Could not verify processing time."
        print("   ğŸ‘‰ Result: Could not calculate exact date due to missing data.")

    # 5. Concierge Agent
    print(f"\nâœ�ï¸� [Concierge Agent] Drafting final report...")
    prompt_3 = (
        f"User: {citizenship}->{destination} on {travel_date_str}. "
        f"Details: {visa_details}. Deadline: {deadline_info}. "
        "Write final response."
    )
    final_report = concierge_agent.run(prompt_3)

    print("\n" + "="*60)
    print("âœ… FINAL AGENT REPORT")
    print("="*60)
    print(final_report)



# --------------------------------------------------------------------------------
# 6. DEMO CALL
# --------------------------------------------------------------------------------

# Example: Indian citizen travelling to Brazil in ~45 days
example_citizenship = "Indian"
example_destination = "Brazil"
example_travel_date = (datetime.today() + timedelta(days=45)).strftime("%Y-%m-%d")

run_agent_workflow(example_citizenship, example_destination, example_travel_date)





