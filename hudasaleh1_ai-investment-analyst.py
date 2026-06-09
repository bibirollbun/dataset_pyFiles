!pip install -U -q google-adk

import os
import time
import requests
import re
import json
from google import genai
from google.genai import types
from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool, google_search, ToolContext
from google.adk.agents.callback_context import CallbackContext
from kaggle_secrets import UserSecretsClient
from IPython.display import Markdown, display
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  
# --- CONFIGURATION ---
# We use Gemini 2.0 Flash for stability (20 RPM limit)
MODEL_NAME = "gemini-2.0-flash"

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… API Key loaded from Kaggle Secrets")
except:
    print("âš ï¸� Warning: Could not load secrets. Ensure 'GOOGLE_API_KEY' is set.")

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
print(f"âœ… System Ready. Using Model: {MODEL_NAME}")


def parse_search_results(callback_context: CallbackContext):
    """
    Takes the raw text from the Search Agent, strips Markdown, 
    and saves 'url' and 'company' to the shared State.
    """
    print(f"DEBUG ATTRIBUTES: {dir(callback_context)}")
    # Update variable usage inside the function
    raw_output = callback_context.state.get("search_result", "")
    
    print(f"\nğŸ§¹ [Callback A] Parsing Output: {raw_output[:50]}...")

    # 1. Clean the Markdown
    json_match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
    clean_json_str = json_match.group(1) if json_match else raw_output.strip()
    print(clean_json_str)
    try:
        data = json.loads(clean_json_str)
        
        # 2. Update the State
        # Update variable usage here too
        callback_context.state["company_name"] = data.get("company")
        callback_context.state["pdf_url"] = data.get("url")
        callback_context.state["status"] = data.get("status")
        
        print(f"âœ… State Updated: Found URL for {data.get('company')}")
        
    except json.JSONDecodeError:
        print("â�Œ Error: Could not parse JSON from agent output.")
        callback_context.state["status"] = "ERROR"

InitialSearchAgent = LlmAgent(
    name="InitialSearchAgent",
    model=Gemini(model=MODEL_NAME),
    tools=[google_search],
    output_key="search_result",
    after_agent_callback=parse_search_results,
    instruction="""
    You are an expert Financial Researcher.
    Find the *direct PDF URL* of the latest Annual Report (2024 or 2023) for the user's company.
    
    STRATEGY:
    - Search: "The company provided investor relations annual report 2024 filetype:pdf"
    - If 2024 is not out, find 2023.
    
    OUTPUT FORMAT (JSON ONLY):
    {"status": "FOUND", "url": "https://...pdf", "company": "..."}
    """
)


# --- A. CALLBACK: DOWNLOAD & UPLOAD ---

def setup_file_tools(callback_context: CallbackContext):
    # Check if 'client' is visible
    if 'client' not in globals():
        print("â�Œ CRITICAL ERROR: 'client' variable is missing. Run the setup cell.")
        return

    state = callback_context.state
    pdf_url = state.get("pdf_url")
    company = state.get("company_name", "Unknown")

    if not pdf_url or state.get("status") != "FOUND":
        print("â�­ï¸� No PDF URL found. Agent will run text-only.")
        return

    print(f"\nğŸ“‚ [Callback B] Processing Report for {company}...")
    
    try:
        # 1. Download & Upload (Only if not already done)
        store_name = state.get("file_store_name")
        print(store_name)
        if not store_name:
            # Download
            response = requests.get(pdf_url, headers={'User-Agent': 'Mozilla/5.0'}, allow_redirects=True, timeout=20)
            file_name = f"{company}_report.pdf"
            with open(file_name, 'wb') as f:
                f.write(response.content)

            # Upload
            print(f"Uploading to Gemini File API...")
            file_search_store = client.file_search_stores.create(
                config=types.CreateFileSearchStoreConfig(display_name=f'{company} Store')
            )
            upload_op = client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=file_search_store.name,
                file=file_name,
                config=types.UploadToFileSearchStoreConfig(display_name='Annual Report')
            )
            
            # Wait for processing
            while not (upload_op := client.operations.get(upload_op)).done:
                time.sleep(2)
                print(".", end="")
            
            store_name = file_search_store.name
            state["file_store_name"] = store_name
            print(f"\nâœ… Store Ready: {store_name}")

            
    except Exception as e:
        print(f"â�Œ Callback Error: {e}")

# --- B. TOOL: DYNAMIC CONTEXT ---
def ask_annual_report(tool_context: ToolContext, question: str) -> str:
    """
    Queries the currently loaded Annual Report PDF.
    """
    # 1. Get Store Name from Context (NOT from LLM argument)
    store_name = tool_context.state.get("file_store_name")
    
    # Safety Check
    if not store_name:
        return "Error: No Annual Report file is currently loaded. Please wait for the download."

    print(f"ğŸ”� [Tool] Querying Store '{store_name}': {question[:40]}...")
    
    # Rate Limit Protection
    time.sleep(2) 
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash", # Use 1.5 Flash for better limits
            contents=question,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(file_search_store_names=[store_name])
                )]
            )
        )
        return response.text
    except Exception as e:
        return f"Tool Error: {str(e)}"

print("âœ… Tool 'ask_annual_report' defined.")

# --- C. AGENT DEFINITION ---
analyst_agent = LlmAgent(
    name="FileAnalystAgent",
    model=Gemini(model=MODEL_NAME),
    output_key="company_analysis",
    tools=[FunctionTool(ask_annual_report)],
    before_agent_callback=setup_file_tools,
    instruction="""
    You are a Senior Investment Analyst.
    You have a tool `ask_annual_report` to read the company's report.
    Alway use file_store_name stored in the state as store_name and add fileSearchStores/ before it if not available

    YOUR MISSION:
    You must generate a comprehensive Investment Memo covering exactly these sections:
    1. **Company Overview**: Business model, key markets, and scale.
    2. **Financial Performance**: Full P&L Extraction (Revenue, Gross Profit, Operating Expenses, Net Income, EPS).
    3. **Strategic Execution**: Key strategic pillars and achievements this year.
    4. **Products & Services**: Main revenue drivers and new product launches.
    5. **Governance & Risk**: Board structure and top 3 material risks (e.g., currency, regulation).
    6. **Future Plans**: Guidance for next year and long-term targets.


    EXECUTION PLAN:
    - You must call `ask_annual_report` SEPARATELY for each section to ensure you get full details. 
    - Do not try to get everything in one tool call.
    - For 'Financial Performance', explicitly ask for the "Consolidated Statement of Income" or "Profit and Loss".
    
    OUTPUT FORMAT:
    Produce a clean Markdown report with headers for each section.
    """
)



# Force a 15s break between Analyst and GapFinder to refill quota
def cooldown(callback_context: CallbackContext):
    print("\nâ�³ [System] Cooling down (15s) to manage API Quota...")
    time.sleep(15)


def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}


gap_finder = LlmAgent(
    name="GapFinderAgent",
    model=Gemini(model=MODEL_NAME),
    tools=[FunctionTool(exit_loop)],
    before_agent_callback=cooldown, 
    output_key="gap_analysis",
    instruction="""
    You are a Strict Editor. Review the information in {company_analysis}.
    
    REQUIRED SECTIONS:
    1. Company Overview
    2. Financial Performance (Must have Revenue, Net Income numbers)
    3. Strategic Execution
    4. Products & Services
    5. Governance & Risk
    6. Future Plans

    DECISION LOGIC:
    1. If ALL sections are present and detailed:
       - You MUST call the tool `exit_gap_loop` immediately.
    2. If information is missing:
       - Do NOT call the tool.
       - Output a text list starting with "MISSING:".
       - Example: "MISSING: Specific 2024 Revenue numbers, details on ESG goals."
    """
)

supplemental = LlmAgent(
    name="SupplementalAgent",
    model=Gemini(model=MODEL_NAME),
    tools=[google_search],
    output_key="company_analysis",
    instruction="""
    You are a Researcher. 
    Your goal is to fill the gaps identified in {gap_analysis} and update the report in {company_analysis}.

    INPUTS:
    - Current Report: {company_analysis}
    - Gaps to Fill: {gap_analysis}

    INSTRUCTIONS:
    1. Look at the "MISSING" items in the gap analysis.
    2. Use Google Search to find this specific data.
    3. MERGE your findings into the existing {company_analysis}.
    4. Ensure the output is the FULL, UPDATED report (Markdown format).
    5. Do not delete existing good information, only add/refine.
    """
)

gap_loop = LoopAgent(
    name="GapClosureLoop",
    sub_agents=[gap_finder, supplemental],
    max_iterations=3
)


pipeline = SequentialAgent(
    name="InvestmentPipeline",
    sub_agents=[InitialSearchAgent, analyst_agent, gap_loop]
)
runner = InMemoryRunner(
    agent=pipeline,
    plugins=[
        LoggingPlugin()
    ],  # Add the plugin. Handles standard Observability logging across ALL agents
)
company_to_test = "Etisalat" # <--- CHANGE COMPANY HERE

print(f"ğŸš€ Starting Analysis for: {company_to_test}...")
best_report_content = ""


prompt = f"Create investment meno for the company: {company_to_test}"
response_turns = await runner.run_debug(prompt) 



# --- SMART CAPTURE & CLEANING LOGIC ---

# 1. Get the raw text
text_output = response_turns[-1].content.parts[0].text
    


# --- DISPLAY RESULTS ---
if text_output:
    print("\n" + "="*50)
    print("ğŸ“� FINAL EXTRACTED REPORT")
    print("="*50 + "\n")
    display(Markdown(text_output))
    
    # Save file
    filename = f"{company_to_test}_Investment_Memo.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(best_report_content)
    print(f"\nğŸ’¾ Saved file locally: {filename}")
else:
    print("â�Œ Error: No report text was captured.")

