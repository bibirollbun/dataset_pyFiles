# 1.1 Install dependencies
# pip install google-adk (Not needed in Kaggle environment)


#1.2 Configure Gemini API key
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


#1.3 Import ADK components
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")

# 1.4 MODEL CONFIG
# Using Gemini 2.5 Flash-Lite for high speed and low latency
MODEL_NAME = "gemini-2.5-flash-lite" 
print(f"ðŸ¤– Active Model: {MODEL_NAME}")


#1.5 Configure retry options
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

print("âœ… Retry config successful.")


#2.1 Define Tools

from datetime import datetime, timedelta

# --- TOOL 1: GOOGLE SEARCH ---
# Imported directly above as `google_search`. 
# This connects Gemini to live search results.

# --- TOOL 2: DATE CALCULATOR (Custom Python Tool) ---
def date_calculator_tool(start_date_str: str, days: int):
    """
    Calculates a deadline date given a start date and number of days.
    Format start_date_str as 'YYYY-MM-DD'.
    Useful for checking legal deadlines (e.g., 21-day deposit return).
    """
    print(f"ðŸ§® [Tool Triggered] Calculating: {start_date_str} + {days} days")
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = start_date + timedelta(days=days)
        return end_date.strftime("%Y-%m-%d")
    except ValueError:
        return "Error: Date format must be YYYY-MM-DD"

# Wrap the custom function for the ADK
calc_tool_adk = FunctionTool(date_calculator_tool)

print("âœ… Tools Initialized: Google Search & Date Calculator")


# --- AGENT 1: INTAKE SPECIALIST (The "Face" of the system, Can be ADK WebUI)---
# Role: Summarize the messy user input into clean facts.
intake_agent = Agent(
    name="IntakeSpecialist",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    You are an empathetic legal intake specialist for 'CivicAlly'.
    YOUR GOAL: chat with the user to gather these four facts:
    1. ISSUE: (e.g., Security Deposit not returned, Eviction)
    2. DATE_EVENT: (When did the lease end or incident happen?)
    3. AMOUNT: (Currency amount involved)
    4. LOCATION: (City/State for jurisdiction)

    BEHAVIOR:
    - Ask one question at a time. Don't overwhelm the user.
    - If you are missing facts, keep asking.
    - CRITICAL: When (and ONLY when) you have ALL 4 facts, output a final summary starting with the text "CASE_READY:".
    
    Example Final Output:
    CASE_READY:
    - Issue: [specific issue]
    - Date: [YYYY-MM-DD format]
    - Amount:  $[amount]
    - Location: [City, State]
      """,
    
   # Output_key is not required as this will be parsed manually
   
)
print("âœ… intake_agent created.")

# --- AGENT 2: LEGAL RESEARCHER (Back-office) ---
# Role: Find the specific law using Google Search.
research_agent = Agent(
    name="LegalResearcher",
     model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    You are a rigorous legal researcher supporting in the background. You do NOT talk to the user.
    
    INPUT: You will receive a structured case summary with Issue, Date, Amount, and Location. These case facts are provided by the Intake Specialist.
    
    ACTION:
    Use the Google Search tool to find the SPECIFIC civil codes or statutes for the location.
    Example: If California + Deposit, search for 'California Civil Code 1950.5 text'.
    
    Output a summary of the APPLICABLE LAW, citing specific codes found in the search results.
    DO NOT ask questions. Work with the information provided.
    Output format: "Applicable Law: [Code Citation] - [Brief summary]"
    """,
    tools=[google_search], # <--- Native Google Search Grounding
    output_key="legal_research"
)
print("âœ… research_agent created.")

# --- AGENT 3: STRATEGIST & DRAFTER ---
# Role: Calculate deadlines and write the final letter.
drafting_agent = Agent(
    name="StrategistDrafter",
     model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    You are a senior legal aid attorney.
    
    STEPS:
    1. Analyze the case facts against the {legal_research}.
    2. Use the 'date_calculator_tool' to verify if deadlines were missed.
       (e.g., If law says 21 days, calculate [Date_Event + 21 days]).
    3. DRAFT A FORMAL DEMAND LETTER.
    
    The letter should be professional, cite the specific codes found by the researcher, 
    mention the calculated deadline violation, and clearly state the demand.
    DO NOT ask questions. Draft the letter using the provided information.
    """,
    tools=[calc_tool_adk], # <--- Custom Calculator Tool
    output_key="final_draft"
)

print("âœ… drafting_agent created.")

# --- ORCHESTRATION ---
# Link them in a specific order. Use the SequentialAgent when applicable when all case_facts are known. 
# For interactive user session, SequentialAgent does NOT apply as the user interfaces only with intake_agent
# civically_swarm  = SequentialAgent(
   # name="CivicAlly_Swarm",
  #  sub_agents=[intake_agent, research_agent, drafting_agent],
#)

# print(f"âœ… CivicAlly Swarm Ready with {MODEL_NAME}")



# --- USER SCENARIO WITH INTERACTIVE INPUT. Apply Long Running Pause for real user input ---

import asyncio

MANUAL_MODE = True # Set True for Video recording, False for Submission

async def run_smart_session():
    print("ðŸš€ Launching CivicAlly...")
    print("="*50)
    
    # Create a runner for intake agent (maintains conversation state)
    intake_runner = InMemoryRunner(agent=intake_agent)
    
    current_input = "Hi, I need help with a landlord issue."
    
    while True:
        print(f"\nðŸ‘¤ User: {current_input}")
        
        # Use run_debug - THIS WORKS
        response_list = await intake_runner.run_debug(current_input)
        
        # Get the last response
        if response_list:
            last_event = response_list[-1]
            
            # Extract text from the event
            if hasattr(last_event, 'content') and hasattr(last_event.content, 'parts'):
                response_text = last_event.content.parts[0].text
            elif hasattr(last_event, 'text'):
                response_text = last_event.text
            else:
                response_text = str(last_event)
            
            response_text = response_text.strip()
            
            # Check if ready
            if "CASE_READY:" in response_text:
                print(f"\nðŸ¤– CivicAlly:\n{response_text}")
                print("\n" + "="*50)
                print("âœ… Triggering Research & Drafting...")
                print("="*50 + "\n")
                
                # Create back office swarm
                back_office_swarm = SequentialAgent(
                    name="BackOffice",
                    sub_agents=[research_agent, drafting_agent]
                )
                
                # Run back office
                back_runner = InMemoryRunner(agent=back_office_swarm)
                back_results = await back_runner.run_debug(response_text)
                
                # Get final draft
                if back_results:
                    final_event = back_results[-1]
                    
                    if hasattr(final_event, 'content') and hasattr(final_event.content, 'parts'):
                        draft_text = final_event.content.parts[0].text
                    elif hasattr(final_event, 'text'):
                        draft_text = final_event.text
                    else:
                        draft_text = str(final_event)
                    
                    print("\n" + "="*50)
                    print("ðŸ“„ FINAL DEMAND LETTER")
                    print("="*50)
                    print(draft_text)
                
                break
            else:
                print(f"\nðŸ¤– CivicAlly: {response_text}")
        
        # Get next input
        if MANUAL_MODE:
            current_input = input("\n(Your reply) > ")
            if current_input.lower() in ['exit', 'quit']:
                print("\nâœ‹ Session ended.")
                break
        else:
            break

# Execute
await run_smart_session()
          


# --- CREATE SUBMISSION.CSV FILE ---
import pandas as pd
submission_data = {
    'id': ['civically_agent_run'],
    'status': ['success'],
    'message': ['Agent swarm executed and draft generated']
}

df = pd.DataFrame(submission_data)
df.to_csv('submission.csv', index=False)

print("âœ… Created submission.csv to meet Kaggle submission requirements.")

