!pip install PyPDF2


import os
from kaggle_secrets import UserSecretsClient
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
import json
import requests
import sqlite3
import subprocess
import time
import uuid
import PyPDF2

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import warnings

warnings.filterwarnings("ignore")

print("âœ… ADK components imported successfully.")

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )
    retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
    
)


APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session
MODEL_NAME = "gemini-2.5-flash-lite"

db_url = "sqlite:///my_agent_data.db"  # Local SQLite file




async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")

def readMedicalReport(name:str):
    print("âœ… hello, Fetching report for "+name)
    report="";
    try:
        f = open('../input/medicine/'+name+'.pdf','rb')
        pdf_reader = PyPDF2.PdfReader (f)
        report_length=(len(pdf_reader.pages))
        for index in range(0,report_length):
            report = report+pdf_reader.pages[index].extract_text()
       # page_one_text = page_one.extract_text()
        print("ðŸ“„"+report);
    except:
        report='No Reports found for '+name+'. Make sure you uploaded the report by name \''+name+'.pdf\''
    return report;
    print("âœ… PDF Reader Function created")
    
print(f"ðŸ“„Test: {readMedicalReport('report3')}")

report_vendor = LlmAgent(
    name="report_vendor",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
   instruction="""Your only job is to use the 'readMedicalReport' function based on patient's name obtained from input and then read the returned data 
                 
    """,
    tools=[FunctionTool(readMedicalReport)],
    output_key="report_data",
)

cardio_agent = LlmAgent(
    name="cardio_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
   instruction="""You are a specialized research agent for cardiovascular health. Your only job is 
                  to read the provided data {report_data} and  to analyse/interpret the same using 
                  google search, Explain each parameter obtained from {report_data} that is relevant to cardiology and Highlight any alarming/critical data in the input,
                If no cardiology specific parameters are present, Return ' No tests determinental to cardiology found'
       If no abnormal parameters found, Return 'All cardiology related parameters normal'
    """,
    tools=[google_search],
    output_key="cardiology_analysis",
)

print("ðŸ¤– Cardiology agent created")

urology_agent = LlmAgent(
    name="urology_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
   instruction="""You are a specialized research agent in field of urology. Your only job is 
                  to read the provided data {report_data} and  to analyse/interpret the same using 
                  google search, Explain each parameter obtained from {report_data} that is relevant to urology and Highlight any alarming/critical data in the input,
                If no urology specific parameters are present, Return ' No tests determinental to urology found'
                If no abnormal parameters found, Return 'All urology releated parameters normal'
    """,
    tools=[google_search],
    output_key="urology_analysis",
)

print("ðŸ¤– Urology agent created")

pulmonology_agent = LlmAgent(
    name="pulmonology_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
   instruction="""You are a specialized research agent in field of pulmonology. Your only job is 
                  to read the provided data {report_data} and  to analyse/interpret the same using 
                  google search, Explain each parameter obtained from {report_data} that is relevant to pulmonology and Highlight any alarming/critical data in the input,
                If no pulmonology specific parameters are present, Return ' No tests determinental to pulmonology found'
                If no abnormal parameters found, Return 'All pulmonology releated parameters normal'
    """,
    tools=[google_search],
    output_key="pulmonology_analysis",
)

print("ðŸ¤– Pulmonology agent created")

nurology_agent = LlmAgent(
    name="nurology_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
   instruction="""You are a specialized research agent in field of nurology. Your only job is 
                  to read the provided data {report_data} and  to analyse/interpret the same using 
                  google search, Explain each parameter obtained from {report_data} that is relevant to nurology and Highlight any alarming/critical data in the input,
                If no nurology specific parameters are present, Return ' No tests determinental to nurology found'
                If no abnormal parameters found, Return 'All nurology releated parameters normal'
    """,
    tools=[google_search],
    output_key="nurology_analysis",
)

print("ðŸ¤– Nurology agent created")

aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these report interpretations findings into a single interpreted health summary:

    **Cardiology:**
    {cardiology_analysis}
    
    **Urology:**
    {urology_analysis}
    
    **Pulmonology:**
    {pulmonology_analysis}

    **Nurology:**
    {nurology_analysis}
    
    Your summary should highlight health concerns and suggest corrective actions to be taken""",
    output_key="health_summary",
)

print("ðŸ¤– aggregator_agent created.")

dietitian_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="dietitian_agent",
    description="Independent agent to suggest diet based on input specifications",
    instruction="""
    You are a specialist dietitian, Use google search to recommend diet based on {health_summary}. 
    """,
    tools=[google_search],
    output_key="dietitian_recommendations",
)

meal_planner_agent = Agent(
    name="MealPlannerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Read the provided diet recomendation: {dietitian_recommendations}
Create a meal plan based on {dietitian_recommendations}, Make the cusine Indian by default unless specified otherwise
in {dietitian_recommendations}. """,
    output_key="final_summary",
)

print("ðŸ¤– Meal_Planner_agent created.")

medico_team = ParallelAgent(
    name="MedicoTeam",
    sub_agents=[cardio_agent, urology_agent, pulmonology_agent, nurology_agent],
)

system_agent = SequentialAgent(
    name="system_agent",
    sub_agents=[report_vendor,medico_team, aggregator_agent,dietitian_agent,meal_planner_agent],
)

print("ðŸ¤– System Agent Created.")

medical_app_compacting = App(
    name="medical_app_compacting",
    root_agent=system_agent,
    # This is the new part!
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 invocations
        overlap_size=1,  # Keep 1 previous turn for context
    ),
)

db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

# Create a new runner for our upgraded app
medical_runner_compacting = Runner(
    app=medical_app_compacting, session_service=session_service
)

print("âœ… root_agent created.")
session_service = DatabaseSessionService(db_url=db_url)
runner = Runner(agent=system_agent, app_name=APP_NAME, session_service=session_service)
print("âœ… Runner created.")

response = await run_session(
    medical_runner_compacting,
    [
      "Hello! patient's name is sam"
    ],
    "session-0B",

)


