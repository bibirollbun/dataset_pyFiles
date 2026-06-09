import os
from kaggle_secrets import UserSecretsClient

print(os.listdir("/kaggle/working/"))

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API key setup complete.")
except Exception as e:
    print(
        f"Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import AgentTool, FunctionTool, google_search, load_memory, preload_memory
from google.genai import types


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")


    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )


    if isinstance(user_queries, str):
        user_queries = [user_queries]

 
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")


print("✅ Helper functions defined.")

APP_NAME = "MemoryDemo"
USER_ID = "demo_user"


data_ingestion_agent = Agent(
    name="DataIngestionAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are a Data Ingestion Agent. Your job is to expect questions and/or read a provided dictionary with sales, inventory, 
    and supplier datasets. If you recieved data, clean missing or inconsistent data, and produce structured outputs 
    for downstream agents. Return as a dictionary containing 'sales', 'inventory', 'suppliers'.
    """,
    tools=[],  # Could add file-read or DB tools if available
    output_key="cleaned_data"
)
print("data_ingestion_agent created.")


demand_forecasting_agent = Agent(
    name="DemandForecastingAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Consider the cleaned sales and inventory data provided by: {cleaned_data}
    You are a Demand Forecasting Agent. Using the cleaned sales and inventory data, 
    forecast demand per SKU for the next 3 periods. Provide numerical forecasts with 
    prediction intervals. Return as a dictionary with SKU as key and forecast as value.
    You can us 'google_search' to guide yourself with the best procedure to analyse the data
    """,
    tools=[google_search],  
    output_key="demand_forecasts"
)
print("demand_forecasting_agent created.")


inventory_optimization_agent = Agent(
    name="InventoryOptimizationAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Following this outline strictly: {demand_forecasts}
    You are an Inventory Optimization Agent. Using the demand forecasts and current stock 
    levels, calculate optimal stock levels and safety stock per SKU. Suggest order quantities 
    to minimize holding costs while avoiding stockouts. Return as a dictionary with SKU as key.
    Generate inventory replenishment plans.
    Access your 'memory' if need to, to provide more insight of suggestion you gave in the past and contrast them with 
    current events.
    """,
    tools=[],  # Could include solver tools (PuLP/Gurobi) if integrated
    output_key="inventory_recommendations"
)
print("inventory_optimization_agent created.")


procurement_optimization_agent = Agent(
    name="ProcurementOptimizationAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are a Procurement Optimization Agent. Using the inventory recommendations and supplier 
    data, determine optimal supplier allocation, order quantities, and delivery schedule. Detect 
    anomalies in supplier costs or performance. Return a procurement plan dictionary.
    Access your 'memory' if need to, to provide more insight of suggestion you gave in the past and contrast them with 
    current events. Prioritize suppliers for risk mitigation or compliance.
    """,
    tools=[load_memory],  # Could include anomaly detection or optimization solver tools
    output_key="procurement_plan"
)
print("procurement_optimization_agent created.")


reporting_agent = Agent(
    name="ReportingAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are a Reporting Agent. Collect outputs from {demand_forecasts}, {inventory_recommendations},
    and {procurement_plan}. Generate executive-ready summaries from each topic, with bullet points of data important
    , dashboards, 
    key KPIs, and recommendations in human-readable text.
    Access your 'memory' if need to, to provide more insight of suggestion you gave in the past and contrast them with 
    current events.
    """,
    tools=[load_memory],  # Could include dashboarding tools if integrated
    output_key="report_summary"
)
print("reporting_agent created.")


root_agent = SequentialAgent(
    name="SequentialWorflow",
    sub_agents=[data_ingestion_agent, demand_forecasting_agent, inventory_optimization_agent, procurement_optimization_agent, reporting_agent],
)


print("Sequential Agent created.")


session_service = InMemorySessionService()
memory_service = (InMemoryMemoryService()) 

runner = Runner(
    agent=root_agent,
    app_name="MemoryDemo",
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
)





# Create runner with BOTH services
runner = Runner(
    agent=root_agent,
    app_name="MemoryDemo",
    session_service=session_service,
    memory_service=memory_service,
)

# Run asynchronously
response = await run_session(
    runner,
    "Give me some insight on what you can do.",
    "ID_Conversation:01",  # Session ID
)



session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id="ID_Conversation:01"
)

print("Session contains:")
for event in session.events:
    text = (
        event.content.parts[0].text[:60]
        if event.content and event.content.parts
        else "(empty)"
    )
    print(f"  {event.content.role}: {text}...")


response = await runner.run_debug(
    "Give an example of what you are capable of."
)

