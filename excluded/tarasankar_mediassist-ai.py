%pip install google-adk


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import AgentTool, FunctionTool, google_search, ToolContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

print("âœ… ADK components imported successfully.")


from kaggle_secrets import UserSecretsClient
secret_label = "GOOGLE_API_KEY"
secret_value = UserSecretsClient().get_secret(secret_label)


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


import pydantic, json
import logging, asyncio, time
from datetime import datetime


log_file = "syslog.log"
log_file_json = "syslog.jsonl"

if os.path.exists(log_file):
    os.remove(log_file)

if os.path.exists(log_file_json):
    os.remove(log_file_json)

logging.basicConfig(
    # filename="logger.log",
    filename=log_file,
    level=logging.WARNING,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.WARNING)
f_handler = logging.FileHandler(log_file)
f_handler.setLevel(logging.WARNING)
logger.addHandler(f_handler)

def log_json(data, fname=log_file_json):
    with open(fname, "a") as f:
        f.write(json.dumps(data, default=str) + "\n\n")


async def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> None | LlmResponse:
    logger.info(f"[before_model] Agent: {callback_context.agent_name}; Request: {llm_request}")
    log_json({
        "event": "before_model",
        "agent": callback_context.agent_name,
        "request": repr(llm_request),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # time.time()
    })
    return None  

async def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> None | LlmResponse:
    logger.info(f"[after_model] Agent: {callback_context.agent_name}; Response: {llm_response}")
    log_json({
        "event": "after_model",
        "agent": callback_context.agent_name,
        "response": repr(llm_response),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # time.time()
    })
    return None  

async def after_agent_callback(callback_context: CallbackContext) -> None | types.Content:
    logger.debug(f"[after_agent] Agent: {callback_context.agent_name} finished run")
    log_json({
        "event": "after_agent",
        "agent": callback_context.agent_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # time.time()
    })
    return None  


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

print("Retry configurations created")





# Patient Service Agent: Its job is to process the user text and extract the user's symptoms and location
patient_service_agent = Agent(
    name="PatientServiceAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized Patient Service agent. Your only job is to process the
    input text, extract the user location and disease symptoms and save the user location using the 
    set_user_location tool. 

    DO NOT ANSWER ANY OTHER QUESTION. Please inform the user politely in such a case.
    
    The 'location' information should contain details pertaining to at least 'City' and 'Country'. If 
    that level of details is not provided, try to find the same using the google_search tool.
    
    In case of successful data extraction, the output should be a JSON containing three keys 'status', 
    'location' and 'symptoms' which contain status of the information extraction operation, user 
    location information and disease symptom information respectively.

    Some examples are as follows:

    1. {"status": "Success", "location": "Gariahat, Kolkata, West Bengal, India", "symptoms": "diarrhea, vomiting, and abdominal pain"}

    2. {"status": "Success", "location": "Near Juhu beach, Mumbai, Maharashtra, India", "symptoms": "swelling, vomiting, nausea"}    
    
    ERROR: If either of the 'location' or 'sympton' information is not present, then STOP 
    processing further and please generate the following JSON:

    {"status": "Error","message": "Please provide detailed information regarding your location and symptoms"}
    
    """,
    tools=[google_search],
    output_key="patient_details",  # The result of this agent will be stored in the session state with this key.
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    # before_tool_callback=before_tool_callback,
    # after_tool_callback=after_tool_callback,    
    after_agent_callback=after_agent_callback,    
)

print("âœ… patient_service_agent created.")





# Diagnostician Agent: Its job is to use the google_search tool and find disease matching the disease.

diagnostician_agent = Agent(
    name="DiagnosticianAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized medical diagnostician agent. Your only job is to use the
    google_search tool to find the most likely disease matching the given symptoms in {patient_details}. 
    
    YOUR OUTPUT SHOULD BE JUST THE DISEASE NAME AND NOTHING ELSE. In case the input 
    string does not contain any information pertaining to medical symptoms, please inform the 
    user accordingly""",
    tools=[google_search],
    output_key="disease",  # The result of this agent will be stored in the session state with this key.
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    # before_tool_callback=before_tool_callback,
    # after_tool_callback=after_tool_callback,    
    after_agent_callback=after_agent_callback,    
)

print("âœ… diagnostician_agent created.")





# Physician Agent: Its job is to use the google_search tool and find best drugs for the 
# given disease.

physician_agent = Agent(
    name="PhysicianAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized physician agent. Your only job is to use the
    google_search tool to find the best 3 medicines for the given disease {disease}. 
    
    Your output should be just the names of the names of the medicines and nothing else
    """,
    tools=[google_search],
    output_key="medicines",  
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    # before_tool_callback=before_tool_callback,
    # after_tool_callback=after_tool_callback,    
    after_agent_callback=after_agent_callback,     
)

print("âœ… physician_agent created.")





# Store Locator Agent: Its job is to use the google_search tool and find most popular drug stores in  
# the given location.

store_locator_agent = Agent(
    name="StoreLocatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized drug store locator agent. Your only job is to use the
    google_search tool to find the 3 most popular and big medicines shops in the location given 
    in {patient_details}. 
    
    Your output should be just the names of the drug stores and nothing else
    """,
    tools=[google_search],
    output_key="drug_stores",  
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    # before_tool_callback=before_tool_callback,
    # after_tool_callback=after_tool_callback,    
    after_agent_callback=after_agent_callback,     
)

print("âœ… store_locator_agent created.")





# Clinic Locator Agent: Its job is to use the google_search tool and find most popular clinicss in  
# the given location.

clinic_locator_agent = Agent(
    name="ClinicLocatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized clinic locator agent. Your only job is to use the
    google_search tool to find the 3 most popular and big medical clinics present in the location given 
    in {patient_details}. 
    
    Your output should be just the names of the clinics and nothing else
    """,
    tools=[google_search],
    output_key="clinics",  
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    # before_tool_callback=before_tool_callback,
    # after_tool_callback=after_tool_callback,    
    after_agent_callback=after_agent_callback,     
)

print("âœ… clinic_locator_agent created.")





# The ParallelAgent runs all its sub-agents simultaneously.
parallel_medical_assist_team = ParallelAgent(
    name="ParallelMedicalAssistTeam",
    sub_agents=[physician_agent, store_locator_agent, clinic_locator_agent],
)





def get_current_time() -> dict:
    """
    Returns the current timestamp 

    Args:
        None.

    Returns:
        Dictionary with status and timestamp.

        Examples:
        
        Success: {'status': 'success', 'current_timestamp': '2025-12-01 02:57:52'}
        Error: {"status": "error", "error_message": "Operation could not be performed"}
    """

    try:                
        return {"status": "success", "current_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except:        
        return {"status": "error", "error_message": "Operation could not be performed"}


# The AggregatorAgent runs after the parallel step to synthesize the results.
aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Use `get_current_time()` to get the current timestamp. You are forbidden to 
    generate any timestamp information yourself and must use the result from the tool.
    
    Combine the following findings into a single summary for the patient:

    **Possible Diagnosis:**
    {disease}
    
    **Corresponding medicines:**
    {medicines}
    
    **Nearby Drug Stores:**
    {drug_stores}

    **Nearby Clinics:**
    {clinics}    
    
    Your summary should have 4 distinct sections - 
    
    a) Possible Diagnosis
    b) Corresponding medicines
    c) Nearby Drug Stores
    d) Nearby Clinics
    
    It should also highlight that these findings were derived based on the user provided input 
    mentioned in {patient_details}. It should also urge the user that they should consult a 
    physician at the earliest opportunity.

    You must also mention the date and time at which the report is generated. For this purpose, 
    you must use the current timestamp information obtained from the 'get_current_time' tool
    """,
    output_key="user_summary",  
    tools=[get_current_time],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    # before_tool_callback=before_tool_callback,
    # after_tool_callback=after_tool_callback,    
    after_agent_callback=after_agent_callback,    
)

print("âœ… aggregator_agent created.")





# This SequentialAgent defines the high-level workflow: processes the user input first, then runs the 
# parallel team and finally runs the aggregator

root_agent = SequentialAgent(
    name="CompleteMediAssistSystem",
    sub_agents=[patient_service_agent, diagnostician_agent, parallel_medical_assist_team, aggregator_agent],
)





async def make_runner(agent):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="my_app", user_id="user1", session_id="sess1"
    )
    runner = Runner(
        agent=agent,
        app_name="my_app",
        session_service=session_service,
    )
    return session, runner


async def run_agent(query: str):
    session, runner = await make_runner(root_agent)
    user_msg = types.Content(role="user", parts=[types.Part(text=query)])
    events = runner.run_async(user_id="user1", session_id="sess1", new_message=user_msg)

    async for event in events:
        # Print final agent response
        if event.is_final_response():
            text = event.content.parts[0].text
            print("FINAL RESPONSE:", text)





await run_agent("swelling of legs, fatigue, nausea, shortness of breath, very less urination, checking from Quadra2, Hadapsar")


await run_agent("High temperature, cough, checking from Gariahat")




