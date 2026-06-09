import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API key setup complete. Let's crush this!")
except Exception as e:
    print(f"Authentication error: Ensure you have an API key in Kaggle Secrets. Details: {e}") 


from google.adk.agents import LlmAgent, Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search, AgentTool, FunctionTool, ToolContext
from google.adk.tools import load_memory, preload_memory
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import DatabaseSessionService


print("ADK components imported successfully!")


import warnings
warnings.filterwarnings("ignore")


retry_config = types.HttpRetryOptions(
    attempts = 5,      # max tries
    exp_base = 7,      # delay multiplier
    initial_delay = 1, # seconds before 1st retry
    http_status_codes = [429, 500, 503, 504] # retry if these errors are seen
)

print("Retry config done!")


async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
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


print("Helper functions defined.")


def get_step_count_for_day(method: str) -> dict:
    """
    This function contains the database of the user's step count for each day of the week.
    It is possible to add the number of steps across several days by calling the function iteratively with a different 'method' or day of week each time and getting the sum of the results.
    """
    step_database = {
        "monday": 5500,
        "tuesday": 6700,
        "wednesday": 7400,
        "thursday": 8020,
        "friday": 4869,
        "saturday": 10101,
        "sunday": 11143
    }

    steps = step_database.get(method.lower())
    if steps is not None:
        return {"status": "success", "step_count": steps}
    else:
        return {
            "status": "error",
            "error_message": f"Step count for {method} not found!",
        }

print("Step lookup function created! :)")
print(f"Test: {get_step_count_for_day('Monday')}")


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

print("Callback created.")


# From Agents Intensive Course hands-on notebooks
calculation_agent = LlmAgent(
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


# Sources Agent: Finds 3 to 5 trusted sources on the topic at hand.
sources_agent = Agent(
    name="SourcesAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""

    Use the google_search too to find relevant resources from trusted sites. 
    Write about what they have to say about the current topic.
    Include your citations.
    
    """,
    tools = [google_search],
    output_key="trusted_sources",  
)

print("Sources agent created.")


# Analysis Agent: Check what the sources agree on and what they disagree about.
analysis_agent = Agent(
    name="AnalysisAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    1. Analyze the content of {trusted_sources}.
    2. Write about what they mostly agree on in detail.
    3. Write about their disagreements in detail.
    """,
    output_key="agent_analysis",  
)

print("Analysis agent created.")


# Summary Agent: Summarizes and polishes the draft from the analysis agent.
summary_agent = Agent(
    name="SummaryAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    
    Summarize this: {agent_analysis}
    Keep it under 200 words.
    Fix grammatical errors and improve the flow for clarity.
    
    """,
    output_key="research_summary", 
)

print("Summary agent created.")


research_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[sources_agent, analysis_agent, summary_agent],
)

print("Sequential Agent created.")


fitness_buddy = LlmAgent(
    name = "FitBud",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = """ 
    
    You are a helpful accountability partner for a fitness enthusiast.
    The user's goal is to walk 10,000 steps per day. 
    Perform the following to help achieve this goal:
    1. You MUST use the 'load_memory' tool BEFORE all user interactions. This will help you answer questions.
    2. If the user is interested in a certain day, use the tool 'get_step_count_for_day()' to get their step count on that day. 
    3. If the user asks for walking statistics, provide the total, average, and standard deviation in easy to read format.
    4. If calculations are necessary, use the calculation_agent tool for accurate computations. 
    5. If the user asks about an illness or injury, use the research_agent tool to get medical information. Underscore the importance of consulting a doctor.
    6. Evaluate the amount they were able to accomplish. If they underperformed, give advice on how to improve. Provide encouragement.
    
    """,
    tools = [load_memory, get_step_count_for_day, AgentTool(agent=calculation_agent), AgentTool(agent=research_agent)],
    after_agent_callback = auto_save_to_memory,
)

print("Fitness buddy defined!")


APP_NAME = "MyFitBud"  
USER_ID = "scout"   
SESSION = "default"   

MODEL_NAME = "gemini-2.5-flash-lite"

session_service = InMemorySessionService()
memory_service = (
    InMemoryMemoryService()
)


runner = Runner(agent = fitness_buddy, 
                app_name = APP_NAME, 
                session_service = session_service,
                memory_service = memory_service,
               )

print("Runner created! Let's orchestrate!")
print("Now with memory support!")


await run_session(
    runner, ["Hi, FitBud! My name is Rose and I like to walk.", "Hello! What's my name?"],
    "test-db-session-01",      
)


await run_session(
    runner, ["What did we talk about earlier?"],
    "test-db-session-02",      
)


await run_session(
    runner, ["How did I do on Thursday?"],
    "test-db-session-01",      
)


await run_session(
    runner, ["Can you give me my stats for the week?"],
    "test-db-session-01",      
)


await run_session(
    runner, ["What do you think about boosting my goal to 20,000 steps per day?"],
    "test-db-session-02",      
)


await run_session(
    runner, ["Does walking faster give better results"],
    "test-db-session-02",      
)


await run_session(
    runner, ["I'm sick. Is it okay if I don't walk much today?"],
    "test-db-session-02",      
)


await run_session(
    runner, ["I twisted my ankle and now the injured feet is swollen. When can I resume walking?"],
    "test-db-session-02",      
)

