import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Missing 'GOOGLE_API_KEY'. Details: {e}"
    )


from google.adk.agents import Agent, LlmAgent, SequentialAgent, LoopAgent #ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import AgentTool, FunctionTool, google_search, preload_memory #load_memory
from google.genai import types

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams

print("âœ… ADK components imported successfully.")


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
                    print(f"{event.content.role}: > {text}")

print("âœ… Helper functions defined.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


"""
from google import genai
client = genai.Client(api_key=GOOGLE_API_KEY)

# To check available embedding models:
#for m in client.models.list():
#     print(m.name)
"""
model = "gemini-2.5-flash" 


# Github Agent: Its job is to fetch information using MCP.

# Retrieve secrets (GitHub token and repo name)
GITHUB_TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")

github_agent = Agent(
    model=Gemini(
        model=model, 
        #retry_options=retry_config
    ),
    name="github_agent",
    instruction="""Fetch code from GitHub""", 
    tools=[
        McpToolset(
            connection_params=StreamableHTTPServerParams(
                url="https://api.githubcopilot.com/mcp/",
                headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "X-MCP-Toolsets": "all",
                    "X-MCP-Readonly": "true"
                },
            ),
        )
    ],
    output_key="codebase",
)
print("âœ… github_agent defined.")


# Google Search Agent: Its job is to use the `google_search` tool and present findings.
google_search_agent = Agent(
    model=Gemini(
        model=model,
        retry_options=retry_config #TODO
    ),
    name="google_search",
    instruction="""Use Google Search to get and summerize (max 300 words) the up-to-date relevent 
    developer documentation for the detected programming language, platform, or platform specific frameworks of the {codebase}.
    Ouput format:
    Developer Documentation Findings: [...]
    """, 
    output_key="dev_docs_findings",
    tools=[google_search], 
)
print("âœ… google_search_agent defined.")


# Inital Draft Agent creates the first draft based on given task, code and dev docs.
initial_draft_agent = Agent(
    model=Gemini(
        model=model,
        retry_options=retry_config #TODO
    ),
    name="initial_draft_agent",
    instruction="""You are a helpful assistant, perform the given task for provided inputs: 
    - Input Code: {codebase}
    - Developer Documentation Findings: {dev_docs_findings}
    """,
    output_key="current_draft", # Stores the first draft in the state.
)
print("âœ… initial_draft_agent defined.")


# Critic Agent: Its only job is to provide feedback or the approval signal.
critic_agent = Agent(
    name="critic_agent",
    model=Gemini(
        model=model,
        retry_options=retry_config
    ),
    instruction="""You are a constructive critic. Review the input provided below, always try to find flaws.
    Input: {current_draft}
    
    Evaluate the input:
    - If no flaws found, you MUST respond with the exact phrase: "APPROVED"
    - Otherwise, provide 2-3 short, specific, actionable suggestions for improvements.
    \n""",
    output_key="critique",  # Stores the feedback in the state. 
)
print("âœ… critic_agent created.")


# Function used by the Refiner Agent to exit the loop.
def exit_loop():
    "Call this function ONLY when the critique is 'APPROVED', indicating the Evaluate the text is finished and no more changes are needed."
    return {"status": "Approved", "message": "Approved. Exiting refinement loop."}

print("âœ… exit_loop function created.")


# Refiner Agent: Its job is to refine the draft based on critique OR call the `exit_loop` function.
refiner_agent = Agent(
    name="refiner_agent",
    model=Gemini(
        model=model,
        retry_options=retry_config
    ), 
    instruction="""You are a refiner. You have a draft to refine and critique:

    Draft to refine: {current_draft}
    Critique: {critique}
    
    Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
    - OTHERWISE, refine draft to fully incorporate the feedback from the critique (output the refined draft only, no comments or descriptions).
    """,
    output_key="current_draft",  # It overwrites the story with the new, refined version.
    tools=[
        FunctionTool(exit_loop),
    ],  
)
"""
    NOT APPROVED
    Feedback: [...]
    Refined draft: [...]

Output format exmaple: 
{
    "status": "Previous version NOT APPROVED"
    "task_": "Pass along the task summary from the given instructions", 
    "refined_draft": "Applied refinements on original draft based on feedback"
    "dev_docs_findings": "Most critical points/findings to watch out for from developer documentation"
}\n"""
print("âœ… refiner_agent created.")


# Loop Agent: Contains the agents that runs repeatedly: Critic -> Refiner.
story_refinement_loop = LoopAgent(
    name="story_refinement_loop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=3,  # Prevents infinite loops
)

# Root Agent: sequential agent that defines the overall workflow: Initial Draft -> Refinement Loop.
root_agent = SequentialAgent(
    name="story_pipeline",
    sub_agents=[github_agent, google_search_agent, initial_draft_agent, story_refinement_loop],
)

print("âœ… story_refinement_loop and root_agent created.")


# TODO
memory_service = (
    InMemoryMemoryService()
)  # ADK's built-in Memory Service for development and testing
print("âœ… memory_service created")

# Create Session Service
session_service = InMemorySessionService()  # Handles conversations

# Define session and memory constants 
APP_NAME = "DemoApp"
USER_ID = "demo_user"

# Create runner with BOTH services
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
)


STORY = "story"
CODE = "code"

GITHUB_OWNER = UserSecretsClient().get_secret("GITHUB_OWNER")
GITHUB_REPO_NAME = UserSecretsClient().get_secret("GITHUB_REPO")
GITHUB_FILE_PATH = UserSecretsClient().get_secret("GITHUB_FILE_PATH")

# TODO
def get_instuctions(instruction_type = CODE):

    if (instruction_type == CODE):
        task = "Provide a new improved verison of this code (focus on code without comments)."
    else:
        task = "Write a short raoser story based on this code to make developers laugh."

    code_path = f"""
    - Owner: '{GITHUB_OWNER}'
    - Repository: '{GITHUB_REPO_NAME}'
    - Path: '{GITHUB_FILE_PATH}'
    """ 

    instruction = f""" 
    Code Path: {code_path}
    Task: {task}
    """
    return instruction



# Run with instructions to generate improved version of the provided code.
instruction = get_instuctions(instruction_type = CODE)
session_id = "conversation-01"
await run_session(
    runner,
    instruction,
    session_id,
)

session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=session_id
)

await memory_service.add_session_to_memory(session)
print("âœ… Session added to memory!") 


# Memory Agent: It's job is to answer user questions.   
memory_agent = LlmAgent(
    model=Gemini(model=model, retry_options=retry_config),
    name="memory_agent",
    instruction="Answer user questions.",
    tools=[preload_memory],
)
memory_runner = Runner(
    agent=memory_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,  # Enables the memory service
)

# Later improvement - to save after each turn, equip agent with: 
#after_agent_callback=auto_save_to_memory 

#async def auto_save_to_memory(callback_context):
#    """Automatically save session to memory after each agent turn."""
#    await callback_context._invocation_context.memory_service.add_session_to_memory(
#        callback_context._invocation_context.session
#    )
#print("âœ… Callback created.")


# Ask some questions to understand what's memorized:

await run_session(
    memory_runner,
    "What was the last refined version?",
    session_id,
)

await run_session(
    memory_runner,
    "What was the original code version?",
    session_id,
)

await run_session(
    memory_runner,
    "What was the story about in short?", 
    session_id,
)

await run_session(
    memory_runner,
    "Can you give the story a one liner punchy title it deserves?",
    session_id,
)


# Function to see the content of current session
def print_session_contents(session=session):
    print("ğŸ“� Session contains:")
    for event in session.events:
        try:
            text = (
                event.content.parts[0].text[:300]
                if event.content and event.content.parts
                else "(empty)"
            )
            print(f"{event.content.role}: {text}...")
            print("______")
        except Exception as e:
            print("")
            
#print_session_contents(session=session)


# Run with instructions to write a funny story based on the provided code.
instruction = get_instuctions(instruction_type = STORY)
session_id = "conversation-02"
await run_session(
    runner,
    instruction,
    session_id,
)

session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=session_id
)

await memory_service.add_session_to_memory(session)
print("âœ… Session added to memory!")  


# Ask some questions to understand what's memorized:

await run_session(
    memory_runner,
    "What was the last refined version?",
    session_id,
)

await run_session(
    memory_runner,
    "What was the original code version?",
    session_id,
)

await run_session(
    memory_runner,
    "What was the story about in short?", 
    session_id,
)

await run_session(
    memory_runner,
    "Can you give the story a one liner punchy title it deserves?",
    session_id,
)


#print_session_contents(session=session)

