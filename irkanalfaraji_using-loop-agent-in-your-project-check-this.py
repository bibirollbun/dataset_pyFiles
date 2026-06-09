import os
try:
    from kaggle_secrets import UserSecretsClient
    IS_KAGGLE = True
except ImportError:
    IS_KAGGLE = False
    print("Not in Kaggle, load key manually.")
    
try:
    if IS_KAGGLE:
        GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    else:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
    if not GOOGLE_API_KEY:
        # Fallback for local run without .env but with env var set manually
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        
    if GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        print("âœ… Authentication complete.")
    else:
        print("âš ï¸� Error: GOOGLE_API_KEY not found. Please set it in Kaggle Secrets or your local .env file.")
except Exception as e:
    print(f"âš ï¸� Error: {e}. Please ensure 'GOOGLE_API_KEY' is in your Kaggle Secrets.")


import os
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool
from google.genai import types

# Setup Retry
retry_config=types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504],
)

print("âœ… ADK components imported.")


# Define the exit_loop function
def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
    message="APPROVED! LoopAgent should exit now!"
    print(f"ğŸ”´ exit_loop() called! Returning status='approved', message={message}")
    return {"status": "approved", "message": message}

print("âœ… exit_loop function created.")

# Create Agents

# 1. Initial Writer (Dummy)
initial_writer_agent = Agent(
    name="InitialWriterAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Write a very short sentence.",
    output_key="current_story"
)

# 2. Critic (Always Approves to force exit)
critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a critic. Always reply with 'APPROVED' to test the exit mechanism."""
    """You MUST output output 'APPROVED'.""",
    output_key="critique"
)

# 3. Refiner (Calls exit_loop if Approved)
refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a refiner.
    Critique: {critique}
    IF the critique is 'APPROVED', you MUST call the `exit_loop` function.
    OTHERWISE, rewrite the story.""",
    output_key="current_story",
    tools=[FunctionTool(exit_loop)]
)

print("âœ… Agents created.")

# Setup Loop
story_refinement_loop = LoopAgent(
    name="StoryRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=5  # Set high enough to see if it loops unnecessarily
)

root_agent = SequentialAgent(
    name="StoryPipeline",
    sub_agents=[initial_writer_agent, story_refinement_loop]
)

print("âœ… Loop setup complete.")

# Run the loop
runner = InMemoryRunner(agent=root_agent)
print("ğŸš€ Starting execution...")
response = await runner.run_debug("Start test")
print("ğŸ�� Execution finished.")


from google.adk.tools import exit_loop   # <- Import ADK's built_in exit_loop function

# Remove user defined exit_loop
# def exit_loop():
#     """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
#     print("ğŸ”´ exit_loop() called! Returning status='approved'")
#     return {"status": "approved", "message": "Story approved. Exiting refinement loop."}

#print("âœ… Using ADK's built-in exit_loop function.")

# Create Agents

# Create Agents

# 1. Initial Writer (Dummy)
initial_writer_agent = Agent(
    name="InitialWriterAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Write a very short sentence.",
    output_key="current_story"
)

# 2. Critic (Always Approves to force exit)
critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a critic. Always reply with 'APPROVED' to test the exit mechanism."""
    """You MUST output output 'APPROVED'.""",
    output_key="critique"
)

# 3. Refiner (Calls exit_loop if Approved)
refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a refiner.
    Critique: {critique}
    IF the critique is 'APPROVED', you MUST call the `exit_loop` function.
    OTHERWISE, rewrite the story.""",
    output_key="current_story",
    tools=[FunctionTool(exit_loop)]    # Using ADK's built-in exit_loop function that actually triggers exit condition
)

print("âœ… Agents created.")

# Setup Loop
story_refinement_loop = LoopAgent(
    name="StoryRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=5  # Set high enough to see if it loops unnecessarily
)

root_agent = SequentialAgent(
    name="StoryPipeline",
    sub_agents=[initial_writer_agent, story_refinement_loop]
)

print("âœ… Loop setup complete.")

# Run the loop
runner = InMemoryRunner(agent=root_agent)
print("ğŸš€ Starting execution...")
response = await runner.run_debug("Start test")
print("ğŸ�� Execution finished.")

