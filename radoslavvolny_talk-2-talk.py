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


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Agent 1: Its job is to answer questions, use the google_search tool if needed and present findings. He will have session memory available.
agent_respondent = Agent(
    name="Respondent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a discussion partner in a discussion. You are polite and answer the question. No unnecessary reasoning, just a simple, concise answer, as happens in a human-like conversation. 
    Listen carefully, be chatty. If you don't know something, search the internet, you have a tool available for that.""",
    tools=[google_search],
    output_key="answer_1",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… agent_respondent created.")


# Agent 2: Its job is to start a discussion, ask questions. He will have session memory available.
agent_interviewer = Agent(
    name="Interviewer",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Read the provided phrase: {answer_1} and continue conversation. Create a concise response. Do not hesitate to ask questions, as it happens in a human-like conversation.""",
    output_key="answer_2",
)

print("âœ… agent_interviewer - a starter, created.")


# Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
root_agent = Agent(
    name="ChatCoordinator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # This instruction tells the root agent HOW to use its tools (which are the other agents).
   instruction="""You are a discussion coordinator. Your goal is to start and observe a discussion between two participants on the topic from a user and orchestrate a flow. 
   You are also responsible for displaying this dialogue to a user.
1. First, you MUST call the `Respondent` tool and give them a greeting and the question from the user. You MUST display the question you asked and a response as a screenplay.
2. After receiving the answer_1, you MUST call the `Interviewer` tool to get his reaction to this answer. You MUST display the answer starting the line with "Interviewer:".
3. Continue this way. TAKE the Answer answer_2 from `Interviewer` and you MUST CALL `Respondent` tool with exactly that answer. You MUST display the answer, starting the line with "Respondent:". You MUST continue to step 2 - asking 'Interwiever' again.
DO NOT display any other of your comments. 
LIMIT: do not go for more than 20 rounds of questioning.""",    
    # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
    tools=[AgentTool(agent_respondent), AgentTool(agent_interviewer)],
)

print("âœ… root_agent created.")


runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug(
    "Terrible weather tomorrow in Madrid. Why is that?"
)

