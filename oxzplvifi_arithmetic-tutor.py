import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


# Multi-agent system components:
from google.genai import types #day-1a
from google.adk.agents import LlmAgent #day-2a
from google.adk.agents import SequentialAgent #day-1b
from google.adk.models.google_llm import Gemini #day-1a

# Custom and built-in tools:
from google.adk.apps.app import App #day-2b
from google.adk.tools import FunctionTool #day-2b
from google.adk.code_executors import BuiltInCodeExecutor #day-2a

# Sessions & Memory:
from google.adk.runners import InMemoryRunner, Runner #day-3a
from google.adk.sessions import InMemorySessionService #day-3b

print("âœ… Imports completed successfully")


APP_NAME = "default"
USER_ID = "default"
SESSION = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

async def run_session(
    runner_instance: Runner,
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

    # Convert the query string to the ADK Content format
    query = types.Content(role="user", parts=[types.Part(text="bring it!")])

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

print("âœ… Helper functions defined.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Agent powered by LLM:
operation_agent = LlmAgent(
    name="OperationAgent",
    model=MODEL_NAME,
    instruction="""You are a random arithmetic operation generator.
    Your task is to generate a random arithmetic operation, making sure it is not repeated in the current session.
    """,
    output_key="random_operation",
    generate_content_config=types.GenerateContentConfig(
        temperature=2,
    ),
)

# Wrap the agent in an App:
operation_app = App(
    name=APP_NAME,
    root_agent=operation_agent,
)


# Create runner with session management:
session_service = InMemorySessionService()
session = await session_service.create_session(
    app_name=APP_NAME, 
    user_id=USER_ID,
    session_id=SESSION,
)
runner = Runner(
    app=operation_app,
    session_service=session_service,
)


response = await runner.run_debug(
    user_messages="I'm ready!",
    user_id=USER_ID, 
    session_id=session.id, 
)


response = await runner.run_debug(
    user_messages="Another one please!",
    user_id=USER_ID, 
    session_id=session.id, 
)


calculation_agent = LlmAgent(
    name="CalculationAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""You are a python-based calculator.
    Your task consists of the following steps:
    1. Write python code for the arithmetic operation: {random_operation}
    2. Print the result.
    """,
    output_key="calculation_result",
    code_executor=BuiltInCodeExecutor(),  # Use the built-in Code Executor Tool. 
)


# Set the session state and feed it to a new session:
session_state = {
    "random_operation":"5+7"
}
session_service = InMemorySessionService()
session = await session_service.create_session(
    app_name=APP_NAME, 
    user_id=USER_ID,
    session_id=SESSION,
    state=session_state
)

# Create runner with session management:
runner = Runner(
    agent=calculation_agent, 
    app_name=APP_NAME,
    session_service=session_service
)


response = await runner.run_debug(
    user_messages="Show me the python code!",
    user_id=USER_ID, 
    session_id=session.id, 
)


async def get_answer() -> str:
    """
    Requests and returns the student's answer to the arithmetic operation.
    """
    answer = input("Answer: ")
    return '{"student_answer": "' + answer + '"}'

# Create the external tool for requesting the student's answer:
answer_tool = FunctionTool(func=get_answer)

request_agent = LlmAgent(
    model=MODEL_NAME,
    name="RequestAgent",
    instruction="""You are a student answer request agent.
    Your task is to request the student to provide his answer to the question:
    1. Use the `answer_tool` to request the answer from the student
    2. Store the answer as 'student_answer' in the session state
    """,
    tools=[answer_tool],
    output_key="answer_requested",
)


runner = InMemoryRunner(agent=request_agent)
_ = await runner.run_debug("I have the answer ready!")


evaluation_agent = LlmAgent(
    model=MODEL_NAME,
    name="EvaluationAgent",
    description="Processes the student's answer and provides final response",
    instruction="""You are an arithmetic tutor.
    Your task is to provide feedback to the student:
    1. Check the 'student_answer' from session state
    2. Parse the answer JSON
    3. Notify the student about the correctness of his answer by comparing to the true answer: {calculation_result}
    4. Provide a step-by-step procedure for carrying out the operation using mental tricks
    """,
    output_key="answer_evaluation",
)

root_agent = SequentialAgent(
    name="StudentAnswerAgent",
    description="Complete workflow for processing arithmetic answers with human oversight",
    sub_agents=[
        operation_agent, 
        request_agent,
        calculation_agent,
        evaluation_agent
    ],
)

session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent, 
    app_name=APP_NAME, 
    session_service=session_service
)


await run_session(runner, SESSION)


await run_session(runner, SESSION)

