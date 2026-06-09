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


from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


recipe_resolver_agent = Agent(
    name="recipe_resolver",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer resolve recipes.",
    instruction="You are a helpful assistant cooking assistant. Use Google Search for resolving recipes.",
    tools=[google_search],
)

print("âœ… Recipe Resolver Agent defined.")


shopping_list_agent = Agent(
    name="shopping_list",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can order order cooking ingredients",
    instruction="Provide list cooking ingredients, group them by shop and provide approximate prices.",
    tools=[google_search],
)

print("âœ… Shopping List Agent defined.")


root_agent = SequentialAgent(
    name="CookingAssistant",
    sub_agents=[recipe_resolver_agent, shopping_list_agent],
)
print("âœ… Cooking Assistant Agent defined.")


runner = InMemoryRunner(
    agent=root_agent,
    plugins=[
        LoggingPlugin()
    ]
)

print("âœ… Runner created.")


response = await runner.run_debug(
    ["I want to prepare hamburgers for dinner"]
)

