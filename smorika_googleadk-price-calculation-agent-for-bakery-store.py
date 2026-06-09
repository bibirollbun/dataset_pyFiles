!pip install google-adk==1.18.0


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Setup and authentication complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor
from typing import Any, Dict
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner

print("✅ ADK components imported successfully.")


def show_python_code_and_result(response):
    for i in range(len(response)):
        # Check if the response contains a valid function call result from the code executor
        if (
            (response[i].content.parts)
            and (response[i].content.parts[0])
            and (response[i].content.parts[0].function_response)
            and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])


print("✅ Helper functions defined.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Custom tool for getting item price

def get_item_price_method(method: str) -> dict:
    """Looks up the price of items for a given user input.

    This tool simulates looking up a bakery's product item prices based on the name of the items provided by the user.

    Args:
        method: The names of the product items. It should be descriptive,
                e.g., "sausage roll" or "salt bread" or "coffee" or "iced coffee".

    Returns:
        Dictionary with status and item price information.
        Success: {"status": "success", "item price": 400}
        Error: {"status": "error", "error_message": "product item not found"}

    """
    # This simulates looking up a bakery's product item prices

    price_database = {
        "sausage roll": 200,
        "salt bread": 100,
        "croissant": 200,
        "pizza toast": 400,
        "matcha bun": 400,
        "coffee": 150,
        "hot coffee": 150,
        "iced coffee": 150
    }

    price = price_database.get(method.lower())
    if price is not None:
        return {"status": "success", "item price": price}
    else:
        return {
            "status": "error",
            "error_message": f"product item '{method}' not found in the store",
        }


print("✅ item price lookup function created")
print(f"Test: {get_item_price_method('Matcha bun')}")


print(f"Test: {get_item_price_method('hot coffee')}")


print(f"Test: {get_item_price_method('croissant')}")


# Custom tool for getting the tax rate for the purchase type

def get_tax_rate(tax_type: str) -> dict:
    """Looks up and returns the tax type and rate for the specific item purchase.

      Args:
          tax_type: purchase type of either eat in or take home/take away
          (e.g., "eat in" or "take home" or "take away")

      Returns:
          Dictionary with status and tax rate information.
          Success: {"status": "success", "rate": 0.1}
          Error: {"status": "error", "error_message": "Unknown tax type"}
    """

    rate_database = {
        "eat in": 1.10,
        "stay here": 1.10,
        "eat here": 1.10,
        "for here": 1.10,
        "to go": 1.08,
        "take away": 1.08,
        "take home": 1.08

    }

    # Return structured result with status
    rate = rate_database.get(tax_type.lower())

    # print(rate)
    if rate is not None:
        return {"status": "success", "rate": rate}
    else:
        return {
            "status": "error",
            "error_message": f"Unsupported tax type: {tax_type}",
        }

print("✅ Tax rate function created")
print(f"Test: {get_tax_rate('Eat in')}")


print(f"Test: {get_tax_rate('Take home')}")


# Calculation agent with 2 custom function tools

calculation_agent = LlmAgent(
    name="calculation_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a smart price calculation assistant.

    For total price calculation requests:

    1. Use `get_item_price_method()` to find item price
    2. Use `get_tax_rate()` to get the correct tax rate for purchase type
    3. Check the "status" field in each tool's response for errors
    4. Calculate the total price after tax with multiplying the item price with tax rate based on the output from 'get_item_price_method ` and `get_tax_rate` methods and provide a clear breakdown.
    5. First, state the final total price.
        Then, explain how you got that result by showing the intermediate amounts. Your explanation must include: the price of the item and
        what tax rate was applied based on the purchase type for the final price.

    If any tool returns status "error", explain the issue to the user clearly.
    """,
    tools=[get_item_price_method, get_tax_rate],
)

print("✅ Calculation agent created with 2 custom function tools")
print("🔧 Available tools:")
print("  • get_item_price_method - Looks up item price")
print("  • get_tax_rate - Gets the consumption tax rate based on the purchase type")


# Test 1

calculation_runner = InMemoryRunner(agent=calculation_agent)
_ = await calculation_runner.run_debug(
    "I want to take away one matcha bun, please."
  )


# Test 2

calculation_runner = InMemoryRunner(agent=calculation_agent)
_ = await calculation_runner.run_debug(
    "I want to take away two coffee, please."
  )


# Define helper functions that will be reused throughout the notebook

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


print("✅ Helper functions defined.")


# Define scope levels for state keys (following best practices)

USER_NAME_SCOPE_LEVELS = ("temp", "user", "app")

# The 'user:' prefix indicates this is user-specific data.

def save_userinfo(
    tool_context: ToolContext, user_name: str, country: str
) -> Dict[str, Any]:
    """
    Tool to record and save user name and country in session state.

    Args:
        user_name: The username to store in session state
        country: The name of the user's country
    """
    # Write to session state using the 'user:' prefix for user data
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country

    return {"status": "success"}


# This demonstrates how tools can read from session state.

def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user name and country from session state.
    """
    # Read from session state
    user_name = tool_context.state.get("user:name", "Username not found")
    country = tool_context.state.get("user:country", "Country not found")

    return {"status": "success", "user_name": user_name, "country": country}


print("✅ Tools created.")


# Calculation agent with custom tool for session state management

# Configuration
APP_NAME = "default"
USER_ID = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

calculation_agent = LlmAgent(
    name="calculation_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a smart price calculation assistant.

    For total price calculation requests:

    Tools for managing user context:

    1. To record username and country when provided use `save_userinfo` tool.
    2. Use `get_item_price_method()` to find item price
    3. Use `get_tax_rate()` to get the correct tax rate for purchase type
    4. Check the "status" field in each tool's response for errors
    5. Calculate the total price after tax with multiplying the item price with tax rate based on the output from 'get_item_price_method ` and `get_tax_rate` methods and provide a clear breakdown.
    6. First, fetch username and country using `retrieve_userinfo` tool and make a short greeting such as 'Hi Sam from Canada, thank you for your purchase.'
        Next, state the final total price.
        Then, explain how you got that result by showing the intermediate amounts. Your explanation must include: the price of the item and
        what tax rate was applied based on the purchase type for the final price.

    If any tool returns status "error", explain the issue to the user clearly.
    """,
    tools=[get_item_price_method, get_tax_rate, save_userinfo, retrieve_userinfo],
)

# Set up session service and runner
session_service = InMemorySessionService()
runner = Runner(agent=calculation_agent, session_service=session_service, app_name="default")

print("✅ Calculation agent created with additional custom function tool")
print("🔧 new tool added!")



# Retrieve the session and inspect its state

session = await session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id="state-demo-session"
)

print("Session State Contents:")
# print(session.state)
print("\n🔍 Notice the 'user:name' and 'user:country' keys storing our data!")


await run_session(
    runner,
    [
        # Agent shouldn't know the name yet
        "Hi there, I am Tom from Canada",  
        "I would like to take away one sausage roll please.",
    ],
    "state-demo-session",
)


# Start a completely new session (for new customer) - the agent won't know the name of the new customer yet

await run_session(
    runner,
    ["Good morning, I am Noriko from Japan and I want a salt bread to take home"],
    "new-isolated-session",
)








































