!pip install google-adk


import os
import json
import re
import asyncio
from typing import List, Dict, Any

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool, google_search, AgentTool
from google.adk.runners import InMemoryRunner
from google.genai import types


# Setup & Configuration

try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except ImportError:
    print("âš ï¸� Kaggle Secrets not available. Ensure you're in a Kaggle Notebook.")
except KeyError:
    print("ğŸ”‘ Authentication Error: Add 'GOOGLE_API_KEY' to Kaggle secrets.")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)


# Load & Parse User Profile

def load_user_profile(dataset_path: str) -> Dict[str, Any]:
    user_profile = {}
    try:
        with open(dataset_path, 'r') as f:
            data = f.read()

            extract = lambda label: (match.group(1).strip()
                                     if (match := re.search(fr"{label}:\s*(.*)", data, re.DOTALL))
                                     else "")

            user_profile["preferences"] = extract("User Preferences")
            user_profile["dietary_restrictions"] = extract("Dietary Restrictions")
            user_profile["travel_destinations"] = extract("Travel Destinations")
            user_profile["budget"] = extract("Budget")
            user_profile["location"] = extract("Location")

    except FileNotFoundError:
        print(f"â�Œ Error: File not found: {dataset_path}")
    except Exception as e:
        print(f"â�Œ Error loading profile: {e}")
        raise

    return user_profile

DATASET_PATH = "/kaggle/input/agents-intensive-capstone-project/Hackathon dataset.txt"
user_profile = load_user_profile(DATASET_PATH)
print("Loaded User Profile:", user_profile)



# Custom Tools (Meal / Shopping / Travel)

def get_recipe(ingredients: str, dietary_restrictions: str = None) -> str:
    recipe = f"Recipe for {ingredients}."
    if dietary_restrictions:
        recipe += f" (Restrictions: {dietary_restrictions})"
    return recipe

def find_grocery_stores(location: str, items: str) -> str:
    return f"Grocery stores near {location}: Store A, B, C. Items likely available: {items}."

def plan_travel_itinerary(destination: str, budget: str, preferences: str = None) -> str:
    return f"Travel itinerary for {destination}. Budget: {budget}. Preferences: {preferences}."




# Create Agents

meal_planning_agent = Agent(
    name="MealPlanningAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Expert meal planner.",
    tools=[FunctionTool(get_recipe)],
    output_key="meal_plan"
)

shopping_agent = Agent(
    name="ShoppingAssistantAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Shopping assistant.",
    tools=[FunctionTool(find_grocery_stores)],
    output_key="shopping_list"
)

travel_agent = Agent(
    name="TravelCoordinatorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Travel coordinator.",
    tools=[FunctionTool(plan_travel_itinerary)],
    output_key="travel_plan"
)

planner_agent = Agent(
    name="PlannerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Personal concierge agent.",
    tools=[AgentTool(meal_planning_agent), AgentTool(shopping_agent), AgentTool(travel_agent)],
    output_key="final_response"
)

root_agent = SequentialAgent(
    name="ConciergeAgent",
    sub_agents=[planner_agent]
)




# Runner
runner = InMemoryRunner(agent=root_agent)
print("Runner Initialized")



# Runner Function

async def run_concierge(user_request: str):
    session_state = {
        "user_preferences": user_profile.get("preferences", ""),
        "dietary_restrictions": user_profile.get("dietary_restrictions", ""),
        "budget": user_profile.get("budget", ""),
        "location": user_profile.get("location", ""),
        "travel_destinations": user_profile.get("travel_destinations", ""),
    }

    print("\nSession State:", session_state)

    # âœ… FIXED: Remove 'inputs=' keyword
    response = await runner.run_debug({
        "user_request": user_request,
        "session_state": session_state
    })

    print("Final Response:", response)




# Example Usage

async def main():
    print("Starting main function...")

    await run_concierge("I have tomatoes, onions, and garlic. What can I make for dinner?")
    await run_concierge("I want to go to Europe. Where should I travel?")
    await run_concierge("I need to buy pasta, sauce, and bread.")

    print("Finished main function.")



# Safe Async Runner for Kaggle

def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(coro)



# Execute

run_async(main())



# Agent2Agent (A2A) Orchestration with Concierge

class ConciergeA2A:
    """
    Orchestrates multiple agents in a conversational flow using
    the Concierge SequentialAgent setup.
    """
    def __init__(self, runner: InMemoryRunner):
        self.runner = runner

    async def handle_request(self, request: str, session_state: Dict[str, Any]):
        """
        Sends user requests through the Concierge agent and returns responses.
        """
        print(f"\nğŸ’¬ User Request: {request}")
        response = await self.runner.run_debug({
            "user_request": request,
            "session_state": session_state
        })
        print(f"ğŸ¤– Concierge Response: {response}")
        return response

    async def multi_request_flow(self, requests: List[str]):
        """
        Processes multiple requests sequentially using the session_state.
        """
        session_state = {
            "user_preferences": user_profile.get("preferences", ""),
            "dietary_restrictions": user_profile.get("dietary_restrictions", ""),
            "budget": user_profile.get("budget", ""),
            "location": user_profile.get("location", ""),
            "travel_destinations": user_profile.get("travel_destinations", ""),
        }
        results = []
        for req in requests:
            result = await self.handle_request(req, session_state)
            results.append(result)
        return results



# Example Multi-Request Usage with A2A

async def a2a_demo():
    concierge_a2a = ConciergeA2A(runner)

    requests = [
        "I have tomatoes, onions, and garlic. What can I make for dinner?",
        "I need to buy pasta, sauce, and bread.",
        "I want to go to Europe. Where should I travel?",
        "Suggest a dinner recipe considering my dietary restrictions."
    ]

    responses = await concierge_a2a.multi_request_flow(requests)
    print("\nğŸ“Œ All Responses Collected:")
    for i, resp in enumerate(responses, 1):
        print(f"{i}. {resp}")




# Run A2A Demo
run_async(a2a_demo())


# Example: Run Concierge A2A Before Deployment

print("\nğŸš€ 13: Testing Concierge A2A Flow Before Deployment")

# Example multi-request session
a2a_test_requests = [
    "I have chicken, rice, and broccoli. Suggest a recipe.",
    "Where can I buy fresh vegetables near me?",
    "I want a weekend trip within my budget."
]

# Run the Concierge A2A demo for these requests
responses_before_deploy = run_async(ConciergeA2A(runner).multi_request_flow(a2a_test_requests))

print("\nğŸ“Œ Responses Before Deployment:")
for i, resp in enumerate(responses_before_deploy, 1):
    print(f"{i}. {resp}")

print("\nâœ… Concierge A2A flow tested successfully. Ready for deployment using step 5 (Agent Engine).")



# Pre-Deployment Testing + Deployment Prep
print("\nğŸš€ 14: Pre-Deployment Testing & Agent Engine Deployment Prep")

# 14.1: Test Concierge A2A Flow

a2a_test_requests = [
    "I have chicken, rice, and broccoli. Suggest a recipe.",
    "Where can I buy fresh vegetables near me?",
    "I want a weekend trip within my budget."
]

print("\nğŸ“Œ Running pre-deployment A2A tests...")
responses_before_deploy = run_async(ConciergeA2A(runner).multi_request_flow(a2a_test_requests))

print("\nğŸ“Œ Responses Before Deployment:")
for i, resp in enumerate(responses_before_deploy, 1):
    print(f"{i}. {resp}")

print("\nâœ… Pre-deployment Concierge A2A flow tested successfully!")

# 14.2: Agent Engine Deployment Example

import random
PROJECT_ID = os.environ.get("PROJECT_ID", "your-gcp-project-id")

# Deployment configuration
agent_engine_config = {
    "min_instances": 0,
    "max_instances": 1,
    "resource_limits": {"cpu": "1", "memory": "1Gi"}
}
config_path = "/tmp/.agent_engine_config.json"
with open(config_path, "w") as f:
    json.dump(agent_engine_config, f)
print("âœ… Agent Engine deployment config created.")

# Choose region randomly for demo
regions_list = ["europe-west1", "europe-west4", "us-east4", "us-west1"]
deployed_region = random.choice(regions_list)
print(f"âœ… Selected deployment region: {deployed_region}")

# Example deployment command (commented, requires CLI & GCP)
# !adk deploy agent_engine --project=$PROJECT_ID --region=$deployed_region sample_agent --agent_engine_config_file=$config_path

print("âœ… Deployment example ready. Use ADK CLI to deploy your agent to Agent Engine.")





