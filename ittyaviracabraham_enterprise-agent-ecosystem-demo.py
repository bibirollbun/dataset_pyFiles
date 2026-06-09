%pip install -q google-adk[a2a]


import os
import json
import time
import logging
import asyncio
import requests
import subprocess
from typing import Dict, Any

from google.adk.tools import FunctionTool
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from kaggle_secrets import UserSecretsClient

print("✅Installed Necessary libraries")


# Configure logging for observability
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("EnterpriseAgentEcosystem")


# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLEAPIKEY", "GOOGLEAPIKEY")
USER_ID = "default-user"
APP_NAME = "EnterpriseAgentEcosystem"
PRODUCT_CATALOG_PORT = 8001


# --- Retry Configuration ---
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# --- Product catalog ---
from google.adk.tools import FunctionTool

product_catalog = {
    "iphone 15 pro": "iPhone 15 Pro, $999, Low Stock (8 units), 128GB, Titanium finish",
    "samsung galaxy s24": "Samsung Galaxy S24, $799, In Stock (31 units), 256GB, Phantom Black",
    "dell xps 15": "Dell XPS 15, $1299, In Stock (45 units), 15.6' display, 16GB RAM, 512GB SSD",
    "macbook pro 14": "MacBook Pro 14, $1999, In Stock (22 units), M3 Pro chip, 18GB RAM, 512GB SSD",
    "sony wh-1000xm5": "Sony WH-1000XM5 Headphones, $399, In Stock (67 units), Noise-canceling, 30hr battery",
    "ipad air": "iPad Air, $599, In Stock (28 units), 10.9' display, 64GB",
    "lg ultrawide 34": "LG UltraWide 34 Monitor, $499, Out of Stock, Expected next week",
}

def get_product_info(product_name: str) -> str:
    key = product_name.lower().strip()
    if key in product_catalog:
        return f"Product details: {product_catalog[key]}"
    else:
        available = ", ".join([p.title() for p in product_catalog.keys()])
        return f"Sorry, product '{product_name}' not found. Available products: {available}."

get_product_info_tool = FunctionTool(
    func=get_product_info
)


product_catalog_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="product_catalog_agent",  # underscores only, no dashes
    description="External vendor product catalog agent providing product info and availability.",
    instruction=(
        "You are a product catalog specialist from an external vendor. "
        "Use the get_product_info tool to fetch data from the catalog. "
        "Provide accurate product info including price, availability, and specs."
    ),
    tools=[get_product_info_tool]
)


from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
import logging

logging.basicConfig(level=logging.INFO)


# Function to query the catalog
def get_product_info(product_name: str) -> str:
    key = product_name.lower().strip()
    logging.info(f"Received product info request: {product_name}")
    if key in product_catalog:
        return f"Product details: {product_catalog[key]}"
    else:
        available = ", ".join([p.title() for p in product_catalog.keys()])
        return f"Sorry, product '{product_name}' not found. Available products: {available}."

# Wrap with FunctionTool
get_product_info_tool = FunctionTool(func=get_product_info)

# Define the agent (no retry_options and valid name)
product_catalog_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="product_catalog_agent",
    description="Vendor product catalog agent providing product info and availability.",
    instruction=(
        "You are a product catalog specialist. Use get_product_info tool."
    ),
    tools=[get_product_info_tool]
)


# Example: Direct call to the tool (for testing)
print(get_product_info_tool.func("iphone 15 pro"))
print(get_product_info_tool.func("nonexistent product"))


# --- Custom Session State Tools ---
def save_user_info(toolcontext, username: str, country: str):
    toolcontext.state["user_username"] = username
    toolcontext.state["user_country"] = country
    return {"status": "success"}

def retrieve_user_info(toolcontext):
    username = toolcontext.state.get("user_username", "Unknown")
    country = toolcontext.state.get("user_country", "Unknown")
    return {"username": username, "country": country}

save_user_info_tool = FunctionTool(func=save_user_info)
retrieve_user_info_tool = FunctionTool(func=retrieve_user_info)

# --- Customer Support Agent ---
customer_support_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="customer_support_agent",
    description="Customer support agent that retrieves and saves user info.",
    instruction="Use save_user_info to record user name/country, retrieve_user_info to fetch.",
    tools=[save_user_info_tool, retrieve_user_info_tool]
)


from google.adk.memory import InMemoryMemoryService

memory_service = InMemoryMemoryService()


# --- Session and Runner Setup ---
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()
APP_NAME = "EnterpriseAgentDemo"

runner = Runner(agent=customer_support_agent, app_name=APP_NAME, session_service=session_service)


# --- Evaluation Integration ---

def create_evaluation_config():
    eval_config = {
        "criteria": {
            "tool_trajectory_avg_score": 1.0,
            "response_match_score": 0.8
        },
        "description": "Evaluation checks tool usage correctness and response quality."
    }
    with open("eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)
    logger.info("Evaluation configuration created.")

create_evaluation_config()

def evaluate_agent_responses(responses):
    """
    Dummy evaluation: checks if all responses contain expected keywords.
    """
    logger.info("Starting evaluation of agent responses.")
    scores = []
    for response in responses:
        score = 1.0 if "Product details" in response else 0.0
        scores.append(score)
        logger.info(f"Response: {response[:50]}... Score: {score}")

    avg_score = sum(scores)/len(scores) if scores else 0.0
    logger.info(f"Average evaluation score: {avg_score}")
    return avg_score


# --- Interaction and observability demonstration ---

# Synchronous demo: directly use tool or agent functions
def run_demo():
    queries = [
        "iphone 15 pro",
        "lg ultrawide 34",
        "nonexistent product"
    ]
    responses = []
    # Here, get_product_info_tool is used directly 
    for query in queries:
        response = get_product_info_tool.func(query)
        print(f"Agent: {response}")
        responses.append(response)

    # Evaluate responses
    avg_score = evaluate_agent_responses(responses)
    print(f"\nAverage evaluation score: {avg_score}")

run_demo()


import ipywidgets as widgets
from IPython.display import display

product_input = widgets.Text(
    value='iphone 15 pro',
    placeholder='Type product name',
    description='Product:',
)

output = widgets.Output()

def on_submit(change):
    output.clear_output()
    response = get_product_info_tool.func(product_input.value)
    with output:
        print("Agent:", response)

product_input.on_submit(on_submit)

display(product_input, output)

