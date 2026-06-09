# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import asyncio
import json
import os
import warnings
from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types
from google.genai.client import Client

print("Imported Successfully")


# Suppress Python warnings
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('google_genai.types').setLevel(logging.ERROR)
logging.getLogger('google_genai').setLevel(logging.ERROR)
logging.getLogger('google.genai').setLevel(logging.ERROR)


from kaggle_secrets import UserSecretsClient 
user_secrets = UserSecretsClient()

try:
    GOOGLE_API_KEY = user_secrets.get_secret ("GOOGLE_API_KEY") 
    os.environ [ "GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("API Key loaded successfully from Kaggle Secrets") 
except Exception as e:
    print(f"X Error loading API key: {e}")
    print("Please add GOOGLE API KEY to Kaggle Secrets") 
    raise

client = Client(api_key=GOOGLE_API_KEY) 
MODEL_NAME = "gemini-2.5-flash-lite"


SAM_PRODUCTS = [
    {
        "product_id": "SAM-FRG-001",
        "product_name": "SAM Smart Fridge",
        "description": "Premium 4-door refrigerator with smart features, water dispenser, and LED display. Stainless steel finish with 25 cu.ft. capacity.",
        "brand": "SAM",
        "search_tags": ["refrigerator", "fridge", "smart fridge", "french door", "kitchen appliances"],
        "price": "$2,499",
        "features": ["Smart features", "Water dispenser", "LED display", "25 cu.ft. capacity"]
    },
    {
        "product_id": "SAM-WM-002",
        "product_name": "SAM Front Load Washing Machine",
        "description": "Energy-efficient 7.5kg front-load washing machine with inverter motor and steam wash technology.",
        "brand": "SAM",
        "search_tags": ["washing machine", "front load", "laundry", "steam wash", "home appliances"],
        "price": "$899",
        "features": ["7.5kg capacity", "Inverter motor", "Steam wash", "Wi-Fi control"]
    },
    {
        "product_id": "SAM-AC-003",
        "product_name": "SAM Split Air Conditioner",
        "description": "1.5-ton smart inverter split AC with 5-star energy rating, AI auto cooling, and copper condenser.",
        "brand": "SAM",
        "search_tags": ["air conditioner", "AC", "split AC", "inverter AC", "cooling"],
        "price": "$1,199",
        "features": ["1.5-ton capacity", "AI auto cooling", "5-star rating", "Copper condenser"]
    },
    {
        "product_id": "SAM-TV-004",
        "product_name": "SAM 65-inch 4K Smart QLED TV",
        "description": "Ultra-HD 4K QLED smart television with HDR10+, voice assistant, and 120Hz refresh rate.",
        "brand": "SAM",
        "search_tags": ["TV", "smart TV", "QLED", "4K", "Home entertainment"],
        "price": "$1,499",
        "features": ["HDR10+", "120Hz refresh rate", "Voice assistant", "Dolby Atmos"]
    },
    {
        "product_id": "SAM-MWO-005",
        "product_name": "SAM Convection Microwave Oven",
        "description": "28-liter convection microwave with grill mode, auto cook menus, and smart control panel.",
        "brand": "SAM",
        "search_tags": ["microwave oven", "convection oven", "kitchen", "cooking appliances"],
        "price": "$349",
        "features": ["28-liter capacity", "Grill mode", "Auto cook menus", "Smart control panel"]
    },
    {
        "product_id": "SAM-VAC-006",
        "product_name": "SAM Smart Robot Vacuum Cleaner",
        "description": "Self-charging robot vacuum cleaner with LiDAR navigation, mop system, and app control.",
        "brand": "SAM",
        "search_tags": ["robot vacuum", "vacuum cleaner", "smart home", "cleaning"],
        "price": "$599",
        "features": ["LiDAR navigation", "Self-charging", "Vacuum + mop", "App & voice control"]
    }
]

print("Product Database Created")


import json

def search_products(query: str) -> str: 
    """
    Search SAM product database for relevant products.

    Args:
        query: Customer search query (e.g., 'smart home','kitchen', 'outdoor')

    Returns:
        JSON string with matching products
    """
    query_lower = query.lower()
    matching_products = []

    for product in SAM_PRODUCTS:
        search_text = (
            product["product_name"].lower() + " " +
            product["product_id"].lower() + " " +
            product["description"].lower() + " " +
            " ".join(tag.lower() for tag in product["search_tags"])
        )

        query_words = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]
        if any(word in search_text for word in query_words):
            matching_products.append(product)

    if not matching_products:
        return json.dumps({
            "message": "No products found matching your query.",
            "products": [],
            "suggestion": (
                "Try: refrigerator, Washing Machine, Robot Vacuum Cleaner, "
                "Smart QLED TV, Convection Microwave Oven, Split Air Conditioner"
            )
        })

    return json.dumps({
        "found": len(matching_products),
        "products": matching_products
    }, indent=2)


def get_product_details(product_identifier: str) -> str:
    """
    Get detailed information about a specific SAM product

    Args:
        product_identifier: Product name, ID, or keyword

    Returns:
        JSON string with complete product details
    """
    identifier_lower = product_identifier.lower()

    for product in SAM_PRODUCTS:
        # Check product ID
        if identifier_lower in product["product_id"].lower():
            return json.dumps({"success": True, "product": product}, indent=2)

        # Check product name (flexible word matching)
        name_words = product["product_name"].lower().split()
        query_words = identifier_lower.split()
        if any(qw in name_words for qw in query_words if len(qw) > 2):
            return json.dumps({"success": True, "product": product}, indent=2)

        # Check search tags
        if any(identifier_lower in tag.lower() for tag in product["search_tags"]):
            return json.dumps({"success": True, "product": product}, indent=2)

    return json.dumps({
        "success": False,
        "error": f"No product found matching '{product_identifier}'.",
        "available_products": [p["product_name"] for p in SAM_PRODUCTS]
    })

def list_all_products() -> str: 
    """List all available SAM products with prices.""" 
    product_list = [
        {
            "name": p["product_name"],
            "id": p["product_id"],
            "price": p["price"],
            "category": p["search_tags"][0],
            "description": p["description"]
        }
        for p in SAM_PRODUCTS
    ]
    
    return json.dumps({
        "total products": len(SAM_PRODUCTS),
        "products": product_list
    }, indent=2)


async def main():
    """Run the SAM customer support agent."""

    print("\n" + "=" * 80)
    print("SAM CUSTOMER SUPPORT AGENT")
    print("=" * 80)
    print("Powered by Google SDK and Gemini AI")
    print("5 Products Available | Text-Only Responses")
    print(f"Model: {MODEL_NAME}")
    print("=" * 80 + "\n")

    app_name = "sam_support_agent"
    user_id = "customer_001"

    # Define the agent
    support_agent = Agent(
        model=MODEL_NAME,
        name="sam_support_specialist",
        instruction="""You are a friendly SAM customer support specialist.

IMPORTANT RULES:
1. When a customer mentions a product name, use get_product_details to fetch information
2. If the tool returns success, present the product details naturally
3. Always be helpful, clear, and avoid contradicting yourself
4. Include prices and key features when describing products

YOUR TOOLS:
- search_products: Find products by keywords (kitchen, smart home, outdoor, tools)
- get_product_details: Get complete details about a specific product
- list_all_products: Show all available SAM products

SAM PRODUCT CATALOG:
- SAM Smart Refrigerator ($2,499) - Premium kitchen appliance
- SAM Front Load Washing Machine ($899) - Home appliance
- SAM Split Air Conditioner ($1,199) - Cooling device
- SAM 65-inch 4K Smart QLED TV ($1,499) - LED TV
- SAM Convection Microwave Oven ($349) - Kitchen appliance
- SAM Smart Robot Vacuum Cleaner ($599) - Home appliance

Be conversational and use the tools to provide accurate product information!""",
        tools=[search_products, get_product_details, list_all_products],
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    # Runner & session
    runner = InMemoryRunner(agent=support_agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    async def run_query(session: Session, user_message: str):
        """Process customer query and stream the response."""
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )

        print(f"\n{'-' * 80}")
        print(f" YOU: {user_message}")
        print(f"{'-' * 80}")
        print("SAM SUPPORT: ", end="", flush=True)

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        ):
            if getattr(event, "content", None) and event.content.parts:
                part = event.content.parts[0]
                if getattr(part, "text", None):
                    chunk = part.text
                    print(chunk, end="", flush=True)

        print()  # newline after response

    # ===============================
    # INTERACTIVE LOOP (inside main)
    # ===============================
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE ACTIVATED")
    print("=" * 80)
    print("Example queries:")
    print("  - 'Show me all products'")
    print("  - 'SAM Smart Refrigerator'")
    print("  - 'I need something for my kitchen'")
    print("  - 'Tell me about the LED TV'")
    print("  - 'Do you have Home products?'")
    print("\n Type 'exit', 'quit', or 'bye' to end conversation")
    print("=" * 80)

    # Greeting
    await run_query(session, "Hello! What products does SAM offer?")

    while True:
        print("\n" + "-" * 80)
        try:
            # Get user input without blocking event loop
            loop = asyncio.get_running_loop()
            user_input = await loop.run_in_executor(
                None, lambda: input(" YOU: ")
            )

            if user_input.lower().strip() in ["exit", "quit", "q", "bye", "goodbye"]:
                print("\n" + "=" * 80)
                print("Thank you for contacting SAM Customer Support!")
                print("Have a great day!")
                print("=" * 80 + "\n")
                break

            if not user_input.strip():
                print("Please enter a question or type 'exit' to quit.")
                continue

            await run_query(session, user_input.strip())

        except KeyboardInterrupt:
            print("\n\n Conversation interrupted by user.")
            print("Goodbye!\n")
            break

        except Exception as e:
            print(f"\n Error: {e}")
            print("Please try again or type 'exit' to quit.\n")


await main()

