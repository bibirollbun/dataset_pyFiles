import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: {e}"
    )      


import pandas as pd
import sqlite3

# Load the data
df = pd.read_csv('/kaggle/input/product-details/products.csv')

# Create a connection to the SQLite database
conn = sqlite3.connect('product.db')

# Write the data to a table named 'product'
df.to_sql('product', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("âœ… 'product.db' created successfully with a 'product' table.")


!adk create product_expert_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY   


%%writefile product_expert_agent/tools.py

import sqlite3

DB_PATH = '/kaggle/working/product.db'

# Create custom tool: find_products()
def find_products(category: str, skin_type: str = "All", max_price: int = 99999) -> list[str]:
    """
    Finds products in the database matching the given criteria.
    Args:
        category: The product category to search for (e.g., 'Face Wash', 'Moisturizer').
        skin_type: The skin type the product is for (e.g., 'Oily', 'Dry', 'All').
        max_price: The maximum price of the products to return.
        
    Returns:
        A list of product names that match the criteria.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = "SELECT product_name FROM product WHERE category = ? AND (skin_type = ? OR skin_type = 'All') AND price <= ?"
    
    # Parameters in tuple
    params = (category, skin_type, max_price)
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    
    # cursor.fetchall() returns a list of tuples, e.g., [('Neem Cleanser'), ('Citrus Wash')]
    # Flatten it into a simple list of strings.
    product_names = [row[0] for row in results]
    
    return product_names


# get_product_details() tool
def get_product_details(product_name: str) -> dict:
    """
    Gets the detailed information (price, quantity, and description) for a specific product.
    
    Args:
        product_name: The exact name of the product to look up.
        
    Returns:
        A dictionary containing the product's details, or an error message.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Allows columns to access by name
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    
    query = """
    SELECT 
      product_name, price, quantity, description 
    FROM product 
    WHERE 
      product_name LIKE ?
    """
    
    # '%' for a flexible search
    params = (f'%{product_name}%',)
    
    cursor.execute(query, params)
    result = cursor.fetchone()
    conn.close()
    
    if result:
        # Convert the Row object to a dictionary
        return dict(result)
    else:
        return {"error": f"Product '{product_name}' not found."}


%%writefile product_expert_agent/agent.py

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.genai import types   

from product_expert_agent.tools import find_products, get_product_details

# Configure Model Retry on errors
retry_config = types.HttpRetryOptions(
    attempts=5,  
    exp_base=7,  
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  
)

# ----------------------------------
# Agent 1: Product Expert Agent 
# ----------------------------------
product_expert_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ProductExpertAgent",
    description="Skincare product expert agent",
    instruction="""You are an product expert assistant for a skincare store.
    Your job is to help users find the perfect skin care product based on their need.
    **Your knowledge is strictly limited to the information you can get from your tools.**
    **You have access to a database with the following product categories: 'Serum', 'Face Wash', 'Moisturizer', 'Toner'.**
    
    RULES:
    - **When a user asks for a product, you MUST map their request to one of the available categories listed 
       above.** For example:
       
        - If a user asks for 'face serum' or 'glow serum' or 'serum', or 'serums' you MUST use the category 'Serum'.
        - If a user asks for a 'face cleanser' or 'face wash', you MUST use the category 'Face Wash'.
        - If a user asks for a 'hydrating cream' or 'mositurizer', you MUST use the category 'Moisturizer'.
        - If a user asks for a 'face toner' or 'toner', you MUST use the category 'Toner'. 
    
    - To find products, you MUST use the `find_products` tool.
    
    - **CRITICAL INTERPRETATION RULE:** 

    When you receive a list of products from the `find_products` tool, **you MUST analyze the `skin_type` field returned for each product**. When interpreting skin suitability, you MUST rely ONLY on the skin_type field. Never infer suitability from the description.
    
        - **Prioritize:** If the user requested a specific skin type (e.g., 'Oily', 'Dry'), and the tool returns a product with that exact skin type, recommend that product first and state that it is **EXACTLY FOR** that skin type.
        
        - **Fallback:** If only products with `skin_type` set to 'All' are returned, recommend those but strictly state that they are **suitable for all skin types.**

    - Do not list more than three products unless the user asks for more.
    
    - If the user asks for "details", "more details", "more info", "information", 
      "tell me about it", "describe", "explain", or anything similar, 
       you MUST call the `get_product_details` tool for each product they are asking about.

    - When using `get_product_details`, retrieve and present all fields it returns, 
      including price, quantity, skin_type, and a consice explanation. 
      Do NOT omit fields unless the user explicitly asks for only specific information.

    - Never say "I already told you" or "as I said before."
      Always maintain a polite, helpful e-commerce tone.
    
    - If no products are found after searching, inform the user you could not find a match. Do not invent products or categories.
    - Be friendly, helpful, and concise in your final answer.
    """,
    tools=[find_products, get_product_details],
)   

# --------------------------------
# Agent 2: Skin Suitability Agent
# --------------------------------
skin_suitability_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="SkinSuitabilityAgent",
    description="Evaluates whether a recommended product matches the user's skin needs.",
    instruction="""
    You are an evaluator agent.

    INPUT: You will receive:
        - The user's original skin type request.
        - The product(s) selected by ProductExpertAgent.
        - The product details returned via tools.

    TASK:
    - Check whether each product suits the skin type the user asked for.
    - If the product's skin_type matches exactly â†’ say "Perfect match."
    - If product has skin_type = 'All' â†’ say "Suitable for all skin types."
    - If product does NOT match â†’ warn the user.
    - Then give a short final recommendation summary.

    IMPORTANT:
    - You do NOT call any tools. You only evaluate based on the information given to you.
    - You do NOT invent information.
    """,
)

# ----------------------------------
# Multi-Agent Pipeline (Sequential)
# ----------------------------------
root_agent = SequentialAgent(
    name="SkincareSystem",
    description="Two-step skincare recommendation system",
    sub_agents=[
        product_expert_agent,        
        skin_suitability_agent       
    ]
)



# Setup Variables
# Application name
APP_NAME = "SkincareStore"
# User id
USER_ID = "test_user_01"
# Session id
SESSION_ID = "product_expert_session_01"  
# Model name
MODEL_NAME = "gemini-2.5-flash-lite"

# Import DatabaseSessionService
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner

# Import root_agent from product_expert_agent.agent
from product_expert_agent.agent import root_agent

# Setup SQLite database 
db_url = f"sqlite:///{APP_NAME}_sessions.db" 

# Setup DatabaseSessionService
session_service = DatabaseSessionService(db_url=db_url)

# Create a runner with persistent storage
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

response = await runner.run_debug("I need a face wash for my oily skin, something under 500 rupees. Please give me more details about the products", 
                                  session_id=SESSION_ID, user_id=USER_ID)


response = await runner.run_debug("Do you have a face serum under 1500 for dry skin that helps hydrate?", 
                                   session_id=SESSION_ID, user_id=USER_ID)     


response = await runner.run_debug("Do you have any toner for dry or sensitive skin?", session_id=SESSION_ID, user_id=USER_ID)


response = await runner.run_debug("Do you have moisturizer for dry skin under 1000 rupees?", session_id=SESSION_ID, user_id=USER_ID)


 !adk create order_booking_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


# Load the data 
df = pd.read_csv('/kaggle/input/inventory/inventory.csv')

# Create a connection to the SQLite database
conn = sqlite3.connect('inventory.db')

# Inventory data
df.to_sql('inventory', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("âœ… 'inventory.db' created successfully with a 'inventory' table.")


%%writefile order_booking_agent/tools.py

import sqlite3
import uuid
from google.adk.tools import ToolContext, FunctionTool


DB_PATH = "/kaggle/working/inventory.db"

# Auto-approval limit
MAX_AUTO_QTY = 2  

# Create function to check inventory
def check_inventory(product_name: str) -> dict:
    """
    Looks up the given product in the SQLite inventory database.

    Returns:
        dict with product details or {"status": "not_found"}.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT product_id, product_name, stock_level, unit_price, discount_percent
        FROM inventory
        WHERE LOWER(product_name) = LOWER(?)
        """,
        (product_name,)
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"status": "not_found"}

    return {
        "status": "ok",
        "product_id": row[0],
        "product_name": row[1],
        "stock_level": row[2],
        "unit_price": row[3],
        "discount_percent": row[4],
    }

# Create custom tool: place_order() to place orders and to handle human-in-the loop process
def place_order(product_name: str, quantity: int, tool_context: ToolContext) -> dict:
    """
    Tool that:
    1. Checks inventory from SQLite
    2. Enforces max 2 units without approval
    3. If quantity > 2 â†’ uses ToolContext for human approval (pauses!)
    4. On resume â†’ processes approval
    """

    # Step 1 - Fetch product from DB
    product = check_inventory(product_name)

    if product["status"] == "not_found":
        return {
            "status": "error",
            "message": f"Product '{product_name}' not found in inventory."
        }

    if product["stock_level"] <= 0:
        return {
            "status": "out_of_stock",
            "message": f"'{product_name}' is currently out of stock."
        }

    # Step 2 - Quantity -> stock?
    if quantity > product["stock_level"]:
        return {
            "status": "insufficient_stock",
            "message": f"Only {product['stock_level']} units available."
        }

    # Step 3 - Quantity <= MAX_AUTO_QTY -> Auto-approve
    if quantity <= MAX_AUTO_QTY:
        final_price = calculate_final_price(product, quantity)
        return {
            "status": "approved",
            "auto": True,
            "order_id": f"ORD-{uuid.uuid4().hex[:8]}",
            "product": product,
            "quantity": quantity,
            "final_price": final_price,
            "delivery_date": "2 days",
            "message": f"Order auto-approved for {quantity} units of {product_name}."
        }

    # Step 4 - Quantity -> MAX_AUTO_QTY -> Require human approval

    # 4A - First call -> request confirmation
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"User wants to order {quantity} units of {product_name}. Max allowed is {MAX_AUTO_QTY} without approval. Approve?",
            payload={"product_name": product_name, "quantity": quantity},
        )
        return {
            "status": "pending",
            "message": f"Order for {quantity} units requires human approval."
        }

    # 4B - Human approval confirmed -> process final order
      
    if tool_context.tool_confirmation:
        human_decision = tool_context.tool_confirmation.confirmed
    
        if not human_decision:
            # human rejected
            return {
                "status": "rejected",
                "message": "Order was rejected by human."
        }

    # human approved -> continue
    final_price = calculate_final_price(product, quantity)
    return {
        "status": "approved",
        "auto": False,
        "order_id": f"ORD-{uuid.uuid4().hex[:8]}",
        "product": product,
        "quantity": quantity,
        "final_price": final_price,
        "delivery_date": "2 days",
        "message": "Order approved by human."
    } 

# Calculate final price after discount 
def calculate_final_price(product: dict, quantity: int) -> float:
    """
    Apply discount and return final amount.
    """
    price = product["unit_price"] * quantity
    discount = price * (product["discount_percent"] / 100)
    return round(price - discount, 2)

# Register tools for ADK
place_order_tool = FunctionTool(func=place_order)


%%writefile order_booking_agent/agent.py

import uuid
import asyncio
from datetime import datetime, timedelta

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.apps import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from google.genai import types

# Import tools from order_booking_agent/tools.py
from order_booking_agent.tools import place_order_tool

# Retry config for error
retry_config = genai_types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

APP_NAME = "SkincareStore"
USER_ID = "test_user_01"
MODEL_NAME = "gemini-2.5-flash-lite"

# -------------------------------------
#  OrderBookingAgent 
# -------------------------------------
order_booking_agent = LlmAgent(
    name="OrderBookingAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are OrderBookingAgent. Use the provided `place_order` tool to process incoming order requests.

Workflow:
1. When a user requests to order a product, you MUST call the `place_order` tool with the product name and quantity.
2. You MUST interpret the 'status' field returned by the `place_order` tool and act accordingly:
   - If status is "out_of_stock" or "insufficient_stock": Inform the user clearly and STOP.
   - If status is "approved" and auto is True: Reply to the user with:
         "Your order is auto-approved!"
          Then print the order block exactly in this format:
            Order Detail:
            order_id: <order_id>
            product_name: <product_name>
            quantity: <quantity>
            final_price: <final_price>  
            delivery_date: <delivery_date> Stop

   - If status is "pending": This means the order needs approval. You MUST inform the user that human approval is required.
   - If status is "approved" and auto is False: This means the order was resumed after human approval. You MUST inform the user with "Your order is approved by the human!" Then print the same order block (as above) STOP.
   - If status is "rejected": You MUST inform the user that "Your order got Rejected." Do NOT print any order block STOP.
        
""",
    tools=[place_order_tool], 
)

# Resumable App + Runner
order_app = App(
    name="order_booking_app",
    root_agent=order_booking_agent,
    resumability_config=ResumabilityConfig(
        is_resumable=True,
    ),
)

# Define Session Service
session_service = InMemorySessionService()

# Define runner
order_runner = Runner(app=order_app, session_service=session_service)

# Helper functions to handle adk_request_confirmation and printing
def check_for_approval(events):
    """
    Return approval info dict if adk_request_confirmation present in events, else None.
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None) and getattr(part.function_call, "name", "") == "adk_request_confirmation":
                    return {"approval_id": part.function_call.id, "invocation_id": event.invocation_id}
    return None

def print_agent_text(events):
    """Print the agent textual outputs from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(f"Agent > {part.text}")

def create_approval_response(approval_info, approved: bool):
    """
    Create a content object holding the human approval decision to resume the tool.
    """
    confirmation_response = genai_types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return genai_types.Content(role="user", parts=[genai_types.Part(function_response=confirmation_response)])

# Async runner function 
async def run_order_workflow(user_query: str, human_approval: bool = True):
    """        
    user_query: e.g., "Order 3 units of HydraBoost Glow Serum"
    human_approval: True = human approves, False = human rejects
    """

    print("\n" + "="*120)  
    print(f"User > {user_query}")

    SESSION_ID = f"order_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name="order_booking_app",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    #  STEP 1: Send initial user message 
    initial_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_query)]
    )
    events = []
    async for event in order_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=initial_content
    ):
        events.append(event)

    print_agent_text(events)

    # STEP 2: Checks if agent paused for approval 
    approval_info = check_for_approval(events)

    if approval_info:
        # The agent paused and asked for human approval
        print(f"â�¸ï¸� Agent paused for human approval (invocation: {approval_info['invocation_id']})")

        # STEP 3: Simulate human decision 
        print("Simulated Human Decision:", "APPROVE" if human_approval else "REJECT")

        approval_response = create_approval_response(
            approval_info,
            human_approval
        )

        # Resume the paused tool call
        async for event in order_runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=approval_response,
            invocation_id=approval_info["invocation_id"]
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        print(f"Agent > {part.text}")

    print("="*120 + "\n")
    


from order_booking_agent.agent import run_order_workflow

# Small order -> Auto-approval
await run_order_workflow("Order 2 units of HydraBoost Glow Serum") 


# Large order -> human approves
await run_order_workflow("Order 4 units of Vitamin C Brightening Cream Moisturizer", human_approval=True)


# Large order -> human rejects
await run_order_workflow("Order 6 units of Citrus Oil-Control Wash", human_approval=False)  

