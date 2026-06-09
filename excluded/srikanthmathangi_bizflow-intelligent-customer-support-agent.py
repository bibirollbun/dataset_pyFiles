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


# 1. Install the Google Gen AI SDK
!pip install -q -U google-generativeai

import os
import google.generativeai as genai
from google.api_core import retry
import json

# --- 2. SETUP API KEY from Kaggle Secrets ---
# You must have your GEMINI_API_KEY saved as a 'Secret' in your Kaggle Notebook.
# Go to 'Add-ons' -> 'Secrets' -> Label the secret 'GEMINI_API_KEY'
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    # Replace the secret label if you used a different one (e.g., 'GOOGLE_API_KEY')
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY") 
except:
    print("WARNING: Could not load API key from Kaggle Secrets. Trying environment variable.")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_FALLBACK_KEY_HERE")
    
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_FALLBACK_KEY_HERE":
    raise ValueError("GEMINI_API_KEY not found. Please set it in Kaggle Secrets.")

genai.configure(api_key=GEMINI_API_KEY)
print("Gemini API configured successfully.")


# --- MOCK BUSINESS DATABASE ---
orders_db = {
    "ORD-123": {"status": "Shipped", "item": "Wireless Headphones", "returnable": True},
    "ORD-456": {"status": "Delivered", "item": "Gaming Mouse", "returnable": False, "reason": "3-month return window elapsed."},
    "ORD-789": {"status": "Processing", "item": "Monitor", "returnable": True}
}

# --- FEATURE: CUSTOM TOOLS (Python Functions) ---

def check_order_status(order_id: str):
    """
    Retrieves the current status and item description for a given order ID.
    Args:
        order_id: The unique order ID (e.g., 'ORD-123').
    """
    order = orders_db.get(order_id)
    if order:
        # Use json.dumps for clean tool output
        return json.dumps({"order_id": order_id, "status": order['status'], "item": order['item']})
    else:
        return f"Order ID {order_id} not found."

def check_refund_policy(order_id: str):
    """
    Checks if an order is eligible for a refund based on current store policy.
    Args:
        order_id: The unique order ID.
    """
    order = orders_db.get(order_id)
    if not order:
        return "Order not found."
    
    if order['returnable']:
        return f"Order {order_id} is eligible for a return. Please proceed with the return form."
    else:
        return f"I'm sorry, order {order_id} is not eligible for a refund. Reason: {order.get('reason', 'Policy violation')}"

# List of tools to pass to the Specialist Agent
tools_list = [check_order_status, check_refund_policy]

# Initialize the Specialist Agent Model
# ğŸŒŸ FIX: Changing model to gemini-2.5-flash for better availability and function calling support.
specialist_agent_model = genai.GenerativeModel(
    model_name='gemini-2.5-flash', 
    tools=tools_list,
    system_instruction="""
    You are the TechGear Inc. Support Specialist. Your role is to resolve customer issues 
    by executing the provided tools. Be concise, professional, and directly address 
    the customer's query using the information returned by the tools.
    """
)
print("Specialist Agent initialized with Custom Tools using gemini-2.5-flash.")


# --- FEATURE: MULTI-AGENT SYSTEM (Sequential/Routing) ---

# ğŸŒŸ FIX: Changing model to gemini-2.5-flash for better availability.
triage_agent_model = genai.GenerativeModel('gemini-2.5-flash')

def run_multi_agent_flow(user_query: str):
    print(f"\n{'='*50}\nğŸ‘¤ User Query: {user_query}")
    
    # 1. Triage Agent (Router)
    triage_prompt = f"""
    Analyze this user query: '{user_query}'.
    Your output MUST be ONLY one word: 'SUPPORT' (if the query is about orders, returns, or technical help) 
    or 'GENERAL' (if the query is a simple greeting or casual chat).
    """
    # The Triage Agent call was causing the error, now using the fixed model name
    triage_response = triage_agent_model.generate_content(triage_prompt).text.strip().upper()
    print(f"ğŸ¤– Triage Agent: Detected intent -> **{triage_response}**")
    
    # 2. Sequential Handoff to Specialist Agent
    if triage_response == "SUPPORT":
        print("â�¡ï¸� Handoff to Specialist Agent for resolution...")
        
        # Start a chat session (enables automatic history/memory)
        # Note: The specialist_agent_model was initialized in Cell 2
        chat = specialist_agent_model.start_chat(enable_automatic_function_calling=True)
        
        # The Specialist Agent handles the query and executes tools automatically
        response = chat.send_message(user_query)
        
        final_response = response.text
        
    else:
        # Fallback for General queries
        final_response = "Hello! I am BizFlow, your support agent. I can help with orders and returns. Please ask me about an order ID."
        chat = None

    print(f"âœ… FINAL AGENT RESPONSE: {final_response}")
    return final_response, chat.history if chat else []

# Test Case 1: Order Lookup (Requires Tool Call)
result_1, history_1 = run_multi_agent_flow("I need to check on my order, ORD-123.")

# Test Case 2: Refund Check (Requires Tool Call)
result_2, history_2 = run_multi_agent_flow("What's the return policy for order ORD-456?")

# Test Case 3: General Chat (Requires Triage)
result_3, history_3 = run_multi_agent_flow("Hi there, how are you today?")


# --- FEATURE: AGENT EVALUATION ---

def evaluate_performance(user_query: str, agent_response: str, history: list):
    """
    A separate agent (the QA Manager) evaluates the performance of the Specialist Agent.
    """
    # ğŸŒŸ FIX: Changing model to gemini-2.5-pro for complex reasoning.
    evaluator = genai.GenerativeModel('gemini-2.5-pro') 
    
    # Format the history for the evaluator to review the full context
    conversation_summary = f"User Query: {user_query}\nAgent Response: {agent_response}\nFull History: {history}"

    eval_prompt = f"""
    You are a highly experienced QA Manager. Grade the following support interaction on a scale of 1 to 5, 
    where 5 is perfect resolution and 1 is complete failure.
    
    Interaction Details:
    ---
    {conversation_summary}
    ---
    
    Output your analysis with the following strict structure:
    SCORE: [1-5]
    REASONING: [Explain why the score was given, referencing clarity, use of tools, and correctness.]
    """
    
    print("\n" + "="*50)
    print("ğŸ“� Starting QA EVALUATION REPORT...")
    
    eval_result = evaluator.generate_content(eval_prompt)
    print(eval_result.text)

# Run evaluation on Test Case 2 (Refund Check)
evaluate_performance(
    user_query="What's the return policy for order ORD-456?", 
    agent_response=result_2, 
    history=history_2
)

