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


!pip install -q google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
print("âœ… Gemini ADK Client initialized successfully!")


response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Testing ADK setup in Kaggle Notebook."
)

print(response.text)


!mkdir -p project/data
!mkdir -p project/tools
!mkdir -p project/agents
!mkdir -p project/utils


%%writefile project/data/orders.json
[
    {
        "order_id": "ORD1001",
        "customer_name": "John Doe",
        "status": "Delivered",
        "items": [
            {"product": "Wireless Mouse", "price": 599}
        ],
        "invoice_url": "invoice_1001.pdf"
    },
    {
        "order_id": "ORD1002",
        "customer_name": "Anna Smith",
        "status": "Shipped",
        "items": [
            {"product": "Keyboard", "price": 1599}
        ],
        "invoice_url": "invoice_1002.pdf"
    }
]


%%writefile project/tools/order_lookup.py
import json

def order_lookup_tool(order_id: str):
    """Look up order details from JSON file."""
    with open("project/data/orders.json") as f:
        orders = json.load(f)

    for order in orders:
        if order["order_id"].lower() == order_id.lower():
            return order

    return {"error": "Order not found"}


%%writefile project/tools/invoice_tool.py
def invoice_tool(order_id: str):
    """Return invoice file name associated with an order."""
    return {"invoice_url": f"invoice_{order_id[-4:]}.pdf"}


%%writefile project/tools/refund_tool.py
def refund_tool(order_id: str):
    """Simulate refund initialization."""
    return {
        "order_id": order_id,
        "refund_status": "Refund Initiated",
        "refund_amount": "â‚¹1000",
        "processing_time": "5-7 business days"
    }


%%writefile project/tools/image_tool.py
def image_tool(image_path: str):
    """Simulated image understanding."""
    return {"description": "This image appears to show a product (simulated vision output)."}


%%writefile project/agents/ecommerce_agent.py
from google import genai
from project.tools.order_lookup import order_lookup_tool
from project.tools.invoice_tool import invoice_tool
from project.tools.refund_tool import refund_tool
from project.tools.image_tool import image_tool

client = genai.Client()

# Define the tools in ADK format
tools = {
    "order_lookup": order_lookup_tool,
    "invoice_generator": invoice_tool,
    "refund_processor": refund_tool,
    "image_understanding": image_tool
}

def ecommerce_agent(query: str):
    """Routing logic that decides which tool to call based on user query."""
    query_lower = query.lower()

    if "status" in query_lower or "track" in query_lower:
        # Extract order ID
        for word in query.split():
            if word.startswith("ORD"):
                return tools["order_lookup"](word)

    if "invoice" in query_lower:
        for word in query.split():
            if word.startswith("ORD"):
                return tools["invoice_generator"](word)

    if "refund" in query_lower:
        for word in query.split():
            if word.startswith("ORD"):
                return tools["refund_processor"](word)

    if "image" in query_lower:
        return tools["image_understanding"]("product.jpg")

    # Default LLM fallback
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=query
    )
    return {"response": response.text}


%%writefile project/main.py
from project.agents.ecommerce_agent import ecommerce_agent

def run_demo():
    print("ğŸ“¦ E-Commerce Customer Support Agent Ready!")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = ecommerce_agent(query)
        print("Agent:", result)

if __name__ == "__main__":
    run_demo()


from project.agents.ecommerce_agent import ecommerce_agent

print(ecommerce_agent("Check order status for ORD1001"))
print(ecommerce_agent("Generate invoice for ORD1002"))
print(ecommerce_agent("Initiate refund for ORD1001"))
print(ecommerce_agent("Describe the image"))

