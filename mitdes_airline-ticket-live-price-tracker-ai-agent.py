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


pip install google-genai
pip install mcp
pip install python-dotenv
pip install requests



import os
import asyncio
import json
import requests
from dotenv import load_dotenv

from mcp.server import Server
from mcp.types import Tool, TextContent, ToolOutput

load_dotenv()

AMADEUS_KEY = os.getenv("AMADEUS_API_KEY")

# Create MCP server using official SDK
server = Server("flight-price-tool")

@server.tool(
    Tool(
        name="flight_search",
        description="Retrieve real-time flight prices.",
        input_schema={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"}
            },
            "required": ["origin", "destination", "date"]
        }
    )
)
async def flight_search(arguments):
    origin = arguments["origin"]
    destination = arguments["destination"]
    date = arguments["date"]

    url = (
        f"https://test.api.amadeus.com/v2/shopping/flight-offers?"
        f"originLocationCode={origin}&destinationLocationCode={destination}"
        f"&departureDate={date}&adults=1"
    )

    headers = {"Authorization": f"Bearer {AMADEUS_KEY}"}
    r = requests.get(url, headers=headers)

    return ToolOutput(
        content=[
            TextContent(
                type="text",
                text=json.dumps(r.json(), indent=2)
            )
        ]
    )

async def main():
    await server.run_websocket(host="localhost", port=8765)

if __name__ == "__main__":
    asyncio.run(main())



python mcp_flight_tool.py



import asyncio
from mcp.server import Server
from mcp.types import Tool, ToolOutput, TextContent
import firebase_admin
from firebase_admin import credentials, messaging

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

server = Server("notification-tool")

@server.tool(
    Tool(
        name="notify_user",
        description="Send a push notification via Firebase Cloud Messaging.",
        input_schema={
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "message": {"type": "string"}
            },
            "required": ["token", "message"]
        }
    )
)
async def notify_user(args):
    msg = messaging.Message(
        notification=messaging.Notification(
            title="Flight Price Alert",
            body=args["message"]
        ),
        token=args["token"]
    )

    messaging.send(msg)

    return ToolOutput(
        content=[TextContent(type="text", text="Notification sent.")]
    )

async def main():
    await server.run_websocket(host="localhost", port=8770)

if __name__ == "__main__":
    asyncio.run(main())



import asyncio
from google import genai
from mcp import Client

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")

# Connect to MCP tools
flight_tool = Client("ws://localhost:8765")
notify_tool = Client("ws://localhost:8770")

async def check_flight(user):
    # Call real-time price tool
    result = await flight_tool.call_tool("flight_search", {
        "origin": user["origin"],
        "destination": user["destination"],
        "date": user["date"]
    })

    # Parse result
    data = result["output"]["content"][0]["text"]
    # (Parse JSON here as needed)

    # Example: Use Gemini for reasoning
    ai_response = model.generate_content(
        f"Analyze flight data: {data}. User threshold: {user['threshold']}."
    ).text

    # If price under threshold → notify
    if "below threshold" in ai_response.lower():
        await notify_tool.call_tool("notify_user", {
            "token": user["fcm_token"],
            "message": "Flight price dropped below your target!"
        })

    return ai_response

if __name__ == "__main__":
    user = {
        "origin": "JFK",
        "destination": "LAX",
        "date": "2025-01-15",
        "threshold": 250,
        "fcm_token": "USER_DEVICE_TOKEN"
    }
    print(asyncio.run(check_flight(user)))


