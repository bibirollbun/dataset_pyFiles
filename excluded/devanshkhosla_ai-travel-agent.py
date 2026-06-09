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


# Install dependencies
!pip install -q google-generativeai
!pip install -q google-api-python-client>=2.108.0
!pip install -q requests>=2.31.0

print("=" * 70)
print("âœ… CELL 2: Installation Complete!")
print("=" * 70)
print("Installed packages:")
print("  âœ“ google-generativeai (Gemini API)")
print("  âœ“ google-api-python-client (Custom Search)")
print("  âœ“ requests (HTTP client)")
print("=" * 70)
print("Ready to proceed to Cell 3.")
print("=" * 70)


# Configuration - Using Kaggle Secrets (Recommended) or Hardcoded Values

# Try to load from Kaggle secrets first (more secure)
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    SEARCH_ENGINE_ID = user_secrets.get_secret("SEARCH_ENGINE_ID")
    
    print("âœ“ Loaded credentials from Kaggle Secrets")
    use_secrets = True
except Exception as e:
    # Fall back to hardcoded values if secrets not available
    print(f"âš ï¸�  Kaggle Secrets not available ({e})")
    print("Using hardcoded values instead...")
    
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
    SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID_HERE"
    use_secrets = False

# Project ID (usually doesn't need to be secret)
PROJECT_ID = "YOUR_GCP_PROJECT_ID"
LOCATION = "us-central1"

print("=" * 70)
print("âœ… CELL 4: Configuration Set!")
print("=" * 70)
print(f"Project ID: {PROJECT_ID}")
print(f"Location: {LOCATION}")

if use_secrets:
    print("âœ“ Using Kaggle Secrets (secure!)")
    print(f"  API Key: ***{GOOGLE_API_KEY[-4:] if len(GOOGLE_API_KEY) > 4 else '****'}")
    print(f"  Search Engine ID: ***{SEARCH_ENGINE_ID[-4:] if len(SEARCH_ENGINE_ID) > 4 else '****'}")
else:
    print(f"API Key: {'***' + GOOGLE_API_KEY[-4:] if len(GOOGLE_API_KEY) > 4 else '[NOT SET]'}")
    print(f"Search Engine ID: {'***' + SEARCH_ENGINE_ID[-4:] if len(SEARCH_ENGINE_ID) > 4 else '[NOT SET]'}")
    
print("=" * 70)
if not use_secrets and ("YOUR_" in GOOGLE_API_KEY or "YOUR_" in SEARCH_ENGINE_ID or "YOUR_" in PROJECT_ID):
    print("âš ï¸�  WARNING: Please set up Kaggle Secrets or replace placeholder values!")
    print("\nTO USE KAGGLE SECRETS (Recommended):")
    print("1. Click 'Add-ons' â†’ 'Secrets' in Kaggle notebook")
    print("2. Add secret: GOOGLE_API_KEY")
    print("3. Add secret: SEARCH_ENGINE_ID")
    print("4. Toggle 'Make secrets available to this notebook'")
else:
    print("âœ“ Credentials appear to be set (verify they are correct)")
print("=" * 70)




import logging
import requests
from typing import Optional, Dict, Any
import json

import google.generativeai as genai
from google.generativeai import types
from googleapiclient.discovery import build

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 70)
print("âœ… CELL 5: All Imports Successful!")
print("=" * 70)
print("âœ“ google.generativeai â†’ genai, types")
print("âœ“ googleapiclient â†’ Google Custom Search")
print("âœ“ requests â†’ HTTP client")
print("âœ“ logging â†’ Configured")
print("=" * 70)
print("Ready to define tools in the next cells!")
print("=" * 70)



# Define the actual weather function
def get_weather(latitude: float, longitude: float) -> str:
    """
    Fetches the current weather for a given latitude and longitude.
    
    Args:
        latitude: The latitude of the location (e.g., 40.7128 for NYC)
        longitude: The longitude of the location (e.g., -74.0060 for NYC)
    
    Returns:
        A JSON string containing the current weather information
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
            "forecast_days": 1
        }
        
        logger.info(f"Fetching weather for coordinates: ({latitude}, {longitude})")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        weather_data = response.json()
        logger.info(f"Weather data retrieved successfully")
        
        return json.dumps(weather_data)
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching weather data: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    except Exception as e:
        error_msg = f"Unexpected error in get_weather: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

# Define the tool declaration for Gemini (using dictionary, not types.Schema)
get_weather_declaration = {
    "name": "get_weather",
    "description": "Fetches the current weather for a given latitude and longitude using the Open-Meteo API",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "The latitude of the location (e.g., 40.7128 for NYC)"
            },
            "longitude": {
                "type": "number",
                "description": "The longitude of the location (e.g., -74.0060 for NYC)"
            },
        },
        "required": ["latitude", "longitude"]
    }
}

print("=" * 70)
print("âœ… CELL 7: Weather Tool Defined!")
print("=" * 70)
print("âœ“ Function: get_weather(latitude, longitude)")
print("âœ“ API: Open-Meteo (free weather data)")
print("âœ“ Tool Declaration: Dictionary format")
print("=" * 70)
print("Testing weather tool...")
try:
    test_result = get_weather(40.7128, -74.0060)  # NYC coordinates
    print("âœ“ Weather tool test successful!")
    print(f"  Sample data: {test_result[:100]}...")
except Exception as e:
    print(f"âš ï¸�  Weather tool test failed: {e}")
print("=" * 70)


      


# Define the actual Google search function
def google_search(query: str, num_results: int = 5) -> str:
    """
    Performs a Google search and returns the top results.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5)
    
    Returns:
        A JSON string with search results
    """
    try:
        logger.info(f"Performing Google search: {query}")
        # Build the Google Custom Search service client
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        # Execute the search query using the Custom Search Engine
        result = service.cse().list(
            q=query,
            cx=SEARCH_ENGINE_ID,
            num=num_results
        ).execute()
        
        if 'items' not in result:
            return json.dumps({"results": [], "message": "No search results found"})
        
        # Format search results
        results = []
        for item in result['items']:
            results.append({
                "title": item.get('title', 'No title'),
                "url": item.get('link', 'No link'),
                "description": item.get('snippet', 'No description')
            })
        
        return json.dumps({"results": results})
    
    except Exception as e:
        error_msg = f"Error performing Google search: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

# Define the tool declaration for Gemini (using dictionary)
google_search_declaration = {
    "name": "google_search",
    "description": "Performs a Google search and returns top results for travel-related queries including flights, hotels, and destinations",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 5)"
            },
        },
        "required": ["query"]
    }
}

print("=" * 70)
print("âœ… CELL 9: Google Search Tool Created!")
print("=" * 70)
print("âœ“ Function: google_search(query, num_results=5)")
print("âœ“ API: Google Custom Search")
print("âœ“ Tool Declaration: Dictionary format")
print("=" * 70)
print("Search tool is ready to use!")
print("=" * 70)



# Configure Gemini API  
genai.configure(api_key=GOOGLE_API_KEY)

# System instruction for the travel agent
system_instruction = """You are a helpful concierge travel agent. You can help users find 
flights, hotels, and check the weather for their destination.

When you need to check the weather, use the 'get_weather' function.
When you need to search for travel information, use the 'google_search' function.

Always be helpful, friendly, and provide detailed travel recommendations."""

# Wrap function declarations in a tools list
tools = [{
    "function_declarations": [
        get_weather_declaration,
        google_search_declaration
    ]
}]

# Create the Generative Model with tools
print("=" * 70)
print("Creating Gemini model with tools...")
print("=" * 70)

model = genai.GenerativeModel(
    'gemini-2.0-flash-exp',
    tools=tools,
    system_instruction=system_instruction
)

print("=" * 70)
print("âœ… CELL 11: AI Model Created Successfully!")
print("=" * 70)
print("âœ“ Model: gemini-2.0-flash-exp")
print("âœ“ Tools registered:")
print("  - get_weather (Open-Meteo API)")
print("  - google_search (Custom Search API)")
print("âœ“ System instructions configured")
print("=" * 70)
print("Model is ready to chat!")
print("=" * 70)



# ============================================================================
# CELL 13: START INTERACTIVE CHAT
# ============================================================================
# Run this cell to start chatting with your travel agent!

"""
HOW TO USE THE CHAT INTERFACE:
-------------------------------
1. Run this cell to start the interactive chat
2. Type your travel-related questions
3. The agent will use tools automatically when needed
4. Type 'exit', 'quit', or 'bye' to end the conversation

EXAMPLE QUERIES TO TRY:
------------------------
1. "What's the weather like in Paris? The coordinates are 48.8566, 2.3522"
2. "Find me the best hotels in Tokyo"
3. "What are the top tourist attractions in New York City?"
4. "Search for cheap flights from San Francisco to London"
5. "What's the current weather in Sydney? Latitude -33.8688, longitude 151.2093"
"""

# Function mapping for tool execution
functions = {
    'get_weather': get_weather,
    'google_search': google_search
}

def chat_with_agent():
    """Main chat loop with function calling support"""
    
    print("=" * 70)
    print("ğŸŒ� WELCOME TO YOUR AI TRAVEL AGENT! âœˆï¸�")
    print("=" * 70)
    print("\nI'm your personal travel concierge, powered by:")
    print("  ğŸ§  Google Gemini 2.0 (AI brain)")
    print("  ğŸŒ¤ï¸�  Open-Meteo API (weather data)")
    print("  ğŸ”� Google Custom Search (web information)")
    print("\n" + "-" * 70)
    print("WHAT I CAN DO:")
    print("-" * 70)
    print("  âœ“ Find flights, hotels, and travel deals")
    print("  âœ“ Check weather conditions at any location")
    print("  âœ“ Recommend tourist attractions and activities")
    print("  âœ“ Provide travel tips and advice")
    print("  âœ“ Remember our conversation context")
    print("\n" + "-" * 70)
    print("TIPS:")
    print("-" * 70)
    print("  â€¢ Be specific with your requests")
    print("  â€¢ Provide coordinates when asking about weather")
    print("  â€¢ I can search for coordinates if you give me a city name")
    print("  â€¢ Type 'exit', 'quit', or 'bye' to end our chat")
    print("=" * 70)
    print("\nReady to help with your travel plans! ğŸ�’\n")
    
    # Start chat session
    chat = model.start_chat()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'stop']:
                print("\nğŸ‘‹ Thanks for chatting! Have a great trip!")
                break
            
            if not user_input:
                continue
            
            # Generate response
            response = chat.send_message(user_input)
            
            # Check if there's a function call in the response
            function_calls_made = False
            
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls_made = True
                    function_call = part.function_call
                    function_name = function_call.name
                    function_args = dict(function_call.args)
                    
                    print(f"\nğŸ”§ Using tool: {function_name}({', '.join(f'{k}={v}' for k, v in function_args.items())})")
                    
                    # Execute the function
                    if function_name in functions:
                        result = functions[function_name](**function_args)
                    else:
                        result = json.dumps({"error": f"Unknown function: {function_name}"})
                    
                    # Send function result back to model
                    response = chat.send_message({
                        "function_response": {
                            "name": function_name,
                            "response": {"result": result}
                        }
                    })
            
            # Get the final text response
            try:
                agent_response = response.text
                print(f"\nAgent: {agent_response}\n")
            except ValueError:
                # If no text, response might only have function calls
                if not function_calls_made:
                    print("\nAgent: I'm processing your request...\n")
            
        except KeyboardInterrupt:
            print("\n\nğŸ‘‹ Chat interrupted. Thanks for using the travel agent!")
            break
        except Exception as e:
            print(f"\nâ�Œ Error: {e}")
            print("Let's try again. You may need to rephrase your question.\n")

# Start the chat
try:
    chat_with_agent()
except Exception as e:
    print(f"\n\nâ�Œ Fatal error: {e}")
    print("You may need to restart the kernel and run all cells again.")
              


