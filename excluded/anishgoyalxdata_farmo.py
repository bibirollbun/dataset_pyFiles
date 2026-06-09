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


import os
import requests
import json
from typing import Any, Dict, Literal
from google.genai import types
from google.adk.tools import FunctionTool, AgentTool
# CRITICAL: Added AgentTool here as it is required for coordinator_agent
from google.adk.agents import Agent, LlmAgent 
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
print("âœ… ADK components imported successfully.")


APP_NAME = "default"
USER_ID = "default"
SESSION = "default"
MODEL_NAME = 'gemini'


async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    app_name = runner_instance.app_name
    session_service = runner_instance.session_service # Get the service from the runner

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    if user_queries:
        if type(user_queries) == str:
            user_queries = [user_queries]

        for query in user_queries:
            print(f"\nUser > {query}")

            query = types.Content(role="user", parts=[types.Part(text=query)])

            # CRITICAL FIX: Ensure runner.run_async() is used with the 'new_message' argument
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                if event.content and event.content.parts:
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


import os
from kaggle_secrets import UserSecretsClient

try:
    WEATHER_API_KEY = UserSecretsClient().get_secret("WEATHER_API_KEY")
    os.environ["WEATHER_API_KEY"] = WEATHER_API_KEY
    print("âœ… Weather API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'WEATHER_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import requests
import json
from typing import Literal

# Your actual API Key and Base URL
# IMPORTANT: Never expose API keys in a public repository or final code.
# For this capstone, we are demonstrating the functional structure.
WEATHER_BASE_URL = "http://api.weatherapi.com/v1/current.json"

def get_live_weather_forecast(location: str) -> str:
    """Retrieves the live, current weather conditions for the specified location using the WeatherAPI."""
    
    try:
        # Construct the dynamic request URL
        url = f"{WEATHER_BASE_URL}?key={WEATHER_API_KEY}&q={location}&aqi=no"
        
        # Make the actual API call
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        raw_data = response.json()
        
        # Extract and simplify key information for the LLM to process
        current_temp = raw_data['current']['temp_c']
        condition = raw_data['current']['condition']['text']
        humidity = raw_data['current']['humidity']
        city = raw_data['location']['name']
        
        # Return the processed, raw data for the LLM to interpret and advise upon
        return f"""SUCCESS: Live Data Retrieved.
        Location: {city}
        Temperature: {current_temp}Â°C
        Condition: {condition}
        Humidity: {humidity}%
        """
        
    except requests.exceptions.RequestException as e:
        return f"ERROR: API request failed. Status code or connection error: {e}"
    except Exception as e:
        return f"ERROR: Could not parse weather data: {e}"

# Wrap the live function in an ADK Tool
weather_tool = FunctionTool(get_live_weather_forecast)

print("âœ… weather functional tool defined.")


# --- TEST EXECUTION BLOCK ---
test_location = "Raipur"
print("\n--- Starting Direct Weather API Test ---")

result = get_live_weather_forecast(test_location)

print("\n--- Final Function Output ---")
print(result)


# RAG Data Source (Hard-Coded Soil Database - EXPANDED to over 20 crops)
SOIL_DATA = {
    # Cereals
    "WHEAT":    {"NPK": "120:60:40", "pH_range": "6.0-7.5", "Advice": "Requires regular Nitrogen top-dressing."},
    "RICE":     {"NPK": "100:50:50", "pH_range": "5.5-6.5", "Advice": "Prefers slightly acidic soil; manage water carefully."},
    "MAIZE":    {"NPK": "150:75:50", "pH_range": "5.8-7.0", "Advice": "High Nitrogen demand, especially early in the season."},
    "BARLEY":   {"NPK": "80:40:20",  "pH_range": "6.0-7.0", "Advice": "Tolerates saline soil, needs less nitrogen than wheat."},
    "JOWAR":    {"NPK": "80:40:40",  "pH_range": "6.0-8.0", "Advice": "A hardy millet, responsive to Phosphorous application."},
    
    # Pulses (Known for Nitrogen-fixing, thus lower N required)
    "GRAM":     {"NPK": "20:40:20",  "pH_range": "6.0-7.5", "Advice": "Leguminous; minimal Nitrogen required, focus on P & K."},
    "ARHAR":    {"NPK": "20:60:20",  "pH_range": "6.5-7.5", "Advice": "Deep-rooted, needs good drainage and adequate Phosphorous."},
    "MOONG":    {"NPK": "20:40:20",  "pH_range": "6.0-7.0", "Advice": "Short duration crop; balance NPK for quick growth."},
    "LENTIL":   {"NPK": "20:40:40",  "pH_range": "6.0-8.0", "Advice": "Very sensitive to waterlogging; ensures adequate Potassium."},
    "SOYBEAN":  {"NPK": "30:80:40",  "pH_range": "6.0-7.5", "Advice": "Key is Phosphorous and pre-sowing seed inoculation."},

    # Cash & Oilseed Crops
    "SUGARCANE": {"NPK": "200:80:60", "pH_range": "6.5-7.5", "Advice": "Highest NPK demand; multiple Nitrogen splits are critical."},
    "COTTON":   {"NPK": "100:50:50", "pH_range": "5.5-7.5", "Advice": "High Potassium needed for boll development."},
    "MUSTARD":  {"NPK": "80:40:40",  "pH_range": "6.0-7.5", "Advice": "Sulphur is critical for oil content; apply gypsum."},
    "GROUNDNUT": {"NPK": "20:40:40",  "pH_range": "6.0-7.0", "Advice": "Requires Gypsum (Calcium and Sulphur) for pod filling."},
    "SUNFLOWER": {"NPK": "60:80:60",  "pH_range": "6.5-7.5", "Advice": "High B and P requirements for maximum seed yield."},
    
    # Horticulture/Spices/Plantation
    "POTATO":   {"NPK": "150:100:150", "pH_range": "4.8-6.0", "Advice": "Very high K (Potassium) demand for tuber sizing. Prefers acidic soil."},
    "ONION":    {"NPK": "100:50:80", "pH_range": "6.0-7.5", "Advice": "Needs Sulphur for pungency and bulb quality."},
    "CHILLI":   {"NPK": "120:60:60", "pH_range": "6.0-7.0", "Advice": "Needs balanced NPK, adequate Calcium for fruit set."},
    "TEA":      {"NPK": "150:75:75", "pH_range": "4.5-5.5", "Advice": "Extremely acid-loving plant. Avoid high pH soils."},
    "COFFEE":   {"NPK": "80:60:80",  "pH_range": "6.0-6.5", "Advice": "Needs regular foliar spray of micronutrients."},
}

def get_fertilizer_recommendation(crop_name: str) -> str:
    """
    Searches the local knowledge base for NPK ratios and soil pH requirements for the specified crop.
    
    Args:
        crop_name: The name of the crop (e.g., 'Wheat', 'Rice', 'Cotton').
        
    Returns:
        A success or failure message with the detailed fertilizer and soil advice.
    """
    
    # Capitalize and strip to match dictionary keys
    crop = crop_name.upper().strip()
    
    if crop in SOIL_DATA:
        data = SOIL_DATA[crop]
        # Return a structured SUCCESS string for the LLM to interpret
        return f"""SUCCESS: Found data for {crop_name}. 
        - Recommended NPK Ratio (kg/hectare): {data['NPK']} 
        - Ideal Soil pH Range: {data['pH_range']} 
        - Key Advisory: {data['Advice']}"""
    else:
        # Return a structured FAILURE string
        return f"FAILURE: Detailed fertilizer data for '{crop_name}' not available in the knowledge base. Please try a different crop."


# Wrap the local function as an ADK tool
fertilizer_tool = FunctionTool(get_fertilizer_recommendation)

print("âœ… Fertilizer information functional tool defined.")


weather_agent = LlmAgent(
    model='gemini-2.5-flash',
    name="WeatherAdviser",
    instruction="You are a weather expert. Use the get_live_weather_forecast tool to provide current conditions and a brief farming advisory.",
    tools=[weather_tool]
)

fertilizer_agent = LlmAgent(
    model='gemini-2.5-flash',
    name="SoilFertilizerAdviser",
    instruction="You are a soil health expert. Use the get_fertilizer_recommendation tool to retrieve data, and clearly present the NPK ratio, pH, and advisory notes.",
    tools=[fertilizer_tool]
)

# --- CRITICAL FIX: Memory instruction added to GeneralInfoAdviser ---
info_agent = LlmAgent(
    model='gemini-2.5-flash',
    name="GeneralInfoAdviser",
    instruction="""You are a general agriculture expert. Your job is to answer non-specific queries about farming, crop cycles, or government schemes using your internal knowledge. 
    
    CRITICAL RULE: If the user asks a question about a PREVIOUS QUERY, conversation history, or what they asked before, you MUST use the provided context to recall and answer the question accurately. Do NOT claim you have no memory.
    
    You have NO external tools.""",
    tools=[]
)

coordinator_agent = LlmAgent(
    model='gemini-2.5-flash',
    name="FarmerQueryCoordinator",
    instruction="""You are the main assistant and router for farmers' queries. Your primary goal is to determine if a tool is needed.
    
    1. If the query asks about **conversation history** (e.g., 'what did I ask?', 'what was the last query?'), you MUST answer the question directly using the conversation history provided. Do NOT delegate this question.
    
    2. If a tool is needed, delegate to the SINGLE best specialist agent:
        - SoilFertilizerAdviser for NPK/soil pH.
        - WeatherAdviser for rain/temp/forecast.
        - GeneralInfoAdviser for general knowledge/schemes.
        
    Your only action is to either answer the memory question directly or call the correct specialist tool.""",
    tools=[
        AgentTool(fertilizer_agent), 
        AgentTool(weather_agent), 
        AgentTool(info_agent)
    ]
)


session_service = InMemorySessionService()
runner = Runner(agent=coordinator_agent, app_name=APP_NAME, session_service=session_service)


await run_session(
    runner,
    [
        "What NPK is recommended for my rice crop, and what pH should I maintain?",
        "what crop query did i ask?",  # This query should now be answered correctly!
    ],
    "query1-session",
)

