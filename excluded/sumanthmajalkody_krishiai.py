import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import logging
import warnings

# Suppress specific Google GenAI warnings
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
logging.getLogger("google.genai.types").setLevel(logging.ERROR)

# Suppress general warnings to keep the output clean
warnings.filterwarnings("ignore")


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
# Install first: !pip install gTTS
from gtts import gTTS
from IPython.display import Audio, display

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)
print("âœ… Retry confguration complete successfully.")


import requests

def get_real_weather(location: str):
    """
    Fetches agricultural weather data (Soil Moisture, Rain Volume) for yield prediction.
    Args:
        location: City or district name (e.g., Mandya).
    """
    try:
        # 1. Geocoding (Get Lat/Lon for the city)
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo = requests.get(geo_url, params={"name": location, "count": 1, "format": "json"}).json()
        
        if not geo.get('results'):
            return f"Error: Location '{location}' not found."
            
        lat = geo['results'][0]['latitude']
        lon = geo['results'][0]['longitude']
        
        # 2. Fetch AGRI Data (Soil Moisture + Rain Sum)
        # We specifically ask for 'rain_sum' (mm) and 'soil_moisture_0_to_1cm'
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "precipitation_probability_max", "rain_sum"],
            "hourly": ["soil_moisture_0_to_1cm"],
            "timezone": "auto"
        }
        
        w = requests.get(weather_url, params=params).json()
        
        # 3. Process Data
        daily = w['daily']
        # Calculate average soil moisture for the next 24 hours
        avg_soil_moisture = sum(w['hourly']['soil_moisture_0_to_1cm'][:24]) / 24 
        
        # Determine status for the LLM
        rain_mm = daily['rain_sum'][0]
        rain_status = "Rain" if rain_mm > 0.5 else "Clear"
        
        return {
            "location": location,
            "max_temp": f"{daily['temperature_2m_max'][0]}Â°C",
            "rain_chance": f"{daily['precipitation_probability_max'][0]}%",
            "rainfall_amount": f"{rain_mm} mm",     # <-- Vital for Yield Calc
            "soil_moisture": f"{avg_soil_moisture:.2f} mÂ³/mÂ³", # <-- Vital for Yield Calc
            "general_status": rain_status
        }
        
    except Exception as e:
        return f"Weather Tool Error: {str(e)}"

# Weather Agent: Its job is to use the google_search tool and present weather findings.
weather_agent = Agent(
    name="WeatherAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Fetch real-time weather.
    CRITICAL OUTPUT: You MUST explicitly state "RAIN STATUS: [Rain/Clear/Dry]".
    """,
    tools=[get_real_weather],
    output_key="weather_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… weather_agent created.")


# search_market_prices_agent: Its job is to use the google_search tool and present price findings.
search_market_prices_agent = Agent(
    name="PriceAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    Searches the web for current agricultural market prices.
    Args:
        query: The search query (e.g., 'Tomato price in Kolar today').
    """,
    tools=[google_search],
    output_key="price_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… search_market_prices_agent created.")


# --- AGENT 3: SCHEMES (New Feature) ---
scheme_agent = Agent(
    name="SchemeAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are an expert on Indian Government Agricultural Schemes.
    Search for subsidies, insurance (Pradhan Mantri Fasal Bima Yojana), and loans relevant to the farmer's query.
    Summarize eligibility and benefits clearly.
    """,
    tools=[google_search],
)

print("âœ… scheme_agent created.")


def calculate_yield_math(area_in_acres: float = 1.0, base_yield_per_acre: float = 10.0, weather_factor: float = 1.0):
    """
    Pure math tool for yield prediction. 
    ALL arguments are optional to prevent Agent hesitation.
    """
    try:
        # 1. Safe Defaults Logic
        if not area_in_acres: area_in_acres = 1.0
        if not base_yield_per_acre: base_yield_per_acre = 10.0
        if not weather_factor: weather_factor = 1.0
        
        # 2. Calculate
        predicted_yield = area_in_acres * base_yield_per_acre * weather_factor
        
        return {
            "formula": f"{area_in_acres} acres * {base_yield_per_acre} q/acre * {weather_factor} (weather impact)",
            "prediction": f"{predicted_yield:.2f} Quintals",
            "status": "Success"
        }
    except Exception as e:
        return {"error": str(e)}

# --- AGENT: AGRONOMIST (The Yield Expert) ---
agronomist_agent = Agent(
    name="Agronomist",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are an expert Agronomist. 
    You predict yield for ANY crop (Orange, Wheat, Rose, etc.) using your internal knowledge + the math tool.
    
    STEP 1: DECIDE INPUTS (Do this internally):
    - 'base_yield_per_acre': Estimate average yield in Quintals for the specific crop based on your knowledge. 
      (e.g., Orange ~ 80, Wheat ~ 12).
    - 'weather_factor': 
      * If Weather is "Rain" and crop hates rain -> 0.7
      * If Weather is "Clear" and crop likes sun -> 1.1
      * If Weather is "Unknown" -> 1.0
      
    STEP 2: RUN TOOL
    - Call 'calculate_yield_math'.
    - Pass your estimated 'base_yield_per_acre' and 'weather_factor'.
    - Pass 'area_in_acres' (if user didn't say, DO NOT ASK. Just send 1.0).
    """,
    tools=[calculate_yield_math],
)

print("âœ… Agronomist created.")


# --- ROOT AGENT (The Holistic Advisor) ---
root_agent = Agent(
    name="KrishiAI",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are KrishiAI, a comprehensive agricultural expert.
    
    COMPREHENSIVE REPORTING PROTOCOL:
    If the user asks about "growing a crop", "predicting yield", or "planning farming", follow these steps strictly:
    
    1. YIELD PREDICTION: 
       - Call 'WeatherAgent' to get the weather string. (Capture the output).
       - IMMEDIATELY Call 'Agronomist'.
       - CRITICAL: You MUST include the weather string from Step 1 in your message to the Agronomist. 
       - Prompt Example: "Predict yield for [Crop] in [Area]. Weather report: [Insert Weather String]."
       - If user didn't specify area, tell Agronomist to use "1 acre".
       
    2. FINANCIALS (Market & Schemes):
       - Call 'PriceAgent' to find current market prices.
       - Call 'SchemeAgent' to find 1-2 relevant government subsidies.
       
    3. SYNTHESIS:
       - Combine all findings into a structured answer:
         * ğŸŒ¦ï¸� **Weather & Yield:** [Insert Yield details]
         * ğŸ’° **Market Trends:** [Insert Price details]
         * ğŸ“œ **Available Schemes:** [Insert Scheme details]
         
    4. MOTIVATION:
       - Always end with a motivating proverb in the same language as the answer.
    
    CRITICAL: DO NOT ask the user for data. If you are missing something, USE INTELLIGENT DEFAULTS and RUN THE TOOLS immediately.
    """,
    # Register ALL agents
    tools=[
        AgentTool(weather_agent), 
        AgentTool(search_market_prices_agent), 
        AgentTool(agronomist_agent),
        AgentTool(scheme_agent)
    ],
)

print("âœ… Root Agent created.")


# 1. Install Audio Library (if not already installed)
!pip install -q gTTS

import asyncio
import io
import sys
from gtts import gTTS
from IPython.display import Audio, display, clear_output
from contextlib import redirect_stdout

runner = InMemoryRunner(agent=root_agent)

# --- ğŸšœ KrishiAI Smart Demo ---
async def start_demo():
    print("--- ğŸšœ KrishiAI Terminal (Type 'quit' to exit) ---")
    
    # We use a fixed session to keep memory
    session_id = "demo_session_v1"
    
    while True:
        user_input = input("\nFarmer: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Bye Farmer! Remember me if you need anything")
            print("HAPPY FARMING ğŸšœ")
            break
            
        print("Thinking...", end="")
        
        # --- THE FIX: Capture the output of run_debug ---
        # Since runner.run() crashes, we use run_debug() and record what it says.
        capture_buffer = io.StringIO()
        
        try:
            with redirect_stdout(capture_buffer):
                # This prints to our hidden buffer instead of the screen
                await runner.run_debug(user_input)
                
            # Now we get the text back
            full_output = capture_buffer.getvalue()
            
            # Show the text to the user
            print(f"\r{full_output}") 
            
            # Extract just the Agent's answer for Audio
            # We look for the part after the ">" symbol
            if ">" in full_output:
                agent_response = full_output.split(">")[-1].strip()
                
                # Generate Audio
                if len(agent_response) > 0:
                    tts = gTTS(agent_response, lang='en')
                    tts.save('krishi_reply.mp3')
                    print("Play the below audio to listen to my response.")
                    print("I can also translate my response in your regional language.")
                    print("  - Just say Tell me same in Kannada / Regional Language of your choice ")
                    display(Audio('krishi_reply.mp3', autoplay=True))
                    
        except Exception as e:
            print(f"\nâ�Œ Error: {e}")

# Start the loop
await start_demo()

