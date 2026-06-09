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


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


# Define helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


# Install Project AEROGUARD libraries
!pip install -U google-generativeai openmeteo-requests requests-cache retry-requests census us


import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

print("ğŸ”‘ Setting up Access Credentials...")

try:
    user_secrets = UserSecretsClient()
    
    # 1. Google Cloud Key (The Brain)
    google_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=google_key)
    print("   âœ… Google API Key loaded.")
    
    # 2. Census Bureau Key (The Demographer)
    # We set this as an env var for the library to use automatically
    os.environ["CENSUS_API_KEY"] = user_secrets.get_secret("CENSUS_API_KEY")
    print("   âœ… Census API Key loaded.")
    
except Exception as e:
    print(f"   â�Œ Secret Error: {e}")
    print("   ğŸ‘‰ Go to 'Add-ons' > 'Secrets' and verify your keys.")


import openmeteo_requests
import requests_cache
from retry_requests import retry
from census import Census
from us import states

# --- Tool 1: The Air Quality Sentinel ---
def check_air_quality_risk(lat: float, lon: float) -> dict:
    """
    Retrieves real-time Air Quality data (AQI, PM2.5) for a location.
    Checks for smog, wildfire smoke, and hazardous particulate matter.
    """
    print(f"ğŸŒ«ï¸� [AeroGuard Sentinel] Sampling air quality at: {lat}, {lon}...")
    
    # Setup Robust Client with Caching
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["us_aqi", "pm2_5", "carbon_monoxide"],
        "timezone": "auto"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        current = response.Current()
        
        us_aqi = current.Variables(0).Value()
        pm2_5 = current.Variables(1).Value()
        
        # Risk Logic
        risk_level = "LOW"
        if us_aqi > 150: risk_level = "CRITICAL (Hazardous - Deploy Oxygen)"
        elif us_aqi > 100: risk_level = "HIGH (Unhealthy)"
        elif us_aqi > 50: risk_level = "MODERATE"
        
        return {
            "status": "success",
            "source": "Open-Meteo Real-Time Network",
            "data": {
                "US_AQI": int(us_aqi),
                "PM2_5": round(pm2_5, 2),
                "risk_assessment": risk_level
            }
        }
    except Exception as e:
        return {"error": f"Air Quality Check Failed: {e}"}

# --- Tool 2: The Census Demographer ---
def get_vulnerable_population(zip_code: str) -> dict:
    """
    Queries US Census data to find the ratio of elderly populations (Age 65+).
    Used to identify communities most vulnerable to respiratory distress.
    """
    print(f"ğŸ‘¥ [Census Demographer] Analyzing demographics for Zip: {zip_code}...")
    
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key: return {"status": "error", "msg": "Missing Census Key"}

    try:
        c = Census(api_key)
        # B01001_001E = Total Pop, B01001_020E = Males 65-66 (Base for extrapolation)
        data = c.acs5.state_zipcode(['NAME', 'B01001_001E', 'B01001_020E'], states.CA.fips, zip_code)
        
        if data:
            total = int(data[0]['B01001_001E'])
            # Extrapolate total elderly from the specific age bracket returned
            elderly_estimate = int(data[0]['B01001_020E']) * 12 
            ratio = round(elderly_estimate / total, 2)
            
            risk = "LOW"
            if ratio > 0.20: risk = "CRITICAL"
            elif ratio > 0.15: risk = "HIGH"
            
            return {
                "status": "success",
                "zip_code": zip_code,
                "demographics": {
                    "total_pop": total,
                    "elderly_ratio": ratio
                },
                "vulnerability": risk
            }
        return {"status": "error", "message": "Zip code not found in Census DB"}
            
    except Exception as e:
        return {"error": str(e)}

print("âœ… AEROGUARD Tools registered.")


# Configure the "Brain"
target_model = "models/gemini-flash-latest" 

print(f"ğŸš€ Initializing AEROGUARD with: {target_model}")

model = genai.GenerativeModel(
    model_name=target_model, 
    # Bind our Python functions directly to the model
    tools=[check_air_quality_risk, get_vulnerable_population],
    system_instruction="""
    You are AEROGUARD, an Autonomous Public Health Agent.
    
    YOUR PROTOCOL:
    1. RECEIVE a health alert for a specific location/zip code.
    2. CALL 'check_air_quality_risk' to get real-time AQI levels.
    3. CALL 'get_vulnerable_population' to assess the density of at-risk elderly residents.
    4. SYNTHESIZE a "HEALTH DEFENSE PLAN":
       - If AQI > 100 AND Vulnerability is HIGH/CRITICAL: Recommend immediate mobile clinic dispatch.
       - If AQI is LOW: Recommend monitoring only.
    
    Always output your final plan in a structured, professional format.
    """
)
print("âœ… Agent Online.")


import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown
import time

# 1. Setup Authentication
try:
    user_secrets = UserSecretsClient()
    google_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=google_key)
    print("âœ… Google API Key loaded.")
except Exception as e:
    print("â�Œ ERROR: Please add 'GOOGLE_API_KEY' to Kaggle Secrets.")

# 2. Define the Mission
mission_prompt = """
ALERT: Respiratory distress reports spiking in Zip Code 90012 (Los Angeles, CA).
Coordinates: 34.0522, -118.2437.
Assess the air quality threat and the local vulnerable population to plan a response.
"""

# 3. Configure Model with "Report Mode" Instructions
target_model = "models/gemini-flash-latest" 

print(f"ğŸš€ Initializing AEROGUARD with: {target_model}")

model = genai.GenerativeModel(
    model_name=target_model, 
    # FIX: Updated to use the correct AEROGUARD tool names
    tools=[check_air_quality_risk, get_vulnerable_population],
    system_instruction="""
    You are AEROGUARD, an Autonomous Public Health Agent.
    
    PROTOCOL:
    1. Assess the air quality threat using the 'check_air_quality_risk' tool.
    2. Assess the human vulnerability (elderly population) using the 'get_vulnerable_population' tool.
    3. Synthesize a "HEALTH DEFENSE PLAN".
    
    FORMATTING RULES:
    - Output the final report in **Markdown**.
    - Use '### Headers' for sections.
    - Use '**Bold**' for critical metrics (AQI, Elderly Ratio).
    - Use a Markdown Table for the Logistics Plan.
    - If data is missing, estimate and flag it.
    """
)

# 4. Execute with Pretty Printing
max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"\n--- ğŸš¨ INCOMING HEALTH ALERT (Attempt {attempt+1}) ğŸš¨ ---")
        print("Thinking... (Connecting to Open-Meteo and Census Bureau streams)")
        
        chat = model.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(mission_prompt)
        
        # --- THE MAGIC LINE: Renders the text as rich HTML/Markdown ---
        display(Markdown(response.text)) 
        
        # PROOF OF WORK: Trace
        print("\n--- ğŸ•µï¸� ORCHESTRATION TRACE ---")
        tools_used = False
        for part in chat.history:
            if part.role == "model" and part.parts[0].function_call:
                tool_name = part.parts[0].function_call.name
                print(f"ğŸ”¹ Agent Called Tool: {tool_name}")
                tools_used = True
            if part.role == "user" and part.parts[0].function_response:
                print(f"ğŸ”¸ Tool Returned Data")
        
        if not tools_used:
            print("âš ï¸� Note: Agent reasoned without tools.")
            
        break 

    except Exception as e:
        print(f"âš ï¸� Error on attempt {attempt+1}: {e}")
        if "429" in str(e):
            time.sleep(10)
        else:
            break

