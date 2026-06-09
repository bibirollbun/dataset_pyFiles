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


!pip install -q google-genai requests



from kaggle_secrets import UserSecretsClient
from google import genai

# Load keys
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

print("GOOGLE_API_KEY found:", bool(api_key))

# Initialize client
client = genai.Client(api_key=api_key)
print("GenAI client initialized!")



# We use the safe model from your list:
MODEL = "models/gemini-2.5-flash"
print("Using MODEL =", MODEL)



import requests
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
owm_key = user_secrets.get_secret("OPENWEATHER_API_KEY")
print("OPENWEATHER_API_KEY found:", bool(owm_key))

def get_destination_weather(city: str) -> str:
    """Get real weather data using OpenWeatherMap."""
    if not owm_key:
        return f"Sunny, 22Â°C in {city} (mocked fallback)"

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": owm_key, "units": "metric"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels = data["main"].get("feels_like")
        wind = data.get("wind", {}).get("speed")

        text = f"{desc}, {temp}Â°C"
        if feels is not None:
            text += f" (feels like {feels}Â°C)"
        if wind is not None:
            text += f", wind {wind} m/s"
        
        return text

    except Exception as e:
        print("Weather API failed:", e)
        return f"Sunny, 22Â°C in {city} (fallback)"



from google.genai import types

weather_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_destination_weather",
            description="Get current weather for a city.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"city": types.Schema(type=types.Type.STRING)},
                required=["city"],
            ),
        )
    ]
)



import json, time, traceback
from google.genai import types

chat_history = []

def safe_extract(response):
    """Extract text safely from Gemini response."""
    if hasattr(response, "text") and response.text:
        return response.text

    try:
        cand = response.candidates[0]
        parts = cand.content.parts
        return " ".join([getattr(p, "text", str(p)) for p in parts])
    except:
        return str(response)

def handle_calls(response):
    """Handle function calls (like weather calls)."""
    if not response.function_calls:
        return False

    for call in response.function_calls:
        args = call.args
        if isinstance(args, str):
            args = json.loads(args)

        city = args.get("city")

        print(f"[AGENT THOUGHT] Calling tool: {call.name}, city={city}")
        result = get_destination_weather(city)

        print("[TOOL OUTPUT]:", result)

        tool_part = types.Part.from_function_response(
            name="get_destination_weather",
            response={"result": result}
        )

        chat_history.append(response.candidates[0].content)
        chat_history.append(types.Content(role="tool", parts=[tool_part]))

        final = client.models.generate_content(model=MODEL, contents=chat_history)
        print("ðŸ¤– Final Answer:", safe_extract(final))

        chat_history.append(final.candidates[0].content)
        return True

    return False

def agent_chat(user_input):
    print("\nðŸ‘¤ USER:", user_input)
    
    chat_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=chat_history,
            config=types.GenerateContentConfig(tools=[weather_tool])
        )

        if handle_calls(response):
            return
        
        text = safe_extract(response)
        print("ðŸ¤– AGENT:", text)

        chat_history.append(response.candidates[0].content)

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()



agent_chat("I want to go to Tokyo.")
time.sleep(1)
agent_chat("What is the weather?")


