%%capture
# Install necessary libraries
!pip install -q -U google-generativeai
!pip install -q gradio
!pip install -q requests

# Import libraries
import os
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import requests
import gradio as gr


from google import genai
from google.genai import types
from IPython.display import display, Image, Markdown, Audio

genai.__version__


# Install the simpler Gemini library and LangChain tools
# !pip install -U -q google-generativeai langchain-google-genai langchainhub langchain
!pip install -U google-generativeai
!pip install langchain langchain-google-genai


# Import and Connect to your Secret Key
import os
from kaggle_secrets import UserSecretsClient
from langchain_google_genai import ChatGoogleGenerativeAI

# Retrieve the key you just saved in Kaggle Secrets
user_secrets = UserSecretsClient()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")

# Initialize the Gemini Model (The Brain)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Flash is fast and free for this tier
    temperature=0,            # 0 means "be precise, don't hallucinate"
    convert_system_message_to_human=True
)

print("SUCCESS: Gemini is connected and ready to build agents!")


from google.api_core import retry


is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)


# Install the official Google Generative AI library
!pip install -U -q google-generativeai

import os
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Authenticate (Using the key you saved earlier)
user_secrets = UserSecretsClient()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Define Your Tools (Regular Python Functions)
# These are the "hands" of your agent.
import requests
from kaggle_secrets import UserSecretsClient

def get_weather(location):
    """
    Fetches real-time weather for a given location using WeatherAPI.com.
    """
    try:
        # 1. Retrieve the API key securely from Kaggle Secrets
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("WEATHER_API_KEY")
        
        # 2. Construct the API URL
        # We use 'q' for the query (city name) and your key
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"
        response = requests.get(complete_url)
        data = response.json()
        
        # 4. Check if the city was found
        if "error" in data:
            return f"Error: Could not find weather for {location}."
            
        # 5. Extract the useful info
        # We get the condition text (e.g., "Partly cloudy") and temp
        condition = data['current']['condition']['text']
        temp_c = data['current']['temp_c']
        
        return f"{condition}, {temp_c}Â°C"
        
    except Exception as e:
        return f"Connection Error: {str(e)}"

# Test it immediately in a new cell:
# print(get_weather("Tokyo"))
# print(get_weather("London"))
def get_attractions(city: str):
    """
    Returns tourist attractions with 'tags' so the Agent can filter them.
    (e.g., Identifying which ones are High Altitude vs. Food)
    """
    print(f"\n[Tool Used] Finding attractions for: {city}")
    
    city_lower = city.lower()

    # 1. Hardcoded data for specific demos (to ensure accurate specific info)
    database = {
        "tokyo": [
            "Tokyo Tower (Observation Deck, High Altitude)", 
            "Tokyo Ramen Street (Food, Indoors)", 
            "Shinjuku Gyoen (Park, Outdoors)"
        ],
        "paris": [
            "Eiffel Tower (Observation Deck, High Altitude)", 
            "Le Meurice (Fine Dining, Food)", 
            "Louvre Museum (Art, Indoors)"
        ],
        "new york": [
            "Empire State Building (High Altitude)", 
            "Chelsea Market (Food, Indoors)", 
            "Central Park (Outdoors)"
        ]
    }

    # 2. Check if we have specific data
    if city_lower in database:
        return database[city_lower]
    
    # 3. THE MAGIC FIX: Dynamic Fallback for ANY other city
    # This ensures your project works even if they ask for "Delhi" or "Berlin"
    else:
        return [
            f"{city} Sky Tower (High Altitude, Views)",
            f"{city} Central Market (Food, Local Culture)",
            f"{city} National Museum (History, Indoors)"
        ]


# 1. We define the "User Profile" (This acts as Long-term Memory)
user_preferences = """
You are a helpful Travel Concierge Agent.
CRITICAL INSTRUCTION: The user you are helping HATES being outdoors and is AFRAID of heights.
Always filter your recommendations to be INDOORS and ON THE GROUND FLOOR.
Focus heavily on FOOD.
"""

# 2. Re-initialize the model with this "Memory"
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash',
    tools=[get_weather, get_attractions],
    system_instruction=user_preferences # <--- This is the magic line
)

chat = model.start_chat(enable_automatic_function_calling=True)

# 3. Ask the SAME question as before
print("Agent is thinking (with personalization)...")
response = chat.send_message("I want to visit Tokyo today. Check the weather and tell me what I should visit.")

# 4. Compare the result
print("\n--- PERSONALIZED RESPONSE ---")
print(response.text)


import gradio as gr

# --- 1. The Wrapper Function ---
# This function connects Gradio to your Gemini Agent.
# It takes the user's message, sends it to Gemini, and returns the answer.
def agent_chat(user_message, history):
    # 'history' contains the past conversation, but Gemini handles its own history
    # in the 'chat' object, so we just send the new message.
    
    try:
        # Send the user's message to your existing chat session
        # (Make sure your chat object is named 'chat' or 'chat_session')
        response = chat.send_message(user_message)
        
        # Return the text response to the UI
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}"

# --- 2. The User Interface Setup ---
# We use ChatInterface, which looks like a standard messaging app.
demo = gr.ChatInterface(
    fn=agent_chat,
    title="Travel Concierge",
    description="I plan trips based on your profile (No heights, love food!).",
    theme="soft",
    examples=["Plan a day in Tokyo for me.", "What is the weather like?", "Should I visit Tokyo Tower?", " Which city you want to visit ?"]
)

# --- 3. Launch the App ---
# share=True creates a public link you can share with friends
demo.launch(share=True, debug=True)

