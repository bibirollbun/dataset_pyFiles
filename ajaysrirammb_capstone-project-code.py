# --- INSTALL & IMPORT ---
# Run this cell first to set everything up
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import os

# --- 1. SETUP API KEY ---
# IMPORTANT: You must add your 'GOOGLE_API_KEY' in the "Add-ons" -> "Secrets" menu in Kaggle.
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("âœ… API Key configured successfully!")
except Exception as e:
    print("âš ï¸� ERROR: Could not find API Key.")
    print("Please go to 'Add-ons' > 'Secrets' in the top menu.")
    print("Add a new secret called 'GOOGLE_API_KEY' and paste your Gemini API key there.")

# --- 2. DEFINE THE TOOL ---
# This is the function the agent will "call" to get data
def get_weather(city: str):
    """
    Returns the current weather for a given city.
    """
    # We are mocking the data to keep it simple for the Capstone
    weather_data = {
        "London": "Rainy, 15Â°C",
        "New York": "Sunny, 25Â°C",
        "Mumbai": "Humid, 30Â°C",
        "Tokyo": "Cloudy, 18Â°C",
        "Paris": "Windy, 12Â°C",
        "Chennai": "Hot and Humid, 32Â°C"
    }
    # Default response if city is not in our list
    return weather_data.get(city, "Sunny, 20Â°C")

# --- 3. INITIALIZE THE AGENT ---
# We connect the tool to the model here
tools_for_agent = [get_weather]

model = genai.GenerativeModel(
    model_name='gemini-2.0-flash', 
    tools=tools_for_agent,
    system_instruction="""
    You are a helpful Outfit Concierge Agent.
    Your goal is to help users decide what to wear.
    1. When a user mentions a city, ALWAYS use the 'get_weather' tool to check the forecast.
    2. Based on the weather return, suggest a practical and stylish outfit.
    3. Be friendly and concise.
    """
)

# Start the chat with automatic function calling enabled
chat = model.start_chat(enable_automatic_function_calling=True)

# --- 4. RUN THE DEMO ---
# This simulates the user asking a question
user_question = "I am traveling to London tomorrow. What should I wear?"

print(f"ğŸ‘¤ User: {user_question}")
print("... Agent is thinking and checking weather tools ...")

response = chat.send_message(user_question)

print(f"ğŸ¤– Agent: {response.text}")

