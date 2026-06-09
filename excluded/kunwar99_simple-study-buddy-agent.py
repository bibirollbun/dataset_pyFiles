# --- SIMPLE STUDY BUDDY AGENT (CERTIFICATE VERSION) ---
import datetime
import sys
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient

# 1. SETUP
# *** IMPORTANT: Change this to match exactly what you typed in Add-ons > Secrets ***
secret_label = "GOOGLE_API_KEY" 

try:
    secret_value = UserSecretsClient().get_secret(secret_label)
    client = genai.Client(api_key=secret_value)
    print(f"[\u2705 SETUP] Success! API Key loaded.")
except Exception as e:
    print(f"\n[\u274C ERROR] Could not find a secret named '{secret_label}'.")
    print("1. Click 'Add-ons' in the top menu -> 'Secrets'.")
    print(f"2. Make sure the Label is exactly: {secret_label}")
    print("3. Make sure the API Key is pasted in the Value box.")
    print("4. Toggle the checkbox to 'Attached'.")
    # Stop the code here so it doesn't crash later
    sys.exit("Stopping execution. Please fix the API Key first.")

# 2. DEFINE TOOLS
def get_current_time():
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def set_study_timer(minutes: int, topic: str):
    """Sets a study timer."""
    return f"Timer set for {minutes} minutes. Focus on: {topic}."

agent_tools = [get_current_time, set_study_timer]

# 3. INITIALIZE AGENT
# We only run this if client exists
agent = client.chats.create(
    model='gemini-2.0-flash',
    config=types.GenerateContentConfig(
        tools=agent_tools,
        temperature=0.7
    )
)

# 4. RUN AGENT
def run_study_buddy(user_input):
    print(f"\n--- NEW INTERACTION ---")
    print(f"[\u1F442 INPUT] User said: '{user_input}'")
    response = agent.send_message(user_input)
    print(f"[\u1F916 AGENT] Response: {response.text}")
    print("-----------------------")

# Test Run
run_study_buddy("Hi, I'm Alex. What time is it?")
run_study_buddy("Set a timer for 25 minutes for History.")

