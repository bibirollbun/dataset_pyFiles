# !pip install -q -U google-generativeai


import google.generativeai as genai
from google.colab import userdata # Agar Colab use kar rahe ho
import os
import time
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
    print(GOOGLE_API_KEY)
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

print("Setup Complete! API Key Configured.")


# 1. Tumhara Portfolio Database (Mock Data)
portfolio_data = {
    "excel automation": "I created a Python script that converts complex Excel sheets to JSON, helping web apps load data faster.",
    "3d portfolio": "An interactive 3D website built using Three.js and React to showcase my creative skills.",
    "ai agent": "A smart assistant built during Google's AI Intensive course that helps recruiters schedule interviews."
}

# 2. Tool Functions
def get_project_details(project_name: str):
    """Retrieves details about a specific project from Manish's portfolio."""
    print(f"\n[TOOL USAGE] Searching for project: {project_name}...")
    
    # Simple search logic
    for key, value in portfolio_data.items():
        if project_name.lower() in key:
            return f"Project: {key.title()}\nDetails: {value}"
    
    return "Project not found. Available projects: Excel Automation, 3D Portfolio, AI Agent."

def check_availability(day: str):
    """Checks if Manish is available for an interview on a specific day."""
    print(f"\n[TOOL USAGE] Checking calendar for: {day}...")
    
    weekend = ["saturday", "sunday"]
    if day.lower() in weekend:
        return f" Manish is not available on {day} (Weekend). Please suggest a weekday."
    else:
        return f"Yes, Manish is available on {day} between 10 AM and 6 PM IST."

def send_message_to_manish(recruiter_name: str, message: str):
    """Sends a contact message to Manish."""
    print(f"\n[TOOL USAGE] Sending email...")
    # Real app me yaha email logic hoga
    return f"Message sent successfully! Manish has received a notification from {recruiter_name}."

# 3. Tools List
my_tools = [get_project_details, check_availability, send_message_to_manish]
print("Tools Defined Successfully!")


# System Instructions: Agent ko batana ki wo kaun hai
instruction = """
You are 'PortfolioBot', an AI assistant for manish.
Your goal is to help recruiters understand manish's skills and schedule interviews.
Always be professional, polite, and concise.
Use the available tools to answer questions.
If a user asks about a project, look it up.
If
 a user wants to meet, check availability first.
"""

# Model Initialization
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite', # Fast model
    tools=my_tools,
    system_instruction=instruction
)

# Chat Session Start (Automatic Orchestration enabled)
chat = model.start_chat(enable_automatic_function_calling=True)
print("Agent is Ready! ðŸ¤–")


def talk_to_agent(message):
    try:
        print(f"User: {message}")
        response = chat.send_message(message)
        print(f"Agent: {response.text}")
        print("-" * 50)
        # Rate Limit bachane ke liye 4 second ka wait
        time.sleep(4) 
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Agar '429' error hai toh 1-2 minute wait karein.")

# --- TEST SCENARIOS ---

# 1. Project Enquiry
talk_to_agent("Tell me about your Excel Automation project.")

# 2. Availability Check
talk_to_agent("Are you free for an interview on Sunday?")

# 3. Scheduling (Reasoning)
talk_to_agent("Okay, what about Monday?")

# 4. Action (Sending Message)
talk_to_agent("Great, please tell Manish that Alice wants to meet on Monday.")

