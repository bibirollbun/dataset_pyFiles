!pip install -U google-generativeai


# --- STEP 1: INSTALL & SETUP ---
!pip install -q -U google-generativeai

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import os

# secure retrieval of the key
user_secrets = UserSecretsClient()
try:
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("âœ… API Connected")
except:
    print("â�Œ Error: API Key not found in Secrets")

# --- STEP 2: DEFINE TOOLS ---
# The agent uses these functions to interact with the "world"

score_board = {"current_score": 0}

def update_score(points: int, reason: str):
    """Updates the user's score based on their answer quality."""
    score_board["current_score"] += points
    return f"Score updated. Current Score: {score_board['current_score']}. Reason: {reason}"

def get_current_score():
    """Retrieves the current score."""
    return f"Current Score: {score_board['current_score']}"

# List of tools to pass to the model
interview_tools = [update_score, get_current_score]

# --- STEP 3: INITIALIZE AGENT ---
# We use Gemini 1.5 Flash (fast & cheap) with automatic tool use enabled

model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite',
    tools=interview_tools,
    system_instruction="You are a strict Python Interviewer. Ask 1 technical question. "
                       "If the user answers correctly, use the tool to add 10 points. "
                       "If wrong, deduct 5. Always state the new score."
)

# Start the chat with automatic function calling enabled
chat = model.start_chat(enable_automatic_function_calling=True)

# --- STEP 4: EXECUTION LOOP ---
def interview_session(user_input):
    print(f"\nğŸ‘¤ User: {user_input}")
    response = chat.send_message(user_input)
    print(f"ğŸ¤– Agent: {response.text}")

# --- TEST RUN ---
# Scenario: User starts, gets a question, answers it (mocking the flow)
interview_session("I am ready for my Python interview.")
interview_session("The difference between a list and a tuple is that lists are mutable and tuples are immutable.")

