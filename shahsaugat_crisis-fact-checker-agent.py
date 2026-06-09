# Install the new Google SDK
!pip install -q -U google-genai


import os
from google.genai import types
from google.genai import Client
from kaggle_secrets import UserSecretsClient

# --- 1. SETUP API KEY ---
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GEMINI_API_KEY")
except Exception as e:
    print("â�Œ Error: Could not find GEMINI_API_KEY in Add-ons -> Secrets.")
    print("Make sure you added the secret with the exact name: GEMINI_API_KEY")
    raise e

# --- 2. INITIALIZE CLIENT ---
# We use the stable 1.5 model to avoid Client Errors
client = Client(api_key=api_key)
MODEL_ID = "gemini-1.5-flash" 

# --- 3. DEFINE AGENT ---
class FactCheckerAgent:
    def __init__(self):
        self.tools = [types.Tool(google_search=types.GoogleSearch())]

    def check_claim(self, user_claim):
        print(f"ğŸ”� Investigating: '{user_claim}'...")
        
        system_instruction = """
        You are a Fact-Checker. 
        1. Search Google for the user's claim. 
        2. Return a verdict (TRUE/FALSE) and a summary with sources.
        """
        
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=user_claim,
                config=types.GenerateContentConfig(
                    tools=self.tools,
                    system_instruction=system_instruction,
                    temperature=0.3
                )
            )
            return response.text
        except Exception as e:
            return f"â�Œ API Error: {str(e)}"

# --- 4. TEST IT ---
my_agent = FactCheckerAgent()
print(f"âœ… Agent Ready (Model: {MODEL_ID})")

# Run a simple test
print(my_agent.check_claim("When was the last Olympics held in Paris?"))


# Test Case 1: A Real Event (modify date as needed to test)
claim1 = "There was a massive earthquake in Japan on January 1st, 2024."
result1 = my_agent.check_claim(claim1)
print(f"\n--- Result ---\n{result1}\n")

# Test Case 2: A Fake Rumor
claim2 = "The Eiffel Tower caught fire yesterday."
result2 = my_agent.check_claim(claim2)
print(f"\n--- Result ---\n{result2}\n")

