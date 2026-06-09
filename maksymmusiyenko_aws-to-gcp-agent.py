# Install the Google GenAI SDK
!pip install -q -U google-genai

import os
from google import genai
from kaggle_secrets import UserSecretsClient

print("--- 1. Setup & Auth ---")

# Retrieve the API key securely from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    # Key name GOOGLE_API_KEY is used as per environment settings
    api_key = user_secrets.get_secret("GOOGLE_API_KEY") 
    
    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)
    print("✅ Success! API Client initialized and ready.")
except Exception as e:
    print(f"❌ Error during setup: {e}")
    print("Please ensure the API key named GOOGLE_API_KEY is correctly set in Add-ons -> Secrets.")


# --- 2. Define Custom Tool (Fulfills the 'Tools' requirement) ---
def get_cloud_comparison(aws_service: str) -> str:
    """
    Takes an AWS service name and returns its Google Cloud equivalent.
    """
    database = {
        "ec2": "Google Compute Engine (GCE)",
        "s3": "Google Cloud Storage (GCS)",
        "lambda": "Google Cloud Functions",
        "rds": "Google Cloud SQL",
        "vpc": "Google Virtual Private Cloud (VPC)",
        "iam": "Cloud IAM",
        "route53": "Cloud DNS",
        "sns": "Pub/Sub"
    }
    
    key = aws_service.lower().strip()
    result = database.get(key)
    
    if result:
        return f"The GCP equivalent for AWS {aws_service} is: {result}."
    else:
        return "Check Google Cloud Documentation."

# --- 3. Agent Configuration (Fulfills the 'LLM' requirement) ---
system_instruction = """
You are the "AWS to GCP Agent", an expert Cloud Architect and mentor.
Your goal is to assist AWS professionals in migrating their knowledge and concepts to Google Cloud Platform (GCP).

Rules:
1. Respond concisely and professionally in English.
2. When the user asks about an AWS service, you MUST use the provided tool 'get_cloud_comparison' to find the official GCP analogue.
"""

# Creating the chat session (Fulfills the 'Sessions & Memory' requirement)
chat = client.chats.create(
    model='gemini-2.0-flash', 
    config={
        'system_instruction': system_instruction,
        'tools': [get_cloud_comparison],
    }
)

print("✅ Agent initialized (Using Gemini 2.0 Flash).")


print("--- TEST 1: Tool Utilization (AWS -> GCP) ---")
try:
    response1 = chat.send_message("I use EC2 in AWS. What is the equivalent in Google Cloud?")
    print(f"User: I use EC2...\nAgent: {response1.text}")
except Exception as e:
    print(f"❌ Error in Test 1 (API call failed): {e}")

print("\n--- TEST 2: Memory/Session Check ---")
try:
    response2 = chat.send_message("What are the main benefits of scaling this service in GCP?")
    print(f"User: What are benefits...\nAgent: {response2.text}")
except Exception as e:
    print(f"❌ Error in Test 2 (API call failed): {e}")

