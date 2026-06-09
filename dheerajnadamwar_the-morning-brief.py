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
import time
from typing import List, Dict


from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models import Gemini
#from google.adk.tools import Tool
from google.adk.runners import Runner

# ==========================================
# SECTION 1: CONFIGURATION & SETUP
# ==========================================

# Initialize the Gemini 1.5 Pro model (Multimodal capabilities)
model_config = Gemini(
    model_name="gemini-1.5-pro",
    temperature=0.2
)

# Define the specific sender we want to filter for
TARGET_SENDER = "xyz@comp.com"

# ==========================================
# SECTION 2: CUSTOM TOOL DEFINITIONS
# ==========================================

class GmailReadTool(Tool):
    """
    Tool to fetch the latest unread email from a specific sender.
    """
    name = "gmail_fetcher"
    description = "Fetches the body of the most recent email from a specific sender address."

    def execute(self, sender_email: str) -> str:
        """
        Simulates calling the Gmail API to find emails from 'sender_email'.
        """
        print(f"\n[System Log] ğŸ“¨ GMAIL TOOL TRIGGERED: Searching for '{sender_email}'...")
        
        # --- MOCK API RESPONSE ---
        # In real life, you would use: service.users().messages().list(q=f"from:{sender_email}")
        time.sleep(1) # Simulate network latency
        
        email_content = """
        FROM: boss@company.com
        SUBJECT: Daily Industry Update - Urgent Review
        BODY:
        Good morning,
        
        Please review these two items for our morning strategy meeting.
        
        1. Competitor Video Analysis: https://youtube.com/watch?v=example_video
        2. Market Report Article: https://techcrunch.com/example_article
        
        Need a summary on my WhatsApp by 8:00 AM.
        """
        print(f"[System Log] âœ… Email found and retrieved.")
        return email_content


class WhatsAppSenderTool(Tool):
    """
    Tool to send the final summarized text via WhatsApp.
    """
    name = "whatsapp_sender"
    description = "Sends a text message to the user's registered phone number."

    def execute(self, message_body: str) -> str:
        """
        Simulates calling the Twilio API to send a WhatsApp message.
        """
        print(f"\n[System Log] ğŸ“± WHATSAPP TOOL TRIGGERED: Sending Payload...")
        
        # --- MOCK API RESPONSE ---
        # In real life: client.messages.create(body=message_body, from_='whatsapp:+123', to='whatsapp:+456')
        
        print("-" * 40)
        print(f"ğŸš€ FINAL OUTPUT SENT TO USER:\n{message_body}")
        print("-" * 40)
        
        return "Status: Message Delivered Successfully (200 OK)"

# ==========================================
# SECTION 3: AGENT DEFINITIONS
# ==========================================

# --- Agent 1: The Gatekeeper (Input) ---
# Checks Gmail and extracts the raw text.
input_agent = LlmAgent(
    name="Inbox_Manager",
    model=model_config,
    instruction=f"""
    You are an intelligent Inbox Manager.
    1. Your goal is to find the latest email specifically from '{TARGET_SENDER}'.
    2. Use the 'gmail_fetcher' tool to get the email content.
    3. Output the email body text exactly as received so other agents can read it.
    """,
    tools=[GmailReadTool()]
)

# --- Agent 2: Video Analyst (Parallel Worker A) ---
# Focuses ONLY on YouTube links.
video_analyst = LlmAgent(
    name="Video_Analyst",
    model=model_config,
    instruction="""
    You are a Video Content Analyst.
    1. Scan the input text for YouTube URLs.
    2. If a URL is found, simulate 'watching' it (using your multimodal capabilities).
    3. Output a structured summary:
       - **VIDEO TOPIC**: [One phrase]
       - **KEY INSIGHTS**: [3 bullet points]
    If no video link is present, return "No video content found."
    """
)

# --- Agent 3: Web Researcher (Parallel Worker B) ---
# Focuses ONLY on article/blog links.
web_researcher = LlmAgent(
    name="Web_Researcher",
    model=model_config,
    instruction="""
    You are a Web Researcher.
    1. Scan the input text for standard web URLs (non-video).
    2. Read the content of the article.
    3. Output a structured summary:
       - **ARTICLE TITLE**: [One phrase]
       - **SUMMARY**: [Brief paragraph]
    If no article link is present, return "No article content found."
    """
)

# --- Agent 4: The Broadcaster (Output) ---
# Aggregates the parallel outputs and uses the tool to send.
broadcaster_agent = LlmAgent(
    name="WhatsApp_Broadcaster",
    model=model_config,
    instruction="""
    You are the Chief Editor of the 'Morning Brief'.
    1. You will receive two inputs: one from the Video Analyst and one from the Web Researcher.
    2. Combine them into a single, professional message formatted for WhatsApp (use emojis).
    3. Use the 'whatsapp_sender' tool to deliver the message.
    4. Do not ask for confirmation; just send it.
    """,
    tools=[WhatsAppSenderTool()]
)

# ==========================================
# SECTION 4: ORCHESTRATION (WORKFLOW)
# ==========================================

# 1. Define the Parallel Block
# This runs Agent 2 and Agent 3 at the same time.
parallel_processing_block = ParallelAgent(
    name="Parallel_Analysis_Workflow",
    agents=[video_analyst, web_researcher],
    mode="wait_all" # Wait for both to finish before moving to the next step
)

# 2. Define the Main Pipeline
# Sequential flow: Input -> Parallel Processing -> Output
main_pipeline = SequentialAgent(
    name="Morning_Brief_Pipeline",
    agents=[
        input_agent,               # Step 1: Get Email
        parallel_processing_block, # Step 2: Analyze Video & Text (Async)
        broadcaster_agent          # Step 3: Send WhatsApp
    ]
)

# ==========================================
# SECTION 5: EXECUTION
# ==========================================

if __name__ == "__main__":
    print("ğŸ¤– SYSTEM STARTUP: Initializing Morning Brief Agents...")
    
    # Initialize the runner with our main pipeline
    runner = Runner(agent=main_pipeline)
    
    # Start the process
    # We pass a generic trigger because the Input Agent knows to look for the email itself.
    try:
        runner.run(input="Start the 8 AM daily briefing sequence.")
        print("\nâœ… SYSTEM SHUTDOWN: Workflow completed successfully.")
        
    except Exception as e:
        print(f"\nâ�Œ SYSTEM ERROR: {e}")

