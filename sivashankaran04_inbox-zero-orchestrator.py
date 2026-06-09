# Install the Google ADK and Generative AI libraries
!pip install -q google-adk google-generativeai pydantic
print("âœ… Libraries installed successfully.")


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


pip install google-adk


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


from pydantic import BaseModel, Field
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner

# --- 2.1: Tool Definitions (The Actions) ---

def draft_email(recipient: str, subject: str, body: str) -> dict:
    """
    Drafts an email with the specified recipient, subject, and body.
    The email is NOT sent, only drafted and saved for review.
    """
    print(f"--- TOOL CALLED: DRAFT EMAIL ---")
    return {
        "status": "success",
        "message": f"Drafted email for {recipient} with subject: '{subject}'."
    }

def summarize_content(content: str) -> dict:
    """
    Generates a concise summary of a long email body.
    """
    # In a real agent, the LLM would usually perform the summarization 
    # as a final step, but defining this as a tool helps guide the agent's logic.
    print(f"--- TOOL CALLED: SUMMARIZE ---")
    return {
        "status": "success",
        "summary": "Key points: Meeting confirmed for next week. Action item: Need to send the report by EOD.",
        "note": "This is a mocked summary. In production, the LLM would generate the real summary."
    }

def archive_email(email_id: str) -> dict:
    """
    Moves a processed email from the inbox to the archive folder.
    """
    print(f"--- TOOL CALLED: ARCHIVE EMAIL ---")
    return {
        "status": "success",
        "message": f"Email {email_id} has been archived."
    }

def create_reminder(task_description: str, due_date: str) -> dict:
    """
    Creates a follow-up reminder in the user's task list for a specific due date.
    """
    print(f"--- TOOL CALLED: CREATE REMINDER ---")
    return {
        "status": "success",
        "message": f"Reminder created: '{task_description}' due on {due_date}."
    }

# Combine tools into a list
INBOX_TOOLS = [draft_email, summarize_content, archive_email, create_reminder]
print("âœ… Custom tools defined.")


ORCHESTRATOR_INSTRUCTION = """
You are the 'Inbox Zero Orchestrator', a single-agent system specializing in email triage.
Your only task is to analyze an unread email (provided as a prompt) and determine the
single, most appropriate action to achieve 'Inbox Zero' status for that email.

Follow these rules strictly:
1. Classification: You must classify the email as one of the following:
   - 'URGENT': Requires an immediate reply (Use draft_email).
   - 'FYI': Information only, needs summarization (Use summarize_content and archive_email).
   - 'FOLLOW_UP': Requires a reminder for a later action (Use create_reminder, and optional draft_email).
   - 'JUNK': Spam or unwanted mail (Use archive_email).
2. Tool Use: You MUST select and use the appropriate tool(s) based on your classification.
3. Output: After classifying the email and calling the tool(s), you MUST generate a FINAL, brief conversational message to the user. This message must START by clearly stating the assigned classification in a single sentence (e.g., "I've classified this email as URGENT.") and then summarize the actions taken.
"""

# --- 3.2: Create the Agent Instance (No change needed here, just re-run) ---
inbox_agent = Agent(
    name="Inbox_Zero_Orchestrator",
    model=Gemini(model="gemini-2.5-flash-lite"),
    description="An autonomous agent for email triage and action planning.",
    instruction=ORCHESTRATOR_INSTRUCTION, # This will use the new instruction
    tools=INBOX_TOOLS,
)

print("âœ… Inbox agent created...")


# --- 4.1: Create the Runner ---
# The Runner orchestrates the conversation and tool calls.
runner = InMemoryRunner(agent=inbox_agent)
print("âœ… Runner created.")

# ----------------------------------------------------------------------
## ğŸ§ª Test Case 1: Urgent Request
# ----------------------------------------------------------------------
print("\n--- TEST CASE 1: URGENT REQUEST ---")
urgent_email = (
    "Subject: URGENT: Q4 Budget Review required by 3 PM\n"
    "Body: Hi team, please review the attached Q4 budget slide deck IMMEDIATELY "
    "and provide your feedback by 3 PM TODAY. We cannot proceed without it. - CFO"
)

# 1. Run the debug session, which returns a list of Event objects
response_urgent_events = await runner.run_debug(urgent_email) # Renamed variable for clarity

# 2. Extract result for Test Case 1 (Find the FIRST final response)
final_message_text_urgent = None
for event in response_urgent_events:
    # Check for the message safely
    if hasattr(event, 'message') and event.message:
        # Check for final response flag AND text existence
        if event.is_final_response() and hasattr(event.message, 'text'):
            final_message_text_urgent = event.message.text
            # Use 'continue' to keep looking for the absolute last message,
            # OR use 'break' to stop immediately and take the first one found.
            # To get the FIRST detailed response, we will BREAK here:
            break

if final_message_text_urgent:
    print(f"\nAGENT RESPONSE (Urgent):\n{final_message_text_urgent}")


# ----------------------------------------------------------------------
## ğŸ§ª Test Case 2: Follow-up Required
# ----------------------------------------------------------------------
print("\n--- TEST CASE 2: FOLLOW-UP REQUIRED ---")
followup_email = (
    "Subject: Next steps for the Marketing Campaign\n"
    "Body: Let's finalize the copy for the new campaign. Can you please "
    "send me the finalized draft by the end of next week, Friday the 22nd?"
)

response_followup_events = await runner.run_debug(followup_email) # Renamed variable for clarity

# 3. Extract result for Test Case 2 (Find the FIRST final response)
final_message_text_followup = None
for event in response_followup_events:
    if hasattr(event, 'message') and event.message:
        if event.is_final_response() and hasattr(event.message, 'text'):
            final_message_text_followup = event.message.text
            # BREAK here to capture the FIRST detailed response
            break

if final_message_text_followup:
    print(f"\nAGENT RESPONSE (Follow-up):\n{final_message_text_followup}")

