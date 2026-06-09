# Install Google ADK , LangGraph and the Google GenAI adapter
# !pip install google-adk

import os
import json
import base64
import datetime
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
from typing import Any, Dict
from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.protobuf import timestamp_pb2

# --- SETUP API KEYS and GMAIL Creds---
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar"
    ]
creds = None   

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    GOOGLE_USER_TOKEN = UserSecretsClient().get_secret("GOOGLE_USER_TOKEN")
    os.environ["GOOGLE_USER_TOKEN"] = GOOGLE_USER_TOKEN
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    print("âœ… API Key Setup Complete.")
    token_info = json.loads(os.environ["GOOGLE_USER_TOKEN"])
    creds = Credentials.from_authorized_user_info(token_info, SCOPES)
    print("âœ… Gmail and Calendar Token Setup Complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY'and 'GOOGLE_USER_TOKEN' to your Kaggle secrets. Details: {e}"
    )


# These functions interact directly with the Google APIs.

def clean_email_body(payload):
    """
    Recursively extracts plain text from the complex Gmail payload (Multipart/HTML).
    """
    body = ""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data')
                if data:
                    body += base64.urlsafe_b64decode(data).decode()
            elif 'parts' in part: # Nested parts
                body += clean_email_body(part)
    elif 'body' in payload:
        data = payload['body'].get('data')
        if data:
            body += base64.urlsafe_b64decode(data).decode()
            
    return body
    
# This function searches in the email based on user query. For Example - Did I receive any birthday invites this week?
def search_inbox_for_content(search_query: str):
    """
    Searches Gmail and returns the FULL content of the emails found.
    Args:
        search_query: The search term (e.g., 'subject:schedule', 'from:school')
    """
    if not creds: return "Error: Authentication failed."
    
    service = build('gmail', 'v1', credentials=creds)
    print(f"   [SYSTEM] Searching Inbox for: '{search_query}'...")

    # 1. Search for IDs
    results = service.users().messages().list(userId='me', q=search_query, maxResults=3).execute()
    messages = results.get('messages', [])
    
    if not messages:
        return "No emails found."

    # 2. Fetch Full Content
    full_email_data = []
    for msg in messages:
        # 'full' format gives us the payload (body)
        txt = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        
        headers = txt['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown")
        date_sent = next((h['value'] for h in headers if h['name'] == 'Date'), "Unknown")
        
        # Decode the body
        raw_body = clean_email_body(txt['payload'])
        
        # Truncate very long emails to save context (optional)
        clean_text = raw_body[:4000] 
        
        full_email_data.append(
            f"--- EMAIL START ---\nFrom: {sender}\nDate Sent: {date_sent}\nSubject: {subject}\nBody: {clean_text}\n--- EMAIL END ---"
        )

    return "\n\n".join(full_email_data)

# This function lists the Calendar Events already scheduled and lists events for next 45 days

def list_calendar_events(days_ahead: int = 45):
    """
    Lists calendar events for the upcoming days.
    Args:
        days_ahead: Number of days to look ahead (default is 45).
    """
    if not creds: return "Error: Auth failed."

    service = build('calendar', 'v3', credentials=creds)
    print(f"   [CALENDAR] Checking schedule for next {days_ahead} days...")

    # Get time range in ISO format
    now = datetime.datetime.utcnow()
    time_min = now.isoformat() + 'Z'
    time_max = (now + datetime.timedelta(days=days_ahead)).isoformat() + 'Z'

    events_result = service.events().list(
        calendarId='primary', timeMin=time_min, timeMax=time_max,
        maxResults=20, singleEvents=True, orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])

    if not events:
        return "No upcoming events found."

    calendar_summary = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        summary = event.get('summary', 'Busy')
        calendar_summary.append(f"- {start}: {summary}")

    return "\n".join(calendar_summary)

# This function Creates Event in the Calendar

def create_calendar_event(title: str, start_iso: str, end_iso: str, location: str = "TBD"):
    """
    Creates a new event on the Google Calendar.
    Args:
        title: The title of the event.
        start_iso: Start time in ISO format (e.g. '2025-12-12T14:00:00') and ensure its eastern time
        end_iso: End time in ISO format (e.g. '2025-12-12T15:00:00') and ensure its eastern time
        location: (Optional) Location of the event.
    """
    if not creds: return "Error: Auth failed."
    
    service = build('calendar', 'v3', credentials=creds)
    print(f"   [CALENDAR] Creating Event: '{title}' at {start_iso}...")

    event_body = {
        'summary': title,
        'location': location,
        'start': {'dateTime': start_iso, 'timeZone': 'UTC'},
        'end': {'dateTime': end_iso, 'timeZone': 'UTC'},
    }

    try:
        event = service.events().insert(calendarId='primary', body=event_body).execute()
        return f"Success! Event created: {event.get('htmlLink')}"
    except Exception as e:
        return f"Error creating event: {str(e)}"


# These are focused agents. They act as "Context Servers" - they don't plan, they just execute.
email_agent_model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite',
    tools=[search_inbox_for_content],
    system_instruction="""
    You are the Email Specialist.
    Your Job: Execute searches on Gmail and return FACTUAL extracts (Date, Time, Event Name, Sender).
    Do NOT offer to book things. Do NOT be conversational. Just report the data found.
    """
)

calendar_agent_model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite',
    tools=[list_calendar_events, create_calendar_event],
    system_instruction="""
    You are the Calendar Specialist.
    Your Job: Check availability and book events.
    Rules:
    1. Always use ISO 8601 format (YYYY-MM-DDTHH:MM:SS) for the tool arguments.
    2. Before creating any event, strictly check availability, report specific conflicts if they exist.
    3. It is safe to assume the closest date and this year and always look for future dates
    """
)



# This agent interfaces with the user and manages the specialists.

# 1. Define "Bridge Functions" (A2A Protocol)
# The Orchestrator calls these Python functions, which in turn spin up a Specialist Agent session.

def call_email_specialist(task_description: str):
    """
    Delegates a research task to the Email Specialist Agent.
    Args:
        task_description: E.g. "Find the email from school about the play."
    """
    print(f" ğŸ”„ [Orchestrator] Delegating to Email Agent: '{task_description}'")
    chat = email_agent_model.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(task_description)
    return f"[EMAIL REPORT]: {response.text}"

def call_calendar_specialist(task_description: str):
    """
    Delegates a scheduling task or calendar search to the Calendar Specialist Agent.
    Args:
        task_description: E.g. "Check if Dec 12 6 PM is free" or "Book event..." or "when is my next dentist appointment"
    """
    print(f" ğŸ”„ [Orchestrator] Delegating to Calendar Agent: '{task_description}'")
    chat = calendar_agent_model.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(task_description)
    return f"[CALENDAR REPORT]: {response.text}"

# 2. Configure the Orchestrator
orchestrator_tools = [call_email_specialist, call_calendar_specialist]

orchestrator_model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite',
    tools=orchestrator_tools,
    system_instruction="""
    You are ORBIT, the Autonomous Family Chief of Staff.
    
    You manage a team of specialists:
    1. Email Specialist (Finds information)
    2. Calendar Specialist (Finds Event Details, Checks availability & Books events)
    
    YOUR WORKFLOW:
    - Receive a request from the user.
    - Break it down into steps.
    - Delegate steps to your specialists using the available tools.
    - Synthesize their reports into a final, helpful answer for the user.
    
    EXAMPLE:
    User: "Find the Science Fair email and book it."
    You:
      1. call_email_specialist("Search for Science Fair email and extract date/time.")
      2. (Receive date: Dec 12, 2pm)
      3. call_calendar_specialist("Check availability on 2025-12-12.")
      4. (Receive: Free)
      5. call_calendar_specialist("Book Science Fair on 2025-12-12 at 14:00.")
      6. Reply to user: "Done. I found the email and booked it for you."
    """
)



# Start the persistent session
orbit_session = orchestrator_model.start_chat(enable_automatic_function_calling=True)


print("--- ğŸ�¬ ORBIT DEMO: Yoga Conflict ---")
user_input = "Can you book a dinner reservation for this Tuesday Dec 2, 2025 from 6 PM to 8 PM at The Venetian?"

response = orbit_session.send_message(user_input)

print("\n--- ğŸ�� FINAL OUTPUT ---")
print(f"\nğŸª� ORBIT: {response.text}\n")
print("-" * 50)


print("--- ğŸ�¬ ORBIT DEMO: Email Synthesis ---")
user_input = "Find email related to dinner with Northeastern friends and book it?"

response = orbit_session.send_message(user_input)

print("\n--- ğŸ�� FINAL OUTPUT ---")
print(f"\nğŸª� ORBIT: {response.text}\n")
print("-" * 50)


# def main():
#     print("\n" + "="*50)
#     print("ğŸª� ORBIT: SYSTEM ONLINE")
#     print("="*50)
#     print("Capable of: Reading Emails, Checking Calendar, Booking Events.")
#     print("Type 'exit' to quit.\n")

#     while True:
#         try:
#             user_input = input("ğŸ‘¤ YOU: ")
#             if user_input.lower() in ['exit', 'quit']:
#                 print("ğŸ›‘ Orbit shutting down.")
#                 break
            
#             print("\nğŸ¤– ORBIT IS THINKING...")
#             response = orbit_session.send_message(user_input)
            
#             print(f"\nğŸª� ORBIT: {response.text}\n")
#             print("-" * 50)
            
#         except Exception as e:
#             print(f"â�Œ SYSTEM ERROR: {e}")

# if __name__ == "__main__":
#         main()

