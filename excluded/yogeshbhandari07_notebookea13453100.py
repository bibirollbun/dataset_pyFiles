# Install necessary libraries for Gemini API and Search
!pip install -q -U google-generativeai duckduckgo-search

import google.generativeai as genai
from duckduckgo_search import DDGS
from kaggle_secrets import UserSecretsClient
import datetime
import os
from IPython.display import display, Markdown

# Setup API Key securely
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print(api_key)
    print("âœ… API Key Configured Successfully")
except Exception as e:
    print(f"âš ï¸� Error: {e}")
    print("Please ensure 'GOOGLE_API_KEY' is added in the Secrets add-on.")


# --- TOOL 1: REAL-TIME SEARCH ---
def search_web(query: str):
    """
    Searches the internet for real-time information using DuckDuckGo.
    Use this tool when you need current events, latest news, specific facts, or documentation.
    """
    try:
        print(f"ğŸ”� [Agent Action] Searching the web for: '{query}'...")
        # Getting top 3 results to keep context concise
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        
        # Formatting results for the LLM to read easily
        formatted_results = "\n".join(
            [f"- Title: {r['title']}\n  Snippet: {r['body']}\n  Source: {r['href']}" for r in results]
        )
        return formatted_results
    except Exception as e:
        return f"Search Error: {str(e)}"

# --- TOOL 2: TEMPORAL AWARENESS ---
def get_current_datetime():
    """
    Returns the current date and time.
    ALWAYS call this tool before answering questions about "latest" news, "today", or "upcoming" events.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Register tools in a list
tools_list = [search_web, get_current_datetime]
print("âœ… Tools Registered: search_web, get_current_datetime")


# Define the System Instruction (The Persona & Quality Control)
system_instruction = """
ROLE:
You are 'Nexus', an elite Technical Research Assistant.

OBJECTIVE:
Provide accurate, up-to-date, and context-aware answers by synthesizing internal knowledge with real-time web data.

OPERATIONAL RULES:
1. **Time Awareness:** If a user asks for "latest" or "today's" news, YOU MUST first check the date using 'get_current_datetime'.
2. **Citations:** When using 'search_web', you MUST include the source URL in your response (e.g., [Source Name](url)).
3. **Context:** You have memory. If a user says "Tell me more", refer to the previous topic seamlessly.
4. **Formatting:** Use Markdown. Use bolding for key entities and lists for clarity.

TONE:
Professional, Concise, and Insightful.
"""

# Initialize the Model
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite', # Change kiya hai
    tools=tools_list,
    system_instruction=system_instruction
)

# Start the Chat Session (This handles MEMORY automatically)
chat = model.start_chat(enable_automatic_function_calling=True)

print("âœ… Agent 'Nexus' is Online and Ready.")


def ask_nexus(prompt):
    """
    Helper function to display the chat in a nice format
    """
    display(Markdown(f"### ğŸ‘¤ User: {prompt}"))
    
    try:
        # Send message to the agent (History is preserved in 'chat')
        response = chat.send_message(prompt)
        display(Markdown(f"### ğŸ¤– Nexus:\n{response.text}"))
        print("-" * 80) # Separator
    except Exception as e:
        print(f"Error: {e}")

# --- TEST SCENARIO 1: Complex Query with multiple tools ---
# Agent should: Check Date -> Search Web -> Answer
ask_nexus("Check today's date and tell me the latest major updates or news about 'React 19' or 'Laravel 11'.")

# --- TEST SCENARIO 2: Context / Memory Check ---
# Agent should: Remember we are talking about React/Laravel without being told again.
ask_nexus("Are there any breaking changes mentioned in that news? summarizing them briefly.")




