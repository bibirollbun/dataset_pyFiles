!pip install -q -U google-genai


# --- IMPORTS & SETUP ---
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown

# 1. Get the API Key safely from your Secrets (which you already set up!)
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
except Exception as e:
    print(f"Error: Could not find API Key. Make sure the label in Secrets is exactly 'GOOGLE_API_KEY'. Details: {e}")

# 2. Connect to Gemini
client = genai.Client(api_key=GOOGLE_API_KEY)

# --- THE AGENT CONFIGURATION ---

# A. Create the Search Tool
# This allows the agent to look up real-time info (weather, prices)
google_search_tool = types.Tool(
    google_search=types.GoogleSearch()
)

# B. Define the Personality (System Instruction)
system_instruction = """
You are 'The Smart Trip Agent', a helpful travel concierge. 
Your goal is to plan 3-day itineraries.

CRITICAL RULES:
1. ALWAYS use the 'google_search' tool to find real flight prices, hotel costs, and weather. Never guess.
2. Structure your response in Markdown. Use a table for the daily itinerary.
3. If the user doesn't state a budget, ask for one politely.
4. Be concise but friendly.
"""

# C. Create the Chat Session (Memory)
# This 'chat' object automatically remembers previous messages!
chat = client.chats.create(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        tools=[google_search_tool],
        system_instruction=system_instruction,
        temperature=0.7 # A bit of creativity
    )
)

# --- HELPER FUNCTION ---
def chat_with_agent(user_input):
    """
    This function sends your text to the agent and displays the reply nicely.
    """
    # Show what you said
    display(Markdown(f"**User:** {user_input}"))
    
    # Get response from Agent
    try:
        response = chat.send_message(user_input)
        # Show what the Agent said
        display(Markdown(f"**Agent:** {response.text}"))
    except Exception as e:
        print(f"An error occurred: {e}")

print("✅ Agent is ready! Go to the next cell to talk to it.")


import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import warnings

# Hide messy warnings for the demo
warnings.filterwarnings('ignore')

# --- CHAT HISTORY STORAGE ---
# Global list to store the conversation
if 'chat_history' not in globals():
    chat_history = [] 

# --- WIDGETS ---
output_box = widgets.Output() 
text_input = widgets.Text(
    placeholder="Type your request (e.g., 'Plan a trip to Goa')...", 
    layout=widgets.Layout(width='80%')
)
send_btn = widgets.Button(
    description="Send", 
    button_style='primary',
    layout=widgets.Layout(width='15%')
)

def render_chat():
    with output_box:
        clear_output(wait=True)
        for chat_line in chat_history:
            display(Markdown(chat_line))

def on_send_click(b):
    user_msg = text_input.value
    if not user_msg: return
    
    # 1. Add User to History
    chat_history.append(f"**You:** {user_msg}")
    text_input.value = ''
    
    # 2. Show "Thinking..."
    render_chat()
    with output_box:
        print("Concierge is thinking...") 
    
    try:
        # 3. Get AI Response
        response = chat.send_message(user_msg)
        
        # 4. Add AI to History
        chat_history.append(f"**Concierge:** {response.text}")
        chat_history.append("---") 
        
        # 5. Final Render
        render_chat()
            
    except Exception as e:
        with output_box:
            print(f"Error: {e}")

# Link button (We removed the 'on_submit' line to fix your error)
send_btn.on_click(on_send_click)

# Display
print("Chat Interface Ready! (Click 'Send' to chat)")
display(widgets.HBox([text_input, send_btn]))
display(output_box)




