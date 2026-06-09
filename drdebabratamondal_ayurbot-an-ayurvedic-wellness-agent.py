# Install dependencies and force update conflicting system libraries

import warnings
import os

warnings.filterwarnings('ignore')

!pip install -q -U google-generativeai langchain-google-genai langchain google-cloud-translate

print("âœ… Dependencies installed successfully.")



import os
from kaggle_secrets import UserSecretsClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
from langchain_core.messages import HumanMessage, SystemMessage

# Retrieve API Key from Kaggle Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

print("âœ… Environment Setup Complete.")



@tool
def lookup_ayurvedic_remedy(condition: str):
    """
    Consults the internal Ayurvedic database for herbs and remedies based on a specific condition or Dosha imbalance.
    Input should be a condition like 'joint pain', 'insomnia', 'low energy', or 'digestion'.
    """
    # A simple mock database for demonstration
    remedies = {
        "joint pain": "Mahanarayan Oil for local application and Shallaki (Boswellia) internally to reduce inflammation.",
        "insomnia": "Brahmi and Jatamansi tea before bed. Warm milk with nutmeg.",
        "digestion": "Triphala powder with warm water at night. Avoid cold drinks.",
        "low energy": "Ashwagandha Rasayana and Chyawanprash for vitality.",
        "skin issues": "Neem and Manjistha to purify the blood.",
        "muscle recovery": "Turmeric milk (Golden Milk) and Ashwagandha for muscle repair."
    }
    
    # Simple keyword matching logic
    condition = condition.lower()
    for key in remedies:
        if key in condition:
            return remedies[key]
    
    return "General advice: Drink warm water and practice light yoga. Consult a specialist for specific herbs."

print("âœ… Tool 'lookup_ayurvedic_remedy' registered.")



# 1. Initializing the Model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 2. Bind the tool to the model
llm_with_tools = llm.bind_tools([lookup_ayurvedic_remedy])

# 3. Define the System Context (Persona)
system_prompt = """
You are an expert Ayurvedic Physician specializing in Nadi Vidya and Marma Therapy.
Your goal is to help users by identifying their potential Dosha imbalance based on symptoms and prescribing natural remedies.

PROTOCOL:
1. Analyze the user's symptom.
2. ALWAYS use the 'lookup_ayurvedic_remedy' tool to find the correct herb.
3. Combine the tool output with your own knowledge to give a holistic recommendation (diet + lifestyle).
4. Keep answers concise and empathetic.
"""

print("âœ… Ayurvedic Agent Initialized.")



from langchain_core.messages import ToolMessage

def get_ayurvedic_consultation(user_query):
    print(f"\nğŸ‘¤ Patient: {user_query}")
    print("ğŸ¤– AyurBot is thinking...")
    
    # Create the message history
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    
    # First call: Model decides to use the tool
    ai_msg_1 = llm_with_tools.invoke(messages)
    messages.append(ai_msg_1)
    
    # Check if the model wanted to use a tool
    if ai_msg_1.tool_calls:
        # Loop through ALL tool calls (in case it tries to call multiple things)
        for tool_call in ai_msg_1.tool_calls:
            print(f"   (Agent is consulting the database for: {tool_call['args']})")
            
            # Execute the tool
            tool_result = lookup_ayurvedic_remedy.invoke(tool_call)
            
            # Create a proper ToolMessage with the matching ID
            tool_msg = ToolMessage(
                tool_call_id=tool_call['id'], # This links the result to the request
                content=str(tool_result),
                name=tool_call['name']
            )
            messages.append(tool_msg)
        
        # Final call: Model generates the answer using the tool results
        final_response = llm_with_tools.invoke(messages)
        print(f"ğŸ‘¨â€�âš•ï¸� Physician: {final_response.content}")
    else:
        # No tool needed, just print the response
        print(f"ğŸ‘¨â€�âš•ï¸� Physician: {ai_msg_1.content}")

# --- TEST CASES ---
print("--- TEST 1: RUNNER RECOVERY ---")
get_ayurvedic_consultation("I just finished a marathon and my muscles are very sore and tired.")

print("\n--- TEST 2: GENERAL ANXIETY ---")
get_ayurvedic_consultation("I have trouble sleeping at night and feel anxious.")



# --- INTERACTIVE CONSULTATION MODE ---
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
import warnings
import logging

logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

def on_submit(b):
    # Get the user query
    user_query = input_box.value
    if not user_query:
        return
        
    # Clear previous output and show processing status
    with output_area:
        clear_output()
        print(f"ğŸ‘¤ Patient: {user_query}")
        print("ğŸ¤– AyurBot is thinking... (Accessing Ayurvedic Knowledge Base)")
        
        # --- LOGIC COPIED FROM get_ayurvedic_consultation ---
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            # We wrap execution in a try-block to catch 429s silently if needed
            ai_msg_1 = llm_with_tools.invoke(messages)
            messages.append(ai_msg_1)
            
            if ai_msg_1.tool_calls:
                for tool_call in ai_msg_1.tool_calls:
                    # Show a cleaner status message
                    print(f"   ğŸŒ¿ Identifying Herbs for: {tool_call['args']}...")
                    
                    tool_result = lookup_ayurvedic_remedy.invoke(tool_call)
                    tool_msg = ToolMessage(
                        tool_call_id=tool_call['id'],
                        content=str(tool_result),
                        name=tool_call['name']
                    )
                    messages.append(tool_msg)
                
                final_response = llm_with_tools.invoke(messages)
                response_text = final_response.content
            else:
                response_text = ai_msg_1.content
            
            # Display nice Markdown response
            display(Markdown(f"### ğŸ‘¨â€�âš•ï¸� Dr. AyurBot's Recommendation:\n\n{response_text}"))
            
        except Exception as e:
            # Only show critical errors, hide the retry warnings
            if "429" in str(e):
                print("â�³ API Busy. Retrying automatically...")
            else:
                print(f"â�Œ Error: {str(e)}")

# --- WIDGET SETUP ---
title = widgets.HTML("<h2>ğŸŒ¿ Talk to Dr. AyurBot</h2><p>Describe your symptoms (e.g., 'I have joint pain' or 'How to recover from running?')</p>")
input_box = widgets.Text(placeholder="Type your symptom here...", layout=widgets.Layout(width='70%'))
submit_btn = widgets.Button(description="Get Remedy", button_style='success', icon='leaf')
output_area = widgets.Output()

submit_btn.on_click(on_submit)

display(title, widgets.HBox([input_box, submit_btn]), output_area)


