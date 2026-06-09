import google.generativeai as genai
import os
import json

# --- CONFIGURATION ---
# 1. PASTE YOUR NEW API KEY HERE
os.environ["GOOGLE_API_KEY"] = "AIzaSyALiySd7FyZA7MrPanU0tJtcura_p3A9Zo"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- TOOL DEFINITION ---
def check_user_inventory(category: str):
    """
    Reads the local 'user_inventory.json' file and retrieves items 
    for a specific category.
    """
    file_path = 'user_inventory.json'
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Normalize keys to lower case
        data = {k.lower(): v for k, v in data.items()}
        
        return data.get(category.lower(), f"Category '{category}' not found in inventory.")
        
    except FileNotFoundError:
        return "Error: Inventory database file not found."

# --- AGENT 1: THE WATCHER (Uses Search Logic) ---
def watcher_agent(location):
    print(f"\n--- [AGENT 1] The Watcher is scanning {location} ---")
    
    # Using Flash for speed
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    You are a Crisis Monitor. 
    Simulate a realistic HEATWAVE emergency alert for {location}.
    Provide: 
    1. Severity Level (High).
    2. Specific risks (Dehydration, Power Outages).
    Keep it concise.
    """
    
    response = model.generate_content(prompt)
    print(f"[!] ALERT DETECTED:\n{response.text}")
    return response.text

# --- AGENT 2: THE QUARTERMASTER (Uses Function Calling) ---
def quartermaster_agent(threat_context):
    print("\n--- [AGENT 2] The Quartermaster is checking supplies ---")
    
    tools = [check_user_inventory]
    # Using Flash for tool use
    model = genai.GenerativeModel('gemini-2.0-flash', tools=tools)
    
    chat = model.start_chat(enable_automatic_function_calling=True)
    
    prompt = f"""
    Context: {threat_context}
    
    TASK: Check the user's inventory for items CRITICAL to this specific threat.
    For a Heatwave, you MUST check for 'water' and 'power' (for fans/cooling).
    
    Output a list of what the user HAS and what is MISSING.
    """
    
    response = chat.send_message(prompt)
    print(f"[+] INVENTORY STATUS:\n{response.text}")
    return response.text

# --- AGENT 3: THE STRATEGIST (Reasoning & Output) ---
def strategist_agent(threat_context, inventory_status):
    print("\n--- [AGENT 3] The Strategist is building the plan ---")
    
    # Using Flash to prevent Quota Errors (429)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    You are a Crisis Commander.
    
    INPUT:
    1. THREAT: {threat_context}
    2. SUPPLY STATUS: {inventory_status}
    
    TASK:
    Generate a prioritized Action Plan.
    IF the user is missing Water -> Make "Get Water" the #1 Priority in bold.
    IF the user is missing Power -> Suggest alternative cooling methods.
    
    Keep the tone urgent but helpful.
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- MAIN ORCHESTRATION ---
# This block is what actually runs the agents. 
# Make sure it is NOT indented (it should touch the left edge).
if __name__ == "__main__":
    user_location = "Solapur, India" 
    
    # Run the Chain
    threat_data = watcher_agent(user_location)
    gap_data = quartermaster_agent(threat_data)
    final_plan = strategist_agent(threat_data, gap_data)
    
    print("\n" + "="*30)
    print("FINAL OUTPUT FOR USER")
    print("="*30)
    print(final_plan)


{
  "food": [
    "canned beans", 
    "granola bars", 
    "rice (5kg)"
  ],
  "medical": [
    "bandages", 
    "antiseptic cream", 
    "painkillers"
  ],
  "power": [
    "flashlight", 
    "AA batteries"
  ],
  "docs": [
    "passport_copy.pdf", 
    "insurance_policy.pdf"
  ]
}

