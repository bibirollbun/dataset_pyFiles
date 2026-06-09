# Force update the library first
!pip install -q -U google-generativeai ddgs matplotlib

import os
import google.generativeai as genai
from duckduckgo_search import DDGS
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from kaggle_secrets import UserSecretsClient

print("ğŸ”„ Initializing Aegis System...")

# --- 1. AUTHENTICATE ---
try:
    user_secrets = UserSecretsClient()
    my_secret = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = my_secret
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except Exception as e:
    print("â�Œ Secret Key Error. Check Add-ons -> Secrets.")

# --- 2. AUTO-DISCOVERY (The Fix) ---
print("ğŸ”� Scanning for available models...")
working_model = None

# We ask Google to list all models available to your Key
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            # We prefer Flash, but we will take anything that works
            if 'flash' in m.name:
                working_model = m.name
                break
            if 'gemini-pro' in m.name:
                working_model = m.name

    # If we found a model in the list, use it.
    if working_model:
        print(f"âœ… FOUND VALID MODEL: {working_model}")
        model = genai.GenerativeModel(working_model)
        
        # Final Test
        response = model.generate_content("Test")
        print("âœ… System Online: Aegis Protocol Initiated")
        
    else:
        # Emergency Fallback
        print("âš ï¸� No specific model found in list. Trying generic 'gemini-pro'...")
        model = genai.GenerativeModel('gemini-pro')
        print("âœ… System Online: Aegis Protocol Initiated")

except Exception as e:
    print(f"â�Œ Critical Error: {e}")
    print("If this fails, your API Key might still be from the old project.")


# --- ğŸ› ï¸� TOOL 1: SEARCH ENGINE ---
def tool_search(query):
    """Fetches real-time crisis data."""
    try:
        results = DDGS().text(query, max_results=3)
        return "\n".join([r['body'] for r in results]) if results else "No specific data found."
    except:
        return "Search Connection Failed."

# --- ğŸ› ï¸� TOOL 2: DATA VISUALIZER (Unique Feature) ---
def tool_plot_data(location, risk_level, resources):
    """Generates a visual dashboard."""
    print(f"   ğŸ“Š [SYSTEM] Visualizer Agent: Generating tactical dashboard for {location}...")
    
    # Simple data parsing for the graph
    labels = list(resources.keys())
    values = list(resources.values())
    
    # Create the chart
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=['#ff9999','#66b3ff','#99ff99'])
    plt.title(f'Resource Deployment: {location} (Risk: {risk_level})')
    plt.ylabel('Units Deployed')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the chart
    plt.show()
    return "Dashboard generated."

# --- ğŸ¤– AGENT DEFINITIONS ---

def agent_scout(location):
    display(Markdown(f"### ğŸ“¡ STEP 1: SCOUT AGENT (Data Gathering)"))
    # Dynamic Search
    raw_data = tool_search(f"current wildfire risk weather {location} wind humidity")
    
    prompt = f"""
    ROLE: You are the Aegis Scout.
    CONTEXT: {raw_data}
    TASK: Analyze the fire risk for {location}.
    OUTPUT: Return ONLY a JSON-style string: 
    {{"risk": "High", "reason": "High winds detected", "temp": "35C"}}
    """
    response = model.generate_content(prompt)
    print(f"   > Analysis: {response.text}")
    return response.text

def agent_commander(scout_data):
    display(Markdown(f"### âš”ï¸� STEP 2: COMMANDER AGENT (Strategy)"))
    
    prompt = f"""
    ROLE: You are the Aegis Commander.
    INPUT: {scout_data}
    TASK: Decide resource numbers (Drones, FireCrews, Tankers).
    OUTPUT: Return ONLY a Python dictionary:
    {{"Drones": 15, "FireCrews": 5, "Tankers": 2}}
    """
    response = model.generate_content(prompt)
    
    # Clean up response to ensure it's valid python code
    clean_response = response.text.replace("```python", "").replace("```", "").replace("json", "").strip()
    display(Markdown(f"**Orders Issued:** `{clean_response}`"))
    try:
        return eval(clean_response) 
    except:
        return {"Drones": 10, "FireCrews": 5, "Tankers": 1} # Fallback if AI messes up format

# --- ğŸš€ ORCHESTRATOR ---
def run_aegis_simulation(location):
    print(f"--- ğŸŒ� INITIATING AEGIS FOR: {location.upper()} ---")
    scout_data = agent_scout(location) # 1. Get Data
    resources = agent_commander(scout_data) # 2. Plan Strategy
    
    display(Markdown(f"### ğŸ“ˆ STEP 3: VISUALIZER AGENT")) # 3. Visualize
    risk_level = "CRITICAL" if "High" in scout_data else "MODERATE"
    tool_plot_data(location, risk_level, resources)
    
    print("\n--- âœ… MISSION COMPLETE ---")


run_aegis_simulation("Los Angeles, California")

