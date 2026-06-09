import os
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient


try:
    user_secrets = UserSecretsClient()
    # Ensure you have set 'GOOGLE_API_KEY' in Kaggle Secrets
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
except Exception:
    # Fallback for local testing (manual entry)
    GOOGLE_API_KEY = "YOUR_API_KEY_HERE" 

# Initialize the Client
client = genai.Client(api_key=GOOGLE_API_KEY)

print("âœ… Venture_Path System Initialized.")


class Agent:
    """
    A modular Agent class that can use Google Search to find real-world info.
    """
    def __init__(self, name, role, instructions, model="gemini-2.0-flash", use_search=True):
        self.name = name
        self.role = role
        self.model = model
        self.instructions = instructions
        self.tools = []
        
        # Enable Native Google Search if required
        if use_search:
            self.tools.append(types.Tool(google_search=types.GoogleSearch()))

    def run(self, task, context=""):
        print(f"\nğŸ”„ {self.name} ({self.role}) is working...")
        
        # Construct the System Prompt
        system_prompt = f"""
        YOU ARE: {self.name}, a {self.role}.
        
        YOUR CORE INSTRUCTIONS:
        {self.instructions}
        
        CONTEXT FROM TEAM:
        {context}
        """
        
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=f"{system_prompt}\n\nCURRENT TASK: {task}",
                config=types.GenerateContentConfig(
                    tools=self.tools,
                    temperature=0.3, # Low temp for factual accuracy
                )
            )
            return response.text
        except Exception as e:
            return f"Error in {self.name}: {e}"

print("âœ… Agent Framework Loaded.")


# 1. TransportArchitect (Formerly RoutePlanner)
# Renamed to be broad enough for Trains, Buses, or Flights.
transport_agent = Agent(
    name="TransportArchitect",
    role="Logistics Specialist",
    use_search=True,
    instructions="""
    You are an expert multi-modal travel planner.
    
    YOUR GOAL: 
    Find the best connection between an Origin and a Destination.
    
    STRATEGY:
    1. FIRST, search for a DIRECT TRAIN. If found, list it (Train Name, Number, Approx Time).
    2. IF NO DIRECT TRAIN, search for a 'Broken Route' (e.g., Train to a Hub City -> Bus/Train to Destination).
    3. VALIDATE the connection times (ensure the user can actually make the transfer).
    
    OUTPUT FORMAT:
    - Primary Option: [Details]
    - Total Duration: [Hours]
    - Transfer Point: [City Name or 'None']
    """
)

# 2. BudgetGuardian (Finance)
# Estimates costs based on the transport mode found.
budget_agent = Agent(
    name="BudgetGuardian",
    role="Cost Analyst",
    use_search=False, # Uses logic/estimates based on the Architect's findings
    instructions="""
    You are a travel accountant. 
    Read the itinerary provided by the TransportArchitect.
    
    ESTIMATION RULES (Mental Math):
    - Train (Sleeper/3AC): Approx â‚¹3 per km or â‚¹200 per hour of travel.
    - Bus/Taxi: Approx â‚¹1000 per inter-city leg.
    - Food/Misc: Flat â‚¹800 per travel day.
    
    TASK:
    Calculate the Total Estimated Budget for 1 Person. 
    Break it down by Ticket Cost + Food/Misc.
    """
)

# 3. LocalGuide (Concierge)
# Finds the cultural gems.
guide_agent = Agent(
    name="LocalGuide",
    role="Destination Concierge",
    use_search=True,
    instructions="""
    You are a local expert for the Final Destination.
    
    TASK:
    1. Identify 3 'Non-Tourist' hidden gems or top rated spots to visit.
    2. Identify 2 FAMOUS LOCAL MEALS that the city is known for (specific dish names).
    
    TONE: 
    Exciting, inviting, and specific.
    """
)

print("âœ… Venture_Path Assembled: TransportArchitect, BudgetGuardian, LocalGuide.")


def run_venturepath(origin, destination, budget):
    print(f"ğŸŒ� VENTURE_PATH: {origin} â�¡ï¸�  {destination}")
    print("="*70)
    
    # --- PHASE 1: LOGISTICS ---
    route_plan = transport_agent.run(
        task=f"Find a route from {origin} to {destination}. Prioritize Trains, but use Bus/Shared Cab if rail is missing."
    )
    print(f"\nğŸ“� [Logistics Plan]:\n{route_plan}")
    
    # --- PHASE 2: PARALLEL PROCESSING (Cost & Culture) ---
    # In a production app, these would run at the same time.
    
    cost_estimate = budget_agent.run(
        task=f"Estimate the cost for this trip under {budget}.",
        context=route_plan
    )
    print(f"\nğŸ’° [Budget Estimate]:\n{cost_estimate}")
    
    local_secrets = guide_agent.run(
        task=f"What should I do and eat in {destination}?",
        context=route_plan
    )
    print(f"\nâœ¨ [Local Experience]:\n{local_secrets}")
    
    print("\n" + "="*70)
    print("âœ… ITINERARY GENERATION COMPLETE")


# Testing a route that typically requires a Train + Road connection
# Example: Mumbai to Shillong (Train to Guwahati, Road to Shillong)
run_venturepath("Mumbai (CSMT)", "Shillong, Meghalaya", 10000)


import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import time

# ==========================================
# 1. THE HTML TEMPLATE (Dynamic F-String)
# ==========================================
def generate_html(origin, destination, logistics_txt, budget_txt, guide_txt):
    # We clean up the raw AI text to make it look good in HTML
    # Converting newlines to <br> for basic formatting
    logistics_html = logistics_txt.replace("\n", "<br>")
    budget_html = budget_txt.replace("\n", "<br>")
    guide_html = guide_txt.replace("\n", "<br>")

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
            :root {{ --primary: #2A9D8F; --secondary: #E9C46A; --accent: #E76F51; --dark: #264653; --light: #F4F1DE; --glass: rgba(255, 255, 255, 0.95); }}
            body {{ font-family: 'Outfit', sans-serif; background: transparent; color: #333; margin: 0; }}
            .container {{ max-width: 900px; margin: 20px auto; background: var(--glass); border-radius: 24px; padding: 40px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2); animation: slideUp 0.8s ease-out; }}
            .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px dashed #ccc; padding-bottom: 20px; }}
            .brand {{ font-weight: 800; color: var(--primary); letter-spacing: 2px; text-transform: uppercase; font-size: 0.9rem; }}
            h1 {{ font-size: 2.2rem; margin: 10px 0; color: var(--dark); }}
            .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
            @media(max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
            .card {{ background: #fff; border-radius: 16px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; }}
            h2 {{ color: var(--dark); border-left: 5px solid var(--accent); padding-left: 15px; margin-top: 0; }}
            .ai-text {{ font-size: 0.95rem; line-height: 1.6; color: #444; white-space: pre-wrap; }}
            .budget-box {{ background: var(--dark); color: white; border-radius: 16px; padding: 25px; }}
            .budget-box h2 {{ color: white; border-color: var(--secondary); }}
            @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
            <div class="brand">Venture_Path AI</div>
            <h1>{origin} <span style="color:#ccc">â�”</span> {destination}</h1>
            <div class="tagline">AI Generated Travel Plan</div>
        </div>

        <div class="grid">
            <div class="main-content">
                <div class="card">
                    <h2>ğŸ“� Logistics Plan</h2>
                    <div class="ai-text">{logistics_txt}</div>
                </div>
                <div class="card">
                    <h2>âœ¨ Local Experience</h2>
                    <div class="ai-text">{guide_txt}</div>
                </div>
            </div>

            <div class="sidebar">
                <div class="budget-box">
                    <h2>ğŸ’° Est. Budget</h2>
                    <div class="ai-text" style="color: rgba(255,255,255,0.9)">{budget_txt}</div>
                </div>
                <div class="card" style="margin-top:20px">
                    <h2>âš ï¸� Notes</h2>
                    <small>Prices are estimates. Check live availability.</small>
                </div>
            </div>
        </div>
    </div>
    </body>
    </html>
    """
    return html_template

# ==========================================
# 2. THE INPUT UI (Ipywidgets)
# ==========================================

# Style definitions
style = {'description_width': 'initial'}
layout = widgets.Layout(width='98%', margin='5px')
btn_layout = widgets.Layout(width='100%', margin='20px 0px')

# Widget Elements
w_origin = widgets.Text(description='From:', placeholder='e.g., Mumbai', style=style, layout=layout)
w_dest = widgets.Text(description='To:', placeholder='e.g., Shillong', style=style, layout=layout)
w_btn = widgets.Button(description='ğŸš€ Plan My Adventure', button_style='success', layout=btn_layout)
w_output = widgets.Output()

# ==========================================
# 3. THE LOGIC (Button Click Event)
# ==========================================

def on_button_click(b):
    origin = w_origin.value
    dest = w_dest.value
    
    if not origin or not dest:
        with w_output:
            clear_output()
            print("â�Œ Please enter both Origin and Destination.")
        return

    with w_output:
        clear_output()
        # Display a simple loader
        display(HTML("""<div style="text-align:center; padding: 20px;">
                        <img src="https://i.gifer.com/ZZ5H.gif" width="50"><br>
                        <b style="color:#2A9D8F">Venture_Path is planning your trip...</b>
                        <p>Consulting Maps, Calculating Budget, Finding Hidden Gems...</p>
                        </div>"""))
        
        try:
            # --- CALL THE AI AGENTS ---
            # 1. Logistics
            logistics_res = transport_agent.run(task=f"Find route from {origin} to {dest}. Prioritize Trains, but use Bus/Shared Cab if rail is missing.")
            time.sleep(2) # Prevent rate limiting
            
            # 2. Budget (Pass logistics as context)
            budget_res = budget_agent.run(task=f"Estimate the cost for this trip under minimal budget.", context=logistics_res)
            time.sleep(2)
            
            # 3. Guide (Pass logistics as context to know the destination city)
            guide_res = guide_agent.run(task=f"Guide for {dest} if destination is famous for temple, religion and worship then show first worship places and what should I do and eat in {dest}", context=logistics_res)
            
            # --- RENDER THE RESULT ---
            clear_output() # Clear the loader
            final_html = generate_html(origin, dest, logistics_res, budget_res, guide_res)
            display(HTML(final_html))
            
        except Exception as e:
            clear_output()
            print(f"âš ï¸� Error: {e}")

w_btn.on_click(on_button_click)

# ==========================================
# 4. DISPLAY THE APP
# ==========================================
display(HTML("<h2 style='color:#264653'>ğŸ—ºï¸� Start Your Journey</h2>"))
display(widgets.VBox([w_origin, w_dest, w_btn, w_output]))




