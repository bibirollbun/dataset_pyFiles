# Install required packages
!pip install google-genai gradio reportlab matplotlib pandas pillow pyngrok -q
print("âœ… All packages installed successfully!")


import gradio as gr
from google import genai
from google.genai import types
import json
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from io import BytesIO
import os
import socket

# Get API key from Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GEMINI_API_KEY")
    print("âœ… API key loaded from Kaggle secrets")
except Exception as e:
    # REPLACE THIS WITH YOUR ACTUAL KEY IF NOT ON KAGGLE
    api_key = "YOUR_API_KEY_HERE" 
    print("âš ï¸� Using hardcoded API key or Placeholder.")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Use stable model
MODEL = 'gemini-2.0-flash'
print(f"âœ… Gemini client initialized with model: {MODEL}")


# Master System Prompt for Odyssey Orchestrator
SYSTEM_PROMPT = """### SYSTEM ROLE: ODYSSEY ORCHESTRATOR
You are "Odyssey," an elite AI Travel Concierge powered by a Multi-Agent System. Your goal is to deliver hyper-personalized, safe, and financially optimized travel experiences.

### 1. CORE ARCHITECTURE (THE TEAM)
You are the Manager Agent. You delegate tasks to these specialized sub-agents:
- **CURATOR (Planner):** Drafts itineraries based on "Vibe," Budget, and Logistics.
- **GUARDIAN (Safety):** REAL-TIME monitor for geolocation safety, medical triage, and dietary restrictions.
- **CFO (Finance):** Tracks spending, converts currency, and re-optimizes budgets if limits are breached.
- **EVALUATOR (Quality Control):** A distinct "Critic" loop that verifies facts (e.g., "Is the Louvre open on Tuesdays?") BEFORE the user sees the plan.
- **BIOGRAPHER (Summary):** Synthesizes the entire trip into a "Travel DNA" report and generates offline-ready PDFs.

### 2. MEMORY PROTOCOLS
- **Long-Term Memory:** You must access the user's profile data. If a user mentions dietary restrictions, medical conditions, or preferences, remember them for all future interactions.
- **Session Context:** Maintain awareness of the current Trip ID, Location, Daily Spend, and Budget Status.

### 3. OPERATIONAL RULES
- **Safety First:** If a user mentions "pain," "sick," "emergency," or "help," DROP all other tasks. Activate the Guardian Agent to find the nearest hospital/pharmacy immediately.
- **The "No-Hallucination" Gate:** No itinerary is presented to the user until the EVALUATOR agent returns a `STATUS: PASS`.
- **Offline Readiness:** If the user asks for a download or export, trigger the offline PDF generation.
- **Budget Enforcement:** If spending exceeds budget, CFO agent must immediately suggest cost-cutting alternatives.

### 4. OUTPUT FORMATTING
- When providing an itinerary, ALWAYS structure it clearly with "Day 1:", "Day 2:", etc.
- Use Bullet points for activities.
- Bold key locations.

### 5. USER PROFILE STRUCTURE
Maintain and reference:
- Name and basic demographics
- Budget preferences and spending history
- Dietary restrictions and allergies (CRITICAL)
- Medical conditions and medications
- Travel style preferences (adventure, relaxation, cultural, etc.)
"""

print("âœ… Odyssey Master System Prompt configured")


class MemoryBank:
    """Persistent storage for user profiles and trip data"""
    
    def __init__(self):
        self.user_profile = {
            "name": "Traveler",
            "dietary_restrictions": [],
            "allergies": [],
            "medical_conditions": [],
            "travel_style": "balanced"
        }
        self.current_trip = {
            "destination": None,
            "budget": 0,
            "spent": 0,
            "latest_plan": "", # CRITICAL: Stores the full text of the plan for PDF
        }
        self.logs = []
    
    def update_profile(self, category, value):
        if category in self.user_profile:
            if isinstance(self.user_profile[category], list):
                if value not in self.user_profile[category]:
                    self.user_profile[category].append(value)
            self.log_event("MEMORY", "Profile Update", f"Added '{value}' to {category}")

    def log_event(self, agent, action, details):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] ğŸ¤– {agent.upper()}: {action} -> {details}"
        self.logs.append(entry)
    
    def get_profile_summary(self):
        return f"""
USER PROFILE:
- Name: {self.user_profile['name']}
- Allergies: {', '.join(self.user_profile['allergies']) or 'None'}
- Dietary: {', '.join(self.user_profile['dietary_restrictions']) or 'None'}

CURRENT TRIP DATA:
- Destination: {self.current_trip['destination'] or 'Not set'}
- Budget: ${self.current_trip['budget']}
"""
    
    def get_logs(self):
        return "\n".join(self.logs)

# Initialize global memory bank
memory = MemoryBank()
print("âœ… Memory Bank initialized")


class OdysseyManager:
    """Main orchestrator that coordinates all agents"""
    
    def __init__(self):
        self.conversation_history = []
    
    def chat(self, user_message, chat_history):
        if not user_message.strip():
            return chat_history, ""
        
        # Log user message
        memory.log_event("USER", "Message", user_message)
        
        # Build context
        context = f"{SYSTEM_PROMPT}\n\n{memory.get_profile_summary()}\n\nLOGS:\n{memory.get_logs()[-500:]}"
        self.conversation_history.append({"role": "user", "content": user_message})
        
        try:
            # Call Gemini
            search_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(
                tools=[search_tool],
                system_instruction=context,
                temperature=0.7
            )
            
            response = client.models.generate_content(
                model=MODEL,
                contents=user_message,
                config=config
            )
            
            assistant_message = response.text
            
            # --- CRITICAL FIX FOR PDF CONTENT ---
            # If the response looks like a plan (long text), save it explicitly for the PDF
            if len(assistant_message) > 300:
                memory.current_trip['latest_plan'] = assistant_message
                memory.log_event("MANAGER", "Plan Saved", "Itinerary stored for PDF generation")
            
            # Extract basic data
            if "trip to" in user_message.lower():
                words = user_message.split()
                if "to" in words:
                    try:
                        dest = words[words.index("to")+1]
                        memory.current_trip['destination'] = dest.capitalize()
                    except: pass
            
            # Log and return
            memory.log_event("MANAGER", "Response", f"{len(assistant_message)} chars")
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            chat_history.append([user_message, assistant_message])
            return chat_history, ""
        
        except Exception as e:
            err = f"Error: {str(e)}"
            chat_history.append([user_message, err])
            return chat_history, ""

    def reset_conversation(self):
        self.conversation_history = []
        return [], ""

odyssey = OdysseyManager()
print("âœ… Odyssey Manager initialized")


from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

class BiographerAgent:
    @staticmethod
    def generate_pdf(trip_data, user_profile):
        try:
            filename = f"Odyssey_Trip_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
            filepath = os.path.join(os.getcwd(), filename)
            
            # 1. GET CONTENT
            # We prioritize the text stored in 'latest_plan'
            full_content = trip_data.get('latest_plan', "")
            
            if not full_content:
                full_content = "No itinerary found. Please ask Odyssey to plan a trip first."
            
            # 2. SETUP PDF
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # 3. STYLES
            title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=24, spaceAfter=20, textColor=colors.HexColor('#2563eb'))
            h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=16, spaceBefore=15, textColor=colors.HexColor('#0ea5e9'))
            body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=11, leading=14)
            
            # 4. HEADER
            story.append(Paragraph(f"âœˆï¸� Trip to {trip_data.get('destination', 'Unknown')}", title_style))
            story.append(Paragraph(f"Traveler: {user_profile['name']}", body_style))
            story.append(Spacer(1, 0.2*inch))
            
            # 5. CONTENT PARSING (Markdown to PDF)
            lines = full_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.1*inch))
                    continue
                
                # Headers
                if line.startswith('##') or line.startswith('**Day'):
                    clean = line.replace('#', '').replace('*', '').strip()
                    story.append(Paragraph(clean, h2_style))
                # Bullets
                elif line.startswith('-') or line.startswith('* '):
                    clean = line.replace('*', '').replace('-', '').strip()
                    story.append(Paragraph(f"â€¢ {clean}", body_style))
                # Normal text
                else:
                    clean = line.replace('**', '') # Remove bold markdown
                    story.append(Paragraph(clean, body_style))
            
            # 6. FOOTER / EMERGENCY
            story.append(PageBreak())
            story.append(Paragraph("ğŸš¨ EMERGENCY INFO", h2_style))
            story.append(Paragraph(f"Allergies: {', '.join(user_profile['allergies']) or 'None'}", body_style))
            story.append(Paragraph("Global Emergency: 112 / 911", body_style))
            
            doc.build(story)
            return filepath
            
        except Exception as e:
            print(f"PDF Error: {e}")
            return None

def export_callback():
    return BiographerAgent.generate_pdf(memory.current_trip, memory.user_profile)

print("âœ… Biographer Agent Ready")


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# âœˆï¸� ODYSSEY AI")
    
    with gr.Row():
        with gr.Column(scale=1):
            logs_box = gr.Textbox(label="System Logs", lines=10, interactive=False)
            btn_pdf = gr.Button("ğŸ“„ Download Trip PDF", variant="primary")
            file_pdf = gr.File(label="Your Guide")
            
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, type="tuples")
            msg_input = gr.Textbox(placeholder="Plan a 5-day trip to Tokyo...")
            btn_send = gr.Button("Send")

    # Events
    def respond(msg, hist):
        return odyssey.chat(msg, hist)
    
    def get_logs():
        return memory.get_logs()
        
    def download():
        path = export_callback()
        return path if path else None

    msg_input.submit(respond, [msg_input, chatbot], [chatbot, msg_input])
    btn_send.click(respond, [msg_input, chatbot], [chatbot, msg_input])
    btn_pdf.click(download, None, file_pdf)
    chatbot.change(get_logs, None, logs_box)

print("âœ… UI Built")


from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient
import socket

# 1. Authenticate
try:
    user_secrets = UserSecretsClient()
    ngrok_token = user_secrets.get_secret("NGROK_AUTH_TOKEN")
    ngrok.set_auth_token(ngrok_token)
    print("âœ… Ngrok authenticated")
except:
    print("âš ï¸� Ngrok token missing (Check Kaggle Secrets)")

# 2. AGGRESSIVE CLEANUP (Fixes ERR_NGROK_108)
print("ğŸ”„ Cleaning up old processes...")
import os
os.system("killall ngrok") # Kills all ngrok processes
time.sleep(2) # Wait for system to release lock

# 3. Find Free Port
def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

# 4. Launch
try:
    PORT = get_free_port()
    print(f"âœ… Starting on port {PORT}")
    
    # Connect Ngrok
    public_url = ngrok.connect(PORT).public_url
    print(f"ğŸ”— PUBLIC URL: {public_url}")
    
    # Launch Gradio
    demo.launch(server_port=PORT, inline=False, share=False)
    
except Exception as e:
    print(f"â�Œ Error: {e}")

