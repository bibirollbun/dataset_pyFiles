# HealthGuard AI - Multi-Agent Healthcare Assistant
# Capstone Project for 5-Day AI Agents Intensive Course with Google
# Track: Agents for Good (Healthcare)

"""
## Project Overview
HealthGuard AI is a multi-agent healthcare assistant that helps users:
1. Understand their symptoms
2. Get preliminary health guidance  
3. Find nearby healthcare facilities

## Key Concepts Demonstrated (3+)
1. Multi-agent System - Multiple specialized agents working together
2. Custom Tools - Health information lookup and facility finder
3. LLM-powered Agents - Using Gemini for intelligent responses
4. Session Management - User conversation tracking
"""

print("ğŸ�¥ HealthGuard AI - Multi-Agent Healthcare System")
print("="*50)


# Install required packages
!pip install -q google-adk google-genai
print("âœ… Packages installed successfully")


# Setup API key and imports
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key configured successfully")
except Exception as e:
    print(f"âš ï¸� Please add GOOGLE_API_KEY to Kaggle secrets: {e}")


# Import ADK components
import uuid
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool

print("âœ… ADK components imported successfully")


# ========== CUSTOM TOOLS ==========
# Tool 1: Symptom Analyzer
def analyze_symptoms(symptoms: str) -> dict:
    """Analyzes symptoms and provides possible conditions.
    Args:
        symptoms: A description of the patient's symptoms
    Returns:
        Analysis with possible conditions and recommendations
    """
    symptom_db = {
        "headache": ["Tension headache", "Migraine", "Dehydration"],
        "fever": ["Viral infection", "Flu", "COVID-19"],
        "cough": ["Common cold", "Bronchitis", "Allergies"],
        "fatigue": ["Anemia", "Sleep disorder", "Stress"],
        "chest pain": ["Muscle strain", "Anxiety", "SEEK IMMEDIATE CARE"]
    }
    found = []
    for s, c in symptom_db.items():
        if s in symptoms.lower():
            found.extend(c)
    return {"symptoms": symptoms, "possible_conditions": list(set(found)) or ["Consult a doctor"], "note": "Please see a healthcare professional"}

# Tool 2: Healthcare Facility Finder  
def find_facility(location: str, facility_type: str = "hospital") -> dict:
    """Finds nearby healthcare facilities."""
    facilities = {
        "hospital": [{"name": "City General Hospital", "distance": "2.5 km"}],
        "clinic": [{"name": "Family Care Clinic", "distance": "1.0 km"}],
        "pharmacy": [{"name": "HealthPlus Pharmacy", "distance": "0.5 km"}]
    }
    return {"location": location, "type": facility_type, "facilities": facilities.get(facility_type.lower(), facilities["hospital"])}

print("âœ… Custom tools created: analyze_symptoms, find_facility")


# ========== MULTI-AGENT SYSTEM ==========
# Create FunctionTools
symptom_tool = FunctionTool(analyze_symptoms)
facility_tool = FunctionTool(find_facility)

# Define Model
model = Gemini(model="gemini-2.0-flash")

# Sub-Agent 1: Symptom Analyzer
symptom_agent = LlmAgent(
    name="SymptomAnalyzer",
    model=model,
    instruction="""You analyze patient symptoms and provide preliminary insights.
    Always remind users to consult healthcare professionals.
    Use the analyze_symptoms tool.""",
    tools=[symptom_tool]
)

# Sub-Agent 2: Healthcare Locator
locator_agent = LlmAgent(
    name="HealthcareLocator", 
    model=model,
    instruction="""You help find nearby healthcare facilities.
    Use the find_facility tool.""",
    tools=[facility_tool]
)

# Main Orchestrator Agent
health_guard = LlmAgent(
    name="HealthGuardAI",
    model=model,
    instruction="""You are HealthGuard AI, a friendly healthcare assistant.
    Delegate symptom analysis to SymptomAnalyzer.
    Delegate facility search to HealthcareLocator.
    IMPORTANT: You are NOT a replacement for professional medical advice.""",
    sub_agents=[symptom_agent, locator_agent]
)

print("âœ… Multi-agent system created")
print("ğŸ�¥ Agents: HealthGuardAI (main), SymptomAnalyzer, HealthcareLocator")


# ========== SESSION & RUNNER (Memory Management) ==========
session_service = InMemorySessionService()

runner = Runner(
    agent=health_guard,
    app_name="HealthGuardAI",
    session_service=session_service
)

USER_ID = "patient_001"
SESSION_ID = str(uuid.uuid4())
session = session_service.create_session(app_name="HealthGuardAI", user_id=USER_ID, session_id=SESSION_ID)

print("âœ… Session created with memory management")
print(f"Session ID: {SESSION_ID[:8]}...")


# ========== DEMO: Test the Agent ==========
import asyncio

async def chat(user_message: str):
    """Send a message to HealthGuard AI and get response"""
    print(f"\nğŸ‘¤ User: {user_message}")
    print("-" * 40)
    
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_message):
        if hasattr(event, 'content') and event.content:
            if hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"ğŸ¦  HealthGuard AI: {part.text}")

# Test the multi-agent system
print("\n" + "="*50)
print("ğŸ�¥ HEALTHGUARD AI - DEMO")
print("="*50)

await chat("I have a headache and fever. What could it be?")
await chat("Can you find me a nearby clinic?")

