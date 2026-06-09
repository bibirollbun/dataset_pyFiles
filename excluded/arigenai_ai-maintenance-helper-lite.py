!pip install --upgrade google-adk google-generativeai nest_asyncio


import os
import json
import asyncio
from datetime import datetime
import google.generativeai as genai
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from kaggle_secrets import UserSecretsClient
import nest_asyncio
nest_asyncio.apply()

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")


os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

print("Setup complete!")


def severity_rater(symptoms: str, error_code: str = "") -> dict:
    """
    Rates severity from 1-10 based on symptoms.
    """
    high = ['fire', 'smoke', 'spark', 'burning', 'leak', 'shock']
    medium = ['noise', 'vibration', 'error', 'not draining', 'not spinning', 'e21']
    score = 5  # default

    text = f"{symptoms} {error_code}".lower()
    if any(word in text for word in high):
        score = 9
    elif any(word in text for word in medium):
        score = 6
    else:
        score = 4

    return {
        "severity_rating": score,
        "reason": f"Keywords found: {[w for w in high+medium if w in text][:3]}",
        "safety_note": "Unplug immediately if score ≥ 8"
    }

print("severity_rater tool ready!")


from google.adk.agents import Agent
from google.adk.tools import google_search

search_agent = Agent(
    name="search",
    model="gemini-2.0-flash",
    instruction="Search the web for service manuals, error code databases, forums, YouTube repair videos, official documentation, and technical bulletins for ANY machine — consumer electronics, vehicles, home appliances, or heavy industrial equipment. Return bullet points with sources.",
    tools=[google_search]  # Tools needed here
)

diagnosis_agent = Agent(
    name="diagnosis",
    model="gemini-2.0-flash",
    instruction="Use severity_rater. Output partial JSON: likely_causes, possible_checks, severity_rating.",
    tools=[severity_rater]  # Tools needed here
)

maintenance_agent = Agent(
    name="coordinator",
    model="gemini-2.0-flash",
    instruction="""You are the world's best diagnostic technician capable of troubleshooting ANY machine ever built: 
    - Consumer: phones, laptops, cars, bikes, drones, cameras, watches
    - Home appliances: washing machines, refrigerators, ACs, microwaves, ovens
    - Industrial: CNC machines, injection molding, compressors, pumps, motors, PLCs, robots, turbines, generators, HVAC chillers, conveyor systems, etc.

    ALWAYS respond with a single, beautifully formatted, human-readable JSON using proper indentation and spaces.
    Use this exact structure (pretty-print is mandatory):
    {
      "machine": "full name + brand + model if mentioned",
      "problem": "short summary",
      "likely_causes": [...],
      "diagnostic_steps": [...],
      "severity_rating": "1–10 or Low/Medium/High/Critical",
      "immediate_safe_actions": [...],
      "caution_notes": [...],
      "recommended_next_step": "self-fix / local technician / authorized service / factory support"
    }
    Never wrap in markdown. Always make it pretty and easy to read.""",
    tools=[]  
)

print("Agents ready!")


from google.adk.tools.preload_memory_tool import PreloadMemoryTool
for agent in [search_agent, diagnosis_agent, maintenance_agent]:
    agent.tools.append(PreloadMemoryTool())
print("Memory tool added")


import uuid
import json
import re
import asyncio
from google.genai import types
from google.adk.runners import Runner

APP_NAME = "maint"
USER_ID = "user"

async def run_one(agent, q):
    fresh_session_id = str(uuid.uuid4())
    s = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=fresh_session_id
    )
    r = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service
    )
    msg = types.Content(role="user", parts=[types.Part(text=q)])
    events = []
    async for e in r.run_async(user_id=USER_ID, session_id=s.id, new_message=msg):
        events.append(e)
    # Safely extract text
    if events and hasattr(events[-1], 'content') and events[-1].content.parts:
        return events[-1].content.parts[0].text
    return "No response"

# SUPER ROBUST JSON EXTRACTOR
def extract_json(text):
    # Remove markdown code blocks
    text = re.sub(r'json\s*|\s*', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # Find the first { and last } 
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1 or end == 0:
        return None
    
    json_str = text[start:end]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Last resort: add spaces after commas and colons
        json_str = re.sub(r'([}\]])([{\[])', r'\1, \2', json_str)  # fix missing commas
        json_str = re.sub(r'([a-zA-Z0-9"\]])\s*([{\[])', r'\1, \2', json_str)
        json_str = re.sub(r'([}\]])[ \t\r\n]+([}\]])', r'\1\2', json_str)
        json_str = json_str.replace('}{', '},{')  # fix missing commas between objects
        try:
            return json.loads(json_str)
        except:
            return {"error": "JSON parse failed", "raw": text[:1000]}

async def run_diagnosis(q):
    search_res, diag_res = await asyncio.gather(
        run_one(search_agent, q),
        run_one(diagnosis_agent, q)
    )
    coord_input = f"{q}\n\nSEARCH RESULTS:\n{search_res}\n\nPARTIAL DIAGNOSIS:\n{diag_res}"
    final_res = await run_one(maintenance_agent, coord_input)
    
    result = extract_json(final_res)
    
    # Store in memory
    sev = result.get("severity_rating", "?") if isinstance(result, dict) else "?"
    diag_session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=str(uuid.uuid4())
    )
    diag_session.state = {"diagnosis": f"Previous diagnosis: {q} → Severity {sev}"}
    await memory_service.add_session_to_memory(diag_session)
    
    return result

# RUN IT
print("Running diagnosis...")
result = asyncio.run(run_diagnosis("Washing machine Whirlpool WTW5000DW error E21, water not draining."))
print("\nFINAL DIAGNOSIS:")
print(json.dumps(result, indent=2))

print("\nRunning follow-up...")
result2 = asyncio.run(run_diagnosis("What if the pump is noisy?"))
print("\nFOLLOW-UP (with Memory Recall):")
print(json.dumps(result2, indent=2))

