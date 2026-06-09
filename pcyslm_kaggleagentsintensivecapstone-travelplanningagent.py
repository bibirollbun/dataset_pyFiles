# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q google-adk litellm
!pip install --upgrade transformers accelerate bitsandbytes -q


import asyncio
import os
import torch
import gc
import json
import requests
import datetime
import time
import threading
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

# Transformers & ADK
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.adk.agents import LlmAgent, Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import Session
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool
from google.adk.apps import App
import nest_asyncio
from flask import Flask, request, jsonify
import sqlite3

# ==========================================
# 0. æ¸…ç�†èˆ‡æº–å‚™ (No-Key Mode)
# ==========================================
for key in ['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY']:
    if key in os.environ:
        del os.environ[key]

print("ğŸ”’ å·²æ¸…é™¤å¤–éƒ¨ API é‡‘é‘°")
print("ğŸ§¹ æ¸…ç�† GPU è¨˜æ†¶é«”...")
torch.cuda.empty_cache()
gc.collect()

# æ¨¡æ“¬ ADK Types (é�¿å…� Kaggle ç’°å¢ƒç¼ºä»¶)
def get_adk_types_module():
    try:
        import google.adk.types as t
        return t
    except ImportError:
        if 'types' in Runner.run_async.__globals__:
            return Runner.run_async.__globals__['types']
        class MockPart:
            def __init__(self, text): self.text = text
        class MockContent:
            def __init__(self, role, parts): self.role = role; self.parts = parts
        class MockTypes:
            Part = MockPart
            Content = MockContent
        return MockTypes()

adk_types = get_adk_types_module()
print("âœ… Types æ¨¡çµ„è¼‰å…¥")


# ==========================================
# 1. Model Load 
# ==========================================
MODEL_PATH = "/kaggle/input/gemma-3/transformers/gemma-3-4b-it/1"
#MODEL_PATH = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2"

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "google/gemma-3-4b-it"

print(f"ğŸ“‚Loading Model from: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        # æ ¹æ“šè¨˜æ†¶é«”å‹•æ…‹èª¿æ•´è¼‰å…¥æ–¹å¼�
        if free_memory < 8e9:
            hf_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, load_in_8bit=True, device_map="auto", trust_remote_code=True
            )
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
            )
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float32, trust_remote_code=True
        )
    print(f"âœ… Model Load Completed")
except Exception as e:
    print(f"â�Œ Model Load Failed: {e}")
    exit(1)


# ==========================================
# 2. Local Model (Gemma-3-4b-it) Inference Server with OpenAI API Compatibility
# ==========================================
app = Flask(__name__)

# Add Health Check, allowing LiteLLM to verify the connection
@app.route('/v1/models', methods=['GET'])
def models():
    return jsonify({"data": [{"id": "gpt-3.5-turbo", "object": "model"}]})

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Simulate the OpenAI Chat Completions API"""
    try:
        data = request.json
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 1024)  # Increase generation length to ensure complete answers
        
        # Construct Prompt (Convert Message List to String)
        prompt = ""
        system_instruction = ""
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                system_instruction += f"System: {content}\n"
            elif role == 'user':
                prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == 'assistant':
                prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            elif role == 'tool':
                prompt += f"<start_of_turn>tool\n{content}<end_of_turn>\n"

        # Combine System Prompt and Conversation
        full_prompt = f"<start_of_turn>system\n{system_instruction}<end_of_turn>\n{prompt}<start_of_turn>model\n"
        
        # Generate
        inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return jsonify({
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "local-gemma",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })
        
    except Exception as e:
        print(f"[API Error] {e}")
        return jsonify({"error": str(e)}), 500

def run_server():
    app.run(host='127.0.0.1', port=8000, debug=False, use_reloader=False)

# Start Server
print("ğŸš€ Starting Local API Server...")
threading.Thread(target=run_server, daemon=True).start()

# Wait for Server to be ready
print("â�³ Waiting for Server connection...")
server_ready = False
for _ in range(15):
    try:
        requests.get("http://127.0.0.1:8000/v1/models", timeout=1)
        server_ready = True
        print("âœ… Server is ready!")
        break
    except:
        time.sleep(1)

if not server_ready:
    print("â�Œ Server startup failed. Please check Port 8000")


# ==========================================
# 3. Define Tools (Web Search, Map, etc.)
# ==========================================

def search_web_duckduckgo(query: str) -> str:
    """[Search Tool] Search the web for real-time prices and information."""
    print(f"    ğŸ”� [Web Search] {query}")
    url = "https://html.duckduckgo.com/html/"
    try:
        r = requests.post(url, data={'q': query}, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        results = []
        for res in soup.find_all('div', class_='result__body')[:3]:
            t = res.find('a', class_='result__a')
            s = res.find('a', class_='result__snippet')
            if t and s:
                results.append(f"Title: {t.text.strip()}\nSummary: {s.text.strip()}")
        return "\n---\n".join(results) if results else "No data found"
    except Exception as e:
        return f"Search Error: {e}"

def search_osm_location(query: str) -> str:
    """[Location Tool] Query location latitude, longitude, and address."""
    print(f"    ğŸŒ� [Location] {query}")
    url = "https://nominatim.openstreetmap.org/search"
    try:
        r = requests.get(url, params={'q': query, 'format': 'json', 'limit': 1}, headers={'User-Agent': 'TravelAgent/1.0'}, timeout=5)
        data = r.json()
        if data:
            return f"Location: {data[0]['display_name']}\nCoordinates: ({data[0]['lat']}, {data[0]['lon']})"
        return "Location not found"
    except Exception as e:
        return f"Map API Error: {e}"

def get_coordinates(query: str) -> tuple[float, float] | None:
    # This is a simplification of your existing search_osm_location logic, assuming it returns (lat, lon)
    # In reality, you need to parse the location name and obtain the coordinates
    # ... (Internal logic calling search_osm_location)
    try:
        # NOTE: This line requires the actual 'search_osm_location' function to be defined elsewhere.
        location_data = search_osm_location(query) 
        
        # Simple parsing example; real parsing needs to be more robust
        if "åº§æ¨™" in location_data:
            # The original code's parsing for the Chinese string "åº§æ¨™: (" is kept for demonstration 
            # based on the assumption that location_data is still returning the string in Chinese from search_osm_location.
            # However, if search_osm_location is also translated, this parsing logic must be updated.
            # Assuming location_data is still in the original format for now:
            coords = location_data.split("åº§æ¨™: (")[1].split(")")[0].split(', ')
            return float(coords[0]), float(coords[1]) # (lat, lon)
        return None
    except:
        return None

def check_traffic_time_osrm(origin: str, destination: str, mode: str = "driving") -> str:
    """[Traffic Tool] Use OSRM public server for travel time."""
    print(f"    ğŸš¦ [Traffic Check - OSRM] {origin} to {destination} by {mode}")
    
    # 1. Get coordinates for origin and destination
    origin_coords = get_coordinates(origin)
    destination_coords = get_coordinates(destination)

    if not origin_coords or not destination_coords:
        return "Error: Could not find coordinates for origin or destination."

    # OSRM uses (lon, lat) format
    lon1, lat1 = origin_coords[1], origin_coords[0]
    lon2, lat2 = destination_coords[1], destination_coords[0]

    # 2. Construct OSRM API request (using the specified mode, defaults to driving)
    # OSRM Public API Endpoint (router.project-osrm.org)
    osrm_url = f"http://router.project-osrm.org/route/v1/{mode}/{lon1},{lat1};{lon2},{lat2}"

    try:
        r = requests.get(osrm_url, timeout=5)
        r.raise_for_status() # Check for HTTP errors
        data = r.json()

        if data.get('code') == 'Ok' and data['routes']:
            # duration is in seconds
            duration_sec = data['routes'][0]['duration']
            distance_m = data['routes'][0]['distance']
            
            # Convert to hours and minutes
            minutes = int(duration_sec // 60)
            hours = minutes // 60
            minutes %= 60
            
            distance_km = distance_m / 1000

            return f"Travel Time: {hours}h {minutes}m ({int(duration_sec)} seconds)\nDistance: {distance_km:.2f} km"
        else:
            return f"Routing Error: {data.get('code', 'Unknown Error')}"
            
    except requests.exceptions.RequestException as e:
        return f"OSRM API Connection Error: {e}"
    except Exception as e:
        return f"Processing Error: {e}"

def search_hotels(location: str) -> str:
    """[Hotel Tool] Search for hotels and prices in the area."""
    query = f"hotels in {location} price per night"
    print(f"    ğŸ�¨ [Hotel Search] {query}")
    return search_web_duckduckgo(query)

print("Function Tool Load Completed")

class SqliteSessionService(InMemorySessionService):
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self._init_db()
        print(f"âœ… SqliteSessionService initialized: {db_path}")

    def _init_db(self):
        """åˆ�å§‹åŒ–è³‡æ–™åº«è¡¨æ ¼"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS sessions
                         (session_id TEXT PRIMARY KEY, 
                          history_json TEXT,
                          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()
            conn.close()
            print("âœ… Database table created/verified")
        except Exception as e:
            print(f"âš ï¸� DB Init Error: {e}")

    async def get_session(self, session_id: str) -> Session:
        """å¾�è³‡æ–™åº«è®€å�– session"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT history_json FROM sessions WHERE session_id=?", (session_id,))
            row = c.fetchone()
            conn.close()

            session = Session(session_id=session_id)
            if row and row[0]:
                print(f"ğŸ“– Loading session {session_id} from DB")
                history_data = json.loads(row[0])
                # æ ¹æ“š ADK çš„å¯¦éš›çµ�æ§‹é‡�å»º history
                for item in history_data:
                    session.history.append(item)
            else:
                print(f"ğŸ†• Creating new session {session_id}")
            
            return session
        except Exception as e:
            print(f"âš ï¸� Get Session Error: {e}")
            return Session(session_id=session_id)

    async def save_session(self, session: Session):
        """å„²å­˜ session åˆ°è³‡æ–™åº«"""
        try:
            # å°‡ history åº�åˆ—åŒ–
            history_list = []
            for msg in session.history:
                if isinstance(msg, dict):
                    history_list.append(msg)
                else:
                    # è™•ç�†ç‰©ä»¶é¡�å�‹çš„è¨Šæ�¯
                    try:
                        # å˜—è©¦ä½¿ç”¨ model_dump (Pydantic v2) æˆ– dict()
                        if hasattr(msg, 'model_dump'):
                            history_list.append(msg.model_dump())
                        elif hasattr(msg, 'dict'):
                            history_list.append(msg.dict())
                        else:
                            # æ‰‹å‹•è½‰æ�›
                            msg_dict = {
                                "role": getattr(msg, 'role', 'unknown'),
                                "content": str(getattr(msg, 'content', ''))
                            }
                            history_list.append(msg_dict)
                    except Exception as e:
                        print(f"âš ï¸� Error converting message: {e}")
                        continue
            
            history_json = json.dumps(history_list, ensure_ascii=False)

            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""INSERT INTO sessions (session_id, history_json, updated_at) 
                         VALUES (?, ?, CURRENT_TIMESTAMP) 
                         ON CONFLICT(session_id) DO UPDATE SET 
                         history_json=excluded.history_json,
                         updated_at=CURRENT_TIMESTAMP""",
                      (session.session_id, history_json))
            conn.commit()
            conn.close()
            print(f"ğŸ’¾ Session {session.session_id} saved ({len(history_list)} messages)")
        except Exception as e:
            print(f"âš ï¸� Save Session Error: {e}")
            import traceback
            traceback.print_exc()

    async def delete_session(self, session_id: str):
        """åˆªé™¤ session"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            conn.commit()
            conn.close()
            print(f"ğŸ—‘ï¸� Session {session_id} deleted")
        except Exception as e:
            print(f"âš ï¸� Delete Session Error: {e}")


# ==========================================
# 4. Define Model and Agents
# ==========================================
     
# Set environment variables for LiteLLM usage
os.environ['OPENAI_API_BASE'] = 'http://127.0.0.1:8000/v1'
os.environ['OPENAI_API_KEY'] = 'dummy-key'

# Core Model Instance (Shared by all Agents)
# Use "gpt-3.5-turbo" as the name to pass LiteLLM validation, 
# but the request actually goes to our Local Gemma server.
local_model = LiteLlm(
    model="openai/gpt-3.5-turbo", 
    api_base="http://127.0.0.1:8000/v1"
)

agent_flight_all_in_one = Agent(
    name="Flight_Service",
    description="Search flights and provide top 3 recommendations (Best Budget, Best Time, Best Overall). Use when user needs flights between cities.",
    instruction="""You are a flight service expert.

**Your Task:**
1. Use search_web_duckduckgo to find flight options
2. Analyze and recommend TOP 3 options:
   - Best Budget
   - Best Time  
   - Best Overall
3. Save results to {flight_recommendations}""",
    model=local_model,
    tools=[search_web_duckduckgo],
    output_key="flight_recommendations"
)

agent_hotel_all_in_one = Agent(
    name="Hotel_Service",
    description="Search hotels and provide top 3 location-optimized recommendations. Use when user needs accommodation.",
    instruction="""You are a hotel service expert.
    
**Your Task:**
1. Use search_hotels to find options
2. Analyze locations and recommend TOP 3:
   - Best Location
   - Best Budget
   - Best Overall
3. Save results to {hotel_recommendations}""",
    model=local_model,
    tools=[search_hotels],
    output_key="hotel_recommendations"
)

agent_itinerary_all_in_one = Agent(
    name="Itinerary_Service",
    description="Create optimized daily itineraries with traffic analysis and route optimization. ALWAYS use for trip planning.",
    instruction="""You are an itinerary service expert.
    According to the {flight_recommendations}, {hotel_recommendations}:
**Your Task:**
1. Use search_osm_location to find attractions
2. Use check_traffic_time_osrm to check travel times
3. Create optimized daily schedules with transportation
4. Save to {optimized_itinerary}""",
    model=local_model,
    tools=[search_osm_location, search_web_duckduckgo, check_traffic_time_osrm],
    output_key="optimized_itinerary"
)

flight_tool = AgentTool(agent=agent_flight_all_in_one)
hotel_tool = AgentTool(agent=agent_hotel_all_in_one)
itinerary_tool = AgentTool(agent=agent_itinerary_all_in_one)


coord_agent = Agent(
    name="CoordAgent",
    description="Generates raw, detailed travel plans using search tools.",
    instruction="""
    You are the content generator.
    1. Analyze the request and call necessary tools (Flight, Hotel, Itinerary).
    2. Generate a DATA-RICH, detailed report.
    3. Include all technical details (prices, times, flight numbers).
    """,
    model=local_model,
    tools=[flight_tool, hotel_tool, itinerary_tool], 
    output_key="final_proposal"
)

agent_summary_optimizer = Agent(
    name="Summary_Optimizer",
    description="Transforms text into structured tables.",
    instruction="""
    You are a formatting engine based on the information {final_proposal}.
    
    **INPUT:** You will receive a detailed travel text.
    
    **TASK:** Refine the input into a structured summary:
    1. ğŸ’° Budget Table
    2. ğŸ—“ï¸� Daily Schedule Table
    3. âœ… Action Checklist
    
    **CONSTRAINT:** Do not invent new facts. Just format the input.
    """,
    model=local_model,
)

travel_sequence_agent = SequentialAgent(
    name="Travel_Sequence_Pipeline",
    #description="A strict pipeline that plans the trip first, then formats the output.",
    sub_agents=[coord_agent, agent_summary_optimizer] 
)

travel_sequence_tool = AgentTool(agent=travel_sequence_agent)

root_agent = Agent(
    name="Master_Router",
    description="Intelligent Gatekeeper that routes requests based on intent.",
    instruction="""
    You are the Master Router.

    **YOUR JOB:**
    Evaluate the user's input string.

    **DECISION TREE:**
    
    1. **IF** the input is related to Travel, Vacations, Flights, Hotels, or Itineraries:
       -> **ACTION:** Call the `Travel_Sequence_Pipeline` tool.
       -> Pass the user's ORIGINAL request directly to it.
    
    2. **IF** the input is anything else (Greeting, Coding, Math, Weather):
       -> **ACTION:** Answer directly using your own knowledge.
       -> DO NOT call the travel tool.
    """,
    model=local_model,
    tools=[travel_sequence_tool], 
    #show_tool_calls=True,
    #markdown=True
)

print("âœ… Architecture Updated with SequentialAgent")
print("   Flow: [Root] -> [Sequential: Coord -> Summary]")


DB_FOLDER = "data"   
my_db_service = SqliteSessionService(db_path="travel_agent.db")

# Create App with root_agent
travel_app = App(
    name="TravelPlanningAgent",  # App name
    root_agent=root_agent  # The decision-making manager agent
)
print("âœ… App 'TravelPlanningAgent' created")

# Verify root_agent exists for ADK Web
if "root_agent" in globals():
    print("\nâœ… [ADK Web Check] 'root_agent' variable found!")
    print(f"   â†’ Ready for: adk web --port 8080")
else:
    print("\nâ�Œ [ADK Web Check] ERROR: 'root_agent' variable is MISSING!")


runner = InMemoryRunner(
    app=travel_app,  # Pass the App object
    #session_service=my_db_service,  # Enable session management
)


initial_request = "ä¸‹æ˜ŸæœŸå…­æˆ‘è¦�åœ¨æ�±äº¬5æ—¥æ—…é�Šï¼Œæœ‰æ²’æœ‰é…’åº—å»ºè­°?"
response = await runner.run_debug(initial_request)
print("\n--- Pipeline Execution Summary ---")
print(f"Initial Request: {initial_request}")
print(f"Final Report: {response}")




initial_request = "ä¸‹æ˜ŸæœŸå…­æˆ‘è¦�åœ¨æ�±äº¬5æ—¥æ—…é�Šï¼Œæœ‰æ²’æœ‰è¡Œç¨‹ä¸Šçš„å»ºè­°? åŒ…å�«é…’åº—ã€�èˆªç�­ã€�äº¤é€šèˆ‡è·¯ç·šé �ä¼°èŠ±è²»ã€‚"
response = await runner.run_debug(initial_request)
print("\n--- Pipeline Execution Summary ---")
print(f"Initial Request: {initial_request}")
print(f"Final Report: {response}")


initial_request = "I am taking a 5-day trip to Tokyo starting next Saturday. Do you have any itinerary suggestions? Please include details on hotels, flight, transportation, and estimated costs."
response = await runner.run_debug(initial_request)
print("\n--- Pipeline Execution Summary ---")
print(f"Initial Request: {initial_request}")
print(f"Final Report: {response}")

