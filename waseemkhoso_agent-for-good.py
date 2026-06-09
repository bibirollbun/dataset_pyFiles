# Agents Intensive Capstone — Colab Notebook (Track: Agents for Good)
# Project: Disaster Relief Multi-Agent System (Colab-style)

"""
Colab Notebook: Disaster Relief Multi-Agent System using Gemini + ADK
Track: Agents for Good
Author: Waseem Abbas  (ready-to-submit)

This notebook implements a multi-agent system that helps communities during disasters by: 
- Providing real-time (or simulated) disaster updates
- Looking up relief centers (custom tool)
- Generating short SMS-friendly alerts
- Managing sessions & memory (InMemory + Memory Bank)
- Observability (logging + simple tracing)
- Agent evaluation (A2A evaluator)
"""

# -------------------------
# 0. Environment Setup (Colab / Kaggle)
# -------------------------

# !pip install -q google-genai adk kagglehub rich xlsxwriter

# -------------------------
# 1. CONFIG: Add API Keys
# -------------------------

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # set before running LLM cells
GOOGLE_CUSTOM_SEARCH_KEY = "YOUR_GOOGLE_SEARCH_API_KEY"  # optional
GOOGLE_CUSTOM_SEARCH_CX = "YOUR_SEARCH_ENGINE_ID"  # optional

# -------------------------
# 2. Imports & Utilities
# -------------------------

import time, json, logging
from typing import Dict, Any, Optional, List
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('disaster_agent')

def trace(step: str, info: Optional[Dict[str, Any]] = None):
    logger.info(f"TRACE: {step} | {json.dumps(info or {})}")

# -------------------------
# 3. Simulated Dataset: Relief Centers
# -------------------------

relief_data = [
    {"city": "Karachi", "center": "Edhi Flood Relief Camp", "contact": "0300-111111", "status": "Open"},
    {"city": "Lahore", "center": "Rescue1122 Emergency Camp", "contact": "042-1122", "status": "Open"},
    {"city": "Quetta", "center": "Red Crescent Center", "contact": "081-9201", "status": "Full"},
    {"city": "Peshawar", "center": "Alkhidmat Camp", "contact": "091-111222", "status": "Open"},
    {"city": "Hyderabad", "center": "Army Flood Relief Camp", "contact": "022-444222", "status": "Open"},
]
relief_df = pd.DataFrame(relief_data)

# -------------------------
# 4. ADK + Gemini: Agent Setup
# -------------------------

class SimpleLLM:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        trace('llm.generate', {'model': self.model, 'prompt': prompt[:120]})
        if self.api_key and self.api_key != "YOUR_GEMINI_API_KEY":
            return "[Simulated Gemini response — replace with real Gemini call when API key is set]"
        if "flood" in prompt.lower():
            return "Move to higher ground. Avoid electricity. Keep emergency kit ready."
        if "relief center" in prompt.lower() or "nearest" in prompt.lower():
            return "I can look up relief centers if you provide a city — use the relief lookup tool."
        return "I'm here to help. Provide more details (city, disaster type)."

llm = SimpleLLM(api_key=GEMINI_API_KEY)

# -------------------------
# 5. Tools (Custom + Built-in placeholders)
# -------------------------

def lookup_relief_center_tool(city: str) -> Dict[str, Any]:
    trace('tool.lookup_relief_center', {'city': city})
    m = relief_df[relief_df['city'].str.lower() == city.lower()]
    if m.empty:
        return {"found": False, "message": f"No relief center found for {city}."}
    row = m.iloc[0].to_dict()
    return {"found": True, "center": row['center'], "contact": row['contact'], "status": row['status']}

def google_search_tool(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    trace('tool.google_search', {'query': query})
    if GOOGLE_CUSTOM_SEARCH_KEY and GOOGLE_CUSTOM_SEARCH_KEY != "YOUR_GOOGLE_SEARCH_API_KEY":
        return [{'title': 'Simulated news', 'snippet': 'Latest flood update — simulated', 'link': 'https://example.com'}]
    return [{'title': 'Simulated news', 'snippet': 'No live search available in demo mode.', 'link': ''}]

def code_exec_tool(code: str) -> Dict[str, Any]:
    trace('tool.code_exec', {'code_snippet': code[:80]})
    try:
        local_vars = {}
        exec(code, {'pd': pd, 'relief_df': relief_df}, local_vars)
        return {'success': True, 'result_vars': list(local_vars.keys())}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# -------------------------
# 6. Sessions & Memory
# -------------------------

class InMemorySessionService:
    def __init__(self):
        self.sessions = {}
    def start(self, session_id: str):
        self.sessions[session_id] = {'history': [], 'created_at': time.time()}
        trace('session.start', {'session_id': session_id})
    def add_message(self, session_id: str, message: str):
        self.sessions.setdefault(session_id, {'history': []})['history'].append({'ts': time.time(), 'msg': message})
    def get_history(self, session_id: str):
        return self.sessions.get(session_id, {}).get('history', [])

class MemoryBank:
    def __init__(self):
        self.memory = {}
    def store(self, user_id: str, key: str, value: Any):
        self.memory.setdefault(user_id, {})[key] = value
        trace('memory.store', {'user_id': user_id, 'key': key})
    def recall(self, user_id: str, key: str) -> Any:
        return self.memory.get(user_id, {}).get(key)

session_service = InMemorySessionService()
memory_bank = MemoryBank()

# -------------------------
# 7. Multi-Agent Orchestration
# -------------------------

class DisasterAssistantAgent:
    def __init__(self, llm: SimpleLLM):
        self.llm = llm
    def handle_request(self, user_id: str, session_id: str, user_message: str) -> Dict[str, Any]:
        trace('agent.handle_request.start', {'user_id': user_id, 'session_id': session_id})
        session_service.add_message(session_id, user_message)
        msg = user_message.lower()

        if 'relief' in msg or 'center' in msg or 'nearest' in msg:
            parts = user_message.split(' in ')
            if len(parts) > 1:
                city = parts[-1].strip().split()[0]
                tool_result = lookup_relief_center_tool(city)
                session_service.add_message(session_id, f"tool_result: {tool_result}")
                memory_bank.store(user_id, 'last_city', city)
                return {'type': 'relief_lookup', 'result': tool_result}
            else:
                return {'type': 'ask_for_city', 'message': 'Please tell me your city, e.g. "nearest relief center in Karachi".'}

        if 'update' in msg or 'news' in msg or 'situation' in msg:
            search_results = google_search_tool(user_message, top_k=3)
            return {'type': 'search', 'results': search_results}

        if 'alert' in msg or 'warning' in msg:
            prompt = f"Generate a short SMS alert: {user_message}"
            alert_text = self.llm.generate(prompt, max_tokens=80)
            short = alert_text[:140]
            return {'type': 'alert', 'alert': short}

        reply = self.llm.generate(user_message)
        return {'type': 'reply', 'reply': reply}

class EvaluatorAgent:
    def __init__(self):
        pass
    def evaluate(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        trace('evaluator.evaluate', {'agent_output_type': agent_output.get('type')})
        text = json.dumps(agent_output)
        unsafe_keywords = ['suicide','harm','explosive']
        for kw in unsafe_keywords:
            if kw in text.lower():
                return {'ok': False, 'reason': f'unsafe keyword found: {kw}'}
        return {'ok': True}

disaster_agent = DisasterAssistantAgent(llm)
evaluator = EvaluatorAgent()

# -------------------------
# 8. Demo Interactions
# -------------------------

user_id = 'user_001'
session_id = 'sess_001'
session_service.start(session_id)

queries = [
    'Find nearest relief center in Karachi',
    'Give me flood safety tips',
    'Generate an alert: Heavy flooding expected tomorrow in Sindh',
    'Show latest situation update for Sindh floods'
]

for q in queries:
    print('=== USER:', q)
    out = disaster_agent.handle_request(user_id, session_id, q)
    eval_result = evaluator.evaluate(out)
    print('AGENT OUTPUT:', out)
    print('EVALUATION:', eval_result)
    print('-----------------------------------')

# -------------------------
# 9. Observability & Metrics
# -------------------------

metrics = {'requests': 0, 'relief_lookups': 0, 'alerts_generated': 0}

def instrumented_handle(user_id: str, session_id: str, message: str):
    metrics['requests'] += 1
    out = disaster_agent.handle_request(user_id, session_id, message)
    if out.get('type') == 'relief_lookup':
        metrics['relief_lookups'] += 1
    if out.get('type') == 'alert':
        metrics['alerts_generated'] += 1
    trace('metrics.update', metrics)
    return out

print('--- Instrumented Run Example ---')
resp = instrumented_handle(user_id, session_id, 'Nearest relief center in Lahore')
print(resp)
print('Metrics:', metrics)

# -------------------------
# 10. Test Cases
# -------------------------

def test_relief_lookup():
    out = disaster_agent.handle_request('user_01', 's1', 'Nearest relief center in Quetta')
    assert out['type'] == 'relief_lookup'
    assert out['result']['found'] == True
    print('test_relief_lookup passed')

def test_alert_generation():
    out = disaster_agent.handle_request('user_01', 's1', 'Generate an alert: Earthquake expected near Karachi')
    assert out['type'] == 'alert'
    print('test_alert_generation passed')

print('--- Running Tests ---')
test_relief_lookup()
test_alert_generation()


