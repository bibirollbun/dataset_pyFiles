!pip install aiohttp


from kaggle_secrets import UserSecretsClient
import os
client = UserSecretsClient()
key = client.get_secret("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = key
if key:
    print("✅ GEMINI_API_KEY imported successfully")
else:
    print("❌ GEMINI_API_KEY not found")


import os
import json
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
print("ENV check complete: Python imports successful")
key = os.getenv("GEMINI_API_KEY")
if key:
    print("GEMINI_API_KEY found")
else:
    print("GEMINI_API_KEY not found; running in MOCK_MODE")


class SessionMemory:
    def __init__(self, session_id=None, memory_file="memory_store.json"):
        self.session_id = session_id or str(uuid.uuid4())
        self.memory_file = memory_file
        self.state = {"session_id": self.session_id, "conversations": [], "facts": {}}
        try:
            with open(self.memory_file, "r") as f:
                data = json.load(f)
            if self.session_id in data:
                self.state = data[self.session_id]
        except Exception:
            pass
        print("SessionMemory initialized for session:", self.session_id)
    def write_memory(self, key, value):
        self.state["facts"][key] = value
        self._persist()
    def append_conversation(self, role, text):
        entry = {"role": role, "text": text, "ts": time.time()}
        self.state["conversations"].append(entry)
        self._persist()
    def _persist(self):
        try:
            all_data = {}
            try:
                with open(self.memory_file, "r") as f:
                    all_data = json.load(f)
            except Exception:
                all_data = {}
            all_data[self.session_id] = self.state
            with open(self.memory_file, "w") as f:
                json.dump(all_data, f, indent=2)
        except Exception:
            pass
    def read_memory(self, key=None):
        if key:
            return self.state["facts"].get(key)
        return self.state
session = SessionMemory()
print("Session created and ready")



def mock_search(query):
    results = [
        {"title": "India Top Attractions", "snippet": "Temple, Market, Lake", "url": "https://example.com/india-attractions"},
        {"title": "Indore Travel Tips", "snippet": "Food, Travel, Stay", "url": "https://example.com/indore-tips"}
    ]
    print("Tool: mock_search executed for query:", query)
    return results

def simple_itinerary(days, interests):
    itinerary = []
    for d in range(1, days+1):
        itinerary.append({"day": d, "plan": f"Visit {interests} and local market on day {d}"})
    print("Tool: simple_itinerary created for days:", days)
    return itinerary
print("Custom tools loaded")



class PlannerAgent:
    def __init__(self, memory):
        self.memory = memory
    def plan(self, user_goal):
        self.memory.append_conversation("planner", f"Planning for: {user_goal}")
        plan = {"goal": user_goal, "days": 2, "interests": "local culture"}
        print("PlannerAgent: plan created")
        return plan

class InfoAgent:
    def __init__(self, memory):
        self.memory = memory
    def fetch(self, item):
        res = mock_search(item)
        self.memory.append_conversation("info", f"Fetched info for: {item}")
        print("InfoAgent: fetch complete for:", item)
        return res

class ExecutorAgent:
    def __init__(self, memory):
        self.memory = memory
        self.paused = False
    def execute_itinerary(self, itinerary):
        results = []
        for step in itinerary:
            while self.paused:
                time.sleep(0.1)
            result = {"day": step["day"], "status": "ok", "action": step["plan"]}
            results.append(result)
            self.memory.append_conversation("executor", f"Executed day {step['day']}")
            time.sleep(0.2)
        print("ExecutorAgent: execution finished")
        return results
    def pause(self):
        self.paused = True
        print("ExecutorAgent: paused")
    def resume(self):
        self.paused = False
        print("ExecutorAgent: resumed")

class Orchestrator:
    def __init__(self, memory):
        self.memory = memory
        self.planner = PlannerAgent(memory)
        self.info = InfoAgent(memory)
        self.executor = ExecutorAgent(memory)
    def run(self, user_goal):
        plan = self.planner.plan(user_goal)
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(self.info.fetch, user_goal), ex.submit(self.info.fetch, f"{user_goal} food")]
            info_results = []
            for fut in as_completed(futures):
                info_results.extend(fut.result())
        itinerary = simple_itinerary(plan["days"], plan["interests"])
        exec_results = self.executor.execute_itinerary(itinerary)
        self.memory.write_memory("last_plan", plan)
        print("Orchestrator: run complete")
        return {"plan": plan, "info": info_results, "itinerary": itinerary, "execution": exec_results}
orch = Orchestrator(session)
print("Orchestrator initialized and agents ready")



user_goal = "short trip to India"
result = orch.run(user_goal)
print("Demo run complete. Summary:")
print("Plan:", result["plan"])
print("Number of info results fetched:", len(result["info"]))
print("Itinerary days:", len(result["itinerary"]))
print("Execution records:", len(result["execution"]))



orch.executor.pause()
print("Executor state after pause:", orch.executor.paused)
time.sleep(0.5)
orch.executor.resume()
print("Executor state after resume:", orch.executor.paused)



def simple_eval(exec_results):
    success = sum(1 for r in exec_results if r.get("status") == "ok")
    total = len(exec_results)
    score = int((success/total)*100) if total>0 else 0
    print("Evaluation completed with score:", score)
    return {"score": score, "success": success, "total": total}
eval_out = simple_eval(result["execution"])
print("Evaluation output stored")


