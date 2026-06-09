# # LifePilot — AI Concierge Meal Planner Agent (Kaggle Notebook)
# 
# **Notes**
# - This notebook runs in **mock mode** by default (no API keys required).
# - Designed for reproducible evaluation: deterministic outputs in mock mode.
# - Toggle real LLM usage by setting environment variable `USE_MOCKS=false` and providing your own LLM client (instructions in a later cell).
# 
# Sections:
# 1. Setup & Configuration
# 2. Utilities & Simple Memory
# 3. Mock LLM + Tool Wrappers
# 4. Agent Implementations (Planner, Nutrition, Shopping, Logging)
# 5. Orchestrator (sequential + parallel)
# 6. Evaluation & Tests
# 7. Demo runs & Final Output



# basic imports and logging
import os
import json
import time
import logging
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configuration flags (safe defaults)
USE_MOCKS = os.getenv("USE_MOCKS", "true").lower() in ("1", "true", "yes")
# If you want to test with a real LLM, set USE_MOCKS=false in your Kaggle runtime environment,
# then implement the RealLLMClient below and provide keys using Kaggle Secrets (do NOT commit keys).
logging.info(f"USE_MOCKS={USE_MOCKS}")



class InMemorySession:
    def __init__(self):
        self._store = {}
        self._history = []

    def set(self, key: str, value: Any):
        self._store[key] = value
        self._history.append({"time": time.time(), "action": "set", "key": key})

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    def dump(self):
        return dict(self._store)

    def history(self):
        return list(self._history)

def pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))



# Mock LLM client -- deterministic canned responses for specific prompts
class MockLLMClient:
    def generate(self, prompt: str, max_tokens=512, temperature=0.0) -> Dict[str, Any]:
        # Use keywords in prompt to decide which canned response to return.
        p = prompt.lower()
        if "create a 7-day meal plan" in p or "plan my week" in p:
            # deterministic simple 7-day plan (short versions)
            plan = {
                "Monday": {"breakfast": "Oats with banana", "lunch": "Quinoa salad", "dinner": "Tofu stir-fry"},
                "Tuesday": {"breakfast": "Greek yogurt + berries", "lunch": "Lentil soup", "dinner": "Veggie pasta"},
                "Wednesday": {"breakfast": "Smoothie bowl", "lunch": "Chickpea salad", "dinner": "Paneer curry"},
                "Thursday": {"breakfast": "Avocado toast", "lunch": "Rice & dal", "dinner": "Stir-fried veggies + rice"},
                "Friday": {"breakfast": "Egg scramble", "lunch": "Falafel wrap", "dinner": "Veggie pizza"},
                "Saturday": {"breakfast": "Pancakes + fruit", "lunch": "Grain bowl", "dinner": "Veggie tacos"},
                "Sunday": {"breakfast": "French toast", "lunch": "Noodle salad", "dinner": "Mushroom risotto"}
            }
            return {"content": json.dumps(plan)}
        elif "analyze nutrition" in p or "calculate macros" in p:
            # simple pseudo-calculation based on a standard vegetarian higher-protein plan
            nutrition = {
                "average_daily_calories": 2100,
                "average_daily_protein_g": 95,
                "average_daily_carbs_g": 260,
                "average_daily_fat_g": 60,
                "notes": "Estimates based on portion heuristics; replace with real nutrition API for production."
            }
            return {"content": json.dumps(nutrition)}
        elif "consolidate shopping list" in p or "shopping list" in p:
            shopping = [
                {"item": "Oats", "quantity": "500g"},
                {"item": "Bananas", "quantity": "7"},
                {"item": "Quinoa", "quantity": "400g"},
                {"item": "Tofu", "quantity": "600g"},
                {"item": "Mixed vegetables", "quantity": "2 kg"},
                {"item": "Greek yogurt", "quantity": "500g"},
                {"item": "Lentils", "quantity": "500g"},
                {"item": "Paneer", "quantity": "400g"},
                {"item": "Rice", "quantity": "2 kg"},
                {"item": "Pasta", "quantity": "500g"}
            ]
            return {"content": json.dumps(shopping)}
        else:
            # generic fallback
            return {"content": json.dumps({"text": "mock response for: " + (p[:80] + "...")})}

# Placeholder for a real LLM client wrapper (NOT implemented here)
class RealLLMClient:
    def __init__(self, api_key: str):
        raise NotImplementedError("Real LLM client is not implemented in this public notebook. Use USE_MOCKS=true.")

def get_llm_client():
    if USE_MOCKS:
        return MockLLMClient()
    else:
        # In a private environment you would create and return a real LLM client here,
        # e.g. using Google generativeai or another provider configured via secrets.
        raise RuntimeError("Real LLM client setup is not provided in the public notebook. Use USE_MOCKS=true.")



class PlannerAgent:
    "Generates a 7-day meal plan based on user preferences."
    def __init__(self, llm_client):
        self.llm = llm_client

    def plan_week(self, user_pref: str) -> Dict[str, Dict[str, str]]:
        prompt = f"Create a 7-day meal plan. Preferences: {user_pref}. Return JSON object of days->meals."
        resp = self.llm.generate(prompt)
        # parse deterministic response
        content = resp.get("content", "{}")
        try:
            plan = json.loads(content)
            assert isinstance(plan, dict)
            return plan
        except Exception as e:
            logging.exception("PlannerAgent failed to parse LLM output; returning empty plan.")
            return {}

class NutritionAgent:
    "Analyzes a meal plan and returns estimated nutrition metrics."
    def __init__(self, llm_client):
        self.llm = llm_client

    def analyze(self, meal_plan: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Analyze nutrition and calculate macros for this meal plan: {json.dumps(meal_plan)}. Return JSON."
        resp = self.llm.generate(prompt)
        content = resp.get("content", "{}")
        try:
            nutrition = json.loads(content)
            assert isinstance(nutrition, dict)
            return nutrition
        except Exception:
            logging.exception("NutritionAgent parsing failed; returning fallback nutrition.")
            return {
                "average_daily_calories": 2000,
                "average_daily_protein_g": 80,
                "average_daily_carbs_g": 250,
                "average_daily_fat_g": 60
            }

class ShoppingAgent:
    "Consolidates all recipes into a shopping list."
    def __init__(self, llm_client):
        self.llm = llm_client

    def build_list(self, meal_plan: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt = f"Consolidate ingredients and create a shopping list from: {json.dumps(meal_plan)}. Return JSON array."
        resp = self.llm.generate(prompt)
        content = resp.get("content", "[]")
        try:
            shopping = json.loads(content)
            assert isinstance(shopping, list)
            return shopping
        except Exception:
            logging.exception("ShoppingAgent parsing failed; returning empty list.")
            return []

class LoggingAgent:
    "Records events and basic metrics."
    def __init__(self, session: InMemorySession):
        self.session = session
        self.logs = []

    def log(self, message: str, meta: Dict[str, Any] = None):
        entry = {"time": time.time(), "message": message, "meta": meta or {}}
        self.logs.append(entry)
        logging.info(f"[LoggingAgent] {message} | meta: {meta}")
        # also store a short trace in session for evaluation
        hist = self.session.get("_logs", [])
        hist.append(entry)
        self.session.set("_logs", hist)

    def get_logs(self):
        return list(self.logs)



class Orchestrator:
    def __init__(self, llm_client):
        self.session = InMemorySession()
        self.llm = llm_client
        self.planner = PlannerAgent(self.llm)
        self.nutrition = NutritionAgent(self.llm)
        self.shopping = ShoppingAgent(self.llm)
        self.logger = LoggingAgent(self.session)

    def run(self, user_pref: str = "vegetarian, high-protein"):
        # Start
        self.logger.log("Orchestrator start", {"user_pref": user_pref})
        self.session.set("status", "starting")

        # 1. Planner (sequential)
        self.logger.log("Calling PlannerAgent")
        meal_plan = self.planner.plan_week(user_pref)
        self.session.set("meal_plan", meal_plan)
        if not meal_plan:
            self.logger.log("Planner returned empty plan", {})
            return {"error": "Planner failed to produce plan"}

        # 2. Trigger Nutrition and Shopping in parallel
        self.logger.log("Triggering NutritionAgent and ShoppingAgent in parallel")
        results = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {
                ex.submit(self.nutrition.analyze, meal_plan): "nutrition",
                ex.submit(self.shopping.build_list, meal_plan): "shopping"
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    res = fut.result()
                    self.session.set(name, res)
                    results[name] = res
                    self.logger.log(f"{name} completed", {"size": len(res) if hasattr(res, "__len__") else None})
                except Exception as e:
                    logging.exception(f"{name} failed")
                    results[name] = None

        # 3. Finalize
        final = {
            "meal_plan": meal_plan,
            "nutrition": results.get("nutrition"),
            "shopping_list": results.get("shopping")
        }
        self.session.set("final_output", final)
        self.session.set("status", "completed")
        self.logger.log("Orchestrator completed", {"final_keys": list(final.keys())})
        return final


def run_basic_tests():
    llm = get_llm_client()
    orch = Orchestrator(llm)
    result = orch.run("vegetarian, high-protein")

    assert isinstance(result, dict), "Result should be a dict"
    assert "meal_plan" in result, "meal_plan required"
    assert "nutrition" in result, "nutrition required"
    assert "shopping_list" in result, "shopping_list required"

    # meal_plan should have 7 days in mock output
    meal_plan = result["meal_plan"]
    assert isinstance(meal_plan, dict), "meal_plan must be a dict"
    assert len(meal_plan.keys()) >= 7, "meal_plan should contain 7 or more days"

    # nutrition must contain expected keys in mock
    nutrition = result["nutrition"]
    assert isinstance(nutrition, dict), "nutrition must be a dict"
    for k in ("average_daily_calories", "average_daily_protein_g"):
        assert k in nutrition, f"nutrition missing {k}"

    # shopping_list should be a list with items
    shopping = result["shopping_list"]
    assert isinstance(shopping, list), "shopping_list must be a list"
    assert len(shopping) > 0, "shopping_list should not be empty"

    print("All basic tests passed.")
    return result

# Run tests when the notebook is executed top-to-bottom (safe, deterministic)
result_for_tests = run_basic_tests()


print("\n=== FINAL OUTPUT ===\n")
pretty(result_for_tests)

print("\n=== SESSION DUMP ===\n")
session_dump = Orchestrator(get_llm_client()).session.dump()  # new empty session demo
pretty(session_dump)



!zip -r all_output_files.zip .

