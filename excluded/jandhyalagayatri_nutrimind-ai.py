!pip install -q langchain langgraph google-generativeai python-dotenv pandas pydantic requests



import os
import json
import time
import uuid
import base64
import re
import datetime
import pathlib
import traceback
from PIL import Image
import pandas as pd

from typing import TypedDict, Dict, Any, Optional

import requests
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Load secrets from Kaggle UI
user_secrets = UserSecretsClient()

GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
EDAMAM_APP_ID = user_secrets.get_secret("EDAMAM_APP_ID")
EDAMAM_APP_KEY = user_secrets.get_secret("EDAMAM_APP_KEY")

print("Gemini key set:", bool(GEMINI_API_KEY))
print("Edamam id set:", bool(EDAMAM_APP_ID))
print("Edamam key set:", bool(EDAMAM_APP_KEY))

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Folders for logs, runs, memory (Kaggle-safe)
pathlib.Path("logs").mkdir(exist_ok=True)
pathlib.Path("runs").mkdir(exist_ok=True)
pathlib.Path("memory").mkdir(exist_ok=True)



def log(node: str, message: str, level: str = "INFO"):
    ts = datetime.datetime.utcnow().isoformat()
    line = f"{ts} [{level}] {node}: {message}\n"
    with open("logs/pipeline.log", "a") as f:
        f.write(line)
    print(line.strip())


def save_run_state(run_id: str, state: Dict[str, Any]):
    with open(f"runs/{run_id}.json", "w") as f:
        json.dump(state, f, indent=2)
    log("orchestrator", f"Saved run state {run_id}")


def load_run_state(run_id: str) -> Optional[Dict[str, Any]]:
    p = f"runs/{run_id}.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_memory(name: str, data: Any):
    with open(f"memory/{name}.json", "w") as f:
        json.dump(data, f, indent=2)
    log("memory", f"Saved {name}")


def load_memory(name: str, default=None):
    p = f"memory/{name}.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return default if default is not None else {}



def log(node: str, message: str, level: str = "INFO"):
    ts = datetime.datetime.utcnow().isoformat()
    line = f"{ts} [{level}] {node}: {message}\n"
    with open("logs/pipeline.log", "a") as f:
        f.write(line)
    print(line.strip())


def save_run_state(run_id: str, state: Dict[str, Any]):
    with open(f"runs/{run_id}.json", "w") as f:
        json.dump(state, f, indent=2)
    log("orchestrator", f"Saved run state {run_id}")


def load_run_state(run_id: str) -> Optional[Dict[str, Any]]:
    p = f"runs/{run_id}.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_memory(name: str, data: Any):
    with open(f"memory/{name}.json", "w") as f:
        json.dump(data, f, indent=2)
    log("memory", f"Saved {name}")


def load_memory(name: str, default=None):
    p = f"memory/{name}.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return default if default is not None else {}



from typing import Optional
from langgraph.graph import StateGraph, END

class AgentState(TypedDict, total=False):
    run_id: str
    image_path: str
    user_profile: Dict[str, Any]   # optional later (goals, conditions, etc.)
    agent1_result: Dict[str, Any]
    agent2_result: Dict[str, Any]
    agent3_result: Dict[str, Any]



def load_image_for_gemini(image_path: str) -> Dict[str, Any]:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(img_bytes).decode()
    }


def agent1_image_analyzer(state: AgentState) -> AgentState:
    node = "Agent1_ImageAnalyzer"
    image_path = state["image_path"]

    try:
        image_data = load_image_for_gemini(image_path)

        prompt = """
You are a STRICT food image analyzer.

Your job:
1. Identify visible food items from ANY cuisine.
2. Food names must be short and generic (rice, curry, roti, dosa, fries, salad, pizza slice, etc.)
3. If you don't know exact dish, use generic labels like "vegetable curry", "fried snack", "gravy".
4. Allowed portion labels: small, medium, large, one piece, two pieces.
5. Estimate realistic weight in grams: "80g", "150g", etc.
6. Return STRICT JSON only in this format:

{
  "food_name": "short meal description",
  "items": [
    {"name": "rice", "portion": "medium", "weight": "180g"},
    {"name": "vegetable curry", "portion": "small", "weight": "80g"}
  ],
  "confidence": 0.95
}

No extra text, no explanation.
"""

        model = genai.GenerativeModel("models/gemini-2.5-flash")

        resp = model.generate_content(
            [prompt, image_data],
            generation_config={"response_mime_type": "application/json"}
        )

        text = resp.text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)

        log(node, f"Detected {len(data.get('items', []))} items")
        return {**state, "agent1_result": data}

    except Exception as e:
        log(node, f"ERROR: {repr(e)}", level="ERROR")
        traceback.print_exc()
        fallback = {"food_name": "unknown meal", "items": [], "confidence": 0.0}
        return {**state, "agent1_result": fallback}



EDAMAM_URL = "https://api.edamam.com/api/nutrition-data"

def build_edamam_query(item: Dict[str, Any]) -> str:
    weight = item.get("weight", "")
    # normalize "150g" -> "150 g"
    if isinstance(weight, str) and weight.endswith("g") and " " not in weight:
        weight = weight.replace("g", " g")
    return f"{weight} {item['name']}".strip()


def get_calories_edamam(query: str) -> float:
    node = "Agent2_Edamam"
    params = {
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY,
        "ingr": query
    }
    try:
        r = requests.get(EDAMAM_URL, params=params, timeout=10)
        log(node, f"Status {r.status_code} URL {r.url}")
        data = r.json()
    except Exception as e:
        log(node, f"HTTP error: {e}", level="ERROR")
        return 0.0

    # Try direct calories field, else parsed nutrients
    try:
        cal = data.get("calories", 0)
        if not cal:
            cal = data["ingredients"][0]["parsed"][0]["nutrients"]["ENERC_KCAL"]["quantity"]
        return float(cal)
    except Exception:
        return 0.0


def estimate_calories_gemini(item_name: str, weight: str) -> float:
    node = "Agent2b_GeminiFallback"
    try:
        prompt = f"""
Estimate calories (kcal) for the given food. Return ONLY a number without units.

Food: {item_name}
Weight: {weight}

Example of correct output: 230
"""

        model = genai.GenerativeModel("models/gemini-2.5-flash")
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        nums = re.findall(r"\d+\.?\d*", text)
        if not nums:
            log(node, f"No numeric value in output: {text}", level="WARN")
            return 0.0
        val = float(nums[0])
        log(node, f"Gemini fallback -> {item_name} {weight}: {val} kcal")
        return val
    except Exception as e:
        log(node, f"ERROR: {e}", level="ERROR")
        return 0.0


def agent2_calorie_calculator(state: AgentState) -> AgentState:
    node = "Agent2_CalorieCalculator"
    agent1 = state["agent1_result"]
    items = agent1.get("items", [])

    enriched_items = []
    total_cal = 0.0

    for item in items:
        query = build_edamam_query(item)
        cal = get_calories_edamam(query)
        if cal == 0.0:
            log(node, f"Edamam returned 0 for '{query}', using Gemini fallback")
            cal = estimate_calories_gemini(item["name"], item.get("weight", ""))

        item_with_cal = {**item, "calories": cal}
        enriched_items.append(item_with_cal)
        total_cal += cal
        log(node, f"{item['name']} ({item.get('weight')}): {cal} kcal")

    result = {
        "meal": agent1.get("food_name", "unknown meal"),
        "confidence": agent1.get("confidence", 0.0),
        "items": enriched_items,
        "total_calories": total_cal,
        "user_profile": state.get("user_profile", {})
    }

    return {**state, "agent2_result": result}



def agent3_dietician(state: AgentState) -> AgentState:
    node = "Agent3_Dietician"
    meal_data = state["agent2_result"]

    try:
        prompt = f"""
You are a certified dietician AI.

Analyze this meal JSON and return STRICT JSON advice:

Meal JSON:
{json.dumps(meal_data, indent=2)}

Return ONLY JSON:

{{
  "meal": "string",
  "health_rating": "short description e.g. 'Moderate â€“ slightly high in refined carbs'",
  "issues": ["issue 1", "issue 2"],
  "improvements": ["tip 1", "tip 2"],
  "alternatives": ["alternative 1", "alternative 2"],
  "personalized_plan": "1 short paragraph personalized to user_profile (if present)"
}}

Do not add any explanation outside the JSON.
"""

        model = genai.GenerativeModel("models/gemini-2.5-flash")
        resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        text = resp.text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        advice = json.loads(text)
        log(node, "Dietician advice generated")
        return {**state, "agent3_result": advice}

    except Exception as e:
        log(node, f"ERROR: {e}", level="ERROR")
        traceback.print_exc()
        fallback = {
            "meal": meal_data.get("meal", ""),
            "health_rating": "Unavailable",
            "issues": [],
            "improvements": [],
            "alternatives": [],
            "personalized_plan": ""
        }
        return {**state, "agent3_result": fallback}



# Build a LangGraph StateGraph

graph = StateGraph(AgentState)

graph.add_node("agent1_image", agent1_image_analyzer)
graph.add_node("agent2_calories", agent2_calorie_calculator)
graph.add_node("agent3_dietician", agent3_dietician)

graph.set_entry_point("agent1_image")
graph.add_edge("agent1_image", "agent2_calories")
graph.add_edge("agent2_calories", "agent3_dietician")
graph.add_edge("agent3_dietician", END)

app = graph.compile()
log("graph", "LangGraph pipeline compiled")



def run_full_meal_pipeline(
    image_path: str,
    user_profile: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    save_history: bool = True
) -> AgentState:
    """
    High-level orchestrator around LangGraph app.
    Adds run_id, user_profile, memory logging, and state persistence.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    init_state: AgentState = {
        "run_id": run_id,
        "image_path": image_path,
    }
    if user_profile:
        init_state["user_profile"] = user_profile

    log("orchestrator", f"Starting run {run_id}")

    # Run the LangGraph pipeline
    final_state: AgentState = app.invoke(init_state)

    # Save full final state (for pause/resume/evaluation)
    save_run_state(run_id, dict(final_state))

    # Save light history for user (only meal info)
    if save_history:
        history = load_memory("history", default=[])
        history.append({
            "run_id": run_id,
            "timestamp": time.time(),
            "meal": final_state.get("agent2_result", {})
        })
        save_memory("history", history)

    log("orchestrator", f"Run {run_id} finished")
    return final_state


def demo_pause_resume():
    """
    Tiny demonstration of a 'long-running job' with pause/resume using saved state.
    Not tied to the main pipeline; just shows concept.
    """
    rid = str(uuid.uuid4())[:8]
    state = {"run_id": rid, "stage": "started", "progress": 0}
    save_run_state(rid, state)
    log("pause_demo", f"Job {rid} started")

    # simulate some progress
    time.sleep(0.5)
    state["stage"] = "halfway"
    state["progress"] = 50
    save_run_state(rid, state)
    log("pause_demo", f"Job {rid} halfway, saved")

    # "resume"
    resumed = load_run_state(rid)
    log("pause_demo", f"Resumed job {rid} at stage={resumed['stage']} progress={resumed['progress']}")

    # finish
    time.sleep(0.5)
    state["stage"] = "done"
    state["progress"] = 100
    save_run_state(rid, state)
    log("pause_demo", f"Job {rid} completed")

    return rid



IMAGE_PATH = "/kaggle/input/food-images/meals_img1.jpg"
img = Image.open(IMAGE_PATH)
img



# Example:
IMAGE_PATH = "/kaggle/input/food-images/meals_img1.jpg"

img = Image.open(IMAGE_PATH)


assert GEMINI_API_KEY, "GEMINI_API_KEY not set"
assert EDAMAM_APP_ID and EDAMAM_APP_KEY, "Edamam credentials not set"

user_profile = {
    "age": 25,
    "gender": "female",
    "goal": "weight_loss",
    "diet_type": "vegetarian",
    "conditions": ["pcos"]
}

final_state = run_full_meal_pipeline(
    IMAGE_PATH,
    user_profile=user_profile,
    save_history=True
)

# ---------------------------------------
# Agent 1 â€” Image Analysis (TABLE)
# ---------------------------------------
print("=== Agent 1 â€” Image Analysis ===")

agent1 = final_state["agent1_result"]
import pandas as pd

if "items" in agent1 and len(agent1["items"]) > 0:
    df_agent1 = pd.DataFrame(agent1["items"])
    display(df_agent1)
else:
    print("(No items detected)")

print(f"\nMeal: {agent1.get('food_name', '')}")
print(f"Confidence: {agent1.get('confidence', '')}")


# ---------------------------------------
# Agent 2 â€” Enhanced Table (Calories Added)
# ---------------------------------------
print("\n=== Agent 2 â€” Calories Added to Agent 1 Table ===")

agent2 = final_state["agent2_result"]

# Agent2 already contains the same items + calories column
df_agent2 = pd.DataFrame(agent2["items"])
display(df_agent2)

print(f"\nTotal Calories: {agent2['total_calories']}")


# ---------------------------------------
# Agent 3 â€” Dietician Advice (Markdown)
# ---------------------------------------
print("\n=== Agent 3 â€” Dietician Advice ===")

from IPython.display import Markdown
advice = final_state["agent3_result"]

md = f"""
### ğŸ¥— Dietician Review for **{advice['meal']}**

**Health Rating:**  
- {advice['health_rating']}

---

### âš  Issues Identified
""" + "\n".join([f"- {issue}" for issue in advice["issues"]]) + """

---

### âœ” Improvements Suggested
""" + "\n".join([f"- {impr}" for impr in advice["improvements"]]) + """

---

### ğŸ�½ Healthier Alternatives
""" + "\n".join([f"- {alt}" for alt in advice["alternatives"]]) + """

---

### ğŸ�¯ Personalized Plan
**"{advice['personalized_plan']}"**
"""

display(Markdown(md))



demo_run_id = demo_pause_resume()
print("Pause/resume demo run_id:", demo_run_id)





