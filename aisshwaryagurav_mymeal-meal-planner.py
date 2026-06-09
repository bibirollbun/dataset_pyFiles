# ============================================================
#   FULL MULTI-AGENT MEAL PLANNING SYSTEM — N Meal Plans
#   Supports:
#   ✔ User chooses N number of meal plans
#   ✔ Multi-agent (parallel + sequential + loop)
#   ✔ LLM-powered planner (mock)
#   ✔ Tools (recipe search, nutrition)
#   ✔ Session memory + long-term SQLite memory
#   ✔ Logging / Observability
# ============================================================

import os, time, json, sqlite3, logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------
# Logging
# ------------------------
logger = logging.getLogger("meal_agent")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# ============================================================
#   MOCK LLM (Replace with Gemini/OpenAI if available)
# ============================================================
class LLMClient:
    def generate(self, prompt, system_prompt=""):
        # Basic template output so notebook runs with no API keys
        return (
            "DAY 1:\n"
            "  Breakfast: Oatmeal + Banana (350 cal)\n"
            "  Lunch: Grilled Chicken Salad (450 cal)\n"
            "  Dinner: Salmon & Quinoa (600 cal)\n"
            "\n"
            "DAY 2:\n"
            "  Breakfast: Yogurt + Berries (300 cal)\n"
            "  Lunch: Veggie Wrap (500 cal)\n"
            "  Dinner: Tofu Stir Fry (550 cal)\n"
            "\n"
            "DAY 3:\n"
            "  Breakfast: Smoothie (320 cal)\n"
            "  Lunch: Turkey Sandwich (480 cal)\n"
            "  Dinner: Pasta Marinara (620 cal)\n"
        )

llm = LLMClient()


# ============================================================
#   SESSION MEMORY
# ============================================================
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}

    def get(self, sid):
        return self.sessions.setdefault(sid, {"history": []})

    def append(self, sid, role, message):
        self.get(sid)["history"].append({
            "role": role,
            "message": message,
            "ts": time.time()
        })

session_service = InMemorySessionService()


# ============================================================
#   LONG-TERM MEMORY (SQLite)
# ============================================================
DB_PATH = "/kaggle/working/meal_memory.sqlite"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT, ts REAL)")
conn.commit()

class MemoryBank:
    def store(self, key, value):
        conn.execute("REPLACE INTO memory VALUES (?, ?, ?)", (key, json.dumps(value), time.time()))
        conn.commit()

    def fetch(self, key):
        r = conn.execute("SELECT value FROM memory WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

memory_bank = MemoryBank()


# ============================================================
#   MCP TOOL REGISTRY
# ============================================================
class MCPClientMock:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, func):
        self.tools[name] = func

    def call(self, name, *args, **kwargs):
        return self.tools[name](*args, **kwargs)

mcp = MCPClientMock()

# --- Static recipes dataset ----
RECIPES = [
    {"title": "Oatmeal + Banana", "cal": 350},
    {"title": "Grilled Chicken Salad", "cal": 450},
    {"title": "Salmon & Quinoa", "cal": 600},
    {"title": "Yogurt + Berries", "cal": 300},
    {"title": "Veggie Wrap", "cal": 500},
    {"title": "Tofu Stir Fry", "cal": 550},
    {"title": "Smoothie", "cal": 320},
    {"title": "Turkey Sandwich", "cal": 480},
    {"title": "Pasta Marinara", "cal": 620},
]

# ---- Tool: Recipe Search ----
def recipe_search(keyword):
    result = [r for r in RECIPES if keyword.lower() in r["title"].lower()]
    return result if result else RECIPES[:3]

# ---- Tool: Total Nutrition ----
def nutrition_calc(recipes):
    return sum(r["cal"] for r in recipes)

mcp.register_tool("recipe_search", recipe_search)
mcp.register_tool("nutrition_calc", nutrition_calc)


# ============================================================
#   MULTI-AGENT SYSTEM
# ============================================================
class AgentBase:
    def __init__(self, name):
        self.name = name

    def log(self, msg):
        logger.info(f"[{self.name}] {msg}")


# ---------------------------
# PARALLEL RETRIEVER AGENT
# ---------------------------
class RetrieverAgent(AgentBase):
    def retrieve(self, query):
        self.log(f"Searching recipes for '{query}'")
        return mcp.call("recipe_search", query)


# ---------------------------
# LLM-POWERED PLANNER AGENT
# ---------------------------
class PlannerAgent(AgentBase):
    def create_plan(self, profile, recipes):
        prompt = f"Create a {profile['days']}-day meal plan using these recipes:\n{recipes}"
        session_service.append("meal", "planner_prompt", prompt)
        plan = llm.generate(prompt)
        session_service.append("meal", "planner_output", plan)
        return plan


# ---------------------------
# LOOP EXECUTOR AGENT
# ---------------------------
class ExecutorAgent(AgentBase):
    def evaluate(self, plan_text):
        used = []
        for r in RECIPES:
            if r["title"].lower() in plan_text.lower():
                used.append(r)

        total_cal = mcp.call("nutrition_calc", used)
        self.log(f"Plan Calories = {total_cal}")

        return {
            "recipes_used": used,
            "total_calories": total_cal,
            "ok": 1400 <= total_cal <= 2500
        }


# ============================================================
#   ORCHESTRATOR — N Meal Plans
# ============================================================
def generate_meal_plans(profile, queries, n):
    retrievers = [RetrieverAgent(f"retriever_{i}") for i in range(len(queries))]

    # ---- PARALLEL retrieval ----
    all_found = []
    with ThreadPoolExecutor(max_workers=len(retrievers)) as ex:
        futures = {ex.submit(r.retrieve, q): (r, q) for r, q in zip(retrievers, queries)}
        for f in as_completed(futures):
            all_found.extend(f.result())

    recipes = list({r["title"]: r for r in all_found}.values())

    planner = PlannerAgent("planner")
    executor = ExecutorAgent("executor")

    meal_plans = []

    for i in range(n):
        print(f"\n================ PLAN {i+1} ================\n")

        plan = planner.create_plan(profile, recipes)
        print(plan)

        evaluation = executor.evaluate(plan)
        print("\nCalories:", evaluation["total_calories"])
        print("OK (Calorie Range):", evaluation["ok"])

        meal_plans.append({"plan": plan, "eval": evaluation})

    return meal_plans


# ============================================================
#   RUN — CHOOSE N MEAL PLANS
# ============================================================

N = 5   

profile = {
    "days": 3,
    "goal": "healthy eating",
    "calorie_target": (1400, 2500)
}

meal_plans = generate_meal_plans(
    profile,
    queries=["breakfast", "lunch", "dinner"],
    n=N
)

print("\nDONE! Generated", len(meal_plans), "meal plans.")


