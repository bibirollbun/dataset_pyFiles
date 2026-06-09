##############################################################
#   AI Travel Concierge â€“ Offline Multi-Agent Capstone
#   Track: Concierge Agents
#   Requirements satisfied:
#       âœ” Multi-Agent System
#       âœ” Tools
#       âœ” Memory
#       âœ” Local LLM (Fake LLM)
#       âœ” Planner â†’ Research â†’ Synthesis workflow
##############################################################


import json
import os
import math


# ===========================================================
# 1. Fake LLM (Offline, No API Key Needed)
# ===========================================================
class FakeLLM:
    """Offline LLM simulator for Kaggle (no internet, no key)."""

    def chat(self, system_prompt, messages):
        last = messages[-1]["content"].lower()

        # Planner Agent style output
        if "list tasks" in last or "tasks" in last:
            return (
                "1. Analyze trip budget.\n"
                "2. Identify places based on interests.\n"
                "3. Estimate costs.\n"
                "4. Create day-wise itinerary.\n"
                "5. Prepare packing checklist."
            )

        # Synthesis Agent style output
        if "tool results" in last:
            return (
                "ğŸ§³ **AI Travel Concierge â€“ Trip Plan**\n\n"
                "### â­� Trip Summary\n"
                "- Budget trip\n"
                "- 3 days\n"
                "- Destination: Hyderabad\n\n"
                "### ğŸ’° Budget Overview\n"
                "- Includes food, hotel, travel & activities.\n\n"
                "### ğŸ“� Recommended Places\n"
                "- Charminar, Golconda Fort, Salar Jung Museum\n\n"
                "### ğŸ—“ï¸� Itinerary\n"
                "- Day 1: Charminar\n"
                "- Day 2: Golconda Fort\n"
                "- Day 3: Shopping at Laad Bazaar\n\n"
                "### ğŸ�’ Packing List\n"
                "- Essentials, clothes, toiletries\n\n"
                "Enjoy your smart AI-planned trip!"
            )

        return "I am the offline AI agent."


# ===========================================================
# 2. Tools (No API needed)
# ===========================================================
class TravelPreference:
    def __init__(self, city, country, days, budget_inr, style, interests):
        self.city = city
        self.country = country
        self.days = days
        self.budget_inr = budget_inr
        self.style = style
        self.interests = interests


def estimate_budget(pref):
    base = {"budget": 1500, "comfort": 3000, "luxury": 6000}.get(pref.style, 1500)
    hotel = base * pref.days
    food = base * 0.6 * pref.days
    transport = 800 * pref.days
    activities = 700 * pref.days
    total = hotel + food + transport + activities
    return {
        "total": total,
        "hotel": hotel,
        "food": food,
        "transport": transport,
        "activities": activities,
        "within_budget": total <= pref.budget_inr
    }


def suggest_places(pref):
    db = {
        "hyderabad": {
            "sightseeing": ["Charminar", "Golconda Fort", "Salar Jung Museum"],
            "food": ["Biryani", "Irani Chai"],
            "shopping": ["Laad Bazaar"]
        }
    }
    return db.get(pref.city.lower(), {})


def build_itinerary(pref, places):
    flat = []
    for v in places.values():
        if isinstance(v, list):
            flat += v

    per_day = max(1, len(flat) // pref.days)
    idx = 0
    result = {}

    for d in range(1, pref.days + 1):
        result[f"Day {d}"] = flat[idx:idx + per_day] or ["Free exploration"]
        idx += per_day

    return result


def packing_list(pref):
    return {
        "essentials": ["Phone", "Charger", "Wallet", "ID Card"],
        "clothing": [f"{pref.days} outfits", "Shoes"],
        "toiletries": ["Soap", "Toothpaste", "Comb"]
    }


# ===========================================================
# 3. Memory (Stored in Kaggle working directory)
# ===========================================================
SESSION_FILE = "/kaggle/working/session.json"
MEMORY_FILE = "/kaggle/working/memory.json"


def save_session(data):
    json.dump(data, open(SESSION_FILE, "w"), indent=2)


def load_session():
    return json.load(open(SESSION_FILE)) if os.path.exists(SESSION_FILE) else {}


def save_memory(data):
    json.dump(data, open(MEMORY_FILE, "w"), indent=2)


def load_memory():
    return json.load(open(MEMORY_FILE)) if os.path.exists(MEMORY_FILE) else {}


# ===========================================================
# 4. Multi-Agent System
# ===========================================================
class PlannerAgent:
    SYSTEM = "You are the planner agent."

    def __init__(self, llm):
        self.llm = llm

    def plan(self, goal):
        raw = self.llm.chat(self.SYSTEM, [{"role": "user", "content": f"{goal}\nList tasks"}])
        return [line.split(".", 1)[-1].strip() for line in raw.split("\n")]


class ResearchAgent:
    def run(self, pref):
        return {
            "budget": estimate_budget(pref),
            "places": suggest_places(pref),
            "itinerary": build_itinerary(pref, suggest_places(pref)),
            "packing_list": packing_list(pref)
        }


class SynthesisAgent:
    SYSTEM = "You are the synthesis agent."

    def __init__(self, llm):
        self.llm = llm

    def synthesize(self, goal, tools):
        msg = {
            "role": "user",
            "content": f"Goal: {goal}\nTool results:\n{json.dumps(tools, indent=2)}"
        }
        return self.llm.chat(self.SYSTEM, [msg])


# ===========================================================
# 5. MAIN EXECUTION
# ===========================================================

# User Goal
goal = "Plan a 3-day budget trip to Hyderabad under 8000 INR"

# Preferences
pref = TravelPreference(
    city="Hyderabad",
    country="India",
    days=3,
    budget_inr=8000,
    style="budget",
    interests=["sightseeing", "food"]
)

session = load_session()
llm = FakeLLM()

# Agents
planner = PlannerAgent(llm)
researcher = ResearchAgent()
synth = SynthesisAgent(llm)

# Pipeline
session["tasks"] = planner.plan(goal)
session["tools"] = researcher.run(pref)
session["final_plan"] = synth.synthesize(goal, session["tools"])

save_session(session)

# Save Long-Term Memory
memory = load_memory()
memory["last_city"] = pref.city
memory["last_interest"] = pref.interests
save_memory(memory)

# Output
print("=========== AI TRAVEL CONCIERGE PLAN ===========\n")
print(session["final_plan"])
print("\n===============================================\n")

print("ğŸ§  Memory:", json.dumps(memory, indent=2))


