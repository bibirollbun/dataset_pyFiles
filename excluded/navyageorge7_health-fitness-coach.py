"""
simple_health_coach.py

A compact, single-file AI Health & Fitness Coach prototype (simulated).
Designed in the same spirit as your travel agent example:
- ToolManager simulates external tools/APIs (wearable data, nutrition DB, workout DB).
- Multiple agents (DataCollector, NutritionAgent, WorkoutAgent, ProgressAgent).
- Orchestrator (CoachAgent) composes personalized plans and stores memories.
- InMemorySessionService (short-term) + MemoryBank (persistent long-term).

This is self-contained (no external APIs required). Run with:
    python simple_health_coach.py

Feel free to adapt the ToolManager to call real APIs (Fitbit/Apple Health/Nutrition APIs).
"""

import json
import os
import datetime
import random
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


# ---------------------------
# Sessions & Memory
# ---------------------------

class InMemorySessionService:
    """Short-term session store for active coaching sessions."""
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str, user_id: str, request: dict):
        self.sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "request": request,
            "draft_plan": None,
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        return self.sessions[session_id]

    def update_session(self, session_id: str, key: str, value: Any):
        if session_id in self.sessions:
            self.sessions[session_id][key] = value

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)


class MemoryBank:
    """Tiny long-term memory persisted to disk as JSON."""
    def __init__(self, filename="health_memory.json"):
        self.filename = filename
        self.data = {"users": {}}
        self._load()

    def _load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"users": {}}

    def _save(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_user_profile(self, user_id: str) -> dict:
        return self.data["users"].get(user_id, {})

    def update_user_profile(self, user_id: str, profile: dict):
        self.data["users"][user_id] = profile
        self._save()

    def append_progress(self, user_id: str, progress_entry: dict):
        profile = self.get_user_profile(user_id)
        progress = profile.get("progress", [])
        progress.append(progress_entry)
        profile["progress"] = progress
        self.update_user_profile(user_id, profile)


# ---------------------------
# Tool Manager (simulated external services)
# ---------------------------

class ToolManager:
    """
    Simulates external tools:
    - wearable_data: returns recent steps, sleep, heart rate
    - nutrition_db: gives calorie/macros for food items
    - workout_db: suggests workouts by difficulty and time
    - recipe_tool: gives a simple recipe suggestion
    """
    def wearable_data(self, user_id: str, days: int = 7) -> dict:
        # Random-ish simulation of past week data
        today = datetime.date.today()
        records = []
        for i in range(days):
            d = (today - datetime.timedelta(days=i)).isoformat()
            records.append({
                "date": d,
                "steps": random.randint(2000, 12000),
                "sleep_hrs": round(random.uniform(5.0, 8.5), 1),
                "resting_hr": random.randint(55, 80)
            })
        print(f"-> TOOL: Retrieved wearable data for {user_id} ({days} days)")
        return {"user_id": user_id, "records": records}

    def nutrition_db(self, food_query: str) -> dict:
        # Mock nutrition lookup
        sample_db = {
            "oatmeal": {"calories": 150, "protein_g": 5, "fat_g": 3, "carb_g": 27},
            "chicken breast": {"calories": 165, "protein_g": 31, "fat_g": 3.6, "carb_g": 0},
            "banana": {"calories": 105, "protein_g": 1.3, "fat_g": 0.4, "carb_g": 27},
            "salmon": {"calories": 208, "protein_g": 20, "fat_g": 13, "carb_g": 0}
        }
        print(f"-> TOOL: Nutrition lookup for '{food_query}'")
        return sample_db.get(food_query.lower(), {"calories": 200, "protein_g": 8, "fat_g": 8, "carb_g": 25})

    def workout_db(self, goal: str, duration_min: int, level: str) -> List[dict]:
        # Return 2-3 sample workouts
        print(f"-> TOOL: Query workouts for goal='{goal}', duration={duration_min}min, level='{level}'")
        base = []
        if goal == "fat_loss":
            base = [
                {"name": "HIIT Circuit", "duration_min": duration_min, "intensity": "high", "description": "Tabata cycling of sprints & bodyweight moves."},
                {"name": "Interval Run", "duration_min": duration_min, "intensity": "medium", "description": "Warmup + intervals 1:2 work/rest."}
            ]
        elif goal == "muscle_gain":
            base = [
                {"name": "Upper Body Strength", "duration_min": duration_min, "intensity": "medium", "description": "Push/pull compound lifts."},
                {"name": "Lower Body Strength", "duration_min": duration_min, "intensity": "medium", "description": "Squats, lunges, deadlifts."}
            ]
        else:
            base = [
                {"name": "Full Body Mix", "duration_min": duration_min, "intensity": "low", "description": "Mobility + circuit."}
            ]
        return base

    def recipe_tool(self, target_calories: int, protein_g: int) -> dict:
        # Mock a simple recipe
        print(f"-> TOOL: Generating recipe for ~{target_calories} kcal, {protein_g}g protein")
        return {
            "title": "Grilled Salmon Bowl",
            "calories": target_calories,
            "protein_g": protein_g,
            "ingredients": ["salmon fillet", "brown rice", "mixed greens", "lemon", "olive oil"],
            "instructions": "Grill salmon; serve over rice and greens; dress with lemon & oil."
        }


# ---------------------------
# Agents
# ---------------------------

@dataclass
class UserRequest:
    user_id: str
    age: int
    weight_kg: float
    height_cm: float
    goal: str  # e.g., "fat_loss", "muscle_gain", "wellness"
    activity_level: str  # "sedentary", "light", "active"
    dietary_pref: str  # "omnivore", "vegetarian", "pescatarian"
    days: int  # plan length in days


class DataCollectorAgent:
    """Collects recent wearable and profile data using ToolManager."""
    def __init__(self, tools: ToolManager):
        self.tools = tools

    def collect(self, user_req: UserRequest) -> dict:
        wearable = self.tools.wearable_data(user_req.user_id, days=7)
        profile = {
            "age": user_req.age,
            "weight_kg": user_req.weight_kg,
            "height_cm": user_req.height_cm,
            "bmi": round(user_req.weight_kg / ((user_req.height_cm / 100) ** 2), 1),
            "goal": user_req.goal,
            "activity_level": user_req.activity_level,
            "dietary_pref": user_req.dietary_pref
        }
        print("-> AGENT: Collected profile & wearable data")
        return {"profile": profile, "wearable": wearable}


class NutritionAgent:
    """Generates daily meal suggestions and caloric targets."""
    def __init__(self, tools: ToolManager):
        self.tools = tools

    def estimate_tdee(self, profile: dict) -> int:
        # Very simple TDEE estimation (Mifflin-St Jeor simplified)
        weight = profile["weight_kg"]
        height = profile["height_cm"]
        age = profile["age"]
        # assume male for simplicity (you can extend)
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
        activity_factor = {"sedentary": 1.2, "light": 1.375, "active": 1.55}.get(profile["activity_level"], 1.375)
        tdee = int(bmr * activity_factor)
        print(f"-> AGENT: Estimated TDEE = {tdee} kcal/day")
        return tdee

    def plan_meals(self, profile: dict, days: int) -> List[dict]:
        tdee = self.estimate_tdee(profile)
        # adjust for goals
        if profile["goal"] == "fat_loss":
            daily_cal = int(tdee * 0.8)
        elif profile["goal"] == "muscle_gain":
            daily_cal = int(tdee * 1.1)
        else:
            daily_cal = tdee
        protein_target = int(profile["weight_kg"] * (2.0 if profile["goal"] == "muscle_gain" else 1.4))
        plans = []
        for d in range(days):
            # Use recipe_tool for dinner and simple items for other meals
            recipe = self.tools.recipe_tool(target_calories=int(daily_cal * 0.4), protein_g=int(protein_target * 0.5))
            breakfast = {"name": "Oatmeal with banana", **self.tools.nutrition_db("oatmeal")}
            lunch = {"name": "Grilled chicken salad", **self.tools.nutrition_db("chicken breast")}
            dinner = {"name": recipe["title"], "calories": recipe["calories"], "protein_g": recipe["protein_g"]}
            plans.append({
                "day": d + 1,
                "date": (datetime.date.today() + datetime.timedelta(days=d)).isoformat(),
                "daily_calories": daily_cal,
                "protein_target_g": protein_target,
                "meals": {"breakfast": breakfast, "lunch": lunch, "dinner": dinner}
            })
        print("-> AGENT: Generated meal plans")
        return plans


class WorkoutAgent:
    """Creates workout suggestions aligned with user's goal and constraints."""
    def __init__(self, tools: ToolManager):
        self.tools = tools

    def create_workouts(self, profile: dict, days: int) -> List[dict]:
        workouts = []
        # choose duration based on activity_level
        dur_map = {"sedentary": 25, "light": 40, "active": 60}
        duration = dur_map.get(profile["activity_level"], 35)
        level = "beginner" if profile["activity_level"] == "sedentary" else "intermediate"
        for d in range(days):
            # alternate cardio & strength for balance
            goal = "fat_loss" if profile["goal"] == "fat_loss" and d % 2 == 0 else "muscle_gain" if profile["goal"] == "muscle_gain" and d % 2 == 1 else "wellness"
            suggestions = self.tools.workout_db(goal, duration, level)
            chosen = random.choice(suggestions)
            workouts.append({
                "day": d + 1,
                "date": (datetime.date.today() + datetime.timedelta(days=d)).isoformat(),
                "workout": chosen
            })
        print("-> AGENT: Crafted workout schedule")
        return workouts


class ProgressAgent:
    """Evaluates recent wearable data vs plan and produces a short progress note."""
    def evaluate(self, wearable: dict, profile: dict) -> dict:
        records = wearable.get("records", [])
        avg_steps = sum(r["steps"] for r in records) / len(records) if records else 0
        avg_sleep = sum(r["sleep_hrs"] for r in records) / len(records) if records else 0
        note = "Keep going!"
        if avg_steps < 5000:
            note = "Try to increase daily movement — add short walks."
        elif avg_steps > 8000:
            note = "Great activity levels — maintain consistency."
        if avg_sleep < 6.0:
            note += " Aim for better sleep hygiene."
        print("-> AGENT: Progress evaluation complete")
        return {"avg_steps": int(avg_steps), "avg_sleep": round(avg_sleep, 1), "note": note}


# ---------------------------
# Orchestrator (Coach)
# ---------------------------

class CoachAgent:
    """
    The orchestrator that calls agents, collects their outputs,
    synthesizes a friendly coaching plan, and updates memory.
    Demonstrates:
      - Multi-agent decomposition
      - Tool usage via ToolManager
      - Sessions & Memory
    """
    def __init__(self, tools: ToolManager, session_svc: InMemorySessionService, memory: MemoryBank):
        self.tools = tools
        self.session_svc = session_svc
        self.memory = memory
        self.data_collector = DataCollectorAgent(tools)
        self.nutrition_agent = NutritionAgent(tools)
        self.workout_agent = WorkoutAgent(tools)
        self.progress_agent = ProgressAgent()

    def create_plan(self, session_id: str) -> dict:
        session = self.session_svc.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
        req = session["request"]
        user_req = UserRequest(**req)

        # 1) Collect data (wearables + profile)
        collected = self.data_collector.collect(user_req)
        profile = collected["profile"]
        wearable = collected["wearable"]

        # 2) Nutrition plan
        meal_plans = self.nutrition_agent.plan_meals(profile, user_req.days)

        # 3) Workout plan
        workout_plans = self.workout_agent.create_workouts(profile, user_req.days)

        # 4) Progress evaluation
        progress = self.progress_agent.evaluate(wearable, profile)

        # Compose final plan
        plan = {
            "user_id": user_req.user_id,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "profile_summary": profile,
            "meal_plans": meal_plans,
            "workout_plans": workout_plans,
            "progress_summary": progress,
            "notes": self._compose_notes(profile, progress)
        }

        # Save draft plan in session
        self.session_svc.update_session(session_id, "draft_plan", plan)

        # Persist a short memory entry
        mem_entry = {
            "timestamp": plan["created_at"],
            "goal": profile["goal"],
            "weight_kg": profile["weight_kg"],
            "bmi": profile["bmi"]
        }
        self.memory.append_progress(user_req.user_id, mem_entry)

        return plan

    def _compose_notes(self, profile: dict, progress: dict) -> str:
        notes = []
        notes.append(f"Goal: {profile['goal']}. BMI: {profile['bmi']}.")
        notes.append(progress["note"])
        if profile["goal"] == "fat_loss":
            notes.append("Calorie target set moderately below estimated TDEE.")
        elif profile["goal"] == "muscle_gain":
            notes.append("Protein target dialed up to support muscle synthesis.")
        return " ".join(notes)


# ---------------------------
# Demo / Main
# ---------------------------

def pretty_print_plan(plan: dict):
    print("\n" + "="*70)
    print("PERSONALIZED HEALTH & FITNESS PLAN")
    print("="*70)
    print(f"User: {plan['user_id']} — Created at {plan['created_at']}")
    ps = plan["profile_summary"]
    print(f"Age {ps['age']}, Weight {ps['weight_kg']}kg, Height {ps['height_cm']}cm, BMI {ps['bmi']}")
    print("-"*70)
    print("Progress Summary:")
    p = plan["progress_summary"]
    print(f"Avg steps (last 7 days): {p['avg_steps']}, Avg sleep: {p['avg_sleep']} hrs")
    print("Notes:", p["note"])
    print("-"*70)
    print("Sample Day Plans (first 3 days):")
    for day in plan["meal_plans"][:3]:
        print(f"Day {day['day']} ({day['date']}): {day['daily_calories']} kcal | Protein target: {day['protein_target_g']}g")
        print("  Breakfast:", day["meals"]["breakfast"]["name"])
        print("  Lunch:", day["meals"]["lunch"]["name"])
        print("  Dinner:", day["meals"]["dinner"]["name"])
    print("-"*70)
    print("Sample Workouts (first 3 days):")
    for w in plan["workout_plans"][:3]:
        print(f"Day {w['day']} ({w['date']}): {w['workout']['name']} — {w['workout']['description']}")
    print("-"*70)
    print("General Notes:", plan["notes"])
    print("="*70 + "\n")


if __name__ == "__main__":
    # Setup services
    session_svc = InMemorySessionService()
    memory = MemoryBank()
    tools = ToolManager()

    # Instantiate coach
    coach = CoachAgent(tools, session_svc, memory)

    # Create a user request (you can modify these values)
    user_req = {
        "user_id": "user_abc",
        "age": 28,
        "weight_kg": 78.0,
        "height_cm": 175.0,
        "goal": "fat_loss",          # options: fat_loss, muscle_gain, wellness
        "activity_level": "light",   # options: sedentary, light, active
        "dietary_pref": "omnivore",  # e.g., vegetarian, pescatarian
        "days": 7
    }

    # Create session
    sess_id = "sess_1001"
    session_svc.create_session(sess_id, user_req["user_id"], user_req)

    # Generate plan
    plan = coach.create_plan(sess_id)

    # Pretty-print the plan
    pretty_print_plan(plan)

    # Show saved memory snippet
    print("Saved memory snippet for user:")
    print(json.dumps(memory.get_user_profile(user_req["user_id"]), indent=2))


