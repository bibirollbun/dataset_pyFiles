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



from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random
import json



@dataclass
class InMemorySessionState:
    user_id: str = "default_user"
    diet: Optional[str] = None                 # e.g., "vegetarian", "vegan", "non-veg"
    calorie_goal: Optional[int] = None         # daily calorie goal
    preferences: List[str] = field(default_factory=list)  # cuisine likes/dislikes
    meal_plan: Dict[str, Dict[str, str]] = field(default_factory=dict)  # day -> {breakfast:..., lunch:..., dinner:...}
    messages: List[Dict] = field(default_factory=list)   # conversation history

    def add_message(self, role: str, content: str):
        self.messages.append({
            "time": datetime.now().isoformat(),
            "role": role,
            "content": content
        })

    def set_diet(self, diet: str):
        self.diet = diet

    def set_calorie_goal(self, calories: int):
        self.calorie_goal = calories

    def save_day_plan(self, day: str, day_plan: Dict[str, str]):
        self.meal_plan[day.lower()] = day_plan

    def get_plan(self) -> Dict[str, Dict[str, str]]:
        return self.meal_plan


class CalorieTool:
    """
    A simplistic calorie lookup/calculation tool. In a real submission this could call a nutrition API.
    Here we keep a small dictionary and a fallback heuristic.
    """
    # Base database (meal name -> estimated calories)
    CAL_DB = {
        "oatmeal with fruits": 300,
        "paneer sandwich": 350,
        "veg smoothie": 250,
        "dal rice bowl": 600,
        "paneer salad": 450,
        "veg khichdi": 400,
        "veg soup": 200,
        "grilled veggies": 300,
        "tofu curry": 500,
        "eggs and toast": 350,
        "chicken salad": 420
    }

    def estimate(self, meal_name: str) -> int:
        key = meal_name.lower()
        if key in self.CAL_DB:
            return self.CAL_DB[key]
        # fallback heuristic: base + random small variance
        base = 400
        variance = random.randint(-50, 100)
        return base + variance

    def batch_estimate(self, meals: List[str]) -> Dict[str, int]:
        return {m: self.estimate(m) for m in meals}



class IntentAgent:
    """
    Very simple keyword-based intent classifier.
    Returns a tuple (intent, metadata)
    """
    def classify(self, message: str) -> Tuple[str, Dict]:
        txt = message.lower()
        meta = {}
        # set diet
        if any(k in txt for k in ["vegetarian", "vegan", "non-veg", "non veg", "nonvegetarian"]):
            meta['diet'] = txt
            return "set_diet", meta
        # set calories
        if "calorie" in txt or "calories" in txt:
            digits = ''.join([c for c in txt if c.isdigit()])
            if digits:
                meta['calories'] = int(digits)
                return "set_calories", meta
            return "ask_calories", meta
        # plan creation
        if any(k in txt for k in ["plan", "week", "generate plan", "create plan"]):
            # optional: parse days or "week"
            if "week" in txt:
                meta['days'] = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
            elif "monday" in txt or "tuesday" in txt:
                # naive single-day detection
                days = [d for d in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"] if d in txt]
                meta['days'] = days or ["monday"]
            else:
                meta['days'] = ["monday"]
            return "create_plan", meta
        # suggest specific meal
        if any(k in txt for k in ["breakfast", "lunch", "dinner"]):
            return "suggest_meal", {}
        # show plan
        if any(k in txt for k in ["show plan", "show my plan", "show plan for", "display plan"]):
            return "show_plan", {}
        # fallback/general
        return "general", {}
class MealPlannerAgent:
    """
    Receives an intent and uses the CalorieTool to compose meal plans.
    Sequential agent: runs after IntentAgent.
    """
    def __init__(self, calorie_tool: CalorieTool):
        self.cal_tool = calorie_tool

        # A small catalogue of meals keyed by meal type for quick sampling
        self.catalog = {
            "breakfast": ["Oatmeal with fruits", "Paneer sandwich", "Veg smoothie", "Eggs and toast"],
            "lunch": ["Dal rice bowl", "Paneer salad", "Veg khichdi", "Chicken salad"],
            "dinner": ["Veg soup", "Grilled veggies", "Tofu curry"]
        }

    def _filter_by_diet(self, meal: str, diet: Optional[str]) -> bool:
        # Simplified dietary filter
        m = meal.lower()
        if not diet:
            return True
        diet = diet.lower()
        if "vegan" in diet:
            # disallow paneer, eggs, chicken, tofu is okay
            if any(x in m for x in ["paneer", "eggs", "chicken"]):
                return False
        if "vegetarian" in diet:
            # disallow chicken
            if "chicken" in m:
                return False
        if "non-veg" in diet or "non veg" in diet or "nonvegetarian" in diet:
            return True
        return True

    def suggest_meal(self, meal_type: str, session: InMemorySessionState) -> Tuple[str, int]:
        meal_type = meal_type.lower()
        candidates = self.catalog.get(meal_type, [])
        # choose first matching by diet preference
        for cand in candidates:
            if self._filter_by_diet(cand, session.diet):
                calories = self.cal_tool.estimate(cand)
                return cand, calories
        # fallback
        fallback = candidates[0] if candidates else "Simple salad"
        return fallback, self.cal_tool.estimate(fallback)

    def create_day_plan(self, day: str, session: InMemorySessionState) -> Dict[str, Dict]:
        """
        Create a plan for a single day (breakfast, lunch, dinner),
        estimate calories for each meal using CalorieTool,
        and store in session.
        """
        day_plan = {}
        meals = {}
        total_cal = 0
        for meal_type in ["breakfast", "lunch", "dinner"]:
            name, cal = self.suggest_meal(meal_type, session)
            meals[meal_type] = name
            total_cal += cal
            day_plan[meal_type] = {
                "name": name,
                "calories": cal
            }
        # Save into session (planner agent decides structure)
        session.save_day_plan(day, {k: day_plan[k]["name"] for k in day_plan})
        return {"day": day, "meals": day_plan, "total_calories": total_cal}

    def create_week_plan(self, days: List[str], session: InMemorySessionState) -> Dict[str, Dict]:
        """
        Create plans for multiple days. Returns mapping day -> day_plan_details
        """
        out = {}
        for d in days:
            out[d] = self.create_day_plan(d, session)
        return out
class ReplyAgent:
    """
    Formats replies to the user. Runs after MealPlannerAgent in the sequence.
    """
    def format_create_plan_response(self, plan_result: Dict) -> str:
        # plan_result can be a day plan or week plan
        if "day" in plan_result:
            day = plan_result["day"].capitalize()
            lines = [f"Meal plan for {day}:"]
            for t, info in plan_result["meals"].items():
                lines.append(f"  {t.capitalize()}: {info['name']} ({info['calories']} kcal)")
            lines.append(f"  Total calories (est.): {plan_result['total_calories']} kcal")
            return "\n".join(lines)
        # else assume week mapping
        lines = []
        for d, details in plan_result.items():
            lines.append(self.format_create_plan_response(details))
            lines.append("-" * 30)
        return "\n".join(lines)

    def format_suggest_meal(self, meal_name: str, calories: int) -> str:
        return f"Suggested {meal_name} — estimated {calories} kcal."

    def format_show_plan(self, session: InMemorySessionState) -> str:
        plan = session.get_plan()
        if not plan:
            return "No meal plan found in this session."
        return json.dumps(plan, indent=2)



class Coordinator:
    def __init__(self):
        self.session = InMemorySessionState()
        self.intent_agent = IntentAgent()
        self.calorie_tool = CalorieTool()
        self.planner_agent = MealPlannerAgent(self.calorie_tool)
        self.reply_agent = ReplyAgent()

    def handle(self, message: str) -> Dict:
        # 1) record message
        self.session.add_message("user", message)

        # 2) Intent classification (first agent)
        intent, meta = self.intent_agent.classify(message)

        # 3) Planner agent (second agent) - sequential flow
        reply_text = "Sorry, I didn't understand that."
        payload = {}
        if intent == "set_diet":
            # store diet in session
            # try to extract diet keyword simply
            diet_keywords = ["vegan", "vegetarian", "non-veg", "non veg", "nonvegetarian"]
            chosen = next((k for k in diet_keywords if k in message.lower()), message.lower())
            self.session.set_diet(chosen)
            reply_text = f"Diet preference updated to: {chosen}"
            payload = {"diet": chosen}

        elif intent == "set_calories":
            cal = meta.get("calories")
            if cal:
                self.session.set_calorie_goal(cal)
                reply_text = f"Daily calorie goal set to {cal} kcal."
                payload = {"calorie_goal": cal}
            else:
                reply_text = "I couldn't detect the calorie number — please specify e.g. 'set calories to 1800'."

        elif intent == "create_plan":
            days = meta.get("days", ["monday"])
            week_plan = self.planner_agent.create_week_plan(days, self.session)
            reply_text = self.reply_agent.format_create_plan_response(week_plan)
            payload = {"plan": week_plan}

        elif intent == "suggest_meal":
            # naive detection which meal type the user asked for
            if "breakfast" in message.lower():
                meal, cal = self.planner_agent.suggest_meal("breakfast", self.session)
                reply_text = self.reply_agent.format_suggest_meal(meal, cal)
                payload = {"meal": meal, "calories": cal}
            elif "lunch" in message.lower():
                meal, cal = self.planner_agent.suggest_meal("lunch", self.session)
                reply_text = self.reply_agent.format_suggest_meal(meal, cal)
                payload = {"meal": meal, "calories": cal}
            elif "dinner" in message.lower():
                meal, cal = self.planner_agent.suggest_meal("dinner", self.session)
                reply_text = self.reply_agent.format_suggest_meal(meal, cal)
                payload = {"meal": meal, "calories": cal}
            else:
                reply_text = "Which meal? breakfast / lunch / dinner?"
        elif intent == "show_plan":
            reply_text = self.reply_agent.format_show_plan(self.session)
            payload = {"plan": self.session.get_plan()}
        elif intent == "general":
            reply_text = "I can help with: set diet, set calories, create plan (e.g. 'create week plan'), suggest breakfast/lunch/dinner, or show plan."

        # 4) save agent reply to session history
        self.session.add_message("agent", reply_text)

        # 5) return structured output for UI / grading
        return {
            "intent": intent,
            "meta": meta,
            "reply": reply_text,
            "payload": payload,
            "session_snapshot": {
                "diet": self.session.diet,
                "calorie_goal": self.session.calorie_goal,
                "stored_plan_days": list(self.session.meal_plan.keys())
            }
        }


if __name__ == "__main__":
    c = Coordinator()

    demo_messages = [
        "I am vegetarian.",
        "Set my calorie goal to 1800.",
        "Create a week plan for me.",
        "Suggest a breakfast.",
        "Show my plan."
    ]

    for m in demo_messages:
        out = c.handle(m)
        print("\nUSER:", m)
        print("REPLY:")
        print(out["reply"])
        # print a short session snapshot
        print("SESSION SNAPSHOT:", out["session_snapshot"])
        print("-" * 60)

    # Print full stored session messages and final plan for verification / submission
    print("\n--- Full Conversation History (session.messages) ---")
    for msg in c.session.messages:
        print(f"{msg['time']} | {msg['role']}: {msg['content']}")
    print("\n--- Final Meal Plan (session.meal_plan) ---")
    print(json.dumps(c.session.meal_plan, indent=2))


