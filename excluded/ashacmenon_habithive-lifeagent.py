
import os
import sqlite3
import json
import requests
import random
import time
from datetime import datetime, timedelta
import uuid

# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
LLM_MODEL = "gpt-3.5-turbo"

CITY = "Chennai"
LAT = 13.0827
LON = 80.2707
DB_PATH = "digest_corrected.db"

# DB helpers
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS expenses(
        id TEXT PRIMARY KEY, ts TEXT, amount REAL, category TEXT, note TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS habits(
        id TEXT PRIMARY KEY, name TEXT, frequency TEXT, last_done TEXT, streak INTEGER)""")
    conn.commit()
    return conn

DB = init_db()

def add_expense(amount, category, note=""):
    DB.execute("INSERT INTO expenses VALUES(?,?,?,?,?)",
               (str(uuid.uuid4()), datetime.utcnow().isoformat(), amount, category, note))
    DB.commit()

def list_expenses(days=7):
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = DB.execute("SELECT amount,category FROM expenses WHERE ts>=?", (since,)).fetchall()
    return [{"amount": r[0], "category": r[1]} for r in rows]

def add_habit(name):
    DB.execute("INSERT OR IGNORE INTO habits VALUES(?,?,?,?,?)",
               (str(uuid.uuid4()), name, "daily", None, 0))
    DB.commit()

def list_habits():
    rows = DB.execute("SELECT name FROM habits").fetchall()
    return [r[0] for r in rows]

# Fill sample data if empty
if not list_habits():
    add_habit("Drink Water")
    add_habit("Study 1 hour")

if not list_expenses():
    add_expense(40, "food")
    add_expense(120, "transport")

# LLM wrapper with fact-aware mock fallback
class LLM:
    def __init__(self, key=OPENAI_API_KEY):
        self.key = key
        self.real = bool(key)

    def _mock_weather_advice(self, temp, wind):
        if temp is None:
            return "Weather data unavailable."
        t = float(temp)
        adv = []
        if t >= 30:
            adv.append("Stay hydrated and avoid prolonged sun exposure.")
        elif t <= 18:
            adv.append("It may be cool — consider a light layer.")
        else:
            adv.append("Comfortable temperatures — a good day for focused work.")
        if wind and float(wind) > 20:
            adv.append("Windy conditions — secure loose items.")
        return " ".join(adv)

    def _mock_budget_advice(self, totals):
        if not totals:
            return "No expenses recorded — consider tracking small purchases."
        # find largest category
        max_cat = max(totals.items(), key=lambda x: x[1])
        cat, amt = max_cat
        if amt > 500:
            return f"High spending in {cat} — review recurring costs there."
        if cat in ("food", "dining", "coffee"):
            return "Cut small food/coffee purchases to save a noticeable amount."
        return f"Spending largely on {cat}. Try trimming non-essential items."

    def _mock_habit_advice(self, habit_names):
        if not habit_names:
            return "No habits tracked — start with one simple habit today (e.g., 10-min walk)."
        if len(habit_names) >= 3:
            return "You have multiple habits — focus on completing the most important one today."
        return f"Keep at '{habit_names[0]}' and try to perform it at the same time daily."

    def _mock_motivation(self, context):
        # short motivational line referencing habits if present
        habits = context.split(",") if context else []
        if habits and habits[0].strip():
            return f"Small consistent steps with {habits[0].strip()} will compound into big gains."
        return random.choice([
            "Take one small, meaningful step today.",
            "Consistency wins — do a little every day.",
            "Focus on one task and finish it."
        ])

    def generate(self, system, text):
        # system param ignored in mock, used for real LLM
        if not self.real:
            # route to specific mock based on system hint
            s = system.lower() if system else ""
            if "weather" in s:
                # expect text like "temp X, wind Y"
                temp = None; wind = None
                try:
                    parts = text.replace("°C","").replace("km/h","").split(",")
                    for p in parts:
                        if "temp" in p.lower() or "temperature" in p.lower():
                            temp = ''.join([c for c in p if (c.isdigit() or c=='.' or c=='-')])
                        if "wind" in p.lower():
                            wind = ''.join([c for c in p if (c.isdigit() or c=='.')])
                except:
                    pass
                return self._mock_weather_advice(temp, wind)
            if "spend" in s or "finance" in s or "budget" in s:
                try:
                    totals = json.loads(text)
                except:
                    totals = {}
                return self._mock_budget_advice(totals)
            if "habit" in s:
                # text is comma-separated names
                names = [n.strip() for n in text.split(",")] if text else []
                return self._mock_habit_advice(names)
            if "motivation" in s or "motivat" in s:
                return self._mock_motivation(text)
            # generic fallback
            return random.choice([
                "Focus on small wins today.",
                "Keep tasks manageable and consistent."
            ])

        # Real OpenAI call path
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
        body = {
            "model": LLM_MODEL,
            "messages":[{"role":"system","content":system},{"role":"user","content":text}],
            "max_tokens":120,
            "temperature":0.7
        }
        try:
            r = requests.post(OPENAI_API_URL, headers=headers, json=body, timeout=15)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[llm_error] {e}"

llm = LLM()

# Agents that produce raw facts and ask LLM for advice (LLM uses raw facts only)
class WeatherAgent:
    def get_raw(self, lat, lon):
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            data = requests.get(url, timeout=5).json().get("current_weather", {})
            return data  # may contain temperature, windspeed etc.
        except:
            return {}

    def output(self):
        raw = self.get_raw(LAT, LON)
        if raw:
            t = raw.get("temperature"); w = raw.get("windspeed")
            fact = f"Temperature {t}°C, wind {w} km/h"
        else:
            fact = "Weather data unavailable"
        advice = llm.generate("Weather advisor", fact)
        return fact, advice

class BudgetAgent:
    def output(self, days=7):
        exps = list_expenses(days)
        totals = {}
        for e in exps:
            totals[e["category"]] = totals.get(e["category"], 0) + e["amount"]
        fact = totals  # dict
        advice = llm.generate("Finance advisor", json.dumps(fact))
        return fact, advice

class HabitAgent:
    def output(self):
        names = list_habits()
        fact = names  # list
        names_str = ", ".join(names) if names else ""
        advice = llm.generate("Habit coach", names_str)
        return fact, advice

class MotivationAgent:
    def output(self, habit_context):
        # pass only habit names or empty string
        advice = llm.generate("Motivation generator", habit_context)
        return advice

class SummaryAgent:
    def output(self, weather_fact, budget_fact, habits_fact, weather_advice, budget_advice, habit_advice, motiv):
       
        raw = {
            "weather_fact": weather_fact,
            "budget_fact": budget_fact,
            "habits_fact": habits_fact
        }
        prompt = (
            f"Create one concise 1-line summary (no repetition). "
            f"Facts: {json.dumps(raw)}. "
            f"Tips (one sentence each): weather_tip: {weather_advice}; budget_tip: {budget_advice}; habit_tip: {habit_advice}; motivation: {motiv}."
            " Now produce exactly one short sentence that synthesizes these facts and tips without repeating phrases."
        )
        return llm.generate("Daily summary writer", prompt)

# Run agents
weather_agent = WeatherAgent()
budget_agent = BudgetAgent()
habit_agent = HabitAgent()
mot_agent = MotivationAgent()
summary_agent = SummaryAgent()

weather_fact, weather_advice = weather_agent.output()
budget_fact, budget_advice = budget_agent.output()
habits_fact, habit_advice = habit_agent.output()
habit_context = ", ".join(habits_fact) if habits_fact else ""
motivation_line = mot_agent.output(habit_context)
final_summary = summary_agent.output(weather_fact, budget_fact, habits_fact, weather_advice, budget_advice, habit_advice, motivation_line)

# Print structured digest with facts preserved and concise advice
print("DAILY DIGEST —", CITY)
print()
print("Weather:", weather_fact)
print("Weather Tip:", weather_advice)
print("Spending (totals):", budget_fact)
print("Spending Tip:", budget_advice)
print("Habits:", habits_fact)
print("Habit Tip:", habit_advice)
print("Motivation:", motivation_line)
print("Today's Summary:", final_summary)


