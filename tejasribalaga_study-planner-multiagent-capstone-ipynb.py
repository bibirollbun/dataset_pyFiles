# Imports & basic helpers
import json, os, datetime, uuid, math
from typing import List, Dict, Any
from dataclasses import dataclass
print("Ready — imports successful.")



# === User profile (customize if you want) ===
USER_PROFILE = {
    "subjects": ["Maths","Science","Programming","English","Social","Telugu","Hindi"],
    "days": 30,
    "daily_hours": 3,
    "difficulty": {
        "Maths":"hard","Science":"hard","Programming":"hard",
        "English":"easy","Social":"medium","Telugu":"medium","Hindi":"medium"
    },
    # optional: supply a custom topic bank to override defaults
    "topic_bank": {
        # "Maths": ["Arithmetic","Algebra","Geometry"]  # example override
    }
}
print("USER_PROFILE set. Subjects:", USER_PROFILE["subjects"])



# Memory store (file-backed)
class MemoryStore:
    def __init__(self, path="study_memory.json"):
        self.path = path
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"runs": [], "plan": None, "schedule": None, "progress": {}}
        else:
            self.data = {"runs": [], "plan": None, "schedule": None, "progress": {}}
            self._save()
    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
    def save_plan(self, plan, schedule):
        self.data["plan"] = plan
        self.data["schedule"] = schedule
        self._save()
    def add_run(self, run):
        self.data["runs"].append(run); self._save()
    def update_progress(self, day, record):
        self.data["progress"].setdefault(str(day), []).append(record); self._save()
    def get_progress(self):
        return self.data["progress"]
    def path_exists(self):
        return os.path.exists(self.path)

# quick check
mem = MemoryStore()
print("Memory file:", mem.path, "exists:", mem.path_exists())



# Planner Agent (prototype)
TOPICS_BANK = {
    "Maths":["Arithmetic & Number Sense","Fractions & Decimals","Algebra basics","Geometry basics","Word problems","Revision"],
    "Science":["Biology basics","Physics basics","Chemistry basics","Experiments & application","Revision"],
    "Programming":["Syntax & variables","Control flow","Functions & modularity","Data types & structures","Small projects & debugging","Revision"],
    "English":["Grammar basics","Reading comprehension","Writing & sentences","Vocabulary & spelling","Practice"],
    "Social":["History basics","Civics basics","Geography basics","Maps & current events","Revision"],
    "Telugu":["Reading & comprehension","Grammar basics","Writing practice","Vocabulary","Short passages"],
    "Hindi":["Reading & comprehension","Grammar basics","Writing practice","Vocabulary","Short passages"]
}

class PlannerAgent:
    def __init__(self, memory: MemoryStore):
        self.mem = memory
    def allocate_hours(self, total_hours, difficulty_map):
        weights={"hard":3,"medium":2,"easy":1}
        subj_weights = {s: weights.get(difficulty_map.get(s,"medium"),2) for s in difficulty_map}
        total_w = sum(subj_weights.values())
        unit = total_hours / total_w
        subj_hours = {s: round(unit * w) for s,w in subj_weights.items()}
        # fix rounding to match total_hours
        diff = int(total_hours - sum(subj_hours.values()))
        subs_sorted = sorted(subj_hours.keys(), key=lambda s: subj_weights[s], reverse=True)
        i = 0
        while diff != 0:
            subj_hours[subs_sorted[i % len(subs_sorted)]] += (1 if diff > 0 else -1)
            diff += -1 if diff > 0 else 1
            i += 1
        return subj_hours

    def plan(self, user_profile):
        total_hours = user_profile["days"] * user_profile["daily_hours"]
        subj_hours = self.allocate_hours(total_hours, user_profile["difficulty"])
        topics_map = {}
        for s,h in subj_hours.items():
            bank = user_profile.get("topic_bank", {}).get(s, TOPICS_BANK.get(s, [s+" - General"]))
            per_topic = round(h / len(bank), 2) if len(bank) > 0 else h
            topics_map[s] = [{"name": t, "est_hours": per_topic} for t in bank]
        plan = {"subject_hours": subj_hours, "topics": topics_map}
        return plan

# quick planner demo
planner = PlannerAgent(mem)
sample_plan = planner.plan(USER_PROFILE)
print("Planner sample subject_hours:", sample_plan["subject_hours"])



# Scheduler Agent
class SchedulerAgent:
    def __init__(self, memory: MemoryStore):
        self.mem = memory
    def schedule(self, plan: Dict[str,Any], days:int, daily_hours:int):
        daily_minutes = daily_hours * 60
        schedule = {d: [] for d in range(1, days+1)}
        flattened = []
        for s, topics in plan["topics"].items():
            for t in topics:
                mins = int(round(t["est_hours"] * 60))
                flattened.append({"subject": s, "topic": t["name"], "mins": max(15, mins)})
        cur_day = 1
        for item in flattened:
            while item["mins"] > 0 and cur_day <= days:
                used = sum(x["mins"] for x in schedule[cur_day])
                free = daily_minutes - used
                if free <= 0:
                    cur_day += 1
                    continue
                take = min(free, min(60, item["mins"]))
                schedule[cur_day].append({"subject": item["subject"], "topic": item["topic"], "mins": take})
                item["mins"] -= take
        # save to memory
        self.mem.save_plan(plan, schedule)
        return schedule

# quick scheduler demo
scheduler = SchedulerAgent(mem)
sample_schedule = scheduler.schedule(sample_plan, USER_PROFILE["days"], USER_PROFILE["daily_hours"])
print("Day 1 sample slots:", sample_schedule[1][:5])



# Evaluator Agent
class EvaluatorAgent:
    def __init__(self, memory: MemoryStore):
        self.mem = memory
    def simulated_checkin(self, day:int, day_slots:List[Dict[str,Any]]):
        completed=[]
        for s in day_slots:
            ok = s["mins"] <= 45 or (hash(s["topic"]+str(day)) % 10) < 8
            completed.append({"subject": s["subject"], "topic": s["topic"], "mins": s["mins"], "completed": ok})
        self.mem.update_progress(day, {"slots": completed, "timestamp": str(datetime.datetime.now())})
        return completed
    def interactive_checkin(self, day:int, day_slots:List[Dict[str,Any]]):
        print(f"\nDay {day} check-in (interactive). Please reply 'y' or 'n'.")
        completed=[]
        for i,s in enumerate(day_slots, start=1):
            resp = input(f"[{i}] {s['subject']} - {s['topic']} ({s['mins']} min) - Completed? (y/n): ").strip().lower()
            ok = resp == 'y'
            completed.append({"subject": s["subject"], "topic": s["topic"], "mins": s["mins"], "completed": ok})
        self.mem.update_progress(day, {"slots": completed, "timestamp": str(datetime.datetime.now())})
        return completed



# ============================
# REQUIRED IMPORTS
# ============================
import uuid
import datetime
from typing import Dict, Any

# Make sure these classes are already defined:
# PlannerAgent, SchedulerAgent, EvaluatorAgent, Memory
# Example: planner = PlannerAgent(mem)

# ============================
# ORCHESTRATOR CLASS
# ============================

class Orchestrator:
    def __init__(self, planner, scheduler, evaluator, memory):
        self.planner = planner
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.mem = memory

    def run(self, user_profile: Dict[str,Any], interactive: bool=False, simulate_days: int=7):
        # Step 1: Generate plan
        plan = self.planner.plan(user_profile)

        # Step 2: Generate schedule
        schedule = self.scheduler.schedule(
            plan, 
            user_profile["days"], 
            user_profile["daily_hours"]
        )

        # Step 3: Save run metadata
        run_id = str(uuid.uuid4())[:8]
        self.mem.add_run({
            "id": run_id,
            "created": str(datetime.datetime.now()),
            "subject_hours": plan["subject_hours"]
        })

        # Step 4: Simulate or run interactively
        days_to_run = user_profile["days"] if interactive else min(user_profile["days"], simulate_days)

        for d in range(1, days_to_run + 1):
            day_slots = schedule.get(d, [])
            print(f"\n=== Day {d} schedule ({sum(s['mins'] for s in day_slots)} min) ===")
            for s in day_slots:
                print(f" - {s['subject']}: {s['topic']} ({s['mins']} min)")

            # evaluator decides: interactive or simulated
            if interactive:
                result = self.evaluator.interactive_checkin(d, day_slots)
            else:
                result = self.evaluator.simulated_checkin(d, day_slots)

            # Completion %
            done = sum(x["mins"] for x in result if x.get("completed"))
            tot = sum(x["mins"] for x in result)
            pct = round(100 * done / tot, 2) if tot > 0 else 0.0

            print(f"Day {d} completion: {pct}%")

            if pct < 60:
                print(" - Suggestion: Consider rebalancing next day's schedule.")

        print("\nRun saved to:", self.mem.path)

        return {
            "run_id": run_id,
            "subject_hours": plan["subject_hours"],
            "days_run": days_to_run
        }

# ============================
# INSTANTIATE AGENTS & ORCHESTRATOR
# ============================

planner = PlannerAgent(mem)
scheduler = SchedulerAgent(mem)
evaluator = EvaluatorAgent(mem)

orchestrator = Orchestrator(planner, scheduler, evaluator, mem)

print("Orchestrator successfully created!")



# Run a quick simulated demo (first 7 days)
result = orchestrator.run(USER_PROFILE, interactive=False, simulate_days=7)
print("\nResult summary:", result)



# Interactive run: change simulate_days if you want fewer days
# WARNING: interactive=True will request y/n input in the notebook output.
# Set interactive=False if you only want simulation.
interactive_run = False   # <-- set True to try the interactive mode
if interactive_run:
    res = orchestrator.run(USER_PROFILE, interactive=True, simulate_days=USER_PROFILE["days"])
    print("Interactive run finished:", res)
else:
    print("Interactive run skipped (interactive_run=False).")



# Display memory contents (pretty)
with open(mem.path, "r") as f:
    data = json.load(f)
print("Plan summary (subject_hours):")
print(json.dumps(data.get("plan", {}).get("subject_hours", {}), indent=2))
print("\nSample schedule: Day 1")
print(json.dumps(data.get("schedule", {}).get("1", []), indent=2))
print("\nProgress (keys):", list(data.get("progress", {}).keys())[:10])


