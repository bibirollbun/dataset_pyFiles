# --------------------------------------------------------
# SMARTFIT – Multi-Agent Workout & Diet Planning System
# --------------------------------------------------------

import json
from datetime import datetime, timedelta

# -------------------------------
# MEMORY (Long Term Storage)
# -------------------------------
memory = {
    "user_profile": {},
    "weekly_plans": {},
    "progress": []
}

# -------------------------------
# TOOL 1: Simple Calorie Calculator
# -------------------------------
food_database = {
    "oats": {"cal": 389, "protein": 17, "carbs": 66, "fat": 7},
    "egg": {"cal": 155, "protein": 13, "carbs": 1.1, "fat": 11},
    "banana": {"cal": 89, "protein": 1.1, "carbs": 23, "fat": 0.3},
    "chicken breast": {"cal": 165, "protein": 31, "carbs": 0, "fat": 3.6},
    "rice": {"cal": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
}

def calorie_tool(food_name, grams):
    """Return calorie and macro info."""
    if food_name not in food_database:
        return {"error": "Food not found"}

    base = food_database[food_name]
    factor = grams / 100
    return {
        "cal": round(base["cal"] * factor, 1),
        "protein": round(base["protein"] * factor, 1),
        "carbs": round(base["carbs"] * factor, 1),
        "fat": round(base["fat"] * factor, 1),
    }

# -------------------------------
# AGENT 1: Profile Builder Agent
# -------------------------------
def profile_agent(name, age, weight, height, goal, diet_type):
    profile = {
        "name": name,
        "age": age,
        "weight": weight,
        "height": height,
        "goal": goal,
        "diet_type": diet_type
    }
    memory["user_profile"] = profile
    return profile

# -------------------------------
# AGENT 2: Workout Planner Agent
# -------------------------------
def workout_agent(profile):
    goal = profile["goal"]
    
    if goal == "weight_loss":
        plan = {
            "Mon": ["Cardio 30 mins", "Pushups 3x12", "Squats 3x15"],
            "Wed": ["Cycling 20 mins", "Lunges 3x12", "Plank 3x1 min"],
            "Fri": ["Jogging 20 mins", "Burpees 3x10", "Situps 3x15"]
        }
    else:
        plan = {
            "Mon": ["Bench Press 4x8", "Pushups 3x12", "Dips 3x10"],
            "Wed": ["Squats 4x8", "Lunges 3x12", "Calf Raises 3x20"],
            "Fri": ["Deadlift 4x6", "Rows 3x10", "Plank 3x1 min"]
        }

    memory["weekly_plans"]["workout"] = plan
    return plan

# -------------------------------
# AGENT 3: Diet Planner Agent
# -------------------------------
def diet_agent(profile):
    weight = profile["weight"]

    # formula: 30 calories per kg
    daily_calories = weight * 30
    
    # sample meals using tool
    breakfast = calorie_tool("oats", 60)
    lunch = calorie_tool("chicken breast", 150)
    dinner = calorie_tool("rice", 150)
    
    diet_plan = {
        "target_calories": daily_calories,
        "meals": {
            "breakfast": {"food": "oats", "nutrition": breakfast},
            "lunch": {"food": "chicken breast", "nutrition": lunch},
            "dinner": {"food": "rice", "nutrition": dinner}
        }
    }
    
    memory["weekly_plans"]["diet"] = diet_plan
    return diet_plan

# -------------------------------
# AGENT 4: Progress Tracking Loop Agent
# -------------------------------
def progress_agent(new_weight):
    profile = memory["user_profile"]
    old_weight = profile["weight"]

    change = round(new_weight - old_weight, 2)
    
    record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "old_weight": old_weight,
        "new_weight": new_weight,
        "change": change
    }
    
    memory["progress"].append(record)
    
    # update long-term memory
    memory["user_profile"]["weight"] = new_weight

    return record

# -------------------------------
# RUN ENTIRE SYSTEM
# -------------------------------

print("=== SMARTFIT CAPSTONE PROJECT RUNNING ===\n")

# Step 1: Build profile
profile = profile_agent(
    name="Riya",
    age=20,
    weight=65,
    height=165,
    goal="weight_loss",
    diet_type="veg"
)

print("User Profile Saved:\n", json.dumps(profile, indent=2), "\n")

# Step 2: Generate Workout Plan
workout_plan = workout_agent(profile)
print("Workout Plan Generated:\n", json.dumps(workout_plan, indent=2), "\n")

# Step 3: Generate Diet Plan
diet_plan = diet_agent(profile)
print("Diet Plan Generated:\n", json.dumps(diet_plan, indent=2), "\n")

# Step 4: Weekly Progress Update (Loop Agent)
print("--- Weekly Progress Loop Simulation ---")
week1 = progress_agent(64.2)
week2 = progress_agent(63.5)
print("Week 1:", week1)
print("Week 2:", week2)

print("\n=== FINAL MEMORY STATE ===")
print(json.dumps(memory, indent=2))

