# ==============================================================
# AI FITNESS & DIET PLANNER AGENT (KAGGLE-FRIENDLY)
# ==============================================================

import pandas as pd
import random

# ==============================================================
# 1) LOAD NUTRITION DATASET (Update dataset path here)
# ==============================================================

dataset_path = "/kaggle/input/nutrition/nutrition.csv" 
# Example path. Replace with your actual dataset.

try:
    df = pd.read_csv(dataset_path)
except:
    print("â�Œ Error: Please update dataset_path to point to your Kaggle dataset.")
    df = None


# ==============================================================
# 2) BMR + CALORIE CALCULATIONS
# ==============================================================

def calculate_bmr(gender, weight, height, age):
    if gender.lower() == "male":
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

activity_multipliers = {
    "low": 1.2,
    "medium": 1.55,
    "high": 1.75
}


# ==============================================================
# 3) DIET PLAN GENERATOR
# ==============================================================

def generate_meal_plan(target_calories):
    if df is None:
        return ["Dataset not loaded. Cannot generate meals."]

    meals = []

    # Randomly generate 3 meals + snacks
    for _ in range(4):
        item = df.sample(1).iloc[0]
        meals.append(f"{item['Shrt_Desc']} ({round(item['Energy_Kcal'])} kcal)")

    return meals


# ==============================================================
# 4) WORKOUT PLAN GENERATOR
# ==============================================================

def generate_workout(goal):
    if goal == "weight loss":
        return [
            "30 minutes brisk walking",
            "20 minutes HIIT",
            "15 minutes core training",
            "Light stretching"
        ]
    elif goal == "muscle gain":
        return [
            "Chest + Triceps (45 mins)",
            "Back + Biceps (45 mins)",
            "Leg Day (1 hour)",
            "Shoulders + Core (45 mins)"
        ]
    else:
        return [
            "20 minutes jogging",
            "Full-body strength circuit (30 mins)",
            "Yoga / mobility (20 mins)"
        ]


# ==============================================================
# 5) MAIN AGENT INTERACTION
# ==============================================================

def run_agent():
    print("=== AI FITNESS & DIET PLANNER ===")

    goal = input("Goal (weight loss / muscle gain / maintain): ").strip().lower()
    age = int(input("Age: "))
    height = float(input("Height in cm: "))
    weight = float(input("Weight in kg: "))
    gender = input("Gender (male/female): ").strip().lower()
    activity = input("Activity level (low/medium/high): ").strip().lower()

    # Calorie calculation
    bmr = calculate_bmr(gender, weight, height, age)
    total_calories = bmr * activity_multipliers.get(activity, 1.2)

    print("\n=== YOUR DAILY FITNESS REPORT ===")
    print(f"BMR: {round(bmr)} kcal/day")
    print(f"Daily Calorie Needs: {round(total_calories)} kcal")
    
    print("\n--- Recommended Meal Plan ---")
    meals = generate_meal_plan(total_calories)
    for m in meals:
        print("â€¢", m)

    print("\n--- Recommended Workout Plan ---")
    workout = generate_workout(goal)
    for w in workout:
        print("â€¢", w)

    print("\nStay consistent! ğŸ’ªğŸ”¥")


# ==============================================================
# Run the agent
# ==============================================================

run_agent()


