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


# Concierge Agent System - Imports
import datetime
import random


# Inbuilt user dataset (no manual input required)
user_profile = {
    "name": "Mithiesh",
    "age": 22,
    "wake_time": "06:30 AM",
    "sleep_time": "10:00 PM",
    "diet_preference": "Vegetarian",
    "activity_level": "Moderate",
    "goals": ["study", "stay healthy", "daily productivity"]
}

user_profile


class SchedulePlannerAgent:
    def create_schedule(self, profile):
        schedule = {
            "06:30 AM": "Wake up",
            "07:00 AM": "Exercise (20 mins yoga)",
            "07:30 AM": "Breakfast",
            "09:00 AM": "Study Session 1",
            "12:30 PM": "Lunch Break",
            "02:00 PM": "Study Session 2",
            "05:00 PM": "Tea Break / Relax",
            "06:00 PM": "Light Walk",
            "08:00 PM": "Dinner",
            "10:00 PM": "Sleep"
        }
        return schedule

schedule_agent = SchedulePlannerAgent()
daily_schedule = schedule_agent.create_schedule(user_profile)
daily_schedule


class ReminderAgent:
    def generate_reminders(self):
        reminders = [
            "Drink 3L water today ğŸ’§",
            "Take deep breaths every 2 hours ğŸ§˜",
            "Maintain correct posture while studying",
            "Avoid long mobile usage ğŸ“µ"
        ]
        return reminders

reminder_agent = ReminderAgent()
reminders = reminder_agent.generate_reminders()
reminders


class MealPlannerAgent:
    def veg_meals(self):
        breakfast = ["Oats Upma", "Idly + Sambar", "Veg Sandwich", "Poha"]
        lunch = ["Veg Thali", "Curd Rice", "Dal + Roti + Sabzi", "Veg Biryani"]
        dinner = ["Chapati + Curry", "Lemon Rice", "Mixed Veg Soup", "Dosa"]
        
        return {
            "Breakfast": random.choice(breakfast),
            "Lunch": random.choice(lunch),
            "Dinner": random.choice(dinner)
        }

meal_agent = MealPlannerAgent()
meal_suggestions = meal_agent.veg_meals()
meal_suggestions


class FitnessAgent:
    def workout_plan(self, activity_level):
        if activity_level == "Low":
            return ["10-min walk", "5-min stretching"]
        elif activity_level == "Moderate":
            return ["20-min yoga", "15-min brisk walk", "Beginner core workout"]
        else:
            return ["30-min running", "20-min strength training"]
        
fitness_agent = FitnessAgent()
fitness_plan = fitness_agent.workout_plan(user_profile["activity_level"])
fitness_plan


class SupervisorAgent:
    def __init__(self, schedule, reminders, meals, fitness):
        self.schedule = schedule
        self.reminders = reminders
        self.meals = meals
        self.fitness = fitness

    def generate_daily_summary(self, name):
        summary = {
            "User": name,
            "Daily Schedule": self.schedule,
            "Reminders": self.reminders,
            "Meal Plan": self.meals,
            "Fitness Plan": self.fitness,
            "Generated On": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return summary


supervisor = SupervisorAgent(
    daily_schedule,
    reminders,
    meal_suggestions,
    fitness_plan
)

daily_summary = supervisor.generate_daily_summary(user_profile["name"])
daily_summary


print("===== DAILY CONCIERGE SUMMARY =====")

print("\nğŸ‘¤ USER:", daily_summary["User"])

print("\nğŸ“… DAILY SCHEDULE")
for time, activity in daily_summary["Daily Schedule"].items():
    print(f"{time} â�� {activity}")

print("\nâ�° REMINDERS")
for r in daily_summary["Reminders"]:
    print("â€¢", r)

print("\nğŸ�½ï¸� MEAL PLAN")
for meal, food in daily_summary["Meal Plan"].items():
    print(f"{meal}: {food}")

print("\nğŸ�‹ï¸� FITNESS PLAN")
for exercise in daily_summary["Fitness Plan"]:
    print("â€¢", exercise)

print("\nğŸ“Œ SUMMARY GENERATED ON:", daily_summary["Generated On"])

