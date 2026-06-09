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


# 1. SETUP & INSTALLS
!pip install google-generativeai --quiet --disable-pip-version-check

import os
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Optional

import google.generativeai as genai

print("Meal Maker Agent - Initialization")
print("-" * 60)



from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

print("Meal Maker Agent - Initialization")
print("-" * 60)

# Read from Kaggle Secrets (Add-ons → Secrets)
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")  # must match your secret name

if not GEMINI_API_KEY:
    raise ValueError("Kaggle secret 'GEMINI_API_KEY' is empty or missing. Check Add-ons → Secrets.")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/gemini-2.5-flash"
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

print("Gemini configured.")
print("Model:", MODEL_NAME)
print("-" * 60)



from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import json
import google.generativeai as genai

# -------------------------------------------------
# CONFIGURE GEMINI
# -------------------------------------------------

model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------------------------
# DATA MODELS
# -------------------------------------------------

class DietType(Enum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    KETO = "keto"
    PALEO = "paleo"
    MEDITERRANEAN = "mediterranean"
    OMNIVORE = "omnivore"
    CUSTOM = "custom"

@dataclass
class Meal:
    name: str
    meal_type: str
    estimated_calories: int
    ingredients_used: List[str]
    instructions: List[str]
    tags: Optional[List[str]] = None

@dataclass
class MealPlan:
    total_calories: int
    meals: List[Meal]
    notes: Optional[str] = None

@dataclass
class MealRequest:
    diet_type: DietType
    calorie_target: int
    meals_per_day: int
    available_ingredients: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    avoid_ingredients: Optional[List[str]] = None
    cuisine_preferences: Optional[List[str]] = None
    cooking_time_limit_minutes: Optional[int] = None
    servings: int = 1

# -------------------------------------------------
# GEMINI MEAL PLANNER
# -------------------------------------------------

class MealPlannerAgent:

    def call_gemini(self, prompt: str) -> str:
        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("Gemini returned an empty response")
        return response.text

    def generate_meal_plan(self, request: MealRequest) -> MealPlan:

        prompt = f"""
        Generate a structured meal plan.
        RETURN STRICT JSON ONLY.

        Requirements:
        - Diet: {request.diet_type.value}
        - Total calories: {request.calorie_target}
        - Meals per day: {request.meals_per_day}
        - Ingredients available: {request.available_ingredients}
        - Avoid: {request.avoid_ingredients}
        - Allergies: {request.allergies}
        - Cuisine preferences: {request.cuisine_preferences}
        - Time limit: {request.cooking_time_limit_minutes} minutes

        JSON Format:
        {{
            "total_calories": number,
            "meals": [
                {{
                    "name": "string",
                    "meal_type": "breakfast/lunch/dinner/snack",
                    "estimated_calories": number,
                    "ingredients_used": ["..."],
                    "instructions": ["..."],
                    "tags": ["..."]
                }}
            ],
            "notes": "string"
        }}
        """

        raw = self.call_gemini(prompt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting JSON if Gemini adds extra text
            cleaned = raw[raw.find("{") : raw.rfind("}") + 1]
            data = json.loads(cleaned)

        meals_list = []
        for m in data.get("meals", []):
            meals_list.append(Meal(
                name=m.get("name", ""),
                meal_type=m.get("meal_type", ""),
                estimated_calories=m.get("estimated_calories", 0),
                ingredients_used=m.get("ingredients_used", []),
                instructions=m.get("instructions", []),
                tags=m.get("tags", []),
            ))

        return MealPlan(
            total_calories=data.get("total_calories", request.calorie_target),
            meals=meals_list,
            notes=data.get("notes", ""),
        )

# -------------------------------------------------
# TEST RUN
# -------------------------------------------------

planner = MealPlannerAgent()

request = MealRequest(
    diet_type=DietType.VEGETARIAN,
    calorie_target=2000,
    meals_per_day=3,
    available_ingredients=["paneer", "rice", "tomato", "onion", "noodles", "curd", "spices", "oil","carrot","cauliflower","potato","green chilies", "coriander"],
    avoid_ingredients=["mushroom"],
    cuisine_preferences=["Chinese"],
    cooking_time_limit_minutes=30,
)

plan = planner.generate_meal_plan(request)

# BEAUTIFUL PRINT
print("\nDaily Meal Plan")
print("--------------------------")
print(f"Total Calories: {plan.total_calories}\n")

for m in plan.meals:
    print(f"{m.meal_type.upper()} — {m.name} ({m.estimated_calories} kcal)")
    print("Ingredients:", ", ".join(m.ingredients_used))
    print("Instructions:")
    for step in m.instructions:
        print(" -", step)
    print()

if plan.notes:
    print("Notes:", plan.notes)


