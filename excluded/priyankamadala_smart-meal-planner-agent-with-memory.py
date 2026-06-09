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



import json
import time
import random

class SimpleLLM:
    def generate(self, prompt):
        options = [
            "This weekly plan focuses on healthy and quick meals.",
            "This meal plan is budget-friendly and easy to prepare.",
            "This plan balances nutrition and simplicity."
        ]
        return random.choice(options)

llm = SimpleLLM()

class MemoryBank:
    def __init__(self):
        self.data = {}

    def save(self, user, preferences):
        self.data[user] = preferences

    def load(self, user):
        return self.data.get(user, "No preferences saved")

memory = MemoryBank()

def fetch_recipe(name):
    return {
        "dish": name,
        "ingredients": ["vegetables", "grains", "olive oil"],
        "steps": ["Cook vegetables", "Add grains", "Serve hot"]
    }
    
logs = []

def log(message):
    logs.append(message)
    print("LOG:", message)

def meal_planner_agent(user, preferences):
    log("Agent started")
    memory.save(user, preferences)

    overview = llm.generate(preferences)
    plan = []

    for day in range(1, 8):
        dish = f"Meal {day}"
        recipe = fetch_recipe(dish)
        plan.append(recipe)

    shopping_list = ["vegetables", "grains", "olive oil"]

    log("Agent finished")

    return {
        "overview": overview,
        "meal_plan": plan,
        "shopping_list": shopping_list
    }
    
result = meal_planner_agent(
    user="User1",
    preferences="Vegetarian, low budget"
)

result


