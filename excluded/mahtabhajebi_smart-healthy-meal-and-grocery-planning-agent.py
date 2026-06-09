# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # numeric calculations
import pandas as pd # data handling, tables

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Install required libraries from requirements.txt
!pip install -r /kaggle/input/healthyplan-recipe-dataset/requirements.txt


# the other Libraries also imported here
import sys
sys.path.append('/kaggle/input/healthyplan-recipe-dataset')  # Make agent scripts importable

import json
from tabulate import tabulate  # optional for nicely printing tables

# Optional standard libraries often used by agents
import random
import math
import copy


#to check what’s actually uploaded
#import os

#for dirname, _, filenames in os.walk('/kaggle/input/healthyplan-recipe-dataset/'):
   # print("Folder:", dirname)
   # for filename in filenames:
       # print("  ", filename)



# Load Dataset
dataset_path = '/kaggle/input/healthyplan-recipe-dataset/sample_recipes.json'

with open(dataset_path, 'r') as f:
    recipes = json.load(f)

print(f"Loaded {len(recipes)} recipes")


# Import agents
from requirement_agent import RequirementAgent
from research_agent import ResearchAgent
from planner_agent import PlannerAgent
from grocery_agent import GroceryAgent


## run multi-agent workflow
# 1. Collect user profile/preferences (fixed for Kaggle, no interactive input)
user_profile = {
    "diet": "vegetarian",
    "calories_per_day": 1800,
    "excludes": ["nuts"],
    "max_time_min": 30,
    "priorities": ["nutrition", "calories", "time"]
}

# 2. Fetch candidate recipes
from research_agent import ResearchAgent
research_agent = ResearchAgent()
research_agent.data_path = '/kaggle/input/healthyplan-recipe-dataset/sample_recipes.json'
candidate_recipes = research_agent.fetch_candidates(user_profile)

# 3. Build weekly meal plan
# Use full recipe dictionaries for each day
weekly_plan = []
for day in range(7):
    weekly_plan.append(candidate_recipes[day % len(candidate_recipes)])

# 4. Generate grocery list
from grocery_agent import GroceryAgent
grocery_agent = GroceryAgent()
grocery_list = grocery_agent.make_grocery_list(weekly_plan, user_profile)
print("\n### Grocery List ###\n")
for item, qty in grocery_list.items():
    print(f"- {item}: {qty}")


# display weekly meal plan
from tabulate import tabulate
table = [(i+1, r['title'], r['calories']) for i, r in enumerate(weekly_plan)]
print("\n### 7-Day Healthy Meal Plan ###\n")
print(tabulate(table, headers=["Day", "Recipe", "Calories"], tablefmt="grid"))

# display the grocery list
print("\n### Grocery List ###\n")
for item, qty in grocery_list.items():
    print(f"- {item}: {qty}")

# save the grocery list to a file
import json
with open("grocery_list.json", "w") as f:
    json.dump(grocery_list, f, indent=2)

