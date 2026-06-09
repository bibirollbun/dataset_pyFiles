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


import os

def write(path, content):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"âœ… Created: {path}")

# -----------------------------
# requirements.txt
# -----------------------------
write("requirements.txt", """
fastapi
uvicorn
requests
pydantic
""")

# -----------------------------
# main.py (The Orchestrator)
# -----------------------------
write("main.py", """
from agents.ingredient_interpreter import interpret_ingredients
from agents.recipe_generator import generate_recipes
from agents.ranker import rank_recipes
from memory import save_session

def run_smartchef(raw_input, diet=None, event=None):
    print("="*60)
    print(f"ğŸ�³ SmartChef AI Initialized")
    print(f"ğŸ‘¤ User Input: '{raw_input}' | Diet: {diet} | Event: {event}")
    print("="*60)

    # --- STEP 1: INTERPRET ---
    print("\\nğŸ§  Agent 1 (Interpreter): Analyzing ingredients...")
    clean_ingredients = interpret_ingredients(raw_input)
    print(f"   -> Identified: {[i['canonical'] for i in clean_ingredients]}")

    # --- MEMORY SAVE ---
    save_session("user_01", {"input": raw_input, "clean": clean_ingredients})

    # --- STEP 2: GENERATE ---
    print("\\nğŸ¥˜ Agent 2 (Generator): Creating detailed recipes...")
    recipes = generate_recipes(clean_ingredients)
    
    # --- STEP 3: RANK & FILTER ---
    print(f"\\nâ­� Agent 3 (Ranker): Personalizing for '{diet}' & '{event}'...")
    final_menu = rank_recipes(recipes, diet, event)

    # --- FINAL OUTPUT ---
    print("\\n" + "="*60)
    print(f"ğŸ�½ï¸�  FINAL MENU RECOMMENDATIONS  ğŸ�½ï¸�")
    print("="*60)
    
    if not final_menu:
        print("No matching recipes found for your diet preferences.")
    
    for i, r in enumerate(final_menu, 1):
        print(f"\\nOption {i}: {r['title'].upper()}")
        print(f"â�±ï¸�  Time: {r['time_mins']} mins | ğŸ”¥ Calories: {r['nutrition']['calories']}")
        print(f"ğŸ“� Description: {r['description']}")
        print("-" * 40)
        print("Instructions:")
        for step in r['steps']:
            print(f" - {step}")
        print("-" * 60)

if __name__ == "__main__":
    # You can change this input to test different scenarios
    run_smartchef(
        "2 eggs, some tomato, cheddar cheese, bread", 
        diet="vegetarian", 
        event="breakfast"
    )
""")


# -----------------------------
# agents/__init__.py
# -----------------------------
write("agents/__init__.py", "")

# -----------------------------
# Agent 1: Ingredient Interpreter
# -----------------------------
write("agents/ingredient_interpreter.py", """
import re

SYNONYMS = {
    "eggs": "egg", "tomatoes": "tomato", "cheese": "cheddar", 
    "bread": "toast", "scallions": "onion"
}

def interpret_ingredients(raw_text):
    # Simple logic to split commas and normalize names
    parts = [p.strip().lower() for p in raw_text.split(",") if p.strip()]
    cleaned = []
    for p in parts:
        # Check simple synonyms
        name = p
        for k, v in SYNONYMS.items():
            if k in p:
                name = v
                break
        # Remove numbers for canonical name (basic logic)
        name = re.sub(r'\d+', '', name).strip()
        
        cleaned.append({
            "original": p,
            "canonical": name
        })
    return cleaned
""")

# -----------------------------
# Agent 2: Recipe Generator (Now with Time & Steps!)
# -----------------------------
write("agents/recipe_generator.py", """
from tools.nutrition import calculate_nutrition

def generate_recipes(ingredients):
    ing_names = [i['canonical'] for i in ingredients]
    
    # Mock Recipe Logic based on ingredients
    # In a real app, this would use an LLM or DB search
    recipes = []

    # Recipe A: Omelet
    if 'egg' in ing_names:
        recipes.append({
            "title": "Fluffy Cheese Omelet",
            "time_mins": 10,
            "description": "A classic breakfast staple made fluffy and cheesy.",
            "ingredients": ing_names,
            "steps": [
                "Whisk eggs in a bowl until frothy.",
                "Heat pan and add a little oil/butter.",
                "Pour eggs in, lift edges to cook evenly.",
                "Add cheese/toppings and fold.",
                "Serve hot with toast."
            ],
            "nutrition": calculate_nutrition(ing_names)
        })

    # Recipe B: Sandwich
    if 'bread' in ing_names and 'tomato' in ing_names:
        recipes.append({
            "title": "Grilled Tomato Melt",
            "time_mins": 15,
            "description": "Crispy grilled bread with melted cheese and fresh tomato.",
            "ingredients": ing_names,
            "steps": [
                "Slice tomato and cheese.",
                "Butter the outside of the bread slices.",
                "Layer cheese and tomato inside.",
                "Grill on medium heat until golden brown."
            ],
            "nutrition": calculate_nutrition(ing_names)
        })

    # Recipe C: Scramble (Fallback)
    recipes.append({
        "title": "Quick Veggie Scramble",
        "time_mins": 8,
        "description": "Fast, nutritious, and messy in a good way.",
        "ingredients": ing_names,
        "steps": [
            "Chop all veggies.",
            "SautÃ© veggies in a pan for 2 mins.",
            "Crack eggs directly into pan.",
            "Scramble until cooked through."
        ],
        "nutrition": calculate_nutrition(ing_names)
    })

    return recipes
""")

# -----------------------------
# Agent 3: Ranker
# -----------------------------
write("agents/ranker.py", """
def rank_recipes(recipes, diet=None, event=None):
    scored = []
    for r in recipes:
        score = 0
        
        # Event Logic
        if event == "breakfast" and "egg" in r['title'].lower():
            score += 5
        if event == "lunch" and "sandwich" in r['title'].lower():
            score += 5
            
        # Diet Logic (Simple Filter)
        valid = True
        if diet == "vegan" and ("egg" in r['title'].lower() or "cheese" in r['title'].lower()):
            valid = False
            
        if valid:
            scored.append((score, r))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored]
""")


# -----------------------------
# tools/nutrition.py
# -----------------------------
write("tools/__init__.py", "")
write("tools/nutrition.py", """
def calculate_nutrition(ingredients):
    # Mock calorie estimator
    base = 100
    if 'egg' in ingredients: base += 70
    if 'cheese' in ingredients: base += 110
    if 'bread' in ingredients: base += 80
    if 'tomato' in ingredients: base += 20
    return {"calories": base, "protein": "High"}
""")

# -----------------------------
# memory.py
# -----------------------------
write("memory.py", """
_session_store = {}

def save_session(user_id, data):
    _session_store[user_id] = data
    # In a real app, this would write to a DB
""")


!python main.py

