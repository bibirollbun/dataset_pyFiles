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


# ===============================================================
# ğŸ�‹ï¸� NutriPlan AI â€” Gym Edition (Dynamic Meal Variety + Memory)
# ===============================================================

import gradio as gr
import json, random, datetime

# -----------------------------
# 1ï¸�âƒ£ Session Memory
# -----------------------------
class SessionMemory:
    """JSON-backed session memory that remembers previous plans."""
    def __init__(self, path="memory.json"):
        self.path = path
        try:
            with open(path, "r") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {"history": []}
    def save(self, key, value):
        self.data[key] = value
        with open(self.path, "w") as f:
            json.dump(self.data, f)
    def get(self, key, default=None):
        return self.data.get(key, default)
    def add_history(self, meal_names):
        """Stores previous meal names to avoid repetition."""
        history = set(self.data.get("history", []))
        history.update(meal_names)
        self.data["history"] = list(history)[-10:]  # keep last 10 only
        with open(self.path, "w") as f:
            json.dump(self.data, f)
    def get_history(self):
        return set(self.data.get("history", []))

# -----------------------------
# 2ï¸�âƒ£ Recipe Database
# -----------------------------
def recipe_database():
    return {
        "bulking": [
            # Veg
            {"name": "Protein Oats", "cal": 650, "protein": 40, "carbs": 55, "fat": 10, "ingredients": ["oats","milk","protein powder","banana"], "veg": True},
            {"name": "Paneer Wrap", "cal": 700, "protein": 35, "carbs": 60, "fat": 20, "ingredients": ["paneer","onion","tortilla","yogurt"], "veg": True},
            {"name": "Vegetable Biryani", "cal": 750, "protein": 30, "carbs": 90, "fat": 15, "ingredients": ["rice","peas","carrot","paneer"], "veg": True},
            {"name": "Tofu Rice Bowl", "cal": 700, "protein": 35, "carbs": 70, "fat": 12, "ingredients": ["tofu","rice","soy sauce","broccoli"], "veg": True},
            # Non-veg
            {"name": "Chicken & Rice Bowl", "cal": 750, "protein": 45, "carbs": 70, "fat": 15, "ingredients": ["chicken","rice","olive oil","spinach"], "veg": False},
            {"name": "Egg & Oats Scramble", "cal": 600, "protein": 35, "carbs": 45, "fat": 12, "ingredients": ["eggs","oats","milk"], "veg": False},
        ],
        "weight_loss": [
            {"name": "Greek Yogurt Parfait", "cal": 300, "protein": 20, "carbs": 25, "fat": 5, "ingredients": ["yogurt","berries","honey"], "veg": True},
            {"name": "Veggie Stir Fry", "cal": 350, "protein": 15, "carbs": 40, "fat": 10, "ingredients": ["tofu","bell pepper","soy sauce"], "veg": True},
            {"name": "Lentil Soup", "cal": 400, "protein": 25, "carbs": 30, "fat": 6, "ingredients": ["lentils","carrots","celery"], "veg": True},
            {"name": "Quinoa Salad", "cal": 380, "protein": 22, "carbs": 35, "fat": 8, "ingredients": ["quinoa","tomatoes","olive oil"], "veg": True},
            {"name": "Grilled Salmon & Veggies", "cal": 450, "protein": 35, "carbs": 20, "fat": 10, "ingredients": ["salmon","broccoli","olive oil"], "veg": False},
        ],
        "weight_gain": [
            {"name": "Nut Butter Smoothie", "cal": 600, "protein": 25, "carbs": 50, "fat": 25, "ingredients": ["peanut butter","milk","banana"], "veg": True},
            {"name": "Paneer Rice Bowl", "cal": 750, "protein": 35, "carbs": 80, "fat": 15, "ingredients": ["paneer","rice","capsicum","onion"], "veg": True},
            {"name": "Cheese Sandwich", "cal": 650, "protein": 30, "carbs": 70, "fat": 18, "ingredients": ["bread","cheese","butter"], "veg": True},
            {"name": "Beef Pasta Bowl", "cal": 800, "protein": 50, "carbs": 90, "fat": 20, "ingredients": ["beef","pasta","tomato sauce","cheese"], "veg": False},
        ],
        "muscle_gain": [
            {"name": "Egg White Omelette", "cal": 400, "protein": 35, "carbs": 10, "fat": 8, "ingredients": ["egg whites","veggies","olive oil"], "veg": False},
            {"name": "Quinoa & Chickpea Bowl", "cal": 500, "protein": 25, "carbs": 60, "fat": 10, "ingredients": ["quinoa","chickpeas","spinach"], "veg": True},
            {"name": "Tofu Power Bowl", "cal": 550, "protein": 30, "carbs": 50, "fat": 12, "ingredients": ["tofu","rice","veggies"], "veg": True},
            {"name": "Paneer Protein Curry", "cal": 600, "protein": 40, "carbs": 50, "fat": 15, "ingredients": ["paneer","tomato","peas"], "veg": True},
        ]
    }

# -----------------------------
# 3ï¸�âƒ£ Tools
# -----------------------------
def get_recipes(diet, goal, excluded, memory):
    db = recipe_database()
    pool = db.get(goal, [])
    if diet == "vegetarian":
        pool = [r for r in pool if r["veg"]]
    elif diet == "vegan":
        pool = [r for r in pool if r["veg"] and not any(i in r["ingredients"] for i in ["milk","cheese","yogurt","paneer"])]
    # Remove allergens
    pool = [r for r in pool if all(a.lower() not in r["name"].lower() for a in excluded)]

    # Remove meals already in recent history
    prev = memory.get_history()
    fresh = [r for r in pool if r["name"] not in prev]
    if not fresh:
        fresh = pool  # reset if everything used

    # Randomly pick 2 different meals each time
    selected = random.sample(fresh, k=min(2, len(fresh)))
    memory.add_history([m["name"] for m in selected])
    return selected

def calculate_nutrition(plan, goal):
    total = sum(m["cal"] for m in plan)
    protein = sum(m["protein"] for m in plan)
    carbs = sum(m["carbs"] for m in plan)
    fat = sum(m["fat"] for m in plan)
    targets = {
        "bulking": (2600, 3500),
        "weight_loss": (1400, 1900),
        "weight_gain": (2200, 2700),
        "muscle_gain": (2000, 2600)
    }
    low, high = targets.get(goal, (1800, 2500))
    within = low <= total <= high
    status = "âœ… On target!" if within else "âš ï¸� Adjust meal sizes."
    return {
        "total_calories": total,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
        "goal_range": f"{low}-{high}",
        "status": status
    }

def build_grocery_list(plan):
    items = set()
    for meal in plan:
        items.update(meal["ingredients"])
    return sorted(items)

# -----------------------------
# 4ï¸�âƒ£ Agent Reasoning (Dynamic)
# -----------------------------
def nutriplan_agent(diet, goal, allergies):
    allergies = [a.strip() for a in allergies.split(",") if a.strip()]
    memory = SessionMemory()
    memory.save("diet", diet)
    memory.save("goal", goal)
    memory.save("allergies", allergies)

    # Dynamic meal selection
    meals = get_recipes(diet, goal, allergies, memory)
    nutrition = calculate_nutrition(meals, goal)
    grocery = build_grocery_list(meals)

    text = f"### ğŸ¥— {goal.replace('_',' ').title()} Plan ({nutrition['total_calories']} kcal)\n"
    for m in meals:
        text += f"- **{m['name']}** ({m['cal']} kcal, {m['protein']}g P, {m['carbs']}g C, {m['fat']}g F)\n"
    text += f"\n**Macros Summary:** {nutrition['protein']} P / {nutrition['carbs']} C / {nutrition['fat']} F\n"
    text += f"Target Range: {nutrition['goal_range']} kcal â†’ {nutrition['status']}\n\n"
    text += f"### ğŸ›’ Grocery List\n" + ", ".join(grocery)

    return text

# -----------------------------
# 5ï¸�âƒ£ Gradio UI
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ğŸ�‹ï¸� NutriPlan AI â€” Gym Edition (Dynamic Variety)")
    gr.Markdown("ADK-inspired Concierge Agent: Tools Â· Memory Â· Reasoning Â· Context")

    with gr.Row():
        diet = gr.Radio(["vegetarian","vegan","default"], label="ğŸ¥¦ Diet Type", value="vegetarian")
        goal = gr.Dropdown(["bulking","weight_loss","weight_gain","muscle_gain"], label="ğŸ�� Fitness Goal", value="bulking")
    allergies = gr.Textbox(label="Allergies (comma separated)", placeholder="e.g., nuts, shellfish")

    generate_btn = gr.Button("ğŸ�½ï¸� Generate Meal Plan")
    output = gr.Markdown(label="Meal Plan Output")

    generate_btn.click(nutriplan_agent, inputs=[diet, goal, allergies], outputs=[output])

    gr.Markdown("---")
    gr.Markdown("ğŸ’¾ SessionMemory prevents repeats. Run multiple times to see different meal combos!")

demo.launch(share=True, prevent_thread_lock=True)



