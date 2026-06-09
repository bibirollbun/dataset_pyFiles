# ==========================
# Ultimate Interactive Meal Planner
# ==========================
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output

# ==========================
# Days of week
# ==========================
days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ==========================
# Meals dataset
# ==========================
meal_options = {
    "Veg Biryani": {"ingredients":["Rice","Vegetables","Spices","Oil","Peanuts"], "category":"Lunch/Dinner", "calories":600, "protein":15, "carbs":90, "fat":20, "tags":["Vegetarian"]},
    "Idli + Sambar": {"ingredients":["Rice","Urad Dal","Lentils","Vegetables","Spices"], "category":"Breakfast", "calories":350, "protein":10, "carbs":60, "fat":5, "tags":["Vegetarian","Gluten-Free"]},
    "Curd Rice": {"ingredients":["Rice","Curd","Salt","Spices"], "category":"Lunch/Dinner", "calories":400, "protein":12, "carbs":70, "fat":8, "tags":["Vegetarian","Gluten-Free"]},
    "Oats Porridge": {"ingredients":["Oats","Milk","Sugar"], "category":"Breakfast", "calories":250, "protein":8, "carbs":45, "fat":4, "tags":["Vegetarian"]},
    "Paneer Butter Masala": {"ingredients":["Paneer","Butter","Tomato","Spices"], "category":"Lunch/Dinner", "calories":500, "protein":20, "carbs":25, "fat":30, "tags":["Vegetarian","Gluten-Free"]},
    "Mixed Fruit Salad": {"ingredients":["Apple","Banana","Orange","Grapes"], "category":"Snack", "calories":150, "protein":2, "carbs":35, "fat":0, "tags":["Vegan","Gluten-Free"]},
    "Chole Bhature": {"ingredients":["Chickpeas","Flour","Oil","Spices"], "category":"Breakfast/Lunch", "calories":550, "protein":18, "carbs":75, "fat":20, "tags":["Vegetarian"]},
    "Protein Smoothie": {"ingredients":["Milk","Whey Protein","Banana"], "category":"Breakfast/Snack", "calories":200, "protein":20, "carbs":25, "fat":2, "tags":["Gluten-Free"]},
    "Grilled Chicken Salad": {"ingredients":["Chicken","Lettuce","Tomato","Olive Oil"], "category":"Lunch/Dinner", "calories":350, "protein":30, "carbs":10, "fat":15, "tags":["Gluten-Free"]},
    "Veg Sandwich": {"ingredients":["Bread","Vegetables","Cheese"], "category":"Snack", "calories":300, "protein":12, "carbs":40, "fat":10, "tags":["Vegetarian"]}
}

# ==========================
# User settings
# ==========================
user_restrictions = ["peanuts"]  # Allergies/dislikes
user_dislikes = ["Idli + Sambar"]  # meals user doesn't want
user_tags = ["Vegetarian","Gluten-Free"]  # preferred diet

# ==========================
# Sliders for daily limits
# ==========================
cal_slider = widgets.IntSlider(value=2200, min=1000, max=4000, step=50, description='Calories')
protein_slider = widgets.IntSlider(value=150, min=50, max=400, step=5, description='Protein')
carbs_slider = widgets.IntSlider(value=300, min=50, max=500, step=5, description='Carbs')
fat_slider = widgets.IntSlider(value=70, min=10, max=150, step=2, description='Fat')
display(widgets.HBox([cal_slider, protein_slider, carbs_slider, fat_slider]))

# ==========================
# Get allowed meals based on preferences
# ==========================
def get_allowed_meals():
    allowed = []
    for meal, info in meal_options.items():
        if meal in user_dislikes: continue
        if any(ing.lower() in user_restrictions for ing in info["ingredients"]): continue
        if not any(tag in info["tags"] for tag in user_tags): continue
        allowed.append(meal)
    return allowed

dropdowns = {day: widgets.Dropdown(options=get_allowed_meals(), description=day+":") for day in days}
display(widgets.VBox(list(dropdowns.values())))

# ==========================
# Generate weekly plan
# ==========================
def generate_plan(selected_meals):
    plan = {day:selected_meals[day] for day in days}
    shopping_list = defaultdict(int)
    weekly_macros = {"calories":0,"protein":0,"carbs":0,"fat":0}
    for meal in plan.values():
        for ingredient in meal_options[meal]["ingredients"]:
            shopping_list[ingredient] += 1
        for macro in weekly_macros:
            weekly_macros[macro] += meal_options[meal][macro]
    return plan, shopping_list, weekly_macros

def check_limits(plan):
    daily_nutrition = {}
    for day in plan:
        meal = plan[day]
        daily_nutrition[day] = meal_options[meal]
    return daily_nutrition

# ==========================
# Plot nutrition
# ==========================
def plot_macros(daily_nutrition):
    df = pd.DataFrame(daily_nutrition).T
    sns.set(style="whitegrid")
    df_plot = df[["calories","protein","carbs","fat"]]
    df_plot.plot(kind="bar", figsize=(12,6), colormap='Set2')
    plt.title("Daily Nutrition Overview")
    plt.ylabel("Amount")
    plt.xticks(rotation=45)
    plt.show()
    for day,row in df.iterrows():
        plt.figure(figsize=(4,4))
        plt.pie([row["protein"], row["carbs"], row["fat"]],
                labels=["Protein","Carbs","Fat"], autopct='%1.1f%%', startangle=140,
                colors=['#ff9999','#66b3ff','#99ff99'])
        plt.title(f"Macro distribution: {day}")
        plt.show()

# ==========================
# Output widgets
# ==========================
plan_output = widgets.Output()
shopping_output = widgets.Output()
macro_output = widgets.Output()
suggestion_output = widgets.Output()

# ==========================
# Buttons
# ==========================
manual_button = widgets.Button(description="Manual Plan")
random_button = widgets.Button(description="Random Plan")
optimized_button = widgets.Button(description="Macro-Optimized Plan")
display(manual_button, random_button, optimized_button, plan_output, shopping_output, macro_output, suggestion_output)

# ==========================
# Display results & export CSV
# ==========================
def display_results(plan, shopping_list, weekly_macros, daily_nutrition):
    with plan_output:
        clear_output()
        print("=== Weekly Meal Plan ===")
        for day, meal in plan.items():
            info = meal_options[meal]
            warning = ""
            limits = {"calories":cal_slider.value,"protein":protein_slider.value,
                      "carbs":carbs_slider.value,"fat":fat_slider.value}
            for macro in limits:
                if info[macro] > limits[macro]: warning += f"{macro.upper()} ⚠️ "
            warning = warning if warning else "✅ Within limits"
            print(f"{day}: {meal} (Calories: {info['calories']}, Protein: {info['protein']}g, Carbs: {info['carbs']}g, Fat: {info['fat']}g) {warning}")
    
    with shopping_output:
        clear_output()
        print("=== Shopping List ===")
        for ing, qty in shopping_list.items():
            print(f"{ing}: {qty}")
    
    with macro_output:
        clear_output()
        print("=== Weekly Macros ===")
        print(f"Calories: {weekly_macros['calories']} kcal | Protein: {weekly_macros['protein']}g | Carbs: {weekly_macros['carbs']}g | Fat: {weekly_macros['fat']}g")
        plot_macros(daily_nutrition)
    
    with suggestion_output:
        clear_output()
        print("=== AI Meal Substitutions ===")
        for day, meal in plan.items():
            alternatives = [m for m in get_allowed_meals() if m!=meal and meal_options[m]["category"]==meal_options[meal]["category"]]
            suggestion = random.choice(alternatives) if alternatives else "None"
            print(f"{day} alternative: {suggestion}")
    
    # Export CSV to /kaggle/working
    plan_df = pd.DataFrame.from_dict(plan, orient='index', columns=['Meal'])
    shopping_df = pd.DataFrame(list(shopping_list.items()), columns=['Ingredient','Qty'])
    plan_df.to_csv("/kaggle/working/weekly_plan.csv", index=False)
    shopping_df.to_csv("/kaggle/working/shopping_list.csv", index=False)
    print("✅ Weekly plan & shopping list exported to /kaggle/working/")

# ==========================
# Button actions
# ==========================
def on_manual_click(b):
    selected = {day:dropdowns[day].value for day in days}
    plan, shop, macros = generate_plan(selected)
    daily = check_limits(plan)
    display_results(plan, shop, macros, daily)

def on_random_click(b):
    selected = {day:random.choice(get_allowed_meals()) for day in days}
    for day in days: dropdowns[day].value = selected[day]
    plan, shop, macros = generate_plan(selected)
    daily = check_limits(plan)
    display_results(plan, shop, macros, daily)

def on_optimized_click(b):
    selected = {}
    limits = {"calories":cal_slider.value,"protein":protein_slider.value,"carbs":carbs_slider.value,"fat":fat_slider.value}
    for day in days:
        best = min(get_allowed_meals(), key=lambda m: sum(max(0, meal_options[m][macro]-limits[macro]) for macro in limits))
        selected[day] = best
    for day in days: dropdowns[day].value = selected[day]
    plan, shop, macros = generate_plan(selected)
    daily = check_limits(plan)
    display_results(plan, shop, macros, daily)

manual_button.on_click(on_manual_click)
random_button.on_click(on_random_click)
optimized_button.on_click(on_optimized_click)

