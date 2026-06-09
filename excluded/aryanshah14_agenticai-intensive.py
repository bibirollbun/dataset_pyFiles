!pip install google-generativeai anthropic langchain langchain-google-genai python-dotenv pandas numpy matplotlib seaborn plotly -q


import google.generativeai as genai
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("âœ… All libraries imported successfully!")


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("api_key")


genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

print("âœ… API configured successfully!")


class UserProfileManager:
    def __init__(self):
        self.profile = {
            'name': '',
            'dietary_restrictions': [],
            'allergies': [],
            'cuisine_preferences': [],
            'daily_calorie_target': 2000,
            'daily_protein_target': 50,
            'daily_carbs_target': 250,
            'daily_fat_target': 70,
            'weekly_budget': 100,
            'household_size': 1,
            'cooking_skill': 'intermediate',
            'available_time': 'moderate',
            'kitchen_equipment': ['stove', 'oven', 'microwave']
        }
    
    def create_profile(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.profile:
                self.profile[key] = value
        return self.profile
    
    def get_profile(self):
        return self.profile
    
    def display_profile(self):
        print("\n" + "="*60)
        print("ğŸ‘¤ USER PROFILE")
        print("="*60)
        for key, value in self.profile.items():
            key_formatted = key.replace('_', ' ').title()
            print(f"  {key_formatted:.<30} {value}")
        print("="*60 + "\n")


class RecipeDiscoveryAgent:
    def __init__(self, model):
        self.model = model
    
    def generate_recipes(self, user_profile: Dict, num_recipes: int = 21, meal_types: List[str] = None) -> List[Dict]:
        if meal_types is None:
            meal_types = ['breakfast', 'lunch', 'dinner']
        
        recipes_per_type = num_recipes // len(meal_types)
        all_recipes = []
        
        for meal_type in meal_types:
            prompt = f"""
Generate {recipes_per_type} diverse {meal_type} recipes in JSON format with the following constraints:

USER PROFILE:
- Dietary Restrictions: {user_profile['dietary_restrictions']}
- Allergies: {user_profile['allergies']}
- Cuisine Preferences: {user_profile['cuisine_preferences']}
- Cooking Skill: {user_profile['cooking_skill']}
- Available Time: {user_profile['available_time']}
- Daily Calorie Target: {user_profile['daily_calorie_target']} cal
- Household Size: {user_profile['household_size']} person(s)

For each recipe, provide:
{{
  "name": "Recipe name",
  "meal_type": "{meal_type}",
  "cuisine": "Cuisine type",
  "prep_time": <minutes as integer>,
  "cook_time": <minutes as integer>,
  "servings": {user_profile['household_size']},
  "difficulty": "easy/medium/hard",
  "ingredients": [
    {{"item": "ingredient name", "quantity": "amount", "estimated_cost": <price in dollars>}}
  ],
  "instructions": ["step 1", "step 2", ...],
  "nutrition": {{
    "calories": <number>,
    "protein": <grams>,
    "carbs": <grams>,
    "fat": <grams>,
    "fiber": <grams>
  }},
  "tags": ["quick", "healthy", etc.]
}}

Return ONLY a JSON array of recipes, no additional text.
"""
            
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                recipes = json.loads(response_text.strip())
                all_recipes.extend(recipes)
                
            except Exception as e:
                print(f"âš ï¸� Error generating {meal_type} recipes: {e}")
                continue
        
        return all_recipes
    
    def score_recipe(self, recipe: Dict, user_profile: Dict) -> float:
        score = 0.0
        
        calorie_diff = abs(recipe['nutrition']['calories'] - (user_profile['daily_calorie_target'] / 3))
        calorie_score = max(0, 1 - (calorie_diff / user_profile['daily_calorie_target']))
        score += calorie_score * 0.3
        
        total_cost = sum(ing['estimated_cost'] for ing in recipe['ingredients'])
        if total_cost <= (user_profile['weekly_budget'] / 21):
            score += 0.3
        else:
            score += 0.3 * ((user_profile['weekly_budget'] / 21) / total_cost)
        
        total_time = recipe['prep_time'] + recipe['cook_time']
        if user_profile['available_time'] == 'limited' and total_time <= 30:
            score += 0.2
        elif user_profile['available_time'] == 'moderate' and total_time <= 60:
            score += 0.2
        else:
            score += 0.1
        
        if recipe['cuisine'].lower() in [c.lower() for c in user_profile['cuisine_preferences']]:
            score += 0.2
        
        return min(score, 1.0)


class NutritionAnalyzer:
    def __init__(self):
        pass
    
    def analyze_recipe(self, recipe: Dict) -> Dict:
        nutrition = recipe['nutrition']
        analysis = {
            'total_calories': nutrition['calories'],
            'macros': {
                'protein': nutrition['protein'],
                'carbs': nutrition['carbs'],
                'fat': nutrition['fat']
            },
            'fiber': nutrition.get('fiber', 0)
        }
        return analysis
    
    def analyze_meal_plan(self, meal_plan: List[Dict]) -> Dict:
        total_nutrition = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0
        }
        
        for meal in meal_plan:
            total_nutrition['calories'] += meal['nutrition']['calories']
            total_nutrition['protein'] += meal['nutrition']['protein']
            total_nutrition['carbs'] += meal['nutrition']['carbs']
            total_nutrition['fat'] += meal['nutrition']['fat']
            total_nutrition['fiber'] += meal['nutrition'].get('fiber', 0)
        
        return total_nutrition
    
    def check_balance(self, daily_nutrition: Dict, targets: Dict) -> Dict:
        balance = {}
        balance['calories'] = (daily_nutrition['calories'] / targets['daily_calorie_target']) * 100
        balance['protein'] = (daily_nutrition['protein'] / targets['daily_protein_target']) * 100
        balance['carbs'] = (daily_nutrition['carbs'] / targets['daily_carbs_target']) * 100
        balance['fat'] = (daily_nutrition['fat'] / targets['daily_fat_target']) * 100
        
        return balance


class BudgetOptimizer:
    def __init__(self):
        pass
    
    def calculate_recipe_cost(self, recipe: Dict) -> float:
        return sum(ing['estimated_cost'] for ing in recipe['ingredients'])
    
    def calculate_plan_cost(self, meal_plan: List[Dict]) -> float:
        return sum(self.calculate_recipe_cost(meal) for meal in meal_plan)
    
    def optimize_plan(self, recipes: List[Dict], budget: float, target_meals: int) -> List[Dict]:
        sorted_recipes = sorted(recipes, key=lambda x: self.calculate_recipe_cost(x))
        
        selected_recipes = []
        current_cost = 0
        meal_type_count = {'breakfast': 0, 'lunch': 0, 'dinner': 0}
        target_per_type = target_meals // 3
        
        for recipe in sorted_recipes:
            recipe_cost = self.calculate_recipe_cost(recipe)
            meal_type = recipe['meal_type']
            
            if (current_cost + recipe_cost <= budget and 
                meal_type_count[meal_type] < target_per_type):
                selected_recipes.append(recipe)
                current_cost += recipe_cost
                meal_type_count[meal_type] += 1
                
                if len(selected_recipes) >= target_meals:
                    break
        
        return selected_recipes


class ShoppingListGenerator:
    def __init__(self):
        pass
    
    def generate_shopping_list(self, meal_plan: List[Dict]) -> Dict:
        shopping_list = {}
        
        for meal in meal_plan:
            for ingredient in meal['ingredients']:
                item = ingredient['item']
                if item not in shopping_list:
                    shopping_list[item] = {
                        'quantity': ingredient['quantity'],
                        'estimated_cost': ingredient['estimated_cost'],
                        'used_in': [meal['name']]
                    }
                else:
                    shopping_list[item]['used_in'].append(meal['name'])
        
        return shopping_list
    
    def display_shopping_list(self, shopping_list: Dict) -> pd.DataFrame:
        data = []
        for item, details in shopping_list.items():
            data.append({
                'Item': item,
                'Quantity': details['quantity'],
                'Est. Cost ($)': f"${details['estimated_cost']:.2f}",
                'Used In': len(details['used_in']),
                'Recipes': ', '.join(details['used_in'][:2]) + ('...' if len(details['used_in']) > 2 else '')
            })
        
        df = pd.DataFrame(data)
        return df


class MealScheduler:
    def __init__(self):
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def create_weekly_schedule(self, recipes: List[Dict]) -> Dict:
        schedule = {day: {'breakfast': None, 'lunch': None, 'dinner': None} for day in self.days}
        
        breakfast_recipes = [r for r in recipes if r['meal_type'] == 'breakfast']
        lunch_recipes = [r for r in recipes if r['meal_type'] == 'lunch']
        dinner_recipes = [r for r in recipes if r['meal_type'] == 'dinner']
        
        for i, day in enumerate(self.days):
            if i < len(breakfast_recipes):
                schedule[day]['breakfast'] = breakfast_recipes[i]
            if i < len(lunch_recipes):
                schedule[day]['lunch'] = lunch_recipes[i]
            if i < len(dinner_recipes):
                schedule[day]['dinner'] = dinner_recipes[i]
        
        return schedule
    
    def display_schedule(self, schedule: Dict) -> pd.DataFrame:
        data = []
        for day, meals in schedule.items():
            row = {'Day': day}
            for meal_type, recipe in meals.items():
                if recipe:
                    row[meal_type.capitalize()] = f"{recipe['name']} ({recipe['nutrition']['calories']} cal)"
                else:
                    row[meal_type.capitalize()] = 'Not assigned'
            data.append(row)
        
        return pd.DataFrame(data)


class SmartMealPlannerAgent:
    def __init__(self, model):
        self.profile_manager = UserProfileManager()
        self.recipe_agent = RecipeDiscoveryAgent(model)
        self.nutrition_analyzer = NutritionAnalyzer()
        self.budget_optimizer = BudgetOptimizer()
        self.shopping_list_gen = ShoppingListGenerator()
        self.meal_scheduler = MealScheduler()
        
        self.generated_recipes = []
        self.meal_plan = []
        self.weekly_schedule = {}
        self.shopping_list = {}
    
    def setup_user_profile(self, **kwargs):
        print("\nğŸ”§ Setting up user profile...")
        profile = self.profile_manager.create_profile(**kwargs)
        self.profile_manager.display_profile()
        return profile
    
    def generate_meal_plan(self, num_recipes: int = 21):
        print("\nğŸ”� Generating recipes based on your preferences...")
        user_profile = self.profile_manager.get_profile()
        
        self.generated_recipes = self.recipe_agent.generate_recipes(user_profile, num_recipes)
        print(f"âœ… Generated {len(self.generated_recipes)} recipes!")
        
        print("\nğŸ’° Optimizing meal plan within budget...")
        self.meal_plan = self.budget_optimizer.optimize_plan(
            self.generated_recipes, 
            user_profile['weekly_budget'], 
            21
        )
        print(f"âœ… Selected {len(self.meal_plan)} meals within budget!")
        
        print("\nğŸ“… Creating weekly schedule...")
        self.weekly_schedule = self.meal_scheduler.create_weekly_schedule(self.meal_plan)
        print("âœ… Weekly schedule created!")
        
        print("\nğŸ›’ Generating shopping list...")
        self.shopping_list = self.shopping_list_gen.generate_shopping_list(self.meal_plan)
        print("âœ… Shopping list generated!")
        
        return {
            'recipes': self.generated_recipes,
            'meal_plan': self.meal_plan,
            'schedule': self.weekly_schedule,
            'shopping_list': self.shopping_list
        }
    
    def display_results(self):
        print("\n" + "="*80)
        print("ğŸ“Š MEAL PLAN SUMMARY")
        print("="*80)
        
        total_cost = self.budget_optimizer.calculate_plan_cost(self.meal_plan)
        user_profile = self.profile_manager.get_profile()
        budget = user_profile['weekly_budget']
        
        print(f"\nğŸ’µ Budget Status:")
        print(f"  Weekly Budget: ${budget:.2f}")
        print(f"  Total Cost: ${total_cost:.2f}")
        print(f"  Remaining: ${budget - total_cost:.2f}")
        print(f"  Budget Utilization: {(total_cost/budget)*100:.1f}%")
        
        weekly_nutrition = self.nutrition_analyzer.analyze_meal_plan(self.meal_plan)
        daily_avg = {k: v/7 for k, v in weekly_nutrition.items()}
        
        print(f"\nğŸ¥— Daily Average Nutrition:")
        print(f"  Calories: {daily_avg['calories']:.0f} kcal")
        print(f"  Protein: {daily_avg['protein']:.1f}g")
        print(f"  Carbs: {daily_avg['carbs']:.1f}g")
        print(f"  Fat: {daily_avg['fat']:.1f}g")
        print(f"  Fiber: {daily_avg['fiber']:.1f}g")
        
        print(f"\nğŸ�½ï¸� Meal Distribution:")
        meal_counts = {'breakfast': 0, 'lunch': 0, 'dinner': 0}
        for meal in self.meal_plan:
            meal_counts[meal['meal_type']] += 1
        for meal_type, count in meal_counts.items():
            print(f"  {meal_type.capitalize()}: {count} meals")
        
        print("\n" + "="*80)


class MealPlanVisualizer:
    def __init__(self):
        pass
    
    def plot_nutrition_breakdown(self, meal_plan: List[Dict], user_profile: Dict):
        daily_nutrition = {'calories': [], 'protein': [], 'carbs': [], 'fat': []}
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for i in range(7):
            day_meals = [m for m in meal_plan[i*3:(i+1)*3]]
            day_total = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
            for meal in day_meals:
                day_total['calories'] += meal['nutrition']['calories']
                day_total['protein'] += meal['nutrition']['protein']
                day_total['carbs'] += meal['nutrition']['carbs']
                day_total['fat'] += meal['nutrition']['fat']
            
            for key in daily_nutrition:
                daily_nutrition[key].append(day_total[key])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Calories', 'Daily Protein', 'Daily Carbs', 'Daily Fat'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        fig.add_trace(
            go.Bar(x=days, y=daily_nutrition['calories'], name='Calories', marker_color='indianred'),
            row=1, col=1
        )
        fig.add_hline(y=user_profile['daily_calorie_target'], line_dash="dash", 
                     line_color="red", annotation_text="Target", row=1, col=1)
        
        fig.add_trace(
            go.Bar(x=days, y=daily_nutrition['protein'], name='Protein', marker_color='lightblue'),
            row=1, col=2
        )
        fig.add_hline(y=user_profile['daily_protein_target'], line_dash="dash",
                     line_color="blue", annotation_text="Target", row=1, col=2)
        
        fig.add_trace(
            go.Bar(x=days, y=daily_nutrition['carbs'], name='Carbs', marker_color='lightgreen'),
            row=2, col=1
        )
        fig.add_hline(y=user_profile['daily_carbs_target'], line_dash="dash",
                     line_color="green", annotation_text="Target", row=2, col=1)
        
        fig.add_trace(
            go.Bar(x=days, y=daily_nutrition['fat'], name='Fat', marker_color='lightyellow'),
            row=2, col=2
        )
        fig.add_hline(y=user_profile['daily_fat_target'], line_dash="dash",
                     line_color="orange", annotation_text="Target", row=2, col=2)
        
        fig.update_layout(height=700, showlegend=False, title_text="Weekly Nutrition Tracking")
        fig.show()
    
    def plot_budget_breakdown(self, meal_plan: List[Dict], budget: float):
        meal_costs = []
        meal_names = []
        
        for meal in meal_plan:
            cost = sum(ing['estimated_cost'] for ing in meal['ingredients'])
            meal_costs.append(cost)
            meal_names.append(meal['name'][:20])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=meal_names,
            y=meal_costs,
            marker_color='lightseagreen',
            text=[f'${c:.2f}' for c in meal_costs],
            textposition='auto',
        ))
        
        avg_cost = sum(meal_costs) / len(meal_costs)
        fig.add_hline(y=avg_cost, line_dash="dash", line_color="red",
                     annotation_text=f"Avg: ${avg_cost:.2f}")
        
        fig.update_layout(
            title="Cost per Meal",
            xaxis_title="Meals",
            yaxis_title="Cost ($)",
            height=500,
            xaxis_tickangle=-45
        )
        fig.show()
    
    def plot_macro_distribution(self, meal_plan: List[Dict]):
        total_nutrition = {'Protein': 0, 'Carbs': 0, 'Fat': 0}
        
        for meal in meal_plan:
            total_nutrition['Protein'] += meal['nutrition']['protein']
            total_nutrition['Carbs'] += meal['nutrition']['carbs']
            total_nutrition['Fat'] += meal['nutrition']['fat']
        
        fig = go.Figure(data=[go.Pie(
            labels=list(total_nutrition.keys()),
            values=list(total_nutrition.values()),
            hole=.3,
            marker_colors=['lightblue', 'lightgreen', 'lightyellow']
        )])
        
        fig.update_layout(title="Weekly Macronutrient Distribution")
        fig.show()


agent = SmartMealPlannerAgent(model)

print("\n" + "="*80)
print("ğŸ�½ï¸� SMART MEAL PLANNER AGENT - DEMO")
print("="*80)


user_profile = agent.setup_user_profile(
    name='Sarah',
    dietary_restrictions=['vegetarian'],
    allergies=['peanuts'],
    cuisine_preferences=['Italian', 'Mexican', 'Asian'],
    daily_calorie_target=2000,
    daily_protein_target=60,
    daily_carbs_target=250,
    daily_fat_target=65,
    weekly_budget=120,
    household_size=1,
    cooking_skill='intermediate',
    available_time='moderate'
)


results = agent.generate_meal_plan(num_recipes=21)


agent.display_results()


schedule_df = agent.meal_scheduler.display_schedule(agent.weekly_schedule)
print("\nğŸ“… WEEKLY MEAL SCHEDULE")
print("="*120)
display(schedule_df)


shopping_df = agent.shopping_list_gen.display_shopping_list(agent.shopping_list)
print("\nğŸ›’ SHOPPING LIST")
print("="*100)
display(shopping_df)

total_shopping_cost = sum([details['estimated_cost'] for details in agent.shopping_list.values()])
print(f"\nğŸ’° Total Shopping Cost: ${total_shopping_cost:.2f}")


if agent.meal_plan:
    sample_recipe = agent.meal_plan[0]
    
    print("\n" + "="*80)
    print(f"ğŸ“– SAMPLE RECIPE: {sample_recipe['name']}")
    print("="*80)
    print(f"\nğŸ�½ï¸� Type: {sample_recipe['meal_type'].capitalize()}")
    print(f"ğŸŒ� Cuisine: {sample_recipe['cuisine']}")
    print(f"â�±ï¸� Prep Time: {sample_recipe['prep_time']} min | Cook Time: {sample_recipe['cook_time']} min")
    print(f"ğŸ‘¥ Servings: {sample_recipe['servings']}")
    print(f"ğŸ“Š Difficulty: {sample_recipe['difficulty'].capitalize()}")
    
    print(f"\nğŸ¥— Nutrition (per serving):")
    for key, value in sample_recipe['nutrition'].items():
        print(f"  â€¢ {key.capitalize()}: {value}{'g' if key != 'calories' else ' kcal'}")
    
    print(f"\nğŸ›’ Ingredients:")
    for ing in sample_recipe['ingredients']:
        print(f"  â€¢ {ing['quantity']} {ing['item']} (${ing['estimated_cost']:.2f})")
    
    print(f"\nğŸ‘¨â€�ğŸ�³ Instructions:")
    for i, step in enumerate(sample_recipe['instructions'], 1):
        print(f"  {i}. {step}")
    
    print("\n" + "="*80)


visualizer = MealPlanVisualizer()

print("\nğŸ“ˆ Generating analytics visualizations...\n")


visualizer.plot_nutrition_breakdown(agent.meal_plan, user_profile)


visualizer.plot_budget_breakdown(agent.meal_plan, user_profile['weekly_budget'])


visualizer.plot_macro_distribution(agent.meal_plan)


export_data = {
    'user_profile': user_profile,
    'meal_plan': agent.meal_plan,
    'weekly_schedule': agent.weekly_schedule,
    'shopping_list': agent.shopping_list,
    'generated_at': datetime.now().isoformat()
}

with open('meal_plan_export.json', 'w') as f:
    json.dump(export_data, f, indent=2)

print("\nâœ… Meal plan exported to 'meal_plan_export.json'")
print("ğŸ“¥ You can download this file to use with meal planning apps or share with family!")

