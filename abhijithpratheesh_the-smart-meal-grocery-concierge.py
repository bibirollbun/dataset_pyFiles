# Install the Google Generative AI SDK
!pip install -q -U google-generativeai


import os
import json
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List, Dict, Any
from kaggle_secrets import UserSecretsClient

# --- CONFIGURATION ---
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("âœ… Google API Key successfully configured.")
except Exception as e:
    print("âš ï¸� Error: Could not find 'GOOGLE_API_KEY'.")

# Setup the Gemini Model
MODEL_NAME = "gemini-2.5-flash-lite" 
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}


class GroceryDatabaseTool:
    """
    A simulated tool that agents use to find the price and aisle of ingredients.
    In a real-world scenario, this would connect to a Walmart/Instacart API.
    """
    def __init__(self):
        # Simulated database of items
        self.db = {
            "chicken": {"price": 5.99, "aisle": "Meat"},
            "rice": {"price": 2.50, "aisle": "Grains"},
            "spinach": {"price": 1.99, "aisle": "Produce"},
            "tofu": {"price": 3.00, "aisle": "Vegetarian"},
            "milk": {"price": 3.50, "aisle": "Dairy"},
            "eggs": {"price": 4.00, "aisle": "Dairy"},
            "bread": {"price": 2.99, "aisle": "Bakery"},
            "apple": {"price": 0.99, "aisle": "Produce"},
        }

    def search(self, item_name: str) -> Dict:
        """Looks up an item in the database."""
        item_name = item_name.lower().strip()
        # Simple fuzzy match
        for key in self.db:
            if key in item_name or item_name in key:
                return {"item": key, **self.db[key]}
        return {"item": item_name, "price": "Unknown", "aisle": "General"}

# Initialize the tool
grocery_tool = GroceryDatabaseTool()
print("âœ… Grocery Tool Initialized")


@dataclass
class Memory:
    history: List[Dict] = field(default_factory=list)
    user_profile: Dict = field(default_factory=lambda: {"diet": [], "allergies": []})

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
    
    def update_profile(self, key: str, value: Any):
        if key in self.user_profile:
            if isinstance(self.user_profile[key], list):
                self.user_profile[key].append(value)
            else:
                self.user_profile[key] = value

    def get_context(self) -> str:
        """Compiles history and profile into a context string for the LLM."""
        context = f"User Profile: {json.dumps(self.user_profile)}\n"
        context += "Conversation History:\n"
        for msg in self.history[-5:]: # Keep last 5 turns for context window
            context += f"{msg['role']}: {msg['content']}\n"
        return context


class BaseAgent:
    def __init__(self, name, model_name=MODEL_NAME):
        self.name = name
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )

    def generate(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            print(f"â�Œ Error in {self.name}: {e}")
            return {"error": str(e)}

class MealPlannerAgent(BaseAgent):
    """Agent 1: Decides WHAT to eat based on user request."""
    def plan(self, user_request, memory_context):
        prompt = f"""
        You are a professional Chef and Nutritionist.
        Context: {memory_context}
        User Request: {user_request}
        
        Task: Create a simple meal plan (max 3 meals) based on the request and context.
        Output JSON format: {{ "meals": [ {{ "name": "Meal Name", "ingredients": ["item1", "item2"] }} ] }}
        """
        return self.generate(prompt)

class ShoppingAgent(BaseAgent):
    """Agent 2: Takes the plan and creates a shopping list using the Tool."""
    def shop(self, meal_plan_json):
        # 1. Extract ingredients
        ingredients = []
        for meal in meal_plan_json.get("meals", []):
            ingredients.extend(meal.get("ingredients", []))
        
        # 2. USE THE TOOL (Agent calls the tool logic)
        final_list = []
        total_est_cost = 0.0
        
        for item in set(ingredients): # Remove duplicates
            # The Agent "calls" the tool here
            data = grocery_tool.search(item)
            final_list.append(data)
            if isinstance(data['price'], float):
                total_est_cost += data['price']
        
        return {
            "shopping_list": final_list,
            "estimated_total": round(total_est_cost, 2),
            "currency": "USD"
        }


class ConciergeSystem:
    def __init__(self):
        self.memory = Memory()
        self.planner = MealPlannerAgent("Chef")
        self.shopper = ShoppingAgent("Shopper")

    def process_request(self, user_input):
        print(f"\nğŸ”¹ USER: {user_input}")
        self.memory.add_message("user", user_input)

        # 1. Check if we need to update memory (Simple keyword check for demo)
        if "vegan" in user_input.lower():
            self.memory.update_profile("diet", "vegan")
            print("   (ğŸ“� Memory Updated: User is Vegan)")
        if "allergic" in user_input.lower():
            self.memory.update_profile("allergies", user_input.split("allergic to")[-1].strip())

        # 2. Step 1: Meal Planner Agent
        print("   (ğŸ‘¨â€�ğŸ�³ Chef Agent is thinking...)")
        context = self.memory.get_context()
        meal_plan = self.planner.plan(user_input, context)
        print(f"   (ğŸ’¡ Proposed Plan: {len(meal_plan.get('meals', []))} meals)")

        # 3. Step 2: Shopping Agent
        print("   (ğŸ›’ Shopper Agent is calculating prices...)")
        shopping_list = self.shopper.shop(meal_plan)

        # 4. Final Output Generation
        final_response = {
            "meal_plan": meal_plan,
            "shopping_list": shopping_list
        }
        
        # Save to memory
        self.memory.add_message("assistant", str(final_response))
        return final_response


# Initialize the System
bot = ConciergeSystem()

# --- TEST SCENARIO 1: Simple Request ---
response1 = bot.process_request("I need a dinner plan for tonight involving chicken and rice.")
print("\nğŸ“‹ --- RESULT 1 ---")
print(json.dumps(response1, indent=2))

# --- TEST SCENARIO 2: Context Awareness (Memory) ---
# The bot should remember the history and context
response2 = bot.process_request("Actually, swap the chicken for tofu. I'm cooking for a vegetarian friend.")
print("\nğŸ“‹ --- RESULT 2 ---")
print(json.dumps(response2, indent=2))

