# Install the Google Generative AI SDK
!pip install -q -U google-generativeai

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# --- KAGGLE SECRETS SETUP ---
user_secrets = UserSecretsClient()
# Retrieving the key as requested
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")

# Configure the SDK with your API key
genai.configure(api_key=secret_value_0)

# Setup the model (using Flash for speed and efficiency)
MODEL_ID = "gemini-2.5-flash"

print("âœ… Setup complete. API Key configured.")


class Agent:
    def __init__(self, name, model_id, system_instruction):
        self.name = name
        self.model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system_instruction
        )

    def generate(self, prompt):
        # Generate content based on the prompt
        response = self.model.generate_content(prompt)
        return response.text


# --- 1. The Chef Agent ---
chef_instruction = """
You are a creative Chef.
Take a list of ingredients and generate a DRAFT recipe.
Include: Name, Ingredients, and Instructions.
"""
chef_agent = Agent("Chef", MODEL_ID, chef_instruction)

# --- 2. The Critique Agent (NEW) ---
# Role: Quality Control & Safety
critique_instruction = """
You are a Senior Executive Chef and Editor.
Your input will be a 'Draft Recipe'.
Your Goal: Refine and Polish the recipe.
1. Check for clarity: Are the steps easy to follow?
2. Check for completeness: Are quantities listed?
3. Safety: Are cooking temperatures mentioned?
4. Output the FINAL improved recipe. 
   - If the draft is good, output it as is.
   - If it needs work, rewrite it.
   - DO NOT include conversational filler like "Here is the fixed recipe". Just output the recipe.
"""
critique_agent = Agent("Critic", MODEL_ID, critique_instruction)

# --- 3. The Grocery Agent ---
grocery_instruction = """
You are a helpful Grocery Assistant.
Your input will be a FINAL recipe.
Extract a 'Shopping List' from that recipe.
- Ignore basic pantry items (salt, pepper, water, oil).
- Group items by aisle.
"""
grocery_agent = Agent("Grocery", MODEL_ID, grocery_instruction)

# --- 4. The Nutrition Agent ---
nutrition_instruction = """
You are a Nutritionist.
Your input will be a FINAL recipe.
Provide:
- Estimated Calories per serving.
- Macronutrient breakdown.
- A one-sentence health tip.
"""
nutrition_agent = Agent("Nutritionist", MODEL_ID, nutrition_instruction)

print("âœ… All 4 Agents Initialized (Chef -> Critic -> Grocery -> Nutrition)")


def run_meal_planner_flow(user_ingredients):
    print(f"ğŸ”¹ INPUT: {user_ingredients}")
    print("=" * 60)

    # --- STEP 1: Chef Agent (Drafting) ---
    print("ğŸ‘¨â€�ğŸ�³ Chef Agent is drafting a recipe...")
    draft_recipe = chef_agent.generate(f"Create a recipe using these ingredients: {user_ingredients}")
    print("...Draft complete.")
    
    # --- STEP 2: Critique Agent (Refining) ---
    print("ğŸ§� Critic Agent is reviewing and polishing...")
    # The Critic takes the DRAFT as input
    final_recipe = critique_agent.generate(f"Review and refine this recipe:\n{draft_recipe}")
    
    print(f"\nğŸ“„ FINAL RECIPE (Approved by Critic):\n{final_recipe}")
    print("-" * 60)

    # --- STEP 3: Grocery Agent ---
    # Critical: Grocery Agent now uses the FINAL_RECIPE, not the draft
    print("ğŸ›’ Grocery Agent is making the shopping list...")
    shopping_list_output = grocery_agent.generate(f"Create a shopping list for this recipe:\n{final_recipe}")
    print(f"\n{shopping_list_output}")
    print("-" * 60)

    # --- STEP 4: Nutrition Agent ---
    # Nutrition Agent also uses the FINAL_RECIPE
    print("ğŸ�� Nutrition Agent is analyzing the meal...")
    nutrition_output = nutrition_agent.generate(f"Analyze the nutrition for this recipe:\n{final_recipe}")
    print(f"\n{nutrition_output}")

    print("=" * 60)
    print("âœ… Multi-Agent Flow Complete")


# Test Case
my_ingredients = "Chicken breast, spinach, tomatoes, and pasta."

run_meal_planner_flow(my_ingredients)

