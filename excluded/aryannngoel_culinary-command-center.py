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


# =====================================================================
# MULTI-AGENT SYSTEM: CULINARY COMMAND CENTER
# Concierge Agent Track: Automating weekly meal planning and shopping.
# =====================================================================

class PantryVault:
    """The high-security asset storage (Long-Term Memory)."""
    def __init__(self):
        # Stored Preferences and Assets
        self.preferences = {
            "dietary_restrictions": ["vegetarian", "no mushrooms"],
            "max_cook_time_min": 45,
            "favorite_cuisines": ["Italian", "Mexican"],
            "pantry_inventory": {"rice": "3 cups", "black beans": "2 cans", "onion": "3"},
            "budget_limit": 100.00
        }

    def retrieve_asset(self, key):
        """Retrieves critical mission assets (preferences)."""
        return self.preferences.get(key)

class BattleLog:
    """The shared, real-time command board (Session State)."""
    def __init__(self, initial_plan=None):
        self.current_plan = initial_plan if initial_plan is not None else {}

    def log_update(self, key, value):
        """Logs a critical update to the mission plan."""
        self.current_plan[key] = value

COMMAND_CENTER_MEMORY = PantryVault()

class RecipeVaultTool:
    """The secured database of culinary schematics (Custom Tool)."""
    def search_schematics(self, constraints):
        """Searches the vault based on mission constraints (diet, time, cost)."""
        print(f"-> TOOL ACCESS: Chef AI querying Recipe Vault with constraints: {constraints}")

        if "vegetarian" in constraints.get("diet", []):
            return [
                {"name": "Black Bean Tacos (Flavor Bomb)", "ingredients": ["black beans", "tortillas", "onion", "lime"], "cost": 15.00},
                {"name": "Creamy Tomato Pasta (Quick Deploy)", "ingredients": ["pasta", "canned tomatoes", "cream", "garlic"], "cost": 12.00}
            ]
        return []

RECIPE_VAULT = RecipeVaultTool()

def tactical_llm_reasoning(prompt, model_name="Gemini 2.5 Flash"):
    """Simulates the sophisticated reasoning of the AI core (Gemini integration)."""
    print(f"   (AI CORE: Using {model_name} for high-level tactical analysis on: '{prompt[:40]}...')")
    return "TACTICAL_RESPONSE_RECEIVED"

class Commander:
    """Agent 1: The Mission Briefing Specialist (GoalAgent)."""
    def run_briefing(self, user_goal: str, log: BattleLog):

        # Human-to-AI translation and context retrieval
        prompt = (
            f"User Distress Signal: '{user_goal}'. "
            f"Pantry Vault Assets: {COMMAND_CENTER_MEMORY.preferences}. "
            "Extract Critical Success Factors (CSFs): diet, budget, meal count."
        )
        tactical_llm_reasoning(prompt)

        # Simulation of extracted CSFs
        csf = {
            "meal_count": 5,
            "diet": ["vegetarian"],
            "budget": COMMAND_CENTER_MEMORY.retrieve_asset("budget_limit")
        }

        log.log_update("critical_success_factors", csf)
        print(f"**COMMANDER COMPLETE.** CSFs locked down and logged: {csf}")
        return log.current_plan

class ChefAI:
    """Agent 2: The Culinary Architect (MealPlannerAgent)."""
    def run_design_phase(self, log: BattleLog):
        csf = log.current_plan.get("critical_success_factors", {})

        # Tool Use: Access the Recipe Vault for schematics
        available_schematics = RECIPE_VAULT.search_schematics(csf)

        # Context Compaction & LLM Reasoning for optimal selection
        prompt = (
            f"Available Schematics: {available_schematics}. "
            f"Pantry Inventory: {COMMAND_CENTER_MEMORY.retrieve_asset('pantry_inventory')}. "
            "Engineer the menu: select 2 recipes that maximize flavor and minimize asset expenditure."
        )
        tactical_llm_reasoning(prompt)

        # Simulated engineered menu
        engineered_menu = [
            {"day": "Monday - Quick Deploy", "recipe": "Black Bean Tacos (Flavor Bomb)", "materials_needed": {"tortillas": 12, "lime": 2}},
            {"day": "Tuesday - Low Resource", "recipe": "Creamy Tomato Pasta (Quick Deploy)", "materials_needed": {"pasta": 1, "cream": 1}}
        ]

        log.log_update("engineered_menu", engineered_menu)
        print(f"**CHEF AI COMPLETE.** Tactical Menu designed for 2 mission days.")
        return log.current_plan

class Quartermaster:
    """Agent 3: The Supply Chain Strategist (ShoppingAgent)."""
    def run_supply_chain_audit(self, log: BattleLog):
        engineered_menu = log.current_plan.get("engineered_menu", [])

        # Consolidate materials required
        purchase_order = {}
        for item in engineered_menu:
            for material, quantity in item["materials_needed"].items():
                purchase_order[material] = purchase_order.get(material, 0) + quantity

        # Final LLM Review: Format the purchase order for field agent (user)
        prompt = f"Final calculated purchase order (materials): {purchase_order}. Format this as a mobile checklist for immediate deployment."
        final_purchase_order = tactical_llm_reasoning(prompt, model_name="Gemini 2.5 Flash")

        log.log_update("final_purchase_order", final_purchase_order)
        print(f"**QUARTERMASTER COMPLETE.** Supply chain optimized.")
        print("-" * 30)
        print("FINAL MATERIALS REQUIRED (Purchase Order):")
        print(purchase_order)
        return final_purchase_order

def execute_culinary_command_center(user_signal: str):
    """The main operation launch sequence, demonstrating the Multi-Agent System."""
    print("✨ STARTING CULINARY COMMAND CENTER: OPERATION DINNER SAVIOR ✨")

    # 1. Initialize Battle Log
    mission_log = BattleLog()

    # 2. Sequential Deployment (A2A Protocol)

    # AGENT 1: Commander
    print("\n--- 1. DEPLOYING COMMANDER (Mission Briefing) ---")
    commander = Commander()
    commander.run_briefing(user_signal, mission_log)

    # AGENT 2: Chef AI
    print("\n--- 2. DEPLOYING CHEF AI (Design Phase) ---")
    chef_ai = ChefAI()
    chef_ai.run_design_phase(mission_log)

    # AGENT 3: Quartermaster
    print("\n--- 3. DEPLOYING QUARTERMASTER (Supply Audit) ---")
    quartermaster = Quartermaster()
    quartermaster.run_supply_chain_audit(mission_log)

    print("\n✅ MISSION COMPLETE: DINNER SAVED.")

# --- LAUNCH SEQUENCE ---
user_distress_signal = "I need a tactical plan for 5 vegetarian dinners this week. Budget is critical."
execute_culinary_command_center(user_distress_signal)

