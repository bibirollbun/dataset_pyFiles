# smart_grocery_agent.py
from datetime import datetime
from collections import defaultdict, Counter
import random
import math
import json

# ---------------------------
# Memory: stores preferences and history
# ---------------------------
class Memory:
    def __init__(self, max_history=50):
        self.max_history = max_history
        self.history = []  # list of dicts: {role, content, time}
        self.persistent = {
            "preferences": {},
            "allergies": [],
            "diet": None,
            "household_size": 1,
            "budget_per_week": None,
            "favorite_recipes": []
        }

    def add_message(self, role, content):
        entry = {"role": role, "content": content, "time": datetime.utcnow().isoformat()}
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_recent(self, n=5):
        return self.history[-n:]

    def set_preference(self, key, val):
        self.persistent["preferences"][key] = val

    def get_preference(self, key, default=None):
        return self.persistent["preferences"].get(key, default)

    def update_profile(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.persistent:
                self.persistent[k] = v

    def to_dict(self):
        return {"history": self.history, "persistent": self.persistent}


# ---------------------------
# MealPlanner: picks recipes for a week
# ---------------------------
class MealPlanner:
    def __init__(self, recipe_db=None):
        # Simple recipe DB: name -> dict(ingredients:list of (item,qty,unit), servings:int, category)
        self.recipe_db = recipe_db or self._default_recipes()

    def _default_recipes(self):
        # minimal example recipes; quantity units are arbitrary units
        return {
            "Vegetable Stir Fry": {
                "ingredients": [("broccoli", 1, "head"), ("carrot", 2, "pcs"), ("soy_sauce", 2, "tbsp"), ("rice", 1, "cup")],
                "servings": 2,
                "category": "veg"
            },
            "Pasta with Tomato Sauce": {
                "ingredients": [("pasta", 200, "g"), ("tomato", 3, "pcs"), ("olive_oil", 2, "tbsp")],
                "servings": 2,
                "category": "veg"
            },
            "Chicken Curry": {
                "ingredients": [("chicken", 400, "g"), ("onion", 1, "pcs"), ("tomato", 2, "pcs"), ("rice", 1, "cup")],
                "servings": 3,
                "category": "non-veg"
            },
            "Oat Porridge": {
                "ingredients": [("oats", 100, "g"), ("milk", 200, "ml"), ("banana", 1, "pcs")],
                "servings": 1,
                "category": "veg"
            },
            "Egg Sandwich": {
                "ingredients": [("egg", 2, "pcs"), ("bread", 2, "slices"), ("butter", 1, "tbsp")],
                "servings": 1,
                "category": "non-veg"
            }
        }

    def plan_week(self, user_profile):
        """
        user_profile: dict with keys: diet ('veg'/'non-veg'/None), household_size, avoid (list), favorites(list)
        returns: list of 7 meal dicts {day, meals: [recipe_names]}
        """
        diet = user_profile.get("diet")
        household_size = max(1, user_profile.get("household_size", 1))
        avoid = set(user_profile.get("avoid", []))
        favorites = user_profile.get("favorite_recipes", [])

        # Candidate recipes filtered by diet + avoid
        candidates = []
        for name, info in self.recipe_db.items():
            if diet == "veg" and info.get("category") == "non-veg":
                continue
            if name in avoid:
                continue
            candidates.append(name)

        if not candidates:
            candidates = list(self.recipe_db.keys())

        # select 7 meals (one main meal per day) trying to include favorites
        week = []
        for day in range(7):
            if favorites and random.random() < 0.4:
                choice = random.choice([f for f in favorites if f in candidates] or candidates)
            else:
                choice = random.choice(candidates)
            week.append({"day": day + 1, "recipe": choice, "servings": self.recipe_db[choice]["servings"] * household_size})
        return week


# ---------------------------
# InventoryAgent
# ---------------------------
class InventoryAgent:
    def __init__(self, user_inventory=None):
        # inventory: item -> quantity (in same unit semantics as recipes, loosely)
        self.inventory = user_inventory or {}

    def has_item(self, item):
        return self.inventory.get(item, 0) > 0

    def consume(self, item, qty):
        if item in self.inventory:
            self.inventory[item] = max(0, self.inventory[item] - qty)

    def missing_items_for_recipe(self, recipe, recipe_db):
        missing = []
        for ing, qty, unit in recipe_db[recipe]["ingredients"]:
            inv_qty = self.inventory.get(ing, 0)
            if inv_qty < qty:
                missing.append((ing, qty - inv_qty, unit))
        return missing

    def aggregate_missing_for_week(self, week_plan, recipe_db):
        aggregate = defaultdict(lambda: [0, ""])
        for entry in week_plan:
            recipe = entry["recipe"]
            times = 1  # assume one unit per day; servings incorporated in recipe ingredient quantities in more advanced setups
            for ing, qty, unit in recipe_db[recipe]["ingredients"]:
                aggregate[ing][0] += qty * times
                aggregate[ing][1] = unit
        # subtract inventory
        result = []
        for ing, (qty, unit) in aggregate.items():
            inv_qty = self.inventory.get(ing, 0)
            needed = max(0, qty - inv_qty)
            if needed > 0:
                result.append((ing, needed, unit))
        return result


# ---------------------------
# BudgetAgent
# ---------------------------
class BudgetAgent:
    def __init__(self, price_catalog=None):
        # price_catalog: item -> price per unit (unit consistent with recipe_db)
        self.price_catalog = price_catalog or self._default_prices()

    def _default_prices(self):
        return {
            "broccoli": 40.0,    # price per head
            "carrot": 8.0,       # per piece
            "soy_sauce": 50.0,   # per bottle - note mismatch; in this simple model we use relative values
            "rice": 60.0,        # per kg or per cup scaled
            "pasta": 80.0,
            "tomato": 10.0,
            "olive_oil": 180.0,
            "chicken": 200.0,
            "onion": 5.0,
            "oats": 150.0,
            "milk": 60.0,
            "banana": 8.0,
            "egg": 6.0,
            "bread": 40.0,
            "butter": 120.0
        }

    def estimate_cost(self, items):
        """
        items: list of (item, qty, unit)
        returns estimated total cost and breakdown list of (item, qty, unit, price_est)
        """
        total = 0.0
        breakdown = []
        for item, qty, unit in items:
            price_unit = self.price_catalog.get(item, None)
            if price_unit is None:
                # fallback estimate
                price_unit = 50.0
            # This simplistic model assumes qty is roughly in units that align with price_unit
            est = price_unit * qty
            breakdown.append((item, qty, unit, est))
            total += est
        return total, breakdown

    def propose_swaps(self, breakdown, budget_limit):
        """
        If total > budget, propose items to swap or remove.
        Simple heuristic: propose removing optional items or substituting high-cost items.
        """
        total = sum(x[3] for x in breakdown)
        if total <= budget_limit:
            return [], total

        # sort by cost desc
        sorted_by_cost = sorted(breakdown, key=lambda x: x[3], reverse=True)
        suggestions = []
        reduced_total = total
        for item, qty, unit, est in sorted_by_cost:
            if reduced_total <= budget_limit:
                break
            # propose removal or cheaper alternative
            # For demo, propose reducing qty by half (if qty>1) else mark optional removal
            if qty > 1:
                suggestions.append(f"Reduce {item} quantity from {qty} to {max(1, math.floor(qty/2))}")
                reduced_total -= est * 0.5
            else:
                suggestions.append(f"Consider removing optional item: {item}")
                reduced_total -= est
        return suggestions, reduced_total


# ---------------------------
# GroceryListAgent
# ---------------------------
class GroceryListAgent:
    def __init__(self):
        pass

    def categorize(self, items):
        # simple categories by keywords
        cats = defaultdict(list)
        vegs = {"broccoli", "carrot", "tomato", "onion", "banana"}
        dairy = {"milk", "butter"}
        grains = {"rice", "pasta", "bread", "oats"}
        proteins = {"chicken", "egg"}
        for item, qty, unit in items:
            if item in vegs:
                cats["Vegetables/Fruits"].append((item, qty, unit))
            elif item in dairy:
                cats["Dairy"].append((item, qty, unit))
            elif item in grains:
                cats["Grains"].append((item, qty, unit))
            elif item in proteins:
                cats["Proteins"].append((item, qty, unit))
            else:
                cats["Others"].append((item, qty, unit))
        return dict(cats)


# ---------------------------
# Coordinator: orchestrates everything
# ---------------------------
class Coordinator:
    def __init__(self, memory=None, meal_planner=None, inventory_agent=None, budget_agent=None, grocery_agent=None):
        self.memory = memory or Memory()
        self.meal_planner = meal_planner or MealPlanner()
        self.inventory_agent = inventory_agent or InventoryAgent()
        self.budget_agent = budget_agent or BudgetAgent()
        self.grocery_agent = grocery_agent or GroceryListAgent()

    def intake_user_profile(self, profile):
        # profile may have diet, household_size, budget_per_week, allergies, favorites
        self.memory.update_profile(**profile)
        self.memory.add_message("system", f"Updated profile: {profile}")

    def generate_plan_and_list(self):
        profile = self.memory.persistent.copy()
        week_plan = self.meal_planner.plan_week(profile)
        missing_items = self.inventory_agent.aggregate_missing_for_week(week_plan, self.meal_planner.recipe_db)

        # estimate cost
        total_cost, breakdown = self.budget_agent.estimate_cost(missing_items)
        budget = profile.get("budget_per_week", None)

        suggestions = []
        reduced_total = total_cost
        if budget is not None and total_cost > budget:
            suggestions, reduced_total = self.budget_agent.propose_swaps(breakdown, budget)

        categorized = self.grocery_agent.categorize(missing_items)

        # store into memory
        result = {
            "week_plan": week_plan,
            "missing_items": missing_items,
            "cost_estimate": total_cost,
            "cost_breakdown": breakdown,
            "budget_suggestions": suggestions,
            "projected_cost_after_suggestions": reduced_total,
            "categorized_list": categorized
        }
        self.memory.add_message("agent", f"Generated weekly plan and list: cost {total_cost}")
        return result

    def pretty_print(self, result):
        print("=== 7-Day Meal Plan ===")
        for entry in result["week_plan"]:
            day = entry["day"]
            recipe = entry["recipe"]
            print(f"Day {day}: {recipe}")

        print("\n=== Grocery List (Aggregated Missing Items) ===")
        for item, qty, unit in result["missing_items"]:
            print(f"- {item}: {qty} {unit}")

        print(f"\nEstimated total cost: ₹{result['cost_estimate']:.2f}")
        if result["budget_suggestions"]:
            print("\nBudget suggestions:")
            for s in result["budget_suggestions"]:
                print(f"- {s}")
            print(f"\nProjected cost after suggestions: ₹{result['projected_cost_after_suggestions']:.2f}")

        print("\n=== Categorized List ===")
        for cat, items in result["categorized_list"].items():
            print(f"\n{cat}:")
            for it, q, u in items:
                print(f"  - {it}: {q} {u}")


# ---------------------------
# Example usage (if run as script)
# ---------------------------
if __name__ == "__main__":
    # create components
    mem = Memory()
    # sample user inventory: user already has 1 broccoli, 1 rice
    inv = InventoryAgent(user_inventory={"broccoli": 1, "rice": 1})
    planner = MealPlanner()
    budget = BudgetAgent()
    grocery = GroceryListAgent()
    coord = Coordinator(memory=mem, meal_planner=planner, inventory_agent=inv, budget_agent=budget, grocery_agent=grocery)

    # user sets profile
    coord.intake_user_profile({
        "diet": "veg",
        "household_size": 2,
        "budget_per_week": 800,
        "favorite_recipes": ["Pasta with Tomato Sauce"]
    })

    result = coord.generate_plan_and_list()
    coord.pretty_print(result)

    # Dump result to JSON for saving to repo / Kaggle
    with open("smart_grocery_output.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


