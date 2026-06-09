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


#!/usr/bin/env python3
"""
MealPlanPro - Automated Weekly Planner (single-file runnable script)

Core concepts implemented:
1) Agent-to-Agent (A2A) Protocol: PlannerAgent -> ListAgent payload transfer (JSON).
2) Context Engineering: Custom JSON schemas + basic validation enforced for both meal plan and shopping list.
3) Sessions & Memory: DietaryMemory class persists allergy/diet restrictions and filters meal choices.

Author: Capstone demo
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import math
import copy

# ----------------------------
# Simple schema validator (fallback if jsonschema is not available)
# ----------------------------
try:
    import jsonschema  # type: ignore
    HAVE_JSONSCHEMA = True
except Exception:
    HAVE_JSONSCHEMA = False

def run_schema_validation(instance: dict, schema: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate instance against schema.
    First try to use jsonschema (if installed). Otherwise, use a simple structural validator
    that enforces required fields and basic types (sufficient for this demo).
    Returns (is_valid, error_message_or_none)
    """
    if HAVE_JSONSCHEMA:
        try:
            jsonschema.validate(instance=instance, schema=schema)
            return True, None
        except Exception as e:
            return False, str(e)

    # Basic manual checks (limited but explicit)
    def type_name(expected):
        return expected if isinstance(expected, str) else str(expected)

    # A very small structural validator for our use-cases:
    def check(obj, schema_node, path="root"):
        # required
        req = schema_node.get("required") or []
        for r in req:
            if r not in obj:
                return False, f"Missing required field '{r}' at {path}"
        # properties
        props = schema_node.get("properties") or {}
        for k, v in props.items():
            if k in obj:
                val = obj[k]
                expected_type = v.get("type")
                if expected_type:
                    if expected_type == "array":
                        if not isinstance(val, list):
                            return False, f"Field {path}.{k} expected array"
                        # items
                        items_schema = v.get("items")
                        if items_schema:
                            for idx, item in enumerate(val):
                                ok, err = check(item, items_schema, path=f"{path}.{k}[{idx}]")
                                if not ok:
                                    return ok, err
                    elif expected_type == "object":
                        if not isinstance(val, dict):
                            return False, f"Field {path}.{k} expected object"
                        ok, err = check(val, v, path=f"{path}.{k}")
                        if not ok:
                            return ok, err
                    else:
                        py_ok = (
                            (expected_type == "string" and isinstance(val, str)) or
                            (expected_type == "number" and (isinstance(val, int) or isinstance(val, float))) or
                            (expected_type == "integer" and isinstance(val, int)) or
                            (expected_type == "boolean" and isinstance(val, bool))
                        )
                        if not py_ok:
                            return False, f"Field {path}.{k} expected {expected_type}"
        return True, None

    return check(instance, schema, path="root")

# ----------------------------
# JSON Schemas for structured output
# ----------------------------
MEAL_PLAN_SCHEMA = {
    "type": "object",
    "required": ["id", "user_input", "days", "metadata"],
    "properties": {
        "id": {"type": "string"},
        "user_input": {"type": "string"},
        "metadata": {
            "type": "object",
            "required": ["planner", "generated_at", "days_count"],
            "properties": {
                "planner": {"type": "string"},
                "generated_at": {"type": "string"},
                "days_count": {"type": "integer"},
            }
        },
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["day_label", "meals"],
                "properties": {
                    "day_label": {"type": "string"},
                    "meals": {
                        "type": "object",
                        # we expect keys Breakfast,Lunch,Dinner - but allow flexible meals
                        "properties": {
                            "Breakfast": {"type": "object"},
                            "Lunch": {"type": "object"},
                            "Dinner": {"type": "object"},
                        }
                    }
                }
            }
        }
    }
}

SHOPPING_LIST_SCHEMA = {
    "type": "object",
    "required": ["id", "source_mealplan_id", "items", "metadata"],
    "properties": {
        "id": {"type": "string"},
        "source_mealplan_id": {"type": "string"},
        "metadata": {
            "type": "object",
            "required": ["list_agent", "generated_at", "items_count"],
            "properties": {
                "list_agent": {"type": "string"},
                "generated_at": {"type": "string"},
                "items_count": {"type": "integer"},
            }
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "total_quantity": {"type": "number"},
                    "unit": {"type": "string"},
                    "notes": {"type": "string"},
                }
            }
        }
    }
}

# ----------------------------
# Memory: DietaryMemory class
# ----------------------------
class DietaryMemory:
    """
    Simple class-based memory to "remember" user allergy/dietary restrictions.
    For demo we hardcode but expose interface to query.
    """
    def __init__(self, restrictions: Optional[List[str]] = None):
        # store as lowercase normalized tokens
        self.restrictions = set(r.lower() for r in (restrictions or []))

    def add_restriction(self, item: str):
        self.restrictions.add(item.lower())

    def remove_restriction(self, item: str):
        self.restrictions.discard(item.lower())

    def list(self) -> List[str]:
        return sorted(self.restrictions)

    def conflicts_with_ingredient(self, ingredient_name: str) -> bool:
        name = ingredient_name.lower()
        for r in self.restrictions:
            if r in name or name in r:
                return True
        return False

    def allows_recipe(self, recipe: Dict[str, Any]) -> bool:
        # recipe is expected to have "ingredients": List[{"name":...}]
        for ingr in recipe.get("ingredients", []):
            name = ingr.get("name", "")
            if self.conflicts_with_ingredient(name):
                return False
        return True

# ----------------------------
# Simple recipe catalog
# Each recipe is structured to help schema + shopping list aggregation
# ----------------------------
RECIPE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "Greek Yogurt Parfait",
        "tags": ["vegetarian", "breakfast", "low_fat"],
        "ingredients": [
            {"name": "greek yogurt", "quantity": 400, "unit": "g"},
            {"name": "mixed berries", "quantity": 200, "unit": "g"},
            {"name": "honey", "quantity": 2, "unit": "tbsp"},
            {"name": "granola", "quantity": 100, "unit": "g"},
        ]
    },
    {
        "name": "Chickpea Salad",
        "tags": ["vegetarian", "lunch", "high_protein"],
        "ingredients": [
            {"name": "canned chickpeas", "quantity": 400, "unit": "g"},
            {"name": "cucumber", "quantity": 1, "unit": "piece"},
            {"name": "tomato", "quantity": 2, "unit": "piece"},
            {"name": "olive oil", "quantity": 2, "unit": "tbsp"},
            {"name": "lemon juice", "quantity": 1, "unit": "tbsp"},
        ]
    },
    {
        "name": "Stir-Fried Tofu and Vegetables",
        "tags": ["vegetarian", "dinner", "high_protein"],
        "ingredients": [
            {"name": "firm tofu", "quantity": 350, "unit": "g"},
            {"name": "broccoli", "quantity": 200, "unit": "g"},
            {"name": "bell pepper", "quantity": 1, "unit": "piece"},
            {"name": "soy sauce", "quantity": 2, "unit": "tbsp"},
            {"name": "sesame oil", "quantity": 1, "unit": "tbsp"},
        ]
    },
    {
        "name": "Peanut Butter Banana Toast",
        "tags": ["breakfast", "vegetarian"],
        "ingredients": [
            {"name": "bread slices", "quantity": 4, "unit": "piece"},
            {"name": "banana", "quantity": 2, "unit": "piece"},
            {"name": "peanut butter", "quantity": 4, "unit": "tbsp"},
        ]
    },
    {
        "name": "Grilled Salmon Salad",
        "tags": ["lunch", "pescatarian", "high_protein"],
        "ingredients": [
            {"name": "salmon fillet", "quantity": 300, "unit": "g"},
            {"name": "mixed greens", "quantity": 150, "unit": "g"},
            {"name": "olive oil", "quantity": 1, "unit": "tbsp"},
            {"name": "lemon", "quantity": 1, "unit": "piece"},
        ]
    },
    {
        "name": "Shrimp Fried Rice",
        "tags": ["dinner", "high_protein"],
        "ingredients": [
            {"name": "shrimp", "quantity": 300, "unit": "g"},
            {"name": "rice", "quantity": 300, "unit": "g"},
            {"name": "egg", "quantity": 2, "unit": "piece"},
            {"name": "soy sauce", "quantity": 2, "unit": "tbsp"},
        ]
    },
    {
        "name": "Lentil Soup",
        "tags": ["vegetarian", "dinner", "high_fiber"],
        "ingredients": [
            {"name": "red lentils", "quantity": 250, "unit": "g"},
            {"name": "carrot", "quantity": 2, "unit": "piece"},
            {"name": "onion", "quantity": 1, "unit": "piece"},
            {"name": "vegetable stock", "quantity": 1, "unit": "L"},
        ]
    },
]

# ----------------------------
# Helper utilities
# ----------------------------
def normalize_name(name: str) -> str:
    return name.strip().lower()

def merge_quantities(a_qty: float, a_unit: Optional[str], b_qty: float, b_unit: Optional[str]) -> Tuple[float, Optional[str]]:
    """
    If units match (case-insensitive), return summed quantity and unit. Otherwise return original in separate item
    Note: This is intentionally simple — full unit conversion is out of scope for the capstone demo.
    """
    if not a_unit and not b_unit:
        return a_qty + b_qty, None
    if a_unit and b_unit and a_unit.lower() == b_unit.lower():
        return a_qty + b_qty, a_unit
    # Units differ — cannot merge semantically. Caller should keep separate entries. We'll signal by returning NaN as indicator.
    return float("nan"), None

# ----------------------------
# PlannerAgent: generates meal plans using memory & user input
# ----------------------------
class PlannerAgent:
    def __init__(self, memory: DietaryMemory, catalog: List[Dict[str, Any]]):
        self.memory = memory
        self.catalog = catalog
        self.name = "PlannerAgent-v1"

    def choose_recipes_for_meal_type(self, meal_tag: str, user_pref: Optional[str]) -> List[Dict[str, Any]]:
        """
        Select candidate recipes that match meal_tag and optional user_pref.
        """
        candidates = []
        for recipe in self.catalog:
            tags = recipe.get("tags", [])
            if meal_tag in tags:
                if user_pref:
                    if user_pref.lower() in [t.lower() for t in tags]:
                        candidates.append(recipe)
                else:
                    candidates.append(recipe)
        # if none found with strict preference, relax preference
        if not candidates and user_pref:
            for recipe in self.catalog:
                if meal_tag in recipe.get("tags", []):
                    candidates.append(recipe)
        return candidates

    def filter_by_memory(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in recipes if self.memory.allows_recipe(r)]

    def generate_3day_plan(self, user_input: str) -> Dict[str, Any]:
        """
        Generate a 3-day plan (Breakfast, Lunch, Dinner) based on user_input and DietaryMemory filtering.
        Produce structured JSON suitable for A2A transfer.
        """
        days = []
        meal_types = ["Breakfast", "Lunch", "Dinner"]
        # Basic strategy: try to pick recipes that match both meal and user_input (e.g., 'vegetarian' or 'high_protein')
        # we'll cycle through simple deterministic choices to keep demo reproducible
        for day_idx in range(1, 4):
            meals_obj = {}
            for meal in meal_types:
                # choose candidate recipes
                candidates = self.choose_recipes_for_meal_type(meal.lower(), user_input)
                if not candidates:
                    candidates = self.choose_recipes_for_meal_type(meal.lower(), None)
                # filter by memory
                candidates = self.filter_by_memory(candidates)
                if not candidates:
                    # fallback: any memory-allowed recipe
                    candidates = [r for r in self.catalog if self.memory.allows_recipe(r)]
                # deterministic pick: rotate choice by day index
                if candidates:
                    chosen = candidates[(day_idx - 1) % len(candidates)]
                else:
                    chosen = {
                        "name": f"Simple {meal} (no-match)",
                        "tags": [meal.lower()],
                        "ingredients": []
                    }
                meals_obj[meal] = {
                    "name": chosen["name"],
                    "tags": chosen.get("tags", []),
                    "ingredients": chosen.get("ingredients", [])
                }
            days.append({"day_label": f"Day {day_idx}", "meals": meals_obj})

        payload = {
            "id": str(uuid.uuid4()),
            "user_input": user_input,
            "metadata": {
                "planner": self.name,
                "generated_at": "2025-11-25T12:00:00+05:30",  # static demo timestamp (can be datetime.now)
                "days_count": len(days)
            },
            "days": days
        }

        # Validate against MEAL_PLAN_SCHEMA
        ok, err = run_schema_validation(payload, MEAL_PLAN_SCHEMA)
        if not ok:
            raise ValueError(f"PlannerAgent produced invalid meal plan payload: {err}")

        return payload

# ----------------------------
# ListAgent: A2A responder - converts meal plan into aggregated shopping list
# ----------------------------
class ListAgent:
    def __init__(self, name: str = "ListAgent-v1"):
        self.name = name

    def aggregate_shopping_list(self, mealplan_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expect mealplan_payload validated. Produce shopping list: aggregate ingredient names, sum quantities when possible.
        """
        source_id = mealplan_payload.get("id", "unknown")
        aggregator: Dict[Tuple[str, Optional[str]], float] = defaultdict(float)
        separate_items: List[Dict[str, Any]] = []

        # Walk through all recipes' ingredients
        for day in mealplan_payload.get("days", []):
            meals = day.get("meals", {})
            for meal_name, meal in meals.items():
                for ingr in meal.get("ingredients", []):
                    name_raw = ingr.get("name")
                    if not name_raw:
                        continue
                    name = normalize_name(name_raw)
                    qty = ingr.get("quantity")
                    unit = ingr.get("unit")
                    if qty is None:
                        # if no quantity, just add as a note (1 unit)
                        separate_items.append({"name": name_raw, "total_quantity": 1, "unit": unit or "", "notes": f"From {meal.get('name')}"})
                        continue
                    key = (name, unit.lower() if isinstance(unit, str) else unit)
                    # naive merge
                    aggregator[key] += float(qty)

        # Post-process aggregator into list items — try to merge same names with different units into separate notes
        items_out: List[Dict[str, Any]] = []
        name_to_units: Dict[str, List[Tuple[Optional[str], float]]] = defaultdict(list)
        for (name, unit), qty in aggregator.items():
            name_to_units[name].append((unit, qty))

        for name, unit_qtys in name_to_units.items():
            if len(unit_qtys) == 1:
                unit, qty = unit_qtys[0]
                items_out.append({
                    "name": name,
                    "total_quantity": round(qty, 2),
                    "unit": unit or "",
                    "notes": ""
                })
            else:
                # multiple units for same name — list each as separate line with unit in notes
                for unit, qty in unit_qtys:
                    items_out.append({
                        "name": name,
                        "total_quantity": round(qty, 2),
                        "unit": unit or "",
                        "notes": f"Unit: {unit or 'n/a'}"
                    })

        # add separate items collected earlier (no-quantity cases)
        items_out.extend(separate_items)

        # Sort items alphabetically for readability
        items_out = sorted(items_out, key=lambda x: x["name"])

        shopping_list_payload = {
            "id": str(uuid.uuid4()),
            "source_mealplan_id": source_id,
            "metadata": {
                "list_agent": self.name,
                "generated_at": "2025-11-25T12:01:00+05:30",
                "items_count": len(items_out)
            },
            "items": items_out
        }

        ok, err = run_schema_validation(shopping_list_payload, SHOPPING_LIST_SCHEMA)
        if not ok:
            raise ValueError(f"ListAgent produced invalid shopping list payload: {err}")

        return shopping_list_payload

# ----------------------------
# Demonstration / Main execution flow
# ----------------------------
def pretty_print_json(obj: Any):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def main_demo():
    print("\n=== MealPlanPro: Automated Weekly Planner (Concierge Agents) ===\n")

    # 1) Hardcode memory (per spec): user allergies/dietary restrictions
    memory = DietaryMemory(restrictions=["peanuts", "shellfish"])  # persistent for the session
    print("DietaryMemory (hardcoded):", memory.list())
    print()

    # 2) Create PlannerAgent using memory & catalog
    planner = PlannerAgent(memory=memory, catalog=RECIPE_CATALOG)

    # Example user input for planner
    user_input = "vegetarian"  # could be 'high_protein', 'pescatarian', etc.
    print(f"User input to PlannerAgent: '{user_input}'\n")

    # 3) Planner generates a 3-day meal plan (A2A payload)
    meal_plan_payload = planner.generate_3day_plan(user_input=user_input)
    print("PlannerAgent -> Generated Meal Plan (A2A payload):")
    pretty_print_json(meal_plan_payload)
    print()

    # 4) Simulated A2A transfer: send meal_plan_payload to ListAgent (A2A responder)
    list_agent = ListAgent()
    shopping_list_payload = list_agent.aggregate_shopping_list(meal_plan_payload)
    print("ListAgent -> Generated Shopping List (A2A response):")
    pretty_print_json(shopping_list_payload)
    print()

    # 5) Show how memory influenced choices (filtering)
    print("Notes:")
    print(f"- Memory restrictions: {memory.list()}")
    # Show any chosen meal that would have conflicted if memory didn't filter
    conflicts = []
    for day in meal_plan_payload["days"]:
        for mkey, meal in day["meals"].items():
            # check ingredients if conflict with memory
            for ingr in meal.get("ingredients", []):
                if memory.conflicts_with_ingredient(ingr.get("name", "")):
                    conflicts.append((day["day_label"], mkey, meal["name"], ingr["name"]))
    if conflicts:
        print("- Warning: planner produced recipes containing restricted items (should be filtered).")
        for c in conflicts:
            print(f"  * {c}")
    else:
        print("- All planned recipes respect the memory restrictions (no ingredients contain 'peanuts' or 'shellfish').")

    print("\nDemo complete. You may adapt the recipe catalog, memory, and aggregation rules for a fuller production system.\n")

if __name__ == "__main__":
    main_demo()


