# -------------------------------------------------------
# RoboFlow Agent - Freestyle Track Capstone Project
# Kaggle-Compatible Version
# -------------------------------------------------------

import os
import json
import textwrap
from typing import Dict, Any
from pathlib import Path

# -------------------------------------------------------
# Agent Class
# -------------------------------------------------------

class RoboFlowAgent:
    """
    An AI Agent that automates micro-tasks:
    - Summaries
    - Meal suggestions
    - Study planning
    - Productivity assistance
    """

    def __init__(self, name: str = "RoboFlow Agent"):
        self.name = name

    def summarize(self, text: str) -> str:
        """Return a short summary of text."""
        lines = text.split(".")
        summary = ". ".join(lines[:2])
        return f"Summary → {summary.strip()}."

    def meal_plan(self, calories: int = 1800) -> Dict[str, Any]:
        """Return a simple meal plan by calorie goal."""
        return {
            "target_calories": calories,
            "breakfast": "Oats + peanut butter + banana",
            "lunch": "Dal + rice + vegetable sabzi",
            "dinner": "Roti + paneer + salad",
            "snack": "Dry fruits or fruit bowl"
        }

    def study_plan(self, hours: int = 2) -> Dict[str, Any]:
        """Return a study plan based on hours available."""
        return {
            "total_hours": hours,
            "plan": [
                {"task": "Concept learning", "time": f"{hours*0.5:.1f} hr"},
                {"task": "Practice questions", "time": f"{hours*0.3:.1f} hr"},
                {"task": "Revision", "time": f"{hours*0.2:.1f} hr"},
            ]
        }

    def chat(self, query: str) -> Any:
        """Main Agent logic."""
        query_lower = query.lower()

        if "summary" in query_lower:
            return self.summarize(query)

        if "meal" in query_lower or "food" in query_lower:
            return self.meal_plan()

        if "study" in query_lower or "prepare" in query_lower:
            return self.study_plan()

        return "I can help with summaries, meal planning, or study scheduling."


agent = RoboFlowAgent()

print("1) Summary Example →")
print(agent.chat("Make summary of AI and robotics. They are growing fast."))

print("\n2) Meal Plan →")
print(agent.chat("Give me a meal plan"))

print("\n3) Study Plan →")
print(agent.chat("Create a study plan for me"))

