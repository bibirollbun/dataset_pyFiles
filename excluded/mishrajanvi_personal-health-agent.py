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


import uuid
from typing import Dict, Any, List

print("Python is working")


import uuid
from typing import Dict, Any, List

# Session & Memory classes

class SessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_profile: Dict[str, Any]) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "profile": user_profile,
            "history": []
        }
        return session_id

    def get_profile(self, session_id: str) -> Dict[str, Any]:
        return self.sessions[session_id]["profile"]

    def add_history(self, session_id: str, event: Dict[str, Any]):
        self.sessions[session_id]["history"].append(event)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions[session_id]["history"]

class MemoryBank:
    def __init__(self):
        self.food_events: List[Dict[str, Any]] = []

    def save_event(self, event: Dict[str, Any]):
        self.food_events.append(event)

    def summarize_patterns(self) -> str:
        if not self.food_events:
            return "No previous food history."
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for e in self.food_events:
            g = e.get("grade")
            if g in counts:
                counts[g] += 1
        parts = []
        for grade, c in counts.items():
            if c > 0:
                parts.append(f"{c} item(s) with grade {grade}")
        return "History summary: " + ", ".join(parts)

# Ingredient database and tool

INGREDIENT_DB = {
    "sugar": {"type": "nutrient", "risk": "high_sugar", "notes": "Too much sugar is not good for obesity and diabetes."},
    "salt": {"type": "nutrient", "risk": "high_sodium", "notes": "High sodium intake is not good for blood pressure."},
    "sodium": {"type": "nutrient", "risk": "high_sodium", "notes": "High sodium intake is not good for blood pressure."},
    "trans fat": {"type": "fat", "risk": "trans_fat", "notes": "Trans fat increases risk of heart disease."},
    "saturated fat": {"type": "fat", "risk": "sat_fat", "notes": "Too much saturated fat can increase cholesterol."},
    "monosodium glutamate": {"type": "additive", "risk": "additive_maybe_sensitive", "notes": "Some people are sensitive to MSG."},
    "sodium nitrite": {"type": "additive", "risk": "nitrite", "notes": "Used in processed meats; high intake is not recommended."},
    "artificial colour": {"type": "additive", "risk": "colour_additive", "notes": "Some artificial colours can cause sensitivity in children."},
    "milk": {"type": "allergen", "risk": "allergen", "notes": "Common allergen; dangerous for people with milk allergy."},
    "peanuts": {"type": "allergen", "risk": "allergen", "notes": "Common allergen; can cause strong reactions."},
    "wheat": {"type": "allergen", "risk": "allergen", "notes": "Contains gluten; problem for people with celiac disease."}
}

COMMON_ALLERGENS = ["milk", "eggs", "peanuts", "tree nuts", "soy", "wheat", "fish", "shellfish", "sesame"]

def ingredient_info_tool(ingredient: str) -> Dict[str, Any]:
    key = ingredient.lower().strip()
    if key in INGREDIENT_DB:
        info = INGREDIENT_DB[key].copy()
        info["known"] = True
        info["name"] = ingredient
        return info
    else:
        return {
            "known": False,
            "name": ingredient,
            "type": "unknown",
            "risk": "unknown",
            "notes": "No specific information in local database."
        }

# Agents

class LabelReaderAgent:
    def run(self, raw_ingredients: str, nutrition_info: Dict[str, float]) -> Dict[str, Any]:
        ingredients = [i.strip().lower() for i in raw_ingredients.split(",") if i.strip()]
        return {"ingredients": ingredients, "nutrition": nutrition_info}

class RiskAnalyzerAgent:
    def run(self, parsed_label: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        ingredients = parsed_label["ingredients"]
        nutrition = parsed_label["nutrition"]
        age = user_profile.get("age")
        allergies = [a.lower() for a in user_profile.get("allergies", [])]
        diseases = [d.lower() for d in user_profile.get("diseases", [])]
        ingredient_details = []
        warnings = []
        for ing in ingredients:
            info = ingredient_info_tool(ing)
            ingredient_details.append(info)
            for allergen in COMMON_ALLERGENS:
                if allergen in ing and allergen in allergies:
                    warnings.append(f"Contains your allergen: {allergen}.")
            if info["risk"] == "allergen":
                warnings.append(f"Contains common allergen: {info['name']}.")
            if info["risk"] in ["nitrite", "colour_additive", "additive_maybe_sensitive"]:
                warnings.append(f"Contains additive: {info['name']} ({info['notes']}).")
        sugar = nutrition.get("sugar_g", 0.0)
        sodium = nutrition.get("sodium_mg", 0.0)
        sat_fat = nutrition.get("sat_fat_g", 0.0)
        trans_fat = nutrition.get("trans_fat_g", 0.0)
        if sugar > 15:
            warnings.append("High sugar content; not good for daily use.")
        if sodium > 400:
            warnings.append("High sodium content; not good for blood pressure.")
        if sat_fat > 5:
            warnings.append("High saturated fat; may increase cholesterol.")
        if trans_fat > 0:
            warnings.append("Contains trans fat; should be avoided.")
        if age is not None and age < 18 and sugar > 10:
            warnings.append("For your age, this product is high in sugar.")
        if "diabetes" in diseases and sugar > 5:
            warnings.append("You have diabetes; sugar level is high for you.")
        if "hypertension" in diseases and sodium > 300:
            warnings.append("You have hypertension; sodium level is high for you.")
        return {"ingredient_details": ingredient_details, "warnings": warnings, "nutrition": nutrition}

class GradingAgent:
    def run(self, risk_report: Dict[str, Any]) -> Dict[str, Any]:
        nutrition = risk_report["nutrition"]
        warnings = risk_report["warnings"]
        sugar = nutrition.get("sugar_g", 0.0)
        sodium = nutrition.get("sodium_mg", 0.0)
        sat_fat = nutrition.get("sat_fat_g", 0.0)
        trans_fat = nutrition.get("trans_fat_g", 0.0)
        score = 0.0
        if sugar <= 5:
            score += 2
        elif sugar <= 15:
            score += 1
        else:
            score -= 1
        if sodium <= 200:
            score += 2
        elif sodium <= 400:
            score += 1
        else:
            score -= 1
        if sat_fat <= 2:
            score += 2
        elif sat_fat <= 5:
            score += 1
        else:
            score -= 1
        if trans_fat == 0:
            score += 1
        else:
            score -= 2
        score -= 0.3 * len(warnings)
        if score >= 5:
            grade = "A"
        elif score >= 2:
            grade = "B"
        elif score >= -1:
            grade = "C"
        else:
            grade = "D"
        explanation = f"Overall grade for you: {grade}."

        if warnings:
            explanation += "Reasons / warnings:"

            for w in warnings:
                explanation += f"- {w}"

        else:
            explanation += "No major warnings detected with current rules."
        return {
            "grade": grade,
            "explanation": explanation.strip(),
            "warnings": warnings
        }

# Gemini API client example (replace with your real client)

class GeminiAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize Gemini client here

    def generate_text(self, prompt: str, max_tokens: int = 200) -> Dict[str, str]:
        # Replace with actual API call to Gemini LLM
        simulated_response = {
            "text": (
                "Based on your profile and the food data, this product is graded D because it contains "
                "high sugar levels, allergens like milk and peanuts, and unhealthy additives. "
                "It is recommended to avoid this product for better health."
            )
        }
        return simulated_response

class GeminiExplanationAgent:
    def __init__(self, gemini_client: GeminiAPIClient):
        self.client = gemini_client

    def run(self, grading_report: Dict[str, Any], risk_report: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"You are a nutrition assistant providing a simple explanation."

            f"User Profile: {user_profile}"

            f"Food Grade: {grading_report['grade']}"

            f"Warnings: {', '.join(risk_report['warnings']) if risk_report['warnings'] else 'No warnings'}"

            f"Nutrition Details: {risk_report['nutrition']}"


            f"Please generate a clear, friendly, and concise explanation for the user."
        )
        response = self.client.generate_text(prompt=prompt, max_tokens=200)
        explanation = response.get("text", "").strip()
        return {
            "explanation": explanation,
            "grade": grading_report["grade"],
            "warnings": risk_report["warnings"],
        }

class FoodHealthOrchestrator:
    def __init__(self, session_service: SessionService, memory_bank: MemoryBank, gemini_explainer: GeminiExplanationAgent = None):
        self.session_service = session_service
        self.memory_bank = memory_bank
        self.label_reader = LabelReaderAgent()
        self.risk_analyzer = RiskAnalyzerAgent()
        self.grading_agent = GradingAgent()
        self.gemini_explainer = gemini_explainer

    def process_food(self, session_id: str, raw_ingredients: str, nutrition_info: Dict[str, float], food_name: str = "Unknown Food") -> Dict[str, Any]:
        user_profile = self.session_service.get_profile(session_id)
        parsed_label = self.label_reader.run(raw_ingredients, nutrition_info)
        risk_report = self.risk_analyzer.run(parsed_label, user_profile)
        grading_result = self.grading_agent.run(risk_report)
        # Use Gemini explanation if available
        if self.gemini_explainer:
            gemini_result = self.gemini_explainer.run(grading_result, risk_report, user_profile)
            explanation = gemini_result["explanation"]
        else:
            explanation = grading_result["explanation"]

        event = {
            "food_name": food_name,
            "ingredients": parsed_label["ingredients"],
            "nutrition": nutrition_info,
            "warnings": grading_result["warnings"],
            "grade": grading_result["grade"]
        }
        self.session_service.add_history(session_id, event)
        self.memory_bank.save_event(event)
        history_summary = self.memory_bank.summarize_patterns()
        return {"food_name": food_name, "grade": grading_result["grade"], "explanation": explanation, "history_summary": history_summary}

def demo_run():
    session_service = SessionService()
    memory_bank = MemoryBank()
    user_profile = {
        "age": 15,
        "gender": "female",
        "allergies": ["milk", "peanuts"],
        "diseases": ["none"]
    }
    session_id = session_service.create_session(user_profile)
    food_name = "Chocolate Snack Bar"
    raw_ingredients = "Sugar, Milk powder, Cocoa butter, Peanuts, Artificial colour"
    nutrition_info = {"sugar_g": 22.0, "sodium_mg": 150.0, "sat_fat_g": 7.0, "trans_fat_g": 0.5}

    gemini_client = GeminiAPIClient(api_key="YOUR_REAL_API_KEY")
    gemini_explainer = GeminiExplanationAgent(gemini_client)

    orchestrator = FoodHealthOrchestrator(session_service, memory_bank, gemini_explainer)
    result = orchestrator.process_food(session_id, raw_ingredients, nutrition_info, food_name)
    print("="*60)
    print(f"Food: {result['food_name']}")
    print("-"*60)
    print(result["explanation"])
    print("-"*60)
    print(result["history_summary"])
    print("="*60)

if __name__ == "__main__":
    demo_run()

