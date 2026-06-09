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


import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from enum import Enum


from google.adk import Agent, Runner
from google.adk.tools import AgentTool


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


class UserCondition(str, Enum):
    PRE_DIABETIC = "pre_diabetic"
    DIABETIC = "diabetic"
    HYPERTENSION = "hypertension"
    KIDNEY_DISEASE = "kidney_disease"
    HEART_DISEASE = "heart_disease"
    NONE = "none"

class DietaryRestriction(str, Enum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    LACTOSE_INTOLERANT = "lactose_intolerant"
    KOSHER = "kosher"
    HALAL = "halal"
    NONE = "none"

class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"


class Medicine(BaseModel):
    name: str
    dosage: str
    frequency: str  # e.g., "once daily", "twice daily"
    condition_treated: str
    interactions: List[str] = Field(default_factory=list)  # Known food/nutrient interactions

class Supplement(BaseModel):
    name: str
    nutrient: str  # e.g., "Vitamin D", "Iron", "B12"
    dose_amount: float
    dose_unit: str  # e.g., "IU", "mg", "mcg"
    frequency: str


class UserProfile(BaseModel):
    user_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    activity_level: ActivityLevel = ActivityLevel.MODERATELY_ACTIVE
    conditions: List[UserCondition] = Field(default_factory=list)
    dietary_restrictions: List[DietaryRestriction] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    medicines: List[Medicine] = Field(default_factory=list)
    supplements: List[Supplement] = Field(default_factory=list)

class UserMemory(BaseModel):
    """Long-term memory for user's stable attributes and patterns"""
    user_id: str
    profile: UserProfile
    dietary_patterns: List[str] = Field(default_factory=list)  # e.g., "frequently exceeds sugar at breakfast"
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class NutritionGoal(BaseModel):
    calories_min: float
    calories_max: float
    protein_g_min: float
    protein_g_max: float
    carbs_g_min: float
    carbs_g_max: float
    fat_g_min: float
    fat_g_max: float
    fiber_g_min: float
    sugar_g_max: float
    sodium_mg_max: float
    saturated_fat_g_max: float
    # Micronutrients
    vitamin_d_iu_min: Optional[float] = None
    vitamin_b12_mcg_min: Optional[float] = None
    iron_mg_min: Optional[float] = None
    calcium_mg_min: Optional[float] = None
    potassium_mg_min: Optional[float] = None


class FoodLogEntry(BaseModel):
    food_name: str
    quantity: float
    unit: str  # e.g., "cup", "oz", "g", "piece"
    meal: str  # e.g., "breakfast", "lunch", "dinner", "snack"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class NutrientTotals(BaseModel):
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    sodium_mg: float = 0.0
    saturated_fat_g: float = 0.0
    vitamin_d_iu: float = 0.0
    vitamin_b12_mcg: float = 0.0
    iron_mg: float = 0.0
    calcium_mg: float = 0.0
    potassium_mg: float = 0.0
    vitamin_k_mcg: float = 0.0

class GoalStatus(BaseModel):
    nutrient: str
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    actual: float
    status: str  # "within_target", "below_target", "slightly_over", "significantly_over"
    percentage_of_target: float

class DailyNutritionSummary(BaseModel):
    date: str
    user_id: str
    food_entries: List[FoodLogEntry]
    nutrient_totals: NutrientTotals
    goal_statuses: List[GoalStatus]
    restriction_violations: List[str] = Field(default_factory=list)
    allergy_violations: List[str] = Field(default_factory=list)
    overall_assessment: str


class InteractionWarning(BaseModel):
    severity: str  # "low", "medium", "high"
    medicine_or_supplement: str
    nutrient_or_food: str
    description: str
    recommendation: str

class MedicineInteractionReport(BaseModel):
    date: str
    user_id: str
    warnings: List[InteractionWarning] = Field(default_factory=list)
    nutrient_upper_limit_concerns: List[str] = Field(default_factory=list)
    overall_safety_status: str


class RecommendationReport(BaseModel):
    date: str
    user_id: str
    nutrition_summary: DailyNutritionSummary
    medicine_report: MedicineInteractionReport
    key_insights: List[str]
    recommendations: List[str]
    praise_points: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str]


NUTRITION_DATABASE = {
    # Grains & Carbs
    "rice_cooked": {"serving": "1 cup", "calories": 205, "protein_g": 4.3, "carbs_g": 45, "fat_g": 0.4,
                    "fiber_g": 0.6, "sugar_g": 0.1, "sodium_mg": 2, "saturated_fat_g": 0.1,
                    "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.8, "calcium_mg": 16,
                    "potassium_mg": 55, "vitamin_k_mcg": 0},
    "brown_rice_cooked": {"serving": "1 cup", "calories": 216, "protein_g": 5, "carbs_g": 45, "fat_g": 1.8,
                          "fiber_g": 3.5, "sugar_g": 0.7, "sodium_mg": 10, "saturated_fat_g": 0.4,
                          "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 1.0, "calcium_mg": 20,
                          "potassium_mg": 84, "vitamin_k_mcg": 1.2},
    "whole_wheat_bread": {"serving": "1 slice", "calories": 80, "protein_g": 4, "carbs_g": 14, "fat_g": 1,
                          "fiber_g": 2, "sugar_g": 2, "sodium_mg": 150, "saturated_fat_g": 0.2,
                          "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.9, "calcium_mg": 30,
                          "potassium_mg": 81, "vitamin_k_mcg": 0.5},
    "oatmeal_cooked": {"serving": "1 cup", "calories": 166, "protein_g": 5.9, "carbs_g": 28, "fat_g": 3.6,
                       "fiber_g": 4, "sugar_g": 0.6, "sodium_mg": 9, "saturated_fat_g": 0.6,
                       "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 2.1, "calcium_mg": 21,
                       "potassium_mg": 164, "vitamin_k_mcg": 2},
    # Proteins
    "chicken_breast_cooked": {"serving": "3 oz", "calories": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6,
                              "fiber_g": 0, "sugar_g": 0, "sodium_mg": 74, "saturated_fat_g": 1,
                              "vitamin_d_iu": 9, "vitamin_b12_mcg": 0.3, "iron_mg": 0.9, "calcium_mg": 13,
                              "potassium_mg": 256, "vitamin_k_mcg": 0.3},
    "salmon_cooked": {"serving": "3 oz", "calories": 175, "protein_g": 19, "carbs_g": 0, "fat_g": 10.5,
                      "fiber_g": 0, "sugar_g": 0, "sodium_mg": 50, "saturated_fat_g": 2.1,
                      "vitamin_d_iu": 447, "vitamin_b12_mcg": 2.4, "iron_mg": 0.3, "calcium_mg": 13,
                      "potassium_mg": 326, "vitamin_k_mcg": 0.4},
    "eggs": {"serving": "1 large", "calories": 72, "protein_g": 6.3, "carbs_g": 0.4, "fat_g": 4.8,
             "fiber_g": 0, "sugar_g": 0.2, "sodium_mg": 71, "saturated_fat_g": 1.6,
             "vitamin_d_iu": 44, "vitamin_b12_mcg": 0.6, "iron_mg": 0.9, "calcium_mg": 28,
             "potassium_mg": 69, "vitamin_k_mcg": 0.3},
    "tofu_firm": {"serving": "3 oz", "calories": 70, "protein_g": 8, "carbs_g": 2, "fat_g": 4,
                  "fiber_g": 1, "sugar_g": 0, "sodium_mg": 7, "saturated_fat_g": 0.5,
                  "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 1.8, "calcium_mg": 253,
                  "potassium_mg": 150, "vitamin_k_mcg": 2},
    "greek_yogurt": {"serving": "1 cup", "calories": 100, "protein_g": 17, "carbs_g": 6, "fat_g": 0.7,
                     "fiber_g": 0, "sugar_g": 4, "sodium_mg": 65, "saturated_fat_g": 0.2,
                     "vitamin_d_iu": 0, "vitamin_b12_mcg": 1.3, "iron_mg": 0.1, "calcium_mg": 200,
                     "potassium_mg": 240, "vitamin_k_mcg": 0.2},
    # Vegetables
    "broccoli_cooked": {"serving": "1 cup", "calories": 55, "protein_g": 3.7, "carbs_g": 11, "fat_g": 0.6,
                        "fiber_g": 5.1, "sugar_g": 2.2, "sodium_mg": 64, "saturated_fat_g": 0.1,
                        "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 1, "calcium_mg": 62,
                        "potassium_mg": 457, "vitamin_k_mcg": 220},
    "spinach_cooked": {"serving": "1 cup", "calories": 41, "protein_g": 5.3, "carbs_g": 6.8, "fat_g": 0.5,
                       "fiber_g": 4.3, "sugar_g": 0.8, "sodium_mg": 126, "saturated_fat_g": 0.1,
                       "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 6.4, "calcium_mg": 245,
                       "potassium_mg": 839, "vitamin_k_mcg": 888},
    "carrots_raw": {"serving": "1 medium", "calories": 25, "protein_g": 0.6, "carbs_g": 6, "fat_g": 0.1,
                    "fiber_g": 1.7, "sugar_g": 3, "sodium_mg": 42, "saturated_fat_g": 0,
                    "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.2, "calcium_mg": 20,
                    "potassium_mg": 195, "vitamin_k_mcg": 8},
    "sweet_potato_baked": {"serving": "1 medium", "calories": 103, "protein_g": 2.3, "carbs_g": 24, "fat_g": 0.2,
                           "fiber_g": 3.8, "sugar_g": 7.4, "sodium_mg": 41, "saturated_fat_g": 0,
                           "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.8, "calcium_mg": 43,
                           "potassium_mg": 542, "vitamin_k_mcg": 2.3},
    # Fruits
    "apple": {"serving": "1 medium", "calories": 95, "protein_g": 0.5, "carbs_g": 25, "fat_g": 0.3,
              "fiber_g": 4.4, "sugar_g": 19, "sodium_mg": 2, "saturated_fat_g": 0,
              "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.2, "calcium_mg": 11,
              "potassium_mg": 195, "vitamin_k_mcg": 4},
    "banana": {"serving": "1 medium", "calories": 105, "protein_g": 1.3, "carbs_g": 27, "fat_g": 0.4,
               "fiber_g": 3.1, "sugar_g": 14, "sodium_mg": 1, "saturated_fat_g": 0.1,
               "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.3, "calcium_mg": 6,
               "potassium_mg": 422, "vitamin_k_mcg": 0.6},
    "orange": {"serving": "1 medium", "calories": 62, "protein_g": 1.2, "carbs_g": 15, "fat_g": 0.2,
               "fiber_g": 3.1, "sugar_g": 12, "sodium_mg": 0, "saturated_fat_g": 0,
               "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.1, "calcium_mg": 52,
               "potassium_mg": 237, "vitamin_k_mcg": 0},
    "berries_mixed": {"serving": "1 cup", "calories": 70, "protein_g": 1, "carbs_g": 17, "fat_g": 0.5,
                      "fiber_g": 4, "sugar_g": 11, "sodium_mg": 1, "saturated_fat_g": 0,
                      "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.6, "calcium_mg": 20,
                      "potassium_mg": 180, "vitamin_k_mcg": 15},
    # Dairy & Alternatives
    "milk_skim": {"serving": "1 cup", "calories": 83, "protein_g": 8.3, "carbs_g": 12, "fat_g": 0.2,
                  "fiber_g": 0, "sugar_g": 12, "sodium_mg": 103, "saturated_fat_g": 0.1,
                  "vitamin_d_iu": 115, "vitamin_b12_mcg": 1.2, "iron_mg": 0.1, "calcium_mg": 299,
                  "potassium_mg": 382, "vitamin_k_mcg": 0.5},
    "cheese_cheddar": {"serving": "1 oz", "calories": 114, "protein_g": 7, "carbs_g": 0.4, "fat_g": 9.4,
                       "fiber_g": 0, "sugar_g": 0.1, "sodium_mg": 176, "saturated_fat_g": 6,
                       "vitamin_d_iu": 6, "vitamin_b12_mcg": 0.2, "iron_mg": 0.2, "calcium_mg": 204,
                       "potassium_mg": 28, "vitamin_k_mcg": 2.4},
    "almond_milk": {"serving": "1 cup", "calories": 30, "protein_g": 1, "carbs_g": 1, "fat_g": 2.5,
                    "fiber_g": 0, "sugar_g": 0, "sodium_mg": 170, "saturated_fat_g": 0,
                    "vitamin_d_iu": 100, "vitamin_b12_mcg": 0, "iron_mg": 0.4, "calcium_mg": 450,
                    "potassium_mg": 160, "vitamin_k_mcg": 0},
    # Nuts & Seeds
    "almonds": {"serving": "1 oz (23 nuts)", "calories": 164, "protein_g": 6, "carbs_g": 6, "fat_g": 14,
                "fiber_g": 3.5, "sugar_g": 1.2, "sodium_mg": 0, "saturated_fat_g": 1.1,
                "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 1.1, "calcium_mg": 76,
                "potassium_mg": 208, "vitamin_k_mcg": 0},
    "peanut_butter": {"serving": "2 tbsp", "calories": 188, "protein_g": 8, "carbs_g": 7, "fat_g": 16,
                      "fiber_g": 2, "sugar_g": 3, "sodium_mg": 147, "saturated_fat_g": 3.3,
                      "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.6, "calcium_mg": 17,
                      "potassium_mg": 208, "vitamin_k_mcg": 0},
    # Snacks & Others
    "pizza_slice": {"serving": "1 slice", "calories": 285, "protein_g": 12, "carbs_g": 36, "fat_g": 10,
                    "fiber_g": 2, "sugar_g": 4, "sodium_mg": 640, "saturated_fat_g": 4.5,
                    "vitamin_d_iu": 0, "vitamin_b12_mcg": 0.3, "iron_mg": 2.5, "calcium_mg": 220,
                    "potassium_mg": 230, "vitamin_k_mcg": 10},
    "chocolate_bar": {"serving": "1 bar (43g)", "calories": 235, "protein_g": 3, "carbs_g": 26, "fat_g": 13,
                      "fiber_g": 2, "sugar_g": 23, "sodium_mg": 35, "saturated_fat_g": 8,
                      "vitamin_d_iu": 0, "vitamin_b12_mcg": 0.3, "iron_mg": 2.3, "calcium_mg": 84,
                      "potassium_mg": 200, "vitamin_k_mcg": 5},
    "avocado": {"serving": "1/2 medium", "calories": 120, "protein_g": 1.5, "carbs_g": 6, "fat_g": 11,
                "fiber_g": 5, "sugar_g": 0.5, "sodium_mg": 5, "saturated_fat_g": 1.6,
                "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.4, "calcium_mg": 9,
                "potassium_mg": 345, "vitamin_k_mcg": 14},
    "olive_oil": {"serving": "1 tbsp", "calories": 119, "protein_g": 0, "carbs_g": 0, "fat_g": 13.5,
                  "fiber_g": 0, "sugar_g": 0, "sodium_mg": 0, "saturated_fat_g": 1.9,
                  "vitamin_d_iu": 0, "vitamin_b12_mcg": 0, "iron_mg": 0.1, "calcium_mg": 0,
                  "potassium_mg": 0, "vitamin_k_mcg": 8.1},
}


class NutritionLookup:
    """Abstraction layer for nutrition data - can be swapped with API or CSV later"""
    def __init__(self, database: Dict = None):
        self.database = database or NUTRITION_DATABASE

    def search_food(self, food_name: str) -> List[str]:
        """Fuzzy search for food names"""
        food_name_lower = food_name.lower().replace(" ", "_")
        matches = []
        for key in self.database.keys():
            if food_name_lower in key or key in food_name_lower:
                matches.append(key)
       
        # Exact match first
        if food_name_lower in self.database:
            return [food_name_lower]
       
        return matches if matches else []

    def get_nutrition(self, food_key: str, quantity: float = 1.0) -> Optional[NutrientTotals]:
        """Get nutrition info for a food, scaled by quantity"""
        if food_key not in self.database:
            return None

        data = self.database[food_key]
       
        return NutrientTotals(
            calories=data["calories"] * quantity,
            protein_g=data["protein_g"] * quantity,
            carbs_g=data["carbs_g"] * quantity,
            fat_g=data["fat_g"] * quantity,
            fiber_g=data["fiber_g"] * quantity,
            sugar_g=data["sugar_g"] * quantity,
            sodium_mg=data["sodium_mg"] * quantity,
            saturated_fat_g=data["saturated_fat_g"] * quantity,
            vitamin_d_iu=data["vitamin_d_iu"] * quantity,
            vitamin_b12_mcg=data["vitamin_b12_mcg"] * quantity,
            iron_mg=data["iron_mg"] * quantity,
            calcium_mg=data["calcium_mg"] * quantity,
            potassium_mg=data["potassium_mg"] * quantity,
            vitamin_k_mcg=data["vitamin_k_mcg"] * quantity,
        )
   
    def get_serving_info(self, food_key: str) -> Optional[str]:
        """Get serving size information"""
        if food_key not in self.database:
            return None
        return self.database[food_key]["serving"]


nutrition_lookup = NutritionLookup()
print(f"âœ“ Nutrition database loaded with {len(NUTRITION_DATABASE)} foods")


class GoalDerivationEngine:
    """Derives personalized nutrition goals based on user profile and conditions"""
    @staticmethod
    def calculate_bmr(profile: UserProfile) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor if data available"""
        if not all([profile.weight_kg, profile.height_cm, profile.age, profile.sex]):
            return None
       
        if profile.sex.lower() == "male":
            bmr = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age + 5
        else:
            bmr = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age - 161
        return bmr
   
    @staticmethod
    def apply_activity_multiplier(bmr: float, activity_level: ActivityLevel) -> float:
        """Apply activity level multiplier to BMR"""
        multipliers = {
            ActivityLevel.SEDENTARY: 1.2,
            ActivityLevel.LIGHTLY_ACTIVE: 1.375,
            ActivityLevel.MODERATELY_ACTIVE: 1.55,
            ActivityLevel.VERY_ACTIVE: 1.725,
            ActivityLevel.EXTREMELY_ACTIVE: 1.9,
        }
        return bmr * multipliers[activity_level]
    @staticmethod
    def derive_goals(profile: UserProfile, custom_calorie_target: Optional[float] = None) -> NutritionGoal:
        """Derive nutrition goals from user profile"""
        # Calculate baseline calories
        bmr = GoalDerivationEngine.calculate_bmr(profile)
        if bmr:
            tdee = GoalDerivationEngine.apply_activity_multiplier(bmr, profile.activity_level)
            base_calories = tdee
        else:
            # Default if no physical data
            base_calories = 2000

        # Override with custom target if provided
        if custom_calorie_target:
            base_calories = custom_calorie_target
       
        # Adjust for conditions
        for condition in profile.conditions:
            if condition == UserCondition.PRE_DIABETIC or condition == UserCondition.DIABETIC:
                # Lower calorie target for weight management
                base_calories = min(base_calories, 1800)
            elif condition == UserCondition.HEART_DISEASE:
                base_calories = min(base_calories, 1800)

        # Set calorie range (Â±10%)
        cal_min = base_calories * 0.9
        cal_max = base_calories * 1.1

        # Default macro distribution (40% carbs, 30% protein, 30% fat)
        protein_g_min = (base_calories * 0.25) / 4  # 25-35% of calories
        protein_g_max = (base_calories * 0.35) / 4
        carbs_g_min = (base_calories * 0.35) / 4    # 35-45% of calories
        carbs_g_max = (base_calories * 0.45) / 4
        fat_g_min = (base_calories * 0.25) / 9      # 25-35% of calories
        fat_g_max = (base_calories * 0.35) / 9

        # Fiber (14g per 1000 calories)
        fiber_g_min = (base_calories / 1000) * 14
       
        # Sugar limits
        sugar_g_max = base_calories * 0.10 / 4  # Max 10% of calories from added sugar
       
        # Sodium
        sodium_mg_max = 2300  # Standard recommendation

        # Saturated fat
        saturated_fat_g_max = (base_calories * 0.10) / 9  # Max 10% of calories

        # Adjust for specific conditions
        for condition in profile.conditions:
            if condition == UserCondition.PRE_DIABETIC or condition == UserCondition.DIABETIC:
                # Lower carbs, especially sugar
                carbs_g_max = (base_calories * 0.40) / 4
                sugar_g_max = base_calories * 0.05 / 4  # Max 5% from sugar
                fiber_g_min = (base_calories / 1000) * 20  # Higher fiber
            elif condition == UserCondition.HYPERTENSION:
                # Lower sodium
                sodium_mg_max = 1500
            elif condition == UserCondition.KIDNEY_DISEASE:
                # Lower protein and sodium
                protein_g_max = (base_calories * 0.25) / 4
                sodium_mg_max = 2000
            elif condition == UserCondition.HEART_DISEASE:
                # Lower saturated fat and sodium
                saturated_fat_g_max = (base_calories * 0.07) / 9
                sodium_mg_max = 1500

        return NutritionGoal(
            calories_min=cal_min,
            calories_max=cal_max,
            protein_g_min=protein_g_min,
            protein_g_max=protein_g_max,
            carbs_g_min=carbs_g_min,
            carbs_g_max=carbs_g_max,
            fat_g_min=fat_g_min,
            fat_g_max=fat_g_max,
            fiber_g_min=fiber_g_min,
            sugar_g_max=sugar_g_max,
            sodium_mg_max=sodium_mg_max,
            saturated_fat_g_max=saturated_fat_g_max,
            vitamin_d_iu_min=600,
            vitamin_b12_mcg_min=2.4,
            iron_mg_min=8 if profile.sex == "male" else 18,
            calcium_mg_min=1000,
            potassium_mg_min=2600 if profile.sex == "female" else 3400,
        )


class RestrictionChecker:
    """Checks if foods violate dietary restrictions or allergies"""
    # Food categories for restrictions
    ANIMAL_PRODUCTS = ["chicken_breast_cooked", "salmon_cooked", "eggs", "milk_skim",
                       "cheese_cheddar", "greek_yogurt", "pizza_slice"]
    MEAT_PRODUCTS = ["chicken_breast_cooked", "salmon_cooked"]
    DAIRY_PRODUCTS = ["milk_skim", "cheese_cheddar", "greek_yogurt"]
    GLUTEN_CONTAINING = ["whole_wheat_bread", "pizza_slice", "oatmeal_cooked"]
    
    # Allergy mappings
    ALLERGY_FOODS = {
        "nuts": ["almonds", "peanut_butter"],
        "peanuts": ["peanut_butter"],
        "tree nuts": ["almonds"],
        "dairy": ["milk_skim", "cheese_cheddar", "greek_yogurt"],
        "eggs": ["eggs"],
        "fish": ["salmon_cooked"],
        "shellfish": [],
        "soy": ["tofu_firm"],
        "wheat": ["whole_wheat_bread", "pizza_slice"],
        "gluten": ["whole_wheat_bread", "pizza_slice", "oatmeal_cooked"],
    }

    @staticmethod
    def check_restrictions(food_key: str, restrictions: List[DietaryRestriction]) -> List[str]:
        """Check if food violates dietary restrictions"""
        violations = []
       
        for restriction in restrictions:
            if restriction == DietaryRestriction.VEGAN:
                if food_key in RestrictionChecker.ANIMAL_PRODUCTS:
                    violations.append(f"{food_key} violates vegan restriction")
            elif restriction == DietaryRestriction.VEGETARIAN:
                if food_key in RestrictionChecker.MEAT_PRODUCTS:
                    violations.append(f"{food_key} violates vegetarian restriction")
            elif restriction == DietaryRestriction.LACTOSE_INTOLERANT:
                if food_key in RestrictionChecker.DAIRY_PRODUCTS:
                    violations.append(f"{food_key} contains lactose")
            elif restriction == DietaryRestriction.GLUTEN_FREE:
                if food_key in RestrictionChecker.GLUTEN_CONTAINING:
                    violations.append(f"{food_key} contains gluten")
        return violations

    @staticmethod
    def check_allergies(food_key: str, allergies: List[str]) -> List[str]:
        """Check if food triggers allergies"""
        violations = []
        for allergy in allergies:
            allergy_lower = allergy.lower()
            if allergy_lower in RestrictionChecker.ALLERGY_FOODS:
                if food_key in RestrictionChecker.ALLERGY_FOODS[allergy_lower]:
                    violations.append(f"âš ï¸� ALLERGY WARNING: {food_key} contains {allergy}")
        return violations


def lookup_food_nutrition(food_name: str, quantity: float = 1.0) -> str:
    """
    Tool to lookup nutrition information for a food item.
    Returns nutrition data scaled by quantity.
    """
    matches = nutrition_lookup.search_food(food_name)
   
    if not matches:
        return json.dumps({
            "success": False,
            "message": f"Food '{food_name}' not found in database. Try a different name or add it manually.",
            "suggestions": []
        })
   
    # Use first match
    food_key = matches[0]
    nutrition = nutrition_lookup.get_nutrition(food_key, quantity)
    serving = nutrition_lookup.get_serving_info(food_key)

    if nutrition:
        return json.dumps({
            "success": True,
            "food_key": food_key,
            "serving_size": serving,
            "quantity_multiplier": quantity,
            "nutrition": nutrition.model_dump()
        }, indent=2)

    return json.dumps({"success": False, "message": "Error retrieving nutrition data"})

def aggregate_daily_nutrition(food_entries_json: str) -> str:
    """
    Tool to aggregate nutrition from multiple food entries.
    Expects JSON string of list of FoodLogEntry objects.
    """
    try:
        entries_data = json.loads(food_entries_json)
        entries = [FoodLogEntry(**entry) for entry in entries_data]
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
   
    total = NutrientTotals()
    processed_foods = []

    for entry in entries:
        matches = nutrition_lookup.search_food(entry.food_name)
        if matches:
            food_key = matches[0]
            nutrition = nutrition_lookup.get_nutrition(food_key, entry.quantity)
            if nutrition:
                # Add to totals
                total.calories += nutrition.calories
                total.protein_g += nutrition.protein_g
                total.carbs_g += nutrition.carbs_g
                total.fat_g += nutrition.fat_g
                total.fiber_g += nutrition.fiber_g
                total.sugar_g += nutrition.sugar_g
                total.sodium_mg += nutrition.sodium_mg
                total.saturated_fat_g += nutrition.saturated_fat_g
                total.vitamin_d_iu += nutrition.vitamin_d_iu
                total.vitamin_b12_mcg += nutrition.vitamin_b12_mcg
                total.iron_mg += nutrition.iron_mg
                total.calcium_mg += nutrition.calcium_mg
                total.potassium_mg += nutrition.potassium_mg
                total.vitamin_k_mcg += nutrition.vitamin_k_mcg
               
                processed_foods.append({
                    "food": entry.food_name,
                    "matched_key": food_key,
                    "quantity": entry.quantity,
                    "meal": entry.meal
                })

    return json.dumps({
        "success": True,
        "total_nutrients": total.model_dump(),
        "processed_foods": processed_foods,
        "total_foods": len(processed_foods)
    }, indent=2)

 
def compare_to_goals(nutrient_totals_json: str, goals_json: str, profile_json: str) -> str:
    """
    Tool to compare actual nutrient intake to goals.
    Returns status for each nutrient and flags violations.
    """
    try:
        totals = NutrientTotals(**json.loads(nutrient_totals_json))
        goals = NutritionGoal(**json.loads(goals_json))
        profile = UserProfile(**json.loads(profile_json))
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
    
    statuses = []
   
    # Check calories
    cal_status = "within_target"
    if totals.calories < goals.calories_min:
        cal_status = "below_target"
    elif totals.calories > goals.calories_max * 1.2:
        cal_status = "significantly_over"
    elif totals.calories > goals.calories_max:
        cal_status = "slightly_over"
   
    statuses.append(GoalStatus(
        nutrient="calories",
        target_min=goals.calories_min,
        target_max=goals.calories_max,
        actual=totals.calories,
        status=cal_status,
        percentage_of_target=(totals.calories / goals.calories_max) * 100
    ))

    # Check protein
    prot_status = "within_target"
    if totals.protein_g < goals.protein_g_min:
        prot_status = "below_target"
    elif totals.protein_g > goals.protein_g_max:
        prot_status = "slightly_over"
    statuses.append(GoalStatus(
        nutrient="protein",
        target_min=goals.protein_g_min,
        target_max=goals.protein_g_max,
        actual=totals.protein_g,
        status=prot_status,
        percentage_of_target=(totals.protein_g / goals.protein_g_max) * 100
    ))
   
    # Check carbs
    carb_status = "within_target"
    if totals.carbs_g < goals.carbs_g_min:
        carb_status = "below_target"
    elif totals.carbs_g > goals.carbs_g_max * 1.2:
        carb_status = "significantly_over"
    elif totals.carbs_g > goals.carbs_g_max:
        carb_status = "slightly_over"

    statuses.append(GoalStatus(
        nutrient="carbs",
        target_min=goals.carbs_g_min,
        target_max=goals.carbs_g_max,
        actual=totals.carbs_g,
        status=carb_status,
        percentage_of_target=(totals.carbs_g / goals.carbs_g_max) * 100
    ))

    # Check sugar (critical for diabetic/pre-diabetic)
    sugar_status = "within_target"
    if totals.sugar_g > goals.sugar_g_max * 1.5:
        sugar_status = "significantly_over"
    elif totals.sugar_g > goals.sugar_g_max:
        sugar_status = "slightly_over"
   
    statuses.append(GoalStatus(
        nutrient="sugar",
        target_max=goals.sugar_g_max,
        actual=totals.sugar_g,
        status=sugar_status,
        percentage_of_target=(totals.sugar_g / goals.sugar_g_max) * 100
    ))

    # Check sodium
    sodium_status = "within_target"
    if totals.sodium_mg > goals.sodium_mg_max * 1.3:
        sodium_status = "significantly_over"
    elif totals.sodium_mg > goals.sodium_mg_max:
        sodium_status = "slightly_over"

    statuses.append(GoalStatus(
        nutrient="sodium",
        target_max=goals.sodium_mg_max,
        actual=totals.sodium_mg,
        status=sodium_status,
        percentage_of_target=(totals.sodium_mg / goals.sodium_mg_max) * 100
    ))

    # Check fiber
    fiber_status = "within_target"
    if totals.fiber_g < goals.fiber_g_min * 0.7:
        fiber_status = "below_target"
    
    statuses.append(GoalStatus(
        nutrient="fiber",
        target_min=goals.fiber_g_min,
        actual=totals.fiber_g,
        status=fiber_status,
        percentage_of_target=(totals.fiber_g / goals.fiber_g_min) * 100 if goals.fiber_g_min > 0 else 100
    ))

    return json.dumps({
        "success": True,
        "goal_statuses": [s.model_dump() for s in statuses],
        "critical_issues": [
            s.nutrient for s in statuses
            if s.status == "significantly_over" or (s.status == "below_target" and s.nutrient in ["protein", "fiber"])
        ]
    }, indent=2)


def check_restrictions_and_allergies(food_entries_json: str, profile_json: str) -> str:
    """
    Tool to check if any foods violate dietary restrictions or trigger allergies.
    """
    try:
        entries_data = json.loads(food_entries_json)
        entries = [FoodLogEntry(**entry) for entry in entries_data]
        profile = UserProfile(**json.loads(profile_json))
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

    restriction_violations = []
    allergy_violations = []
    for entry in entries:
        matches = nutrition_lookup.search_food(entry.food_name)
        if matches:
            food_key = matches[0]
            # Check restrictions
            rest_viols = RestrictionChecker.check_restrictions(food_key, profile.dietary_restrictions)
            restriction_violations.extend(rest_viols)
            # Check allergies
            allergy_viols = RestrictionChecker.check_allergies(food_key, profile.allergies)
            allergy_violations.extend(allergy_viols)
   
    return json.dumps({
        "success": True,
        "restriction_violations": restriction_violations,
        "allergy_violations": allergy_violations,
        "safe_to_consume": len(allergy_violations) == 0
    }, indent=2)


nutrition_tools = [lookup_food_nutrition,aggregate_daily_nutrition,compare_to_goals,check_restrictions_and_allergies]


nutrition_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="NutritionAgent",
    instruction="""You are a nutrition analysis specialist. Your role is to:
    1. Parse and interpret food intake logs
    2. Look up nutrition information for foods
    3. Aggregate total daily nutrient intake
    4. Compare intake against personalized nutrition goals
    5. Check for dietary restriction and allergy violations
    6. Provide clear, actionable nutrition feedback

    Use the available tools to analyze the user's food intake and generate a comprehensive
    DailyNutritionSummary. Focus on identifying whether the user met their goals, exceeded
    limits (especially calories, sugar, sodium), or fell short on important nutrients.
    Be encouraging when goals are met, and constructive when suggesting improvements.""",
    tools=nutrition_tools,
)

print("âœ“ Nutrition Agent and tools created")


class MedicineInteractionEngine:
    """Rules engine for medicine-nutrient and supplement interactions"""
    # Nutrient upper limits (Tolerable Upper Intake Levels)
    NUTRIENT_UPPER_LIMITS = {
        "vitamin_d_iu": 4000,
        "vitamin_b12_mcg": 1000,  # Generally safe, but flag high doses
        "iron_mg": 45,
        "calcium_mg": 2500,
        "vitamin_k_mcg": 1000,  # Important for anticoagulant interactions
        "sodium_mg": 2300,
    }

    # Medicine-nutrient interactions (simplified, educational)
    MEDICINE_INTERACTIONS = {
        "warfarin": {
            "nutrient": "vitamin_k_mcg",
            "threshold": 200,  # Daily vitamin K intake
            "severity": "high",
            "description": "High vitamin K intake can interfere with warfarin effectiveness",
            "recommendation": "Maintain consistent vitamin K intake; avoid sudden large increases from leafy greens"
        },

        "coumadin": {
            "nutrient": "vitamin_k_mcg",
            "threshold": 200,
            "severity": "high",
            "description": "High vitamin K intake can interfere with anticoagulant effectiveness",
            "recommendation": "Maintain consistent vitamin K intake"
        },

        "lisinopril": {
            "nutrient": "potassium_mg",
            "threshold": 4000,
            "severity": "medium",
            "description": "ACE inhibitors can increase potassium levels; high dietary potassium may be risky",
            "recommendation": "Monitor potassium intake; avoid excessive potassium supplements"
        },

        "hydrochlorothiazide": {
            "nutrient": "sodium_mg",
            "threshold": 1500,
            "severity": "medium",
            "description": "Diuretic effectiveness reduced with high sodium intake",
            "recommendation": "Limit sodium to enhance medication effectiveness"
        },

        "levothyroxine": {
            "nutrient": "calcium_mg",
            "threshold": 1000,
            "severity": "low",
            "description": "High calcium can interfere with thyroid medication absorption",
            "recommendation": "Take medication 4 hours apart from calcium-rich foods or supplements"
        },
        
        "metformin": {
            "nutrient": "vitamin_b12_mcg",
            "threshold": 2.4,
            "severity": "low",
            "description": "Long-term metformin use can reduce B12 absorption",
            "recommendation": "Ensure adequate B12 intake; consider supplementation if deficient"
        },

    }

   

    # Foods that interact with certain medications
    FOOD_MEDICINE_INTERACTIONS = {
        "grapefruit": [
            "simvastatin", "atorvastatin", "lovastatin",  # Statins
            "amlodipine", "felodipine",  # Calcium channel blockers
        ],
    }

   

    @staticmethod
    def check_supplement_limits(nutrient_totals: NutrientTotals, supplements: List[Supplement]) -> List[str]:
        """Check if supplements + diet exceed safe upper limits"""
        concerns = []
    
        # Calculate total intake including supplements
        total_with_supplements = {
            "vitamin_d_iu": nutrient_totals.vitamin_d_iu,
            "vitamin_b12_mcg": nutrient_totals.vitamin_b12_mcg,
            "iron_mg": nutrient_totals.iron_mg,
            "calcium_mg": nutrient_totals.calcium_mg,
        }

       

        for supplement in supplements:
            nutrient_key = supplement.nutrient.lower().replace(" ", "_").replace("-", "_")
           
            # Map supplement nutrients to our tracking
            if "vitamin d" in supplement.nutrient.lower():
                total_with_supplements["vitamin_d_iu"] += supplement.dose_amount
            elif "b12" in supplement.nutrient.lower() or "b-12" in supplement.nutrient.lower():
                total_with_supplements["vitamin_b12_mcg"] += supplement.dose_amount
            elif "iron" in supplement.nutrient.lower():
                total_with_supplements["iron_mg"] += supplement.dose_amount
            elif "calcium" in supplement.nutrient.lower():
                total_with_supplements["calcium_mg"] += supplement.dose_amount
       
        # Check against limits
        for nutrient, total in total_with_supplements.items():
            if nutrient in MedicineInteractionEngine.NUTRIENT_UPPER_LIMITS:
                limit = MedicineInteractionEngine.NUTRIENT_UPPER_LIMITS[nutrient]
                if total > limit:
                    concerns.append(
                        f"âš ï¸� {nutrient.replace('_', ' ').title()}: {total:.1f} exceeds safe upper limit of {limit}"
                    ) 

        return concerns

    @staticmethod
    def check_medicine_nutrient_interactions(
        nutrient_totals: NutrientTotals,
        medicines: List[Medicine]
    ) -> List[InteractionWarning]:
        """Check for medicine-nutrient interactions"""
        warnings = []

        for medicine in medicines:
            med_name_lower = medicine.name.lower()
           
            # Check if this medicine has known interactions
            for med_key, interaction in MedicineInteractionEngine.MEDICINE_INTERACTIONS.items():
                if med_key in med_name_lower:
                    nutrient = interaction["nutrient"]
                    threshold = interaction["threshold"]
                   
                    # Get actual intake
                    actual = getattr(nutrient_totals, nutrient, 0)

                    if actual > threshold:
                        warnings.append(InteractionWarning(
                            severity=interaction["severity"],
                            medicine_or_supplement=medicine.name,
                            nutrient_or_food=nutrient.replace("_", " ").title(),
                            description=interaction["description"],
                            recommendation=interaction["recommendation"]
                        ))
        return warnings

    @staticmethod
    def generate_medicine_report(
        date: str,
        user_id: str,
        nutrient_totals: NutrientTotals,
        profile: UserProfile
    ) -> MedicineInteractionReport:
        """Generate comprehensive medicine interaction report"""
        warnings = MedicineInteractionEngine.check_medicine_nutrient_interactions(
            nutrient_totals, profile.medicines
        )

        upper_limit_concerns = MedicineInteractionEngine.check_supplement_limits(
            nutrient_totals, profile.supplements
        )

        # Determine overall safety status
        if any(w.severity == "high" for w in warnings):
            safety_status = "âš ï¸� HIGH PRIORITY: Review with healthcare provider"
        elif any(w.severity == "medium" for w in warnings) or upper_limit_concerns:
            safety_status = "âš ï¸� CAUTION: Monitor interactions"
        elif warnings:
            safety_status = "âœ“ Low concern, but be aware"
        else:
            safety_status = "âœ“ No significant interactions detected"
        return MedicineInteractionReport(
            date=date,
            user_id=user_id,
            warnings=warnings,
            nutrient_upper_limit_concerns=upper_limit_concerns,
            overall_safety_status=safety_status
        )


def check_medicine_interactions(nutrient_totals_json: str, profile_json: str, date: str) -> str:
    """
    Tool to check for medicine-nutrient and supplement interactions.
    Returns a MedicineInteractionReport.
    """
    try:
        totals = NutrientTotals(**json.loads(nutrient_totals_json))
        profile = UserProfile(**json.loads(profile_json))
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

    report = MedicineInteractionEngine.generate_medicine_report(
        date=date,
        user_id=profile.user_id,
        nutrient_totals=totals,
        profile=profile
    )

    return json.dumps({
        "success": True,
        "report": report.model_dump()
    }, indent=2)

 
def get_medicine_schedule(profile_json: str) -> str:
    """
    Tool to get the user's medicine and supplement schedule.
    """
    try:
        profile = UserProfile(**json.loads(profile_json))
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
   
    schedule = {
        "medicines": [
            {
                "name": med.name,
                "dosage": med.dosage,
                "frequency": med.frequency,
                "condition": med.condition_treated
            }
            for med in profile.medicines
        ],

        "supplements": [
            {
                "name": supp.name,
                "nutrient": supp.nutrient,
                "dose": f"{supp.dose_amount} {supp.dose_unit}",
                "frequency": supp.frequency
            }
            for supp in profile.supplements
        ]
    }

    return json.dumps({
        "success": True,
        "schedule": schedule,
        "total_medicines": len(profile.medicines),
        "total_supplements": len(profile.supplements)

    }, indent=2)


medicine_tools = [check_medicine_interactions,get_medicine_schedule]

medicine_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="MedicineAgent",
    instruction="""You are a medicine and supplement interaction specialist. Your role is to:
    1. Track the user's medicines and supplements
    2. Check for potential medicine-nutrient interactions
    3. Verify that nutrient intake (diet + supplements) doesn't exceed safe upper limits
    4. Identify potential concerns with specific food-medicine combinations
    5. Provide clear safety recommendations
   
    IMPORTANT DISCLAIMER: Your analysis is educational and based on general guidelines.
    Always recommend users consult their healthcare provider for personalized medical advice,
    especially for high-severity interactions.
	
    Use the available tools to analyze medicine and supplement interactions with the user's
    daily nutrition intake.""",
    tools=medicine_tools,
)

print("âœ“ Medicine Agent and interaction rules created")


class UserMemoryStore:
    """Simple in-memory store for user profiles and long-term patterns"""
    def __init__(self):
        self.memories: Dict[str, UserMemory] = {}
   
    def save_memory(self, user_memory: UserMemory):
        """Save or update user memory"""
        self.memories[user_memory.user_id] = user_memory

    def get_memory(self, user_id: str) -> Optional[UserMemory]:
        """Retrieve user memory"""
        return self.memories.get(user_id)
   
    def add_dietary_pattern(self, user_id: str, pattern: str):
        """Add a dietary pattern observation to user's memory"""
        if user_id in self.memories:
            if pattern not in self.memories[user_id].dietary_patterns:
                self.memories[user_id].dietary_patterns.append(pattern)
                self.memories[user_id].last_updated = datetime.now().isoformat()

    def get_dietary_patterns(self, user_id: str) -> List[str]:
        """Get user's dietary patterns"""
        if user_id in self.memories:
            return self.memories[user_id].dietary_patterns
        return []
 
# Initialize global memory store
user_memory_store = UserMemoryStore()


def parse_user_profile(profile_description: str) -> str:
    """
    Tool to parse natural language user profile into structured UserProfile.
    """
    try:
        # Try to parse as JSON first
        profile_data = json.loads(profile_description)
        profile = UserProfile(**profile_data)
       
        return json.dumps({
            "success": True,
            "profile": profile.model_dump()
        }, indent=2)

    except json.JSONDecodeError:
        # If not JSON, return instruction for proper format
        return json.dumps({
            "success": False,
            "message": "Please provide profile as JSON with fields: user_id, age, sex, weight_kg, height_cm, activity_level, conditions, dietary_restrictions, allergies, medicines, supplements"
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

def derive_nutrition_goals(profile_json: str, custom_calorie_target: Optional[float] = None) -> str:
    """
    Tool to derive personalized nutrition goals from user profile.
    """
    try:
        profile = UserProfile(**json.loads(profile_json))
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
    goals = GoalDerivationEngine.derive_goals(profile, custom_calorie_target)
   
    return json.dumps({
        "success": True,
        "goals": goals.model_dump(),
        "derivation_notes": {
            "based_on_conditions": [c.value for c in profile.conditions],
            "custom_calorie_override": custom_calorie_target is not None
        }
    }, indent=2)

def save_user_memory(profile_json: str) -> str:
    """
    Tool to save user profile to long-term memory.
    """
    try:
        profile = UserProfile(**json.loads(profile_json))
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

    # Check if user already has memory
    existing_memory = user_memory_store.get_memory(profile.user_id)
   
    if existing_memory:
        # Update existing memory
        existing_memory.profile = profile
        existing_memory.last_updated = datetime.now().isoformat()
        user_memory_store.save_memory(existing_memory)
        action = "updated"
    else:
        # Create new memory
        new_memory = UserMemory(
            user_id=profile.user_id,
            profile=profile,
            dietary_patterns=[]
        )

        user_memory_store.save_memory(new_memory)
        action = "created"

    return json.dumps({
        "success": True,
        "action": action,
        "user_id": profile.user_id
    })

def retrieve_user_memory(user_id: str) -> str:
    """
    Tool to retrieve user's stored profile and dietary patterns from memory.
    """
    memory = user_memory_store.get_memory(user_id)
  
    if memory:
        return json.dumps({
            "success": True,
            "memory_found": True,
            "profile": memory.profile.model_dump(),
            "dietary_patterns": memory.dietary_patterns,
            "last_updated": memory.last_updated
        }, indent=2)
    else:
        return json.dumps({
            "success": True,
            "memory_found": False,
            "message": f"No stored memory for user {user_id}"
        })


def add_dietary_pattern_to_memory(user_id: str, pattern: str) -> str:
    """
    Tool to add a dietary pattern observation to user's long-term memory.
    Examples: 'frequently exceeds sugar at breakfast', 'often forgets to log snacks'
    """
    user_memory_store.add_dietary_pattern(user_id, pattern)
   
    return json.dumps({
        "success": True,
        "pattern_added": pattern,
        "total_patterns": len(user_memory_store.get_dietary_patterns(user_id))
    })


profile_tools = [parse_user_profile, derive_nutrition_goals,save_user_memory,retrieve_user_memory,add_dietary_pattern_to_memory]

profile_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="ProfileAgent",
    instruction="""You are a user profile management specialist. Your role is to:
    1. Parse and validate user profile information
    2. Retrieve stored user profiles and dietary patterns from long-term memory
    3. Derive personalized nutrition goals based on user's conditions, restrictions, and preferences
    4. Save updated profiles to memory
    5. Track long-term dietary patterns and issues
   
    When a user provides their information, first check if they have existing memory.
    If they do, retrieve it and note any patterns from previous sessions.

    Always validate that required fields are present and make sense.
    Be helpful in explaining what information is needed if the profile is incomplete.
   
    Use the memory system to provide continuity across sessions and personalize advice
    based on historical patterns.""",
    tools=profile_tools,
)

print("âœ“ Profile Agent and memory system created")


from google.adk import runners
from google.adk.agents import ParallelAgent
from google.genai import types
import asyncio


# Parallel agent that runs Nutrition and Medicine agents concurrently.
parallel_health_agent = ParallelAgent(
    name="ParallelHealthAgent",
    description="Runs nutrition and medicine agents in parallel over the same daily context.",
    sub_agents=[nutrition_agent, medicine_agent],
)

async def _run_parallel_commentary_async(prompt: str) -> Dict[str, str]:
    """Run the parallel agent via an inâ€‘memory runner and collect perâ€‘agent commentary."""
    runner = runners.InMemoryRunner(
        agent=parallel_health_agent,
        app_name="nutrition_app",
    )
    session = await runner.session_service.create_session(
        app_name="nutrition_app",
        user_id="demo_user",
    )
    user_message = types.Content(role="user", parts=[types.Part(text=prompt)])
    per_agent_text: Dict[str, str] = {}
 
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            texts = [
                getattr(part, "text", "")
                for part in event.content.parts
                if getattr(part, "text", None)
            ]
            if not texts:
                continue
            author = getattr(event, "author", parallel_health_agent.name)
            per_agent_text.setdefault(author, "")
            per_agent_text[author] += " ".join(texts).strip() + "\n"
 
    return per_agent_text

 

def run_parallel_commentary(prompt: str) -> Dict[str, str]:
    """Synchronous helper to get commentary from the ParallelHealthAgent."""
    try:
        return asyncio.run(_run_parallel_commentary_async(prompt))
    except RuntimeError:
        # If an event loop is already running (e.g. in some notebook environments),
        # reuse it instead of creating a new one.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_run_parallel_commentary_async(prompt))
 
def analyze_nutrition_parallel(
    profile_json: str,
    goals_json: str,
    food_entries_json: str,
    date: str,
) -> str:
    """
    Tool that coordinates nutrition and medicine analysis.
    It computes structured nutrition and medicine reports with deterministic
    Python logic, then uses an ADK ParallelAgent to have NutritionAgent and
    MedicineAgent generate natural language commentary in parallel.
    """
    try:
        profile = UserProfile(**json.loads(profile_json))
        goals = NutritionGoal(**json.loads(goals_json))
        entries_data = json.loads(food_entries_json)
        entries = [FoodLogEntry(**entry) for entry in entries_data]
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
 
    # Aggregate nutrition deterministically.
    total = NutrientTotals()
    processed_foods = []
    restriction_violations: List[str] = []
    allergy_violations: List[str] = []
 
    for entry in entries:
        matches = nutrition_lookup.search_food(entry.food_name)
        if matches:
            food_key = matches[0]
            nutrition = nutrition_lookup.get_nutrition(food_key, entry.quantity)
 
            if nutrition:
                # Add to totals
                for field in total.model_fields:
                    current = getattr(total, field)
                    to_add = getattr(nutrition, field)
                    setattr(total, field, current + to_add)
 
                processed_foods.append(
                    {
                        "food": entry.food_name,
                        "matched_key": food_key,
                        "quantity": entry.quantity,
                        "meal": entry.meal,
                    }
                )
 
                # Check restrictions and allergies
                restriction_violations.extend(
                    RestrictionChecker.check_restrictions(
                        food_key, profile.dietary_restrictions
                    )
                )
                allergy_violations.extend(
                    RestrictionChecker.check_allergies(food_key, profile.allergies)
                )

    # Compare to goals
    goal_statuses: List[Dict[str, Any]] = []
 
    # Calories
    cal_status = "within_target"
    if total.calories < goals.calories_min:
        cal_status = "below_target"
    elif total.calories > goals.calories_max * 1.2:
        cal_status = "significantly_over"
    elif total.calories > goals.calories_max:
        cal_status = "slightly_over"
 
    goal_statuses.append(
        {
            "nutrient": "calories",
            "target_min": goals.calories_min,
            "target_max": goals.calories_max,
            "actual": total.calories,
            "status": cal_status,
            "percentage_of_target": (total.calories / goals.calories_max) * 100,
        }
    )
 
    # Sugar (critical)
    sugar_status = "within_target"
    if total.sugar_g > goals.sugar_g_max * 1.5:
        sugar_status = "significantly_over"
    elif total.sugar_g > goals.sugar_g_max:
        sugar_status = "slightly_over"
 
    goal_statuses.append(
        {
            "nutrient": "sugar",
            "target_max": goals.sugar_g_max,
            "actual": total.sugar_g,
            "status": sugar_status,
            "percentage_of_target": (total.sugar_g / goals.sugar_g_max) * 100,
        }
    )
 
    # Sodium
    sodium_status = "within_target"
    if total.sodium_mg > goals.sodium_mg_max * 1.3:
        sodium_status = "significantly_over"
    elif total.sodium_mg > goals.sodium_mg_max:
        sodium_status = "slightly_over"
 
    goal_statuses.append(
        {
            "nutrient": "sodium",
            "target_max": goals.sodium_mg_max,
            "actual": total.sodium_mg,
            "status": sodium_status,
            "percentage_of_target": (total.sodium_mg / goals.sodium_mg_max) * 100,
        }
    )
    # Create nutrition summary
    overall = (
        "Goals met"
        if all(s["status"] == "within_target" for s in goal_statuses)
        else "Some goals exceeded or not met"
    )
 
    nutrition_summary = {
        "date": date,
        "user_id": profile.user_id,
        "food_entries": [e.model_dump() for e in entries],
        "nutrient_totals": total.model_dump(),
        "goal_statuses": goal_statuses,
        "restriction_violations": restriction_violations,
        "allergy_violations": allergy_violations,
        "overall_assessment": overall,
    }

    # Deterministic medicine report
    medicine_report = MedicineInteractionEngine.generate_medicine_report(
        date=date,
        user_id=profile.user_id,
        nutrient_totals=total,
        profile=profile,
    )

    # Use ADK ParallelAgent to generate natural-language commentary in parallel.
    prompt = (
        "You are part of a multi-agent nutrition analysis system.\n\n"
        f"User profile (JSON):\n{profile_json}\n\n"
        f"Goals (JSON):\n{goals_json}\n\n"
        f"Aggregated nutrient totals (JSON):\n{json.dumps(total.model_dump(), indent=2)}\n\n"
        f"Dietary restriction violations: {restriction_violations}\n\n"
        f"Allergy violations: {allergy_violations}\n\n"
        "As the agent named in the 'author' field, provide a short, user-friendly "
        "summary (2â€“3 bullet points) of the key findings from your perspective. "
        "Do not repeat raw JSON; focus on insights."
    )
    agent_commentary = run_parallel_commentary(prompt)
 
    return json.dumps(
        {
            "success": True,
            "nutrition_summary": nutrition_summary,
            "medicine_report": medicine_report.model_dump(),
            "parallel_commentary": agent_commentary,
            "note": "Nutrition and medicine analyses computed deterministically, "
            "with parallel agents generating commentary via ADK.",
        },
        indent=2,
    )



def generate_final_recommendations(
    nutrition_summary_json: str,
    medicine_report_json: str,
    profile_json: str
) -> str:
    """
    Tool to generate final comprehensive recommendations by merging
    nutrition and medicine analyses.
    """
    try:
        nutrition_summary = DailyNutritionSummary(**json.loads(nutrition_summary_json))
        medicine_report = MedicineInteractionReport(**json.loads(medicine_report_json))
        profile = UserProfile(**json.loads(profile_json))
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

    key_insights = []
    recommendations = []
    praise_points = []
    areas_for_improvement = []
    # Analyze nutrition results
    for status in nutrition_summary.goal_statuses:
        if status.status == "within_target":
            praise_points.append(f"âœ“ {status.nutrient.title()} intake is on target")
        elif status.status == "significantly_over":
            key_insights.append(f"âš ï¸� {status.nutrient.title()} significantly exceeded ({status.percentage_of_target:.0f}% of target)")
            areas_for_improvement.append(f"Reduce {status.nutrient}")
            # Specific recommendations
            if status.nutrient == "calories":
                recommendations.append("Consider smaller portions or lower-calorie alternatives")
            elif status.nutrient == "sugar":
                recommendations.append("Limit sugary foods and beverages; choose whole fruits over processed sweets")
            elif status.nutrient == "sodium":
                recommendations.append("Reduce processed foods and added salt; use herbs and spices for flavor")
        elif status.status == "slightly_over":
            key_insights.append(f"âš ï¸� {status.nutrient.title()} slightly over target ({status.percentage_of_target:.0f}%)")
        elif status.status == "below_target":
            areas_for_improvement.append(f"Increase {status.nutrient}")
            if status.nutrient == "protein":
                recommendations.append("Add more protein sources like lean meats, fish, eggs, or legumes")
            elif status.nutrient == "fiber":
                recommendations.append("Increase fiber with whole grains, vegetables, and fruits")
   
    # Check for allergy violations
    if nutrition_summary.allergy_violations:
        key_insights.append(f"ğŸš¨ ALLERGY ALERT: {len(nutrition_summary.allergy_violations)} violation(s) detected")
        recommendations.append("URGENT: Review foods consumed for allergens")
    # Check for restriction violations
    if nutrition_summary.restriction_violations:
        key_insights.append(f"Dietary restriction violations: {len(nutrition_summary.restriction_violations)}")
   
    # Analyze medicine interactions
    if medicine_report.warnings:
        high_severity = [w for w in medicine_report.warnings if w.severity == "high"]
        if high_severity:
            key_insights.append(f"ğŸš¨ {len(high_severity)} high-priority medicine interaction(s)")
            recommendations.append("Consult healthcare provider about medicine-nutrient interactions")

        for warning in medicine_report.warnings:
            recommendations.append(f"{warning.medicine_or_supplement}: {warning.recommendation}")

    if medicine_report.nutrient_upper_limit_concerns:
        key_insights.append(f"âš ï¸� {len(medicine_report.nutrient_upper_limit_concerns)} nutrient(s) exceed safe limits")
        recommendations.append("Review supplement dosages with healthcare provider")
   

    # Condition-specific insights
    for condition in profile.conditions:
        if condition == UserCondition.PRE_DIABETIC or condition == UserCondition.DIABETIC:
            sugar_status = next((s for s in nutrition_summary.goal_statuses if s.nutrient == "sugar"), None)
            if sugar_status and sugar_status.status != "within_target":
                key_insights.append(f"Blood sugar management: Sugar intake is {sugar_status.status}")
                recommendations.append("Focus on low-glycemic foods and consistent meal timing")
        elif condition == UserCondition.HYPERTENSION:
            sodium_status = next((s for s in nutrition_summary.goal_statuses if s.nutrient == "sodium"), None)
            if sodium_status and sodium_status.status != "within_target":
                key_insights.append(f"Blood pressure management: Sodium is {sodium_status.status}")
                recommendations.append("DASH diet principles: more fruits, vegetables, and low-fat dairy")

    # Create final report
    report = RecommendationReport(
        date=nutrition_summary.date,
        user_id=profile.user_id,
        nutrition_summary=nutrition_summary,
        medicine_report=medicine_report,
        key_insights=key_insights,
        recommendations=recommendations,
        praise_points=praise_points,
        areas_for_improvement=areas_for_improvement
    )

    return json.dumps({
        "success": True,
        "report": report.model_dump()
    }, indent=2)


coordinator_tools = [analyze_nutrition_parallel,generate_final_recommendations]

coordinator_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="CoordinatorAgent",
    instruction="""You are the Nutrition Analyst Coordinator. You orchestrate multiple specialized agents:
    - ProfileAgent: Manages user profiles and memory
    - NutritionAgent: Analyzes food intake and nutrition
    - MedicineAgent: Checks medicine and supplement interactions
    Your workflow:
    1. First, work with ProfileAgent to get/create user profile and derive goals
    2. Then, coordinate PARALLEL analysis by NutritionAgent and MedicineAgent
       - Both agents analyze the same data independently
       - Nutrition focuses on food intake vs goals
       - Medicine focuses on interactions and safety
    3. Finally, merge their results into a comprehensive RecommendationReport
    Provide clear, actionable, empathetic feedback. Celebrate successes and offer
    constructive guidance for improvements. Always prioritize safety (allergies,
    high-severity interactions) in your communication.
   
    Remember: This is educational guidance, not medical advice. Encourage users
    to consult healthcare providers for medical decisions.""",
    tools=coordinator_tools,
)

print("âœ“ Coordinator Agent created for multi-agent orchestration")



DIETARY_GUIDELINES_CORPUS = {
    "pre_diabetic_tips": """
    Pre-Diabetic Diet Guidelines:
    - Focus on low glycemic index foods that don't spike blood sugar
    - Aim for 25-30g fiber per day from whole grains, vegetables, legumes
    - Limit added sugars to less than 5-10% of daily calories
    - Choose complex carbohydrates over simple sugars
    - Include lean proteins with each meal to stabilize blood sugar
    - Eat regular meals at consistent times
    - Portion control is key for weight management
    - Stay hydrated with water instead of sugary beverages
    - Include healthy fats from nuts, seeds, avocado, olive oil
    - Monitor carbohydrate intake and spread throughout the day
    """,
  
    "hypertension_diet": """
    DASH Diet for Hypertension:
    - Limit sodium to 1,500-2,300mg per day
    - Increase potassium-rich foods (bananas, sweet potatoes, spinach)
    - Focus on fruits and vegetables (4-5 servings each daily)
    - Choose low-fat or fat-free dairy products
    - Include whole grains instead of refined grains
    - Select lean meats, poultry, and fish
    - Limit red meat and processed meats
    - Reduce saturated and trans fats
    - Avoid processed and packaged foods high in sodium
    - Use herbs and spices instead of salt for flavoring
    - Limit alcohol consumption
    """,
   
    "balanced_nutrition": """
    Balanced Nutrition Principles:
    - Eat a variety of foods from all food groups
    - Fill half your plate with fruits and vegetables
    - Choose whole grains over refined grains
    - Include lean protein sources (fish, poultry, beans, nuts)
    - Limit saturated fats to less than 10% of calories
    - Keep added sugars to less than 10% of calories
    - Adequate fiber intake: 25g for women, 38g for men
    - Stay hydrated with 8-10 cups of water daily
    - Practice portion control
    - Limit processed and ultra-processed foods
    - Include healthy fats from plant sources
    """,

    "vitamin_mineral_basics": """
    Essential Vitamins and Minerals:
    - Vitamin D: Important for bone health, immune function (600-800 IU daily)
    - Vitamin B12: Critical for nerve function, red blood cells (2.4 mcg daily)
    - Iron: Essential for oxygen transport (8mg men, 18mg women daily)
    - Calcium: Bone health, muscle function (1000-1200mg daily)
    - Potassium: Blood pressure regulation, heart health (2600-3400mg daily)
    - Fiber: Digestive health, blood sugar control (25-38g daily)
   
    Upper Limits (don't exceed):
    - Vitamin D: 4000 IU daily
    - Iron: 45mg daily
    - Calcium: 2500mg daily
	
    Note: Supplements should complement, not replace, a balanced diet.
    """,

    "weight_management": """
    Healthy Weight Management:
    - Create a moderate calorie deficit (500-750 cal/day for 1-1.5 lb/week loss)
    - Don't go below 1200 calories/day for women, 1500 for men
    - Focus on nutrient-dense, low-calorie foods
    - Increase physical activity gradually
    - Eat protein with each meal to preserve muscle mass
    - Stay hydrated - sometimes thirst mimics hunger
    - Get adequate sleep (7-9 hours) - affects hunger hormones
    - Practice mindful eating - eat slowly, recognize fullness
    - Plan meals and snacks to avoid impulsive eating
    - Track food intake for awareness
    - Be patient - sustainable weight loss takes time
    """,
}


class SimpleRAG:
    """Simple retrieval system for dietary guidelines"""
    def __init__(self, corpus: Dict[str, str]):
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, str]]:
        """
        Simple keyword-based retrieval.
        In production, would use embeddings and semantic search.
        """
        query_lower = query.lower()
       
        # Score each document based on keyword matches
        scores = {}
        for doc_id, content in self.corpus.items():
            content_lower = content.lower()
            # Simple scoring: count keyword matches
            score = 0
            keywords = query_lower.split()
            for keyword in keywords:
                if len(keyword) > 3:  # Skip short words
                    score += content_lower.count(keyword)
            scores[doc_id] = score
       
        # Get top-k documents
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]

        results = []
        for doc_id, score in top_docs:
            if score > 0:  # Only return if there's some match
                results.append({
                    "doc_id": doc_id,
                    "content": self.corpus[doc_id],
                    "score": score
                })

        return results

# Initialize RAG system
rag_system = SimpleRAG(DIETARY_GUIDELINES_CORPUS)


def retrieve_dietary_guidelines(query: str, top_k: int = 2) -> str:
    """
    Tool to retrieve relevant dietary guidelines based on query.
    Uses simple keyword matching for retrieval.
    """
    results = rag_system.retrieve(query, top_k)
   
    if not results:
        return json.dumps({
            "success": True,
            "results": [],
            "message": "No relevant guidelines found for this query"
        })
   
    return json.dumps({
        "success": True,
        "results": [
            {
                "topic": r["doc_id"],
                "content": r["content"].strip(),
                "relevance_score": r["score"]
            }
            for r in results
        ],
        "count": len(results)
    }, indent=2)

# Add RAG tool to coordinator
rag_tool = retrieve_dietary_guidelines
 
# Update coordinator with RAG capability
coordinator_agent_with_rag = Agent(
    model="gemini-2.5-flash-lite",
    name="CoordinatorAgent",
    instruction="""You are the Nutrition Analyst Coordinator. You orchestrate multiple specialized agents:
    - ProfileAgent: Manages user profiles and memory
    - NutritionAgent: Analyzes food intake and nutrition
    - MedicineAgent: Checks medicine and supplement interactions
   
    Your workflow:
    1. First, work with ProfileAgent to get/create user profile and derive goals
    2. Then, coordinate PARALLEL analysis by NutritionAgent and MedicineAgent
       - Both agents analyze the same data independently
       - Nutrition focuses on food intake vs goals
       - Medicine focuses on interactions and safety
    3. Use RAG to retrieve relevant dietary guidelines based on user's conditions
    4. Finally, merge all results into a comprehensive RecommendationReport
   
    When providing recommendations, reference retrieved guidelines to support your advice.
    Provide clear, actionable, empathetic feedback. Celebrate successes and offer
    constructive guidance for improvements. Always prioritize safety (allergies,
    high-severity interactions) in your communication.
   
    Remember: This is educational guidance, not medical advice. Encourage users
    to consult healthcare providers for medical decisions.""",
    tools=coordinator_tools + [rag_tool],
)
print("âœ“ RAG system created with dietary guidelines corpus")


demo_profile_1 = UserProfile(
    user_id="user_001",
    age=45,
    sex="female",
    weight_kg=75,
    height_cm=165,
    activity_level=ActivityLevel.LIGHTLY_ACTIVE,
    conditions=[UserCondition.PRE_DIABETIC],
    dietary_restrictions=[],
    allergies=[],
    medicines=[
        Medicine(
            name="Metformin",
            dosage="500mg",
            frequency="twice daily",
            condition_treated="pre-diabetes",
            interactions=[]
        )
    ],
    supplements=[
        Supplement(
            name="Vitamin D3",
            nutrient="Vitamin D",
            dose_amount=2000,
            dose_unit="IU",
            frequency="once daily"
        )
    ]
)


demo_profile_2 = UserProfile(
    user_id="user_002",
    age=58,
    sex="male",
    weight_kg=90,
    height_cm=178,
    activity_level=ActivityLevel.MODERATELY_ACTIVE,
    conditions=[UserCondition.HYPERTENSION],
    dietary_restrictions=[],
    allergies=[],
    medicines=[
        Medicine(
            name="Lisinopril",
            dosage="10mg",
            frequency="once daily",
            condition_treated="hypertension",
            interactions=["potassium"]
        )
    ],
    supplements=[]
)


demo_profile_3 = UserProfile(
    user_id="user_003",
    age=28,
    sex="female",
    weight_kg=60,
    height_cm=168,
    activity_level=ActivityLevel.VERY_ACTIVE,
    conditions=[],
    dietary_restrictions=[DietaryRestriction.VEGETARIAN],
    allergies=["nuts", "peanuts"],
    medicines=[],
    supplements=[
        Supplement(
            name="B12",
            nutrient="Vitamin B12",
            dose_amount=500,
            dose_unit="mcg",
            frequency="once daily"
        ),
        Supplement(
            name="Iron",
            nutrient="Iron",
            dose_amount=18,
            dose_unit="mg",
            frequency="once daily"
        )
    ]
)


demo_food_log_1a = [
    FoodLogEntry(food_name="oatmeal", quantity=1, unit="cup", meal="breakfast"),
    FoodLogEntry(food_name="banana", quantity=1, unit="medium", meal="breakfast"),
    FoodLogEntry(food_name="milk_skim", quantity=1, unit="cup", meal="breakfast"),
    FoodLogEntry(food_name="chicken_breast", quantity=1, unit="3 oz", meal="lunch"),
    FoodLogEntry(food_name="brown_rice", quantity=1.5, unit="cup", meal="lunch"),
    FoodLogEntry(food_name="broccoli", quantity=1, unit="cup", meal="lunch"),
    FoodLogEntry(food_name="chocolate_bar", quantity=1, unit="bar", meal="snack"),
    FoodLogEntry(food_name="pizza_slice", quantity=2, unit="slice", meal="dinner"),
    FoodLogEntry(food_name="orange", quantity=1, unit="medium", meal="snack"),
]


demo_food_log_1b = [
    FoodLogEntry(food_name="oatmeal", quantity=0.5, unit="cup", meal="breakfast"),
    FoodLogEntry(food_name="berries_mixed", quantity=0.5, unit="cup", meal="breakfast"),
    FoodLogEntry(food_name="greek_yogurt", quantity=0.5, unit="cup", meal="breakfast"),
    FoodLogEntry(food_name="chicken_breast", quantity=1, unit="3 oz", meal="lunch"),
    FoodLogEntry(food_name="sweet_potato", quantity=0.5, unit="medium", meal="lunch"),
    FoodLogEntry(food_name="spinach", quantity=1, unit="cup", meal="lunch"),
    FoodLogEntry(food_name="almonds", quantity=0.5, unit="oz", meal="snack"),
    FoodLogEntry(food_name="salmon", quantity=1, unit="3 oz", meal="dinner"),
    FoodLogEntry(food_name="broccoli", quantity=1, unit="cup", meal="dinner"),
    FoodLogEntry(food_name="olive_oil", quantity=0.5, unit="tbsp", meal="dinner"),
]


demo_food_log_2 = [
    FoodLogEntry(food_name="eggs", quantity=2, unit="large", meal="breakfast"),
    FoodLogEntry(food_name="whole_wheat_bread", quantity=2, unit="slice", meal="breakfast"),
    FoodLogEntry(food_name="cheese_cheddar", quantity=1, unit="oz", meal="breakfast"),
    FoodLogEntry(food_name="pizza_slice", quantity=3, unit="slice", meal="lunch"),
    FoodLogEntry(food_name="chocolate_bar", quantity=1, unit="bar", meal="snack"),
    FoodLogEntry(food_name="chicken_breast", quantity=1, unit="3 oz", meal="dinner"),
    FoodLogEntry(food_name="rice", quantity=1, unit="cup", meal="dinner"),
]


demo_food_log_3 = [
    FoodLogEntry(food_name="oatmeal", quantity=1, unit="cup", meal="breakfast"),
    FoodLogEntry(food_name="almond_milk", quantity=1, unit="cup", meal="breakfast"),
    FoodLogEntry(food_name="banana", quantity=1, unit="medium", meal="breakfast"),
    FoodLogEntry(food_name="tofu_firm", quantity=1, unit="3 oz", meal="lunch"),
    FoodLogEntry(food_name="brown_rice", quantity=1, unit="cup", meal="lunch"),
    FoodLogEntry(food_name="spinach", quantity=1, unit="cup", meal="lunch"),
    FoodLogEntry(food_name="peanut_butter", quantity=1, unit="2 tbsp", meal="snack"),  # ALLERGY!
    FoodLogEntry(food_name="whole_wheat_bread", quantity=1, unit="slice", meal="snack"),
    FoodLogEntry(food_name="eggs", quantity=2, unit="large", meal="dinner"),
    FoodLogEntry(food_name="avocado", quantity=0.5, unit="medium", meal="dinner"),
]


import nest_asyncio
nest_asyncio.apply()


async def _run_coordinator_with_rag_async(
    profile: UserProfile,
    food_log: List[FoodLogEntry],
    custom_calorie_target: Optional[float] = None,
    date: Optional[str] = None,
) -> str:
    """Run the coordinator_agent_with_rag for a single day.
    Flow:
      1) Call analyze_nutrition_parallel() directly in Python
         (this uses ParallelAgent -> NutritionAgent + MedicineAgent).
      2) Call generate_final_recommendations() directly in Python.
      3) Call retrieve_dietary_guidelines() directly in Python.
      4) Give ALL those results to CoordinatorAgentWithRag to summarize.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # 1) Precompute structured inputs that the tools expect.
    goals = GoalDerivationEngine.derive_goals(profile, custom_calorie_target)
    profile_json = profile.model_dump_json()
    goals_json = goals.model_dump_json()
    food_entries_json = json.dumps([e.model_dump() for e in food_log])

    # 2) Run core tools directly (no LLM tool-calling here).
    analysis_raw = analyze_nutrition_parallel(
        profile_json=profile_json,
        goals_json=goals_json,
        food_entries_json=food_entries_json,
        date=date,
    )

    try:
        analysis = json.loads(analysis_raw)
    except Exception as e:
        return f"Error while analyzing nutrition: could not parse analysis JSON ({e})."

    if not analysis.get("success", False):
        return (
            "Error while analyzing nutrition: "
            f"{analysis.get('error', 'Unknown error from analyze_nutrition_parallel')}."
        )
    nutrition_summary = analysis["nutrition_summary"]
    medicine_report = analysis["medicine_report"]
    parallel_commentary = analysis.get("parallel_commentary", {})
    recs_raw = generate_final_recommendations(
        nutrition_summary_json=json.dumps(nutrition_summary),
        medicine_report_json=json.dumps(medicine_report),
        profile_json=profile_json,
    )
    
    try:
        recs = json.loads(recs_raw)
    except Exception as e:
        return f"Error while generating recommendations: could not parse JSON ({e})."

    if not recs.get("success", False):
        return (
            "Error while generating recommendations: "
            f"{recs.get('error', 'Unknown error from generate_final_recommendations')}."
        )
    final_report = recs["report"]

    # 3) Retrieve dietary guidelines (RAG) for the user's conditions.
    condition_query = " ".join([c.value for c in profile.conditions])
    guidelines_result = {}
    if condition_query:
        try:
            guidelines_raw = retrieve_dietary_guidelines(condition_query, top_k=2)
            guidelines_result = json.loads(guidelines_raw)
        except Exception:
            guidelines_result = {"success": False, "results": []}

    # 4) Now hand everything to CoordinatorAgentWithRag for final narration.
    summarization_prompt = (
        "You are given structured results from a multi-agent nutrition system.\n\n"
        "1) DailyNutritionSummary (JSON):\n"
        f"{json.dumps(nutrition_summary, indent=2)}\n\n"
        "2) MedicineInteractionReport (JSON):\n"
        f"{json.dumps(medicine_report, indent=2)}\n\n"
        "3) Parallel agent commentary (per sub-agent):\n"
        f"{json.dumps(parallel_commentary, indent=2)}\n\n"
        "4) Final RecommendationReport object (JSON):\n"
        f"{json.dumps(final_report, indent=2)}\n\n"
        "5) Retrieved dietary guidelines (if any):\n"
        f"{json.dumps(guidelines_result, indent=2)}\n\n"
        "Using ONLY this information, speak directly to the user.\n"
        "- Start with a one-sentence overall summary of how their day went.\n"
        "- Then give 3â€“5 concise bullet points with the most important insights "
        "(calories, sugar, sodium, medicine safety, etc.).\n"
        "- End with 2â€“3 practical tips for tomorrow.\n"
        "- Finally mention their total counts and the percentage they exceeded"
        "Do not repeat raw JSON; transform it into user-friendly language."
    )
    runner = runners.InMemoryRunner(
        agent=coordinator_agent_with_rag,
        app_name="nutrition_app_rag",
    )
    session = await runner.session_service.create_session(
        app_name="nutrition_app_rag", user_id=profile.user_id
    )
    user_message = types.Content(
        role="user",
        parts=[types.Part(text=summarization_prompt)],
    )
    final_text = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    text = part.text.strip()
                    if text:
                        print(text)
                        print()  # spacing between chunks
                        final_text = text
    return final_text

def run_nutrition_analysis_with_agents(
    profile: UserProfile,
    food_log: List[FoodLogEntry],
    custom_calorie_target: Optional[float] = None,
    date: Optional[str] = None,
) -> str:
    """
    Synchronous wrapper around the coordinator_agent_with_rag demo.
    Safe to call from a normal Kaggle notebook cell.
    """
    """
    Synchronous wrapper around the coordinator_agent_with_rag demo.
    Safe to call from a normal Kaggle notebook cell.
    """
    coro = _run_coordinator_with_rag_async(
        profile=profile,
        food_log=food_log,
        custom_calorie_target=custom_calorie_target,
        date=date,
    )
    
    try:
        # 1. Try to get the running loop
        loop = asyncio.get_running_loop()
        
        # 2. If a loop is already running, run the coroutine in the existing event loop
        #    Note: This is often preferred in notebook environments.
        if loop.is_running():
            # Use run_until_complete on the existing loop (requires creating a Future/Task)
            # A more common approach in notebooks is to use a high-level wrapper like
            # nest_asyncio or just scheduling the task, but for simple sync wrapper:
            
            # Since we are trying to run a coroutine synchronously in an async environment,
            # we use the existing loop and wait for the result.
            return loop.run_until_complete(coro)
            
    except RuntimeError:
        # 3. If no loop is running (i.e., new thread or standard script), create and run a new one
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            # Fallback for environments where get_event_loop() also fails
            # This handles Python 3.7+ environments where asyncio.run() is available
            try:
                return asyncio.run(coro)
            except RuntimeError as e:
                return f"Error running async function: {e}"


print("\n" + "ğŸŒŸ"*40)
print("DEMO SCENARIO 1A: Pre-Diabetic User - EXCEEDING Limits")
print("ğŸŒŸ"*40)
report_1a = run_nutrition_analysis_with_agents(
    profile=demo_profile_1,
    food_log=demo_food_log_1a,
    custom_calorie_target=1200,
    date="2025-11-27"
)


print("\n\n" + "ğŸŒŸ"*40)
print("DEMO SCENARIO 1B: Pre-Diabetic User - WITHIN Target")
print("ğŸŒŸ"*40)
report_1b = run_nutrition_analysis_with_agents(
    profile=demo_profile_1,
    food_log=demo_food_log_1b,
    custom_calorie_target=1200,
    date="2025-11-27"
)


print("\n\n" + "ğŸŒŸ"*40)
print("DEMO SCENARIO 2: Hypertension User - HIGH Sodium")
print("ğŸŒŸ"*40)
report_2 = run_nutrition_analysis_with_agents(
    profile=demo_profile_2,
    food_log=demo_food_log_2,
    date="2025-11-27"
)


print("\n\n" + "ğŸŒŸ"*40)
print("DEMO SCENARIO 3: Vegetarian with ALLERGY Violation")
print("ğŸŒŸ"*40)
report_3 = run_nutrition_analysis_with_agents(
    profile=demo_profile_3,
    food_log=demo_food_log_3,
    date="2025-11-27"
)


print("\n\nâœ… All demo scenarios completed!")

