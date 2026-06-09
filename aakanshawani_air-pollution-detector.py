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


import requests
from typing import Dict, List, Optional

# --- 1. Knowledge Base for Plant Suggestions ---
# Note: In a real-world application, this data would be sourced from a robust database
# and tailored for *native* and local conditions (e.g., city, climate, soil type).
# The plants listed here are generally known for air purification (indoor/outdoor).

PLANT_KNOWLEDGE_BASE: Dict[str, List[str]] = {
    "GOOD": [
        "Ornamental Tree Species (e.g., flowering cherries, native maples)",
        "Drought-Tolerant Groundcovers (to stabilize soil)",
        "Any native flowering shrubs or bushes",
    ],
    "MODERATE": [
        "Snake Plant (Sansevieria trifasciata) - Excellent for formaldehyde/benzene",
        "Spider Plant (Chlorophytum comosum) - Removes xylene/formaldehyde",
        "Money Plant (Epipremnum aureum) - Reduces VOCs like formaldehyde and xylene",
    ],
    "POOR": [
        "Areca Palm (Chrysalidocarpus lutescens) - Effective against xylene, toluene, formaldehyde",
        "Peace Lily (Spathiphyllum) - Absorbs benzene, formaldehyde, and trichloroethylene",
        "Rubber Plant (Ficus elastica) - Removes formaldehyde and increases oxygen",
    ],
    "UNHEALTHY": [
        "Dense Foliage Trees (e.g., White Poplar, American Elm) for roadside particulate matter (PM) reduction",
        "Bamboo Palm (Chamaedorea seifrizii) - Good filter for benzene and trichloroethylene",
        "English Ivy (Hedera helix) - Great for reducing airborne mold and benzene",
    ],
    "SEVERE": [
        "Focus on high-biomass, pollution-tolerant native evergreen trees and shrubs for maximum PM absorption.",
        "Priority on creating green barriers near roads (e.g., hedges, vertical gardens).",
    ],
}


class AirPollutionAgent:
    """
    An AI Agent to detect city road air pollution and suggest suitable plants.
    """
    def __init__(self, api_key: str):
        # In a real app, this would be your API key for a service like Google Air Quality API
        self.api_key = api_key
        self.aqi_endpoint = "https://api.airqualityservice.com/currentConditions" # Placeholder URL

    def _get_air_quality_data(self, lat: float, lng: float) -> Optional[Dict]:
        """
        Simulates fetching real-time air quality data for a given location.
        A real implementation would use the requests library to query a paid/free API.
        """
        print(f"-> Querying Air Quality API for Lat: {lat}, Lng: {lng}...")
        
        # --- Placeholder Logic for Demonstration ---
        try:
            # Simulate a successful API response
            simulated_aqi = 155 # Example: Unhealthy level
            
            # Simulate real pollutant data (PM2.5 is often a major road pollutant)
            simulated_data = {
                "aqi_us": simulated_aqi,
                "category": self._categorize_aqi(simulated_aqi),
                "dominant_pollutant": "PM2.5",
                "pollutant_concentration": 45.0, # in µg/m³
            }
            return simulated_data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        # --- End Placeholder Logic ---

    def _categorize_aqi(self, aqi: int) -> str:
        """Categorizes the AQI into standard health categories."""
        if aqi <= 50:
            return "GOOD"
        elif aqi <= 100:
            return "MODERATE"
        elif aqi <= 150:
            return "POOR"
        elif aqi <= 200:
            return "UNHEALTHY"
        elif aqi <= 300:
            return "SEVERE"
        else:
            return "HAZARDOUS"

    def _suggest_plants(self, pollution_category: str) -> List[str]:
        """Suggests plants based on the pollution category."""
        return PLANT_KNOWLEDGE_BASE.get(pollution_category, ["No specific recommendations found for this category."])

    def run_detection_and_suggest(self, city_road_lat: float, city_road_lng: float):
        """
        The main function to run the agent's full cycle.
        1. Get Air Quality (Perception)
        2. Determine Pollution Level (Analysis)
        3. Suggest Plants (Action/Recommendation)
        """
        print(f"*** Air Pollution Agent Initialized ***")
        
        # 1. Get Air Quality Data
        aq_data = self._get_air_quality_data(city_road_lat, city_road_lng)

        if not aq_data:
            print("Failed to retrieve air quality data. Agent terminating.")
            return

        # 2. Determine Pollution Level
        aqi = aq_data["aqi_us"]
        category = aq_data["category"]
        pollutant = aq_data["dominant_pollutant"]

        print("\n*** Air Quality Analysis ***")
        print(f"Location AQI (US EPA Standard): **{aqi}**")
        print(f"Pollution Level: **{category}**")
        print(f"Dominant Pollutant: **{pollutant}**")
        
        



        # 3. Suggest Plants
        suggested_plants = self._suggest_plants(category)

        print("\n*** Plant Suggestion for Surrounding Area ***")
        print(f"Based on a **{category}** air quality level, here are suggested plants for remediation and beautification:")
        for plant in suggested_plants:
            print(f"* {plant}")
        
        print("\n*Recommendation Note:* Always choose native or well-adapted species and consult local forestry/horticultural guides for best results.")


if __name__ == "__main__":
    # Define a location (e.g., a busy road intersection)
    target_lat = 40.7128  # Example: New York City
    target_lng = -74.0060

    # Initialize and Run the Agent
    agent = AirPollutionAgent(api_key="YOUR_API_KEY_HERE")
    agent.run_detection_and_suggest(target_lat, target_lng)

