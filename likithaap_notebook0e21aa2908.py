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


# ğŸŒ� Intermediate Travel AI Agent
print("Welcome to Travel AI Agent! Type 'exit' to quit.")

# Travel data
destinations = {
    "paris": {
        "places": ["Eiffel Tower", "Louvre Museum", "Seine River Cruise"],
        "food": ["Croissant", "Baguette", "Escargot"],
        "budget_per_day": 100
    },
    "tokyo": {
        "places": ["Shibuya Crossing", "Tokyo Tower", "Meiji Shrine"],
        "food": ["Sushi", "Ramen", "Tempura"],
        "budget_per_day": 120
    },
    "rome": {
        "places": ["Colosseum", "Vatican City", "Trevi Fountain"],
        "food": ["Pasta", "Pizza", "Gelato"],
        "budget_per_day": 90
    },
    "london": {
        "places": ["Big Ben", "London Eye", "Tower Bridge"],
        "food": ["Fish & Chips", "Pie", "Afternoon Tea"],
        "budget_per_day": 110
    }
}

# Packing suggestions
packing_items = {
    "cold": ["Jacket", "Gloves", "Warm socks"],
    "hot": ["T-shirt", "Sunglasses", "Hat", "Sunscreen"],
    "rainy": ["Umbrella", "Raincoat", "Waterproof shoes"]
}

# Functions
def show_places_food(city):
    city = city.lower()
    if city in destinations:
        print("Famous places to visit:", ", ".join(destinations[city]["places"]))
        print("Popular food:", ", ".join(destinations[city]["food"]))
    else:
        print("Sorry, I don't have data for that city. Explore local attractions!")

def estimate_budget(city, days):
    city = city.lower()
    if city in destinations:
        return destinations[city]["budget_per_day"] * days
    else:
        return 80 * days  # default

def suggest_packing(weather):
    return packing_items.get(weather.lower(), ["Clothes and essentials"])

# Variables to remember user choices
current_city = None
trip_days = None

# Main chat loop
while True:
    user_input = input("\nYou: ").lower()
    
    if user_input == "exit":
        print("Travel AI Agent: Goodbye! Have a safe trip! âœˆï¸�")
        break
    
    elif "destination" in user_input or "city" in user_input or "going" in user_input:
        current_city = input("Which city are you visiting? ")
        show_places_food(current_city)
    
    elif "days" in user_input or "duration" in user_input:
        if current_city:
            trip_days = int(input(f"How many days will you spend in {current_city}? "))
            print(f"Travel AI Agent: Got it! {trip_days} days in {current_city}.")
        else:
            print("Please tell me your destination first.")
    
    elif "budget" in user_input or "cost" in user_input or "money" in user_input:
        if current_city and trip_days:
            budget = estimate_budget(current_city, trip_days)
            print(f"Travel AI Agent: Estimated budget for {trip_days} days in {current_city} is ${budget}.")
        else:
            print("Please provide your city and trip duration first.")
    
    elif "packing" in user_input or "pack" in user_input or "weather" in user_input:
        weather = input("What is the weather like (hot/cold/rainy)? ")
        items = suggest_packing(weather)
        print("Travel AI Agent: You should pack:", ", ".join(items))
    
    elif "places" in user_input or "food" in user_input or "sightseeing" in user_input:
        if current_city:
            show_places_food(current_city)
        else:
            print("Please tell me your destination first.")
    
    else:
        print("Travel AI Agent: I can help with destination, budget, trip days, packing, or famous places & food!")


