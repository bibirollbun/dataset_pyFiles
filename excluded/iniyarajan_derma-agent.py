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


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")



# -------------------------------------------
# MyDerma Interactive Skincare Agent - Kaggle
# -------------------------------------------

# Step 1: Install ipywidgets for interactivity
!pip install ipywidgets --quiet

# Step 2: Import libraries
import os
import json
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output

# Step 3: Google API key (if needed for other features)
from kaggle_secrets import UserSecretsClient
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Google API key setup complete.")
except Exception as e:
    print("ðŸ”‘ Google API key not found, some features may not work.")

# Helper function to pretty print JSON
def pretty_print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))

# -------------------------------------------
# Tools / Functions
# -------------------------------------------

def save_profile(profile):
    print(f"\nâœ… Profile saved for {profile['name']}")
    return {"status": "success", "profile": profile}

def get_skincare_recommendation(profile):
    skin_type = profile.get("skin_type", "normal").lower()
    approach = profile.get("approach", "ayurvedic").lower()
    
    remedies = {
        "ayurvedic": {
            "oily": ["Aloe vera gel", "Neem paste mask", "Multani mitti mask"],
            "dry": ["Honey + milk mask", "Coconut oil massage", "Rose water toner"],
            "normal": ["Cucumber mask", "Aloe + honey mask", "Green tea rinse"]
        },
        "medicine": {
            "oily": ["Salicylic acid cleanser", "Oil-free moisturizer"],
            "dry": ["Hydrating cream", "Gentle cleanser"],
            "normal": ["Mild exfoliant", "Balanced moisturizer"]
        }
    }

    skincare_plan = {
        "oily": [
            "Morning: Cleanser + Toner + Oil-free moisturizer",
            "Night: Gentle cleanser + Serum + Light moisturizer"
        ],
        "dry": [
            "Morning: Hydrating cleanser + Rich moisturizer",
            "Night: Creamy cleanser + Night cream + Serum"
        ],
        "normal": [
            "Morning: Gentle cleanser + Moisturizer + Sunscreen",
            "Night: Cleanser + Light moisturizer"
        ]
    }

    diet_plan = {
        "oily": ["Reduce oily food", "Eat fruits & vegetables", "Drink water"],
        "dry": ["Drink milk, coconut water", "Include nuts & seeds", "Hydrating fruits"],
        "normal": ["Balanced diet", "Green vegetables", "Hydrate well"]
    }

    return {
        "recommended_remedies": remedies.get(approach, remedies["ayurvedic"]).get(skin_type),
        "skincare_plan": skincare_plan.get(skin_type),
        "diet_plan": diet_plan.get(skin_type)
    }

def get_dermatologist_referral(profile):
    user_location = profile.get("location", "your city")
    nearby_derma = [
        {"name": "Dr. Skin Care", "clinic": f"{user_location} Dermatology Clinic", "contact": "123-456-7890"},
        {"name": "Dr. Glow", "clinic": f"{user_location} Skin Center", "contact": "987-654-3210"}
    ]
    return {"dermatologists_nearby": nearby_derma}

def book_appointment(derma, date, time):
    return {
        "status": "booked",
        "doctor": derma["name"],
        "clinic": derma["clinic"],
        "date": date,
        "time": time
    }

# -------------------------------------------
# Widgets for Interactive Input
# -------------------------------------------

# User input widgets
name_widget = widgets.Text(value='Iniya', description='Name:')
skin_widget = widgets.Dropdown(options=['oily','dry','normal'], description='Skin Type:')
location_widget = widgets.Text(value='Bengaluru', description='City:')
approach_widget = widgets.Dropdown(options=['ayurvedic','medicine'], description='Approach:')

# Dermatologist appointment widgets
derma_dropdown = widgets.Dropdown(description="Select Doctor:")
date_widget = widgets.Text(value='2025-11-25', description='Date (YYYY-MM-DD)')
time_widget = widgets.Text(value='10:00', description='Time (HH:MM)')
book_button = widgets.Button(description="Book Appointment", button_style='success')

# Run MyDerma button
run_button = widgets.Button(description="Run MyDerma", button_style='info')

# Output display
output = widgets.Output()
display(name_widget, skin_widget, location_widget, approach_widget, run_button, output)

# -------------------------------------------
# Callback Functions
# -------------------------------------------

def run_myderma(btn):
    with output:
        clear_output()
        profile = {
            "name": name_widget.value,
            "skin_type": skin_widget.value,
            "location": location_widget.value,
            "approach": approach_widget.value
        }
        save_profile(profile)
        
        # Skincare recommendation
        print("\n=== Skincare Recommendation ===")
        recommendation = get_skincare_recommendation(profile)
        pretty_print_json(recommendation)
        
        # Dermatologists
        print("\n=== Nearby Dermatologists ===")
        derma_list = get_dermatologist_referral(profile)["dermatologists_nearby"]
        for idx, d in enumerate(derma_list, 1):
            print(f"{idx}. {d['name']} - {d['clinic']} (Contact: {d['contact']})")
        
        # Update derma dropdown options for booking
        derma_dropdown.options = [f"{d['name']} ({d['clinic']})" for d in derma_list]
        display(derma_dropdown, date_widget, time_widget, book_button)

def book_derma_appointment(btn):
    with output:
        clear_output(wait=True)
        derma_index = derma_dropdown.index
        profile = {
            "name": name_widget.value,
            "skin_type": skin_widget.value,
            "location": location_widget.value,
            "approach": approach_widget.value
        }
        derma_list = get_dermatologist_referral(profile)["dermatologists_nearby"]
        derma = derma_list[derma_index]
        appointment = book_appointment(derma, date_widget.value, time_widget.value)
        print("\nâœ… Appointment Booked Successfully!")
        pretty_print_json(appointment)

# -------------------------------------------
# Connect buttons to callbacks
# -------------------------------------------
run_button.on_click(run_myderma)
book_button.on_click(book_derma_appointment)


