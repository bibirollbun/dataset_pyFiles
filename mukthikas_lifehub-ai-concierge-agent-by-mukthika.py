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


import datetime

# ---------------------------
#    Helper: Meal Engine
# ---------------------------
def generate_meals(preferences):
    meals = {
        "breakfast": f"Healthy breakfast based on: {preferences}",
        "lunch": f"Nutrient-balanced lunch following: {preferences}",
        "snacks": f"Light snacks respecting: {preferences}",
        "dinner": f"High-protein dinner following: {preferences}"
    }
    return meals


# ---------------------------
#   Helper: Date Detection
# ---------------------------
def detect_date(text_input):
    today = datetime.date.today()
    text = text_input.lower()

    if "today" in text:
        return today.isoformat()

    if "tomorrow" in text:
        return (today + datetime.timedelta(days=1)).isoformat()

    # Default fallback
    return today.isoformat()


# ---------------------------
#      Main Agent Logic
# ---------------------------
def generate_schedule(text_input, preferences="", attachments=None):
    if attachments is None:
        attachments = []

    date = detect_date(text_input)
    meals = generate_meals(preferences)

    schedule = f"ğŸ“… LifeHub Daily Schedule ({date})\n"
    schedule += f"User Input: {text_input}\n\n"

    schedule += "ğŸ�½ Meals:\n"
    schedule += f"- Breakfast: {meals['breakfast']}\n"
    schedule += f"- Lunch: {meals['lunch']}\n"
    schedule += f"- Snacks: {meals['snacks']}\n"
    schedule += f"- Dinner: {meals['dinner']}\n\n"

    schedule += "ğŸ“� Attachments:\n"
    if attachments:
        for a in attachments:
            schedule += f"- {a}\n"
    else:
        schedule += "- None\n"

    schedule += "\nğŸ§  Summary:\nThis schedule integrates goals, preferences, and uploaded data."

    return schedule


print("LifeHub components loaded successfully.")


example_text = "Plan my day today. I need high-protein meals and I have 2 meetings."
example_prefs = "high protein"
example_attachments = ["meeting_notes.pdf"]

schedule = generate_schedule(example_text, example_prefs, example_attachments)
print(schedule)


text = input("Describe your day: ")
prefs = input("List any preferences (meals, health, schedule): ")

schedule = generate_schedule(text, prefs)
print("\n" + schedule)


