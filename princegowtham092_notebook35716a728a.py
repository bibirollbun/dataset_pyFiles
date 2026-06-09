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


!pip install --upgrade protobuf --quiet
!pip install transformers==4.31.0 --quiet



#SECTION 4 — SETUP CODE
#Install transformers
!pip install transformers --quiet

import json, os, datetime
from collections import Counter
from transformers import pipeline
import random



#SECTION 5 — CORE AGENT CODE
# =========== EMOBUDDY CORE CODE ===========

DATA_FILE = "emobuddy_data.json"

# Load / Save functions
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"entries": []}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Mood detection model
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

# Basic empathetic response generator (no API needed)
MICRO_ACTIONS = [
    "Take 3 slow breaths with your shoulders relaxed.",
    "Drink a glass of water calmly.",
    "Stretch your neck and arms for 20 seconds.",
    "Step away from your screen for 1 minute.",
    "Close your eyes and inhale slowly for 5 seconds."
]

def generate_response(user_text, mood):
    response = ""

    response += f"I’m sorry you’re feeling {mood}. Thanks for sharing that.\n\n"
    response += f"Here's something gentle you can try: **{random.choice(MICRO_ACTIONS)}**\n\n"
    response += "If you want, I can also give you a journaling question."

    return response

# Pattern detection
def detect_patterns(entries):
    if not entries:
        return "No patterns yet."

    moods = [e["mood"] for e in entries]
    times = [datetime.datetime.fromisoformat(e["timestamp"]).hour for e in entries]

    mood_count = Counter(moods).most_common()
    time_count = Counter(times).most_common()

    return {
        "common_moods": mood_count[:3],
        "common_times": time_count[:3]
    }

# ---- MAIN FUNCTION ----
def emobuddy_chat(user_text):
    data = load_data()

    mood = emotion_model(user_text)[0]["label"]

    entry = {
        "text": user_text,
        "mood": mood,
        "timestamp": datetime.datetime.now().isoformat()
    }
    data["entries"].append(entry)
    save_data(data)

    reply = generate_response(user_text, mood)
    return mood, reply



#SECTION 6 — DEMO OUTPUT CELL
# DEMO RUN

sample_input = "I feel stressed these days and nothing is going right."
mood, reply = emobuddy_chat(sample_input)

print("User:", sample_input)
print("Detected Mood:", mood)
print("\nEmoBuddy Response:\n", reply)


