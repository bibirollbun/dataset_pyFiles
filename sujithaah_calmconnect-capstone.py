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

# --- Agent 1: Emotion Detector (Friendly Listener) ---

def analyze_emotion(user_input: str) -> dict:
    """
    Analyzes user text to extract emotion, intensity, and a summary/trigger.
    In a real project, this uses an LLM call with a structured output schema.
    """
    
    # 1. Define the Structured Output Schema (Pydantic or JSON schema)
    # The LLM is prompted to return exactly this structure.
    schema = {
        "emotion": "str (e.g., 'sad', 'stressed', 'overwhelmed')",
        "intensity": "str (low, medium, high)",
        "trigger_summary": "str (brief description of potential cause)"
    }

    # 2. Simulate LLM Call/Parsing (You will replace this with actual LLM API call)
    if "drained" in user_input.lower() or "anxious" in user_input.lower():
        output_data = {
            "emotion": "anxious",
            "intensity": "high",
            "trigger_summary": "General feeling of anxiety and fatigue before the week begins."
        }
        
    elif "tired" in user_input.lower() and "deadline" in user_input.lower():
        output_data = {
            "emotion": "overwhelmed",
            "intensity": "high",
            "trigger_summary": "High stress related to external workload/deadline."
        }
    elif "low" in user_input.lower() or "sad" in user_input.lower():
        output_data = {
            "emotion": "sad",
            "intensity": "medium",
            "trigger_summary": "General feeling of low mood and fatigue."
        }
    else:
        output_data = {
            "emotion": "neutral",
            "intensity": "low",
            "trigger_summary": "No immediate distress detected."
        }

    return output_data


import pandas as pd
from datetime import datetime
from collections import Counter

# --- Agent 3: Mood Memory (Pattern Analyzer) ---

# Global data structure simulating a simple database/memory
MOOD_HISTORY = []

def store_mood(emotion_data: dict):
    """Stores the analyzed emotion data with a timestamp."""
    timestamp = datetime.now()
    record = {
        "timestamp": timestamp,
        "day_of_week": timestamp.strftime('%A'),
        "hour": timestamp.hour,
        "emotion": emotion_data['emotion'],
        "intensity": emotion_data['intensity']
    }
    MOOD_HISTORY.append(record)
    print(f"-> Agent 3: Stored mood for {record['day_of_week']} at {record['hour']}h.")

def detect_recurring_pattern() -> str:
    """
    Analyzes MOOD_HISTORY to find recurring high-intensity negative patterns.
    (In a full build, this uses K-Means clustering on time features.)
    """
    
    if len(MOOD_HISTORY) < 5:
        return "No sufficient history yet for pattern analysis."

    # Filter for high intensity negative moods (simulation)
    negative_moods = [
        r for r in MOOD_HISTORY 
        if r['intensity'] == 'high' and r['emotion'] in ['overwhelmed', 'stressed', 'angry']
    ]

    if not negative_moods:
        return "No recurring high-intensity patterns detected."

    # Simple simulation of pattern detection: find the most common day/hour combo
    day_hour_combos = [(r['day_of_week'], r['hour']) for r in negative_moods]
    pattern_count = Counter(day_hour_combos)
    
    most_common_pattern = pattern_count.most_common(1)
    
    if most_common_pattern and most_common_pattern[0][1] >= 2:
        day, hour = most_common_pattern[0][0]
        return f"Recurring Pattern Detected: High stress peaks frequently around {day} evenings ({hour}h). Trigger likely related to weekly cycle."
    
    return "No significant recurring patterns detected."


# --- Agent 2: Advice Generator (Comforting + Practical) ---

def generate_advice(emotion_data: dict, pattern_insight: str) -> str:
    """
    Generates a balanced, two-part response based on real-time emotion and history.
    This relies heavily on precise LLM prompting (CBT/ACT framework).
    """

    emotion = emotion_data['emotion']
    summary = emotion_data['trigger_summary']

    # 1. COMFORTING TONE (LLM Prompting)
    comfort_lines = f"I hear you. It sounds like you are feeling {emotion} right now, likely due to {summary}. Please know that these feelings are valid, and you are not alone."

    # 2. PATTERN REFERENCE
    pattern_line = ""
    if "Recurring Pattern Detected" in pattern_insight:
        pattern_line = f"I also noticed something in your long-term history: {pattern_insight}. This is a cycle we can work on interrupting!"

    # 3. ACTIONABLE STEPS (LLM Prompting - CBT/ACT)
    # The LLM is prompted to return 2 simple, actionable steps based on the pattern/summary.
    practical_steps = ""
    
    # NEW LOGIC: PRIORITIZE PATTERN INTERRUPTANCE
    if "Recurring Pattern Detected" in pattern_insight:
        # If a pattern is found, the steps are proactive and scheduled
        steps = (
            "1. **Proactive Schedule:** Set a calendar reminder NEXT Saturday (13h) for a 30-minute 'Mental Boundary' activity.",
            "2. **Identify Micro-Trigger:** What small task (e.g., checking email) are you doing right now that initiates the anxiety?"
        )
        practical_steps = "\n\nTry these 2 **Proactive Steps** to interrupt this recurring cycle:\n" + "\n".join(steps)
    
    elif "deadline" in summary:
        # Original logic for immediate crisis
        steps = (
# ... (original steps for 'deadline' crisis)
        )
        practical_steps = "\n\nTry these 2 immediate steps:\n" + "\n".join(steps)
        
    else:
        # Original generic steps for low intensity
        steps = (
# ... (original generic steps)
        )
        practical_steps = "\n\nTry these 2 simple steps:\n" + "\n".join(steps)


    return f"{comfort_lines}\n{pattern_line}{practical_steps}"




def calmconnect_session(user_input: str) -> str:
    """
    The main workflow function coordinating the three agents.
    """
    print("\n--- CalmConnect Session Initiated ---")
    
    # 1. AGENT 1: EMOTION DETECTION
    emotion_output = analyze_emotion(user_input)
    print(f"-> Agent 1 Output: {emotion_output['emotion']} ({emotion_output['intensity']})")

    # 2. AGENT 3: DATA STORAGE
    store_mood(emotion_output)

    # 3. AGENT 3: PATTERN ANALYSIS (Retrieve Long-Term Context)
    long_term_insight = detect_recurring_pattern()
    print(f"-> Agent 3 Insight: {long_term_insight}")

    # 4. AGENT 2: ADVICE GENERATION
    final_advice = generate_advice(emotion_output, long_term_insight)
    
    return final_advice

# --- DEMO SCENARIOS ---

# SCENARIO 1: First Interaction (No Pattern Yet)
# Note: You need to run several sessions for the pattern to be 'detected'
first_input = "I just feel low today. I'm struggling with everything."
response_1 = calmconnect_session(first_input)
print("\n[USER RESPONSE 1]")
print(response_1)
print("---------------------------------------")

# SCENARIO 2: Repeated Pattern (Simulate multiple 'high stress' inputs on the same day)
# For the pattern to work in the simulation, run these inputs sequentially:
print("Simulating 5 previous high-stress inputs on the same day...")
for _ in range(5):
    store_mood(analyze_emotion("I hate this deadline, I'm so stressed and tired."))
print("---------------------------------------")

# SCENARIO 3: Pattern Detected Interaction
pattern_input = "Here we go again, just feeling drained and anxious about the next five days."
response_2 = calmconnect_session(pattern_input)
print("\n[USER RESPONSE 2 - Pattern Detected]")
print(response_2)

