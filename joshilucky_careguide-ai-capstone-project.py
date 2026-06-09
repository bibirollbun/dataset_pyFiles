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


# ----------------------------
# CareGuide AI - Initialization
# ----------------------------

# Data stores
reminders = []         # List of user reminders
habit_logs = []        # Log for completed habits
wellness_data = {}     # Placeholder for knowledge base (will add soon)

# High-risk keywords list for safety filter
risky_keywords = [
    "dose", "mg", "tablet", "capsule", "prescription", "medicine name",
    "severe", "bleeding", "unconscious", "chest pain", "heart pain",
    "emergency", "suicidal", "self-harm"
]

def is_high_risk(text):
    """Check if message contains high-risk keywords."""
    text = text.lower()
    return any(word in text for word in risky_keywords)



def detect_intent(message):
    """Basic rule-based intent detection for classifying user requests."""
    msg = message.lower()

    # Check for reminder setup
    if "remind" in msg or "reminder" in msg or "set" in msg:
        return "set_reminder"
    
    # Check for habit completion updates
    if "done" in msg or "completed" in msg or "finished" in msg or "i did" in msg:
        return "mark_done"
    
    # Check for progress queries
    if "progress" in msg or "summary" in msg or "report" in msg:
        return "check_progress"
    
    # Otherwise assume wellness information query
    return "wellness_info"



# --------------------------------------------
# Basic Wellness Knowledge Base
# --------------------------------------------

wellness_data = {
    "water": "Drinking enough water helps maintain hydration, improve digestion, and support energy levels.",
    "sleep": "Aim for 7-9 hours of sleep daily. Maintain a routine, avoid screens before bedtime, and reduce caffeine.",
    "walking": "Walking improves cardiovascular health, helps maintain weight, and reduces stress.",
    "yoga": "Yoga improves flexibility, breathing, posture, and supports mental relaxation.",
    "balanced diet": "A balanced diet includes vegetables, fruits, proteins, whole grains, and limited sugary or processed foods.",
    "mental health": "Taking breaks, journaling, speaking to trusted people, and practicing mindfulness supports mental well-being."
}



# --------------------------------------------
# Wellness Response
# --------------------------------------------

def wellness_response(message):
    """Return general wellness info based on keyword matching."""
    msg = message.lower()
    
    for key in wellness_data:
        if key in msg:
            return wellness_data[key]
    
    return "Try asking about: water, sleep, yoga, walking, diet, or mental health."



# --------------------------------------------
# Reminder Module
# --------------------------------------------

def set_reminder(message):
    """Extract reminder text and store it."""
    reminder_text = message.replace("remind me to", "").replace("set", "").strip()
    
    if reminder_text:
        reminders.append(reminder_text)
        return f"Reminder added: '{reminder_text}'"
    else:
        return "Please specify the reminder. Example: 'Remind me to drink water at 6 PM'"



def mark_habit_completion(message):
    category = categorize_habit(message)
    habit_logs.append(category)
    return f"Great job! Habit recorded under: {category} 💪"



# --------------------------------------------
# Progress Report Module
# --------------------------------------------

def check_progress():
    if not habit_logs:
        return "No completed habits yet — Let's build consistency! 👍"
    return f"You have completed {len(habit_logs)} habit actions so far. Keep going! 🌟"



# --------------------------------------------
# Main Chatbot Controller
# --------------------------------------------

def careguide_ai(message):
    """Main agent function combining modules and safety."""
    
    # Safety filter first
    if is_high_risk(message):
        return "⚠️ I can only provide general wellness information. For medical or emergency concerns, please contact a certified healthcare professional."

    intent = detect_intent(message)
    
    if intent == "set_reminder":
        return set_reminder(message)
    
    elif intent == "mark_done":
        return mark_habit_completion(message)
    
    elif intent == "check_progress":
        return check_progress()
    
    else:
        return wellness_response(message)



# Testing with example interactions

print(careguide_ai("Remind me to drink water"))
print(careguide_ai("I did yoga today"))
print(careguide_ai("Show my progress"))
print(careguide_ai("What are benefits of walking?"))
print(careguide_ai("What dose of medicine should I take?"))



import csv
import os

# Save reminders to CSV
def save_reminders(filename="reminders.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for reminder in reminders:
            writer.writerow([reminder])
    return "Reminders saved successfully!"

# Load reminders from CSV
def load_reminders(filename="reminders.csv"):
    if os.path.exists(filename):
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    reminders.append(row[0])
        return "Reminders loaded successfully!"
    return "No saved reminder file found."

# Save logs
def save_logs(filename="habit_logs.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for log in habit_logs:
            writer.writerow([log])
    return "Habit logs saved!"

# Load logs
def load_logs(filename="habit_logs.csv"):
    if os.path.exists(filename):
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    habit_logs.append(row[0])
        return "Habit logs loaded!"
    return "No saved habit logs file found."



# Save current data
print(save_reminders())
print(save_logs())

# Clear in-memory data
reminders.clear()
habit_logs.clear()

# Load from CSV
print(load_reminders())
print(load_logs())

print("Loaded reminders:", reminders)
print("Loaded logs:", habit_logs)



import matplotlib.pyplot as plt
from collections import Counter

def show_progress_chart():
    """Display a bar chart of completed habits using matplotlib."""
    
    if not habit_logs:
        return "No habit data to visualize yet."
    
    # Count frequency of completed habits
    habit_counts = Counter(habit_logs)
    
    # Extract labels and values
    labels = list(habit_counts.keys())
    values = list(habit_counts.values())
    
    # Plot bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.xlabel("Habit Type / Description")
    plt.ylabel("Completion Count")
    plt.title("Habit Completion Summary")
    plt.xticks(rotation=20)
    plt.show()

    return "Chart generated successfully!"



# Test Visualization
print(show_progress_chart())



habit_logs.extend(["yoga", "yoga", "water", "sleep", "water", "water"])
print(show_progress_chart())



def start_chat():
    print("👋 Welcome to CareGuide AI!")
    print("Type 'exit' anytime to stop.\n")

    while True:
        user_input = input("You: ")

        # Stop condition
        if user_input.lower() == "exit":
            print("CareGuide AI: Goodbye! Stay healthy and hydrated 💧✨")
            break

        # Get agent reply
        response = careguide_ai(user_input)
        print("CareGuide AI:", response)



start_chat()



# Predefined categories and related keywords
habit_categories = {
    "water": ["water", "drink", "hydration"],
    "yoga": ["yoga", "stretch", "asana"],
    "sleep": ["sleep", "nap", "rest", "bed"],
    "diet": ["diet", "healthy", "fruits", "vegetable", "protein"],
    "walk": ["walk", "steps", "running", "jog"]
}

def categorize_habit(message):
    msg = message.lower()
    for category, keywords in habit_categories.items():
        if any(keyword in msg for keyword in keywords):
            return category
    return "other"  # fallback category



print(mark_habit_completion("I drank two bottles of water"))
print(mark_habit_completion("I did yoga for 20 minutes"))
print(mark_habit_completion("I slept early tonight"))
print(mark_habit_completion("I ate a healthy meal"))
print(habit_logs)



def weekly_summary():
    if not habit_logs:
        return "No weekly data available yet. Try completing some habits! 🌱"

    from collections import Counter
    counts = Counter(habit_logs)

    total = sum(counts.values())
    top = counts.most_common(1)[0][0]

    summary = "📊 **Your Weekly Wellness Summary**\n\n"
    summary += f"• Total habits recorded: **{total}**\n"
    
    for habit, value in counts.items():
        summary += f"• {habit.capitalize()}: **{value} times**\n"

    summary += f"\n🎯 Most consistent habit: **{top.capitalize()}**\n"
    
    # Motivation line
    summary += "\n💡 Keep going — small daily habits create long-term transformation! 💪"

    return summary



print(weekly_summary())



from google import genai
client = genai.Client(api_key="YOUR_API_KEY_HERE")  # removed for security

def ai_motivation():
    """Generate a short motivational wellness quote using Gemini."""
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="Give a short, safe motivational quote under 10 words."
    )
    return response.text



print(ai_motivation())


