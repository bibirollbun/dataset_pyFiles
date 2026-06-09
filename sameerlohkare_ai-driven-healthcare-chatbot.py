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


# Import libraries
import random
import re

# Simulate Large Language Model (LLM)-powered agent with simple intent classification
# Here we use keyword-based intent detection as proxy for LLM in free setup

# Define intents and training data (expandable)
intents = {
    'symptom_query': ['headache', 'fever', 'cough', 'pain', 'cold', 'flu', 'chest pain'],
    'appointment_booking': ['book', 'appointment', 'schedule', 'meet', 'clinic', 'doctor'],
    'symptom_info': ['symptom', 'sign', 'indicate', 'cause', 'disease'],
    'mental_health_support': ['anxiety', 'depression', 'stress', 'mental', 'sad'],
    'exit': ['exit', 'quit', 'bye', 'stop']
}

# Simple function to detect intent from user input
def detect_intent(user_input):
    user_input = user_input.lower()
    for intent, keywords in intents.items():
        if any(word in user_input for word in keywords):
            return intent
    return 'unknown'

# Tool: Simple symptom checker database (custom tool)
symptom_database = {
    'headache': "Possible causes: tension, dehydration, migraine. Stay hydrated and rest. Consult a doctor if severe.",
    'fever': "Fever may be due to infection. Monitor temperature regularly and consult a healthcare provider if >38°C persists.",
    'cough': "Common causes: cold, flu, allergies. Avoid irritants and seek medical care if cough lasts more than 2 weeks.",
    'chest pain': "Chest pain may be serious. Seek emergency help immediately if accompanied by shortness of breath or dizziness.",
    'anxiety': "Anxiety symptoms include worry, restlessness. Consider techniques like mindfulness or consulting a counselor."
}

# Session & memory management: Store conversation history per user (simple in-memory dictionary)
session_memory = []

# Multi-agent system simulation: Sequential agents
# Agent 1: LLM-powered agent for intent detection
# Agent 2: Tool agent for symptom info provision

def llm_agent(user_input):
    intent = detect_intent(user_input)
    return intent

def tool_agent(intent, user_input):
    if intent == 'symptom_query':
        # Find symptoms mentioned
        symptoms_mentioned = [symptom for symptom in symptom_database if symptom in user_input.lower()]
        if symptoms_mentioned:
            responses = [symptom_database[s] for s in symptoms_mentioned]
            return " ".join(responses)
        else:
            return "Please specify your symptom for assistance."
    elif intent == 'appointment_booking':
        return "Please provide preferred date and time for appointment booking."
    elif intent == 'symptom_info':
        return "Please tell me the symptom or condition you want information about."
    elif intent == 'mental_health_support':
        return "Mental health is important. Consider talking to a professional or trying breathing exercises."
    elif intent == 'exit':
        return "Thank you for using the healthcare chatbot. Stay healthy!"
    else:
        return "Sorry, I didn't understand that. Could you please rephrase?"

def chatbot_response(user_input):
    # Save to session memory
    session_memory.append({'user': user_input})

    # Run multi-agent sequentially
    intent = llm_agent(user_input)
    response = tool_agent(intent, user_input)

    # Save response to session memory
    session_memory.append({'bot': response})
    return response

# Interactive chat loop (example)
print("Healthcare Chatbot at your service. Type 'exit' to quit.")
while True:
    user_text = input("You: ")
    if user_text.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: Thank you! Take care!")
        break
    reply = chatbot_response(user_text)
    print("Chatbot:", reply)

# End of notebook

