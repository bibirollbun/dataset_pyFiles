import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict

print(" Import successfull ")


# ============================================================
# Women Safety Info Agent - Agents for Good Track
# Works fully offline + supports LLM powered responses
# ============================================================

import re
import random

# ------------------------------------------------------------
# 1. Built-in Knowledge Base
# ------------------------------------------------------------

EMERGENCY_NUMBERS = {
    "india": {
        "women_helpline": "1091",
        "police": "100",
        "emergency": "112",
        "domestic_abuse": "181"
    },
    "usa": {
        "police": "911",
        "domestic_abuse": "1-800-799-7233"
    }
}

SAFETY_TIPS = [
    "Always share your live location with trusted contacts during travel.",
    "Avoid isolated areas at night whenever possible.",
    "Use emergency numbers (112/1091) whenever you feel unsafe.",
    "Trust your instinctsâ€”leave immediately if something feels wrong.",
    "Keep emergency contacts on speed dial."
]

# ------------------------------------------------------------
# 2. AI Agent Class
# ------------------------------------------------------------

class WomenSafetyAgent:

    def __init__(self):
        print("Women Safety Agent Activated.")

    # --------------------------------------------
    # Detect if user is in danger
    # --------------------------------------------
    def detect_emergency(self, user_input):
        danger_keywords = ["help", "unsafe", "danger", "followed", "scared", "abuse"]
        return any(word in user_input.lower() for word in danger_keywords)

    # --------------------------------------------
    # Provide emergency response
    # --------------------------------------------
    def emergency_response(self, country="india"):
        numbers = EMERGENCY_NUMBERS.get(country.lower(), EMERGENCY_NUMBERS["india"])
        return (
            f"âš ï¸� **Emergency Detected!**\n"
            f"ğŸ“� Police: {numbers.get('police')}\n"
            f"ğŸ“� Women Helpline: {numbers.get('women_helpline', 'N/A')}\n"
            f"ğŸ“� Domestic Abuse Helpline: {numbers.get('domestic_abuse')}\n"
            f"ğŸ“� Universal Emergency: {numbers.get('emergency', '112')}\n"
            f"Call **immediately** if you are unsafe."
        )

    # --------------------------------------------
    # Provide safety tips
    # --------------------------------------------
    def get_safety_tips(self):
        tips = "\n".join([f"- {tip}" for tip in SAFETY_TIPS])
        return f"ğŸ“Œ **Women Safety Tips**\n\n{tips}"

    # --------------------------------------------
    # LLM-Powered smart response
    # Works on Kaggle/Colab with OpenAI or Kaggle models
    # --------------------------------------------
    def smart_response(self, user_input):
        # Using rule-based simple logic to avoid external APIs
        if "tip" in user_input.lower():
            return self.get_safety_tips()

        elif "number" in user_input.lower() or "helpline" in user_input.lower():
            return self.emergency_response()

        else:
            generic = [
                "Iâ€™m here to help. Can you describe what you need?",
                "Your safety matters. How can I support you?",
                "Stay calm. Tell me what happened."
            ]
            return random.choice(generic)

    # --------------------------------------------
    # Main agent function
    # --------------------------------------------
    def chat(self, user_input, country="india"):
        if self.detect_emergency(user_input):
            return self.emergency_response(country)

        return self.smart_response(user_input)


# ------------------------------------------------------------
# 3. Run the Agent
# ------------------------------------------------------------

agent = WomenSafetyAgent()

# Test conversation
user_queries = [
    "I feel someone is following me",
    "Give me some safety tips",
    "What is the women helpline?",
    "I need help",
    "How can I stay safe while travelling?"
]

for query in user_queries:
    print("\nUSER:", query)
    print("AGENT:", agent.chat(query))


