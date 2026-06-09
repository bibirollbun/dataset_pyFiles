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


from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


from dataclasses import dataclass, field
from datetime import datetime

# ============================================
# Conversation Memory Agent
# ============================================
@dataclass
class ChatMemory:
    history: list = field(default_factory=list)
    limit: int = 20

    def remember(self, speaker, text):
        self.history.append({
            "role": speaker,
            "message": text,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.history) > self.limit:
            self.history = self.history[-self.limit:]

    def recent_context(self, n=5):
        context = ""
        for item in self.history[-n:]:
            context += f"{item['role']}: {item['message']}\n"
        return context


# ============================================
# Intent Detection Agent
# ============================================
class IntentDetector:
    def detect(self, msg):
        text = msg.lower()

        if any(word in text for word in ["hi", "hello", "hey"]):
            return "greeting", "low"

        if "refund" in text:
            return "refund", "high"

        if "cancel" in text or "unsubscribe" in text:
            return "cancellation", "high"

        if "invoice" in text or "bill" in text or "payment" in text:
            return "billing", "medium"

        if "help" in text or "assist" in text:
            return "general_help", "low"

        return "unknown", "low"


# ============================================
# Reply Agent
# ============================================
class ResponseAgent:
    def generate(self, intent, urgency):
        responses = {
            "greeting": "Hello! How can I assist you today?",
            "refund": "Understood. Please provide your order number so I can initiate your refund request.",
            "cancellation": "I can help you cancel your subscription. May I have your registered email address?",
            "billing": "It seems you are facing a billing issue. Kindly share your invoice ID for verification.",
            "general_help": "I'm here to help! Please tell me what you need assistance with.",
            "unknown": "I'm not sure I understand. Could you provide more details?"
        }
        return responses.get(intent, "How can I assist you today?")


# ============================================
# Main Chatbot System
# ============================================
class SupportBot:
    def __init__(self):
        self.memory = ChatMemory()
        self.intent_detector = IntentDetector()
        self.response_agent = ResponseAgent()

    def reply(self, user_input):
        # Save user message
        self.memory.remember("User", user_input)

        # Detect intent
        intent, urgency = self.intent_detector.detect(user_input)

        # Generate response
        bot_response = self.response_agent.generate(intent, urgency)

        # Save bot response
        self.memory.remember("Bot", bot_response)

        return bot_response

    def show_history(self):
        return self.memory.recent_context(10)


# ============================================
# Demo
# ============================================
bot = SupportBot()

print(bot.reply("Hello"))
print(bot.reply("I want a refund"))
print(bot.reply("My invoice is missing"))
print(bot.reply("Cancel my subscription"))
print("\nConversation History:\n")
print(bot.show_history())


