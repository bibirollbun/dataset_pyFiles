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
from dataclasses import dataclass, field
from typing import List, Dict

print("Import Successfull.")


import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Memory:
    short_term: List[Dict] = field(default_factory=list)
    long_term: List[Dict] = field(default_factory=list)
    max_short: int = 15
    max_long: int = 200

    def add(self, role, content):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.short_term.append(entry)

        # Trim short-term memory
        if len(self.short_term) > self.max_short:
            self.short_term.pop(0)

        # Auto-store important info to long-term
        if role == "user" and self._is_important(content):
            self.long_term.append(entry)
            if len(self.long_term) > self.max_long:
                self.long_term.pop(0)

    def _is_important(self, text):
        # basic heuristic for long-term memory
        keywords = ["name", "email", "address", "deadlines", "birthday"]
        return any(word in text.lower() for word in keywords)

    def get_context(self):
        """Return recent conversation context for LLM."""
        context = ""
        for m in self.short_term[-6:]:
            context += f"{m['role']}: {m['content']}\n"
        return context

class IntentAgent:
    def classify(self, message):
        text = message.lower()

        # Expanded intent detection
        intent_map = {
            "refund": ["refund", "money back"],
            "cancellation": ["cancel", "stop service"],
            "billing": ["invoice", "bill", "payment"],
            "technical_issue": ["error", "bug", "not working", "issue"],
            "order_status": ["order", "delivery", "tracking"],
            "greeting": ["hello", "hi", "hey"],
            "complaint": ["bad", "worst", "terrible", "dissatisfied"],
            "general_help": ["help", "support"],
        }

        for intent, keywords in intent_map.items():
            if any(k in text for k in keywords):
                urgency = self._determine_urgency(intent)
                return intent, urgency

        return "general", "low"

    def _determine_urgency(self, intent):
        high = ["refund", "cancellation", "complaint", "technical_issue"]
        medium = ["billing", "order_status"]
        if intent in high:
            return "high"
        if intent in medium:
            return "medium"
        return "low"

class SearchTool:
    def run(self, query):
        return f"(Simulated search results about '{query}')"


class FAQTool:
    FAQ_DB = {
        "refund": "Refunds are processed within 5–7 business days.",
        "cancel": "You can cancel anytime from your account settings.",
        "invoice": "Invoices can be downloaded from the Billing section."
    }

    def lookup(self, message):
        for key in self.FAQ_DB:
            if key in message.lower():
                return self.FAQ_DB[key]
        return None


class TaskTool:
    def execute(self, task):
        return f"Executing task: {task} (simulated)"

class ReplyAgent:
    def __init__(self):
        self.search_tool = SearchTool()
        self.faq_tool = FAQTool()
        self.task_tool = TaskTool()

    def create_reply(self, message, intent, urgency, memory_context):
        # First try FAQ
        faq_response = self.faq_tool.lookup(message)
        if faq_response:
            return faq_response

        # Intent-based replies
        responses = {
            "refund": "I understand you want a refund. Please provide your order ID.",
            "cancellation": "Sure, I can assist with cancellation. What is your registered email?",
            "billing": "I can help with billing issues. Can you share your invoice number?",
            "technical_issue": "Sorry you're facing issues. Describe the error in detail.",
            "order_status": "Let me check your order. Please send your order number.",
            "complaint": "I apologize for your experience. Could you explain the problem?",
            "greeting": random.choice(["Hello! How can I assist you?", "Hey there! What can I do for you?"]),
            "general_help": "I'm here to help! Please share more details.",
            "general": "Thank you for your message. How can I assist?"
        }

        base = responses.get(intent, responses["general"])

        # Add urgency tone
        if urgency == "high":
            base = "⚠️ HIGH PRIORITY: " + base

        # Add memory-based personalization
        return base + "\n\n" + "Context so far:\n" + memory_context

class Coordinator:
    def __init__(self):
        self.intent_agent = IntentAgent()
        self.reply_agent = ReplyAgent()
        self.memory = Memory()

    def ask(self, message):
        self.memory.add("user", message)

        intent, urgency = self.intent_agent.classify(message)
        context = self.memory.get_context()
        reply = self.reply_agent.create_reply(message, intent, urgency, context)

        # Save reply to memory
        self.memory.add("agent", reply)

        return {
            "intent": intent,
            "urgency": urgency,
            "reply": reply,
            "timestamp": datetime.now().isoformat()
        }

agent = Coordinator()
test_messages = [
    "Hello",
    "I want to cancel my subscription.",
    "My invoice amount is wrong.",
    "I need a refund please.",
    "I'm facing an error in your app!",
    "Where is my order?",
    "Your service is terrible!",
]

for msg in test_messages:
    print("USER:", msg)
    out = agent.ask(msg)
    print(json.dumps(out, indent=2))
    print("-" * 50)


