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

print("import successful")



import re
from datetime import datetime

# ---------------- Memory ----------------
class Memory:
    def __init__(self):
        self.user_name = ""
        self.last_messages = []

    def add(self, role, message):
        self.last_messages.append({"role": role, "message": message, "time": datetime.now().isoformat()})
        # Save user name if mentioned
        match = re.search(r'my name is ([A-Za-z ]+)', message, re.I)
        if match:
            self.user_name = match.group(1).strip()

    def context(self, n=5):
        return self.last_messages[-n:]

# ---------------- Intent ----------------
class IntentAgent:
    INTENTS = {
        "refund": ["refund", "money back", "return"],
        "cancellation": ["cancel", "unsubscribe", "stop subscription"],
        "billing": ["invoice", "bill", "payment", "charge"],
        "order_status": ["order id", "track", "status", "where is my order"],
        "greeting": ["hello", "hi", "hey"],
        "farewell": ["bye", "goodbye", "thanks", "thank you"]
    }

    def classify(self, message):
        message_lower = message.lower()
        found = [intent for intent, keywords in self.INTENTS.items() if any(k in message_lower for k in keywords)]
        return found if found else ["unknown"]

    def extract_info(self, message):
        info = {}
        order_ids = re.findall(r'order id[: ]*([\w\d-]+)', message, re.I)
        if order_ids:
            info["order_ids"] = order_ids
        emails = re.findall(r'[\w\.-]+@[\w\.-]+', message)
        if emails:
            info["emails"] = emails
        invoices = re.findall(r'invoice[ #]*([\w\d-]+)', message, re.I)
        if invoices:
            info["invoices"] = invoices
        return info

# ---------------- Reply ----------------
class ReplyAgent:
    TEMPLATES = {
        "greeting": "Hi {name}! How can I help you today?",
        "farewell": "Goodbye {name}! Thanks for chatting.",
        "refund": "I can help with refunds{details}.",
        "cancellation": "I can cancel your subscription{details}.",
        "billing": "I can help with billing{details}.",
        "order_status": "I can check your order status{details}.",
        "unknown": "Sorry {name}, I didn't understand that. Can you clarify?"
    }

    def create(self, intents, info, memory):
        name = memory.user_name or ""
        responses = []

        for intent in intents:
            details = ""
            if intent == "refund" and "order_ids" in info:
                details = f" for order(s): {', '.join(info['order_ids'])}"
            elif intent == "cancellation" and "emails" in info:
                details = f" for email(s): {', '.join(info['emails'])}"
            elif intent == "billing" and "invoices" in info:
                details = f" for invoice(s): {', '.join(info['invoices'])}"
            elif intent == "order_status" and "order_ids" in info:
                details = f" for order(s): {', '.join(info['order_ids'])}"

            template = self.TEMPLATES.get(intent, self.TEMPLATES["unknown"])
            responses.append(template.format(name=name, details=details))

        return " ".join(responses)

# ---------------- Coordinator ----------------
class ChatBot:
    def __init__(self):
        self.memory = Memory()
        self.intent_agent = IntentAgent()
        self.reply_agent = ReplyAgent()

    def chat(self, message):
        self.memory.add("user", message)
        intents = self.intent_agent.classify(message)
        info = self.intent_agent.extract_info(message)
        reply = self.reply_agent.create(intents, info, self.memory)
        self.memory.add("bot", reply)
        return reply

# ---------------- Demo ----------------
if __name__ == "__main__":
    bot = ChatBot()
    messages = [
        "Hello, my name is MD ESA.",
        "I want a refund and cancel my subscription.",
        "Here is my order ID: 75GH29 and my email is anu@email.com.",
        "My invoice #INV1928 needs billing help.",
        "Where's order ID 75GH29?",
        "Thanks! bye"
    ]
    for msg in messages:
        print("USER:", msg)
        print("BOT:", bot.chat(msg))
        print("-" * 40)





