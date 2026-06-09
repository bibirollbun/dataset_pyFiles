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
from typing import List,Dict

print("import successfull")


from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
import json

@dataclass
class Memory:
    messages: List[Dict] = field(default_factory = list)
    max_history : int = 20

    def add(self, role, content):
        self.messages.append({
            "role": role,
            "content": content,
            "time": datetime.now().isoformat()
        })
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_context(self):
        out = ""
        for m in self.messages[-5:]:
            out += f"({m['role']}): {m['content']}\n"
        return out

class IntentAgent:
    def classify(self, message):
        text = message.lower()
        if "refund" in text:
            return "refund","high"
        if "cancel" in text:
            return "cancellation","high"
        if "invoice" in text or "bill" in text:
            return "billing","medium"
        if "text" in text:
            return "general_help","low"
        return "general","low"

class ReplyAgent:
    def create_reply(self, intent, urgency):
        if intent == "refund":
            return "I understand you want a refund. Please share your order id so I can assist you further."
        if intent == "cancellation":
            return "I can help you cancel your subscription. Kindly provide your registered email."
        if intent == "billing":
            return "It seems you have a billing concern. Please send your invoice number for verification."
        if intent == "general_help":
            return "Sure, I'm here to help you. Could you please share more details?"
        return "Thank you for your message. How can I assist you today?"

class Coordinator:
    def __init__(self):
        self.intent_agent = IntentAgent()
        self.reply_agent = ReplyAgent()
        self.memory = Memory()

    def ask(self, message):
        self.memory.add("user", message)
        intent, urgency = self.intent_agent.classify(message)
        final_output = {
            "intent": intent,
            "urgency": urgency,
        }
        reply = self.reply_agent.create_reply(intent, urgency)
        final_output["reply"] = reply
        self.memory.add("agent", reply)
        return final_output

agent = Coordinator()
message = [
    "I want to cancel my order.",
    "My invoice amount is wrong.",
    "I need a refund please.",
    "Hi, I need help."
]

def query(msg):
    print("USER", msg)
    out = agent.ask(msg)
    print(json.dumps(out, indent=2))
    print("-" * 50)

for msg in message:
    query(msg)





