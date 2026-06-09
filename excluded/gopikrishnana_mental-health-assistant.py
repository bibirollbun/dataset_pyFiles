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


@dataclass
class Memory:
    messages: List[Dict] = field(default_factory=list)
    max_history: int = 10

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
        # Show only last 3 messages for brevity
        for m in self.messages[-3:]:
            out += f"{m['role']}: {m['content']}\n"
        return out



class TriageAgent:
    def classify(self, message: str) -> tuple[str, str]:
        text = message.lower()
        
        # CRITICAL risk (Needs external intervention)
        if any(k in text for k in ["harm myself", "suicide", "emergency", "kill myself"]):
            return "Crisis", "CRITICAL"
        
        # HIGH risk (Needs immediate coping strategy)
        if any(k in text for k in ["panic", "anxiety", "overwhelmed", "can't breathe"]):
            return "Acute Anxiety", "HIGH"
        
        # MEDIUM risk (Needs empathy and simple tasking)
        if any(k in text for k in ["sad", "depressed", "lonely", "low"]):
            return "Low Mood", "MEDIUM"
            
        return "Neutral", "LOW"


class SupportAgent:
    def create_reply(self, intent: str, urgency: str) -> str:
        if intent == "Crisis":
            return "ðŸš¨ SAFETY ALERT. Please call or text **988** immediately. Your safety is the priority."
        
        if intent == "Acute Anxiety":
            return "Let's try **4-7-8 breathing**. Inhale for 4, hold for 7, exhale for 8. You can do this."
            
        if intent == "Low Mood":
            return "I'm here for you. What is one small step you can take to make today 1% better?"
            
        return "Thanks for reaching out. What's on your mind today?"


class SafetyAgent:
    def check(self, intent: str, urgency: str) -> Dict:
        if urgency == "CRITICAL":
            return {
                "protocol": "CRISIS ALERT",
                "referral": True,
                "note": "Mandatory 988 referral."
            }
        if urgency == "HIGH":
            return {
                "protocol": "Coping Strategy",
                "referral": True,
                "note": "Recommend professional consult."
            }
        return {
            "protocol": "Listening",
            "referral": False,
            "note": "Continue automated support."
        }



class MentalHealthCoordinator:
    def __init__(self):
        self.triage_agent = TriageAgent()
        self.support_agent = SupportAgent()
        self.safety_agent = SafetyAgent()
        self.memory = Memory()

    def ask(self, message: str) -> Dict:
        # 1. User Input & Memory
        self.memory.add("user", message)
        
        # 2. Triage & Classification
        intent, urgency = self.triage_agent.classify(message)
        
        # 3. Generate Reply
        reply = self.support_agent.create_reply(intent, urgency)
        
        # 4. Safety Check
        safety_check = self.safety_agent.check(intent, urgency)

        final_output = {
            "intent": intent,
            "urgency": urgency,
            "reply": reply,
            "safety_check": safety_check,
            "context": self.memory.get_context().strip()
        }

        # 5. Agent Reply & Memory
        self.memory.add("agent", reply)
        return final_output


assistant = MentalHealthCoordinator()

conversations = [
    "Hello, just checking in.",
    "I am just really sad and lonely today.",
    "I'm feeling intense anxiety right now, I need help.",
    "I feel like I want to harm myself."
]

print("--- STARTING MENTAL HEALTH ASSISTANT DEMO ---")
for i, msg in enumerate(conversations):
    print(f"\n[{i+1}] USER: {msg}")
    out = assistant.ask(msg)
    print(json.dumps(out, indent=2))
    print("-" * 50)




