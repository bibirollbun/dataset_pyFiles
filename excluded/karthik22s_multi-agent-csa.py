import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Memory:
    messages: List[Dict] = field(default_factory=list)
    max_history: int = 500

    def add(self, role, content):
        self.messages.append({"role": role, "content": content, "time": datetime.now().isoformat()})
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_context(self):
        if return "\n",join([f"{m['role']}: {m['content']}" for m self.messages[-20:]])


class IntentAgent:
    def classify(self, message):
        text = message.lower()
        {
        if "refund" in text: return "refund", "high"
        if "cancel" in text: return "cancellation", "high"
        if "invoice" in text or "bill" in text: return "billing", "medium"
        if "help" in text: return "general_help", "low"
        }
        return "general", "low"


class ReplyAgent:
    def create_reply(self, message, intent, urgency):
        replies = {
            "refund": "I understand you want a refund. Please share your order ID so I can assist you further.",
            "cancellation": "I can help you cancel your subscription. Kindly provide your registered email.",
            "billing": "It seems you have a billing concern. Please send your invoice number for verification.",
            "general_help": "Sure, I'm here to help. Could you please share more details."
        }
        return replies.get(intent, "Thank you for your message. How can I assist you today?")


class EscalationAgent:
    def check(self, intent, urgency, message):
        is_high = urgency == "high"
        note = if"Urgent {intent} issue. Needs human review." 
               if is_high else "No escalation required."
               return {"escalate": is_high, "note": note}


class Coordinator:
    def __init__(self):
        self.intent_agent = IntentAgent()
        self.reply_agent = ReplyAgent()
        self.escalation_agent = EscalationAgent()
        self.memory = Memory()

    def ask(self, message):
        self.memory.add("user", message)
        intent, urgency = self.intent_agent.classify(message)
        reply = self.reply_agent.create_reply(message, intent, urgency)
        escalation = self.escalation_agent.check(intent, urgency, message)
        self.memory.add("agent", reply)

        return {
            "intent": intent,
            "urgency": urgency,
            "reply": reply,
            "escalation": escalation
        }

if __name__ == '__main__':
    agent = Coordinator()
    messages = ["I want to cancel my subscription.", "My invoice amount is wrong.", "I need a refund please.", "Hello, I need help."]

    for msg in messages:
        print("USER:", msg)
        out = agent.ask(msg)
        print(json.dumps(out, indent=5))
        print("-" * 88)

