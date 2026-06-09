# it_helpdesk_multi_agent.py
# Simple Multi-Agent IT Helpdesk Ticket Classifier & Auto-Responder
# Author: Nandhini-style template (adapt as needed)

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# -------------------------
# Memory System
# -------------------------
@dataclass
class Memory:
    messages: List[Dict] = field(default_factory=list)
    max_history: int = 50

    def add(self, role: str, content: str):
        entry = {
            "role": role,
            "content": content,
            "time": datetime.now().isoformat()
        }
        self.messages.append(entry)
        # keep recent history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_context(self, n: int = 5) -> str:
        out = ""
        for m in self.messages[-n:]:
            out += f"{m['role']}: {m['content']}\n"
        return out

# -------------------------
# Intent Classification Agent
# -------------------------
class IntentAgent:
    """
    Simple rule-based intent classifier for IT tickets.
    Returns: (intent, category_confidence)
    """
    def __init__(self):
        # keywords mapped to intents
        self.map = {
            "network": ["network", "wifi", "internet", "latency", "slow"],
            "hardware": ["keyboard", "mouse", "monitor", "broken", "battery", "power", "hardware", "fan"],
            "software": ["install", "error", "crash", "update", "patch", "license", "bug", "software"],
            "access": ["access", "permission", "cannot login", "login", "locked", "credentials"],
            "printer": ["printer", "printing", "paper jam", "toner", "scan"],
            "other": []
        }

    def classify(self, message: str) -> Tuple[str, float]:
        text = message.lower()
        scores = {k: 0 for k in self.map.keys()}
        words = text.split()
        for intent, kws in self.map.items():
            for kw in kws:
                if kw in text:
                    scores[intent] += 1
        # choose best intent
        best = max(scores.items(), key=lambda x: x[1])
        intent, score = best
        # if no keywords matched, fallback heuristics
        if score == 0:
            # quick heuristics
            if "help" in text or "support" in text:
                return "other", 0.6
            return "other", 0.4
        # confidence scaled (simple)
        total = sum(scores.values()) if sum(scores.values()) > 0 else 1
        confidence = float(score) / total
        return intent, round(confidence, 2)

# -------------------------
# Priority Scorer (small helper)
# -------------------------
class PriorityScorer:
    """
    Scoring based on words indicating urgency, user status, or SLA risk.
    Returns 'low', 'medium', 'high'
    """
    urgent_words = ["urgent", "immediately", "asap", "critical", "down", "cannot", "can't", "failed"]
    def score(self, message: str, intent: str) -> str:
        text = message.lower()
        urgency = 0
        for w in self.urgent_words:
            if w in text:
                urgency += 1
        # if 'down' and network/hardware -> high
        if ("down" in text or "not working" in text) and intent in ["network", "hardware"]:
            urgency += 2
        # simple thresholds
        if urgency >= 2:
            return "high"
        if urgency == 1:
            return "medium"
        return "low"

# -------------------------
# Reply Generator Agent
# -------------------------
class ReplyAgent:
    def create_reply(self, message: str, intent: str, urgency: str, context: str = "") -> str:
        base = ""
        if intent == "network":
            base = ("Thanks — we detected a network issue. "
                    "Please confirm whether this affects multiple users or only you, "
                    "and share the device (Windows/Mac/Linux) and location (office/home).")
        elif intent == "hardware":
            base = ("Looks like a hardware problem. Please provide the device serial number (if available), "
                    "the exact model, and a short description of the failure (e.g., won't power on, noisy fan).")
        elif intent == "software":
            base = ("This appears to be a software issue. Please send the exact error message (or a screenshot), "
                    "application name and version, and the steps to reproduce if possible.")
        elif intent == "access":
            base = ("Access issue detected. Please provide your username/email and the resource you're trying to access. "
                    "If you saw an error, paste it here.")
        elif intent == "printer":
            base = ("Printer-related issue. Share the printer model, error displayed, and whether others can print.")
        else:
            base = ("Thanks for reaching out. Could you share more details so we can help (screenshots, steps, any error text)?")

        # urgency hinting
        if urgency == "high":
            base = "[PRIORITY: HIGH] " + base
        elif urgency == "medium":
            base = "[PRIORITY: MEDIUM] " + base

        # append contextual hint if available
        if context:
            base += f" (Context: {context.strip()})"

        return base

# -------------------------
# Escalation Agent
# -------------------------
class EscalationAgent:
    """
    Decide whether to escalate to L2/L3 or notify on-call.
    Returns a dict with escalate boolean and recommended team.
    """
    def check(self, intent: str, urgency: str, message: str, confidence: float) -> Dict:
        escalate = False
        team = None
        note = "No escalation required."

        # rules:
        if urgency == "high":
            escalate = True
            # route by intent
            if intent == "network":
                team = "Network Ops (L2)"
            elif intent == "hardware":
                team = "Hardware Support (L2)"
            elif intent == "software":
                team = "Application Support (L2)"
            else:
                team = "Service Desk (L2)"
            note = f"High urgency. Route to {team}."
        else:
            # escalate if classifier had low confidence
            if confidence < 0.4:
                escalate = True
                team = "Service Desk (review)"
                note = "Low classification confidence; please review."
        return {
            "escalate": escalate,
            "team": team,
            "note": note
        }

# -------------------------
# Coordinator (Orchestrator)
# -------------------------
class Coordinator:
    def __init__(self):
        self.memory = Memory()
        self.intent_agent = IntentAgent()
        self.priority_scorer = PriorityScorer()
        self.reply_agent = ReplyAgent()
        self.escalation_agent = EscalationAgent()

    def ask(self, message: str, user_id: Optional[str] = None) -> Dict:
        # store incoming
        meta_user = f"user:{user_id}" if user_id else "user"
        self.memory.add(meta_user, message)

        # classify intent
        intent, confidence = self.intent_agent.classify(message)

        # compute urgency
        urgency = self.priority_scorer.score(message, intent)

        # build small context (last 4 messages)
        context = self.memory.get_context(n=4)

        # create reply
        reply = self.reply_agent.create_reply(message, intent, urgency, context=context)

        # escalation decision
        escalation = self.escalation_agent.check(intent, urgency, message, confidence)

        # agent logs reply to memory
        self.memory.add("agent", reply)

        # final structured output
        out = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "intent": intent,
            "confidence": confidence,
            "urgency": urgency,
            "reply": reply,
            "escalation": escalation,
            "context_snapshot": context
        }
        return out

# -------------------------
# Demo / Test
# -------------------------
if __name__ == "__main__":
    agent = Coordinator()

    test_messages = [
        ("user_101", "Our office WiFi is down since 10 AM, urgent - cannot work."),
        ("user_202", "My laptop battery is swelling and it won't charge."),
        ("user_303", "App crashes with error code 0x8f when opening Reports."),
        ("user_404", "I need access to the finance shared drive, my account says permission denied."),
        ("user_505", "Printer 5 has paper jam, tried clearing but still not printing."),
        ("user_606", "Hello, I need support with something but not sure what category it is.")
    ]

    for uid, msg in test_messages:
        print("USER:", uid, msg)
        out = agent.ask(msg, user_id=uid)
        print(json.dumps(out, indent=2))
        print("-" * 60)


