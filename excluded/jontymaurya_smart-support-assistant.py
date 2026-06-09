"""
Smart Support Assistant - Powered by Multiple Agents

A simple multi-agent customer support system with:
- IntentAgent       -> classifies intent and urgency
- ReplyAgent        -> generates a support-style reply
- EscalationAgent   -> decides if human escalation is needed
- CoordinatorAgent  -> orchestrates agents and returns a JSON result
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime


# -------------------------------------------------------------------
# ENUMS & DATA MODELS
# -------------------------------------------------------------------

class IntentLabel(str, Enum):
    REFUND = "refund"
    CANCELLATION = "cancellation"
    BILLING_ISSUE = "billing_issue"
    TECHNICAL_ISSUE = "technical_issue"
    ACCOUNT_ISSUE = "account_issue"
    GENERAL_QUERY = "general_query"
    FEEDBACK = "feedback"
    OTHER = "other"


class UrgencyLabel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class IntentResult:
    intent: IntentLabel
    urgency: UrgencyLabel
    confidence: float
    meta: Dict[str, Any]


@dataclass
class ReplyResult:
    reply_text: str
    tone: str  # e.g. "formal", "friendly"
    language: str  # e.g. "en"


@dataclass
class EscalationResult:
    should_escalate: bool
    reason: Optional[str]
    priority: Optional[str]  # e.g. "P1", "P2", "P3"


@dataclass
class CoordinatorResult:
    customer_id: str
    original_message: str
    timestamp: str
    intent: IntentResult
    reply: ReplyResult
    escalation: EscalationResult
    raw_output: Dict[str, Any]

    def to_json(self, pretty: bool = True) -> str:
        data = asdict(self)
        return json.dumps(data, indent=2 if pretty else None)


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def extract_amount(text: str) -> Optional[float]:
    """
    Very simple regex-based amount extractor.
    Looks for patterns like: $50, 50$, 50 USD, 1,299.99 etc.
    Returns the first amount found, or None.
    """
    pattern = r"(?:\$|₹)?\s*([0-9]{1,3}(?:[,0-9]{3})*(?:\.[0-9]+)?)"
    match = re.search(pattern, text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    return text.strip().lower()


# -------------------------------------------------------------------
# AGENT IMPLEMENTATIONS
# -------------------------------------------------------------------

class IntentAgent:
    """
    Rule-based intent & urgency classifier.
    In a production system this could be a trained classifier or LLM.
    """

    # keyword dictionaries for each intent
    INTENT_KEYWORDS = {
        IntentLabel.REFUND: [
            "refund", "money back", "chargeback", "return my money"
        ],
        IntentLabel.CANCELLATION: [
            "cancel", "cancellation", "stop my subscription", "end my plan"
        ],
        IntentLabel.BILLING_ISSUE: [
            "bill", "billing", "invoice", "charged", "payment issue",
            "overcharged", "wrong amount"
        ],
        IntentLabel.TECHNICAL_ISSUE: [
            "bug", "error", "issue", "not working", "stuck", "crash",
            "unable to login", "can't login", "cannot login"
        ],
        IntentLabel.ACCOUNT_ISSUE: [
            "account", "profile", "password", "username", "login problem",
            "email change"
        ],
        IntentLabel.FEEDBACK: [
            "feedback", "suggestion", "idea", "improve", "love the app",
            "like the app"
        ],
    }

    URGENCY_KEYWORDS_HIGH = [
        "urgent", "asap", "immediately", "right now", "today",
        "fraud", "scam", "hacked", "unauthorized"
    ]

    URGENCY_KEYWORDS_MEDIUM = [
        "soon", "whenever possible", "this week", "help me quickly"
    ]

    def classify_intent(self, message: str) -> IntentResult:
        text = normalize_text(message)

        # 1. Detect intent by simple keyword search
        matched_intent = IntentLabel.OTHER
        matched_score = 0.0

        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(k in text for k in keywords)
            if score > matched_score:
                matched_intent = intent
                matched_score = float(score)

        # default fall-back
        if matched_intent == IntentLabel.OTHER and any(
            w in text for w in ["how", "what", "where", "when", "why"]
        ):
            matched_intent = IntentLabel.GENERAL_QUERY
            matched_score = 1.0

        # rough confidence score
        confidence = min(1.0, 0.3 + 0.2 * matched_score)

        # 2. Detect urgency
        urgency = UrgencyLabel.LOW
        if any(word in text for word in self.URGENCY_KEYWORDS_HIGH):
            urgency = UrgencyLabel.HIGH
        elif any(word in text for word in self.URGENCY_KEYWORDS_MEDIUM):
            urgency = UrgencyLabel.MEDIUM
        else:
            # heuristic: money + refund/billing tends to be medium
            amount = extract_amount(text)
            if amount is not None and amount > 0:
                if amount >= 500:   # large amount => high urgency
                    urgency = UrgencyLabel.HIGH
                else:
                    urgency = UrgencyLabel.MEDIUM

        meta = {
            "amount_detected": extract_amount(text),
            "matched_score": matched_score,
        }

        return IntentResult(
            intent=matched_intent,
            urgency=urgency,
            confidence=confidence,
            meta=meta,
        )


class ReplyAgent:
    """
    Generates a short customer support style reply using intent & urgency.
    """

    def generate_reply(
        self,
        message: str,
        intent_result: IntentResult,
    ) -> ReplyResult:
        intent = intent_result.intent
        urgency = intent_result.urgency

        # Simple templates per intent
        if intent == IntentLabel.REFUND:
            text = (
                "Thanks for reaching out about a refund. "
                "Please share your registered email and the transaction details "
                "so we can review and process this as quickly as possible."
            )
        elif intent == IntentLabel.CANCELLATION:
            text = (
                "I can help you cancel your subscription. "
                "Please confirm your registered email or account ID, and "
                "we'll proceed with the cancellation and share a confirmation."
            )
        elif intent == IntentLabel.BILLING_ISSUE:
            text = (
                "I understand you're facing a billing issue. "
                "Please send us a screenshot of the invoice or the last "
                "charge you received so we can investigate and correct it."
            )
        elif intent == IntentLabel.TECHNICAL_ISSUE:
            text = (
                "Sorry you're experiencing technical trouble. "
                "Please let us know the device you're using and any error "
                "messages you see, and we'll help you fix it."
            )
        elif intent == IntentLabel.ACCOUNT_ISSUE:
            text = (
                "It looks like you're having an account-related issue. "
                "Please confirm your registered email and describe the "
                "problem in a bit more detail so we can assist you."
            )
        elif intent == IntentLabel.FEEDBACK:
            text = (
                "Thank you for sharing your feedback. "
                "We really appreciate you taking the time to help us improve "
                "and will review your suggestions with our team."
            )
        elif intent == IntentLabel.GENERAL_QUERY:
            text = (
                "Thanks for your question. "
                "Let us know any extra details you have so we can give you "
                "a clear and accurate answer."
            )
        else:
            text = (
                "Thanks for contacting support. "
                "We’ve received your message and will review it shortly. "
                "If you can provide any additional details, that will help us "
                "resolve this faster."
            )

        # Add a small urgency-specific line
        if urgency == UrgencyLabel.HIGH:
            text += " This request has been marked as high priority."
        elif urgency == UrgencyLabel.MEDIUM:
            text += " We’ll get back to you as soon as possible."

        return ReplyResult(
            reply_text=text,
            tone="professional_friendly",
            language="en",
        )


class EscalationAgent:
    """
    Decides whether the ticket should be escalated to a human support agent.
    """

    def decide_escalation(
        self,
        message: str,
        intent_result: IntentResult,
    ) -> EscalationResult:
        text = normalize_text(message)
        amount = intent_result.meta.get("amount_detected")
        intent = intent_result.intent
        urgency = intent_result.urgency

        # Base rule: high urgency => escalate
        if urgency == UrgencyLabel.HIGH:
            return EscalationResult(
                should_escalate=True,
                reason="High urgency message.",
                priority="P1",
            )

        # Escalate if money-related and amount is large
        if intent in {IntentLabel.REFUND, IntentLabel.BILLING_ISSUE}:
            if amount is not None and amount >= 200:
                return EscalationResult(
                    should_escalate=True,
                    reason=f"Refund/Billing issue with high amount ({amount}).",
                    priority="P1" if amount >= 1000 else "P2",
                )

        # Escalate if user mentions fraud or hacking
        fraud_words = ["fraud", "scam", "hacked", "unauthorized"]
        if any(w in text for w in fraud_words):
            return EscalationResult(
                should_escalate=True,
                reason="Potential fraud or account compromise.",
                priority="P1",
            )

        # Technical issues can often be auto-handled, but escalate
        # if message mentions repeated failures.
        if intent == IntentLabel.TECHNICAL_ISSUE and any(
            w in text for w in ["again", "multiple times", "keeps happening"]
        ):
            return EscalationResult(
                should_escalate=True,
                reason="Repeated technical failures reported.",
                priority="P2",
            )

        # Default: no escalation (low-impact issue or general query)
        return EscalationResult(
            should_escalate=False,
            reason=None,
            priority=None,
        )


# -------------------------------------------------------------------
# COORDINATOR (BRAIN) AGENT
# -------------------------------------------------------------------

class CoordinatorAgent:
    """
    Orchestrates all agents:
    - Stores a simple in-memory conversation history
    - Calls IntentAgent, ReplyAgent, EscalationAgent
    - Returns a structured JSON-friendly result
    """

    def __init__(self):
        self.intent_agent = IntentAgent()
        self.reply_agent = ReplyAgent()
        self.escalation_agent = EscalationAgent()

        # Very simple "memory": dict[customer_id] -> list of messages
        self.memory: Dict[str, List[Dict[str, Any]]] = {}

    def _store_message(
        self,
        customer_id: str,
        message: str,
        role: str = "user",
    ) -> None:
        history = self.memory.setdefault(customer_id, [])
        history.append(
            {
                "role": role,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def handle_message(
        self,
        customer_id: str,
        message: str,
    ) -> CoordinatorResult:
        """
        Main entry point for the system.
        """
        # Step 1: persist message
        self._store_message(customer_id, message, role="user")

        # Step 2: Intent & urgency
        intent_result = self.intent_agent.classify_intent(message)

        # Step 3: Reply generation
        reply_result = self.reply_agent.generate_reply(message, intent_result)

        # Step 4: Escalation decision
        escalation_result = self.escalation_agent.decide_escalation(
            message, intent_result
        )

        # Step 5: Store assistant reply in memory
        self._store_message(customer_id, reply_result.reply_text, role="assistant")

        # Build final structured result
        result = CoordinatorResult(
            customer_id=customer_id,
            original_message=message,
            timestamp=datetime.utcnow().isoformat(),
            intent=intent_result,
            reply=reply_result,
            escalation=escalation_result,
            raw_output={
                "memory_length": len(self.memory.get(customer_id, [])),
                "conversation_history": self.memory.get(customer_id, []),
            },
        )

        return result


# -------------------------------------------------------------------
# DEMO / TEST HARNESS
# -------------------------------------------------------------------

if __name__ == "__main__":
    """
    Example usage for Kaggle notebook:
    - Run this cell
    - Inspect the JSON for different example messages
    """

    assistant = CoordinatorAgent()

    test_messages = [
        "I need a refund for the last payment of $49, it was charged twice.",
        "Please cancel my subscription immediately. This is urgent.",
        "My invoice amount is wrong, it should not be this high.",
        "The app keeps crashing again and again when I try to log in.",
        "I think my account was hacked. There are unauthorized charges.",
        "Just wanted to say I love the product and have a small suggestion."
    ]

    for i, msg in enumerate(test_messages, start=1):
        customer_id = f"user_{i}"
        print("=" * 80)
        print(f"Customer: {customer_id}")
        print(f"Message : {msg}")
        print("-" * 80)

        result = assistant.handle_message(customer_id, msg)
        print(result.to_json(pretty=True))
        print("\n")


