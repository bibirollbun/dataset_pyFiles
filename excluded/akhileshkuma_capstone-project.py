from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


# ==========================
# Simple logger
# ==========================

def log(message: str) -> None:
    """Simple logging helper."""
    print(f"[{datetime.now().isoformat()}] {message}")


# ==========================
# Core data models
# ==========================

@dataclass
class CategoryState:
    """Stores budget and spending information for a single category."""
    budget: int = 0
    spent: int = 0
    extra_added: int = 0
    extra_add_times: int = 0


@dataclass
class UserState:
    """Stores all persistent state related to a user."""
    user_id: str
    categories: Dict[str, CategoryState] = field(default_factory=dict)

    total_points: int = 0
    level: int = 1
    theme: str = "Basic"

    negative_limit: int = -1_000_000  # -10 lakh
    overspend_strikes: int = 0

    last_login: Optional[datetime] = None


@dataclass
class AgentMessage:
    """Message used for agent-to-agent communication."""
    sender: str
    target: str
    type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ==========================
# Base Agent
# ==========================

class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, users: Dict[str, UserState]):
        self.name = name
        self.users = users

    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle an incoming message and optionally return a follow-up message."""
        raise NotImplementedError("Subclasses must implement handle()")

    def get_user(self, user_id: str) -> UserState:
        """Retrieve or create the UserState for a given user."""
        if user_id not in self.users:
            self.users[user_id] = UserState(user_id=user_id)
        return self.users[user_id]


# ==========================
# Budget Agent
# ==========================

class BudgetAgent(BaseAgent):
    """Handles monthly budget setup and reset."""

    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type != "set_budget":
            return None

        user_id = message.payload["user_id"]
        budgets = message.payload["budgets"]  # dict: category -> amount

        user = self.get_user(user_id)
        for cat, amount in budgets.items():
            if cat not in user.categories:
                user.categories[cat] = CategoryState()
            state = user.categories[cat]
            state.budget = amount
            state.spent = 0
            state.extra_added = 0
            state.extra_add_times = 0

        log(f"[BudgetAgent] Budget set for user={user_id}: {budgets}")
        return None


# ==========================
# Reward Agent
# ==========================

class RewardAgent(BaseAgent):
    """Handles positive reward points for different user actions."""

    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        user_id = message.payload.get("user_id")
        if not user_id:
            return None

        user = self.get_user(user_id)

        if message.type == "payment_success":
            self._add_points(user, 50, "payment")
        elif message.type == "daily_login":
            user.last_login = datetime.utcnow()
            self._add_points(user, 2, "daily_login")
        elif message.type == "budget_check":
            self._add_points(user, 10, "budget_check")
        elif message.type == "weekly_report":
            self._add_points(user, 100, "weekly_report")
        elif message.type == "savings_bonus":
            bonus = min(int(message.payload.get("points", 0)), 500)
            self._add_points(user, bonus, "savings_bonus")
        elif message.type == "challenge_completed":
            reward = min(int(message.payload.get("points", 0)), 500)
            self._add_points(user, reward, "challenge_completed")
        elif message.type == "referral":
            self._add_points(user, 500, "referral")

        return AgentMessage(
            sender=self.name,
            target="LevelAgent",
            type="update_level",
            payload={"user_id": user_id},
        )

    def _add_points(self, user: UserState, points: int, reason: str) -> None:
        """Add reward points to a user with per-event cap."""
        if points > 500:
            points = 500
        user.total_points += points
        log(f"[RewardAgent] Added +{points} points for {reason}. Total={user.total_points}")


# ==========================
# Penalty Agent
# ==========================

class PenaltyAgent(BaseAgent):
    """Applies penalties when user adds extra money above budget."""

    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type != "extra_money_added":
            return None

        user_id = message.payload["user_id"]
        category = message.payload["category"]
        extra_amount = int(message.payload["extra_amount"])

        user = self.get_user(user_id)
        cat_state = user.categories.setdefault(category, CategoryState())

        cat_state.extra_added += extra_amount
        cat_state.extra_add_times += 1

        penalty = self._compute_penalty(extra_amount, cat_state.extra_add_times)
        self._apply_penalty(user, penalty)

        return AgentMessage(
            sender=self.name,
            target="ThemeAgent",
            type="update_theme",
            payload={"user_id": user_id},
        )

    def _compute_penalty(self, extra_amount: int, times_added_same_category: int) -> int:
        """
        Rule:
        - Every 200 units of extra_amount => 100 penalty points
        - Base penalty capped at 500
        - Multiplied by number of times extra was added in this category, then capped again at 500
        """
        chunks = extra_amount // 200
        base_penalty = chunks * 100
        if base_penalty > 500:
            base_penalty = 500

        penalty = base_penalty * max(1, times_added_same_category)
        if penalty > 500:
            penalty = 500

        return penalty

    def _apply_penalty(self, user: UserState, penalty: int) -> None:
        """Apply penalty while respecting negative limit."""
        new_points = user.total_points - penalty
        if new_points < user.negative_limit:
            log("[PenaltyAgent] ERROR: Negative limit reached. Overspend blocked.")
            return

        user.total_points = new_points
        log(f"[PenaltyAgent] Applied penalty -{penalty}. Total={user.total_points}")


# ==========================
# Level Agent
# ==========================

class LevelAgent(BaseAgent):
    """Maps total points to levels using a predefined table."""

    def __init__(self, name: str, users: Dict[str, UserState]):
        super().__init__(name, users)
        # Example curve (you can tune to your 30-level 2 crore design)
        self.level_points = {
            1: 20_000,
            2: 60_000,
            3: 120_000,
            4: 200_000,
            5: 300_000,
            6: 420_000,
            7: 560_000,
            8: 720_000,
            9: 900_000,
            10: 1_100_000,
            30: 20_000_000,
        }

    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type != "update_level":
            return None

        user_id = message.payload["user_id"]
        user = self.get_user(user_id)

        points = max(user.total_points, 0)
        new_level = 1
        for lvl, req in sorted(self.level_points.items()):
            if points >= req:
                new_level = lvl
            else:
                break

        if new_level != user.level:
            log(f"[LevelAgent] Level changed {user.level} -> {new_level}")
            user.level = new_level

        return AgentMessage(
            sender=self.name,
            target="ThemeAgent",
            type="update_theme",
            payload={"user_id": user_id},
        )


# ==========================
# Theme Agent
# ==========================

class ThemeAgent(BaseAgent):
    """Handles theme upgrade and degradation based on points."""

    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type != "update_theme":
            return None

        user_id = message.payload["user_id"]
        user = self.get_user(user_id)

        p = user.total_points
        old_theme = user.theme

        # Simple threshold-based theme scheme
        if p <= -500_000:
            user.theme = "Basic"
        elif p <= -200_000:
            user.theme = "Blue"
        elif p <= -100_000:
            user.theme = "Green"
        elif p <= -50_000:
            user.theme = "Gold"
        elif p > 5_000_000:
            user.theme = "Royal"
        elif p > 1_000_000:
            user.theme = "Premium"

        if user.theme != old_theme:
            log(f"[ThemeAgent] Theme changed {old_theme} -> {user.theme}")

        return None


# ==========================
# Transaction Agent
# ==========================

# IMPORTANT:
# predict_category(description: str) must be defined in your notebook
# by your ML cell. Here we just call it.

class TransactionAgent(BaseAgent):
    """
    Handles payment transactions.
    Uses ML model to predict category from description if category is not provided.
    """

    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type != "transaction":
            return None

        user_id = message.payload["user_id"]
        amount = int(message.payload["amount"])

        category = message.payload.get("category")
        description = message.payload.get("description", "").lower()

        # Use ML + fallback if category is missing
        if category is None:
            try:
                from __main__ import predict_category  # works inside notebook
                predicted = predict_category(description)
                if predicted is not None:
                    category = predicted
                else:
                    raise Exception("ML returned None")
            except Exception as e:
                print(f"[TransactionAgent] ML failed or missing: {e}")

                # Simple rule-based fallback mapping
                if "swiggy" in description or "zomato" in description or "kfc" in description:
                    category = "food"
                elif "uber" in description or "ola" in description or "rapido" in description:
                    category = "travel"
                elif "amazon" in description or "flipkart" in description or "myntra" in description:
                    category = "shopping"
                elif "bill" in description or "recharge" in description or "electricity" in description:
                    category = "bills"
                elif "movie" in description or "ticket" in description or "pvr" in description:
                    category = "entertainment"
                else:
                    category = "other"

        user = self.get_user(user_id)
        cat_state = user.categories.setdefault(category, CategoryState())

        allowed_limit = cat_state.budget + cat_state.extra_added
        new_spent = cat_state.spent + amount

        if new_spent <= allowed_limit:
            cat_state.spent = new_spent
            log(f"[TransactionAgent] Payment accepted. user={user_id}, cat={category}, spent={new_spent}")
            return AgentMessage(
                sender=self.name,
                target="RewardAgent",
                type="payment_success",
                payload={"user_id": user_id},
            )
        else:
            log(f"[TransactionAgent] Payment would exceed budget. user={user_id}, cat={category}")
            return None


# ==========================
# Summary Agent
# ==========================

class SummaryAgent(BaseAgent):
    """Generates simple monthly summary for a user."""

    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type != "generate_summary":
            return None

        user_id = message.payload["user_id"]
        user = self.get_user(user_id)

        summary = {
            "user_id": user.user_id,
            "points": user.total_points,
            "level": user.level,
            "theme": user.theme,
            "categories": {},
        }

        for cat, state in user.categories.items():
            summary["categories"][cat] = {
                "budget": state.budget,
                "spent": state.spent,
                "extra_added": state.extra_added,
            }

        log(f"[SummaryAgent] Summary generated for user={user_id}: {summary}")
        return None


# ==========================
# Multi-Agent System Orchestrator
# ==========================

class MultiAgentSystem:
    """Main orchestrator that routes messages between agents."""

    def __init__(self):
        self.users: Dict[str, UserState] = {}

        self.agents: Dict[str, BaseAgent] = {
            "BudgetAgent": BudgetAgent("BudgetAgent", self.users),
            "TransactionAgent": TransactionAgent("TransactionAgent", self.users),
            "RewardAgent": RewardAgent("RewardAgent", self.users),
            "PenaltyAgent": PenaltyAgent("PenaltyAgent", self.users),
            "LevelAgent": LevelAgent("LevelAgent", self.users),
            "ThemeAgent": ThemeAgent("ThemeAgent", self.users),
            "SummaryAgent": SummaryAgent("SummaryAgent", self.users),
        }

    def route(self, message: AgentMessage) -> None:
        """Route a message to its target agent and follow any chained messages."""
        current = message
        while current is not None:
            target_name = current.target
            agent = self.agents.get(target_name)
            if agent is None:
                log(f"[System] Unknown agent: {target_name}")
                return
            next_msg = agent.handle(current)
            current = next_msg

    # Convenience helpers
    def set_budget(self, user_id: str, budgets: Dict[str, int]) -> None:
        msg = AgentMessage(
            sender="External",
            target="BudgetAgent",
            type="set_budget",
            payload={"user_id": user_id, "budgets": budgets},
        )
        self.route(msg)

    def add_transaction(
        self,
        user_id: str,
        amount: int,
        category: Optional[str] = None,
        description: str = "",
    ) -> None:
        msg = AgentMessage(
            sender="External",
            target="TransactionAgent",
            type="transaction",
            payload={
                "user_id": user_id,
                "category": category,
                "amount": amount,
                "description": description,
            },
        )
        self.route(msg)

    def add_extra_money(self, user_id: str, category: str, extra_amount: int) -> None:
        msg = AgentMessage(
            sender="External",
            target="PenaltyAgent",
            type="extra_money_added",
            payload={"user_id": user_id, "category": category, "extra_amount": extra_amount},
        )
        self.route(msg)

    def daily_login(self, user_id: str) -> None:
        msg = AgentMessage(
            sender="External",
            target="RewardAgent",
            type="daily_login",
            payload={"user_id": user_id},
        )
        self.route(msg)

    def budget_check(self, user_id: str) -> None:
        msg = AgentMessage(
            sender="External",
            target="RewardAgent",
            type="budget_check",
            payload={"user_id": user_id},
        )
        self.route(msg)

    def generate_summary(self, user_id: str) -> None:
        msg = AgentMessage(
            sender="External",
            target="SummaryAgent",
            type="generate_summary",
            payload={"user_id": user_id},
        )
        self.route(msg)


# ==========================
# Simple demo (for local testing)
# ==========================

if __name__ == "__main__":
    system = MultiAgentSystem()

    system.set_budget("user1", {"food": 5000, "travel": 3000, "shopping": 4000, "bills": 3500})

    system.daily_login("user1")
    system.budget_check("user1")

    # This call expects predict_category() to exist when run from a notebook.
    system.add_transaction(
        user_id="user1",
        amount=300,
        category=None,
        description="Swiggy chicken biryani order"
    )

    system.add_transaction(
        user_id="user1",
        amount=250,
        category=None,
        description="Uber ride to office"
    )

    system.generate_summary("user1")






import pandas as pd

df = pd.read_csv("/kaggle/input/transactions/transactions.csv")
df.head()



# Helper functions to use the trained model

import joblib
from pathlib import Path

VECTOR_PATH = Path("/kaggle/working/txn_vectorizer.pkl")
MODEL_PATH = Path("/kaggle/working/txn_model.pkl")

_vectorizer = None
_model = None

def load_txn_model():
    """Load the vectorizer and model once."""
    global _vectorizer, _model
    if _vectorizer is None or _model is None:
        if not VECTOR_PATH.exists() or not MODEL_PATH.exists():
            raise FileNotFoundError("Model files not found. Train model first.")
        _vectorizer = joblib.load(VECTOR_PATH)
        _model = joblib.load(MODEL_PATH)

def predict_category(description: str) -> str:
    """Predict the category from a text description."""
    load_txn_model()
    X = _vectorizer.transform([description])
    return _model.predict(X)[0]



system = MultiAgentSystem()

# 1) Budget set
system.set_budget("user1", {
    "food": 5000,
    "travel": 3000,
    "shopping": 4000,
    "bills": 3500
})

# 2) Daily login + budget check
system.daily_login("user1")
system.budget_check("user1")

# 3) Transaction using ONLY description → ML should auto-predict
system.add_transaction(
    user_id="user1",
    category=None,
    amount=300,
    description="Swiggy chicken biryani order"
)

system.add_transaction(
    user_id="user1",
    category=None,
    amount=250,
    description="Uber ride to office"
)

system.add_transaction(
    user_id="user1",
    category=None,
    amount=700,
    description="Amazon headphone purchase"
)

# 4) Summary
system.generate_summary("user1")



print(predict_category("Swiggy pizza order"))
print(predict_category("Flipkart shoe purchase"))
print(predict_category("Jio postpaid bill"))
print(predict_category("Ola cab to airport"))



system = MultiAgentSystem()

system.set_budget("u1", {"food": 2000, "travel": 1000})

system.add_transaction(
    user_id="u1",
    amount=100,
    category=None,
    description="Swiggy biryani"
)

system.add_transaction(
    user_id="u1",
    amount=150,
    category=None,
    description="Uber ride to college"
)

system.generate_summary("u1")



class FinancialAdvisorAgent(BaseAgent):
    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type == "financial_advice":
            user_id = message.payload["user_id"]
            user = self.get_user(user_id)
            
            # Analyze spending pattern
            analysis = self._analyze_spending(user)
            
            # Get LLM advice
            advice = self._call_gemini_api(analysis)
            
            return AgentMessage(
                sender=self.name,
                target="UserInterface",
                type="show_advice", 
                payload={"user_id": user_id, "advice": advice}
            )


import sqlite3
import json

class DatabaseManager:
    def save_user_state(self, user: UserState):
        # Save to SQLite/JSON file
        pass
        
    def load_user_state(self, user_id: str) -> UserState:
        # Load from database
        pass


from flask import Flask, request, jsonify

app = Flask(__name__)
system = MultiAgentSystem()

@app.route('/api/transaction', methods=['POST'])
def add_transaction():
    data = request.json
    system.add_transaction(
        user_id=data['user_id'],
        amount=data['amount'],
        description=data['description']
    )
    return jsonify({"status": "success"})

