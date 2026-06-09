


import pandas as pd
from datetime import datetime

# A short example dataset for illustration
MARKET_DATA = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=10, freq="D"),
    "market": ["EEX_power"] * 10,
    "price_eur_mwh": [92.3, 94.1, 93.5, 95.0, 96.2, 97.8, 96.9, 98.1, 99.4, 100.2],
})


def load_market_data(market: str = "EEX_power") -> pd.DataFrame:
    """Return a filtered copy of the small sample dataset."""
    df = MARKET_DATA.copy()
    return df[df["market"] == market].reset_index(drop=True)


def compute_risk_metrics(df: pd.DataFrame) -> dict:
    """Compute a few descriptive statistics from the price series."""
    df = df.sort_values("date")

    start_price = df["price_eur_mwh"].iloc[0]
    end_price = df["price_eur_mwh"].iloc[-1]
    pct_change = (end_price - start_price) / start_price * 100

    daily_returns = df["price_eur_mwh"].pct_change().dropna()
    volatility = float(daily_returns.std())

    max_drawdown = float((df["price_eur_mwh"].cummax() - df["price_eur_mwh"]).max())

    return {
        "start_price": float(start_price),
        "end_price": float(end_price),
        "pct_change": float(pct_change),
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "n_points": len(df),
    }


# quick check
compute_risk_metrics(load_market_data())


class MemoryStore:
    """
    Minimal memory structure attached to a user_id.
    Only keeps a list of recent questions and responses.
    """

    def __init__(self):
        self._store = {}

    def get_user_state(self, user_id: str) -> dict:
        if user_id not in self._store:
            self._store[user_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "history": [],
            }
        return self._store[user_id]

    def add_entry(self, user_id: str, query: str, response: str):
        state = self.get_user_state(user_id)
        state["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response,
        })


memory = MemoryStore()


def fake_llm(prompt: str) -> str:
    """
    This acts as a stand-in for a real language model.
    It simply returns a formatted echo of the prompt so the logic can run offline.
    """
    return (
        "FAKE MODEL OUTPUT (offline test)\n\n"
        "Prompt excerpt:\n"
        + prompt[:500]
        + "\n\n[In a full environment, this output would come from an actual LLM.]"
    )


def build_prompt(user_query: str,
                 market_df: pd.DataFrame,
                 metrics: dict,
                 user_state: dict) -> str:
    """Create the prompt that would be passed to the model in a real setup."""

    recent = user_state["history"][-3:]
    if recent:
        history_text = "\n\n".join(
            f"Q: {h['query']}\nA: {h['response'][:200]}" for h in recent
        )
    else:
        history_text = "No earlier conversation."

    prompt = f"""
EMIPRA – Energy Market Intelligence & Portfolio Risk Assistant

User question:
{user_query}

Recent conversation:
{history_text}

Market overview (last {metrics['n_points']} days):
- Start: {metrics['start_price']:.2f} EUR/MWh
- End: {metrics['end_price']:.2f} EUR/MWh
- Change: {metrics['pct_change']:.2f}%
- Volatility (daily std): {metrics['volatility']:.4f}
- Max drawdown: {metrics['max_drawdown']:.2f} EUR/MWh

Write a short explanation (3–6 sentences) covering:
1. The general trend and volatility.
2. The main portfolio risks.
3. One possible risk mitigation or hedging idea.
"""
    return prompt.strip()


def run_agent(user_query: str, user_id: str = "demo_user") -> str:
    """Runs the basic agent pipeline with the offline stub."""
    df = load_market_data()
    metrics = compute_risk_metrics(df)
    user_state = memory.get_user_state(user_id)

    prompt = build_prompt(user_query, df, metrics, user_state)
    reply = fake_llm(prompt)

    memory.add_entry(user_id, user_query, reply)
    return reply


# quick example
run_agent("What are the key risks right now?")[:800]


TEST_QUERIES = [
    "Give me a short market summary.",
    "How volatile has the market been?",
    "What should a cautious portfolio manager watch for?",
]


def judge_response(query: str, answer: str) -> dict:
    """
    Placeholder evaluation.
    In a real setup this would call a stronger model to review the answer.
    """
    return {
        "query": query,
        "answer_preview": answer[:180],
        "correctness": 4,
        "clarity": 4,
        "risk_focus": 5,
        "note": "Local stub. Real scoring would come from a model acting as a judge."
    }


def run_evaluation():
    results = []
    for q in TEST_QUERIES:
        ans = run_agent(q, user_id="evaluation_user")
        results.append(judge_response(q, ans))
    return results


run_evaluation()

