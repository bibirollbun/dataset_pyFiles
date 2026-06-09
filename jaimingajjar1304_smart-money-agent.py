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


import pandas as pd

CSV_PATH = "/kaggle/input/smart-money-transactions-dataset/transactions.csv"

df = pd.read_csv(CSV_PATH)
df.head()



# Gemini SDK for Kaggle
!pip install -q google-generativeai rich



# Cell: Gemini / LLM setup with Kaggle Secrets

import os
import json
import google.generativeai as genai


# If you're using Kaggle Secrets:
# 1. Go to: Add-ons â�œ Secrets â�œ Add new secret
# 2. Name: GEMINI_API_KEY
# 3. Value: your real Gemini API key

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
except Exception as e:
    # Fallback if secrets are not available (e.g. running locally)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

print("Gemini key found:", bool(GEMINI_API_KEY))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("âš ï¸� Warning: No GEMINI_API_KEY set. LLM agent will be skipped / mocked.")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import time
import json



class TransactionLoaderAgent:
    def __init__(self, trace_log=None):
        self.trace = trace_log if trace_log is not None else []

    def load_transactions(self, csv_path: str) -> pd.DataFrame:
        t0 = time.time()
        self.trace.append({"step": "load_start", "t": t0, "csv_path": csv_path})

        df = pd.read_csv(csv_path)

        if "trans_date_trans_time" in df.columns:
            df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

        df = df.rename(columns={
            "trans_date_trans_time": "date",
            "merchant": "merchant",
            "amt": "amount",
            "category": "raw_category",
            "transaction_category": "transaction_category"
        })

        df = df.drop_duplicates()
        df["transaction_category"] = df["transaction_category"].fillna("uncategorized")
        df["raw_category"] = df["raw_category"].fillna("unknown")

        if "normalized_category" not in df.columns:
            df["normalized_category"] = df["transaction_category"]

        t1 = time.time()
        self.trace.append({"step": "load_done", "t": t1, "n_rows": len(df)})
        return df



class CategorizationAgent:
    def __init__(self, trace_log=None):
        self.trace = trace_log if trace_log is not None else []
        self.keyword_map = {
            "gas": "Gas & Transport",
            "transport": "Gas & Transport",
            "shopping": "Shopping",
            "misc_net": "Shopping",
            "kids_pets": "Kids & Pets",
            "mobile transaction": "Digital / Mobile",
            "debit": "General Spend",
            "credit": "General Spend"
        }

    def _normalize_one(self, raw_cat: str, trans_cat: str) -> str:
        text = f"{str(raw_cat)} {str(trans_cat)}".lower()
        for k, v in self.keyword_map.items():
            if k in text: 
                return v
        return "Other"

    def categorize(self, df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.time()
        self.trace.append({"step": "categorize_start", "t": t0})

        df = df.copy()
        df["normalized_category"] = df.apply(
            lambda row: self._normalize_one(row["raw_category"], row["transaction_category"]),
            axis=1
        )

        t1 = time.time()
        ncat = df["normalized_category"].nunique()
        self.trace.append({"step": "categorize_done", "t": t1, "n_categories": ncat})
        return df



class InsightsAgent:
    def __init__(self, trace_log=None):
        self.trace = trace_log if trace_log is not None else []

    def compute_insights(self, df: pd.DataFrame) -> dict:
        t0 = time.time()
        self.trace.append({"step": "insights_start", "t": t0})

        insights = {
            "total_spend": df["amount"].sum(),
            "spend_by_category": df.groupby("normalized_category")["amount"].sum().sort_values(ascending=False).to_dict(),
        }

        df2 = df.copy()
        df2["month"] = df2["date"].dt.to_period("M").astype(str)
        insights["spend_by_month"] = df2.groupby("month")["amount"].sum().to_dict()

        t1 = time.time()
        self.trace.append({
            "step": "insights_done",
            "t": t1,
            "total_spend": insights["total_spend"],
            "n_months": len(insights["spend_by_month"])
        })
        return insights



class ReportAgent:
    def __init__(self, monthly_budget=None, trace_log=None):
        self.monthly_budget = monthly_budget
        self.trace = trace_log if trace_log is not None else []

    def generate_report(self, insights: dict) -> str:
        t0 = time.time()
        self.trace.append({"step": "report_start", "t": t0})

        total = insights["total_spend"]
        by_cat = insights["spend_by_category"]
        by_month = insights["spend_by_month"]

        report = []
        report.append("ğŸ“Š Smart Money Agent â€“ Summary Report")
        report.append(f"- Total spend in dataset: ${total:,.2f}")
        report.append("- Top categories:")

        for k, v in list(by_cat.items())[:3]:
            report.append(f"  â€¢ {k}: ${v:,.2f}")

        if self.monthly_budget:
            report.append(f"\n- Budget: ${self.monthly_budget:,.2f}")
            for m, amt in by_month.items():
                status = "â�Œ Over Budget" if amt > self.monthly_budget else "âœ… OK"
                report.append(f"  â€¢ {m}: ${amt:,.2f} ({status})")

        t1 = time.time()
        self.trace.append({"step": "report_done", "t": t1})

        return "\n".join(report)



# Cell: Session memory + evaluator + LLM advisor agent

class SessionMemory:
    """
    Very simple in-memory session store.
    Demonstrates:
      - Sessions & state management
      - Long-term-ish memory across runs in this notebook
    """
    def __init__(self):
        # session_id -> {"insights": ..., "report": ...}
        self._store = {}

    def save(self, session_id: str, insights: dict, report: str):
        self._store[session_id] = {
            "insights": insights,
            "report": report,
        }

    def load(self, session_id: str):
        return self._store.get(session_id, None)


class EvaluatorAgent:
    """
    Simple automatic evaluation of agent quality.
    Checks whether too many transactions fall into 'Other'
    (i.e., categorization is poor).
    Demonstrates: Agent evaluation.
    """
    def __init__(self, max_other_ratio: float = 0.2):
        self.max_other_ratio = max_other_ratio

    def evaluate(self, df: pd.DataFrame, insights: dict) -> dict:
        n_tx = len(df)
        n_other = (df["normalized_category"] == "Other").sum()
        other_ratio = float(n_other) / n_tx if n_tx > 0 else 0.0

        result = {
            "n_transactions": int(n_tx),
            "n_other": int(n_other),
            "other_ratio": round(other_ratio, 4),
            "total_spend": float(insights["total_spend"]),
            "passes_other_ratio_check": other_ratio <= self.max_other_ratio,
        }
        return result


class LlmAdvisorAgent:
    """
    LLM-powered advisor:
    - Takes numeric insights + rule-based report
    - Produces a friendly, natural language advice summary
    """
    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        self.model_name = model_name
        self.llm_available = bool(GEMINI_API_KEY)

        if self.llm_available:
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None

    def advise(self, insights: dict, baseline_report: str, monthly_budget: float | None) -> str:
        if not self.llm_available:
            return (
                "LLM advisor is disabled (no GEMINI_API_KEY). "
                "Baseline report:\n\n" + baseline_report
            )

        prompt = f"""
You are a helpful personal finance assistant.

Here are the numeric insights:
{json.dumps(insights, indent=2)}

Here is a baseline report:
\"\"\"{baseline_report}\"\"\"

Give a short summary + 4 practical improvement tips.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"LLM advisor call failed, falling back to baseline.\n(Error: {str(e)})\n\n{baseline_report}"



# Cell: OrchestratorAgent (updated to use memory, evaluator, and LLM)

class OrchestratorAgent:
    """
    High-level 'Concierge' / Orchestrator agent.
    Demonstrates:
      - Sequential multi-agent pipeline
      - Tool-style LLM usage
      - Observability (trace + metrics)
      - Session & memory integration
      - Agent evaluation
    """
    def __init__(self, monthly_budget: float | None = None, session_memory: SessionMemory | None = None):
        self.trace = []

        # Core sub-agents
        self.loader = TransactionLoaderAgent(self.trace)
        self.categorizer = CategorizationAgent(self.trace)
        self.insights_agent = InsightsAgent(self.trace)
        self.report_agent = ReportAgent(monthly_budget, self.trace)

        # New agents
        self.evaluator = EvaluatorAgent(max_other_ratio=0.2)
        self.llm_advisor = LlmAdvisorAgent()
        self.session_memory = session_memory or SessionMemory()

        self.monthly_budget = monthly_budget

    def run_pipeline(self, csv_path: str, session_id: str = "default_session"):
        t0 = time.time()

        # 1. Load & clean
        df = self.loader.load_transactions(csv_path)

        # 2. Categorize
        df = self.categorizer.categorize(df)

        # 3. Compute numeric insights
        insights = self.insights_agent.compute_insights(df)

        # 4. Rule-based report (deterministic)
        baseline_report = self.report_agent.generate_report(insights)

        # 5. Evaluation of categorization quality
        eval_result = self.evaluator.evaluate(df, insights)

        # 6. LLM-powered advisory layer
        llm_advice = self.llm_advisor.advise(
            insights=insights,
            baseline_report=baseline_report,
            monthly_budget=self.monthly_budget,
        )

        # 7. Save into session memory (simple persistence)
        self.session_memory.save(session_id, insights, baseline_report)

        t1 = time.time()
        metrics = {
            "runtime_sec": t1 - t0,
            "n_transactions": len(df),
            "n_categories": df["normalized_category"].nunique(),
        }

        return {
            "df": df,
            "insights": insights,
            "baseline_report": baseline_report,
            "llm_advice": llm_advice,
            "evaluation": eval_result,
            "trace": self.trace,
            "metrics": metrics,
        }



# Cell: Run full pipeline (with LLM, evaluation, memory, observability)

orch = OrchestratorAgent(monthly_budget=2000, session_memory=SessionMemory())

result = orch.run_pipeline(CSV_PATH, session_id="user_001")

df_clean       = result["df"]
insights       = result["insights"]
baseline_report = result["baseline_report"]
llm_advice     = result["llm_advice"]
evaluation     = result["evaluation"]
trace          = result["trace"]
metrics        = result["metrics"]

print("=== CLEANED (HEAD) ===")
display(df_clean.head())

print("\n=== INSIGHTS (NUMERIC) ===")
print(insights)

print("\n=== BASELINE (RULE-BASED) REPORT ===")
print(baseline_report)

print("\n=== LLM ADVISOR MESSAGE ===")
print(llm_advice)

print("\n=== EVALUATION RESULT ===")
print(evaluation)

print("\n=== METRICS (OBSERVABILITY) ===")
print(metrics)

print("\n=== TRACE LOG (STEP-BY-STEP) ===")
for t in trace:
    print(t)



import json

output_path = "/kaggle/working/smart_money_output.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("SMART MONEY AGENT - FINAL OUTPUT\n\n")
    f.write("=== BASELINE REPORT ===\n")
    f.write(baseline_report)
    f.write("\n\n=== LLM ADVICE ===\n")
    f.write(llm_advice)
    f.write("\n\n=== METRICS ===\n")
    f.write(json.dumps(metrics, indent=2))

print("Saved output file to:", output_path)



!ls /kaggle/working


