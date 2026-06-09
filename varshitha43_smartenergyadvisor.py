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


from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import concurrent.futures
import uuid
import random

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class ApplianceUsage:
    appliance: str
    power_watts: float
    daily_hours: float
    daily_kwh: float
    daily_cost_inr: float

@dataclass
class Recommendation:
    appliance: str
    message: str
    potential_kwh_saved: float
    potential_cost_saved_inr: float

@dataclass
class SavingsScenario:
    reduction_pct: float
    original_monthly_cost_inr: float
    new_monthly_cost_inr: float
    monthly_savings_inr: float

# -----------------------------
# In-memory session service
# (Sessions & Memory concept)
# -----------------------------
class InMemorySessionService:
    """Simple session store keyed by session_id."""
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            "user_id": user_id,
            "created_at": time.time(),
            "metadata": metadata or {},
            "state": {}
        }
        return sid

    def get(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

    def set_state(self, session_id: str, key: str, value: Any):
        if session_id not in self.sessions:
            raise KeyError("session not found")
        self.sessions[session_id]["state"][key] = value

    def get_state(self, session_id: str, key: str, default=None):
        return self.sessions.get(session_id, {}).get("state", {}).get(key, default)

# -----------------------------
# Mock LLM Agent
# (Agent powered by an LLM - mocked)
# -----------------------------
class MockLLMAgent:
    """A mocked LLM agent for explanations & tips.
    In a real system this would call Gemini or another LLM.
    """
    def summarize_usage(self, appliance: str, kwh: float, cost: float) -> str:
        return (f"{appliance} uses about {kwh:.2f} kWh/day (~₹{cost:.2f}/day). "
                f"This contributes significantly to your electricity bill.")

    def generate_energy_tip(self, appliance: str, usage_level: str) -> str:
        if usage_level == "high":
            return (f"For {appliance}: reduce runtime, use timer/thermostat, "
                    "and prefer energy-efficient settings or models.")
        if usage_level == "medium":
            return (f"For {appliance}: schedule usage in off-peak hours and avoid unnecessary standby.")
        return (f"For {appliance}: usage is already efficient; focus on maintaining good habits.")

    def explain_savings(self, reduction_pct: float) -> str:
        return (f"By reducing usage by about {int(reduction_pct*100)}%, you can lower your monthly "
                "electricity cost while maintaining comfort by focusing on the top-consuming devices.")

# -----------------------------
# Energy data and analysis agents
# -----------------------------
class UsageSimulationAgent:
    """Simulates or ingests energy usage data for a household."""
    def __init__(self, price_per_kwh_inr: float = 8.0):
        self.price_per_kwh_inr = price_per_kwh_inr

    def simulate_usage(self) -> List[ApplianceUsage]:
        # Example appliance list with approximate powers
        appliances = [
            ("Fan", 70),
            ("Light", 15),
            ("Fridge", 150),
            ("AC", 1500),
            ("Washing Machine", 500),
            ("TV", 120),
            ("Laptop", 60)
        ]
        usage_data: List[ApplianceUsage] = []
        for name, power in appliances:
            # Random usage with some bias
            if name == "Fridge":
                hours = 20 + random.uniform(2, 4)
            elif name == "AC":
                hours = 4 + random.uniform(1, 3)
            else:
                hours = 1 + random.uniform(1, 8)

            daily_kwh = (power * hours) / 1000.0
            daily_cost = daily_kwh * self.price_per_kwh_inr
            usage_data.append(
                ApplianceUsage(
                    appliance=name,
                    power_watts=power,
                    daily_hours=hours,
                    daily_kwh=daily_kwh,
                    daily_cost_inr=daily_cost
                )
            )
        return usage_data

class ParallelAnalysisAgent:
    """Runs per-appliance analysis in parallel (Parallel Agents concept)."""
    def __init__(self, llm_agent: MockLLMAgent, price_per_kwh_inr: float = 8.0):
        self.llm = llm_agent
        self.price_per_kwh_inr = price_per_kwh_inr

    def analyze_appliance(self, usage: ApplianceUsage) -> Dict[str, Any]:
        # Classify usage level
        if usage.daily_kwh > 3:
            level = "high"
        elif usage.daily_kwh > 1:
            level = "medium"
        else:
            level = "low"

        summary = self.llm.summarize_usage(usage.appliance, usage.daily_kwh, usage.daily_cost_inr)
        tip = self.llm.generate_energy_tip(usage.appliance, level)

        return {
            "appliance": usage.appliance,
            "level": level,
            "summary": summary,
            "tip": tip,
            "daily_kwh": usage.daily_kwh,
            "daily_cost_inr": usage.daily_cost_inr
        }

    def run_parallel(self, usage_list: List[ApplianceUsage]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            fut_to_app = {ex.submit(self.analyze_appliance, u): u for u in usage_list}
            for fut in concurrent.futures.as_completed(fut_to_app):
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"error": str(e)}
                results.append(res)
        return results

class SavingsSimulationAgent:
    """Simulates monthly savings from reducing consumption."""
    def __init__(self, price_per_kwh_inr: float = 8.0):
        self.price_per_kwh_inr = price_per_kwh_inr

    def simulate(self, usage_list: List[ApplianceUsage], reduction_pct: float = 0.10) -> SavingsScenario:
        original_monthly = 0.0
        new_monthly = 0.0
        for u in usage_list:
            orig_daily = u.daily_kwh
            new_daily = orig_daily * (1 - reduction_pct)
            original_monthly += orig_daily * self.price_per_kwh_inr * 30
            new_monthly += new_daily * self.price_per_kwh_inr * 30

        savings = original_monthly - new_monthly
        return SavingsScenario(
            reduction_pct=reduction_pct,
            original_monthly_cost_inr=original_monthly,
            new_monthly_cost_inr=new_monthly,
            monthly_savings_inr=savings
        )

class RecommendationAgent:
    """Builds structured Recommendation objects based on analysis."""
    def __init__(self, price_per_kwh_inr: float = 8.0):
        self.price_per_kwh_inr = price_per_kwh_inr

    def recommend(self, usage_list: List[ApplianceUsage]) -> List[Recommendation]:
        recs: List[Recommendation] = []
        for u in usage_list:
            if u.daily_kwh > 3:
                reduction_pct = 0.20
                msg = (f"{u.appliance}: Very high usage. Try reducing runtime by 20%, "
                       f"use thermostat/timer, and check for inefficient settings.")
            elif u.daily_kwh > 1:
                reduction_pct = 0.10
                msg = (f"{u.appliance}: Moderate usage. Shift some usage to off-peak hours and avoid standby.")
            else:
                reduction_pct = 0.05
                msg = (f"{u.appliance}: Low usage. You are already efficient; maintain current habits.")
            saved_kwh = u.daily_kwh * reduction_pct * 30   # monthly
            saved_cost = saved_kwh * self.price_per_kwh_inr
            recs.append(
                Recommendation(
                    appliance=u.appliance,
                    message=msg,
                    potential_kwh_saved=saved_kwh,
                    potential_cost_saved_inr=saved_cost
                )
            )
        return recs

# -----------------------------
# Coordinator demonstrating:
# - Multi-agent system
# - Parallel agents
# - Sessions & Memory
# - Simple observability
# -----------------------------
class EnergyAdvisorCoordinator:
    def __init__(self, session_service: InMemorySessionService, price_per_kwh_inr: float = 8.0):
        self.sessions = session_service
        self.llm = MockLLMAgent()
        self.usage_agent = UsageSimulationAgent(price_per_kwh_inr)
        self.parallel_agent = ParallelAnalysisAgent(self.llm, price_per_kwh_inr)
        self.savings_agent = SavingsSimulationAgent(price_per_kwh_inr)
        self.recommendation_agent = RecommendationAgent(price_per_kwh_inr)
        self.price_per_kwh_inr = price_per_kwh_inr

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.sessions.create_session(user_id, metadata)

    def generate_plan(self, session_id: str, reduction_pct: float = 0.10) -> Dict[str, Any]:
        print("[LOG] Simulating usage data...")
        usage_data = self.usage_agent.simulate_usage()
        self.sessions.set_state(session_id, "usage_data", usage_data)

        print("[LOG] Running parallel per-appliance analysis...")
        analysis_parallel = self.parallel_agent.run_parallel(usage_data)

        print("[LOG] Running savings simulation...")
        savings_scenario = self.savings_agent.simulate(usage_data, reduction_pct)

        print("[LOG] Generating structured recommendations...")
        recs = self.recommendation_agent.recommend(usage_data)

        print("[LOG] Building summary & storing in session...")
        profile = {
            "usage_data": usage_data,
            "analysis_parallel": analysis_parallel,
            "savings_scenario": savings_scenario,
            "recommendations": recs
        }
        self.sessions.set_state(session_id, "last_plan", profile)

        return profile

    def pretty_print_plan(self, plan: Dict[str, Any]):
        usage_data: List[ApplianceUsage] = plan["usage_data"]
        analysis = plan["analysis_parallel"]
        savings: SavingsScenario = plan["savings_scenario"]
        recs: List[Recommendation] = plan["recommendations"]

        print("\n--- Household Energy Summary ---")
        for u in usage_data:
            print(f"- {u.appliance}: {u.daily_kwh:.2f} kWh/day (~₹{u.daily_cost_inr:.2f}/day)")

        print("\n--- Per-appliance Insights (from MockLLMAgent) ---")
        for a in analysis:
            if "error" in a:
                continue
            print(f"\n{a['appliance']}:")
            print(f"  Level: {a['level']}")
            print(f"  Summary: {a['summary']}")
            print(f"  Tip: {a['tip']}")

        print("\n--- Recommendations & Potential Savings ---")
        for r in recs:
            print(f"\n{r.appliance}:")
            print(f"  Recommendation: {r.message}")
            print(f"  Potential monthly saving: {r.potential_kwh_saved:.2f} kWh "
                  f"(~₹{r.potential_cost_saved_inr:.2f})")

        print("\n--- Scenario: Overall Monthly Savings ---")
        explanation = self.llm.explain_savings(savings.reduction_pct)
        print(explanation)
        print(f"- Original monthly cost: ₹{savings.original_monthly_cost_inr:.2f}")
        print(f"- New monthly cost:      ₹{savings.new_monthly_cost_inr:.2f}")
        print(f"- Estimated savings:     ₹{savings.monthly_savings_inr:.2f} per month")

# -----------------------------
# Demo run for your project
# -----------------------------
if __name__ == "__main__":
    session_service = InMemorySessionService()
    advisor = EnergyAdvisorCoordinator(session_service, price_per_kwh_inr=8.0)

    sid = advisor.create_session(
        user_id="household_001",
        metadata={"region": "IN", "notes": "Demo session for Kaggle Capstone"}
    )
    print("Created session id:", sid)

    plan = advisor.generate_plan(session_id=sid, reduction_pct=0.10)
    advisor.pretty_print_plan(plan)

    print("\nSession stored state keys:", list(session_service.get(sid).get("state", {}).keys()))


