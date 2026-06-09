"""
CAPSTONE PROJECT
Topic: How to use Gemini (Simulation Project)
Project: AI-based Incentive Recommendation using Gemini Agents (Simulated)
Author: Tanisha
"""

import pandas as pd
import numpy as np
import json
import datetime


# -----------------------------
# 1. LOAD SAMPLE SALES DATA
# -----------------------------

def load_data():
    data = {
        "agent_id": ["A001", "A002", "A003", "A004"],
        "agent_name": ["Anita", "Rahul", "Sneha", "Karan"],
        "region": ["West", "North", "East", "South"],
        "total_sales": [120000, 60000, 95000, 45000],
        "quota": [100000, 90000, 90000, 80000],
        "closed_deals": [12, 7, 9, 5],
        "last_active": ["2025-10-18", "2025-09-30", "2025-10-10", "2025-10-05"]
    }
    df = pd.DataFrame(data)
    df["last_active"] = pd.to_datetime(df["last_active"])
    return df


# -----------------------------
# 2. ANALYSIS AGENT
# -----------------------------
def analysis_agent(df):
    df["quota_achievement"] = (df["total_sales"] / df["quota"]) * 100
    df["avg_deal_size"] = df["total_sales"] / df["closed_deals"]
    df["recency_days"] = (datetime.datetime.now() - df["last_active"]).dt.days
    return df


# -----------------------------
# 3. GEMINI-SIMULATED INCENTIVE AGENT
# -----------------------------
def gemini_incentive_agent(df):
    """
    This simulates what Gemini *would* generate.
    No API call. No internet.
    """
    # Condition-based incentive logic
    tiers = []

    for _, row in df.iterrows():
        if row["quota_achievement"] >= 110:
            tiers.append({
                "agent_name": row["agent_name"],
                "tier": "Platinum",
                "bonus": "₹25,000 + Gift Voucher",
                "reason": "Exceeded quota significantly."
            })
        elif row["quota_achievement"] >= 100:
            tiers.append({
                "agent_name": row["agent_name"],
                "tier": "Gold",
                "bonus": "₹15,000",
                "reason": "Achieved quota."
            })
        elif row["quota_achievement"] >= 80:
            tiers.append({
                "agent_name": row["agent_name"],
                "tier": "Silver",
                "bonus": "₹8,000",
                "reason": "Average performance."
            })
        else:
            tiers.append({
                "agent_name": row["agent_name"],
                "tier": "Bronze",
                "bonus": "₹3,000",
                "reason": "Needs improvement."
            })

    # Simulated Gemini response
    final_plan = {
        "incentive_cycle": "Q1 - 2025",
        "tiers_generated_by": "Gemini Simulation Model",
        "result": tiers
    }

    return final_plan


# -----------------------------
# 4. GEMINI-SIMULATED MESSAGE AGENT
# -----------------------------
def gemini_message_agent(agent_name, tier, bonus):
    """
    Simulated message creation as if Gemini generated it.
    """
    return f"""
Subject: Congratulations {agent_name}!

Dear {agent_name},

Based on your performance this quarter, you have been placed in the **{tier} Tier**.
You are eligible for the incentive reward of **{bonus}**.

Keep up the excellent work!

Regards,  
AI Incentive System (Gemini Simulation)
"""


# -----------------------------
# 5. MAIN FUNCTION
# -----------------------------
def main():
    print("Loading data...\n")
    df = load_data()
    print(df, "\n")

    print("Running analysis agent...\n")
    analysis = analysis_agent(df)
    print(analysis, "\n")

    print("Generating incentive plan using Gemini Simulation...\n")
    plan = gemini_incentive_agent(analysis)
    print(json.dumps(plan, indent=4), "\n")

    print("Generating personalized messages...\n")
    messages = []
    for result in plan["result"]:
        msg = gemini_message_agent(
            result["agent_name"],
            result["tier"],
            result["bonus"]
        )
        messages.append(msg)
        print(msg)

    # Save output files
    with open("incentive_plan_simulated.json", "w") as f:
        json.dump(plan, f, indent=4)

    print("Saved incentive_plan_simulated.json")


# Run project
main()


