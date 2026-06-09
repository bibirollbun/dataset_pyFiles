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


DATA_PATH = "/kaggle/input/drug-drug-interactions/db_drug_interactions.csv"
df = pd.read_csv(DATA_PATH)


df.head()


df.columns


# CONFIG
DRUG_A_COL = "Drug 1"
DRUG_B_COL = "Drug 2"
DESCRIPTION_COL = "Interaction Description"
# dataset does not include severity ratings
SEVERITY_COL = None  


# Normalization Helper
def normalize_drug_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


from collections import defaultdict

# Normalize columns
df["_drug_a_norm"] = df[DRUG_A_COL].apply(normalize_drug_name)
df["_drug_b_norm"] = df[DRUG_B_COL].apply(normalize_drug_name)

# Remove invalid entries
df = df[(df["_drug_a_norm"] != "") & (df["_drug_b_norm"] != "")].copy()

# Build lookup: key = frozenset({drug1, drug2})
interaction_lookup = defaultdict(list)

for _, row in df.iterrows():
    a = row["_drug_a_norm"]
    b = row["_drug_b_norm"]
    key = frozenset([a, b])

    record = {
        "drug_a": row[DRUG_A_COL],
        "drug_b": row[DRUG_B_COL],
        "description": row[DESCRIPTION_COL]
    }
    
    interaction_lookup[key].append(record)

len(interaction_lookup)


def tool_check_pair(drug1: str, drug2: str):
    """
    Return interaction records for a pair of drugs.
    """
    d1 = normalize_drug_name(drug1)
    d2 = normalize_drug_name(drug2)
    key = frozenset([d1, d2])
    return interaction_lookup.get(key, [])


def tool_find_interactions_for_list(drug_list):
    """
    Check all unique pairs within a list of drugs.
    """
    normalized = [normalize_drug_name(d) for d in drug_list]
    results = []
    
    n = len(normalized)
    for i in range(n):
        for j in range(i+1, n):
            d1 = normalized[i]
            d2 = normalized[j]
            key = frozenset([d1, d2])
            interactions = interaction_lookup.get(key, [])
            
            if interactions:
                results.append({
                    "drug1_input": drug_list[i],
                    "drug2_input": drug_list[j],
                    "drug1_norm": d1,
                    "drug2_norm": d2,
                    "interactions": interactions
                })
    return results


def tool_format_report(drug_list):
    """
    Create a human-friendly report summarizing interactions.
    """
    interactions = tool_find_interactions_for_list(drug_list)

    if not interactions:
        return (
            "### No interactions found in this dataset.\n"
            "**Important:** Absence of data is NOT confirmation of safety.\n"
            "Consult a pharmacist or doctor for medical decisions."
        )

    lines = []
    lines.append("## Drug–Drug Interaction Report")
    lines.append("### Drugs entered: " + ", ".join(drug_list))
    lines.append("")

    for entry in interactions:
        d1 = entry["drug1_input"]
        d2 = entry["drug2_input"]

        lines.append(f"### Interaction: **{d1}** × **{d2}**")

        for rec in entry["interactions"]:
            desc = rec.get("description") or "No description provided."
            lines.append(f"- **Detail**: {desc}")

        lines.append("")

    lines.append("---")
    lines.append(
        "*This is an educational tool only. It uses a dataset and may not cover all interactions.*"
    )

    return "\n".join(lines)


test_list = ["Aspirin", "Warfarin", "Ibuprofen"]
print(tool_format_report(test_list))


def planner(user_input):
    """
    Simple rule-based planner deciding what the agent should do.
    Returns a plan dictionary.
    """
    user_input = user_input.strip()

    if not user_input:
        return {"action": "ask_clarification", "message": "Please enter at least one drug name."}

    # Split user input into drug names
    if "," in user_input:
        drug_list = [d.strip() for d in user_input.split(",") if d.strip()]
    else:
        # Single drug? Not useful. Ask for more.
        return {
            "action": "ask_clarification",
            "message": "Please enter two or more drugs separated by commas."
        }

    if len(drug_list) < 2:
        return {"action": "ask_clarification", "message": "You must provide at least two drugs."}

    # Main action: run interaction analysis
    return {
        "action": "check_interactions",
        "drugs": drug_list
    }


def agent(user_input):
    """
    Agent loop: plan -> act -> observe -> produce final answer.
    """

    # Step 1: Planning
    plan = planner(user_input)

    if plan["action"] == "ask_clarification":
        return plan["message"]

    if plan["action"] == "check_interactions":
        drugs = plan["drugs"]

        # Step 2: Use tool to check interactions
        report = tool_format_report(drugs)

        # Step 3: Return final answer (no LLM needed)
        return report

    # Default fallback
    return "I'm not sure how to help with that."


def run_demo():
    print("Offline Drug Interaction Agent")
    print("----------------------------------")
    user_input = input("Enter drug names (comma-separated): ")
    print("\n### Agent Output:\n")
    print(agent(user_input))

# Uncomment to run in notebook:
# run_demo()


agent("Aspirin, Warfarin, Ibuprofen")


def evaluator(user_input, agent_output):
    """
    Evaluator Agent:
    Checks consistency between the tool results and the final agent output.
    Returns a dictionary with evaluation results.
    """
    # Parse user input into drug list
    if "," in user_input:
        drug_list = [d.strip() for d in user_input.split(",") if d.strip()]
    else:
        return {"status": "invalid", "message": "Input must contain at least two drugs."}

    # Recompute ground-truth interactions using tools
    tool_results = tool_find_interactions_for_list(drug_list)

    # 1. Check if agent output is empty or malformed
    if not isinstance(agent_output, str) or len(agent_output.strip()) == 0:
        return {
            "status": "fail",
            "reason": "Agent output is empty or not a string.",
            "expected_interactions": tool_results
        }

    # 2. Verify that each interaction from tool layer appears in agent_output
    missing_interactions = []

    for entry in tool_results:
        for rec in entry["interactions"]:
            desc = rec.get("description", "").strip()
            if desc and desc not in agent_output:
                missing_interactions.append(desc)

    # 3. Decide pass/fail status
    if len(tool_results) == 0:
        # No interactions expected -> output is valid as long as it says "no interactions"
        if "No interactions" in agent_output:
            return {"status": "pass", "details": "No interactions expected or found."}
        else:
            return {
                "status": "warning",
                "reason": "No interactions exist, but agent output does not mention this.",
                "agent_output": agent_output
            }

    if missing_interactions:
        return {
            "status": "fail",
            "reason": "Agent output missing interaction descriptions.",
            "missing": missing_interactions,
            "agent_output": agent_output
        }

    # If none missing then pass
    return {
        "status": "pass",
        "details": "Agent output correctly includes all known interactions.",
        "agent_output": agent_output
    }


user_query = "Aspirin, Warfarin, Ibuprofen"
agent_output = agent(user_query)

eval_result = evaluator(user_query, agent_output)

print("### Agent Output:\n")
print(agent_output)
print("\n### Evaluation Result:\n")
print(eval_result)




