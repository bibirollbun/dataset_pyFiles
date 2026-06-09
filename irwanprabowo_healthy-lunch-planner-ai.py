# ============================================
#  IMPORTS AND GEMINI 3.0 SETUP
# ============================================
import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Plot styling
plt.style.use("seaborn-v0_8")
sns.set_theme()

# ============================================
# Gemini 3.0 Flash Setup (silent fallback)
# ============================================
USE_GEMINI = False
client = None

try:
    from google import genai
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
    else:
        USE_GEMINI = False
except Exception:
    USE_GEMINI = False


def call_gemini(prompt: str) -> str:
    """
    Wrapper for Gemini 3.0 Flash with graceful fallback.

    If Gemini is available and the call succeeds, returns the model output.
    Otherwise returns an empty string so the system can use a local fallback.
    """
    if USE_GEMINI and client:
        try:
            response = client.models.generate_content(
                model="gemini-3.0-flash",
                contents=prompt
            )
            if hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception:
            pass
    return ""



# ============================================
#  LOAD MEALS DATASET
# ============================================
# Adjust this path to match your Kaggle dataset if needed
DATA_PATH = "/kaggle/input/meals-large-30/meals_large_30.csv"

df_meals = pd.read_csv(DATA_PATH)
df_meals.head()



# ============================================
#  MEMORY SYSTEM
# ============================================
class InMemorySessionService:
    """Simple in-memory session store."""
    def __init__(self):
        self.sessions = {}

    def get(self, sid):
        return self.sessions.setdefault(sid, {})

    def set(self, sid, data):
        self.sessions[sid] = data


class MemoryBank:
    """
    Long-term memory stored as JSON on disk.
    Used to persist the last run's state and results.
    """
    def __init__(self, path: str = "lunch_memory.json"):
        self.path = path

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def save(self, data: dict):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)


# ============================================
#  TOOLS (CSV, ratings, computations)
# ============================================
def csv_tool():
    """
    Returns the meals as a list of dicts from the loaded dataframe.
    """
    return df_meals.to_dict(orient="records")


def rating_tool(meal_name: str) -> float:
    """
    Mock rating tool (simulated external API).
    Returns a rating between 3.5 and 5.0.
    """
    return round(random.uniform(3.5, 5.0), 1)


def code_execution_tool(meals):
    """
    Compute average price, average calories, and select
    the top 5 recommended meals based on calories and price.
    """
    if not meals:
        return {
            "avg_price": 0.0,
            "avg_calories": 0.0,
            "ranked_meals": []
        }

    avg_price = float(np.mean([m["price"] for m in meals]))
    avg_cal = float(np.mean([m["calories"] for m in meals]))

    ranked = sorted(
        meals,
        key=lambda x: (x["calories"], x["price"])
    )[:5]

    return {
        "avg_price": round(avg_price, 2),
        "avg_calories": round(avg_cal, 2),
        "ranked_meals": ranked
    }



# ================================
#  AGENTS (Planner, Researcher, Coder, Critic)
# ================================
class Planner:
    """
    Planner Agent - interprets user goals and creates a plan.
    Uses Gemini 3.0 Flash when available.
    """
    def run(self, goal: dict) -> str:
        prompt = (
            "You are the Planner Agent in a multi-agent AI system.\n"
            f"The user wants a lunch under {goal['calories']} calories "
            f"and under {goal['budget']} USD.\n"
            "Write a brief plan (3â€“5 bullet-like sentences) describing how "
            "to choose suitable meals from a dataset by:\n"
            "- filtering by calories and price\n"
            "- computing summary statistics\n"
            "- ranking and selecting the top recommendations.\n"
        )

        plan = call_gemini(prompt)
        if plan:
            return plan

        # Local fallback
        return (
            f"Plan: filter meals under {goal['calories']} calories and under "
            f"{goal['budget']} USD, compute averages, rank meals, and pick "
            "the top 5 options."
        )


class Researcher:
    """
    Researcher Agent - loads and enriches meal data.
    """
    def run(self):
        meals = csv_tool()
        enriched = []
        for item in meals:
            row = dict(item)  # copy
            row["rating"] = rating_tool(row["meal"])
            enriched.append(row)
        return enriched


class Coder:
    """
    Coder Agent - performs numeric analysis via the code_execution_tool.
    """
    def run(self, meals):
        return code_execution_tool(meals)


class Critic:
    """
    Critic Agent - evaluates whether the recommendations meet the goal.
    """
    def run(self, summary: dict, goal: dict) -> str:
        avg_price = summary.get("avg_price", 0.0)
        avg_cal = summary.get("avg_calories", 0.0)

        meets_price = avg_price <= goal["budget"]
        meets_cal = avg_cal <= goal["calories"]

        if meets_price and meets_cal:
            return "Recommendation meets both budget and calorie goals."
        if meets_price and not meets_cal:
            return "Average price is in range, but calories exceed the target."
        if meets_cal and not meets_price:
            return "Calories are in range, but average price exceeds the budget."
        return "Recommendations do not fully match the budget or calorie goals."



# ============================================
# Filtering and Visualization Helpers
# ============================================
def filter_meals(meals, goal):
    """
    Filter meals that satisfy both calorie and price constraints.
    """
    filtered = [
        m for m in meals
        if m["calories"] <= goal["calories"] and m["price"] <= goal["budget"]
    ]
    return {
        "filtered_meals": filtered,
        "count": len(filtered),
        "total": len(meals)
    }


def visualize_filtered_meals(meals):
    """
    Generate basic charts for filtered meals.
    """
    df = pd.DataFrame(meals)
    if df.empty:
        return

    # Calories per meal
    plt.figure(figsize=(10, 5))
    plt.bar(df["meal"], df["calories"])
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("Calories")
    plt.title("Calories per Meal (Filtered)")
    plt.tight_layout()
    plt.show()

    # Price vs Calories
    plt.figure(figsize=(7, 5))
    plt.scatter(df["price"], df["calories"])
    plt.xlabel("Price")
    plt.ylabel("Calories")
    plt.title("Price vs Calories (Filtered)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Protein vs Carbs (if columns available)
    required_cols = {"protein", "carbs"}
    if required_cols.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        plt.scatter(df["protein"], df["carbs"])
        plt.xlabel("Protein")
        plt.ylabel("Carbs")
        plt.title("Protein vs Carbs (Filtered)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def radar_chart_top5(summary):
    """
    Radar charts for the top 5 recommended meals.
    """
    df = pd.DataFrame(summary["ranked_meals"])
    if df.empty:
        return

    attributes = ["calories", "protein", "fat", "carbs"]
    num_vars = len(attributes)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for _, row in df.iterrows():
        stats = [row[a] for a in attributes]
        stats += stats[:1]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, stats, linewidth=2)
        ax.fill(angles, stats, alpha=0.25)
        plt.xticks(angles[:-1], attributes)
        plt.title(row["meal"])
        plt.tight_layout()
        plt.show()


def correlation_heatmap(meals):
    """
    Correlation heatmap for key numeric columns.
    """
    df = pd.DataFrame(meals)
    cols = [c for c in ["calories", "protein", "fat", "carbs", "price"] if c in df.columns]
    if len(cols) < 2:
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()


def top5_chart(summary):
    """
    Bar chart for the top 5 recommended meals.
    """
    df = pd.DataFrame(summary["ranked_meals"])
    if df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(df["meal"], df["calories"], color="green")
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("Calories")
    plt.title("Top 5 Recommended Meals (Lower calories preferred)")
    plt.tight_layout()
    plt.show()


def top5_table(summary):
    """
    Styled table of the top 5 recommended meals.
    """
    df = pd.DataFrame(summary["ranked_meals"])
    if df.empty:
        return df  # safe empty DataFrame

    cols = [c for c in ["meal", "calories", "protein", "fat", "carbs", "price", "rating"] if c in df.columns]
    subset = df[cols]

    styled = subset.style.set_caption("Top 5 Recommended Meals").set_table_styles([
        {
            "selector": "caption",
            "props": [
                ("font-size", "16px"),
                ("font-weight", "bold"),
                ("text-align", "left"),
            ],
        }
    ])
    return styled



# ============================================
# Orchestrator Agent
# ============================================
class Orchestrator:
    """
    Orchestrator Agent - coordinates all agents,
    triggers tools, manages memory, and runs visualizations.
    """
    def __init__(self):
        self.session = InMemorySessionService()
        self.memory = MemoryBank()
        self.state = self.memory.load() or {}

    def run(self, goal: dict):
        # Instantiate agents
        planner = Planner()
        researcher = Researcher()
        coder = Coder()
        critic = Critic()

        # 1. Planner: interpret goal and generate plan
        plan = planner.run(goal)

        # 2. Researcher: load and enrich data
        meals = researcher.run()

        # 3. Filter by goal
        filter_result = filter_meals(meals, goal)
        filtered_meals = filter_result["filtered_meals"]

        # Visualize filtered meals
        visualize_filtered_meals(filtered_meals)

        # Choose which set to analyze (fallback to all meals if filter is empty)
        analysis_input = filtered_meals if filtered_meals else meals

        # 4. Coder: compute statistics and ranking
        summary = coder.run(analysis_input)

        # 5. Visual analytics
        correlation_heatmap(meals)
        top5_chart(summary)
        radar_chart_top5(summary)

        # 6. Top-5 table
        styled = top5_table(summary)
        display(styled)

        # 7. Critic: evaluate alignment with goals
        evaluation = critic.run(summary, goal)

        # Save state to long-term memory
        self.state = {
            "goal": goal,
            "plan": plan,
            "summary": summary,
            "evaluation": evaluation,
        }
        self.memory.save(self.state)

        # Return structured summary
        return {
            "goal": goal,
            "plan": plan,
            "filter_result": filter_result,
            "summary": summary,
            "evaluation": evaluation,
        }



# ============================================
# Run the System
# ============================================
goal = {
    "budget": 8.0,   # maximum budget in USD
    "calories": 600  # maximum calories
}

orchestrator = Orchestrator()
results = orchestrator.run(goal)
results




