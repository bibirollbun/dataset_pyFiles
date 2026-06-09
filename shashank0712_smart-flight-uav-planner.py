import json
import matplotlib.pyplot as plt
import math
import random
import os

try:
    from openai import OpenAI
    OPENAI = OpenAI()
except:
    OPENAI = None


# ------------------------- LLM Decision Engine -------------------------

def llm_generate_plan(objective, constraints):
    """Uses OpenAI if available, else produces a local fallback."""
    prompt = f"""
You are an aviation mission planning assistant.
Generate a clear UAV mission plan for:

Objective: {objective}
Constraints: {json.dumps(constraints, indent=2)}

Provide:
1. Mission summary
2. Risk assessment
3. Number of waypoints needed
4. Key recommendations
"""

    if OPENAI and os.getenv("OPENAI_API_KEY"):
        resp = OPENAI.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return resp.output_text

    # Fallback (no API key)
    return (
        f"Mission Plan (Offline Mode)\n"
        f"Objective: {objective}\n"
        f"Suggested 6–10 waypoints.\n"
        f"Risks: wind, battery, obstacles.\n"
        f"Recommendation: maintain altitude 60–90m.\n"
    )


# ------------------------- Waypoint Generator -------------------------

def generate_waypoints(n=8):
    """Basic random waypoint generator for demo."""
    points = []
    x, y = 0, 0
    for _ in range(n):
        x += random.randint(20, 60)
        y += random.randint(-30, 30)
        points.append((x, y))
    return points


# ------------------------- Plotting Function -------------------------

def plot_path(waypoints):
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.title("UAV Flight Path")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("uav_path.png")
    plt.close()


# ------------------------- Report Generator -------------------------

def save_report(plan_text, waypoints):
    report = {
        "mission_plan": plan_text,
        "waypoints": waypoints
    }
    with open("mission_report.json", "w") as f:
        json.dump(report, f, indent=2)


# ------------------------- Main -------------------------

def run_project():
    print("=== Intelligent UAV Mission Planner ===\n")

    # Example inputs
    objective = "Aerial mapping of a 2 km area"
    constraints = {
        "battery_minutes": 25,
        "wind_speed": "12 km/h",
        "payload_weight": "300 grams",
        "no_fly_zones": ["Sector B3", "Sector C1"]
    }

    print("Generating mission plan...")
    plan = llm_generate_plan(objective, constraints)
    print("\n--- Mission Plan ---\n")
    print(plan)

    print("\nGenerating waypoints...")
    wps = generate_waypoints(n=8)
    print(wps)

    print("Plotting path...")
    plot_path(wps)

    print("Saving mission report...")
    save_report(plan, wps)

    print("\nDone! Files created:")
    print(" - uav_path.png")
    print(" - mission_report.json")


if __name__ == "__main__":
    run_project()


