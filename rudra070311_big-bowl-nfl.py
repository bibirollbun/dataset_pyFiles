# ============================================================
# NFL BIG DATA BOWL 2026 — COMPLETE ANALYTICS PIPELINE (1 CELL)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. LOAD DATA (OFFICIAL PATH)
# -------------------------------
BASE = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final"

X = pd.read_csv(f"{BASE}/train/input_2023_w01.csv")
y = pd.read_csv(f"{BASE}/train/output_2023_w01.csv")

# -------------------------------
# 2. SELECT ONE REAL PASS PLAY
# -------------------------------
example_play = X.iloc[0]
GAME_ID = example_play["game_id"]
PLAY_ID = example_play["play_id"]

play_data = X[(X.game_id == GAME_ID) & (X.play_id == PLAY_ID)]
play_outcome = y[(y.game_id == GAME_ID) & (y.play_id == PLAY_ID)]

# -------------------------------
# 3. IDENTIFY KEY PLAYER
# -------------------------------
focus_player = play_data[play_data["player_to_predict"] == True].copy()

# Fallback safety
if focus_player.empty:
    focus_player = play_data.copy()

# -------------------------------
# 4. CORE METRIC: BALL ATTACK EFFICIENCY (BAE)
# -------------------------------
# Distance to ball landing point
focus_player["dist_to_ball"] = np.sqrt(
    (focus_player["x"] - focus_player["ball_land_x"])**2 +
    (focus_player["y"] - focus_player["ball_land_y"])**2
)

# Distance change per frame (reaction quality)
focus_player["dist_delta"] = (
    focus_player["dist_to_ball"].shift(1) - focus_player["dist_to_ball"]
).fillna(0)

# Acceleration-weighted closing speed
focus_player["BAE"] = focus_player["dist_delta"] * (1 + focus_player["a"].clip(lower=0))

# Normalize for interpretability
focus_player["BAE_norm"] = (
    focus_player["BAE"] - focus_player["BAE"].mean()
) / (focus_player["BAE"].std() + 1e-6)

# -------------------------------
# 5. AGGREGATE BY ROLE
# -------------------------------
role_summary = (
    play_data
    .assign(dist_to_ball=lambda d: np.sqrt(
        (d.x - d.ball_land_x)**2 + (d.y - d.ball_land_y)**2
    ))
    .groupby("player_role")["dist_to_ball"]
    .mean()
    .sort_values()
)

# -------------------------------
# 6. VISUALIZATION — MOVEMENT MAP
# -------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(
    play_data["x"],
    play_data["y"],
    alpha=0.15,
    label="All Players"
)

plt.plot(
    focus_player["x"],
    focus_player["y"],
    linewidth=3,
    label="Key Player Path"
)

plt.scatter(
    focus_player["ball_land_x"].iloc[0],
    focus_player["ball_land_y"].iloc[0],
    marker="X",
    s=200,
    label="Ball Landing Point"
)

plt.title("Player Movement After Ball Release")
plt.xlabel("Field X")
plt.ylabel("Field Y")
plt.legend()
plt.show()

# -------------------------------
# 7. VISUALIZATION — BAE OVER TIME
# -------------------------------
plt.figure(figsize=(10, 4))
plt.plot(
    focus_player["frame_id"],
    focus_player["BAE_norm"],
    linewidth=2
)
plt.axhline(0)
plt.title("Ball Attack Efficiency (BAE) Over Time")
plt.xlabel("Frame After Throw")
plt.ylabel("Normalized BAE")
plt.show()

# -------------------------------
# 8. VISUALIZATION — ROLE COMPARISON
# -------------------------------
plt.figure(figsize=(10, 5))
role_summary.plot(kind="barh")
plt.title("Average Distance to Ball Landing Point by Role")
plt.xlabel("Distance (yards)")
plt.show()

# -------------------------------
# 9. FOOTBALL INTERPRETATION
# -------------------------------
print("GAME:", GAME_ID, "| PLAY:", PLAY_ID)
print("KEY PLAYER:", focus_player["player_name"].iloc[0])
print("ROLE:", focus_player["player_role"].iloc[0])
print("\nBAE Insight:")
print(
    "Positive BAE spikes indicate decisive movement toward the ball.\n"
    "Flat or negative values suggest hesitation, poor angles, or late reaction."
)

print("\nRole Distance Ranking (Lower = Better Ball Positioning):")
print(role_summary)


