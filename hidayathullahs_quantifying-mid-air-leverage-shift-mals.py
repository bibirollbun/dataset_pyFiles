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
import numpy as np
import matplotlib.pyplot as plt

from math import atan2



tracking = pd.read_csv(
    "/kaggle/input/nfl-big-data-bowl-2026-analytics/"
    "114239_nfl_competition_files_published_analytics_final/"
    "train/input_2023_w17.csv"
)

tracking.head()



# Select one play to analyze
sample_play = tracking.iloc[0]

game_id = sample_play["game_id"]
play_id = sample_play["play_id"]

play_df = tracking[
    (tracking["game_id"] == game_id) &
    (tracking["play_id"] == play_id)
]

play_df.head()



receiver = play_df[play_df["player_to_predict"] == True].iloc[0]
receiver_id = receiver["nfl_id"]

defender = play_df[play_df["player_role"] == "Defensive Coverage"].iloc[0]
defender_id = defender["nfl_id"]

receiver_id, defender_id



ball_x = receiver["ball_land_x"]
ball_y = receiver["ball_land_y"]

ball_x, ball_y



def angle_to_target(x, y, tx, ty):
    return atan2(ty - y, tx - x)

leverage_scores = []

for frame in sorted(play_df["frame_id"].unique()):
    frame_df = play_df[play_df["frame_id"] == frame]

    try:
        r = frame_df[frame_df["nfl_id"] == receiver_id].iloc[0]
        d = frame_df[frame_df["nfl_id"] == defender_id].iloc[0]
    except:
        continue

    dr = np.sqrt((ball_x - r.x)**2 + (ball_y - r.y)**2)
    dd = np.sqrt((ball_x - d.x)**2 + (ball_y - d.y)**2)

    angle_r = angle_to_target(r.x, r.y, ball_x, ball_y)
    angle_d = angle_to_target(d.x, d.y, ball_x, ball_y)

    leverage = (dd - dr) * np.cos(angle_r - angle_d)
    leverage_scores.append(leverage)

leverage_scores[:5], leverage_scores[-5:]



if len(leverage_scores) > 1:
    mals = leverage_scores[-1] - leverage_scores[0]
else:
    mals = None

mals



defenders = play_df[play_df["player_role"] == "Defensive Coverage"].copy()



defenders["dist_to_ball"] = np.sqrt(
    (defenders["x"] - ball_x)**2 + (defenders["y"] - ball_y)**2
)



primary_defender = defenders.sort_values("dist_to_ball").iloc[0]
defender_id = primary_defender["nfl_id"]
defender_id



leverage_scores = []



for frame in sorted(play_df["frame_id"].unique()):
    frame_df = play_df[play_df["frame_id"] == frame]

    try:
        r = frame_df[frame_df["nfl_id"] == receiver_id].iloc[0]
        d = frame_df[frame_df["nfl_id"] == defender_id].iloc[0]
    except:
        continue

    dr = np.sqrt((ball_x - r.x)**2 + (ball_y - r.y)**2)
    dd = np.sqrt((ball_x - d.x)**2 + (ball_y - d.y)**2)

    angle_r = angle_to_target(r.x, r.y, ball_x, ball_y)
    angle_d = angle_to_target(d.x, d.y, ball_x, ball_y)

    leverage = (dd - dr) * np.cos(angle_r - angle_d)
    leverage_scores.append(leverage)



leverage_scores[:5], leverage_scores[-5:]



mals = leverage_scores[-1] - leverage_scores[0]
mals






plt.plot(leverage_scores)
plt.xlabel("Frame")
plt.ylabel("Leverage")
plt.title("Leverage Over Ball Flight")
plt.show()



results = []



for (gid, pid), play_df in tracking.groupby(["game_id", "play_id"]):

    receivers = play_df[play_df["player_to_predict"] == True]
    defenders = play_df[play_df["player_role"] == "Defensive Coverage"]

    if receivers.empty or defenders.empty:
        continue

    receiver = receivers.iloc[0]
    receiver_id = receiver["nfl_id"]

    ball_x = receiver["ball_land_x"]
    ball_y = receiver["ball_land_y"]

    defenders = defenders.copy()
    defenders["dist_to_ball"] = np.sqrt(
        (defenders["x"] - ball_x)**2 + (defenders["y"] - ball_y)**2
    )

    defender_id = defenders.sort_values("dist_to_ball").iloc[0]["nfl_id"]

    leverage_scores = []

    for frame in sorted(play_df["frame_id"].unique()):
        frame_df = play_df[play_df["frame_id"] == frame]

        try:
            r = frame_df[frame_df["nfl_id"] == receiver_id].iloc[0]
            d = frame_df[frame_df["nfl_id"] == defender_id].iloc[0]
        except:
            continue

        dr = np.sqrt((ball_x - r.x)**2 + (ball_y - r.y)**2)
        dd = np.sqrt((ball_x - d.x)**2 + (ball_y - d.y)**2)

        angle_r = angle_to_target(r.x, r.y, ball_x, ball_y)
        angle_d = angle_to_target(d.x, d.y, ball_x, ball_y)

        leverage = (dd - dr) * np.cos(angle_r - angle_d)
        leverage_scores.append(leverage)

    if len(leverage_scores) < 2:
        continue

    mals = leverage_scores[-1] - leverage_scores[0]

    results.append({
        "game_id": gid,
        "play_id": pid,
        "mals": mals
    })



results_df = pd.DataFrame(results)
results_df.head()



results_df["mals"].describe()



plt.hist(results_df["mals"], bins=50)
plt.xlabel("MALS")
plt.ylabel("Count")
plt.title("Distribution of Mid-Air Leverage Shift")
plt.show()



top_positive = results_df.sort_values("mals", ascending=False).head(1)
top_negative = results_df.sort_values("mals").head(1)

top_positive, top_negative








