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


## importing necessary libraries
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

import plotly.express as go 

import warnings 
warnings.filterwarnings('ignore')

# Show all columns when printing a DataFrame
pd.set_option("display.max_columns", None)

# Show all rows
pd.set_option("display.max_rows", None)


## reading the input_2023_w01 data
input_2023_w01_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv")
input_2023_w01_df.head()


## reading the output_2023_w01 data
output_2023_w01_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w01.csv")
output_2023_w01_df.head()


## 2023_w01 data
big_bowl_2023_w01_df = pd.merge(
        input_2023_w01_df,
        output_2023_w01_df,
        left_on = ['game_id', 'play_id', 'nfl_id', 'frame_id'],
        right_on = ['game_id', 'play_id', 'nfl_id', 'frame_id'],
        how = 'left'
)


big_bowl_2023_w01_df.head(10)


big_bowl_2023_w01_df.game_id.unique(), big_bowl_2023_w01_df.game_id.nunique()


## play id corresponding to every game id
# big_bowl_2023_w01_df.groupby(['game_id'])['play_id'].count()
big_bowl_2023_w01_df.groupby("game_id")["play_id"].apply(list)
big_bowl_2023_w01_df.groupby("game_id")["play_id"].nunique().reset_index(name="distinct_play_count")


big_bowl_2023_w01_df.nfl_id.nunique()


## Visualize the Game
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

%matplotlib notebook

# -------------------------------
# Load one play's input + output
# -------------------------------
# Example: Week 1 input/output
input_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv")
output_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w01.csv")

# Choose a single game and play
game_id = input_df['game_id'].iloc[0]
play_id = input_df['play_id'].iloc[0]

play_input = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]
play_output = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]

# Combine input & output frames
play_df = pd.concat([
        play_input[['game_id','play_id','nfl_id','frame_id','x','y','player_role','player_side']],
        play_output[['game_id','play_id','nfl_id','frame_id','x','y']]
])

play_df[['player_role','player_side']] = play_df[['player_role','player_side']].fillna("")

# Reset frame index (so it runs sequentially)
play_df = play_df.sort_values(by=['frame_id', 'nfl_id']).reset_index(drop=True)

# -------------------------------
# Visualization setup
# -------------------------------
fig, ax = plt.subplots(figsize=(12,6))

# Draw football field (120x53.3 yards)
ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
ax.set_xlabel("Yards (Length of Field)")
ax.set_ylabel("Yards (Width of Field)")
ax.set_title(f"Game {game_id} | Play {play_id}")

# Differentiate offense vs defense
offense_ids = play_input[play_input['player_side'] == "Offense"]['nfl_id'].unique()
defense_ids = play_input[play_input['player_side'] == "Defense"]['nfl_id'].unique()

offense_scatter, = ax.plot([], [], 'o', color='blue', label="Offense")
defense_scatter, = ax.plot([], [], 'o', color='red', label="Defense")
ball_scatter, = ax.plot([], [], 'o', color='brown', markersize=10, label="Ball")

ax.legend(loc="upper right")

# -------------------------------
# Animation function
# -------------------------------
def update(frame):
    frame_data = play_df[play_df['frame_id'] == frame]
    
    # Offense
    off = frame_data[frame_data['nfl_id'].isin(offense_ids)]
    offense_scatter.set_data(off['x'], off['y'])
    
    # Defense
    deff = frame_data[frame_data['nfl_id'].isin(defense_ids)]
    defense_scatter.set_data(deff['x'], deff['y'])
    
    # Ball (approx: use median of offensive player_to_predict or add later if tracked separately)
    if "player_role" in play_input.columns:
        ball = frame_data[frame_data['player_role'] == "Passer"]
        if not ball.empty:
            ball_scatter.set_data(ball['x'], ball['y'])
    
    return offense_scatter, defense_scatter, ball_scatter

# -------------------------------
# Run Animation
# -------------------------------
ani = animation.FuncAnimation(fig, update, 
                              frames=play_df['frame_id'].max(), 
                              interval=200, blit=True)

# plt.show()
HTML(ani.to_jshtml())





