!pip install nfl_tracks


# Data Preparation
import pandas as pd
import matplotlib.pyplot as plt

from nfl import visuals

tracking_data = pd.read_csv(f'/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w02.csv')
context_data = pd.read_csv(f'/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv', dtype={25: str})

game_id = 2023091400
play_id = 3438

data = tracking_data[(tracking_data['game_id'] == game_id) & (tracking_data['play_id'] == play_id)]


play = visuals.Play(data, game_id, play_id, context_data)


# Plots frame 10 of the play on a standard field
fig, ax = play.plot_snap(frameId=10)
plt.show()


# Plots frame 10 using the advanced relay dashboard
fig, ax = play.plot_snap(frameId=10, relay=True)
plt.show()


# Generates a standard animation of the play
# Use kaggle=True to display it in a notebook
standard_animation = play.animate(kaggle=True)
standard_animation


# Generates a relay dashboard animation of the play
relay_animation = play.animate(relay=True, kaggle=True)
relay_animation

