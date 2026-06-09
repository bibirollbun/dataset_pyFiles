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


!pip install sportypy


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from keras.utils import to_categorical


# Load NFL Big Data Bowl 2025 datasets
project_dir = '/kaggle/input/nfl-big-data-bowl-2025'
games = pd.read_csv(f'{project_dir}/games.csv')
plays = pd.read_csv(f'{project_dir}/plays.csv')
players = pd.read_csv(f'{project_dir}/players.csv')
player_play = pd.read_csv(f'{project_dir}/player_play.csv')
tracking_week1 = pd.read_csv(f'{project_dir}/tracking_week_1.csv')
tracking_week2 = pd.read_csv(f'{project_dir}/tracking_week_2.csv')
tracking_week3 = pd.read_csv(f'{project_dir}/tracking_week_3.csv')
tracking_week4 = pd.read_csv(f'{project_dir}/tracking_week_4.csv')
tracking_week5 = pd.read_csv(f'{project_dir}/tracking_week_5.csv')
tracking_week6 = pd.read_csv(f'{project_dir}/tracking_week_6.csv')
tracking_week7 = pd.read_csv(f'{project_dir}/tracking_week_7.csv')
tracking_week8 = pd.read_csv(f'{project_dir}/tracking_week_8.csv')
tracking_week9 = pd.read_csv(f'{project_dir}/tracking_week_9.csv')


games.head()


plays.columns


correlations = ["passLength", "timeToThrow","expectedPointsAdded"]

plays[correlations].corr()


plays['timeToThrow'].mean()


player_play.columns


player_play[player_play['hadRushAttempt']==1].groupby(['teamAbbr']).rushingYards.mean().sort_values(ascending = False).head()


players.columns


# avg_pass_stats_player_named = pd.concat([avg_pass_stats_player, players], axis = 1, join='inner')
avg_pass_stats_player_named = pd.merge(left = avg_pass_stats_player, right = players, how='inner',on='nflId') # get just the offense
avg_pass_stats_player_named[:10]


play_formation = plays[['gameId','playId','offenseFormation','expectedPointsAdded']].groupby(['gameId','playId','offenseFormation']).count()
play_formation = play_formation.reset_index(level=['gameId','playId','offenseFormation'])
# play_formation = play_formation.reset_index()
play_formation


rush_data_players = player_play[player_play['hadRushAttempt']==1].groupby(['nflId']).rushingYards.mean().sort_values(ascending = False).rename_axis('nflId').reset_index()
rush_data_players = pd.merge(left=rush_data_players,right=players,how='left')
rush_data_players


# Draw field

# def draw_field(ax):
#     # Field boundaries
#     ax.set_xlim(0, 120)
#     ax.set_ylim(0, 53.3)
#     ax.set_facecolor('mediumseagreen')  # field color

#     # End zones
#     ax.add_patch(plt.Rectangle((0, 0), 10, 53.3, color='lightblue', alpha=0.3, zorder=0))     # left end zone
#     ax.add_patch(plt.Rectangle((110, 0), 10, 53.3, color='lightblue', alpha=0.3, zorder=0))   # right end zone

#     # Yard lines every 10 yards
#     for x in range(10, 111, 10):
#         ax.plot([x, x], [0, 53.3], color='white', linewidth=1)

#     # Hash marks (simplified version)
#     for x in range(11, 110):
#         ax.plot([x, x], [23.35, 23.95], color='white', linewidth=0.5)  # Bottom hash
#         ax.plot([x, x], [29.35, 29.95], color='white', linewidth=0.5)  # Top hash

#     # Turn off axis
#     ax.axis('off')

#     return ax


from sportypy.surfaces.football import NFLField

# Create the NHL rink
nfl = NFLField()
nfl.draw()


import pandas as pd
project_dir = '/kaggle/input/nfl-big-data-bowl-2025'
# Load data (make sure you've uploaded it to your Kaggle notebook)
tracking = pd.read_csv(f'{project_dir}/tracking_week_1.csv')

# Filter for a specific play (example: playId = 64, gameId = 2022091200)
play_data = tracking[(tracking['gameId'] == 2022091200) & (tracking['playId'] == 64)]
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up figure
fig, ax = plt.subplots(figsize=(12, 6))
from sportypy.surfaces.football import NFLField

# Create the NHL rink
nfl = NFLField()
nfl.draw(ax = ax)
# ax.set_xlim(0, 120)
# ax.set_ylim(0, 53.3)
ax.set_title('NFL Play Animation - Play 64')

# Separate football and players for coloring
players = play_data[play_data['club'].isin(['home', 'away'])]
football = play_data[play_data['club'] == 'football']

# Initialize scatter plot
home_scatter = ax.scatter([], [], c='blue', label='Home')
away_scatter = ax.scatter([], [], c='red', label='Away')
football_scatter = ax.scatter([], [], c='brown', s=30, label='Football')

# Optional: display jersey numbers
texts = []

# Frame list
frames = sorted(play_data['frameId'].unique())

def init():
    empty = np.empty((0, 2))  # Correct shape: 0 rows, 2 columns (x, y)
    home_scatter.set_offsets(empty)
    away_scatter.set_offsets(empty)
    football_scatter.set_offsets(empty)
    return home_scatter, away_scatter, football_scatter

def update(frame):
    global texts
    for txt in texts:
        txt.remove()
    texts = []

    frame_data = play_data[play_data['frameId'] == frame]

    home = frame_data[frame_data['club'] == 'home']
    away = frame_data[frame_data['club'] == 'away']
    ball = frame_data[frame_data['club'] == 'football']

    home_scatter.set_offsets(home[['x', 'y']])
    away_scatter.set_offsets(away[['x', 'y']])
    football_scatter.set_offsets(ball[['x', 'y']])

    # Add jersey numbers
    for _, row in frame_data.iterrows():
        if pd.notnull(row['jerseyNumber']):
            txt = ax.text(row['x'], row['y'] + 0.5, str(int(row['jerseyNumber'])), fontsize=6, ha='center')
            texts.append(txt)

    ax.set_title(f'NFL Play Animation - Frame {frame}')
    return home_scatter, away_scatter, football_scatter, *texts

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=100)

# To display in Kaggle (in notebook)
from IPython.display import HTML
HTML(ani.to_jshtml())




tracking.head()


def distance(x2, x1, y2,y1):
    return math.sqrt(Math.pow(x2-x1));

