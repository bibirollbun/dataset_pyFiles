%%time
import pandas as pd
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from xgboost import XGBRegressor
import tensorflow as tf
from tqdm.notebook import tqdm
import os


%%time
DATA_PATH = '/kaggle/input/nfl-big-data-bowl-2025'


%%time

games = pd.read_csv(os.path.join(DATA_PATH, 'games.csv'), low_memory=True)
plays = pd.read_csv(os.path.join(DATA_PATH, 'plays.csv'), low_memory=True)
players = pd.read_csv(os.path.join(DATA_PATH, 'players.csv'), low_memory=True)
player_play = pd.read_csv(os.path.join(DATA_PATH, 'player_play.csv'), low_memory=True)
tracking_week1 = pd.read_csv(os.path.join(DATA_PATH, 'tracking_week_1.csv'), low_memory=True)

# get 100 rows on eaach datasets for reduce memory
games = games.head(300)
plays = plays.head(300)
players = players.head(300)
player_play = player_play.head(300)
tracking_week1 = tracking_week1.head(300)


print(games.info())
games.describe()
games['week'].value_counts().plot(kind='bar', title='Game week Distribution')


print(plays.info())
plays['down'].value_counts().plot(kind='bar', title='Play down Distribution')
sns.boxplot(x='quarter', y='yardsToGo', data=plays)


print(players.info())
sns.histplot(players['height'], kde=True, bins=15)
sns.histplot(players['weight'], kde=True, bins=15)


# # Example for Week 1
print(tracking_week1.info())
tracking_week1.head()


# Check missing values
print(games.isnull().sum())
print(plays.isnull().sum())
print(players.isnull().sum())
print(player_play.isnull().sum())



%%time
merged_data = plays.merge(games, on='gameId').merge(player_play, on='playId')


%%time
tracking_week1['speed_squared'] = tracking_week1['s'] ** 2
tracking_week1['acceleration'] = np.gradient(tracking_week1['s'])
tracking_week1.head


%%time
# Get summary statistics

# Plot a histogram of play durations
plt.hist(tracking_week1['dis'].head(100), bins=50)
plt.title('Play Duration Distribution')
plt.xlabel('Play Duration (seconds)')
plt.ylabel('Frequency')
plt.show()


%%time
# Plot player trajectories
for player_id in tracking_week1['nflId'].unique():
    player_data = tracking_week1[tracking_week1['nflId'] == player_id]
    plt.plot(player_data['x'], player_data['y'], label=player_id)

# Adjust the figure size and DPI
plt.figure(figsize=(10, 5), dpi=72)

plt.title('Player Trajectories')
plt.xlabel('X Position (yards)')
plt.ylabel('Y Position (yards)')
plt.legend()
plt.show()


play_outcomes = tracking_week1['event'].value_counts()
sns.countplot(x='event', data=tracking_week1)
plt.title('Play Event Distribution')
plt.xlabel('Play Event')
plt.ylabel('Frequency')
plt.show()


# upgrade pip
pip install --upgrade plotly


%%time
import plotly.express as px
fig = px.scatter(plays, x='playId', y='gameId')
fig.show()


%%time
from sklearn.preprocessing import LabelEncoder
# Identify non-numeric columns
non_numeric_cols = merged_data.select_dtypes(exclude=['int64', 'float64']).columns
print(non_numeric_cols)

# Convert non-numeric columns to numeric using LabelEncoder
le = LabelEncoder()
for col in non_numeric_cols:
    merged_data[col] = le.fit_transform(merged_data[col])

# Convert non-numeric columns to numeric using one-hot encoding
merged_data = pd.get_dummies(merged_data, columns=non_numeric_cols)

X = merged_data.drop(['playId'], axis=1)
y = merged_data['nflId']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


%%time

model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)


%%time
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f'Mean Absolute Error: {mae}')

