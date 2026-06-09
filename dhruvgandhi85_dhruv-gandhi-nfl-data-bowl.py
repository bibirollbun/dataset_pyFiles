import pandas as pd
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


folder = '/kaggle/input/nfl-big-data-bowl-2025'

games = pd.read_csv(f'{folder}/games.csv')
players = pd.read_csv(f'{folder}/players.csv')
plays = pd.read_csv(f'{folder}/plays.csv')
player_play = pd.read_csv(f'{folder}/player_play.csv')


# Merge player_play with plays
tracked_data = player_play.merge(
    plays, 
    on=['gameId', 'playId'], 
    how='left'
)


# Merge the resulting data with players
tracked_data = tracked_data.merge(
    players, 
    on=['nflId'], 
    how='left'
)


# Merge the resulting data with games
tracked_data = tracked_data.merge(
    games, 
    on='gameId', 
    how='left'
)


# split the data into only 10% of the games to reduce the size of the data
game_IDs = tracked_data['gameId'].unique()
random_gameIDs = random.sample(list(game_IDs), int(len(game_IDs)*0.1))


# colnames = list(pd.read_csv(f'{folder}/tracking_week_1.csv').columns)
tracked_data_list = []

for week in range(1, 10):
    start = time.time()
    tracking_week = f'{folder}/tracking_week_{week}.csv'
    week_data = pd.read_csv(tracking_week)
    week_data['week'] = week
    week_data = week_data[week_data['gameId'].isin(random_gameIDs)]
    week_data = week_data.merge(
        tracked_data, 
        on=['gameId', 'playId', 'nflId'], 
        how='left'
    )
    tracked_data_list.append(week_data)
    end = time.time()
    print(f'Week {week} took {end - start} seconds')

tracking_weekly_data = pd.concat(tracked_data_list, ignore_index=True)


# print(f"games.columns: {games.columns}")
# print(f"players.columns: {players.columns}")
# print(f"plays.columns: {plays.columns}")
# print(f"player_play.columns: {player_play.columns}")
# print(f"tracking_weekly_datad.columns: {tracking_weekly_data.columns}")


tracking_weekly_data.head(1)
all_cols = list(tracking_weekly_data.columns)
columns_to_keep = ['frameType', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'inMotionAtBallSnap', 'shiftSinceLineset', 'motionSinceLineset', 'quarter', 'down', 'yardsToGo', 'height', 'weight']

model_dataset = tracking_weekly_data[columns_to_keep]

model_dataset['frameType'] = model_dataset['frameType'].apply(lambda x: 1 if x == 'BEFORE_SNAP' else (2 if x == 'AFTER_SNAP' else 0))

model_dataset['shiftSinceLineset'] = model_dataset['shiftSinceLineset'].fillna(False)
model_dataset['shiftSinceLineset'] = model_dataset['shiftSinceLineset'].astype(int)

model_dataset['motionSinceLineset'] = model_dataset['motionSinceLineset'].fillna(False)
model_dataset['motionSinceLineset'] = model_dataset['motionSinceLineset'].astype(int)

model_dataset['inMotionAtBallSnap'] = model_dataset['inMotionAtBallSnap'].fillna(False)
model_dataset['inMotionAtBallSnap'] = model_dataset['inMotionAtBallSnap'].astype(int)

model_dataset['height'] = model_dataset['height'].str.split('-')
model_dataset['height'] = model_dataset['height'].apply(lambda x: int(x[0]) * 12 + int(x[1]) if isinstance(x, list) else None)

model_dataset = model_dataset[~model_dataset['quarter'].isna()]



X = model_dataset.drop(columns=['motionSinceLineset'])
y = model_dataset['motionSinceLineset']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

y_pred_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])

plt.show()




