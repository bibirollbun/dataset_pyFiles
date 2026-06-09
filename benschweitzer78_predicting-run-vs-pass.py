# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


tracking_1 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv')
tracking_2 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_2.csv')
tracking_3 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_3.csv')
tracking_4 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_4.csv')
tracking_5 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_5.csv')
tracking_6 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_6.csv')
tracking_7 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_7.csv')
tracking_8 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_8.csv')
tracking_9 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_9.csv')


plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')
player_play = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')


def preprocess_tracking_data(week):
    # Get linebacker IDs
    wr_ids = players.loc[players['position'].isin(['WR']), 'nflId'].values
    
    # Retrieve and filter tracking data for the week
    week_track_df = globals().get(f"tracking_{week}")
    
    # Filter tracking data to include only linebackers at ball snap
    # Identify plays where a ball snap occurs
    
    snap_plays = week_track_df[
        (week_track_df['nflId'].isin(wr_ids)) & 
        (week_track_df['event'] == 'ball_snap')
    ]

    set_plays = week_track_df[
        (week_track_df['nflId'].isin(wr_ids)) & 
        (week_track_df['event'] == 'line_set')
    ]
    
    return snap_plays, set_plays

def wr_width_finder(play_df, snap_plays, set_plays):
    # Merge play data with linebacker tracking data on gameId and playId
    merged_snap = pd.merge(play_df, snap_plays, on=['gameId', 'playId'])
    merged_set = pd.merge(play_df, set_plays, on=['gameId', 'playId'])

    # Calculate depth based on play direction
    merged_snap['width'] = abs(26.65 - merged_snap['y'])
    merged_set['width'] = abs(26.65 - merged_set['y'])
    
    # Group by play and calculate the average linebacker depth
    avg_depths_snap = merged_snap.groupby(['gameId', 'playId'])['width'].mean().reset_index()
    avg_depths_snap.rename(columns={'width': 'avgWRwidthSnap'}, inplace=True)

    avg_depths_set = merged_set.groupby(['gameId', 'playId'])['width'].mean().reset_index()
    avg_depths_set.rename(columns={'width': 'avgWRwidthSet'}, inplace=True)
    
    # Merge average depths back to the plays DataFrame
    result_snap = pd.merge(play_df, avg_depths_snap, on=['gameId', 'playId'], how='left')
    result_set = pd.merge(play_df, avg_depths_set, on=['gameId', 'playId'], how='left')
    
    return result_snap, result_set

def process_all_weeks(plays):
    # Merge plays with games to add week information
    plays = pd.merge(plays, games[['gameId', 'week']], on='gameId', how='left')
   

    # Preprocess tracking data for each week and calculate LB depth
    all_results_set = []
    all_results_snap = []

    # Filter for passing plays
    #plays = plays[plays['isDropback'] == True]
    
    for week in plays['week'].unique():
        # Preprocess tracking data for this week
        snap_plays, set_plays = preprocess_tracking_data(week)
        
        # Filter plays for this week
        week_plays = plays[plays['week'] == week]
        
        # Calculate linebacker depth for plays in this week
        result_snap, result_set = wr_width_finder(week_plays, snap_plays, set_plays)
        
        all_results_snap.append(result_snap)
        all_results_set.append(result_set)
        
    
    # Concatenate results from all weeks
    final_result_set = pd.concat(all_results_set, ignore_index=True)
    final_result_snap = pd.concat(all_results_snap, ignore_index=True)
    
    return final_result_set, final_result_snap


processed_data_set, processed_data_snap = process_all_weeks(plays)
processed_data_snap.shape


# Drop the last column from each dataset but keep the values
last_col_set = processed_data_set.iloc[:, -1]
last_col_snap = processed_data_snap.iloc[:, -1]

# Keep only the common columns
common_set = processed_data_set.iloc[:, :-1]
common_snap = processed_data_snap.iloc[:, :-1]

# Merge based on all common columns (inner join)
processed_data = pd.merge(common_set, common_snap, how='inner')

# Add the unique last columns to the merged dataframe
processed_data['avgWRwidthSet'] = last_col_set.values
processed_data['avgWRwidthSnap'] = last_col_snap.values


processed_data = processed_data.dropna(subset=['avgWRwidthSet'])
processed_data = processed_data.dropna(subset=['avgWRwidthSnap'])

processed_data['avgWRwidthDiff'] = processed_data['avgWRwidthSnap'] - processed_data['avgWRwidthSet']
processed_data.shape


X = processed_data[['avgWRwidthSnap', 'offenseFormation', 'quarter', 'yardsToGo', 'down']]
y = processed_data['isDropback']

le = LabelEncoder()
y = le.fit_transform(y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = xgb.XGBClassifier(  
    objective='multi:softmax',  
    num_class=2,  
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',) 
           # xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for isDropback')
plt.show()

