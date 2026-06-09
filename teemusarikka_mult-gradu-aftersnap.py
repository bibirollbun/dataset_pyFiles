# Load the necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import keras_tuner as kt
from sklearn.metrics import confusion_matrix
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


# Combine tracking data and filter to contain data from 0.5 seconds after the ball snap
all_tracking_data = [tracking_week1,tracking_week2,tracking_week3,tracking_week4,tracking_week5,tracking_week6,tracking_week7,tracking_week8
                     ,tracking_week9]
tracking_data = pd.concat(all_tracking_data, ignore_index=True)
ball_snap_rows = tracking_data[tracking_data['event'] == "ball_snap"]
tracking_features = tracking_data.iloc[ball_snap_rows.index - 5]
tracking_features = tracking_features.merge(players, on='nflId').reset_index(drop=True)


print(tracking_features.shape)
print(tracking_features.iloc[0])


# Merge more data together to form a more comprehensive dataframe
play = plays.merge(games, on='gameId', how='left').reset_index(drop=True)
df = tracking_features.merge(play, on=['gameId','playId'], how='left').reset_index()


# Modify the x coordinate of each player to represent the distance from the line of scrimmage
df.loc[(df['possessionTeam'] == df['homeTeamAbbr']) & (df['possessionTeam'] == df['yardlineSide']), 'x'] = df['x'] - 10 - df['yardlineNumber']
df.loc[(df['possessionTeam'] == df['homeTeamAbbr']) & (df['possessionTeam'] != df['yardlineSide']),'x'] = df['x'] - 10 - (100 - df['yardlineNumber'])
df.loc[(df['possessionTeam'] != df['homeTeamAbbr']) & (df['possessionTeam'] == df['yardlineSide']), 'x'] = df['yardlineNumber'] + 20 - df['x']
df.loc[(df['possessionTeam'] != df['homeTeamAbbr']) & (df['possessionTeam'] != df['yardlineSide']), 'x'] = df['yardlineNumber'] + 10 - df['x']


# Drop unused columns and group by gameID and playID
df['Id'] = df['gameId'].astype(str) + df['playId'].astype(str)
data_Mult = df[['Id','x','y','s','a','dis','o','dir']]
grouped = data_Mult.groupby(['Id'])
df_Mult = grouped.agg(list).reset_index()


# Flatten the lists into a two dimensional data matrix
x = df_Mult['x'].apply(pd.Series).add_prefix('x_')
y = df_Mult['y'].apply(pd.Series).add_prefix('y_')
s = df_Mult['s'].apply(pd.Series).add_prefix('s_')
a = df_Mult['a'].apply(pd.Series).add_prefix('a_')
ds = df_Mult['dis'].apply(pd.Series).add_prefix('dis_')
o = df_Mult['o'].apply(pd.Series).add_prefix('o_')
dr = df_Mult['dir'].apply(pd.Series).add_prefix('dir_')
df_Mult = pd.concat([df_Mult.drop(['x','y','s','a','dis','o','dir'], axis=1),x,y,s,a,ds,o,dr], axis=1)


print(x.shape)
print(df_Mult.shape)


# Merge together to form a complete dataframe
play['Id'] = play['gameId'].astype(str) + play['playId'].astype(str)
df_Mult = df_Mult.merge(play, on='Id', how='left').reset_index()


# Define target variable and features
target = 'isDropback'
features = ['x_0','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','x_9','x_10','x_11','x_12','x_13','x_14','x_15','x_16','x_17','x_18',
            'x_19','x_20','x_21','y_0','y_1','y_2','y_3','y_4','y_5','y_6','y_7','y_8','y_9','y_10','y_11','y_12','y_13','y_14','y_15',
            'y_16','y_17','y_18','y_19','y_20','y_21','s_0','s_1','s_2','s_3','s_4','s_5','s_6','s_7','s_8','s_9','s_10','s_11','s_12',
            's_13','s_14','s_15','s_16','s_17','s_18','s_19','s_20','s_21','a_0','a_1','a_2','a_3','a_4','a_5','a_6','a_7','a_8','a_9',
            'a_10','a_11','a_12','a_13','a_14','a_15','a_16','a_17','a_18','a_19','a_20','a_21','dis_0','dis_1','dis_2','dis_3','dis_4',
            'dis_5','dis_6','dis_7','dis_8','dis_9','dis_10','dis_11','dis_12','dis_13','dis_14','dis_15','dis_16','dis_17','dis_18',
            'dis_19','dis_20','dis_21','o_0','o_1','o_2','o_3','o_4','o_5','o_6','o_7','o_8','o_9','o_10','o_11','o_12','o_13','o_14',
            'o_15','o_16','o_17','o_18','o_19','o_20','o_21','dir_0','dir_1','dir_2','dir_3','dir_4','dir_5','dir_6','dir_7','dir_8',
            'dir_9','dir_10','dir_11','dir_12','dir_13','dir_14','dir_15','dir_16','dir_17',
            'dir_18','dir_19','dir_20','dir_21']

X = df_Mult[features]
y = df_Mult[target]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print(X_train.shape)
print(y_train.shape)


# Define the model
model = Sequential()
model.add(Input(shape=(154,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
model_history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))


# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert the probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)


# Show the model's test accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Show model summary
model.summary()


# Plot training history for the model
plt.plot(model_history.history['accuracy'], label='Training Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(model_history.history['loss'], label='Training Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.legend()
plt.show()

