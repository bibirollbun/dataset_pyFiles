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
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# Importing some selected datasets
m_season_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
season = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv')
m_tourney_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
m_tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyCompactResults.csv')
m_teams= pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')



# Looking at our datasets
m_season_results.head()


season.head()


m_tourney_seed.head()


m_tourney_results.head()


m_teams.head()


# Merging our Tourney Seeds and Teams

team_seeds = pd.merge(m_tourney_seed, m_teams, on='TeamID', how='inner')
team_seeds.head()


# importing our coaches and conference data
coaches = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeamCoaches.csv')
conference = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeamConferences.csv')
print (coaches.head())
print(conference.head())


# Merging the coaches and conference datasets
coach_conf = pd.merge(coaches, conference, on=['TeamID', 'Season'], how = 'inner')
coach_conf.head()


# Merging the coach_conf with our Team_seed data

team_info = pd.merge(team_seeds, coach_conf, on =['TeamID','Season'], how='inner')
team_info.head()


# reading in our rankings data
rankings = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv')
rankings.head()


# Merging the rankings data with our created team_info dataset

team_info= pd.merge(team_info, rankings, on=['TeamID', 'Season'], how='inner')
team_info.tail()


# Looking at the details of our dataset

team_info.describe().T


# Checking for duplicates
team_info.duplicated().sum()


# Checking for missing values
team_info.isna().sum()


team_info.shape


plt.figure(figsize=(12,15))
sns.barplot(team_info, x='OrdinalRank', y='Seed')


#making a copy of our dataframe

team_info_df = team_info.copy()

# changing our TeamID column name

winning_team= team_info.rename(columns={'TeamID':'WTeamID'})


# Combining our winning_team and m_tourney_results dataframes

winning_team = pd.merge(winning_team,m_tourney_results, on='WTeamID', how='inner')


winning_team.head()


plt.figure(figsize=(12,15))
sns.countplot(winning_team, x='ConfAbbrev', order=winning_team['ConfAbbrev'].value_counts().index)
plt.xticks(rotation=90);


# Adding a column for score differential

winning_team['Score_Diff']=winning_team['WScore'] - winning_team['LScore']


# Simplifying our seeding within the dataframe
winning_team['Seed']=winning_team['Seed'].str.extract('(\d+)')


plt.figure(figsize=(12,15))
sns.histplot(winning_team, x='Score_Diff', hue = 'Seed')


plt.figure(figsize=(12,15))
sns.countplot(winning_team, x='WLoc', hue='Score_Diff')
plt.xticks(rotation=90);


# Looking at our losing team information

# changing our TeamID column name

losing_team= team_info.rename(columns={'TeamID':'LTeamID'})


# Combining our losing_team and m_tourney_results dataframes

losing_team = pd.merge(losing_team,m_tourney_results, on='LTeamID', how='inner')

# Adding a column for score differential

losing_team['Score_Diff']=losing_team['WScore'] - losing_team['LScore']

# Simplifying our seeding within the dataframe
losing_team['Seed']=losing_team['Seed'].str.extract('(\d+)')


# Viewing which conferences lose the most

plt.figure(figsize=(12,15))
sns.countplot(losing_team, x='ConfAbbrev', order=losing_team['ConfAbbrev'].value_counts().index)
plt.xticks(rotation=90);


losing_team.head()


tourney_details=pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')
tourney_details.head()


# filtering to only include data from 2020 and onward

tourney_details=tourney_details[tourney_details['Season']>= 2020]


tourney_details.head()


# filtering Ranking data to only include data from 2020 and onward

rankings=rankings[rankings['Season']>= 2020]


# Merging our rankings with our tourney_details

# Merge rankings with WTeamID
tourney_details = tourney_details.merge(rankings, left_on='WTeamID', right_on='TeamID', how='left')
tourney_details.rename(columns={'OrdinalRank': 'WTeamRank'}, inplace=True)
tourney_details.drop(columns=['TeamID'], inplace=True)  # Remove extra TeamID column





# Building Features to our tourney_details dataframe

def features(df):
    df['FGPerc']=df['WFGM']/df['WFGA'] #field goal percentage
    df['3PtPerc']=df['WFGM3']/df['WFGA3'] #3 point percentage
    df['FTPerc']=df['WFTM']/df['WFTA'] # free throw percentage
    df['TotRb'] = df['WOR'] + df['WDR'] # total rebounds
    df['TOMargin']=df['WTO'] - df['LTO'] # turnover margin

    return df


tourney_details = features(tourney_details)


tourney_details.head()


# filtering out the multiple ranking systems to only keep AP rankings (for simplicity)

tourney_details = tourney_details[tourney_details['SystemName'] =='AP']

tourney_details.head()


sns.histplot(tourney_details, x='FGPerc', kde=True)


sns.histplot(tourney_details, x='3PtPerc', kde=True)


sns.histplot(tourney_details, x='TOMargin', kde=True)


sns.histplot(tourney_details, x='TotRb', kde=True)


plt.figure(figsize=(20,10))
sns.heatmap(tourney_details.corr(numeric_only=True), annot=True, fmt='.1f', cmap='viridis')


!pip install plotly

#Looking at some stats as they apply to rank

px.scatter_3d(tourney_details,x='TOMargin', y='FGPerc',z='WAst', color="WTeamRank")


plt.figure(figsize=(25,12))
sns.countplot(tourney_details, x= 'WTeamID', order=tourney_details['WTeamID'].value_counts().index)
plt.xticks(rotation=90);


# Reading in Teams so we can identify which teams are winnings

teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
teams.head()


# Locating our top 5 teams

team_ID = [1211, 1222,1242,1124,1163]
teams['TeamID'] = teams['TeamID'].astype(int)  # Ensure TeamID is an integer
top_5 = teams[teams['TeamID'].isin(team_ID)][['TeamID', 'TeamName']]
print(top_5.sort_values(by='TeamID', ascending=False))


tourney_details.info()


sns.barplot(tourney_details, x='WTeamRank', y='3PtPerc')


!pip install scikeras
!pip install scikit-learn==1.2.2


# Importing Modeling Libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input, Dropout,BatchNormalization
from scikeras.wrappers import KerasClassifier

import random
from tensorflow.keras import backend
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


# filtering team_seeds data to only include data from 2020 and onward

team_seeds=team_seeds[team_seeds['Season']>= 2020]


#dropping columns in team_seeds to prepare for merging

team_seeds=team_seeds.drop(columns=['TeamName', 'FirstD1Season','LastD1Season'])


tourney_details=tourney_details.drop(columns='Season_y')


# Renaming Season_x to season in the dataset
tourney_details=tourney_details.rename(columns={'Season_x':'Season'})


# Merging the tourney_details and team_seeds dataframes

tourney_details=tourney_details.merge(
    team_seeds[['Seed', 'Season', 'TeamID']], # columns from team_seeds
    left_on=['WTeamID', 'Season'], # Matching WTeamID with TeamID
    right_on=['TeamID','Season'],
    how='left' # Keeping all rows from tourney_details
)


# Renaming Seed column

tourney_details.rename(columns={'Seed':'WSeed'}, inplace=True)

# Dropping TeamID column

tourney_details.drop(columns=['TeamID'], inplace=True)

tourney_details.head()


# Simplifying our seeding within the dataframe
tourney_details['WSeed']=tourney_details['WSeed'].str.extract('(\d+)')



#making a copy of our dataset
df_tourney = tourney_details.copy()

df_tourney.head()


df_tourney_encoded=pd.get_dummies(df_tourney, drop_first=True)


#Splitting our target feature from the predictive features

X=df_tourney_encoded.drop(['WScore'], axis=1)
y=df_tourney_encoded[['WScore']]


# Splitting our data for training

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


backend.clear_session()
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)


from tensorflow import keras
from tensorflow.keras import layers

model=keras.Sequential([
    layers.Dense(128,activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8,activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='Adam', loss='mse' ,metrics=['mae'])



model.summary()


# Training the model


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32)



# Model Evaluation
# Evaluate on test data
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.2f}")


# Predict WScore for new data
predictions = model.predict(X_test)
print(predictions[:5])  # Show first 5 predictions


# Comparing predicted scores to Ground Truth scores

y_test=y_test.values.flatten()

y_pred=model.predict(X_test).flatten()

comparison_df=pd.DataFrame({'Actual WScore': y_test, 'Predicted WScore':y_pred})

print(comparison_df.head())


#Evaluating model metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-Squared (R²): {r2:.2f}")



backend.clear_session()
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)


model2=keras.Sequential([
    layers.Dense(128,activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(8,activation='relu'),
    layers.Dense(1)
])

model2.compile(optimizer='Adam', loss='mse' ,metrics=['mae'])



model2.summary()


history = model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32)


# Model Evaluation
# Evaluate on test data
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.2f}")

