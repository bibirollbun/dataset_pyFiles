import os
import re
import sklearn
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.model_selection import *

pd.set_option('display.max_columns', None)



DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/'
DATA_PATH_M = '/kaggle/input/march-machine-learning-mania-2025/'

for filename in os.listdir(DATA_PATH):
    print(filename)


df_seeds = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv"),
], ignore_index=True)
    
df_seeds



df_seeds


df_seeds.info()


df_season_results = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonCompactResults.csv"),
    pd.read_csv(DATA_PATH + "WRegularSeasonCompactResults.csv"),
], ignore_index=True)

df_season_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)


df_season_results


df_season_results['ScoreGap'] = df_season_results['WScore'] - df_season_results['LScore']


df_season_results


# df_season_resul


num_win = df_season_results.groupby(['Season', 'WTeamID']).count()
num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"})


num_win


num_loss = df_season_results.groupby(['Season', 'LTeamID']).count()
num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"})


num_loss


df_season_results


gap_win = df_season_results.groupby(['Season', 'WTeamID']).mean().reset_index()
gap_win = gap_win[['Season', 'WTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"})


gap_win


gap_loss = df_season_results.groupby(['Season', 'LTeamID']).mean().reset_index()
gap_loss = gap_loss[['Season', 'LTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"})


gap_loss




# Merge all dfs together 
df_features_season_w = df_season_results.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
df_features_season_l = df_season_results.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})

df_features_season = pd.concat([df_features_season_w, df_features_season_l], axis=0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)


df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_loss, on=['Season', 'TeamID'], how='left')


df_features_season


# checking  for null values
df_features_season.isnull().sum()


# filling up the missing values with the median .
# This ensures missing values are filled based on the specific team's past performance


df_features_season['NumWins'] = df_features_season.groupby('TeamID')['NumWins'].transform(lambda x: x.fillna(x.median()))
df_features_season['NumLosses'] = df_features_season.groupby('TeamID')['NumLosses'].transform(lambda x: x.fillna(x.median()))
df_features_season['GapWins'] = df_features_season.groupby('TeamID')['GapWins'].transform(lambda x: x.fillna(x.median()))
df_features_season['GapLosses'] = df_features_season.groupby('TeamID')['GapLosses'].transform(lambda x: x.fillna(x.median()))



df_features_season.isnull().sum()


df_features_season


df_features_season['WinRatio'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])
df_features_season['GapAvg'] = (
    (df_features_season['NumWins'] * df_features_season['GapWins'] - 
    df_features_season['NumLosses'] * df_features_season['GapLosses'])
    / (df_features_season['NumWins'] + df_features_season['NumLosses'])
)

df_features_season.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)


df_features_season




df_tourney_results = pd.concat([
    pd.read_csv(DATA_PATH + "WNCAATourneyCompactResults.csv"),
    pd.read_csv(DATA_PATH + "MNCAATourneyCompactResults.csv"),
], ignore_index=True)
df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)




df_tourney_results


df_tourney_results.isnull().sum().sum()




# Training data
df = df_tourney_results.copy()
df = df[df['Season'] >= 2010].reset_index(drop=True)
# considering team performance from 2010 onwards

df



df_seeds


# SeedL is losing team and SeedW is winning team
df = pd.merge(
    df, 
    df_seeds, 
    how='left', 
    left_on=['Season', 'WTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})


df


df = pd.merge(
    df, 
    df_seeds, 
    how='left', 
    left_on=['Season', 'LTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})


df


def treat_seed(seed):
    return int(re.sub("[^0-9]", "", seed))

df['SeedW'] = df['SeedW'].apply(treat_seed)
df['SeedL'] = df['SeedL'].apply(treat_seed)
df.head(30)


winning_teams = df['WTeamID'].unique()
winning_teams



losing_teams = df['LTeamID'].unique()
losing_teams


all_teams = set(df['WTeamID']).union(set(df['LTeamID']))
all_teams


only_won_teams = set(winning_teams) - set(losing_teams)  # Teams that never lost
only_lost_teams = set(losing_teams) - set(winning_teams)  # Teams that never won
both_won_and_lost = set(winning_teams).intersection(set(losing_teams))  # Teams that won and lost



# print("Teams that have only won games (never lost):", only_won_teams)
# print("Teams that have only lost games (never won):", only_lost_teams)
# print("Teams that have both won and lost games:", both_won_and_lost)
only_won_teams_count = len(only_won_teams)
only_lost_teams_count = len(only_lost_teams)
both_won_and_lost_count = len(both_won_and_lost)



import matplotlib.pyplot as plt

# Data
categories = ["Only Won", "Only Lost", "Won & Lost"]
values = [only_won_teams_count, only_lost_teams_count, both_won_and_lost_count]

# Create Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(categories, values, color=['green', 'red', 'blue'])

# Labels & Title
plt.xlabel("Team Category")
plt.ylabel("Number of Teams")
plt.title("Teams Based on Wins & Losses")
plt.ylim(0, max(values) + 10)  # Add some space above the highest bar

# Display Values on Top of Bars
for i, v in enumerate(values):
    plt.text(i, v + 1, str(v), ha='center', fontsize=12)

# Show Plot
plt.show()



df_features_season



df = pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsW',
    'NumLosses': 'NumLossesW',
    'GapWins': 'GapWinsW',
    'GapLosses': 'GapLossesW',
    'WinRatio': 'WinRatioW',
    'GapAvg': 'GapAvgW',
}).drop(columns='TeamID', axis=1)



df


df = pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsL',
    'NumLosses': 'NumLossesL',
    'GapWins': 'GapWinsL',
    'GapLosses': 'GapLossesL',
    'WinRatio': 'WinRatioL',
    'GapAvg': 'GapAvgL',
}).drop(columns='TeamID', axis=1)


df





def add_loosing_matches(df):
    win_rename = {
        "WTeamID": "TeamIdA", 
        "WScore" : "ScoreA", 
        "LTeamID" : "TeamIdB",
        "LScore": "ScoreB",
     }
    win_rename.update({c : c[:-1] + "A" for c in df.columns if c.endswith('W')})
    win_rename.update({c : c[:-1] + "B" for c in df.columns if c.endswith('L')})
    
    lose_rename = {
        "WTeamID": "TeamIdB", 
        "WScore" : "ScoreB", 
        "LTeamID" : "TeamIdA",
        "LScore": "ScoreA",
    }
    lose_rename.update({c : c[:-1] + "B" for c in df.columns if c.endswith('W')})
    lose_rename.update({c : c[:-1] + "A" for c in df.columns if c.endswith('L')})
    
    win_df = df.copy()
    lose_df = df.copy()
    
    win_df = win_df.rename(columns=win_rename)
    lose_df = lose_df.rename(columns=lose_rename)
    
    return pd.concat([win_df, lose_df], axis=0, sort=False)

df = add_loosing_matches(df)
df.head(30)


df


cols_to_diff = [
    'Seed', 'WinRatio', 'GapAvg'
]

for col in cols_to_diff:
    df[col + 'Diff'] = df[col + 'A'] - df[col + 'B']


df


df_test = pd.read_csv(DATA_PATH+'SampleSubmissionStage2.csv')


df_test


df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))

df_test.head(30)




df_test = pd.merge(
    df_test,
    df_seeds,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'}).fillna('W01')

df_test = pd.merge(
    df_test, 
    df_seeds, 
    how='left', 
    left_on=['Season', 'TeamIdB'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'}).fillna('W01')
df_test['SeedA'] = df_test['SeedA'].apply(treat_seed)
df_test['SeedB'] = df_test['SeedB'].apply(treat_seed)
df_test.head(30)




df_test


df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsA',
    'NumLosses': 'NumLossesA',
    'GapWins': 'GapWinsA',
    'GapLosses': 'GapLossesA',
    'WinRatio': 'WinRatioA',
    'GapAvg': 'GapAvgA',
}).drop(columns='TeamID', axis=1)

df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsB',
    'NumLosses': 'NumLossesB',
    'GapWins': 'GapWinsB',
    'GapLosses': 'GapLossesB',
    'WinRatio': 'WinRatioB',
    'GapAvg': 'GapAvgB',
}).drop(columns='TeamID', axis=1)


df_test


for col in cols_to_diff:
    df_test[col + 'Diff'] = df_test[col + 'A'] - df_test[col + 'B']
    
# Compute Difference in Final Score (ScoreDiff) and whether or not the team won (WinA)
df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
df['WinA'] = (df['ScoreDiff'] > 0).astype(int)
df 


df_test


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# plot ScoreDiff
axes[0, 0].hist(df['ScoreDiff'], bins=20, alpha=0.5)
axes[0, 0].axvline(df['ScoreDiff'].median(), color='red', linestyle='--', label='Median')
axes[0, 0].axvline(df['ScoreDiff'].mean(), color='green', linestyle='--', label='Mean')
axes[0, 0].set_title('ScoreDiff')
axes[0, 0].legend()

# plot SeedDiff
axes[0, 1].hist(df['SeedDiff'], bins=20, alpha=0.5)
axes[0, 1].axvline(df['SeedDiff'].median(), color='red', linestyle='--', label='Median')
axes[0, 1].axvline(df['SeedDiff'].mean(), color='green', linestyle='--', label='Mean')
axes[0, 1].set_title('SeedDiff')
axes[0, 1].legend()

# plot WinRatioDiff
axes[1, 0].hist(df['WinRatioDiff'], bins=20, alpha=0.5)
axes[1, 0].axvline(df['WinRatioDiff'].median(), color='red', linestyle='--', label='Median')
axes[1, 0].axvline(df['WinRatioDiff'].mean(), color='green', linestyle='--', label='Mean')
axes[1, 0].set_title('WinRatioDiff')
axes[1, 0].legend()

# plot GapAvgDiff
axes[1, 1].hist(df['GapAvgDiff'], bins=20, alpha=0.5)
axes[1, 1].axvline(df['GapAvgDiff'].median(), color='red', linestyle='--', label='Median')
axes[1, 1].axvline(df['GapAvgDiff'].mean(), color='green', linestyle='--', label='Mean')
axes[1, 1].set_title('GapAvgDiff')
axes[1, 1].legend()

plt.show()


X = df[["SeedA", "SeedB", 'WinRatioA', 'GapAvgA', 'WinRatioB', 'GapAvgB', 'SeedDiff', 'WinRatioDiff', 'GapAvgDiff']]
y = df[['WinA']]





fig, axs = plt.subplots(3,3, figsize=(12, 10))

# plot the distribution of each feature
for i, col in enumerate(X.columns):
    sns.histplot(data=X, x=col, ax=axs[i//3, i%3], kde=True)
    
# add a title to the figure
fig.suptitle('Distribution of Features', fontsize=16)

# Adjust the layout of the subplots
fig.tight_layout()

# display the plot
plt.show()



from sklearn.preprocessing import StandardScaler

# drop columns that we don't want to scale
X = df.drop(columns=['Season', 'DayNum', 'TeamIdA', 'ScoreA', 'TeamIdB', 'ScoreB', 'WinA', 'ScoreDiff'])

# create the scaler object
scaler = StandardScaler()

# fit the scaler object to the data and transform the data
X = scaler.fit_transform(X)

# assign y values
y = df['WinA']


print("hello")


from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, roc_curve


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def brier_score_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))


pip install XGBoost


import xgboost as xgb
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost classifier
model = xgb.XGBClassifier(
    objective="binary:logistic",  # Ensures probability output
    eval_metric="logloss",        # Optimizing log-loss aligns well with Brier score
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1
)

# Train the model
model.fit(X_train, y_train)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1

# Calculate Brier Score
brier_score = brier_score_loss(y_test, y_probs)
print(f"Brier Score: {brier_score:.4f}")


features = [
    "SeedA", "SeedB", 'WinRatioA', 'GapAvgA', 'WinRatioB', 'GapAvgB', 'SeedDiff', 'WinRatioDiff', 'GapAvgDiff'
]


def rescale(features, df_train, df_val, df_test=None):
    min_ = df_train[features].min()
    max_ = df_train[features].max()
    
    df_train[features] = (df_train[features] - min_) / (max_ - min_)
    df_val[features] = (df_val[features] - min_) / (max_ - min_)
    
    if df_test is not None:
        df_test[features] = (df_test[features] - min_) / (max_ - min_)
        
    return df_train, df_val, df_test


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss

# Assuming X_train, X_test, y_train, y_test are already defined

# Step 1: Train the Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Step 2: Predict probabilities (for class 1)
y_prob = gb_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (Win=1)

# Step 3: Predict actual 0/1 values
y_pred = gb_model.predict(X_test)

# Step 4: Evaluate using Brier Score
brier_score = brier_score_loss(y_test, y_prob)

# Print Results
print(f"Brier Score: {brier_score:.4f}")
print("\nPredicted Probabilities vs. Actual Labels:")
results_df = pd.DataFrame({"Predicted_Prob": y_prob, "Actual_Label": y_test})
print(results_df)  # Show first 10 results

# Step 5: Function to Predict Probabilities for New Data
def predict_probabilities(new_data):
    """
    Given a new dataset (same features as training data),
    returns predicted probabilities for each row.
    """
    return gb_model.predict_proba(new_data)[:, 1]




results_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay

# Assuming X_train, X_test, y_train, y_test are already defined

# Step 1: Train the Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Step 2: Predict probabilities (for class 1)
y_prob = gb_model.predict_proba(X_test)[:, 1]  # Probabilities for Win=1

# Step 3: Predict actual 0/1 values
y_pred = gb_model.predict(X_test)

# Step 4: Evaluate using Accuracy and Brier Score
accuracy = accuracy_score(y_test, y_pred)
brier_score = brier_score_loss(y_test, y_prob)

# Print Results
print(f"Accuracy: {accuracy:.4f}")
print(f"Brier Score: {brier_score:.4f}")

# Step 5: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Step 6: DataFrame for Comparison
results_df = pd.DataFrame({"Predicted_Prob": y_prob, "Predicted_Label": y_pred, "Actual_Label": y_test})
print("\nPredicted Probabilities vs. Actual Labels:")
print(results_df.head(10))

# Step 7: Graphical Distribution of Predicted Probabilities
plt.figure(figsize=(8, 6))
sns.histplot(results_df[results_df["Actual_Label"] == 1]["Predicted_Prob"], bins=25, label="Actual 1", color="blue", alpha=0.6)
sns.histplot(results_df[results_df["Actual_Label"] == 0]["Predicted_Prob"], bins=25, label="Actual 0", color="red", alpha=0.6)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities")
plt.legend()
plt.show()

# Step 8: Function to Predict Probabilities for New Data
def predict_probabilities(new_data):
    """
    Given a new dataset (same features as training data),
    returns predicted probabilities for each row.
    """
    return gb_model.predict_proba(new_data)[:, 1]



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay

# Assuming X_train, X_test, y_train, y_test are already defined

def train_and_evaluate(model, model_name):
    """
    Trains the given model, predicts probabilities and labels, evaluates, 
    and visualizes results.
    """
    print(f"\nTraining {model_name}...\n")
    
    # Train Model
    model.fit(X_train, y_train)
    
    # Predict Probabilities (for class 1)
    y_prob = model.predict_proba(X_test)[:, 1]  

    # Predict Actual Labels
    y_pred = model.predict(X_test)

    # Evaluate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    brier_score = brier_score_loss(y_test, y_prob)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Brier Score: {brier_score:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # Create Results DataFrame
    results_df = pd.DataFrame({"Predicted_Prob": y_prob, "Predicted_Label": y_pred, "Actual_Label": y_test})
    print("\nPredicted Probabilities vs. Actual Labels (First 10 Rows):")
    print(results_df.head(10))

    # Plot Distribution of Predicted Probabilities
    plt.figure(figsize=(8, 6))
    sns.histplot(results_df[results_df["Actual_Label"] == 1]["Predicted_Prob"], bins=25, label="Actual 1 (Win)", color="blue", alpha=0.6)
    sns.histplot(results_df[results_df["Actual_Label"] == 0]["Predicted_Prob"], bins=25, label="Actual 0 (Loss)", color="red", alpha=0.6)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title(f"Probability Distribution - {model_name}")
    plt.legend()
    plt.show()

    return model

# Train and evaluate each model separately
rf_model = train_and_evaluate(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), "Random Forest")
xgb_model = train_and_evaluate(XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, use_label_encoder=False, eval_metric="logloss", random_state=42), "XGBoost")

# Function to predict probabilities for new data
def predict_probabilities(model, new_data):
    return model.predict_proba(new_data)[:, 1]



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay

def train_and_evaluate_logistic():
    """
    Trains Logistic Regression, predicts probabilities and labels,
    evaluates, and visualizes results.
    """
    print("\nTraining Logistic Regression...\n")
    
    # Train Model
    log_model = LogisticRegression(solver='liblinear', random_state=42)
    log_model.fit(X_train, y_train)
    
    # Predict Probabilities (for class 1)
    y_prob = log_model.predict_proba(X_test)[:, 1]

    # Predict Actual Labels
    y_pred = log_model.predict(X_test)

    # Evaluate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    brier_score = brier_score_loss(y_test, y_prob)

    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print(f"Logistic Regression Brier Score: {brier_score:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    # Create Results DataFrame
    results_df = pd.DataFrame({"Predicted_Prob": y_prob, "Predicted_Label": y_pred, "Actual_Label": y_test})
    print("\nPredicted Probabilities vs. Actual Labels (First 10 Rows):")
    print(results_df.head(10))

    # Plot Distribution of Predicted Probabilities
    plt.figure(figsize=(8, 6))
    sns.histplot(results_df[results_df["Actual_Label"] == 1]["Predicted_Prob"], bins=25, label="Actual 1 (Win)", color="blue", alpha=0.6)
    sns.histplot(results_df[results_df["Actual_Label"] == 0]["Predicted_Prob"], bins=25, label="Actual 0 (Loss)", color="red", alpha=0.6)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Probability Distribution - Logistic Regression")
    plt.legend()
    plt.show()

    return log_model

# Train and evaluate Logistic Regression
logistic_model = train_and_evaluate_logistic()

# Function to predict probabilities for new data
def predict_probabilities_logistic(new_data):
    return logistic_model.predict_proba(new_data)[:, 1]



df


dummy = df.drop(columns = 'WinA')


dummy


y_pred =  logistic_model.predict(dummy[features])


dummy['pred'] = y_pred


dummy['WinA'] = df['WinA']


# checking the predicted and actual value on the original created dataset
dummy[dummy['pred'] != dummy['WinA']]


df


df_test


# making predictions on df_test which is the actual submissions we need to submit
y_prob = xgb_model.predict_proba(df_test[features])[:, 1]  # Extract class 1 probabilities



y_prob


df_test['pred'] = y_prob


df_test





# extracting the teamids and their predictions to create a submission file
final_sub = df_test[['ID', 'pred']].copy()
# final_sub.to_csv('submission.csv', index=False)



import matplotlib.pyplot as plt
import seaborn as sns

# Plot histogram
plt.figure(figsize=(8,5))
sns.histplot(final_sub["pred"], bins=30, kde=True)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities")
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming your trained model is named 'model' and your features are in 'X_train'
feature_importance = rf_model.feature_importances_
feature_names = features

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Plot")
plt.gca().invert_yaxis()  # Invert axis to show the most important at the top
plt.show()



features


df_seeds


combined_seeds_df = df_seeds.copy()


combined_seeds_df


combined_seeds_df['Seed'] = combined_seeds_df['Seed'].apply(treat_seed)


combined_seeds_df


TeamA = df_test['TeamIdA']


TeamB = df_test['TeamIdB']


combined_teams = pd.concat([TeamA, TeamB]).drop_duplicates().reset_index(drop=True)

# Converting to DataFrame with column name 'Seeds'
combined_teams_df = pd.DataFrame(combined_teams, columns=["TeamID"])


combined_teams_df


df_seeds


def assign_seed(team_id):
    # Try to use the 2024 seed
    seed_2024 = combined_seeds_df[(combined_seeds_df['TeamID'] == team_id) & (combined_seeds_df['Season'] == 2024)]
    if not seed_2024.empty:
        return seed_2024.iloc[0]['Seed']
    
    # If missing, use the average of past seeds
    past_seeds = combined_seeds_df[(combined_seeds_df['TeamID'] == team_id) & (combined_seeds_df['Season'] >= 2013)]['Seed']
    if not past_seeds.empty:
        return past_seeds.mean()
    
    # If no data, assign median seed
    return combined_seeds_df['Seed'].median()
# new_seed_df = []
# Apply this to your prediction dataset
combined_teams_df['new_Seed'] = combined_teams_df['TeamID'].apply(assign_seed)



combined_teams_df = combined_teams_df.rename(columns = {'new_Seed':'Seeds'})


combined_teams_df


df_test


import pandas as pd

# Example df_test with TeamID columns

# Creating a dictionary for mapping TeamID to Seeds
seed_mapping = dict(zip(combined_teams_df["TeamID"], combined_teams_df["Seeds"]))

# Replacing seedA and seedB based on teamIdA and teamIdB
df_test["SeedA"] = df_test["TeamIdA"].map(seed_mapping)
df_test["SeedB"] = df_test["TeamIdB"].map(seed_mapping)

# Display the updated df_test
print(df_test)



df_test


df_test['SeedDiff'] = df_test['SeedA'] - df_test['SeedB']


df_test


df_test['pred_xgb'] = xgb_model.predict_proba(df_test[features])[:,1]


df_test['pred_rf'] =rf_model.predict_proba(df_test[features])[:,1]


df_test['pred_lg'] = logistic_model.predict_proba(df_test[features])[:,1]


df_test['pred_gb'] = gb_model.predict_proba(df_test[features])[:,1]


df_test


import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with subplots (1 row, 4 columns)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjust width with figsize

# Define column names and titles
columns = ["pred_rf", "pred_xgb", "pred_lg", "pred_gb"]
titles = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
    "Gradient Boosting"
]

# Loop through columns and plot on respective axes
for i, col in enumerate(columns):
    sns.histplot(df_test[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_xlabel("Predicted Probability")
    axes[i].set_ylabel("Frequency")
    axes[i].set_title(f"Distribution of {titles[i]}")

# Adjust layout
plt.tight_layout()

# Show all subplots
plt.show()




df_tourney_boxes = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv"),
], ignore_index=True)
    
df_tourney_boxes



df_regular_boxes = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WRegularSeasonDetailedResults.csv"),
], ignore_index=True)
    
df_regular_boxes






# random forest model predictions
final_df_rf = df_test[['ID', 'pred_rf']].copy()


final_df_rf = final_df_rf.rename(columns = {'pred_rf':'pred'})


# final_df_rf.to_csv('submission_rf.csv',index = False)


# # xgb model  predictions
# final_df_xgb = df_test[['ID', 'pred_xgb']].copy()
# final_df_xgb = final_df_xgb.rename(columns = {'pred_xgb':'pred'})
# final_df_xgb.to_csv('submission_xgb.csv',index = False)


# # gradient boost model predictions
# final_df_gb = df_test[['ID', 'pred_gb']].copy()
# final_df_gb = final_df_rf.rename(columns = {'pred_gb':'pred'})
# final_df_gb.to_csv('submission_gb.csv',index = False)


# final submission
final_df_rf.to_csv('submission.csv',index = False)

