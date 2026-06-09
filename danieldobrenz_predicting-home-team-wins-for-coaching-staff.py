import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.inspection import PartialDependenceDisplay

import warnings

warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")


# Import csv files
games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')

# Set option to display all rows
pd.set_option('display.max_rows', None)


# Check head of games dataframe
games.head()


games.info()


# Check head of plays dataframe
plays.head()


plays.info()


# Start by merging the dataframes
df = pd.merge(games, plays, on='gameId', how='inner')


df.shape


# Clean up column names
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces
df.columns = df.columns.str.lower()  # Convert all column names to lowercase
df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores


df.head().T


# Sort the data frame by user_id and date
df = df.sort_values(by=['gameid', 'quarter', 'gameclock'])


df.head().T


df.shape


df.shape


df.isnull().sum()


df['yardlineside'].value_counts(dropna=False)


# Select Nan rows for yard side line
nan_rows = df[df['yardlineside'].isna()]


nan_rows.T


nan_rows['yardlinenumber'].value_counts()


# Fill the Nan values with 'MIDDLE'
df['yardlineside'] = df['yardlineside'].fillna('MIDDLE')


df['yardlineside'].isnull().sum()


df['offenseformation'].value_counts(dropna=False)


df['offenseformation'] = df['offenseformation'].fillna('OTHER')


df['receiveralignment'].value_counts(dropna=False)


# Select the NaN rows for receiveralignment
nan_rows = df[df['receiveralignment'].isna()]
nan_rows.T


df['receiveralignment'] = df['receiveralignment'].fillna('0x0')


df['receiveralignment'].isnull().sum()


df['playclockatsnap'].value_counts(dropna=False)


# Drop the row
df = df.dropna(subset=['playclockatsnap'])


df['passresult'].value_counts(dropna=False)


# Select the NaN rows for passresult
nan_rows = df[df['passresult'].isna()]
nan_rows.T


df['passresult'] = df['passresult'].fillna('None')


df['passresult'].isnull().sum()


df['passlength'].value_counts(dropna=False)


# Select the NaN rows for passlength
nan_rows = df[df['passlength'].isna()]
nan_rows.T


# Fill the missing values with 0
df['passlength'] = df['passlength'].fillna(0.0)


# Fill the missing values with 0 for both targetx and targety
df['targetx'] = df['targetx'].fillna(0)
df['targety'] = df['targety'].fillna(0)


df['dropbacktype'].value_counts(dropna=False)


df['dropbacktype'] = df['dropbacktype'].fillna('NON_PASSING_PLAY')


df['dropbackdistance'].value_counts(dropna=False)


df['dropbackdistance'] = df['dropbackdistance'].fillna(0.0)


df['passlocationtype'].value_counts(dropna=False)


df['passlocationtype'] = df['passlocationtype'].fillna('NOT_THROWN')


df['timetothrow'].value_counts(dropna=False)


df['timetothrow'] = df['timetothrow'].fillna(0.0)


df['timeintacklebox'].value_counts(dropna=False)


# Select the NaN rows for timeintacklebox
nan_rows = df[df['timeintacklebox'].isna()]
nan_rows.sample().T


df['timeintacklebox'] = df['timeintacklebox'].fillna(0.0)


df['timetosack'].value_counts(dropna=False)


df['timetosack'] = df['timetosack'].fillna(0.0)


df['passtippedatline'].value_counts(dropna=False)


df['passtippedatline'] = df['passtippedatline'].fillna('NON_PASS')


df['unblockedpressure'].value_counts(dropna=False)


# Select the NaN rows for unblocked pressure
nan_rows = df[df['unblockedpressure'].isna()]
nan_rows.sample(10).T


df['unblockedpressure'] = df['unblockedpressure'].fillna('NON_PASSING_PLAY')


df['qbspike'].value_counts(dropna=False)


# Select the NaN rows for QB Spike
nan_rows = df[df['qbspike'].isna()]
nan_rows.sample(2).T


false_rows = df[df['qbspike'] == False]
false_rows.sample(2).T


df['qbspike'] = df['qbspike'].fillna('NON_PASSING_PLAY')


df['qbsneak'].value_counts(dropna=False)


# Select the NaN rows for QB Sneak
nan_rows = df[df['qbsneak'].isna()]
nan_rows.sample(2).T


df['qbsneak'] = df['qbsneak'].fillna('OTHER')


df['rushlocationtype'].value_counts(dropna=False)


df['rushlocationtype'] = df['rushlocationtype'].fillna('PASSING_PLAY')


df['penaltyyards'].value_counts(dropna=False)


df['penaltyyards'] = df['penaltyyards'].fillna(0.0)


df['pff_runconceptprimary'].value_counts(dropna=False)


# Select the NaN rows for run concept primary
nan_rows = df[df['pff_runconceptprimary'].isna()]
nan_rows.sample(10).T


df['pff_runconceptprimary'] = df['pff_runconceptprimary'].fillna('OTHER')


df['pff_runconceptsecondary'].value_counts(dropna=False)


df['pff_runconceptsecondary'] = df['pff_runconceptsecondary'].fillna('OTHER')


df['pff_passcoverage'].value_counts(dropna=False)


df['pff_passcoverage'] = df['pff_passcoverage'].fillna('OTHER')


df['pff_manzone'].value_counts(dropna=False)


# Select the NaN rows for Man Zone
nan_rows = df[df['pff_manzone'].isna()]
nan_rows.sample(10).T


df['pff_manzone'] = df['pff_manzone'].fillna('NONE')


df.isnull().sum()


# If the home team value is greater than the visitor team then I'll assigned a 1 as the positive class and 0 for the negative class
# This is our target variable
df['win'] = (df['hometeamabbr'] > df['visitorteamabbr']).astype(int)


df['win'].head()


plt.figure(figsize=(10, 6))
ax = df['win'].value_counts().plot(kind='bar', color=['blue', 'orange'])

# Adding counts on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

# Labels and title
plt.title('Wins and Losses')
plt.xlabel('Value')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


df.columns


plt.figure(figsize=(10, 6))
df['week'].value_counts().plot(kind='bar')
plt.title('Plays per Week')
plt.xlabel('Value')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


home_counts = df['hometeamabbr'].value_counts()

# Set figure size
plt.figure(figsize=(20, 10))

# Plotting the bar chart
ax = home_counts.plot(kind='bar', color='blue')

# Adding the numbers at the top of each bar
for p in ax.patches:
    height = p.get_height()
    # Adding the count as text above each bar
    plt.text(p.get_x() + p.get_width() / 2, height + 0.2, f'{height}', ha='center', va='bottom', fontsize=12)

# Adding labels and title
plt.title('Plays per Home Team')
plt.xlabel('Home Team')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.show()


visit_counts = df['visitorteamabbr'].value_counts()

# Set figure size
plt.figure(figsize=(20, 10))

# Plotting the bar chart
ax = visit_counts.plot(kind='bar', color='red')

# Adding the numbers at the top of each bar
for p in ax.patches:
    height = p.get_height()
    # Adding the count as text above each bar
    plt.text(p.get_x() + p.get_width() / 2, height + 0.2, f'{height}', ha='center', va='bottom', fontsize=12)

# Adding labels and title
plt.title('Plays per Visiting Team')
plt.xlabel('Visiting Team')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.show()


df.head().T


# Drop columns that we don't need
df.drop(columns=['gameid', 'season', 'playdescription'], inplace=True)


# Convert gamedate column to datetime format
df['gamedate'] = pd.to_datetime(df['gamedate'])


df['gamedate'].info()


# Extract day and month from the 'date' column
df['gamedateday'] = df['gamedate'].dt.day
df['gamedatedaymonth'] = df['gamedate'].dt.month


# Drop gamedate column
df.drop(columns= 'gamedate', inplace=True)


# Convert the 'gametimeeastern'
df['gametimeeastern'] = pd.to_datetime(df['gametimeeastern'], format='%H:%M:%S')


# Extract hour and minute of game time eastern, leave out seconds. 
df['gametimeeasternhour'] = df['gametimeeastern'].dt.hour
df['gametimeeasternminute'] = df['gametimeeastern'].dt.minute


# Drop game time eastern
df.drop(columns='gametimeeastern', inplace=True)


# Drop the abbreviation columns
df.drop(columns=['hometeamabbr','visitorteamabbr'], inplace=True)


df.head().T


# Create the new columns based on the conditions
df['yardlineside_possessionteam'] = df['yardlineside'] == df['possessionteam']
df['yardlineside_defensiveteam'] = df['yardlineside'] == df['defensiveteam']


# Drop the possession team column and defensive team column
df.drop(columns=['possessionteam', 'defensiveteam', 'yardlineside'], inplace=True)


df.head().T


# Convert the 'game clock' column to datetime
df['gameclock'] = pd.to_datetime(df['gameclock'], format='%H:%M')


# Extract hour and minute of game time eastern, leave out seconds. 
df['gameclock_hour'] = df['gameclock'].dt.hour
df['gameclock_minute'] = df['gameclock'].dt.minute


# Drop the game clock column
df.drop(columns='gameclock', inplace=True)


dummies = pd.get_dummies(df['playnullifiedbypenalty'], prefix='playnullifiedbypenalty')
# Add the dummies back to the original dataframe
df = pd.concat([df, dummies], axis=1)


df['playnullifiedbypenalty_N'].value_counts()


df.shape


df.drop(columns=['playnullifiedbypenalty','playnullifiedbypenalty_N'], inplace=True)


# Offense Formation Dummies
dummies = pd.get_dummies(df['offenseformation'], prefix='offensiveform').astype(int)
df = pd.concat([df, dummies], axis=1)

# Drop original column
df.drop(columns='offenseformation', inplace=True)

# Receiver Alignment Dummies
dummies = pd.get_dummies(df['receiveralignment'], prefix='receiveralignment').astype(int)
df = pd.concat([df, dummies], axis=1)

# Passresult Dummies
dummies = pd.get_dummies(df['passresult'], prefix='passresults').astype(int)
df = pd.concat([df, dummies], axis=1)

# Change Play Action to numerical
df['playaction'] = df['playaction'].astype(int)

# Drop Back Dummies
dummies = pd.get_dummies(df['dropbacktype'], prefix='dropbacktype').astype(int)
df = pd.concat([df, dummies], axis=1)

# Pass Location Type Dummies
dummies = pd.get_dummies(df['passlocationtype'], prefix='passlocationtype').astype(int)
df = pd.concat([df, dummies], axis=1)

# Pass Tipped Line Dummies
dummies = pd.get_dummies(df['passtippedatline'], prefix='passtippedatline').astype(int)
df = pd.concat([df, dummies], axis=1)

# Unblocked Pressure Dummies
dummies = pd.get_dummies(df['unblockedpressure'], prefix='unblockedpressure').astype(int)
df = pd.concat([df, dummies], axis=1)

# QB Spike Dummies
dummies = pd.get_dummies(df['qbspike'], prefix='qbspike').astype(int)
df = pd.concat([df, dummies], axis=1)

# Qb Sneak Dummies
dummies = pd.get_dummies(df['qbsneak'], prefix='qbsneak').astype(int)
df = pd.concat([df, dummies], axis=1)

# Rush Location Type Dummies
dummies = pd.get_dummies(df['rushlocationtype'], prefix='rushlocationtype').astype(int)
df = pd.concat([df, dummies], axis=1)

# Is Drop Back Dummies
dummies = pd.get_dummies(df['isdropback'], prefix='isdropback').astype(int)
df = pd.concat([df, dummies], axis=1)

# Run Concept Primary Dummies
dummies = pd.get_dummies(df['pff_runconceptprimary'], prefix='pff_runconceptprimary').astype(int)
df = pd.concat([df, dummies], axis=1)

# Run Concept Secondary Dummies
dummies = pd.get_dummies(df['pff_runconceptsecondary'], prefix='pff_runconceptsecondary').astype(int)
df = pd.concat([df, dummies], axis=1)

# Pass Coverage Dummies
dummies = pd.get_dummies(df['pff_passcoverage'], prefix='pff_passcoverage').astype(int)
df = pd.concat([df, dummies], axis=1)

# Manzone Dummies
dummies = pd.get_dummies(df['pff_manzone'], prefix='pff_manzone').astype(int)
df = pd.concat([df, dummies], axis=1)

# Yard Line Side Possesstion Team Dummies
dummies = pd.get_dummies(df['yardlineside_possessionteam'], prefix='yardlineside_possessionteam').astype(int)
df = pd.concat([df, dummies], axis=1)

# Yard Line Side Defensive Team Dummies
dummies = pd.get_dummies(df['yardlineside_defensiveteam'], prefix='yardlineside_defensiveteam').astype(int)
df = pd.concat([df, dummies], axis=1)


df.head().T


# Checking columns with categorical types to make sure I got all the dummy variables
categorical_columns = df.select_dtypes(include=['object', 'category']).columns


categorical_columns


# I forgot to drop these column when creating dummy variables
df.drop(columns=['receiveralignment', 'passresult', 'dropbacktype', 'passlocationtype',
       'passtippedatline', 'unblockedpressure', 'qbspike', 'qbsneak',
       'rushlocationtype', 'pff_runconceptprimary', 'pff_runconceptsecondary',
       'pff_passcoverage', 'pff_manzone', 'isdropback', 'yardlineside_possessionteam', 'yardlineside_defensiveteam'], inplace=True)


# Re-check columns with categorical types
categorical_columns = df.select_dtypes(include=['object', 'category']).columns


categorical_columns


# Check the shape of the final dataframe
df.shape


df.head().T


# Drop final score columns
df.drop(columns=['homefinalscore', 'visitorfinalscore'], inplace=True)


# Clean the column names
df.columns = (
    df.columns
    .str.strip()                    # Remove leading and trailing whitespace
    .str.lower()                    # Convert to lowercase
    .str.replace(r'[^a-z0-9]', '_') # Replace any non-alphanumeric character with an underscore
    .str.replace(r'__+', '_')       # Replace multiple underscores with a single one
    .str.replace('-', '_')
    .str.replace(' ', '_')
    .str.replace(';', '_')
)


X = df.drop(columns=['win'])
y = df['win']


# Split the data into training and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_train_scores = model.predict_proba(X_train)[:, 1]
    y_pred = model.predict(X_test)
    y_test_scores = model.predict_proba(X_test)[:, 1]

    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("AUC_score_test", round(roc_auc_score(y_test, y_test_scores),2))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))
    print("AUC_score_test", round(roc_auc_score(y_train, y_train_scores),2))

def metric_df(model, X_train, y_train, X_test, y_test,name):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    scores = {name: {"accuracy" : accuracy_score(y_test,y_pred),
    "precision" : precision_score(y_test, y_pred),
    "recall" : recall_score(y_test, y_pred),                          
    "f1" : f1_score(y_test,y_pred),
     "True Negative Rate": confusion_matrix(y_test,y_pred)[0][0]/confusion_matrix(y_test,y_pred).sum(),
     "False Positive Rate": confusion_matrix(y_test,y_pred)[0][1]/confusion_matrix(y_test,y_pred).sum(),
    "False Negative Rate": confusion_matrix(y_test,y_pred)[1][0]/confusion_matrix(y_test,y_pred).sum(),
    "True Positive Rate": confusion_matrix(y_test,y_pred)[1][1]/confusion_matrix(y_test,y_pred).sum()}}
    return pd.DataFrame(scores)


# Initilize the model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

#  Predictions
y_pred= rf_model.predict(X_test)

eval_metric(rf_model, X_train, y_train, X_test, y_test)


skip_this_cell = True

if not skip_this_cell:

    # Grid Search Random Forest
    rf_tune = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    grid_rf = GridSearchCV(estimator=rf_tune, param_grid=param_grid, cv=5, scoring='precision', n_jobs=-1, verbose=1)
    
    # Fit the model
    grid_rf.fit(X_train, y_train)
    
    # Show best parameters
    print(f"Best Parameters: {grid_rf.best_params_}")
    
    # Evaluation Metrics
    eval_metric(grid_rf, X_train, y_train, X_test, y_test)


# Initialize the model
rf_model_tune = RandomForestClassifier(bootstrap=False, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300,random_state=42)

# Fit the model
rf_model_tune.fit(X_train, y_train)

# Evaluation Metrics
eval_metric(rf_model_tune, X_train, y_train, X_test, y_test)


# Feature importances
importance = rf_model_tune.feature_importances_

# Create dataframe
feature_importance_rf = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})

# Sort Values
feature_importance_rf = feature_importance_rf.sort_values(by='Importance', ascending=False)
feature_importance_rf


top_20_features = feature_importance_rf.head(20)

# Plotting the top 20 features
plt.figure(figsize=(10, 6))
plt.barh(top_20_features['Feature'], top_20_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 20 Features - Feature Importance')
plt.gca().invert_yaxis()  # To display the top feature at the top
plt.show()


# Plotting
plt.figure(figsize=(8, 6))
sns.histplot(df[df['win'] == 0]['gamedateday'], label='Loss', kde=True, color='blue', alpha=0.6)
sns.histplot(df[df['win'] == 1]['gamedateday'], label='Win', kde=True, color='orange', alpha=0.6)
plt.xlabel('Game Date Day')
plt.ylabel('Count')
plt.title('Histogram: Game Date Day vs Wins/Losses')
plt.legend()
plt.show()


xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train);
# Predict on the test data
y_pred = xgb_model.predict(X_test)

# Convert probabilities to binary outcomes (for binary classification)
y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

# Get probabilities for the positive class
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation
eval_metric(xgb_model, X_train, y_train, X_test, y_test)


# Get the feature importances
importance = xgb_model.feature_importances_

# Create a DataFrame for feature importance
feature_importance_xgb = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})

# Sort the feature importances
feature_importance_xgb = feature_importance_xgb.sort_values(by='Importance', ascending=False)

# Displaying the top 20 feature importances
top_20_features = feature_importance_xgb.head(20)

# Plotting the top 20 feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_20_features['Feature'], top_20_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances from Default XGBoost Model')
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.show()


# Plotting
plt.figure(figsize=(8, 6))
sns.histplot(df[df['win'] == 0]['gamedatedaymonth'], label='Loss', kde=True, color='blue', alpha=0.6)
sns.histplot(df[df['win'] == 1]['gamedatedaymonth'], label='Win', kde=True, color='orange', alpha=0.6)
plt.xlabel('Game Date Month')
plt.ylabel('Count')
plt.title('Histogram: Game Date Month vs Home Wins/Losses')
plt.legend()
plt.show()


df['gamedatedaymonth'].value_counts()


# First sort dataframe by time


df.head().T


df.drop(columns=['week', 
                 'playid', 
                 'presnaphometeamwinprobability', 
                 'presnapvisitorteamwinprobability', 
                 'hometeamwinprobabilityadded', 
                 'visitorteamwinprobilityadded',
                 'gamedateday',
                 'gamedatedaymonth',
                 'gametimeeasternminute',
                 'gametimeeasternhour'], inplace=True)


# Shift the yards gain 5 times
df['yards_gained_minus_one'] = df['yardsgained'].shift(1)
df['yards_gained_minus_two'] = df['yardsgained'].shift(2)
df['yards_gained_minus_three'] = df['yardsgained'].shift(3)
df['yards_gained_minus_four'] = df['yardsgained'].shift(4)
df['yards_gained_minus_five'] = df['yardsgained'].shift(5)


# Fill the NaN in the new columns with 0.0 since there isn't any previous info
df['yards_gained_minus_one'] = df['yards_gained_minus_one'].fillna(0.0)
df['yards_gained_minus_two'] = df['yards_gained_minus_two'].fillna(0.0)
df['yards_gained_minus_three'] = df['yards_gained_minus_three'].fillna(0.0)
df['yards_gained_minus_four'] = df['yards_gained_minus_four'].fillna(0.0)
df['yards_gained_minus_five'] = df['yards_gained_minus_five'].fillna(0.0) 


# Remove some of the columns with values that are almost continuous 
df.drop(columns=['receiveralignment_3x3', 'pff_runconceptsecondary_cross_lead_lead_qb_runs',
                'pff_runconceptsecondary_inverted_read_option_speed_option'], inplace=True)


# Export to csv
df.to_csv('nfl_df_one.csv', index=False)


df.head().T


X = df.drop(columns=['win'], axis=1)
y = df['win']


# Split the data into training and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


X.shape


y.shape


xgb_model_two = XGBClassifier()
xgb_model_two.fit(X_train, y_train);
# Predict on the test data
y_pred = xgb_model_two.predict(X_test)

# Convert probabilities to binary outcomes (for binary classification)
y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

# Get probabilities for the positive class
y_pred_prob = xgb_model_two.predict_proba(X_test)[:, 1]

# Evaluation
eval_metric(xgb_model_two, X_train, y_train, X_test, y_test)


# Get the feature importances
importance = xgb_model_two.feature_importances_

# Create a DataFrame for feature importance
feature_importance_xgb = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})

# Sort the feature importances
feature_importance_xgb = feature_importance_xgb.sort_values(by='Importance', ascending=False)

# Displaying the top 20 feature importances
top_20_features = feature_importance_xgb.head(20)

# Plotting the top 20 feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_20_features['Feature'], top_20_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances from Default XGBoost Model')
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.show()


# Define the model
xgb = XGBClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Perform Grid Search with the correct estimator and parameters
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, scoring='precision')
grid_search.fit(X_train, y_train)

# Display results
print("Best parameters found: ", grid_search.best_params_)
print("Best precision found: ", grid_search.best_score_)

# Explicitly ensure the best model is trained
best_model = grid_search.best_estimator_

# Now, you can use the best model to predict
y_pred = best_model.predict(X_test)

# Generate the predicted probabilities for the positive class
y_pred_prob = best_model.predict_proba(X_test)[:, 1]


from sklearn.metrics import precision_recall_curve
def plot_precision_recall_vs_threshold(y_true, y_probs, title):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    # Calculate F1 Score
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])

    # Plot Precision, Recall, and F1-Score vs Threshold
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label="Precision", color='b')
    plt.plot(thresholds, recall[:-1], label="Recall", color='g')
    plt.plot(thresholds, f1_scores, label="F1 Score", color='orange')
    plt.xlabel("Decision Threshold (Cutoff for Positive Class Probability)")
    plt.ylabel("Score (Precision, Recall, F1)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()

    # Return Results
    return precision, recall, thresholds
    
# Use the function with your best model
precision, recall, thresholds = plot_precision_recall_vs_threshold(y_test, y_pred_prob, "Precision-Recall vs Threshold")


# Run the model with different thresholds
# Update XG Boost with the tune hyperparameters
xgb = XGBClassifier(random_state=42, colsample_bytree= 1.0, learning_rate= 0.01, max_depth= 7, n_estimators= 50, subsample= 0.8)
xgb.fit(X_train, y_train)

# Get predicted probabilities for the positive class
y_probs = xgb.predict_proba(X_test)[:, 1]

# Generate predicted probabilities for the training and test sets
y_train_probs = xgb.predict_proba(X_train)[:, 1]
y_test_probs = xgb.predict_proba(X_test)[:, 1]

# Set custom threshold
threshold = 0.65

# Convert probabilities to binary predictions
y_train_pred = (y_train_probs >= threshold).astype(int)
y_test_pred = (y_test_probs >= threshold).astype(int)

# Classification reports
train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)

print("Training Set Classification Report:")
print(train_report)

print("\nTest Set Classification Report:")
print(test_report)

# AUC scores
train_auc = roc_auc_score(y_train, y_train_probs)
test_auc = roc_auc_score(y_test, y_test_probs)

print(f"Training Set AUC: {train_auc:.3f}")
print(f"Test Set AUC: {test_auc:.3f}")


eval_metric(best_model, X_train, y_train, X_test, y_test)


# Access the best estimator from the grid search and get feature importances
importance = grid_search.best_estimator_.feature_importances_

# Create a DataFrame 
feature_importance_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance})

# Sort features by importance in descending order
feature_importance_xgb = feature_importance_xgb.sort_values(by='Importance', ascending=False)
feature_importance_xgb

# Displaying the top 20 feature importances
top_20_features = feature_importance_xgb.head(20)

# Plotting the top 20 feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_20_features['Feature'], top_20_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances from Tuned XGBoost Model')
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.show()


# Get probability for positive class (1)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Plot the first feature against the predicted probability
plt.figure(figsize=(10, 6))
plt.scatter(X_test['presnapvisitorscore'], y_prob, alpha=0.6)
plt.xlabel('Pre Snap Visitor Score')
plt.ylabel('Predicted Probability of a Win')
plt.title('Pre Snap Visitor Score vs Predicted Probability for a Win')
plt.show()


# Plot the feature Pre against the predicted probability
plt.figure(figsize=(10, 6))
plt.scatter(X_test['playclockatsnap'], y_prob, alpha=0.5)
plt.xlabel('Play Clock at Snap')
plt.ylabel('Predicted Probability of a Win')
plt.title('Pre Snap Visitor Score vs Predicted Probability for a Win')
plt.show()


# Plot the feature against the predicted probability
plt.figure(figsize=(10, 6))
plt.scatter(X_test['quarter'], y_prob, alpha=0.5)
plt.xlabel('Quarter')
plt.ylabel('Predicted Probability of a Win')
plt.title('Quarter vs Predicted Probability for a Win')
plt.show()


# Plot the feature against the predicted probability
plt.figure(figsize=(10, 6))
plt.scatter(X_test['pff_passcoverage_cover_3'], y_prob, alpha=0.5)
plt.xlabel('Pass Coverage 3')
plt.ylabel('Predicted Probability of a Win')
plt.title('Pass Coverage 3 vs Predicted Probability for a Win')
plt.show()


# Specify the feature names (as strings)
features = ['presnapvisitorscore', 'presnaphomescore', 'playclockatsnap', 'quarter', 'yardlinenumber',
           'absoluteyardlinenumber', 'gameclock_hour', 'yardsgained', 'expectedpoints']

# Create a figure with increased size
fig, ax = plt.subplots(figsize=(18, 12)) 

# Create the ICE plot with 'kind='both'' to show both PDP and ICE
PartialDependenceDisplay.from_estimator(best_model, X_test, features, kind='both', ax=ax)

# Show the plot
plt.show()


# Plotting
plt.figure(figsize=(8, 6))
sns.histplot(df[df['win'] == 0]['presnapvisitorscore'], label='Loss', kde=True, color='blue', alpha=0.6)
sns.histplot(df[df['win'] == 1]['presnapvisitorscore'], label='Win', kde=True, color='red', alpha=0.6)
plt.xlabel('Pre Snap Visitor Score')
plt.ylabel('Count')
plt.title('Histogram: Game Date Day vs Home Wins/Losses')
plt.legend()
plt.show()


# Plotting
plt.figure(figsize=(8, 6))
sns.histplot(df[df['win'] == 0]['playclockatsnap'], label='Loss', kde=True, color='blue', alpha=0.6)
sns.histplot(df[df['win'] == 1]['playclockatsnap'], label='Win', kde=True, color='yellow', alpha=0.6)
plt.xlabel('Play Clock at Snap')
plt.ylabel('Count')
plt.title('Histogram: Game Date Day vs Home Wins/Losses')
plt.legend()
plt.show()


# Plotting
plt.figure(figsize=(8, 6))
sns.histplot(df[df['win'] == 0]['presnaphomescore'], label='Loss', kde=True, color='blue', alpha=0.6)
sns.histplot(df[df['win'] == 1]['presnaphomescore'], label='Win', kde=True, color='orange', alpha=0.6)
plt.xlabel('Pre Snap Home Score')
plt.ylabel('Count')
plt.title('Histogram: Pre Snap Home Score vs Home Wins/Losses')
plt.legend()
plt.show()


# Plotting
plt.figure(figsize=(8, 6))
sns.histplot(df[df['win'] == 0]['quarter'], label='Loss', kde=True, color='blue', alpha=0.6)
sns.histplot(df[df['win'] == 1]['quarter'], label='Win', kde=True, color='red', alpha=0.6)
plt.xlabel('Quarter')
plt.ylabel('Count')
plt.title('Histogram: Quarter vs Home Wins/Losses')
plt.legend()
plt.show()


# Plotting
plt.figure(figsize=(8, 6))
sns.histplot(df[df['win'] == 0]['offensiveform_jumbo'], label='Loss', kde=True, color='blue', alpha=0.6)
sns.histplot(df[df['win'] == 1]['offensiveform_jumbo'], label='Win', kde=True, color='yellow', alpha=0.6)
plt.xlabel('Offensive Form Jumbo')
plt.ylabel('Count')
plt.title('Histogram: Offensive Form Jumbo vs Home Wins/Losses')
plt.legend()
plt.show()


!python --version




