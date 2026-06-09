!pip install kaggle


from google.colab import files  # if you're on Colab
files.upload()  # Choose the 'kaggle.json' file you downloaded from Kaggle


!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json


!kaggle competitions download -c march-machine-learning-mania-2025


!unzip march-machine-learning-mania-2025.zip -d march_data


!ls


# # test
import pandas as pd
cities = pd.read_csv("march_data/Cities.csv")
cities.head()


import pandas as pd
import os

# Define the directory containing the CSV files
directory = "march_data/"

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # Check if the file is a CSV
        file_path = os.path.join(directory, filename)
        print(f"Processing: {filename}")
        try:
            # Read the CSV file
            data = pd.read_csv(file_path)
            # Print the head of the dataset
            print(data.head())
        except Exception as e:
            print(f"Error reading {filename}: {e}")


import pandas as pd
import os

# Define the directory containing the CSV files
directory = "march_data/"

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # Check if the file is a CSV
        file_path = os.path.join(directory, filename)
        try:
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Select numerical type columns
            numerical_columns = data.select_dtypes(include=["number"]).columns

            # Only print numerical columns if they exist
            if len(numerical_columns) > 0:
                print(f"{filename}: {list(numerical_columns)}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")


# # test
import pandas as pd
cities = pd.read_csv("/content/march_data/SampleSubmissionStage1.csv")
cities.head()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# Load main files
df_seeds = pd.read_csv('march_data/MNCAATourneySeeds.csv')
df_results = pd.read_csv('march_data/MNCAATourneyCompactResults.csv')
df_teams = pd.read_csv('march_data/MTeams.csv')
df_regular = pd.read_csv('march_data/MRegularSeasonDetailedResults.csv')
df_ranks = pd.read_csv('march_data/MMasseyOrdinals.csv')

# Clean seed to extract seed number
df_seeds['SeedNum'] = df_seeds['Seed'].str.extract('(\d+)').astype(int)


# Winning stats
win_stats = df_regular.groupby(['Season', 'WTeamID']).agg({
    'WScore': 'mean', 'WFGM': 'mean', 'WFGA': 'mean', 'WFGM3': 'mean', 'WFGA3': 'mean',
    'WFTM': 'mean', 'WFTA': 'mean', 'WAst': 'mean', 'WTO': 'mean', 'WStl': 'mean', 'WBlk': 'mean'
}).reset_index()
win_stats.columns = ['Season', 'TeamID'] + ['Win_' + col for col in win_stats.columns[2:]]

# Losing stats
lose_stats = df_regular.groupby(['Season', 'LTeamID']).agg({
    'LScore': 'mean', 'LFGM': 'mean', 'LFGA': 'mean', 'LFGM3': 'mean', 'LFGA3': 'mean',
    'LFTM': 'mean', 'LFTA': 'mean', 'LAst': 'mean', 'LTO': 'mean', 'LStl': 'mean', 'LBlk': 'mean'
}).reset_index()
lose_stats.columns = ['Season', 'TeamID'] + ['Lose_' + col for col in lose_stats.columns[2:]]

# Combine both
team_stats = pd.merge(win_stats, lose_stats, on=['Season', 'TeamID'], how='outer')
team_stats = team_stats.fillna(0)  # Fill NaNs with 0


rankings = df_ranks[df_ranks['RankingDayNum'] == 133]
avg_ranks = rankings.groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()
avg_ranks.columns = ['Season', 'TeamID', 'MeanOrdinalRank']


def make_features(row):
    season = row['Season']
    team1 = min(row['WTeamID'], row['LTeamID'])
    team2 = max(row['WTeamID'], row['LTeamID'])
    team1_wins = row['WTeamID'] == team1
    return pd.Series([season, team1, team2, 1 if team1_wins else 0])

# Convert tournament results to training format
Xy = df_results.apply(make_features, axis=1)
Xy.columns = ['Season', 'Team1', 'Team2', 'Result']


def add_team_features(df, team_col, prefix):
    df_ = df.copy()

    # Merge seeds
    temp = df_seeds.rename(columns={'TeamID': team_col, 'SeedNum': f'{prefix}Seed'})
    df_ = df_.merge(temp[['Season', team_col, f'{prefix}Seed']], on=['Season', team_col], how='left')

    # Merge stats
    temp = team_stats.rename(columns={'TeamID': team_col})
    df_ = df_.merge(temp, on=['Season', team_col], how='left', suffixes=('', f'_{prefix}_stat'))

    # Merge ranks
    temp = avg_ranks.rename(columns={'TeamID': team_col, 'MeanOrdinalRank': f'{prefix}Rank'})
    df_ = df_.merge(temp[['Season', team_col, f'{prefix}Rank']], on=['Season', team_col], how='left')

    return df_


X = add_team_features(Xy, 'Team1', 'A')
X = add_team_features(X, 'Team2', 'B')


X['SeedDiff'] = X['ASeed'] - X['BSeed']
X['RankDiff'] = X['BRank'] - X['ARank']

features = ['SeedDiff', 'RankDiff']
X_final = X[features]
y = X['Result']


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_final, y, test_size=0.3, random_state=42)

# Build and train Random Forest model
model1 = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model1.fit(X_train, y_train)

# Predict probabilities
preds = model1.predict_proba(X_val)[:, 1]

# Evaluate with log loss
loss = log_loss(y_val, preds)
print("Validation Log Loss (Random Forest):", loss)


from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced']
}

model2 = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(model2, param_grid, n_iter=50, cv=3, scoring='neg_log_loss', n_jobs=-1)
search.fit(X_train, y_train)

print("Best Params:", search.best_params_)
model2 = search.best_estimator_


from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix

# Predict probabilities for the positive class on the validation set
preds_prob = model2.predict_proba(X_val)[:, 1]
validation_log_loss = log_loss(y_val, preds_prob)
print("Validation Log Loss:", validation_log_loss)

# In addition, get the class predictions
preds_class = model2.predict(X_val)

# Evaluate accuracy
accuracy = accuracy_score(y_val, preds_class)
print("Validation Accuracy:", accuracy)

# Print the classification report to see precision, recall and f1-score
print("Classification Report:")
print(classification_report(y_val, preds_class))

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, preds_class))


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# Fill missing values
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X_final)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# Perform PCA with n_components set to the maximum allowed (min(n_samples, n_features))
n_components = min(X_scaled.shape[1], X_scaled.shape[0])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)


# Train/test split using the PCA-transformed data
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model3 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model3.fit(X_train, y_train)

# Predict probabilities and evaluate log loss
preds = model3.predict_proba(X_val)[:, 1]
print("Log Loss with PCA:", log_loss(y_val, preds))


# 1. Scatter Plot of the First Two Principal Components
if X_pca.shape[1] >= 2:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter Plot of the First Two PCA Components')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Target Variable')
    plt.show()
else:
    print("Not enough components to create a scatter plot (at least 2 required).")

# 2. Plotting the Explained Variance Ratio for Each Component
explained_variance = pca.explained_variance_ratio_
components = np.arange(1, len(explained_variance) + 1)

plt.figure(figsize=(8, 6))
plt.bar(components, explained_variance, color='skyblue', alpha=0.7, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Each Principal Component')
plt.xticks(components)
plt.show()

# 3. Plotting the Cumulative Explained Variance
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8, 6))
plt.step(components, cumulative_variance, where='mid', color='red')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xticks(components)
plt.ylim([0, 1.05])
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Probabilities
preds1_prob = model1.predict_proba(X_val)[:, 1]
preds2_prob = model2.predict_proba(X_val)[:, 1]
preds3_prob = model3.predict_proba(X_val)[:, 1]

# Class predictions
preds1_class = model1.predict(X_val)
preds2_class = model2.predict(X_val)
preds3_class = model3.predict(X_val)


fpr1, tpr1, _ = roc_curve(y_val, preds1_prob)
fpr2, tpr2, _ = roc_curve(y_val, preds2_prob)
fpr3, tpr3, _ = roc_curve(y_val, preds3_prob)

auc1 = auc(fpr1, tpr1)
auc2 = auc(fpr2, tpr2)
auc3 = auc(fpr3, tpr3)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f'Model 1 (AUC = {auc1:.3f})')
plt.plot(fpr2, tpr2, label=f'Model 2 (AUC = {auc2:.3f})')
plt.plot(fpr3, tpr3, label=f'Model 3 (AUC = {auc3:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

cm1 = confusion_matrix(y_val, preds1_class)
cm2 = confusion_matrix(y_val, preds2_class)
cm3 = confusion_matrix(y_val, preds3_class)

plot_confusion_matrix(cm1, "Model 1 - Confusion Matrix")
plot_confusion_matrix(cm2, "Model 2 - Confusion Matrix")
plot_confusion_matrix(cm3, "Model 3 - Confusion Matrix")


plt.figure(figsize=(8, 6))
for preds, label in zip(
    [preds1_prob, preds2_prob, preds3_prob],
    ['Model 1', 'Model 2', 'Model 3']
):
    prob_true, prob_pred = calibration_curve(y_val, preds, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=label)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves')
plt.legend()
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# After training
importances = model1.feature_importances_
features = X_final.columns

# Create DataFrame and sort
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance Score')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.show()


import seaborn as sns

corr = X_final.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()



from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
rfe.fit(X_final, y)

selected = X_final.columns[rfe.support_]
print("Top selected features by RFE:", selected.tolist())


test_data = pd.read_csv("march_data/SampleSubmissionStage1.csv")
X_test = test_data


print(X_test.columns)


# First, let's see what test data we're working with
print("Current X_test:")
print(X_test.head())

# We need to create a proper test dataset with team matchups
# Option 1: If you're trying to predict all possible matchups for a tournament

def create_tournament_matchups(season, team_ids):
    """
    Create all possible matchups between teams in a tournament

    Parameters:
    season (int): The season year (e.g., 2024)
    team_ids (list): List of team IDs in the tournament

    Returns:
    DataFrame with all possible matchups
    """
    matchups = []
    for i in range(len(team_ids)):
        for j in range(i+1, len(team_ids)):
            team1 = min(team_ids[i], team_ids[j])
            team2 = max(team_ids[i], team_ids[j])
            matchups.append({
                'Season': season,
                'Team1': team1,
                'Team2': team2
            })
    return pd.DataFrame(matchups)

# Option 2: If you have a specific bracket structure to predict

def create_bracket_matchups(season, bracket_matchups):
    """
    Create matchups from a specific bracket structure

    Parameters:
    season (int): The season year
    bracket_matchups (list): List of tuples with (team1_id, team2_id)

    Returns:
    DataFrame with the specified matchups
    """
    matchups = []
    for team1, team2 in bracket_matchups:
        t1 = min(team1, team2)
        t2 = max(team1, team2)
        matchups.append({
            'Season': season,
            'Team1': t1,
            'Team2': t2
        })
    return pd.DataFrame(matchups)

# Example: Create test data for the current season
# Get the current season from X_test
current_season = X_test['Season'].iloc[0] if not X_test.empty else 2024

# Option 1: Create test data from all tournament teams in the current season
# First, get the tournament teams for the current season
tournament_teams = df_seeds[df_seeds['Season'] == current_season]['TeamID'].unique().tolist()
if tournament_teams:
    print(f"Found {len(tournament_teams)} tournament teams for season {current_season}")
    # Create all possible matchups
    X_test_new = create_tournament_matchups(current_season, tournament_teams)
else:
    print(f"No tournament teams found for season {current_season}")
    # If we can't find tournament teams, create a sample test set with previous tournament teams
    # Get the most recent season with tournament teams
    recent_season = df_seeds['Season'].max()
    tournament_teams = df_seeds[df_seeds['Season'] == recent_season]['TeamID'].unique().tolist()
    print(f"Using {len(tournament_teams)} teams from season {recent_season} as a sample")
    # Create test data with the recent season's teams but the current season
    X_test_new = create_tournament_matchups(current_season, tournament_teams)

print(f"Created {len(X_test_new)} matchups for testing")
print(X_test_new.head())

# Now add the features
X_test_new = add_team_features(X_test_new, 'Team1', 'A')
X_test_new = add_team_features(X_test_new, 'Team2', 'B')

# Calculate the feature differences
X_test_new['SeedDiff'] = X_test_new['ASeed'] - X_test_new['BSeed']
X_test_new['RankDiff'] = X_test_new['BRank'] - X_test_new['ARank']

# Select only the features used in training
X_test_final = X_test_new[features]

# Now predict
predictions = model2.predict_proba(X_test_final)[:, 1]

# Create a results dataframe
results = X_test_new.copy()
results['Probability'] = predictions
results['PredictedWinner'] = np.where(predictions > 0.5, results['Team1'], results['Team2'])

# Merge with team names for better readability
if 'df_teams' in globals():
    results = results.merge(
        df_teams[['TeamID', 'TeamName']],
        left_on='Team1',
        right_on='TeamID',
        how='left'
    ).rename(columns={'TeamName': 'Team1Name'}).drop('TeamID', axis=1)

    results = results.merge(
        df_teams[['TeamID', 'TeamName']],
        left_on='Team2',
        right_on='TeamID',
        how='left'
    ).rename(columns={'TeamName': 'Team2Name'}).drop('TeamID', axis=1)

# Display the first few predictions
print("\nPrediction Results:")
print(results[['Team1', 'Team1Name', 'Team2', 'Team2Name', 'Probability', 'PredictedWinner']].head())

# Save the results
results.to_csv('tournament_predictions.csv', index=False)
print(f"Saved predictions to tournament_predictions.csv")




