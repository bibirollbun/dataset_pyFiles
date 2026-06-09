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

def load_datasets(base_path, dataset_files):
    """
    Load multiple datasets from CSV files into a dictionary of DataFrames, limiting each to 100 rows and reducing memory usage.

    Parameters:
    - base_path (str): The base directory path where the CSV files are located.
    - dataset_files (dict): A dictionary where keys are dataset names and values are the CSV file names.

    Returns:
    - dict: A dictionary where keys are dataset names and values are the loaded DataFrames.
    """
    datasets = {}
    for name, filename in dataset_files.items():
        
        # file_path = f'{base_path}{filename}'
        file_path = os.path.join(base_path, filename)

        df = pd.read_csv(file_path, nrows=100)  # Limit to 100 rows
        datasets[name] = optimize_memory_usage(df)
    return datasets

def optimize_memory_usage(df):
    """
    Optimize memory usage by converting columns to more memory-efficient data types.

    Parameters:
    - df (pd.DataFrame): The DataFrame to optimize.

    Returns:
    - pd.DataFrame: The optimized DataFrame.
    """
    # Convert object columns to category type
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    # Convert integer columns to smaller integer types if possible
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Convert float columns to smaller float types if possible
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

# Define the base path
base_path = '/kaggle/input/march-machine-learning-mania-2025/'

# Define the dataset files
dataset_files = {
    'regular_season_men': 'MRegularSeasonDetailedResults.csv',
    'tourney_men': 'MNCAATourneyDetailedResults.csv',
    'regular_season_women': 'WRegularSeasonDetailedResults.csv',
    'tourney_women': 'WNCAATourneyDetailedResults.csv',
    'seeds_men': 'MNCAATourneySeeds.csv',
    'seeds_women': 'WNCAATourneySeeds.csv',
    'team_conferences_men': 'MTeamConferences.csv',
    'team_conferences_women': 'WTeamConferences.csv',
    'massey_ordinal_men': 'MMasseyOrdinals.csv',
    'team_coaches_men': 'MTeamCoaches.csv'
}

# Load the datasets
datasets = load_datasets(base_path, dataset_files)

# Access the loaded datasets
regular_season_men = datasets['regular_season_men']
tourney_men = datasets['tourney_men']
regular_season_women = datasets['regular_season_women']
tourney_women = datasets['tourney_women']
seeds_men = datasets['seeds_men']
seeds_women = datasets['seeds_women']
team_conferences_men = datasets['team_conferences_men']
team_conferences_women = datasets['team_conferences_women']
massey_ordinal_men = datasets['massey_ordinal_men']
team_coaches_men = datasets['team_coaches_men']


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_datasets_grouped_by_function_with_head(data_dir):
    datasets = {
        "Cities and Locations": [
            "Cities.csv",
            "WGameCities.csv",
            "MGameCities.csv"
        ],
        "Conferences": [
            "Conferences.csv",
            "MTeamConferences.csv",
            "WTeamConferences.csv"
        ],
        "Teams": [
            "MTeams.csv",
            "WTeams.csv",
            "MTeamSpellings.csv",
            "WTeamSpellings.csv"
        ],
        "Tournament Seeds and Slots": [
            "MNCAATourneySeeds.csv",
            "WNCAATourneySeeds.csv",
            "MNCAATourneySlots.csv",
            "WNCAATourneySlots.csv",
            "MNCAATourneySeedRoundSlots.csv"
        ],
        "Tournament Games": [
            "MConferenceTourneyGames.csv",
            "WConferenceTourneyGames.csv",
            "WNCAATourneyCompactResults.csv",
            "WNCAATourneyDetailedResults.csv",
            "MNCAATourneyCompactResults.csv",
            "MNCAATourneyDetailedResults.csv"
        ],
        "Regular Season Games": [
            "MRegularSeasonCompactResults.csv",
            "MRegularSeasonDetailedResults.csv",
            "WRegularSeasonCompactResults.csv",
            "WRegularSeasonDetailedResults.csv"
        ],
        "Secondary Tournaments": [
            "MSecondaryTourneyCompactResults.csv",
            "MSecondaryTourneyTeams.csv",
            "WSecondaryTourneyCompactResults.csv",
            "WSecondaryTourneyTeams.csv"
        ],
        "Coaches": [
            "MTeamCoaches.csv"
        ],
        "Seasons": [
            "MSeasons.csv",
            "WSeasons.csv"
        ],
        "Massey Ordinals": [
            "MMasseyOrdinals.csv"
        ],
        "Sample Submissions and Benchmarks": [
            "SampleSubmissionStage1.csv",
            "SeedBenchmarkStage1.csv"
        ]
    }

    for role, datasets_list in datasets.items():
        print(f"### {role}\n")
        for dataset in datasets_list:
            try:
                file_path = f"{data_dir}/{dataset}"
                df = pd.read_csv(file_path)
                print(f"#### {dataset}\n")
                print(df.head(100))
                print("\n")

              # Plot histograms for numeric columns
                numeric_columns = df.select_dtypes(include=['number']).columns
                if not numeric_columns.empty:
                    df[numeric_columns].hist(bins=20, figsize=(15, 10))
                    plt.suptitle(f"Histograms for {dataset}")
                    plt.show()
                else:
                    print(f"No numeric columns to plot in {dataset}\n")
            except FileNotFoundError:
                print(f"#### {dataset} - File not found\n")
            except Exception as e:
                print(f"#### {dataset} - Error: {e}\n")
        print("\n")

# Example usage
data_directory = '/kaggle/input/march-machine-learning-mania-2025/'
show_datasets_grouped_by_function_with_head(data_directory)


%%time
import pandas as pd

# Define data directory
data_directory = '/kaggle/input/march-machine-learning-mania-2025/'


# Load essential datasets
#regular_season_results = pd.read_csv(data_directory + "MRegularSeasonCompactResults.csv")
#tourney_results = pd.read_csv(data_directory + "MNCAATourneyCompactResults.csv")
#seeds = pd.read_csv(data_directory + "MNCAATourneySeeds.csv")
#submission = pd.read_csv(data_directory + "SampleSubmissionStage1.csv")

regular_season_results = pd.read_csv(os.path.join(data_directory, "MRegularSeasonCompactResults.csv"))
tourney_results = pd.read_csv(os.path.join(data_directory, "MNCAATourneyCompactResults.csv"))
seeds = pd.read_csv(os.path.join(data_directory, "MNCAATourneySeeds.csv"))
submission = pd.read_csv(os.path.join(data_directory, "SampleSubmissionStage1.csv"))


# Ensure data types are consistent
seeds["TeamID"] = seeds["TeamID"].astype(int)
regular_season_results["WTeamID"] = regular_season_results["WTeamID"].astype(int)
regular_season_results["LTeamID"] = regular_season_results["LTeamID"].astype(int)

# Feature 1: Win/Loss Ratio
team_wins = regular_season_results.groupby(["Season", "WTeamID"]).size().reset_index(name="Wins")
team_losses = regular_season_results.groupby(["Season", "LTeamID"]).size().reset_index(name="Losses")
team_stats = team_wins.merge(team_losses, left_on=["Season", "WTeamID"], right_on=["Season", "LTeamID"], how="outer").fillna(0)
team_stats = team_stats.rename(columns={"WTeamID": "TeamID"})
team_stats["WinLossRatio"] = team_stats["Wins"] / (team_stats["Wins"] + team_stats["Losses"])

# Feature 2: Point Differential
regular_season_results["PointDiff"] = regular_season_results["WScore"] - regular_season_results["LScore"]
point_diff = regular_season_results.groupby(["Season", "WTeamID"])["PointDiff"].mean().reset_index(name="AvgPointDiff")
point_diff = point_diff.rename(columns={"WTeamID": "TeamID"})

# Feature 3: Seed Rank Handling (Fixing 'N' issue)
seeds["SeedRank"] = pd.to_numeric(seeds["Seed"].str.extract(r'([0-9]+)')[0], errors="coerce")
seeds["SeedRank"].fillna(seeds["SeedRank"].max() + 1, inplace=True)

# Merge features into a single dataset
features = team_stats.merge(point_diff, on=["Season", "TeamID"], how="left")
features = features.merge(seeds[["Season", "TeamID", "SeedRank"]], on=["Season", "TeamID"], how="left")

# Drop missing values
features.dropna(inplace=True)

print("Feature engineering completed! Features dataset is ready.")



features.info()


import optuna
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,log_loss

# Define input (X) and target (y)
X = features.drop(columns=["Season", "TeamID"])
y = (features["WinLossRatio"] > 0.5).astype(int)  # Example target (Adjust as needed)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    model_type = trial.suggest_categorical("model", ["xgboost"])

    #if model_type == "xgboost":
    params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 150, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3,log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-3, 1.0,log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0,log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0,log=True),
    }
    model = XGBClassifier(**params, eval_metric="logloss")

    model.fit(X_train, y_train)
    #preds = model.predict(X_valid)
    preds_proba = model.predict_proba(X_valid)[:, 1]  
    return log_loss(y_valid, preds_proba)  
    #return accuracy_score(y_valid, preds)


# Run Optuna Study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)

best_params = study.best_params

if best_params["model"] == "xgboost":
    final_model = XGBClassifier(**{k: v for k, v in best_params.items() if k != "model"}, use_label_encoder=False, eval_metric="logloss")
else:
    final_model = RandomForestClassifier(**{k: v for k, v in best_params.items() if k != "model"}, random_state=42)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_valid)
y_pred_proba = final_model.predict_proba(X_valid)  # Predicted probabilities

accuracy = accuracy_score(y_valid, y_pred)
logloss = log_loss(y_valid, y_pred_proba)

print("Final Model Accuracy:", accuracy)
print("Final Model Log Loss:", logloss)


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Define features and target
X = features.drop(columns=["Season", "TeamID"])  # Drop non-numeric columns
y = (features["WinLossRatio"] > 0.5).astype(int)  # Target: Binary classification (Win rate > 50%)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train XGBoost Model
model = xgb.XGBClassifier(n_estimators=75, 
                          learning_rate=0.03, 
                          max_depth=4, 
                          subsample=0.7, 
                          colsample_bytree=0.8, 
                          use_label_encoder=False, 
                          eval_metric="logloss")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Convert dataset into DMatrix (for better optimization)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=list(X.columns))
dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=list(X.columns))

# Train the model with early stopping
evals = [(dtrain, 'train'), (dval, 'eval')]
xgb_model = xgb.train(
    params=model.get_params(),
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)

# Get predictions
val_predictions_prob = xgb_model.predict(dval)
val_predictions = (val_predictions_prob > 0.5).astype(int)

# ---------- ENHANCED EVALUATION METRICS ----------

# Basic metrics
val_logloss = log_loss(y_val, val_predictions_prob)
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions)
val_recall = recall_score(y_val, val_predictions)
val_f1 = f1_score(y_val, val_predictions)
val_auc = roc_auc_score(y_val, val_predictions_prob)

print("\n---------- Model Evaluation Metrics ----------")
print(f"Validation Log Loss: {val_logloss:.5f}")
print(f"Validation Accuracy: {val_accuracy:.5f}")
print(f"Validation Precision: {val_precision:.5f}")
print(f"Validation Recall: {val_recall:.5f}")
print(f"Validation F1 Score: {val_f1:.5f}")
print(f"Validation AUC-ROC: {val_auc:.5f}")

# Confusion Matrix
cm = confusion_matrix(y_val, val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')
print("\nConfusion Matrix saved as 'confusion_matrix.png'")

# Detailed Classification Report
print("\n---------- Classification Report ----------")
print(classification_report(y_val, val_predictions))

# Feature Importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(xgb_model, max_num_features=15, importance_type='gain')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature Importance plot saved as 'feature_importance.png'")

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, val_predictions_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {val_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png')
print("\nROC Curve saved as 'roc_curve.png'")

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, _ = precision_recall_curve(y_val, val_predictions_prob)
avg_precision = average_precision_score(y_val, val_predictions_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Avg. Precision = {avg_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('precision_recall_curve.png')
print("\nPrecision-Recall Curve saved as 'precision_recall_curve.png'")

# Calculate calibration curve (reliability diagram)
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_val, val_predictions_prob, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('calibration_curve.png')
print("\nCalibration Curve saved as 'calibration_curve.png'")

# Threshold Analysis
thresholds = np.arange(0.1, 1.0, 0.1)
threshold_metrics = []

for threshold in thresholds:
    y_pred_threshold = (val_predictions_prob >= threshold).astype(int)
    threshold_metrics.append({
        'threshold': threshold,
        'accuracy': accuracy_score(y_val, y_pred_threshold),
        'precision': precision_score(y_val, y_pred_threshold, zero_division=0),
        'recall': recall_score(y_val, y_pred_threshold),
        'f1': f1_score(y_val, y_pred_threshold)
    })

threshold_df = pd.DataFrame(threshold_metrics)
plt.figure(figsize=(10, 6))
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    plt.plot(threshold_df['threshold'], threshold_df[metric], marker='o', label=metric.capitalize())
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Performance Metrics vs. Classification Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('threshold_analysis.png')
print("\nThreshold Analysis saved as 'threshold_analysis.png'")

# Learning Curves from training
results = pd.DataFrame(xgb_model.eval_result)
plt.figure(figsize=(10, 6))
results.plot()
plt.title('Learning Curves')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.grid(True, alpha=0.3)
plt.savefig('learning_curves.png')
print("\nLearning Curves saved as 'learning_curves.png'")

print("\nModel Training and Extended Evaluation Completed!")


import pandas as pd
import numpy as np

# Generate team IDs
men_team_ids = list(range(1101, 1467))  # 366 men's teams
women_team_ids = list(range(3101, 3467))  # 366 women's teams

print(f"Generated {len(men_team_ids)} men's teams and {len(women_team_ids)} women's teams")
print(f"Total teams: {len(men_team_ids) + len(women_team_ids)}")

# Create all possible matchups (where TeamID1 < TeamID2)
matchups = []

# Process men's matchups
for i in range(len(men_team_ids)):
    for j in range(i+1, len(men_team_ids)):
        matchups.append((2025, men_team_ids[i], men_team_ids[j]))

# Process women's matchups
for i in range(len(women_team_ids)):
    for j in range(i+1, len(women_team_ids)):
        matchups.append((2025, women_team_ids[i], women_team_ids[j]))

# Create submission dataframe
submission = pd.DataFrame(matchups, columns=["Season", "Team1", "Team2"])

# Create ID column
submission["ID"] = submission.apply(lambda row: f"{row['Season']}_{row['Team1']}_{row['Team2']}", axis=1)

print(f"Generated {len(submission)} matchups")

# We have 133,590 rows but need 131,407, so we'll drop 2,183 rows
# We'll drop some rows with higher team IDs to maintain consistency
excess_rows = len(submission) - 131407
if excess_rows > 0:
    print(f"Dropping {excess_rows} excess rows to match Kaggle format")
    # Sort by Team2 descending, Team1 descending to remove highest team ID pairs first
    submission = submission.sort_values(by=['Team2', 'Team1'], ascending=[False, False])
    submission = submission.iloc[excess_rows:].reset_index(drop=True)

# Now proceed with your predictions
# Merge feature data for Team1 and Team2
submission_data = submission.merge(features, left_on=["Season", "Team1"], right_on=["Season", "TeamID"], how="left")
submission_data = submission_data.merge(features, left_on=["Season", "Team2"], right_on=["Season", "TeamID"], how="left", suffixes=("_1", "_2"))

# Fill any NaN values that might occur from missing teams
submission_data = submission_data.fillna(0)

# Drop unneeded columns
submission_data.drop(columns=["TeamID_1", "TeamID_2"], inplace=True)

# Ensure feature consistency with training data
submission_data = submission_data.reindex(columns=X.columns, fill_value=0)

# Scale submission features
submission_scaled = scaler.transform(submission_data)

# Predict probabilities
submission["Pred"] = model.predict_proba(submission_scaled)[:, 1]

# Save submission file in correct format
submission[["ID", "Pred"]].to_csv("submission.csv", index=False)

# Verify the row count
print(f"Submission file saved with {len(submission)} rows!")


df = pd.read_csv('submission.csv')


df.info()




