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
        file_path = f'{base_path}{filename}'
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

    # Define a color scheme for different categories
    category_colors = {
        "Cities and Locations": "skyblue",
        "Conferences": "lightgreen",
        "Teams": "lightcoral",
        "Tournament Seeds and Slots": "lightyellow",
        "Tournament Games": "lightpink",
        "Regular Season Games": "lightsalmon",
        "Secondary Tournaments": "lightblue",
        "Coaches": "lightgray",
        "Seasons": "lightcyan",
        "Massey Ordinals": "lightgoldenrodyellow",
        "Sample Submissions and Benchmarks": "lightsteelblue"
    }

    # Define a color scheme for Men's and Women's datasets
    gender_colors = {
        'M': 'blue',
        'W': 'red'
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

                # Determine the color based on the first letter of the dataset
                first_letter = dataset[0]
                plot_color = gender_colors.get(first_letter, category_colors[role])

                # Plot histograms for numeric columns
                numeric_columns = df.select_dtypes(include=['number']).columns
                if not numeric_columns.empty:
                    plt.figure(figsize=(15, 10))
                    for col in numeric_columns:
                        sns.histplot(df[col], bins=20, color=plot_color, kde=True, label=col)
                    plt.title(f"Histograms for {dataset}")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")
                    plt.legend()
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
regular_season_results = pd.read_csv(data_directory + "MRegularSeasonCompactResults.csv")
tourney_results = pd.read_csv(data_directory + "MNCAATourneyCompactResults.csv")
seeds = pd.read_csv(data_directory + "MNCAATourneySeeds.csv")
submission = pd.read_csv(data_directory + "SampleSubmissionStage1.csv")

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



import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# Define features and target
X = features.drop(columns=["Season", "TeamID"])  # Drop non-numeric columns
y = (features["WinLossRatio"] > 0.5).astype(int)  # Target: Binary classification (Win rate > 50%)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
''''
# Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False
)
'''
# Train XGBoost Model
model = xgb.XGBClassifier(n_estimators=1000, 
                          learning_rate=0.05, 
                          max_depth=5, 
                          subsample=0.7, 
                          colsample_bytree=0.8, 
                          use_label_encoder=False, 
                          eval_metric="logloss")

#model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=True)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Convert dataset into DMatrix (for better optimization)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dval = xgb.DMatrix(X_val_scaled, label=y_val)

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

# Evaluate Model
val_predictions = xgb_model.predict(dval)
val_logloss = log_loss(y_val, val_predictions)
print(f"Validation Log Loss: {val_logloss:.5f}")

print("Model Training Completed!")


# Ensure correct Season and Team ID formats
submission[["Season", "Team1", "Team2"]] = submission["ID"].str.split("_", expand=True).astype(int)

# Merge feature data for Team1 and Team2
submission_data = submission.merge(features, left_on=["Season", "Team1"], right_on=["Season", "TeamID"], how="left")
submission_data = submission_data.merge(features, left_on=["Season", "Team2"], right_on=["Season", "TeamID"], how="left", suffixes=("_1", "_2"))

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
print("Submission file saved successfully!")

