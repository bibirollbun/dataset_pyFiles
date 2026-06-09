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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss









# Define data folder path
data_folder = "/kaggle/input/march-machine-learning-mania-2025"

# Load all necessary datasets
def load_data():
    data_files = [
        "Cities.csv", "Conferences.csv", "MConferenceTourneyGames.csv", "MGameCities.csv", "MMasseyOrdinals.csv",
        "MNCAATourneyCompactResults.csv", "MNCAATourneyDetailedResults.csv", "MNCAATourneySeedRoundSlots.csv", "MNCAATourneySeeds.csv", "MNCAATourneySlots.csv",
        "MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv", "MSeasons.csv", "MSecondaryTourneyCompactResults.csv", "MSecondaryTourneyTeams.csv",
        "MTeamCoaches.csv", "MTeamConferences.csv", "MTeamSpellings.csv", "MTeams.csv", "SampleSubmissionStage1.csv", "SeedBenchmarkStage1.csv",
        "WConferenceTourneyGames.csv", "WGameCities.csv", "WNCAATourneyCompactResults.csv", "WNCAATourneyDetailedResults.csv", "WNCAATourneySeeds.csv", "WNCAATourneySlots.csv",
        "WRegularSeasonCompactResults.csv", "WRegularSeasonDetailedResults.csv", "WSeasons.csv", "WSecondaryTourneyCompactResults.csv", "WSecondaryTourneyTeams.csv",
        "WTeamConferences.csv", "WTeamSpellings.csv", "WTeams.csv"
    ]
    
    data = {}
    for file in data_files:
        try:
            data[file] = pd.read_csv(f"{data_folder}/{file}", encoding="latin1")
        except UnicodeDecodeError:
            print(f"Encoding issue with {file}, retrying with ISO-8859-1...")
            data[file] = pd.read_csv(f"{data_folder}/{file}", encoding="ISO-8859-1")
    
    return data


matchups = pd.read_csv(f"{data_folder}/SampleSubmissionStage1.csv")
print(matchups.head())



data = load_data()
processed_data = preprocess_data(data)
print(processed_data["Win"].value_counts())  # Should show both 0s and 1s
model, scaler = train_model(processed_data)



def preprocess_data(data):
    df = data["MRegularSeasonCompactResults.csv"].copy()
    
    features = ["Season", "WTeamID", "LTeamID", "WScore", "LScore"]
    df = df[features]

    # Create duplicate rows with swapped teams to balance win/loss labels
    df_win = df.copy()
    df_win["Win"] = 1  # Winner's record
    
    df_loss = df.copy()
    df_loss.rename(columns={"WTeamID": "LTeamID", "LTeamID": "WTeamID"}, inplace=True)
    df_loss["Win"] = 0  # Loser's record

    # Combine the datasets
    df = pd.concat([df_win, df_loss], ignore_index=True)
    df.drop(columns=["WScore", "LScore"], inplace=True)  # Remove score columns

    print("Win column distribution:", df["Win"].value_counts())  # Debugging step

    return df



def train_model(df):
    X = df.drop(columns=["Win"])
    y = df["Win"]

    # Ensure both classes exist
    if len(y.unique()) < 2:
        raise ValueError("Dataset contains only one class. Check data preprocessing!")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    print("Validation Log Loss:", log_loss(y_test, y_pred))

    return model, scaler



#Before running generate_predictions(), print the extracted features:
print(matchups.head())  # Ensure Season, WTeamID, and LTeamID exist




"""

def generate_predictions(model, scaler, matchup_file):
    matchups = pd.read_csv(f"{data_folder}/{matchup_file}")

    # Extract Season, WTeamID, LTeamID from ID
    matchups[["Season", "WTeamID", "LTeamID"]] = matchups["ID"].str.split("_", expand=True).astype(int)

    # Ensure the features match the training data
    matchup_features = matchups[["Season", "WTeamID", "LTeamID"]]
    matchup_features_scaled = scaler.transform(matchup_features)

    # Make predictions
    matchups["Pred"] = model.predict_proba(matchup_features_scaled)[:, 1]

    # Save submission file
    matchups[["ID", "Pred"]].to_csv("submission.csv", index=False)
    print("Submission file generated: submission.csv")
"""

def generate_predictions(model, scaler, matchup_file):
    matchups = pd.read_csv(f"{data_folder}/{matchup_file}")
    
    # Check if "ID" column exists
    if "ID" not in matchups.columns:
        raise ValueError("Error: 'ID' column missing in matchup file!")

    # Extract Season, WTeamID, and LTeamID
    try:
        matchups[["Season", "WTeamID", "LTeamID"]] = matchups["ID"].str.split("_", expand=True).astype(int)
    except ValueError:
        raise ValueError("Error: Check ID column format. Expected 'Season_WTeamID_LTeamID'.")

    # Print first few rows for debugging
    print("Matchups data sample:")
    print(matchups.head())

    # Ensure the features match the training data
    matchup_features = matchups[["Season", "WTeamID", "LTeamID"]]
    matchup_features_scaled = scaler.transform(matchup_features)

    # Make predictions
    matchups["Pred"] = model.predict_proba(matchup_features_scaled)[:, 1]

    # Print prediction sample
    print("Predictions sample:")
    print(matchups[["ID", "Pred"]].head())

    # Save submission file
    matchups[["ID", "Pred"]].to_csv("submission.csv", index=False)
    print("✅ Submission file generated: submission.csv")






data = load_data()
processed_data = preprocess_data(data)
model, scaler = train_model(processed_data)
generate_predictions(model, scaler, "SampleSubmissionStage1.csv")



# Run the pipeline
if __name__ == "__main__":
    data = load_data()
    processed_data = preprocess_data(data)
    model, scaler = train_model(processed_data)
    generate_predictions(model, scaler, "SampleSubmissionStage1.csv")

