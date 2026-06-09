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
import os

def load_all_dataframes():
    csv_files = [
        "/kaggle/input/march-machine-learning-mania-2025/Conferences.csv",
        "/kaggle/input/march-machine-learning-mania-2025/SeedBenchmarkStage1.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyDetailedResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeedRoundSlots.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MGameCities.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WSecondaryTourneyCompactResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WGameCities.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySlots.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyTeams.csv",
        "/kaggle/input/march-machine-learning-mania-2025/Cities.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MTeamSpellings.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyCompactResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WTeams.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WConferenceTourneyGames.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySlots.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WSeasons.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MTeamCoaches.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MConferenceTourneyGames.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WTeamSpellings.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MTeamConferences.csv",
        "/kaggle/input/march-machine-learning-mania-2025/MTeams.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WTeamConferences.csv",
        "/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv",
        "/kaggle/input/march-machine-learning-mania-2025/WSecondaryTourneyTeams.csv"
    ]
    
    dataframes = {}
    for file_path in csv_files:
        # Get the file name without the extension, e.g., "Conferences"
        df_name = os.path.splitext(os.path.basename(file_path))[0]
        # Read CSV using the cp1252 encoding to avoid UTF-8 decoding errors
        dataframes[df_name] = pd.read_csv(file_path, encoding='cp1252')
    
    return dataframes

# Load all DataFrames into a dictionary
dfs = load_all_dataframes()

# Make each DataFrame available as a global variable for future calculations.
# Note: Dynamically updating globals() like this is convenient but use it with caution.
for name, df in dfs.items():
    globals()[name] = df

# Example usage: now you can directly access each DataFrame by its name.
# print("Conferences DataFrame:")
# print(Conferences.head())

# print("\nSeedBenchmarkStage1 DataFrame:")
# print(SeedBenchmarkStage1.head())

# You now have access to all the DataFrames:
# Conferences, SeedBenchmarkStage1, WNCAATourneyDetailedResults, WRegularSeasonCompactResults,
# MNCAATourneySeedRoundSlots, MRegularSeasonDetailedResults, MNCAATourneyCompactResults,
# MGameCities, WSecondaryTourneyCompactResults, WGameCities, MSeasons, WNCAATourneySlots,
# MSecondaryTourneyTeams, Cities, MTeamSpellings, MRegularSeasonCompactResults, MMasseyOrdinals,
# MSecondaryTourneyCompactResults, WTeams, WConferenceTourneyGames, MNCAATourneySlots,
# MNCAATourneySeeds, WNCAATourneyCompactResults, WSeasons, WNCAATourneySeeds, MTeamCoaches,
# MConferenceTourneyGames, WRegularSeasonDetailedResults, MNCAATourneyDetailedResults,
# WTeamSpellings, MTeamConferences, MTeams, WTeamConferences, SampleSubmissionStage1,
# WSecondaryTourneyTeams.



# Example accessing one of the dataframes
print("\nSeedBenchmarkStage1 DataFrame:")
print(SeedBenchmarkStage1.head())


def display_dataframes(dfs, head_rows=5):
    """Prints the name and first few rows of each DataFrame in the dictionary."""
    for name, df in dfs.items():
        print(f"{name}\n{'-' * 80}\n{df.head(head_rows)}\n")


display_dataframes(dfs)




