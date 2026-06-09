# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Phase 1: Data Exploration and Understanding
import pandas as pd  # Data manipulation and analysis
import numpy as np   # Numerical operations
import matplotlib.pyplot as plt # Basic plotting
import seaborn as sns # Advanced plotting and visualizations
import re # Used for building the final dataset
from scipy.stats import linregress # Used for Elo calculation
from tqdm import tqdm # Used for Elo calculation

# Phase 2 & 3: Feature Engineering, Data Preprocessing, Modeling, and Evaluation
# --- Install cupy (if needed - Kaggle notebooks usually have it) ---
from sklearn.model_selection import KFold, StratifiedKFold # CPU-based splitting for now
from sklearn.metrics import brier_score_loss
try:
    import torch # used for Bounds
    print("torch is already installed.")
except ImportError:
    print("torch is not installed. Installing it now...")
    !pip install torch  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import torch # used for gpu arrays
    print("torch installed successfully.")
try:
    import cupy as cp # used for gpu arrays
    print("cupy is already installed.")
except ImportError:
    print("cupy is not installed. Installing it now...")
    !pip install cupy  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import cupy as cp  # used for gpu arrays
    print("cupy installed successfully.")
try:
    import cudf # used to convert Pandas Dataframes to gpu Dataframes
    print("cudf is already installed.")
except ImportError:
    print("cudf is not installed. Installing it now...")
    !pip install cudf  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import cudf # used to convert Pandas Dataframes to gpu Dataframes
    print("cudf installed successfully.")
try:
    import cuml # used running well known machine learning techniques on the gpu
    print("cuml is already installed.")
except ImportError:
    print("cuml is not installed. Installing it now...")
    !pip install cuml  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import cuml # used running well known machine learning techniques on the gpu
    print("cuml installed successfully.")
try:
    import xgboost as xgb # Gradient Boosting - XGBoost cuML/RAPIDS XGBoost (ensure RAPIDS is installed correctly)
    print("xgboost is already installed.")
except ImportError:
    print("xgboost is not installed. Installing it now...")
    !pip install xgboost  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import xgboost as xgb # Gradient Boosting - XGBoost cuML/RAPIDS XGBoost (ensure RAPIDS is installed correctly)
    print("xgboost installed successfully.")
try:
    import lightgbm as lgb # Gradient Boosting - LightGBM
    print("lightgbm is already installed.")
except ImportError:
    print("lightgbm is not installed. Installing it now...")
    !pip install lightgbm  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import lightgbm as lgb # Gradient Boosting - LightGBM
    print("lightgbm installed successfully.")
try:
    import catboost # Gradient Boosting - CatBoost
    print("catboost is already installed.")
except ImportError:
    print("catboost is not installed. Installing it now...")
    !pip install catboost  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import catboost # Gradient Boosting - CatBoost
    print("catboost installed successfully.")
try:
    import botorch # Bayesian Optimization with BoTorch
    print("botorch is already installed.")
except ImportError:
    print("botorch is not installed. Installing it now...")
    !pip install botorch  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import botorch # Bayesian Optimization with BoTorch
    print("botorch installed successfully.")
try:
    import ax # Ax for A/B Testing
    print("Ax is already installed.")
except ImportError:
    print("Ax is not installed. Installing it now...")
    !pip install ax-platform  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import ax # Bayesian Optimization with BoTorch
    print("Ax installed successfully.")
try:
    import gpytorch # For using ExactMarginalLogLikelihood
    print("gpytorch is already installed.")
except ImportError:
    print("gpytorch is not installed. Installing it now...")
    !pip install gpytorch  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import gpytorch # For using ExactMarginalLogLikelihood
    print("gpytorch installed successfully.")
from catboost import CatBoostClassifier #Baseline CatBoostClassifier
from cuml.linear_model import LogisticRegression as cuMLLogisticRegression # Baseline Logistic Regression Model for GPU training
from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier # Baseline Random Forest Model
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.sampling import SobolQMCNormalSampler
from botorch.exceptions import OptimizationWarning
from ax.service.ax_client import AxClient, ObjectiveProperties # Import AxClient!


# Phase 4 & 5: Prediction and Submission
import itertools # For generating matchups
import zipfile # For creating submission ZIP (if needed)

# Utility Libraries
import os # Operating system interactions
import time # Time tracking and potentially for rate limiting
import warnings

# --- Install chardet (if needed - Kaggle notebooks usually have it) ---
try:
    import chardet
    print("chardet is already installed.")
except ImportError:
    print("chardet is not installed. Installing it now...")
    !pip install chardet  # Use !pip to run pip commands in Kaggle/Colab notebooks
    import chardet
    print("chardet installed successfully.")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# --- Define data directory for Kaggle Input ---
data_dir = '/kaggle/input/march-machine-learning-mania-2025/'  # Assumes your data is in the competition's input directory


def detect_encoding(filepath):
    """
    Detects the encoding of a text file using chardet.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The detected encoding, or None if detection fails.
    """
    with open(filepath, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# --- Data Directory (Adjust if needed for Kaggle or local) ---
# data_dir = '/kaggle/input/march-machine-learning-mania-2025/' # Kaggle Notebook Path
# data_dir = './' # Local path if data is in the same directory

# --- List of CSV file names (from your data loading script) ---
csv_files = [
    'MTeams.csv', 'WTeams.csv', 'MSeasons.csv', 'WSeasons.csv',
    'MNCAATourneySeeds.csv', 'WNCAATourneySeeds.csv',
    'MRegularSeasonCompactResults.csv', 'WRegularSeasonCompactResults.csv',
    'MNCAATourneyCompactResults.csv', 'WNCAATourneyCompactResults.csv',
    'SampleSubmissionStage1.csv','SampleSubmissionStage2.csv',
    'MRegularSeasonDetailedResults.csv', 'WRegularSeasonDetailedResults.csv',
    'MNCAATourneyDetailedResults.csv', 'WNCAATourneyDetailedResults.csv',
    'Cities.csv', 'MGameCities.csv', 'WGameCities.csv',
    'MMasseyOrdinals.csv', 'MTeamCoaches.csv', 'Conferences.csv',
    'MTeamConferences.csv', 'WTeamConferences.csv',
    'MConferenceTourneyGames.csv', 'WConferenceTourneyGames.csv',
    'MSecondaryTourneyTeams.csv', 'WSecondaryTourneyTeams.csv',
    'MSecondaryTourneyCompactResults.csv', 'WSecondaryTourneyCompactResults.csv',
    'MTeamSpellings.csv', 'WTeamSpellings.csv',
    'MNCAATourneySlots.csv', 'WNCAATourneySlots.csv',
    'MNCAATourneySeedRoundSlots.csv'
]

# --- Detect and print encoding for each file ---
detected_encodings = {}
for filename in csv_files:
    filepath = os.path.join(data_dir, filename)
    encoding = detect_encoding(filepath)
    detected_encodings[filename] = encoding
    print(f"Detected encoding for {filename}: {encoding}")

print("\n--- Detected Encodings Summary ---")
for filename, encoding in detected_encodings.items():
    print(f"{filename}: {encoding}")


# --- Loading DataFrames ---
# Data Section 1 - The Basics
m_teams_df = pd.read_csv(os.path.join(data_dir, 'MTeams.csv'), encoding=detected_encodings['MTeams.csv'])
w_teams_df = pd.read_csv(os.path.join(data_dir, 'WTeams.csv'), encoding=detected_encodings['WTeams.csv'])
m_seasons_df = pd.read_csv(os.path.join(data_dir, 'MSeasons.csv'), encoding=detected_encodings['MSeasons.csv'])
w_seasons_df = pd.read_csv(os.path.join(data_dir, 'WSeasons.csv'), encoding=detected_encodings['WSeasons.csv'])
m_ncaa_tourney_seeds_df = pd.read_csv(os.path.join(data_dir, 'MNCAATourneySeeds.csv'), encoding=detected_encodings['MNCAATourneySeeds.csv'])
w_ncaa_tourney_seeds_df = pd.read_csv(os.path.join(data_dir, 'WNCAATourneySeeds.csv'), encoding=detected_encodings['WNCAATourneySeeds.csv'])
m_regular_season_compact_results_df = pd.read_csv(os.path.join(data_dir, 'MRegularSeasonCompactResults.csv'), encoding=detected_encodings['MRegularSeasonCompactResults.csv'])
w_regular_season_compact_results_df = pd.read_csv(os.path.join(data_dir, 'WRegularSeasonCompactResults.csv'), encoding=detected_encodings['WRegularSeasonCompactResults.csv'])
m_ncaa_tourney_compact_results_df = pd.read_csv(os.path.join(data_dir, 'MNCAATourneyCompactResults.csv'), encoding=detected_encodings['MNCAATourneyCompactResults.csv'])
w_ncaa_tourney_compact_results_df = pd.read_csv(os.path.join(data_dir, 'WNCAATourneyCompactResults.csv'), encoding=detected_encodings['WNCAATourneyCompactResults.csv'])
sample_submission_stage_1_df = pd.read_csv(os.path.join(data_dir, 'SampleSubmissionStage1.csv'), encoding=detected_encodings['SampleSubmissionStage1.csv'])
sample_submission_stage_2_df = pd.read_csv(os.path.join(data_dir, 'SampleSubmissionStage2.csv'), encoding=detected_encodings['SampleSubmissionStage2.csv'])

# Data Section 2 - Team Box Scores (Detailed Results)
m_regular_season_detailed_results_df = pd.read_csv(os.path.join(data_dir, 'MRegularSeasonDetailedResults.csv'), encoding=detected_encodings['MRegularSeasonDetailedResults.csv'])
w_regular_season_detailed_results_df = pd.read_csv(os.path.join(data_dir, 'WRegularSeasonDetailedResults.csv'), encoding=detected_encodings['WRegularSeasonDetailedResults.csv'])
m_ncaa_tourney_detailed_results_df = pd.read_csv(os.path.join(data_dir, 'MNCAATourneyDetailedResults.csv'), encoding=detected_encodings['MNCAATourneyDetailedResults.csv'])
w_ncaa_tourney_detailed_results_df = pd.read_csv(os.path.join(data_dir, 'WNCAATourneyDetailedResults.csv'), encoding=detected_encodings['WNCAATourneyDetailedResults.csv'])

# Data Section 3 - Geography
cities_df = pd.read_csv(os.path.join(data_dir, 'Cities.csv'), encoding=detected_encodings['Cities.csv'])
m_game_cities_df = pd.read_csv(os.path.join(data_dir, 'MGameCities.csv'), encoding=detected_encodings['MGameCities.csv'])
w_game_cities_df = pd.read_csv(os.path.join(data_dir, 'WGameCities.csv'), encoding=detected_encodings['WGameCities.csv'])

# Data Section 4 - Public Rankings
m_massey_ordinals_df = pd.read_csv(os.path.join(data_dir, 'MMasseyOrdinals.csv'), encoding=detected_encodings['MMasseyOrdinals.csv'])

# Data Section 5 - Supplements
m_team_coaches_df = pd.read_csv(os.path.join(data_dir, 'MTeamCoaches.csv'), encoding=detected_encodings['MTeamCoaches.csv'])
conferences_df = pd.read_csv(os.path.join(data_dir, 'Conferences.csv'), encoding=detected_encodings['Conferences.csv'])
m_team_conferences_df = pd.read_csv(os.path.join(data_dir, 'MTeamConferences.csv'), encoding=detected_encodings['MTeamConferences.csv'])
w_team_conferences_df = pd.read_csv(os.path.join(data_dir, 'WTeamConferences.csv'), encoding=detected_encodings['WTeamConferences.csv'])
m_conference_tourney_games_df = pd.read_csv(os.path.join(data_dir, 'MConferenceTourneyGames.csv'), encoding=detected_encodings['MConferenceTourneyGames.csv'])
w_conference_tourney_games_df = pd.read_csv(os.path.join(data_dir, 'WConferenceTourneyGames.csv'), encoding=detected_encodings['WConferenceTourneyGames.csv'])
m_secondary_tourney_teams_df = pd.read_csv(os.path.join(data_dir, 'MSecondaryTourneyTeams.csv'), encoding=detected_encodings['MSecondaryTourneyTeams.csv'])
w_secondary_tourney_teams_df = pd.read_csv(os.path.join(data_dir, 'WSecondaryTourneyTeams.csv'), encoding=detected_encodings['WSecondaryTourneyTeams.csv'])
m_secondary_tourney_compact_results_df = pd.read_csv(os.path.join(data_dir, 'MSecondaryTourneyCompactResults.csv'), encoding=detected_encodings['MSecondaryTourneyCompactResults.csv'])
w_secondary_tourney_compact_results_df = pd.read_csv(os.path.join(data_dir, 'WSecondaryTourneyCompactResults.csv'), encoding=detected_encodings['WSecondaryTourneyCompactResults.csv'])
m_team_spellings_df = pd.read_csv(os.path.join(data_dir, 'MTeamSpellings.csv'), encoding=detected_encodings['MTeamSpellings.csv'])
w_team_spellings_df = pd.read_csv(os.path.join(data_dir, 'WTeamSpellings.csv'), encoding=detected_encodings['WTeamSpellings.csv'])
m_ncaa_tourney_slots_df = pd.read_csv(os.path.join(data_dir, 'MNCAATourneySlots.csv'), encoding=detected_encodings['MNCAATourneySlots.csv'])
w_ncaa_tourney_slots_df = pd.read_csv(os.path.join(data_dir, 'WNCAATourneySlots.csv'), encoding=detected_encodings['WNCAATourneySlots.csv'])
m_ncaa_tourney_seed_round_slots_df = pd.read_csv(os.path.join(data_dir, 'MNCAATourneySeedRoundSlots.csv'), encoding=detected_encodings['MNCAATourneySeedRoundSlots.csv'])

print("All data files loaded into pandas DataFrames!")


# --- Display first few rows of a few DataFrames to verify loading ---
print("\n--- MTeams DataFrame (First 5 rows) ---")
print(m_teams_df.head())

print("\n--- WRegularSeasonCompactResults DataFrame (First 5 rows) ---")
print(w_regular_season_compact_results_df.head())

print("\n--- MMasseyOrdinals DataFrame (First 5 rows) ---")
print(m_massey_ordinals_df.head())


print("\n--- Phase 1.3: Basic EDA ---\n")



# --- 1.3.1: MTeams.csv & WTeams.csv Exploration ---
print("\n--- 1.3.1: MTeams.csv & WTeams.csv ---")
print("\n--- MTeams Info ---")
print(m_teams_df.info())
print("\n--- MTeams Head ---")
print(m_teams_df.head())
print("\n--- WTeams Info ---")
print(w_teams_df.info())
print("\n--- WTeams Head ---")
print(w_teams_df.head())


# --- 1.3.2: MSeasons.csv & WSeasons.csv Exploration ---
print("\n--- 1.3.2: MSeasons.csv & WSeasons.csv ---")
print("\n--- MSeasons Info ---")
print(m_seasons_df.info())
print("\n--- MSeasons Head ---")
print(m_seasons_df.head())
print("\n--- WSeasons Info ---")
print(w_seasons_df.info())
print("\n--- WSeasons Head ---")
print(w_seasons_df.head())


# --- 1.3.3: MNCAATourneySeeds.csv & WNCAATourneySeeds.csv Exploration ---
print("\n--- 1.3.3: MNCAATourneySeeds.csv & WNCAATourneySeeds.csv ---")
print("\n--- MNCAATourneySeeds Info ---")
print(m_ncaa_tourney_seeds_df.info())
print("\n--- MNCAATourneySeeds Head ---")
print(m_ncaa_tourney_seeds_df.head())
print("\n--- WNCAATourneySeeds Info ---")
print(w_ncaa_tourney_seeds_df.info())
print("\n--- WNCAATourneySeeds Head ---")
print(w_ncaa_tourney_seeds_df.head())


# --- 1.3.4: MRegularSeasonCompactResults.csv & WRegularSeasonCompactResults.csv Exploration ---
print("\n--- 1.3.4: MRegularSeasonCompactResults.csv & WRegularSeasonCompactResults.csv ---")
print("\n--- MRegularSeasonCompactResults Info ---")
print(m_regular_season_compact_results_df.info())
print("\n--- MRegularSeasonCompactResults Head ---")
print(m_regular_season_compact_results_df.head())
print("\n--- WRegularSeasonCompactResults Info ---")
print(w_regular_season_compact_results_df.info())
print("\n--- WRegularSeasonCompactResults Head ---")
print(w_regular_season_compact_results_df.head())


# --- 1.3.5: MNCAATourneyCompactResults.csv & WNCAATourneyCompactResults.csv Exploration ---
print("\n--- 1.3.5: MNCAATourneyCompactResults.csv & WNCAATourneyCompactResults.csv ---")
print("\n--- MNCAATourneyCompactResults Info ---")
print(m_ncaa_tourney_compact_results_df.info())
print("\n--- MNCAATourneyCompactResults Head ---")
print(m_ncaa_tourney_compact_results_df.head())
print("\n--- WNCAATourneyCompactResults Info ---")
print(w_ncaa_tourney_compact_results_df.info())
print("\n--- WNCAATourneyCompactResults Head ---")
print(w_ncaa_tourney_compact_results_df.head())


# --- 1.3.6: SampleSubmissionStage1.csv Exploration ---
print("\n--- 1.3.6: SampleSubmissionStage1.csv ---")
print("\n--- SampleSubmissionStage1 Info ---")
print(sample_submission_stage_1_df.info())
print("\n--- SampleSubmissionStage1 Head ---")
print(sample_submission_stage_1_df.head())


print("\n--- Phase 1.4: Relationships and Insights EDA ---\n")


# --- 1.4.1: Win/Loss Ratios over Seasons ---
print("\n--- 1.4.1: Win/Loss Ratios over Seasons ---")

def calculate_win_loss_ratio(results_df, teams_df, gender='Men'):
    """Calculates win/loss ratios and overall win percentage for each team."""
    team_stats = []
    for team_id in teams_df['TeamID']:
        wins = ((results_df['WTeamID'] == team_id) | (results_df['LTeamID'] == team_id)).sum() # Total games played
        wins_actual = (results_df['WTeamID'] == team_id).sum() # Games won
        loss_actual = (results_df['LTeamID'] == team_id).sum() # Games lost
        win_percentage = (wins_actual / wins) * 100 if wins > 0 else 0
        team_stats.append({'TeamID': team_id, 'Wins': wins_actual, 'Losses': loss_actual, 'TotalGames': wins, 'WinPercentage': win_percentage})
    team_stats_df = pd.DataFrame(team_stats)
    print(f"\n--- {gender}'s Team Win/Loss Stats (First 10 rows) ---")
    print(team_stats_df.sort_values(by='WinPercentage', ascending=False).head(10))

    plt.figure(figsize=(10, 6))
    sns.histplot(team_stats_df['WinPercentage'], bins=30, kde=True)
    plt.title(f'{gender}\'s Teams Win Percentage Distribution')
    plt.xlabel('Win Percentage')
    plt.ylabel('Frequency')
    plt.show()
    return team_stats_df

men_team_win_loss_df = calculate_win_loss_ratio(m_regular_season_compact_results_df, m_teams_df, gender='Men')
women_team_win_loss_df = calculate_win_loss_ratio(w_regular_season_compact_results_df, w_teams_df, gender='Women')


# --- 1.4.2: Score Distributions and Point Differences ---
print("\n--- 1.4.2: Score Distributions and Point Differences ---")

def plot_score_distributions(results_df, gender='Men'):
    """Plots score distributions and point differences."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(results_df['WScore'], bins=30, kde=True, color='green', label='Winning Score')
    sns.histplot(results_df['LScore'], bins=30, kde=True, color='red', label='Losing Score')
    plt.title(f'{gender}\'s Game Score Distributions')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    point_diff = results_df['WScore'] - results_df['LScore']
    sns.histplot(point_diff, bins=30, kde=True)
    plt.title(f'{gender}\'s Point Difference Distribution')
    plt.xlabel('Point Difference (Winning - Losing)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

plot_score_distributions(m_regular_season_compact_results_df, gender='Men')
plot_score_distributions(w_regular_season_compact_results_df, gender='Women')


# --- 1.4.3: Home/Away/Neutral Advantage ---
print("\n--- 1.4.3: Home/Away/Neutral Advantage ---")

def analyze_location_advantage(results_df, gender='Men'):
    """Analyzes win percentages by location (Home, Away, Neutral)."""
    location_win_rates = results_df['WLoc'].value_counts(normalize=True) * 100
    print(f"\n--- {gender}'s Location Win Rates (%) ---")
    print(location_win_rates)

    plt.figure(figsize=(6, 6))
    location_win_rates.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title(f'{gender}\'s Win Rate by Location')
    plt.xlabel('Location (WLoc)')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=0)
    plt.show()

analyze_location_advantage(m_regular_season_compact_results_df, gender='Men')
analyze_location_advantage(w_regular_season_compact_results_df, gender='Women')


# --- 1.4.4: Seed Performance in Tournaments (Upsets) ---
print("\n--- 1.4.4: Seed Performance in Tournaments (Upsets) ---")

def analyze_seed_performance(tourney_results_df, tourney_seeds_df, gender='Men'):
    """Analyzes seed performance and upset probabilities in tournaments."""

    # Merge seeds into tournament results
    tourney_results_seeds_df = pd.merge(tourney_results_df, tourney_seeds_df, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Seed': 'WSeed'}).drop('TeamID', axis=1)
    tourney_results_seeds_df = pd.merge(tourney_results_seeds_df, tourney_seeds_df, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Seed': 'LSeed'}).drop('TeamID', axis=1)

    # Parse seed to get numeric seed value
    def get_seed_int(seed):
        return int(seed[1:3])

    tourney_results_seeds_df['WSeed_int'] = tourney_results_seeds_df['WSeed'].apply(get_seed_int)
    tourney_results_seeds_df['LSeed_int'] = tourney_results_seeds_df['LSeed'].apply(get_seed_int)

    print(f"\n--- {gender}'s Tournament Seed Performance (Example First 10 rows) ---")
    print(tourney_results_seeds_df.head(10))

    # Upset analysis (lower seed winning against higher seed)
    upsets = tourney_results_seeds_df[tourney_results_seeds_df['WSeed_int'] > tourney_results_seeds_df['LSeed_int']]
    upset_rate = len(upsets) / len(tourney_results_seeds_df) * 100
    print(f"\n--- {gender}'s Overall Upset Rate (Lower Seed Winning): {upset_rate:.2f}% ---")

    # Analyze upset frequency by seed difference (Example: Seed difference of 5 or more)
    seed_diff_upsets = upsets[upsets['WSeed_int'] - upsets['LSeed_int'] >= 5]
    seed_diff_upset_rate = len(seed_diff_upsets) / len(tourney_results_seeds_df) * 100
    print(f"\n--- {gender}'s Upset Rate (Seed Difference >= 5): {seed_diff_upset_rate:.2f}% ---")


    plt.figure(figsize=(12, 6))
    sns.histplot(upsets['WSeed_int'], bins=16, kde=False, color='purple', label='Winning Seed in Upsets')
    sns.histplot(upsets['LSeed_int'], bins=16, kde=False, color='orange', label='Losing Seed in Upsets')
    plt.title(f'{gender}\'s Seed Distribution in Tournament Upsets')
    plt.xlabel('Seed Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


analyze_seed_performance(m_ncaa_tourney_compact_results_df, m_ncaa_tourney_seeds_df, gender='Men')
analyze_seed_performance(w_ncaa_tourney_compact_results_df, w_ncaa_tourney_seeds_df, gender='Women')


# --- 1.4.5: Data Quality Check (Basic - can be extended) ---
print("\n--- 1.4.5: Data Quality Check ---")

def basic_data_quality_check(results_df, gender='Men'):
    """Performs basic data quality checks."""
    print(f"\n--- {gender}'s Data Quality Checks ---")

    # Check for negative scores (should not exist)
    negative_scores = results_df[(results_df['WScore'] < 0) | (results_df['LScore'] < 0)]
    print(f"\n--- Games with Negative Scores: {len(negative_scores)} ---")

    # Check for score inconsistencies (e.g., LScore > WScore - should not happen by definition)
    score_inconsistencies = results_df[results_df['LScore'] > results_df['WScore']]
    print(f"\n--- Games with Score Inconsistencies (LScore > WScore): {len(score_inconsistencies)} ---")

    # Check for duplicate game entries (based on Season, DayNum, WTeamID, LTeamID - assuming these uniquely identify a game)
    duplicate_games = results_df.duplicated(subset=['Season', 'DayNum', 'WTeamID', 'LTeamID'])
    print(f"\n--- Duplicate Game Entries: {duplicate_games.sum()} ---")

basic_data_quality_check(m_regular_season_compact_results_df, gender='Men')
basic_data_quality_check(w_regular_season_compact_results_df, gender='Women')
basic_data_quality_check(m_ncaa_tourney_compact_results_df, gender='Men')
basic_data_quality_check(w_ncaa_tourney_compact_results_df, gender='Women')



print("\n--- EDA Phase 1.3 & 1.4 Completed ---")


# --- Phase 1.5: Feature Ideas Brainstorming - Python Script (Updated with Non-Conference and Late-Season Features) ---

# --- Review of EDA Results (Based on EDA Output provided) ---
#  (This section summarizes key findings from your EDA output for reference)

print("--- Phase 1.5: Feature Ideas Brainstorming (Updated) ---\n")
print("--- Review of EDA Results (Key Findings) ---\n")

# ... (Keep the EDA results review from the previous script - it's still relevant) ...
print("... (EDA Results Review - Same as before) ...") # Placeholder - keep the original EDA review section


# --- Phase 1.5.1: Feature Brainstorming (Updated) ---
print("\n--- Phase 1.5.1: Feature Brainstorming (Updated) ---\n")
print("--- Potential Features Brainstorm (Updated) ---\n")

potential_features = [
    # --- Team Strength Features ---
    {"feature": "Season Win Percentage",
     "description": "Overall regular season win percentage for each team in a given season.",
     "rationale": "Reflects overall team strength and consistency during the season.",
     "data_source": "RegularSeasonCompactResults"},

    {"feature": "Average Points Scored (Season)",
     "description": "Average points scored per game by a team during the regular season.",
     "rationale": "Measures offensive capability.",
     "data_source": "RegularSeasonCompactResults"},

    {"feature": "Average Points Allowed (Season)",
     "description": "Average points allowed per game by a team during the regular season.",
     "rationale": "Measures defensive capability.",
     "data_source": "RegularSeasonCompactResults"},

    {"feature": "Point Differential (Season)",
     "description": "Average point differential per game (Points Scored - Points Allowed) during the regular season.",
     "rationale": "Overall measure of team performance margin.",
     "data_source": "RegularSeasonCompactResults"},

    {"feature": "Ranking System Rank (e.g., Pomeroy Rank)",
     "description": "Team's rank from a selected ranking system (e.g., Pomeroy) at the end of the regular season.",
     "rationale": "External validated measure of team strength.",
     "data_source": "MMasseyOrdinals"},

    # --- Seed Advantage (Tournament Specific - might not be applicable for all matchups initially) ---
    {"feature": "Seed Difference (Tournament Matchup)",
     "description": "Difference in seed values between two teams in a tournament matchup.",
     "rationale": "Reflects perceived pre-tournament strength difference by tournament seeding.",
     "data_source": "NCAATourneySeeds"},

    {"feature": "Historical Seed Win Rate (vs. Seed Range in Round)",
     "description": "Historical win rate of a seed (e.g., #5 seed) against a specific seed range (e.g., #12 seed to #16 seed) in a given tournament round.",
     "rationale": "Captures historical upset probabilities and seed performance patterns.",
     "data_source": "NCAATourneySeeds, NCAATourneyCompactResults"},

    # --- Game Location Feature ---
    {"feature": "Home Court Advantage (Regular Season)",
     "description": "Binary feature: 1 if Team 1 is playing at home, 0 otherwise (for regular season games).",
     "rationale": "Accounts for home court advantage observed in EDA.",
     "data_source": "RegularSeasonCompactResults"}, # Only relevant for Regular Season, Tournament is mostly Neutral

    # --- Conference Feature ---
    {"feature": "Team Conference (Categorical)",
     "description": "Conference abbreviation for each team.",
     "rationale": "Captures potential conference strength differences.",
     "data_source": "MTeamConferences, WTeamConferences, Conferences"},

    # --- Advanced Stats (If using Detailed Results later) ---
    {"feature": "Field Goal Percentage (Season)",
     "description": "Average field goal percentage during the regular season.",
     "rationale": "Measures shooting efficiency.",
     "data_source": "RegularSeasonDetailedResults"}, # Requires switching to Detailed Results

    {"feature": "Three-Point Percentage (Season)",
     "description": "Average three-point field goal percentage during the regular season.",
     "rationale": "Measures 3-point shooting efficiency.",
     "data_source": "RegularSeasonDetailedResults"}, # Requires switching to Detailed Results

    {"feature": "Free Throw Percentage (Season)",
     "description": "Average free throw percentage during regular season.",
     "rationale": "Measures free throw shooting efficiency.",
     "data_source": "RegularSeasonDetailedResults"}, # Requires switching to Detailed Results

    {"feature": "Rebound Rate (Offensive/Defensive Season)",
     "description": "Offensive and Defensive rebound rates during the regular season.",
     "rationale": "Measures rebounding ability.",
     "data_source": "RegularSeasonDetailedResults"}, # Requires switching to Detailed Results

    {"feature": "Assist-to-Turnover Ratio (Season)",
     "description": "Ratio of assists to turnovers during the regular season.",
     "rationale": "Measures ball-handling and offensive efficiency.",
     "data_source": "RegularSeasonDetailedResults"}, # Requires switching to Detailed Results

    {"feature": "Coach Experience (Years Coaching)",
     "description": "Number of years the head coach has been coaching Division I basketball.",
     "rationale": "Potentially captures coaching experience impact.",
     "data_source": "MTeamCoaches"}, # Men's data only initially

    # --- New Features: Non-Conference and Late Season Strength ---

    {"feature": "Strong Team Non-Conference Win Percentage",
     "description": "Win percentage in non-conference games for teams meeting 'strong team' criteria (>=73% overall win rate AND tournament appearance).",
     "rationale": "Measures performance of strong teams specifically in non-conference play, relevant for tournament matchups.",
     "data_source": "RegularSeasonCompactResults, MTeamConferences, WTeamConferences, MNCAATourneySeeds, WNCAATourneySeeds"}, # Requires multiple data sources

    {"feature": "Late vs. Early Non-Conf Top 10 Win Ratio",
     "description": "Ratio of late regular season non-conference wins against top 10 ranked opponents to early regular season non-conference wins against top 10 ranked opponents.",
     "rationale": "Captures late-season momentum against strong non-conference teams.",
     "data_source": "RegularSeasonCompactResults, MTeamConferences, WTeamConferences, MMasseyOrdinals"} # Requires multiple data sources
]


# --- Phase 1.5.2: Data Source Mapping (Updated - already included in 'data_source' key above) ---
print("\n--- Phase 1.5.2: Data Source Mapping (Updated) ---")
print("   - Data source for each feature is listed in the 'data_source' key in the potential_features list above.")


# --- Phase 1.5.3: Feature Engineering Complexity Assessment (Updated) ---
print("\n--- Phase 1.5.3: Feature Engineering Complexity Assessment (Updated) ---\n")
print("--- Feature Complexity Assessment (Updated - Ranked by Estimated Importance & Complexity) ---\n")

feature_complexity = {
    # --- Highly Important & Relatively Easy/Medium ---
    "Ranking System Rank (e.g., Pomeroy Rank)": "Medium", # High importance, medium complexity (merging, filtering)
    "Point Differential (Season)": "Easy", # High importance, easy to calculate
    "Average Points Scored (Season)": "Easy", # High importance, easy to calculate
    "Average Points Allowed (Season)": "Easy", # High importance, easy to calculate
    "Season Win Percentage": "Easy", # Good baseline, easy to calculate

    # --- Moderately Important & Medium Complexity ---
    "Seed Difference (Tournament Matchup)": "Easy (for tournament matchups, harder for all-pairs)", # Moderate importance in tournament, easy for tournament games
    "Field Goal Percentage (Season)": "Medium", # Moderate importance, detailed results needed
    "Three-Point Percentage (Season)": "Medium", # Moderate importance, detailed results needed
    "Free Throw Percentage (Season)": "Medium", # Moderate importance, detailed results needed
    "Assist-to-Turnover Ratio (Season)": "Medium", # Moderate importance, detailed results needed

    # --- Potentially Important but Higher Complexity ---
    "Strong Team Non-Conference Win Percentage": "Hard", # High potential importance, complex logic and merging
    "Late vs. Early Non-Conf Top 10 Win Ratio": "Hard", # High potential importance, complex logic, ranking data
    "Rebound Rate (Offensive/Defensive Season)": "Medium", # Moderate to high importance, detailed results needed
    "Team Conference (Categorical)": "Easy", # Moderate importance, easy categorical feature
    "Home Court Advantage (Regular Season)": "Easy", # Moderate importance, easy binary feature

    # --- Lower Importance or Very Complex ---
    "Historical Seed Win Rate (vs. Seed Range in Round)": "Hard", # Complex, may not be as impactful as team-level stats
    "Coach Experience (Years Coaching)": "Hard"  # Low to moderate importance, complex, men's data only
}

# --- Rank Features by Estimated Importance and then Complexity (Subjective Ranking) ---
ranked_features = [
    "Ranking System Rank (e.g., Pomeroy Rank)", # 1. High Importance, Medium Complexity
    "Point Differential (Season)",              # 2. High Importance, Easy
    "Average Points Scored (Season)",           # 3. High Importance, Easy
    "Average Points Allowed (Season)",          # 4. High Importance, Easy
    "Season Win Percentage",                    # 5. Good Baseline, Easy
    "Strong Team Non-Conference Win Percentage", # 6. High Potential, Hard - Focus Next
    "Late vs. Early Non-Conf Top 10 Win Ratio",  # 7. High Potential, Hard - Focus Next
    "Seed Difference (Tournament Matchup)",     # 8. Moderate Importance (Tournament), Easy
    "Field Goal Percentage (Season)",           # 9. Moderate Importance, Medium
    "Three-Point Percentage (Season)",          # 10. Moderate Importance, Medium
    "Free Throw Percentage (Season)",           # 11. Moderate Importance, Medium
    "Assist-to-Turnover Ratio (Season)",        # 12. Moderate Importance, Medium
    "Rebound Rate (Offensive/Defensive Season)",# 13. Moderate to High Importance, Medium
    "Team Conference (Categorical)",            # 14. Moderate Importance, Easy
    "Home Court Advantage (Regular Season)",    # 15. Moderate Importance, Easy
    "Historical Seed Win Rate (vs. Seed Range in Round)", # 16. Complex, Lower Impact (Maybe Later)
    "Coach Experience (Years Coaching)"         # 17. Complex, Lower Impact (Maybe Later)
]


print("--- Feature Complexity Assessment (Updated - Ranked List) ---\n")
for rank, feature_name in enumerate(ranked_features, start=1):
    complexity = feature_complexity.get(feature_name, "Unknown")
    print(f"{rank}. Feature: '{feature_name}' - Complexity: {complexity}")


print("\n--- Phase 1.5 Completed (Updated): Feature Brainstorming, Data Source Mapping, Complexity Assessment, Feature Ranking ---")


# --- Phase 2.1: Feature Engineering - Basic Team Statistics ---

print("\n--- Phase 2.1: Feature Engineering - Basic Team Statistics ---\n")

# --- 2.1.1: Data Aggregation Setup ---
print("\n--- 2.1.1: Data Aggregation Setup ---")

def aggregate_results_data(results_df):
    """Aggregates regular season compact results data for feature engineering."""
    winning_team_stats = results_df.groupby(['Season', 'WTeamID']).agg(
        WScore_mean=('WScore', 'mean'),
        LScore_mean_win=('LScore', 'mean'), # Avg LScore when winning
        NumOT_sum_win=('NumOT', 'sum'), # Total OT wins
        GameCount_win=('DayNum', 'count') # Number of wins
    ).reset_index().rename(columns={'WTeamID': 'TeamID'})

    losing_team_stats = results_df.groupby(['Season', 'LTeamID']).agg(
        LScore_mean=('LScore', 'mean'),
        WScore_mean_loss=('WScore', 'mean'), # Avg WScore when losing
        NumOT_sum_loss=('NumOT', 'sum'), # Total OT losses
        GameCount_loss=('DayNum', 'count')  # Number of losses
    ).reset_index().rename(columns={'LTeamID': 'TeamID'})

    team_season_stats = pd.merge(winning_team_stats, losing_team_stats, on=['Season', 'TeamID'], how='outer')
    team_season_stats = team_season_stats.fillna(0) # Fill NaN from outer merge with 0

    print("\n--- Aggregated Winning Team Stats (Example) ---")
    print(winning_team_stats.head())
    print("\n--- Aggregated Losing Team Stats (Example) ---")
    print(losing_team_stats.head())
    print("\n--- Merged Team Season Stats (Example) ---")
    print(team_season_stats.head())
    return team_season_stats

m_team_season_basic_stats_df = aggregate_results_data(m_regular_season_compact_results_df)
w_team_season_basic_stats_df = aggregate_results_data(w_regular_season_compact_results_df)


# --- 2.1.2: Calculate Basic Offensive Stats ---
print("\n--- 2.1.2: Calculate Basic Offensive Stats ---")

def calculate_offensive_stats(team_season_stats):
    """Calculates basic offensive statistics."""
    team_season_stats['AvgPointsScored'] = (team_season_stats['WScore_mean'] * team_season_stats['GameCount_win'] + team_season_stats['LScore_mean'] * team_season_stats['GameCount_loss']) / (team_season_stats['GameCount_win'] + team_season_stats['GameCount_loss'])
    team_season_stats['PointDifferential'] = (team_season_stats['WScore_mean'] - team_season_stats['LScore_mean_win']) # Simplified Point Differential (Win Score - Loss Score when winning) - Can refine if needed

    print("\n--- Team Season Stats with Offensive Stats (Example) ---")
    print(team_season_stats[['Season', 'TeamID', 'AvgPointsScored', 'PointDifferential']].head())
    return team_season_stats

m_team_season_basic_stats_df = calculate_offensive_stats(m_team_season_basic_stats_df)
w_team_season_basic_stats_df = calculate_offensive_stats(w_team_season_basic_stats_df)


# --- 2.1.3: Calculate Basic Defensive Stats ---
print("\n--- 2.1.3: Calculate Basic Defensive Stats ---")

def calculate_defensive_stats(team_season_stats):
    """Calculates basic defensive statistics."""
    team_season_stats['AvgPointsAllowed'] = (team_season_stats['LScore_mean_win'] * team_season_stats['GameCount_win'] + team_season_stats['WScore_mean_loss'] * team_season_stats['GameCount_loss']) / (team_season_stats['GameCount_win'] + team_season_stats['GameCount_loss'])
    team_season_stats['OpponentPointDifferential'] = -team_season_stats['PointDifferential'] # Opponent Point Diff is negative of Team Point Diff

    print("\n--- Team Season Stats with Defensive Stats (Example) ---")
    print(team_season_stats[['Season', 'TeamID', 'AvgPointsAllowed', 'OpponentPointDifferential']].head())
    return team_season_stats

m_team_season_basic_stats_df = calculate_defensive_stats(m_team_season_basic_stats_df)
w_team_season_basic_stats_df = calculate_defensive_stats(w_team_season_basic_stats_df)



# --- 2.1.4: Calculate Basic Stats from Detailed Results (OPTIONAL - skipping for now for brevity, add later if using detailed results) ---
print("\n--- 2.1.4: Calculate Basic Stats from Detailed Results - SKIPPING FOR NOW (OPTIONAL) ---")
# --- Implementation for Detailed Results would go here, if you choose to use Detailed Results DataFrames ---


# --- 2.1.5: Aggregate Season-Level Team Statistics ---
print("\n--- 2.1.5: Aggregate Season-Level Team Statistics ---")
# --- Basic stats are already aggregated in team_season_basic_stats_df DataFrames ---
# --- Further features will be merged into these DataFrames in subsequent steps ---
print("\n--- Basic Season-Level Team Statistics Aggregation Completed ---")


# --- Phase 2.2: Feature Engineering - Ranking System Features ---

print("\n--- Phase 2.2: Feature Engineering - Ranking System Features ---\n")

# --- 2.2.1: Select Ranking Systems ---
print("\n--- 2.2.1: Select Ranking Systems ---")
selected_ranking_systems = ['POM', 'SAG'] # Example: Pomeroy and Sagarin
print(f"--- Selected Ranking Systems: {selected_ranking_systems} ---")


# --- 2.2.2: Extract Ranking Data for Selected Systems ---
print("\n--- 2.2.2: Extract Ranking Data for Selected Systems ---")

def extract_ranking_data(massey_ordinals_df, selected_systems, ranking_day_num=133):
    """Extracts ranking data for selected systems and day number."""
    filtered_rankings = massey_ordinals_df[
        (massey_ordinals_df['SystemName'].isin(selected_systems)) &
        (massey_ordinals_df['RankingDayNum'] == ranking_day_num)
    ]
    print("\n--- Filtered Ranking Data (Example) ---")
    print(filtered_rankings.head())
    return filtered_rankings

m_filtered_rankings_df = extract_ranking_data(m_massey_ordinals_df, selected_ranking_systems)



# --- 2.2.3: Create Team-Season Ranking Features ---
print("\n--- 2.2.3: Create Team-Season Ranking Features ---")

def create_team_season_ranking_features(filtered_rankings_df, selected_systems):
    """Creates team-season ranking features by pivoting the ranking data."""
    ranking_features = filtered_rankings_df.pivot_table(
        index=['Season', 'TeamID'],
        columns='SystemName',
        values='OrdinalRank'
    ).reset_index()
    ranking_features.columns.name = None # Remove column name from multi-index
    ranking_features = ranking_features.rename(columns={system: f'{system}_Rank' for system in selected_systems}) # Rename columns
    print("\n--- Team-Season Ranking Features (Example) ---")
    print(ranking_features.head())
    return ranking_features

m_team_season_ranking_features_df = create_team_season_ranking_features(m_filtered_rankings_df, selected_ranking_systems)

# --- Merge Ranking Features into basic stats DataFrames ---
m_team_season_stats_df = pd.merge(m_team_season_basic_stats_df, m_team_season_ranking_features_df, on=['Season', 'TeamID'], how='left')
w_team_season_stats_df = w_team_season_basic_stats_df # Women's rankings not available in provided data, can use Men's or skip for women



# --- Function to Calculate Women's Team Ranking for All Years (Further Correction) ---
def calculate_womens_ranking_all_seasons(results_df, team_conferences_df, teams_df):
    """
    Calculates a custom ranking for women's teams for each available season based on:
    - Win Percentage
    - Strength of Conference
    - Strength of Schedule

    Args:
        results_df (DataFrame): WRegularSeasonCompactResults DataFrame.
        team_conferences_df (DataFrame): WTeamConferences DataFrame.
        teams_df (DataFrame): WTeams DataFrame.

    Returns:
        DataFrame: DataFrame with TeamID, Season, and CustomRank columns for all seasons.
    """
    all_seasons_rankings = []  # List to store ranking DataFrames for each season

    for season_year in results_df['Season'].unique():  # Iterate through each unique season
        print(f"\n--- Calculating Women's Ranking for Season {season_year} ---")
        season_results = results_df[results_df['Season'] == season_year].copy()
        season_team_conferences = team_conferences_df[team_conferences_df['Season'] == season_year].copy()
        season_teams = teams_df.copy()

        # 1. Calculate Team Win Percentage
        team_wins_series = season_results['WTeamID'].value_counts().rename('Wins')
        team_wins = team_wins_series.reset_index()
        if not team_wins.empty:
            team_wins.columns = ['TeamID', 'Wins']
        else:
            team_wins = pd.DataFrame(columns=['TeamID', 'Wins'])

        team_losses_series = season_results['LTeamID'].value_counts().rename('Losses')
        team_losses = team_losses_series.reset_index()
        if not team_losses.empty:
            team_losses.columns = ['TeamID', 'Losses']
        else:
            team_losses = pd.DataFrame(columns=['TeamID', 'Losses'])

        # Recalculate GamesPlayed directly and ensure it only has 'GamesPlayed' column (No change needed here)
        team_games_played_data = []
        all_team_ids = pd.concat([team_wins['TeamID'], team_losses['TeamID']]).unique()  # Get unique team IDs
        for team_id in all_team_ids:
            wins_count = team_wins[team_wins['TeamID'] == team_id]['Wins'].iloc[0] if team_id in team_wins['TeamID'].values else 0
            losses_count = team_losses[team_losses['TeamID'] == team_id]['Losses'].iloc[0] if team_id in team_losses['TeamID'].values else 0
            games_played = wins_count + losses_count
            team_games_played_data.append({'TeamID': team_id, 'GamesPlayed': games_played})
        team_games_played = pd.DataFrame(team_games_played_data).set_index('TeamID')[['GamesPlayed']] # Explicitly select ONLY 'GamesPlayed' column


        team_win_percentage_data = []
        for team_id in all_team_ids:
            wins_count = team_wins[team_wins['TeamID'] == team_id]['Wins'].iloc[0] if team_id in team_wins['TeamID'].values else 0
            games_played = team_games_played.loc[team_id]['GamesPlayed'] if team_id in team_games_played.index else 0
            win_percentage = (wins_count / games_played) if games_played > 0 else 0
            team_win_percentage_data.append({'TeamID': team_id, 'WinPercentage': win_percentage})
        team_win_percentage = pd.DataFrame(team_win_percentage_data).set_index('TeamID')


        # --- Corrected team_losses DataFrame creation to ensure only 'Losses' column ---
        team_losses_data = []
        all_team_ids_losses = season_results['LTeamID'].unique() # Get unique losing team IDs
        for team_id in all_team_ids_losses:
            losses_count = team_losses_series.get(team_id, 0) # Use .get to handle cases where team_id might not be in team_losses_series
            team_losses_data.append({'TeamID': team_id, 'Losses': losses_count})
        team_losses = pd.DataFrame(team_losses_data).set_index('TeamID')[['Losses']] # Create DataFrame and explicitly select ONLY 'Losses' column


        # --- Combine using pd.concat and explicit column naming ---
        team_stats = pd.concat([team_win_percentage, team_games_played, team_losses], axis=1, join='outer').fillna(0)
        team_stats.columns = ['WinPercentage', 'GamesPlayed', 'Losses'] # Explicitly set column names
        team_stats = team_stats.reset_index()
        
        # Explicitly rename the column that contains the original TeamID index to 'TeamID'
        if 'TeamID' in team_stats.columns: # Check if 'TeamID' column already exists (unlikely but robust)
            team_stats = team_stats.rename(columns={'TeamID': 'TeamID'}) # Rename if it exists, though it shouldn't
        elif 'index' in team_stats.columns: # Check if reset_index() created 'index' column (more likely)
            team_stats = team_stats.rename(columns={'index': 'TeamID'}) # Rename 'index' column to 'TeamID'
        else: # Fallback in case neither 'TeamID' nor 'index' column exists (unexpected)
            team_stats['TeamID'] = team_stats.index # As a last resort, create 'TeamID' from current index (if index still holds TeamIDs)
            print("Warning: 'TeamID' column not found after reset_index(), creating it from index - Please Investigate!")

        # --- ADDED: Explicitly add 'Season' column to team_stats ---
        team_stats['Season'] = season_year # ADDED: Get season_year from loop and assign to 'Season' column
        
        #team_stats = team_stats.rename(columns={'TeamID':'TeamID_index', 'index': 'TeamID'}) # Correct column naming after reset_index
        team_stats['TeamID'] = team_stats['TeamID'].astype(int)
        team_stats = team_stats.set_index('TeamID') # Set TeamID as index again for merging in later steps


        # 2. Calculate Conference Strength (No changes needed)
        conference_win_rates = {}
        for conf_abbrev in season_team_conferences['ConfAbbrev'].unique():
            conf_teams = season_team_conferences[season_team_conferences['ConfAbbrev'] == conf_abbrev]['TeamID'].unique()
            conf_wins = 0
            conf_total_games = 0
            for team_id in conf_teams:
                # --- ADDED CHECK: Skip if team_id is not in team_stats.index ---
                if team_id in team_stats.index: # ADDED CONDITIONAL CHECK - Skip if team_id is missing from team_stats
                    conf_wins += team_stats.loc[team_id]['WinPercentage'] * team_stats.loc[team_id]['GamesPlayed']  # Use .loc for index-based access
                    conf_total_games += team_stats.loc[team_id]['GamesPlayed']  # LINE CAUSING ERROR - Use .loc for index-based access
                # --- If team_id is NOT in team_stats.index, just skip to the next team_id ---access
            conf_strength = (conf_wins / conf_total_games) if conf_total_games > 0 else 0
            conference_win_rates[conf_abbrev] = conf_strength

        team_conference_strength = []
        for team_id_val in team_stats.index: # Iterate through index values
            team_id = int(team_id_val) # Ensure team_id is int
            conf_abbrev = season_team_conferences[season_team_conferences['TeamID'] == team_id]['ConfAbbrev'].iloc[0]
            strength = conference_win_rates.get(conf_abbrev, 0)
            team_conference_strength.append({'TeamID': team_id, 'Season': season_year, 'ConferenceStrength': strength})
        team_conference_strength_df = pd.DataFrame(team_conference_strength)

        # 3. Calculate Strength of Schedule (No changes needed - but adjust index-based access)
        team_schedule_strength = []
        for team_id_val in team_stats.index: # Iterate through index values
            team_id = int(team_id_val) # Ensure team_id is int
            opponent_win_percentages = []

            # Games where team won
            winning_games = season_results[season_results['WTeamID'] == team_id]
            for _, game in winning_games.iterrows():
                opponent_id = game['LTeamID']
                if opponent_id in team_stats.index: # Use .index
                    opponent_win_percentages.append(team_stats.loc[opponent_id]['WinPercentage']) # Use .loc

            # Games where team lost
            losing_games = season_results[season_results['LTeamID'] == team_id]
            for _, game in losing_games.iterrows():
                opponent_id = game['WTeamID']
                if opponent_id in team_stats.index: # Use .index
                    opponent_win_percentages.append(team_stats.loc[opponent_id]['WinPercentage']) # Use .loc

            avg_opponent_win_pct = np.mean(opponent_win_percentages) if opponent_win_percentages else 0
            team_schedule_strength.append({'TeamID': team_id, 'Season': season_year, 'StrengthOfSchedule': avg_opponent_win_pct})
        team_schedule_strength_df = pd.DataFrame(team_schedule_strength)

        # 4. Combine and Calculate Ranking Score (No changes needed)
        team_ranking_data = pd.merge(team_stats.reset_index(), team_conference_strength_df, on=['TeamID', 'Season']) # Reset index before merge
        team_ranking_data = pd.merge(team_ranking_data, team_schedule_strength_df, on=['TeamID', 'Season'])

        # --- Weights for Ranking Components (Adjust as needed) ---
        WEIGHT_WIN_PCT = 0.5
        WEIGHT_CONF_STRENGTH = 0.3
        WEIGHT_SOS = 0.2

        team_ranking_data['RankingScore'] = (
            WEIGHT_WIN_PCT * team_ranking_data['WinPercentage'] +
            WEIGHT_CONF_STRENGTH * team_ranking_data['ConferenceStrength'] +
            WEIGHT_SOS * team_ranking_data['StrengthOfSchedule']
        )

        # 5. Rank Teams (No changes needed)
        team_ranking_data['CustomRank'] = team_ranking_data['RankingScore'].rank(ascending=False, method='min').astype(int)

        season_ranking_output_df = team_ranking_data[['TeamID', 'Season', 'CustomRank']].sort_values(by='CustomRank').reset_index(drop=True)
        all_seasons_rankings.append(season_ranking_output_df)  # Append season ranking to list
        # For Data Analysis Only else comment out 
        #print(f"\n--- Custom Women's Ranking for Season {season_year} (Top 5) ---")  # Reduced to top 5 for brevity in loop
        #print(season_ranking_output_df.head(5))

    # Concatenate rankings from all seasons into a single DataFrame
    womens_rankings_all_seasons_df = pd.concat(all_seasons_rankings, ignore_index=True)
    # For Data Analysis Only else comment out 
    #print(f"\n--- Custom Women's Ranking Calculated for ALL Seasons ---")
    #print(womens_rankings_all_seasons_df.head())
    return womens_rankings_all_seasons_df


# --- Calculate Ranking for ALL Available Seasons ---
womens_custom_ranking_all_seasons_df = calculate_womens_ranking_all_seasons(w_regular_season_compact_results_df, w_team_conferences_df, w_teams_df)


# --- Function to Merge Custom Ranking into team_season_stats_df (No Change Needed in Merge Function) ---
def merge_womens_custom_ranking(team_season_stats_df, womens_ranking_df, rank_feature_name='CustomRank_Womens'):
    """Merges the custom women's ranking into the team_season_stats_df."""
    team_season_stats_ranked = pd.merge(team_season_stats_df, womens_ranking_df[['TeamID', 'Season', 'CustomRank']], on=['TeamID', 'Season'], how='left')  # Left merge to keep all stats
    team_season_stats_ranked = team_season_stats_ranked.rename(columns={'CustomRank': rank_feature_name})  # Rename column for clarity
    # For Data Analysis Only else comment out 
    #print("\n--- Team Season Stats with Custom Women's Ranking (Example - First 5 rows after merge) ---")  # Updated print statement
    #print(team_season_stats_ranked[['Season', 'TeamID', rank_feature_name]].head())
    return team_season_stats_ranked

# --- Assuming w_team_season_stats_df exists from previous phases, merge the ranking ---
# --- If not, you'll need to create w_team_season_stats_df (e.g., from Phase 2.1 outputs) first ---
# Example: Assuming w_team_season_basic_stats_df is available from Phase 2.1
# w_team_season_stats_df = w_team_season_basic_stats_df.copy() # Or however you are initializing w_team_season_stats_df

# Merge Custom Ranking into w_team_season_stats_df
w_team_season_stats_df = merge_womens_custom_ranking(w_team_season_stats_df, womens_custom_ranking_all_seasons_df, rank_feature_name='CustomRank_Womens') # Commented out to avoid re-merge if already done


print("\n--- Women's Custom Ranking Feature Engineering for ALL Seasons Completed ---")


# --- 2.2.4: Create Matchup Ranking Difference Features (Implementation in Phase 2.5.4 - Matchup Dataset Prep) ---
print("\n--- 2.2.4: Create Matchup Ranking Difference Features - IMPLEMENTATION IN PHASE 2.5.4 ---")
print("--- (Matchup Ranking Difference Features will be created when preparing the matchup feature dataset in Phase 2.5.4) ---")


# --- Phase 2.3: Feature Engineering - Seed and Tournament History Features ---

print("\n--- Phase 2.3: Feature Engineering - Seed and Tournament History Features ---\n")

# --- 2.3.1: Merge Seed Data with Team Season Stats ---
print("\n--- 2.3.1: Merge Seed Data with Team Season Stats ---")

def merge_seed_data(team_season_stats, tourney_seeds_df):
    """Merges tournament seed data with team season stats."""

    def get_seed_int(seed): # Helper function to extract integer seed value
        try:
            return int(seed[1:3])
        except: # Handle potential parsing errors (e.g., play-in seeds with letters)
            return 16 # Assign a default high seed value for unparseable seeds

    tourney_seeds_df['Seed_int'] = tourney_seeds_df['Seed'].apply(get_seed_int) # Create integer seed column
    seed_features = tourney_seeds_df[['Season', 'TeamID', 'Seed', 'Seed_int']] # Select relevant columns

    team_season_stats_seeded = pd.merge(team_season_stats, seed_features, on=['Season', 'TeamID'], how='left') # Left merge to keep all team_season_stats
    print("\n--- Team Season Stats with Seed Data (Example) ---")
    print(team_season_stats_seeded.head())
    return team_season_stats_seeded

m_team_season_stats_df = merge_seed_data(m_team_season_stats_df, m_ncaa_tourney_seeds_df)
w_team_season_stats_df = merge_seed_data(w_team_season_stats_df, w_ncaa_tourney_seeds_df)



# --- 2.3.2: Create Seed Difference Feature for Matchups (Implementation in Phase 2.5.4 - Matchup Dataset Prep) ---
print("\n--- 2.3.2: Create Seed Difference Feature for Matchups - IMPLEMENTATION IN PHASE 2.5.4 ---")
print("--- (Seed Difference Feature will be created when preparing the matchup feature dataset in Phase 2.5.4) ---")


# --- 2.3.3: Historical Seed Performance (Tournament Win Rates per Seed - OPTIONAL - Skipping for now) ---
print("\n--- 2.3.3: Historical Seed Performance - SKIPPING FOR NOW (OPTIONAL) ---")
# --- Implementation for Historical Seed Performance would go here ---


# --- Phase 2.4: Feature Engineering - Conference and Coaching Features ---

print("\n--- Phase 2.4: Feature Engineering - Conference and Coaching Features ---\n")

# --- 2.4.1: Merge Conference Data with Team Season Stats ---
print("\n--- 2.4.1: Merge Conference Data with Team Season Stats ---")

def merge_conference_data(team_season_stats, team_conferences_df):
    """Merges conference data with team season stats."""
    conference_features = team_conferences_df[['Season', 'TeamID', 'ConfAbbrev']]
    team_season_stats_conf = pd.merge(team_season_stats, conference_features, on=['Season', 'TeamID'], how='left') # Left merge
    print("\n--- Team Season Stats with Conference Data (Example) ---")
    print(team_season_stats_conf.head())
    return team_season_stats_conf

m_team_season_stats_df = merge_conference_data(m_team_season_stats_df, m_team_conferences_df)
w_team_season_stats_df = merge_conference_data(w_team_season_stats_df, w_team_conferences_df)


# --- 2.4.2: Conference as Categorical Feature ---
print("\n--- 2.4.2: Conference as Categorical Feature ---")
# --- 'ConfAbbrev' column is now available in team_season_stats_df for use as categorical feature ---
print("--- 'ConfAbbrev' column is now available as a categorical feature ---")


# --- 2.4.3: Coaching Features (OPTIONAL - Skipping for now) ---
print("\n--- 2.4.3: Coaching Features - SKIPPING FOR NOW (OPTIONAL) ---")
# --- Implementation for Coaching Features would go here ---


# --- Phase 2.X: Non-Conference and Late-Season Strength Features (From Previous Response - Re-integrating) ---
print("\n--- Phase 2.X: Non-Conference and Late-Season Strength Features (Re-integrating from previous response) ---\n")

# --- 2.X.1: Define "Non-Conference" Games (Re-integrated) ---
# --- 2.X.2: Define "Strong Team" Criteria (Re-integrated) ---
# --- 2.X.3: Engineer "Strong Team Non-Conference Win Percentage" Feature (Re-integrated) ---
# --- 2.X.4: Define "Early" vs. "Late" Regular Season (Re-integrated) ---
# --- 2.X.5: Engineer "Late vs. Early Non-Conf Top 10 Win Ratio" Feature (Re-integrated) ---

# --- Re-run the code block from the previous response for Phase 2.X here ---
# --- (Copy and paste the entire Phase 2.X code block from the previous response into this section) ---

# --- Phase 2.X Code Block from previous response starts ---
print("\n--- Phase 2.X: Non-Conference and Late-Season Strength Features ---\n")

# --- 2.X.1: Define "Non-Conference" Games ---
print("\n--- 2.X.1: Define 'Non-Conference' Games ---")

def is_non_conference_game(game, team_conferences_df):
    """
    Determines if a game is a non-conference game.

    Args:
        game (Series): Row from RegularSeasonCompactResults DataFrame.
        team_conferences_df (DataFrame): Team conference data (MTeamConferences or WTeamConferences).

    Returns:
        bool: True if non-conference, False otherwise.
    """
    season = game['Season']
    team1_id = game['WTeamID']
    team2_id = game['LTeamID']

    team1_conf = team_conferences_df[(team_conferences_df['Season'] == season) & (team_conferences_df['TeamID'] == team1_id)]['ConfAbbrev'].iloc[0]
    team2_conf = team_conferences_df[(team_conferences_df['Season'] == season) & (team_conferences_df['TeamID'] == team2_id)]['ConfAbbrev'].iloc[0]

    return team1_conf != team2_conf

def add_non_conference_flag(results_df, team_conferences_df, gender='Men'):
    """Adds a 'NonConference' column to the results DataFrame."""
    non_conference_flags = []
    for index, game in results_df.iterrows():
        try:  # Use try-except to handle potential missing conference data (rare, but robust)
            is_non_conf = is_non_conference_game(game, team_conferences_df)
            non_conference_flags.append(is_non_conf)
        except:
            non_conference_flags.append(False) # Default to False if conference info missing
            print(f"Warning: Conference info missing for game in Season {game['Season']}, DayNum {game['DayNum']}, TeamIDs {game['WTeamID']}, {game['LTeamID']} ({gender}'s)")

    results_df['NonConference'] = non_conference_flags
    print(f"\n--- Added 'NonConference' flag to {gender}'s Regular Season Results ---")
    print(results_df[['Season', 'DayNum', 'WTeamID', 'LTeamID', 'NonConference']].head())
    return results_df

m_regular_season_compact_results_df = add_non_conference_flag(m_regular_season_compact_results_df, m_team_conferences_df, gender='Men')
w_regular_season_compact_results_df = add_non_conference_flag(w_regular_season_compact_results_df, w_team_conferences_df, gender='Women')


# --- 2.X.2: Define "Strong Team" Criteria ---
print("\n--- 2.X.2: Define 'Strong Team' Criteria ---")

def identify_strong_teams(results_df, teams_df, tourney_seeds_df, win_percentage_threshold=0.73, gender='Men'):
    """
    Identifies 'strong teams' based on win percentage and tournament appearance.

    Args:
        results_df (DataFrame): RegularSeasonCompactResults DataFrame.
        teams_df (DataFrame): Teams DataFrame (MTeams or WTeams).
        tourney_seeds_df (DataFrame): TournamentSeeds DataFrame (MNCAATourneySeeds or WNCAATourneySeeds).
        win_percentage_threshold (float): Win percentage threshold for 'strong team' criteria.
        gender (str): 'Men' or 'Women' for print messages.

    Returns:
        DataFrame: DataFrame with 'StrongTeam' flag added to team_season_stats.
    """
    team_season_win_stats = []
    for season in results_df['Season'].unique():
        season_results = results_df[results_df['Season'] == season]
        for team_id in teams_df['TeamID']: # Iterate through ALL teams, even if not in this season, to ensure consistent structure
            team_games = season_results[((season_results['WTeamID'] == team_id) | (season_results['LTeamID'] == team_id))]
            total_games = len(team_games)
            wins = len(season_results[season_results['WTeamID'] == team_id])
            win_percentage = wins / total_games if total_games > 0 else 0
            is_tournament_team = team_id in tourney_seeds_df[tourney_seeds_df['Season'] == season]['TeamID'].values
            is_strong_team = (win_percentage >= win_percentage_threshold) and is_tournament_team
            team_season_win_stats.append({'Season': season, 'TeamID': team_id, 'WinPercentage': win_percentage, 'IsTournamentTeam': is_tournament_team, 'StrongTeam': is_strong_team})

    team_season_stats_df = pd.DataFrame(team_season_win_stats) # Replaced team_season_stats with team_season_win_stats for clarity

    print(f"\n--- Identified 'Strong Teams' for {gender}'s Data ---")
    print(team_season_stats_df[team_season_stats_df['StrongTeam'] == True].head())
    return team_season_stats_df

m_team_season_strong_teams_df = identify_strong_teams(m_regular_season_compact_results_df, m_teams_df, m_ncaa_tourney_seeds_df, gender='Men')
w_team_season_strong_teams_df = identify_strong_teams(w_regular_season_compact_results_df, w_teams_df, w_ncaa_tourney_seeds_df, gender='Women')


# --- 2.X.3: Engineer "Strong Team Non-Conference Win Percentage" Feature ---
print("\n--- 2.X.3: Engineer 'Strong Team Non-Conference Win Percentage' Feature ---")

def engineer_strong_team_non_conf_win_pct(results_df, team_season_strong_teams_df, gender='Men'):
    """Engineers the 'Strong Team Non-Conference Win Percentage' feature."""
    strong_team_non_conf_win_percentages = []

    for season in results_df['Season'].unique():
        season_results = results_df[results_df['Season'] == season]
        strong_teams_season = team_season_strong_teams_df[(team_season_strong_teams_df['Season'] == season) & (team_season_strong_teams_df['StrongTeam'] == True)]

        for team_id in strong_teams_season['TeamID']: # Iterate ONLY through strong teams for this feature
            non_conf_games = season_results[((season_results['WTeamID'] == team_id) | (season_results['LTeamID'] == team_id)) & (season_results['NonConference'] == True)]
            non_conf_wins = len(non_conf_games[non_conf_games['WTeamID'] == team_id])
            total_non_conf_games = len(non_conf_games)
            non_conf_win_percentage = (non_conf_wins / total_non_conf_games) if total_non_conf_games > 0 else 0 # Handle division by zero

            strong_team_non_conf_win_percentages.append({'Season': season, 'TeamID': team_id, 'StrongTeamNonConfWinPct': non_conf_win_percentage})

    strong_team_non_conf_win_pct_df = pd.DataFrame(strong_team_non_conf_win_percentages)
    print(f"\n--- Engineered 'Strong Team Non-Conference Win Percentage' for {gender}'s Data (Example) ---")
    print(strong_team_non_conf_win_pct_df.head())
    return strong_team_non_conf_win_pct_df

m_strong_team_non_conf_win_pct_df = engineer_strong_team_non_conf_win_pct(m_regular_season_compact_results_df, m_team_season_strong_teams_df, gender='Men')
w_strong_team_non_conf_win_pct_df = engineer_strong_team_non_conf_win_pct(w_regular_season_compact_results_df, w_team_season_strong_teams_df, gender='Women')


# --- 2.X.4: Define "Early" vs. "Late" Regular Season ---
print("\n--- 2.X.4: Define 'Early' vs. 'Late' Regular Season ---")
LATE_SEASON_DAYNUM_CUTOFF = 90 # DayNum 90 as cutoff for early vs. late season
print(f"--- Using DayNum = {LATE_SEASON_DAYNUM_CUTOFF} as cutoff for Early vs. Late Regular Season ---")


# --- 2.X.5: Engineer "Late vs. Early Non-Conf Top 10 Win Ratio" Feature ---
print("\n--- 2.X.5: Engineer 'Late vs. Early Non-Conf Top 10 Win Ratio' Feature ---")

def engineer_late_early_non_conf_top10_win_ratio(results_df, team_conferences_df, massey_ordinals_df, ranking_system='POM', top_n=10, late_season_cutoff_daynum=LATE_SEASON_DAYNUM_CUTOFF, gender='Men'):
    """Engineers the 'Late vs. Early Non-Conf Top 10 Win Ratio' feature."""
    late_early_non_conf_top10_win_ratios = []

    for season in results_df['Season'].unique():
        season_results = results_df[results_df['Season'] == season]
        season_ordinals = massey_ordinals_df[(massey_ordinals_df['Season'] == season) & (massey_ordinals_df['SystemName'] == ranking_system) & (massey_ordinals_df['RankingDayNum'] == 133)] # Using DayNum 133 rankings (pre-tournament)

        for team_id in m_teams_df['TeamID']: # Iterate through ALL teams to create feature for everyone
            early_non_conf_top10_wins = 0
            late_non_conf_top10_wins = 0

            non_conf_games = season_results[((season_results['WTeamID'] == team_id) | (season_results['LTeamID'] == team_id)) & (season_results['NonConference'] == True)]

            for index, game in non_conf_games.iterrows():
                day_num = game['DayNum']
                opponent_id = game['LTeamID'] if game['WTeamID'] == team_id else game['WTeamID'] # Get opponent TeamID

                opponent_ranking_row = season_ordinals[season_ordinals['TeamID'] == opponent_id]
                if not opponent_ranking_row.empty: # Check if ranking data is available for opponent
                    opponent_rank = opponent_ranking_row['OrdinalRank'].iloc[0]
                    if opponent_rank <= top_n: # Check if opponent is Top N ranked
                        if day_num < late_season_cutoff_daynum:
                            early_non_conf_top10_wins += 1
                        else:
                            late_non_conf_top10_wins += 1

            win_ratio = (late_non_conf_top10_wins / early_non_conf_top10_wins) if early_non_conf_top10_wins > 0 else (late_non_conf_top10_wins if late_non_conf_top10_wins > 0 else 0) # Handle division by zero and cases where early wins are zero but late wins exist
            late_early_non_conf_top10_win_ratios.append({'Season': season, 'TeamID': team_id, 'LateEarlyNonConfTop10WinRatio': win_ratio})

    late_early_non_conf_top10_win_ratio_df = pd.DataFrame(late_early_non_conf_top10_win_ratios)
    print(f"\n--- Engineered 'Late vs. Early Non-Conf Top 10 Win Ratio' for {gender}'s Data (Example) ---")
    print(late_early_non_conf_top10_win_ratio_df.head())
    return late_early_non_conf_top10_win_ratio_df

m_late_early_non_conf_top10_win_ratio_df = engineer_late_early_non_conf_top10_win_ratio(m_regular_season_compact_results_df, m_team_conferences_df, m_massey_ordinals_df, gender='Men')
w_late_early_non_conf_top10_win_ratio_df = engineer_late_early_non_conf_top10_win_ratio(w_regular_season_compact_results_df, w_team_conferences_df, m_massey_ordinals_df, gender='Women') # Note: Using Men's Massey Ordinals as Women's rankings are not in this dataset
# --- Phase 2.X Code Block from previous response ends ---


# --- Phase 2.5: Data Preprocessing and Feature Selection ---
print("\n--- Phase 2.5: Data Preprocessing and Feature Selection ---\n")


# --- 2.5.1: Missing Value Handling ---
print("\n--- 2.5.1: Missing Value Handling ---")
# --- Check for missing values in team_season_stats_df (after merging all features) ---
print("\n--- Missing Values Before Handling (Men's Data) ---")
print(m_team_season_stats_df.isnull().sum())
print("\n--- Missing Values Before Handling (Women's Data) ---")
print(w_team_season_stats_df.isnull().sum())


# --- Impute missing CustomRank_Womens with median (Example) ---
rank_col_womens = 'CustomRank_Womens' # Use your custom rank feature name
w_team_season_stats_df[rank_col_womens] = w_team_season_stats_df[rank_col_womens].fillna(w_team_season_stats_df[rank_col_womens].median()) # Median imputation for CustomRank_Womens

print("\n--- Missing Values After Imputation (Women's Data - CustomRank Imputed) ---")
print(w_team_season_stats_df.isnull().sum()) # Re-check missing values after imputation


# --- Impute missing ranking values with median (Example - adjust strategy as needed) ---
for system in selected_ranking_systems:
    rank_col = f'{system}_Rank'
    m_team_season_stats_df[rank_col] = m_team_season_stats_df[rank_col].fillna(m_team_season_stats_df[rank_col].median()) # Median imputation



print("\n--- Missing Values After Imputation (Men's Data) ---")
print(m_team_season_stats_df.isnull().sum()) # Re-check missing values after imputation
print("\n--- Missing Values After Imputation (Women's Data) ---")
print(w_team_season_stats_df.isnull().sum()) # Re-check missing values after imputation


# --- 2.5.2: Feature Scaling/Normalization (Optional - Skipping for GBMs, implement if using LR/NN later) ---
print("\n--- 2.5.2: Feature Scaling/Normalization - SKIPPING FOR GBMs (OPTIONAL for other models) ---")
# --- Scaling/Normalization implementation would go here if needed for models like Logistic Regression or Neural Networks ---


# --- Merge New Features into team_season_stats_df (Before Phase 2.5.3) ---
print("\n--- Merging New Features into team_season_stats_df (Before Phase 2.5.3) ---\n")

# --- Merge Strong Team Non-Conference Win Percentage ---
m_team_season_stats_df = pd.merge(m_team_season_stats_df, m_strong_team_non_conf_win_pct_df, on=['Season', 'TeamID'], how='left')
w_team_season_stats_df = pd.merge(w_team_season_stats_df, w_strong_team_non_conf_win_pct_df, on=['Season', 'TeamID'], how='left')
print("\n--- Merged Strong Team Non-Conference Win Percentage ---")
print(m_team_season_stats_df[['Season', 'TeamID', 'StrongTeamNonConfWinPct']].head())
print(w_team_season_stats_df[['Season', 'TeamID', 'StrongTeamNonConfWinPct']].head())


# --- Merge Late vs. Early Non-Conf Top 10 Win Ratio ---
m_team_season_stats_df = pd.merge(m_team_season_stats_df, m_late_early_non_conf_top10_win_ratio_df, on=['Season', 'TeamID'], how='left')
w_team_season_stats_df = pd.merge(w_team_season_stats_df, w_late_early_non_conf_top10_win_ratio_df, on=['Season', 'TeamID'], how='left')
print("\n--- Merged Late vs. Early Non-Conf Top 10 Win Ratio ---")
print(m_team_season_stats_df[['Season', 'TeamID', 'LateEarlyNonConfTop10WinRatio']].head())
print(w_team_season_stats_df[['Season', 'TeamID', 'LateEarlyNonConfTop10WinRatio']].head())

print("\n--- Merging New Features into team_season_stats_df Completed ---\n")


print("\n--- Sample ConfAbbrev_x and ConfAbbrev_y columns (Men's Data) ---")
print(m_team_season_stats_df[['Season', 'TeamID', 'ConfAbbrev', 'StrongTeamNonConfWinPct','LateEarlyNonConfTop10WinRatio']].head(10))

print("\n--- Sample ConfAbbrev_x and ConfAbbrev_y columns (Women's Data) ---")
print(w_team_season_stats_df[['Season', 'TeamID', 'ConfAbbrev', 'StrongTeamNonConfWinPct','LateEarlyNonConfTop10WinRatio']].head(10))


# --- 2.5.3: Feature Selection (Initial Pass - Basic Selection) ---
print("\n--- 2.5.3: Feature Selection (Initial Pass - Basic Selection) ---")
# --- Select a subset of features to start with (Example - refine based on EDA, feature importance later) ---
selected_features_initial_men = [
    'Season', 'TeamID', 'AvgPointsScored', 'AvgPointsAllowed', 'PointDifferential',
    'POM_Rank', 'SAG_Rank', 'Seed_int', 'ConfAbbrev', 'StrongTeamNonConfWinPct', 'LateEarlyNonConfTop10WinRatio' # Include new features
] # Example feature list - ADJUST AND EXPAND BASED ON EDA AND FEATURE IMPORTANCE

selected_features_initial_women = [
    'Season', 'TeamID', 'AvgPointsScored', 'AvgPointsAllowed', 'PointDifferential',
    'CustomRank_Womens', 'Seed_int', 'ConfAbbrev', 'StrongTeamNonConfWinPct', 'LateEarlyNonConfTop10WinRatio' # Include new features - using Men's Rankings for now
] # Example feature list for women - ADJUST AND EXPAND


m_team_season_stats_selected_features_df = m_team_season_stats_df[selected_features_initial_men].copy() # Create DataFrame with selected features
w_team_season_stats_selected_features_df = w_team_season_stats_df[selected_features_initial_women].copy() # Create DataFrame with selected features

print("\n--- Men's Team Season Stats with Selected Features (Example) ---")
print(m_team_season_stats_selected_features_df.head())
print("\n--- Women's Team Season Stats with Selected Features (Example) ---")
print(w_team_season_stats_selected_features_df.head())


# --- 2.5.4: Prepare Matchup Feature Dataset for Modeling ---
print("\n--- 2.5.4: Prepare Matchup Feature Dataset for Modeling ---")

def prepare_matchup_features(compact_results_df, team_season_stats_df, gender='Men'):
    """Prepares matchup feature dataset for model training."""
    matchup_features = []
    for index, game in compact_results_df.iterrows():
        season = game['Season']
        team1_id = min(game['WTeamID'], game['LTeamID']) # Lower TeamID as Team1
        team2_id = max(game['WTeamID'], game['LTeamID']) # Higher TeamID as Team2
        winner_id = game['WTeamID']

        team1_features = team_season_stats_df[(team_season_stats_df['Season'] == season) & (team_season_stats_df['TeamID'] == team1_id)].iloc[0].to_dict() # Get features for Team1
        team2_features = team_season_stats_df[(team_season_stats_df['Season'] == season) & (team_season_stats_df['TeamID'] == team2_id)].iloc[0].to_dict() # Get features for Team2

        game_features = {
            'Season': season,
            'Team1ID': team1_id,
            'Team2ID': team2_id,
            'Team1_Win': 1 if winner_id == team1_id else 0, # Target variable: 1 if Team1 (lower ID) wins
        }

        # Add Team 1 Features (rename to prefix with 'Team1_')
        for feature_name, feature_value in team1_features.items():
            if feature_name not in ['Season', 'TeamID']: # Avoid duplicate Season and TeamID columns
                game_features[f'Team1_{feature_name}'] = feature_value

        # Add Team 2 Features (rename to prefix with 'Team2_')
        for feature_name, feature_value in team2_features.items():
            if feature_name not in ['Season', 'TeamID']: # Avoid duplicate Season and TeamID columns
                game_features[f'Team2_{feature_name}'] = feature_value

        # Add Matchup Specific Features (e.g., Rank Difference, Seed Difference)
        game_features['RankDiff_POM'] = game_features.get('Team1_POM_Rank', 0) - game_features.get('Team2_POM_Rank', 0) # Rank Difference for POM
        game_features['RankDiff_SAG'] = game_features.get('Team1_SAG_Rank', 0) - game_features.get('Team2_SAG_Rank', 0) # Rank Difference for SAG
        game_features['SeedDiff'] = game_features.get('Team1_Seed_int', 16) - game_features.get('Team2_Seed_int', 16) # Seed Difference

        matchup_features.append(game_features)

    matchup_features_df = pd.DataFrame(matchup_features)
    print("\n--- Matchup Feature Dataset (Example) ---")
    print(matchup_features_df.head())
    return matchup_features_df

m_matchup_features_train_df = prepare_matchup_features(m_regular_season_compact_results_df, m_team_season_stats_selected_features_df, gender='Men')
w_matchup_features_train_df = prepare_matchup_features(w_regular_season_compact_results_df, w_team_season_stats_selected_features_df, gender='Women')


# Iterate through each X and confirm the columns 'RankDiff_POM', 'SeedDiff', ' RankDiff_SAG' exist
for X in [
    m_matchup_features_train_df, w_matchup_features_train_df
]:
  if 'RankDiff_POM' not in X.columns or 'SeedDiff' not in X.columns or 'RankDiff_SAG' not in X.columns:
    print("Columns 'RankDiff_POM', 'SeedDiff', and 'RankDiff_SAG' do not exist in the DataFrame.")
  else:
    print("Columns 'RankDiff_POM', 'SeedDiff', and 'RankDiff_SAG' exist in the DataFrame.")  



# --- One-Hot Encode ConfAbbrev Columns (Men's Data) ---
print("\n--- One-Hot Encoding ConfAbbrev Columns - Men's Data ---\n")

# One-hot encode Team1_ConfAbbrev
m_matchup_features_train_df = pd.get_dummies(m_matchup_features_train_df, columns=['Team1_ConfAbbrev'], prefix='Team1_Conf')
print("\n--- Men's Data - Team1_ConfAbbrev One-Hot Encoded (Example Columns) ---")
conf_cols_team1_men = [col for col in m_matchup_features_train_df.columns if 'Team1_Conf_' in col] # Get newly created one-hot columns
print(m_matchup_features_train_df[conf_cols_team1_men].head())


# One-hot encode Team2_ConfAbbrev
m_matchup_features_train_df = pd.get_dummies(m_matchup_features_train_df, columns=['Team2_ConfAbbrev'], prefix='Team2_Conf')
print("\n--- Men's Data - Team2_ConfAbbrev One-Hot Encoded (Example Columns) ---")
conf_cols_team2_men = [col for col in m_matchup_features_train_df.columns if 'Team2_Conf_' in col] # Get newly created one-hot columns
print(m_matchup_features_train_df[conf_cols_team2_men].head())

print("\n--- Men's Data - ConfAbbrev One-Hot Encoding Completed ---\n")


# --- One-Hot Encode ConfAbbrev Columns (Women's Data) ---
print("\n--- One-Hot Encoding ConfAbbrev Columns - Women's Data ---\n")

# One-hot encode Team1_ConfAbbrev
w_matchup_features_train_df = pd.get_dummies(w_matchup_features_train_df, columns=['Team1_ConfAbbrev'], prefix='Team1_Conf')
print("\n--- Women's Data - Team1_ConfAbbrev One-Hot Encoded (Example Columns) ---")
conf_cols_team1_women = [col for col in w_matchup_features_train_df.columns if 'Team1_Conf_' in col] # Get newly created one-hot columns
print(w_matchup_features_train_df[conf_cols_team1_women].head())


# One-hot encode Team2_ConfAbbrev
w_matchup_features_train_df = pd.get_dummies(w_matchup_features_train_df, columns=['Team2_ConfAbbrev'], prefix='Team2_Conf')
print("\n--- Women's Data - Team2_ConfAbbrev One-Hot Encoded (Example Columns) ---")
conf_cols_team2_women = [col for col in w_matchup_features_train_df.columns if 'Team2_Conf_' in col] # Get newly created one-hot columns
print(w_matchup_features_train_df[conf_cols_team2_women].head())

print("\n--- Women's Data - ConfAbbrev One-Hot Encoding Completed ---\n")


print("\n--- ConfAbbrev One-Hot Encoding Completed for Both Men's and Women's Data ---\n")



# --- Data Quality Check: Handling Null Values (Men's Data) ---
print("\n--- Data Quality Check: Handling Null Values - Men's Data ---\n")

print("\n--- Null Values BEFORE Handling (Men's Data) ---")
print(m_matchup_features_train_df.isnull().sum()) # Check for null values before handling

# Impute missing values with 0 for LateEarlyNonConfTop10WinRatio (or choose another appropriate value like median if preferred)
m_matchup_features_train_df['Team1_LateEarlyNonConfTop10WinRatio'] = m_matchup_features_train_df['Team1_LateEarlyNonConfTop10WinRatio'].fillna(0) # Impute NaN with 0
m_matchup_features_train_df['Team2_LateEarlyNonConfTop10WinRatio'] = m_matchup_features_train_df['Team2_LateEarlyNonConfTop10WinRatio'].fillna(0) # Impute NaN with 0

# Identify numerical columns for imputation (exclude categorical/one-hot encoded columns if any)
numerical_cols_men = m_matchup_features_train_df.select_dtypes(include=np.number).columns
print(f"\n--- Numerical Columns for Imputation (Men's Data): ---\n{numerical_cols_men.tolist()}")


# Impute missing values with median for numerical columns (Men's Data)
for col in numerical_cols_men:
    median_val = m_matchup_features_train_df[col].median()
    m_matchup_features_train_df[col] = m_matchup_features_train_df[col].fillna(median_val)
print("\n--- Null Values AFTER Median Imputation (Men's Data) ---")
print(m_matchup_features_train_df.isnull().sum()) # Re-check for null values after imputation

print("\n--- Null Value Handling Completed for Men's Data ---\n")


# --- Data Quality Check: Handling Null Values (Women's Data) ---
print("\n--- Data Quality Check: Handling Null Values - Women's Data ---\n")

print("\n--- Null Values BEFORE Handling (Women's Data) ---")
print(w_matchup_features_train_df.isnull().sum()) # Check for null values before handling

# Impute missing values with 0 for LateEarlyNonConfTop10WinRatio (or choose another appropriate value like median if preferred)
w_matchup_features_train_df['Team1_LateEarlyNonConfTop10WinRatio'] = w_matchup_features_train_df['Team1_LateEarlyNonConfTop10WinRatio'].fillna(0) # Impute NaN with 0
w_matchup_features_train_df['Team2_LateEarlyNonConfTop10WinRatio'] = w_matchup_features_train_df['Team2_LateEarlyNonConfTop10WinRatio'].fillna(0) # Impute NaN with 0

# Identify numerical columns for imputation (exclude categorical/one-hot encoded columns if any)
numerical_cols_women = w_matchup_features_train_df.select_dtypes(include=np.number).columns
print(f"\n--- Numerical Columns for Imputation (Women's Data): ---\n{numerical_cols_women.tolist()}")


# Impute missing values with median for numerical columns (Women's Data)
for col in numerical_cols_women:
    median_val_women = w_matchup_features_train_df[col].median()
    w_matchup_features_train_df[col] = w_matchup_features_train_df[col].fillna(median_val_women)
print("\n--- Null Values AFTER Median Imputation (Women's Data) ---")
print(w_matchup_features_train_df.isnull().sum()) # Re-check for null values after imputation

print("\n--- Null Value Handling Completed for Women's Data ---\n")

print("\n--- Null Value Handling Completed for Both Men's and Women's Data ---\n")

print("\n--- Phase 2 Completed: Feature Engineering and Data Preprocessing ---")


print('info for the mens feature dataframe: ',m_matchup_features_train_df.info())
print('info for the womens feature dataframe: ',w_matchup_features_train_df.info())


# Use the training dataframe that is before the time split to build full training dataset for Phase 4
m_matchup_features_full_dataset_train_df = m_matchup_features_train_df
w_matchup_features_full_dataset_train_df = w_matchup_features_train_df


# Confirm the number of records and no nulls in each dataframe
print(f"Number of records in m_matchup_features_full_dataset_train_df: {len(m_matchup_features_full_dataset_train_df)}")
print(f"Number of records in w_matchup_features_full_dataset_train_df: {len(w_matchup_features_full_dataset_train_df)}")
print(f"Number of null values in m_matchup_features_full_dataset_train_df: {m_matchup_features_full_dataset_train_df.isnull().sum().sum()}")
print(f"Number of null values in w_matchup_features_full_dataset_train_df: {w_matchup_features_full_dataset_train_df.isnull().sum().sum()}")


def early_compare_features(pred_df, val_df, dataset_name):
    """Compares features between prediction and validation datasets and lists missing features."""
    pred_features = set(pred_df.columns)
    val_features = set(val_df.columns)

    print(f"\n--- Feature Comparison for {dataset_name} ---")
    print(f"Number of features in prediction dataset: {len(pred_features)}")
    print(f"Number of features in validation dataset: {len(val_features)}")

    missing_features = list(val_features - pred_features)  # Features in validation but not in prediction

    if missing_features:
        print(f"\nMissing features in prediction dataset ({len(missing_features)} total):")
        for feature in missing_features[-10:]:  # Get last 10 missing features
            print(f"- {feature}")
    else:
        print("\nNo missing features found in prediction dataset.")

# --- Compare features for men's datasets ---
early_compare_features(m_matchup_features_full_dataset_train_df, m_matchup_features_train_df, "Men's")

# --- Compare features for women's datasets ---
early_compare_features(w_matchup_features_full_dataset_train_df, w_matchup_features_train_df, "Women's")


# --- Prepare X and y DataFrames for Men and Women (Assuming already prepared in Phase 2.5.4) ---
# --- Replace with your actual X_train, y_train preparation code if needed ---
y_train_men = m_matchup_features_train_df['Team1_Win']
feature_cols_men = [col for col in m_matchup_features_train_df.columns if col not in ['Team1_Win', 'Season', 'Team1ID', 'Team2ID']]
X_train_men = m_matchup_features_train_df[feature_cols_men]
print(X_train_men.shape)
print(y_train_men.shape)

y_train_women = w_matchup_features_train_df['Team1_Win']
feature_cols_women = [col for col in w_matchup_features_train_df.columns if col not in ['Team1_Win', 'Season', 'Team1ID', 'Team2ID']]
X_train_women = w_matchup_features_train_df[feature_cols_women]
print(X_train_women.shape)
print(y_train_women.shape)


# Iterate through each X and confirm the columns 'RankDiff_POM', 'SeedDiff', ' RankDiff_SAG' exist
for X in [
    X_train_men, X_train_women
]:
  if 'RankDiff_POM' not in X.columns or 'SeedDiff' not in X.columns or 'RankDiff_SAG' not in X.columns:
    print("Columns 'RankDiff_POM', 'SeedDiff', and 'RankDiff_SAG' do not exist in the DataFrame.")
  else:
    print("Columns 'RankDiff_POM', 'SeedDiff', and 'RankDiff_SAG' exist in the DataFrame.")  



# --- Phase 3.1: Model Selection (GPU Accelerated Baseline Models) ---
print("\n--- Phase 3.1: Model Selection (GPU Accelerated Baseline Models) ---\n")


# --- 3.1.1: cuML Logistic Regression (GPU) ---
print("\n--- 3.1.1: cuML Logistic Regression (GPU) ---")

# Convert Pandas DataFrames to cuDF DataFrames for GPU
X_train_men_cudf = cudf.DataFrame.from_pandas(X_train_men)
y_train_men_cudf = cudf.Series(y_train_men.values) # Or cudf.DataFrame.from_pandas(y_train_men.to_frame())

logreg_model_gpu = cuMLLogisticRegression(random_state=42, solver='qn', max_iter=200) # 'qn' solver is GPU-accelerated in cuML
logreg_model_gpu.fit(X_train_men_cudf, y_train_men_cudf) # Train on GPU data

print("cuML Logistic Regression (GPU) - Training Completed (Men's)")


print("\n--- Data Types of X_train_men_cudf before Logistic Regression Training ---") # DEBUG PRINT
print(X_train_men_cudf.dtypes) # DEBUG PRINT


# --- 3.1.1: cuML Logistic Regression (GPU) - Women's Model ---
print("\n--- 3.1.1: cuML Logistic Regression (GPU) - Women's Model ---")
X_train_women_cudf = cudf.DataFrame.from_pandas(X_train_women)
y_train_women_cudf = cudf.Series(y_train_women.values)
logreg_model_gpu_women = cuMLLogisticRegression(random_state=42, solver='qn', max_iter=200)
logreg_model_gpu_women.fit(X_train_women_cudf, y_train_women_cudf)
print("cuML Logistic Regression (GPU) - Training Completed (Women's)")


print("\n--- Data Types of X_train_women_cudf before Logistic Regression Training ---") # DEBUG PRINT
print(X_train_women_cudf.dtypes) # DEBUG PRINT


# --- 3.1.2: cuML Random Forest (GPU) ---
print("\n--- 3.1.2: cuML Random Forest (GPU) ---")

rf_model_gpu = cuMLRandomForestClassifier(n_estimators=200,random_state=42)
rf_model_gpu.fit(X_train_men_cudf, y_train_men_cudf) # Train on GPU data

print("cuML Random Forest (GPU) - Training Completed (Men's)")


# --- 3.1.2: cuML Random Forest (GPU) - Women's Model ---
print("\n--- 3.1.2: cuML Random Forest (GPU) - Women's Model ---")
rf_model_gpu_women = cuMLRandomForestClassifier(n_estimators=200,random_state=42)
rf_model_gpu_women.fit(X_train_women_cudf, y_train_women_cudf)
print("cuML Random Forest (GPU) - Training Completed (Women's)")


# --- Phase 3.1.3: XGBoost (cuML/RAPIDS GPU) - Men's Model - Corrected for Boolean dtype ---
print("\n--- Phase 3.1.3: XGBoost (cuML/RAPIDS GPU) - Men's Model - Corrected for Boolean dtype ---\n")

# --- Convert boolean columns to int8 for XGBoost GPU compatibility ---
bool_cols_men = X_train_men_cudf.select_dtypes(include='bool').columns # Identify boolean columns
print(f"\n--- Boolean Columns to Convert for XGBoost (Men's): ---\n{bool_cols_men.tolist()}") # Print boolean columns

for col in bool_cols_men:
    X_train_men_cudf[col] = X_train_men_cudf[col].astype('int8') # Convert boolean to int8

print("\n--- Data Types of X_train_men_cudf AFTER Boolean Conversion, Before XGBoost Training ---") # DEBUG PRINT
print(X_train_men_cudf.dtypes) # DEBUG PRINT

print(X_train_men_cudf.info()) # DEBUG PRINT


xgb_model_gpu_men = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist', device = 'cuda', random_state=42, n_estimators=200) # tree_method='gpu_hist' for GPU
xgb_model_gpu_men.fit(X_train_men_cudf, y_train_men_cudf) # Train on GPU data

print("cuML/RAPIDS XGBoost (GPU) - Training Completed (Men's)")


# --- Phase 3.1.3: XGBoost (cuML/RAPIDS GPU) - Women's Model - Corrected for Boolean dtype (for consistency) ---
print("\n--- Phase 3.1.3: XGBoost (cuML/RAPIDS GPU) - Women's Model - Corrected for Boolean dtype (for consistency) ---\n")

# --- Convert boolean columns to int8 for XGBoost GPU compatibility ---
bool_cols_women = X_train_women_cudf.select_dtypes(include='bool').columns # Identify boolean columns
print(f"\n--- Boolean Columns to Convert for XGBoost (Women's): ---\n{bool_cols_women.tolist()}") # Print boolean columns

for col in bool_cols_women:
    X_train_women_cudf[col] = X_train_women_cudf[col].astype('int8') # Convert boolean to int8

print("\n--- Data Types of X_train_women_cudf AFTER Boolean Conversion, Before XGBoost Training ---") # DEBUG PRINT
print(X_train_women_cudf.dtypes) # DEBUG PRINT
print(X_train_women_cudf.info()) # DEBUG PRINT

xgb_model_gpu_women = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist', device = 'cuda', random_state=42, n_estimators=200) # tree_method='gpu_hist' for GPU
xgb_model_gpu_women.fit(X_train_women_cudf, y_train_women_cudf) # Train on GPU data

print("cuML/RAPIDS XGBoost (GPU) - Training Completed (Women's)")


# --- Phase 3.1.3: LightGBM (GPU) - Men's Model - Corrected for Boolean dtype (and consistency) ---
print("\n--- Phase 3.1.3: LightGBM (GPU) - Men's Model - Corrected for Boolean dtype ---\n")

# --- Convert boolean columns to int8 for LightGBM GPU compatibility (and consistency) ---
bool_cols_men_lgbm = X_train_men_cudf.select_dtypes(include='bool').columns # Identify boolean columns (re-select, could be same as XGBoost bool_cols_men)
print(f"\n--- Boolean Columns to Convert for LightGBM (Men's): ---\n{bool_cols_men_lgbm.tolist()}") # Print boolean columns

for col in bool_cols_men_lgbm:
    X_train_men_cudf[col] = X_train_men_cudf[col].astype('int8') # Convert boolean to int8

print("\n--- Data Types of X_train_men_cudf AFTER Boolean Conversion, Before LightGBM Training ---") # DEBUG PRINT
print(X_train_men_cudf.dtypes) # DEBUG 

# --- Convert y_train_men_cudf to CuPy array for LightGBM GPU compatibility ---
y_train_men_cupy = y_train_men_cudf.to_cupy() # Convert target to CuPy array

print("\n--- Data Types of X_train_men_cudf and y_train_men_cupy Before LightGBM Training ---") # DEBUG PRINT
print("X_train_men_cudf dtypes:", X_train_men_cudf.dtypes) # DEBUG PRINT
print("y_train_men_cupy type:", type(y_train_men_cupy)) # DEBUG PRINT


lgbm_model_gpu_men = lgb.LGBMClassifier(objective='binary', boosting_type='gbdt', device='gpu', random_state=42, n_estimators=200) # device='gpu' for GPU
lgbm_model_gpu_men.fit(X_train_men, y_train_men) # Train on GPU data (X as cuDF, y as CuPy)

print("LightGBM (GPU) - Training Completed (Men's)")


# --- Phase 3.1.3: LightGBM (GPU) - Women's Model - Corrected for Target Variable Format (and consistency) ---
print("\n--- Phase 3.1.3: LightGBM (GPU) - Women's Model - Corrected for Target Variable Format ---\n")

# --- Convert boolean columns to int8 (already present from previous correction) ---
bool_cols_women_lgbm = X_train_women_cudf.select_dtypes(include='bool').columns 
print(f"\n--- Boolean Columns to Convert for LightGBM (Women's): ---\n{bool_cols_women_lgbm.tolist()}") # Print boolean columns

for col in bool_cols_women_lgbm:
    X_train_women_cudf[col] = X_train_women_cudf[col].astype('int8') # Convert boolean to int8

print("\n--- Data Types of X_train_men_cudf AFTER Boolean Conversion, Before LightGBM Training ---") # DEBUG PRINT
print(X_train_women_cudf.dtypes) # DEBUG 

# --- Convert y_train_women_cudf series to CuPy array for LightGBM GPU compatibility ---
y_train_women_cupy = cp.asarray(y_train_women.values) # Convert target to CuPy array

print("\n--- Data Types of X_train_women_cudf and y_train_women_cupy Before LightGBM Training ---") # DEBUG PRINT
print("X_train_women_cudf dtypes:", X_train_women_cudf.dtypes) # DEBUG PRINT
print("y_train_women_cupy type:", type(y_train_women_cupy)) # DEBUG PRINT


lgbm_model_gpu_women = lgb.LGBMClassifier(objective='binary', boosting_type='gbdt', device='gpu', random_state=42, n_estimators=200) # device='gpu' for GPU
# --- CORRECTED FIT CALL: Pass CuPy array for y_train ---
lgbm_model_gpu_women.fit(X_train_women, y_train_women) # Train on GPU data (X as cuDF, y as CuPy)

print("LightGBM (GPU) - Training Completed (Women's)")


# --- Phase 3.2: Data Splitting and Validation Strategy ---
print("\n--- Phase 3.2: Data Splitting and Validation Strategy ---\n")

# --- 3.2.1: Time-Based Train-Validation Split ---
print("\n--- 3.2.1: Time-Based Train-Validation Split ---")

validation_season = 2024 # Example: Use 2024 season for validation

def time_based_split(matchup_df, validation_season, dataframe_name): # Added dataframe_name parameter
    """Splits matchup DataFrame into time-based train and validation sets, recognizes dataframe name."""
    train_df = matchup_df[matchup_df['Season'] < validation_season]
    val_df = matchup_df[matchup_df['Season'] == validation_season]
    
    # Dynamically select feature columns based on dataframe_name
    if dataframe_name == 'm_matchup_features_train_df':
        feature_cols = feature_cols_men # Use men's feature columns for men's dataframe
    elif dataframe_name == 'w_matchup_features_train_df':
        feature_cols = feature_cols_women # Use women's feature columns for women's dataframe
    else:
        raise ValueError("Invalid dataframe_name. Must be 'm_matchup_features_train_df' or 'w_matchup_features_train_df'")

    X_train = train_df[feature_cols].copy() 
    y_train = train_df['Team1_Win'].copy()
    X_val = val_df[feature_cols].copy()
    y_val = val_df['Team1_Win'].copy()
    return X_train, y_train, X_val, y_val


# --- Convert boolean columns to int8 (already present from previous correction) ---
bool_cols_men = m_matchup_features_train_df.select_dtypes(include='bool').columns 
print(f"\n--- Boolean Columns to Convert for CUDF (Men's): ---\n{bool_cols_men.tolist()}") # Print boolean columns

for col in bool_cols_men:
    m_matchup_features_train_df[col] = m_matchup_features_train_df[col].astype('int8') # Convert boolean to int8

print("\n--- Data Types of X_train_men_cudf AFTER Boolean Conversion ---") # DEBUG PRINT
print(m_matchup_features_train_df.dtypes) # DEBUG 



# --- Convert boolean columns to int8 (already present from previous correction) ---
bool_cols_women_lgbm = w_matchup_features_train_df.select_dtypes(include='bool').columns 
print(f"\n--- Boolean Columns to Convert for LightGBM (Women's): ---\n{bool_cols_women_lgbm.tolist()}") # Print boolean columns

for col in bool_cols_women_lgbm:
    w_matchup_features_train_df[col] = w_matchup_features_train_df[col].astype('int8') # Convert boolean to int8

print("\n--- Data Types of X_train_men_cudf AFTER Boolean Conversion, Before LightGBM Training ---") # DEBUG PRINT
print(w_matchup_features_train_df.dtypes) # DEBUG 



X_train_men_time, y_train_men_time, X_val_men_time, y_val_men_time = time_based_split(m_matchup_features_train_df, validation_season, dataframe_name='m_matchup_features_train_df') # Pass dataframe_name
X_train_women_time, y_train_women_time, X_val_women_time, y_val_women_time = time_based_split(w_matchup_features_train_df, validation_season, dataframe_name='w_matchup_features_train_df') # Pass dataframe_name


# Convert Time-Split data to cuDF for GPU (optional - can convert later when training/validating)
X_train_men_time_cudf = cudf.DataFrame.from_pandas(X_train_men_time)
y_train_men_time_cudf = cudf.Series(y_train_men_time.values)
X_val_men_time_cudf = cudf.DataFrame.from_pandas(X_val_men_time)
y_val_men_time_cudf = cudf.Series(y_val_men_time.values)

X_train_women_time_cudf = cudf.DataFrame.from_pandas(X_train_women_time)
y_train_women_time_cudf = cudf.Series(y_train_women_time.values)
X_val_women_time_cudf = cudf.DataFrame.from_pandas(X_val_women_time)
y_val_women_time_cudf = cudf.Series(y_val_women_time.values)



print("\n--- Time-Based Train-Validation Split Completed (Men's Data) ---")
print(f"Men's X_train shape: {X_train_men_time.shape}, y_train shape: {y_train_men_time.shape}")
print(f"Men's X_val shape: {X_val_men_time.shape}, y_val shape: {y_val_men_time.shape}")
print("\n--- Time-Based Train-Validation Split Completed (Women's Data) ---")
print(f"Women's X_train shape: {X_train_women_time.shape}, y_train shape: {y_train_women_time.shape}")
print(f"Women's X_val shape: {X_val_women_time.shape}, y_val shape: {y_val_women_time.shape}")


print("\n--- Time-Based Train-Validation Split Completed (Men's Data) ---")
print(f"Men's X_train shape: {X_train_men_time_cudf.shape}, y_train shape: {y_train_men_time_cudf.shape}")
print(f"Men's X_val shape: {X_val_men_time_cudf.shape}, y_val shape: {y_val_men_time_cudf.shape}")
print("\n--- Time-Based Train-Validation Split Completed (Women's Data) ---")
print(f"Women's X_train shape: {X_train_women_time_cudf.shape}, y_train shape: {y_train_women_time_cudf.shape}")
print(f"Women's X_val shape: {X_val_women_time_cudf.shape}, y_val shape: {y_val_women_time_cudf.shape}")


print(X_train_men_time_cudf.info())


print(w_matchup_features_train_df.info())


# --- 3.2.2: K-Fold Cross-Validation Setup - Separate for Men and Women ---
print("\n--- 3.2.2: K-Fold Cross-Validation Setup - Separate for Men and Women ---\n")

n_folds = 5 # Example: 5-fold CV


# K-Fold for Men's Data
kf_men = KFold(n_splits=n_folds, shuffle=True, random_state=42) # KFold for Men
print("\n--- K-Fold Cross-Validation Splitter (Men's Data) ---")
for fold, (train_index, val_index) in enumerate(kf_men.split(X_train_men, y_train_men)): # Split using CPU data for indexing
    print(f"\n--- Fold {fold+1} (Men's) ---")
    print(f"  Train indices (first 5): {train_index[:5]}")
    print(f"  Validation indices (first 5): {val_index[:5]}")
    X_train_fold_cpu, y_train_fold_cpu = X_train_men.iloc[train_index], y_train_men.iloc[train_index] # CPU Fold Data (Pandas)
    X_val_fold_cpu, y_val_fold_cpu = X_train_men.iloc[val_index], y_train_men.iloc[val_index] # CPU Validation Fold Data (Pandas)
    # Convert Fold Data to cuDF for GPU (Do this when you actually train/validate in each fold in Phase 3.3)
    X_train_fold_cudf = cudf.DataFrame.from_pandas(X_train_fold_cpu)
    y_train_fold_cudf = cudf.Series(y_train_fold_cpu.values)
    X_val_fold_cudf = cudf.DataFrame.from_pandas(X_val_fold_cpu)
    y_val_fold_cudf = cudf.Series(y_val_fold_cpu.values)
    print(f"  Fold X_train shape (CPU): {X_train_fold_cpu.shape}, Fold y_train shape (CPU): {y_train_fold_cpu.shape}")
    print(f"  Fold X_val shape (CPU): {X_val_fold_cpu.shape}, Fold y_val shape (CPU): {y_val_fold_cpu.shape}")
    print(f"  Fold X_train shape (GPU): {X_train_fold_cudf.shape}, Fold y_train shape (GPU): {y_train_fold_cudf.shape}")
    print(f"  Fold X_val shape (GPU): {X_val_fold_cudf.shape}, Fold y_val shape (GPU): {y_val_fold_cudf.shape}")
    


# Stratified K-Fold for Women's Data (Example - if class imbalance is suspected)
skf_women = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42) # StratifiedKFold for Women
print("\n--- Stratified K-Fold Cross-Validation Splitter (Women's Data) ---")
for fold, (train_index, val_index) in enumerate(skf_women.split(X_train_women, y_train_women)): # Split using CPU data
    print(f"\n--- Fold {fold+1} (Women's) ---")
    print(f"  Train indices (first 5): {train_index[:5]}")
    print(f"  Validation indices (first 5): {val_index[:5]}")
    X_train_fold_cpu, y_train_fold_cpu = X_train_women.iloc[train_index], y_train_women.iloc[train_index] # CPU Fold Data (Pandas)
    X_val_fold_cpu, y_val_fold_cpu = X_train_women.iloc[val_index], y_train_women.iloc[val_index] # CPU Validation Fold Data (Pandas)
    # Convert Fold Data to cuDF for GPU (Do this when you actually train/validate in each fold in Phase 3.3)
    X_train_fold_cudf = cudf.DataFrame.from_pandas(X_train_fold_cpu)
    y_train_fold_cudf = cudf.Series(y_train_fold_cpu.values)
    X_val_fold_cudf = cudf.DataFrame.from_pandas(X_val_fold_cpu)
    y_val_fold_cudf = cudf.Series(y_val_fold_cpu.values)
    print(f"  Fold X_train shape (CPU): {X_train_fold_cpu.shape}, Fold y_train shape (CPU): {y_train_fold_cpu.shape}")
    print(f"  Fold X_val shape (CPU): {X_val_fold_cpu.shape}, Fold y_val shape (CPU): {y_val_fold_cpu.shape}")
    print(f"  Fold X_train shape (GPU): {X_train_fold_cudf.shape}, Fold y_train shape (GPU): {y_train_fold_cudf.shape}")
    print(f"  Fold X_val shape (GPU): {X_val_fold_cudf.shape}, Fold y_val shape (GPU): {y_val_fold_cudf.shape}")

print("\n--- Phase 3.2: Data Splitting and Validation Strategy Setup Completed (Men's and Women's) ---\n")


# --- Phase 3.3: Model Training and Hyperparameter Tuning - Bayesian Optimization with Ax Client (GPU Accelerated) ---
print("\n--- Phase 3.3: Model Training and Hyperparameter Tuning - Bayesian Optimization with Ax Client (GPU Accelerated) ---\n")


# --- Initialize Ax Client for Experiment Management ---
ax_client_men = AxClient()
ax_client_men.create_experiment(
    name="Men_Model_Selection_and_Tuning",
    parameters=[
        # Model selection parameter
        {"name": "model_type", "type": "choice", "values": ["logreg", "rf", "xgb", "lgbm"]},

        # --- Hyperparameters for Logistic Regression - CORRECTED 'dependents' SPECIFICATION (DICTIONARY) ---
        {"name": "logreg_C", "type": "range", "bounds": [float(1e-4), float(1e2)], "log_scale": False, # Linear scale in log10 domain
         "dependents": {"model_type": ["logreg"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "logreg_fit_intercept", "type": "choice", "values": [0, 1], # 0 for False, 1 for True
         "value_type": "int",  "is_ordered": True}, # CORRECTED: Use dictionary for dependents

        # --- Hyperparameters for Random Forest - CORRECTED 'dependents' SPECIFICATION (DICTIONARY) ---
        {"name": "rf_n_estimators", "type": "range", "bounds": [100, 500], "value_type": "int",
         "dependents": {"model_type": ["rf"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "rf_max_depth", "type": "range", "bounds": [5, 15], "value_type": "int",
         "dependents": {"model_type": ["rf"]}}, # CORRECTED: Use dictionary for dependents

        # --- Hyperparameters for XGBoost - CORRECTED 'dependents' SPECIFICATION (DICTIONARY) ---
        {"name": "xgb_learning_rate", "type": "range", "bounds": [0.01, 0.3], "log_scale": True,
         "dependents": {"model_type": ["xgb"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "xgb_max_depth", "type": "range", "bounds": [3, 10], "value_type": "int",
         "dependents": {"model_type": ["xgb"]}, # CORRECTED: Use dictionary for dependents
        "name": "xgb_reg_alpha", "type": "range", "bounds": [0.0, 1.0],
         "dependents": {"model_type": ["xgb"]}}, # CORRECTED: Use dictionary for dependents

        # --- Hyperparameters for LightGBM - CORRECTED 'dependents' SPECIFICATION (DICTIONARY) ---
        {"name": "lgbm_learning_rate", "type": "range", "bounds": [0.01, 0.3], "log_scale": True,
         "dependents": {"model_type": ["lgbm"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "lgbm_num_leaves", "type": "range", "bounds": [31, 127], "value_type": "int",
         "dependents": {"model_type": ["lgbm"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "lgbm_reg_alpha", "type": "range", "bounds": [0.0, 1.0],
         "dependents": {"model_type": ["lgbm"]}}, # CORRECTED: Use dictionary for dependents
    ],
    # We want our models to focus on improving the value of brier_score by exploring and exploiting different parameter combinations.
    objectives = {"brier_score": ObjectiveProperties(minimize=True)},
)




# --- Evaluation Function for Ax Client - Model Selection and Hyperparameter Tuning ---
def evaluate_model_ax(parameterization):
    """
    Evaluation function for Ax Client, handling Model Selection and Hyperparameter Tuning.
    """
    model_type = parameterization['model_type'] # Get model_type from Ax parameterization

    print(f"\n--- Evaluate_model_ax - Model Type: {model_type}, Hyperparameters: {parameterization} ---") # DEBUG PRINT - Model type and hyperparams

    if model_type == 'logreg':
        # --- cuML Logistic Regression ---
        model = cuMLLogisticRegression(
            C=10**parameterization.get("logreg_C", -2.0), # Get logreg_C, default to 10**-2 if not found
            fit_intercept=bool(parameterization.get("logreg_fit_intercept", 1)), # Get logreg_fit_intercept, default True if not found
            solver='qn',
            random_state=42
        )
    elif model_type == 'rf':
        # --- cuML Random Forest ---
        model = cuMLRandomForestClassifier(
            n_estimators=parameterization.get("rf_n_estimators", 200), # Get rf_n_estimators, default 200 if not found
            max_depth=parameterization.get("rf_max_depth", 10), # Get rf_max_depth, default 10 if not found
            random_state=42
        )
    elif model_type == 'xgb':
        # --- XGBoost (cuML/RAPIDS GPU) ---
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            device='gpu',
            random_state=42,
            learning_rate=parameterization.get("xgb_learning_rate", 0.1), # Get xgb_learning_rate, default 0.1 if not found
            max_depth=parameterization.get("xgb_max_depth", 5), # Get xgb_max_depth, default 5 if not found
            reg_alpha=parameterization.get("xgb_reg_alpha", 0.0) # Get xgb_reg_alpha, default 0.0 if not found
        )
    elif model_type == 'lgbm':
        # --- LightGBM (GPU) ---
        model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            device='gpu',
            random_state=42,
            learning_rate=parameterization.get("lgbm_learning_rate", 0.1), # Get lgbm_learning_rate, default 0.1 if not found
            num_leaves=parameterization.get("lgbm_num_leaves", 31), # Get lgbm_num_leaves, default 31 if not found
            reg_alpha=parameterization.get("lgbm_reg_alpha", 0.0) # Get lgbm_reg_alpha, default 0.0 if not found
        )
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


    # --- Train Model and Predict (Using Time-Based Split Validation Data) ---
    if model_type == 'lgbm':
        # --- LightGBM (GPU) ---
        # Convert y_train_men_time_cudf.values to NumPy array explicitly for LightGBM
        model.fit(X_train_men_time_cudf.to_numpy(), y_train_men_time_cudf.values.get())
    else:
        # --- Other Models (cuML Logistic Regression, cuML Random Forest, XGBoost) ---
        # Use current variables (cuDF Series) for other models
        model.fit(X_train_men_time_cudf, y_train_men_time_cudf.values)
    #model.fit(X_train_men_time_cudf, y_train_men_time_cudf.values) # Train on GPU time-based training data
    print("\n--- Debugging Prediction in evaluate_model_ax ---") # ADDED DEBUG PRINTS
    print("Shape of X_val_men_time_cudf:", X_val_men_time_cudf.shape) # DEBUG PRINT - Shape of validation features
    print("Type of X_val_men_time_cudf:", type(X_val_men_time_cudf)) # DEBUG PRINT - Type of validation features
    if model_type == 'lgbm':
        # --- LightGBM (GPU) ---
        # Convert y_val_men_time_cudf.values to NumPy array explicitly for LightGBM
        val_preds_gpu = model.predict_proba(X_val_men_time_cudf.to_numpy()) # Predict on GPU time-based validation data
    else:
        # --- Other Models (cuML Logistic Regression, cuML Random Forest, XGBoost) ---
        # Use current variables (cuDF Series) for other models
        val_preds_gpu = model.predict_proba(X_val_men_time_cudf) # Predict on GPU time-based validation data
    #val_preds_cpu = val_preds_gpu.values
   # --- MODIFIED: Check if val_preds_gpu is a NumPy array ---
    if isinstance(val_preds_gpu, np.ndarray):
        val_preds_cpu = val_preds_gpu  # If it's a NumPy array, assign directly
    else:
        val_preds_cpu = val_preds_gpu.values.get() # If it's a cuDF Series/DataFrame, get values

    y_val_cpu = y_val_men_time_cudf.values.get()
    print(f"  Shape of y_val_cpu: {y_val_cpu.shape}") # DEBUG PRINT - Shape of validation target
    print("Type of y_val_cpu:", type(y_val_cpu))
    print(f"  Shape of val_preds_cpu: {val_preds_cpu.shape}") # DEBUG PRINT - Shape of validation predictions
    # --- CORRECTED SLICING: Extract probabilities for positive class (column 1) ---
    val_preds_cpu_positive_class = val_preds_cpu[:, 1] # Extract probabilities for positive class (column 1) - CORRECTED SLICING
    # --- MODIFIED: Remove unnecessary .get() call ---
    #val_preds_cpu_positive_class = val_preds_cpu_positive_class.get()
    print(f"  Shape of val_preds_cpu_positive_class: {val_preds_cpu_positive_class.shape}") # DEBUG PRINT - Shape of validation predictions
    print("Type of val_preds_cpu_positive_class:", type(val_preds_cpu_positive_class))
    brier_score_val = brier_score_loss(y_val_cpu, val_preds_cpu_positive_class) # Calculate Brier score on CPU
    print(f"  Calculated Brier Score (Fold): {brier_score_val:.5f}") # DEBUG PRINT - Brier score before negation

    return {"brier_score": (float(brier_score_val), 0.0)} # Return metric as tuple: (mean, SEM) - SEM is 0.0 as we are using a fixed validation set


# --- Run Bayesian Optimization with Ax Client ---
N_TRIALS_AX = 50 # Example number of trials for Ax Client
print("\n--- Running Bayesian Optimization with Ax Client (Men's Data) ---\n")

for i in range(N_TRIALS_AX):
    print(f"\n--- Ax Trial {i+1}/{N_TRIALS_AX} ---")
    trial = ax_client_men.get_next_trial() # Get next trial from Ax Client - suggests hyperparameters
    # --- CORRECTED: Handle both Trial object and tuple return from get_next_trial() ---
    if isinstance(trial, tuple): # Check if trial is a tuple (older Ax versions)
        trial_arm_parameters, trial_index = trial # Unpack tuple if it's a tuple
        suggested_hyperparams = trial_arm_parameters # Hyperparameters are directly in the tuple
    else: # Assume it's a Trial object (newer Ax versions)
        trial_index = trial.index
        suggested_hyperparams = trial.arm.parameter_values # Access hyperparams from trial.arm.parameter_values

    print(f"--- Suggested Parameterization: {suggested_hyperparams} ---") # Print suggested hyperparams

    # --- Evaluate model with suggested hyperparameters using evaluate_model_ax function ---
    evaluation_results = evaluate_model_ax(suggested_hyperparams)
    ax_client_men.complete_trial(trial_index=trial_index, raw_data=evaluation_results) # Report results back to Ax Client


print("\n--- Bayesian Optimization with Ax Client (Men's Data) Completed ---\n")


best_arm_and_metrics_men = ax_client_men.get_best_trial() # Get best trial
print("\n--- Best Trial and Metrics (Men's Data) ---")
print(best_arm_and_metrics_men)


best_parameters_men = ax_client_men.get_best_parameters()[0] # Get best parameters
print("\n--- Best Parameters (Men's Data) ---")
print(best_parameters_men)


print("--- Store Optimization Trace (Optional - for analysis and visualization) for men---")


# --- Store Optimization Trace (Optional - for analysis and visualization) ---
optimization_trace_logreg_men_df = pd.DataFrame(ax_client_men.get_trials_data_frame())
print("\n--- Optimization Trace (First 5 Iterations) ---")
print(optimization_trace_logreg_men_df.head())


print("\n--- Optimization Trace (Last 5 Iterations) ---")
print(optimization_trace_logreg_men_df.tail(5))


# --- Phase 3.3: Model Training and Hyperparameter Tuning - Bayesian Optimization with Ax Client (GPU Accelerated) - WOMEN'S DATA ---
print("\n--- Phase 3.3: Model Training and Hyperparameter Tuning - Bayesian Optimization with Ax Client (GPU Accelerated) - WOMEN'S DATA ---\n")


# --- Initialize Ax Client for Experiment Management - Women's Data ---
ax_client_women = AxClient() # Separate AxClient for women's experiment
ax_client_women.create_experiment(
    name="Women_Model_Selection_and_Tuning", # Different experiment name for women
    parameters=[
        # Model selection parameter
        {"name": "model_type", "type": "choice", "values": ["logreg", "rf", "xgb", "lgbm"]},

        # --- Hyperparameters for Logistic Regression - CORRECTED 'dependents' SPECIFICATION (DICTIONARY) ---
        {"name": "logreg_C", "type": "range", "bounds": [float(1e-4), float(1e2)], "log_scale": False, # Linear scale in log10 domain
         "dependents": {"model_type": ["logreg"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "logreg_fit_intercept", "type": "choice", "values": [0, 1], # 0 for False, 1 for True
         "value_type": "int",  "is_ordered": True}, # CORRECTED: Use dictionary for dependents

        # --- Hyperparameters for Random Forest - CORRECTED 'dependents' SPECIFICATION (DICTIONARY) ---
        {"name": "rf_n_estimators", "type": "range", "bounds": [100, 500], "value_type": "int",
         "dependents": {"model_type": ["rf"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "rf_max_depth", "type": "range", "bounds": [5, 15], "value_type": "int",
         "dependents": {"model_type": ["rf"]}}, # CORRECTED: Use dictionary for dependents

        # --- Hyperparameters for XGBoost - CORRECTED 'dependents' SPECIFICATION (DICTIONARY) ---
        {"name": "xgb_learning_rate", "type": "range", "bounds": [0.01, 0.3], "log_scale": True,
         "dependents": {"model_type": ["xgb"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "xgb_max_depth", "type": "range", "bounds": [3, 10], "value_type": "int",
         "dependents": {"model_type": ["xgb"]}, # CORRECTED: Use dictionary for dependents
        "name": "xgb_reg_alpha", "type": "range", "bounds": [0.0, 1.0],
         "dependents": {"model_type": ["xgb"]}}, # CORRECTED: Use dictionary for dependents

        # --- Hyperparameters for LightGBM - CORRECTED 'dependents' SPECIFICATION (DICTIONARY) ---
        {"name": "lgbm_learning_rate", "type": "range", "bounds": [0.01, 0.3], "log_scale": True,
         "dependents": {"model_type": ["lgbm"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "lgbm_num_leaves", "type": "range", "bounds": [31, 127], "value_type": "int",
         "dependents": {"model_type": ["lgbm"]}}, # CORRECTED: Use dictionary for dependents
        {"name": "lgbm_reg_alpha", "type": "range", "bounds": [0.0, 1.0],
         "dependents": {"model_type": ["lgbm"]}}, # CORRECTED: Use dictionary for dependents
    ],
    # We want our models to focus on improving the value of brier_score by exploring and exploiting different parameter combinations.
    objectives = {"brier_score": ObjectiveProperties(minimize=True)},
)


# --- Evaluation Function for Ax Client - Model Selection and Hyperparameter Tuning - WOMEN'S DATA ---
def evaluate_model_ax_women(parameterization): # Different evaluation function for women's data
    """
    Evaluation function for Ax Client, handling Model Selection and Hyperparameter Tuning - WOMEN'S DATA.
    """
    model_type = parameterization['model_type'] # Get model_type from Ax parameterization

    if model_type == 'logreg':
        # --- cuML Logistic Regression ---
        model = cuMLLogisticRegression(
            C=10**parameterization.get("logreg_C", -2.0),
            fit_intercept=bool(parameterization.get("logreg_fit_intercept", 1)),
            solver='qn',
            random_state=42
        )
    elif model_type == 'rf':
        # --- cuML Random Forest ---
        model = cuMLRandomForestClassifier(
            n_estimators=parameterization.get("rf_n_estimators", 200),
            max_depth=parameterization.get("rf_max_depth", 10),
            random_state=42
        )
    elif model_type == 'xgb':
        # --- XGBoost (cuML/RAPIDS GPU) ---
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            device='gpu',
            random_state=42,
            learning_rate=parameterization.get("xgb_learning_rate", 0.1),
            max_depth=parameterization.get("xgb_max_depth", 5),
            reg_alpha=parameterization.get("xgb_reg_alpha", 0.0)
        )
    elif model_type == 'lgbm':
        # --- LightGBM (GPU) ---
        model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            device='gpu',
            random_state=42,
            learning_rate=parameterization.get("lgbm_learning_rate", 0.1),
            num_leaves=parameterization.get("lgbm_num_leaves", 31),
            reg_alpha=parameterization.get("lgbm_reg_alpha", 0.0)
        )
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


    # --- Train Model and Predict (Using Time-Based Split Validation Data) ---
    if model_type == 'lgbm':
        # --- LightGBM (GPU) ---
        # Convert X_train_women_time_cudf.values to NumPy array explicitly for LightGBM
        model.fit(X_train_women_time_cudf.to_numpy(), y_train_women_time_cudf.values.get())
    else:
        # --- Other Models (cuML Logistic Regression, cuML Random Forest, XGBoost) ---
        # Use current variables (cuDF Series) for other models
        model.fit(X_train_women_time_cudf, y_train_women_time_cudf.values)
    #model.fit(X_train_women_time_cudf, y_train_women_time_cudf.values) # Train on GPU time-based training data
    print("\n--- Debugging Prediction in evaluate_model_ax ---") # ADDED DEBUG PRINTS
    print("Shape of X_val_women_time_cudf:", X_val_women_time_cudf.shape) # DEBUG PRINT - Shape of validation features
    print("Type of X_val_women_time_cudf:", type(X_val_women_time_cudf)) # DEBUG PRINT - Type of validation features
    if model_type == 'lgbm':
        # --- LightGBM (GPU) ---
        # Convert X_val_women_time_cudf.values to NumPy array explicitly for LightGBM
        val_preds_gpu = model.predict_proba(X_val_women_time_cudf.to_numpy()) # Predict on GPU time-based validation data
    else:
        # --- Other Models (cuML Logistic Regression, cuML Random Forest, XGBoost) ---
        # Use current variables (cuDF Series) for other models
        val_preds_gpu = model.predict_proba(X_val_women_time_cudf) # Predict on GPU time-based validation data
    #val_preds_cpu = val_preds_gpu.values
   # --- MODIFIED: Check if val_preds_gpu is a NumPy array ---
    if isinstance(val_preds_gpu, np.ndarray):
        val_preds_cpu = val_preds_gpu  # If it's a NumPy array, assign directly
    else:
        val_preds_cpu = val_preds_gpu.values.get() # If it's a cuDF Series/DataFrame, get values

    y_val_cpu = y_val_women_time_cudf.values.get()
    print(f"  Shape of y_val_cpu: {y_val_cpu.shape}") # DEBUG PRINT - Shape of validation target
    print("Type of y_val_cpu:", type(y_val_cpu))
    print(f"  Shape of val_preds_cpu: {val_preds_cpu.shape}") # DEBUG PRINT - Shape of validation predictions
    # --- CORRECTED SLICING: Extract probabilities for positive class (column 1) ---
    val_preds_cpu_positive_class = val_preds_cpu[:, 1] # Extract probabilities for positive class (column 1) - CORRECTED SLICING
    # --- MODIFIED: Remove unnecessary .get() call ---
    #val_preds_cpu_positive_class = val_preds_cpu_positive_class.get()
    print(f"  Shape of val_preds_cpu_positive_class: {val_preds_cpu_positive_class.shape}") # DEBUG PRINT - Shape of validation predictions
    print("Type of val_preds_cpu_positive_class:", type(val_preds_cpu_positive_class))
    brier_score_val = brier_score_loss(y_val_cpu, val_preds_cpu_positive_class) # Calculate Brier score on CPU
    print(f"  Calculated Brier Score (Fold): {brier_score_val:.5f}") # DEBUG PRINT - Brier score before negation


    return {"brier_score": (float(brier_score_val), 0.0)} # Return metric as tuple: (mean, SEM)


# --- Run Bayesian Optimization with Ax Client - WOMEN'S DATA ---
N_TRIALS_AX = 50 # Example number of trials for Ax Client
print("\n--- Running Bayesian Optimization with Ax Client (Women's Data) ---\n")

for i in range(N_TRIALS_AX):
    print(f"\n--- Ax Trial {i+1}/{N_TRIALS_AX} (Women's) ---") # Indicate Women's data in print
    trial = ax_client_women.get_next_trial() # Get next trial from Ax Client for WOMEN'S experiment
    # --- CORRECTED: Handle both Trial object and tuple return from get_next_trial() ---
    if isinstance(trial, tuple): # Check if trial is a tuple (older Ax versions)
        trial_arm_parameters, trial_index = trial # Unpack tuple if it's a tuple
        suggested_hyperparams = trial_arm_parameters # Hyperparameters are directly in the tuple
    else: # Assume it's a Trial object (newer Ax versions)
        trial_index = trial.index
        suggested_hyperparams = trial.arm.parameter_values # Access hyperparams from trial.arm.parameter_values

    print(f"--- Suggested Parameterization: {suggested_hyperparams} ---") # Print suggested hyperparams

    # --- Evaluate model with suggested hyperparameters using evaluate_model_ax_women function --- # Use WOMEN'S evaluation function
    evaluation_results = evaluate_model_ax_women(suggested_hyperparams)
    # Pass the trial_index which is set correctly in the if-else block
    ax_client_women.complete_trial(trial_index=trial_index, raw_data=evaluation_results) # Report results back to Ax Client for WOMEN'S experiment


print("\n--- Bayesian Optimization with Ax Client (Women's Data) Completed ---\n")


best_arm_and_metrics_women = ax_client_women.get_best_trial() # Get best trial for WOMEN'S experiment
print("\n--- Best Trial and Metrics (Women's Data) ---")
print(best_arm_and_metrics_women)


best_parameters_women = ax_client_women.get_best_parameters()[0] # Get best parameters for WOMEN'S experiment
print("\n--- Best Parameters (Women's Data) ---")
print(best_parameters_women)


# --- Store Optimization Trace (Optional - for analysis and visualization) ---
optimization_trace_women_df = pd.DataFrame(ax_client_women.get_trials_data_frame()) # Get all trial data for women's experiment
print("\n--- Optimization Trace (First 5 Iterations - Women's Data) ---")
print(optimization_trace_women_df.head())


print("\n--- Optimization Trace (Last 5 Iterations - Women's Data) ---")
print(optimization_trace_women_df.tail(5))


# --- 4.1: Confirm dataset has 2025 data in it (Men's) and (Women's) and Build Final Prediction Datasets ---
print("\n--- 4.1.1: Identify 2025 Teams Matchups Completed ---\n")


# --- 4.2: Confirm dataset has 2025 data in it (Men's) and (Women's) and Build Final Prediction Datasets ---
print("\n--- 4.1.2: Confirm dataset has 2025 data in it (Men's) and (Women's) and Build Final Prediction Datasets ---\n")


# --- Convert boolean columns to int8 for Model compatibility ---
bool_cols_men = m_matchup_features_full_dataset_train_df.select_dtypes(include='bool').columns # Identify boolean columns
print(f"\n--- Boolean Columns to Convert for Model (Men's): ---\n{bool_cols_men.tolist()}") # Print boolean columns

for col in bool_cols_men:
    m_matchup_features_full_dataset_train_df[col] = m_matchup_features_full_dataset_train_df[col].astype('int8') # Convert boolean to int8

print("\n--- Data Types of m_matchup_features_full_dataset_train_df AFTER Boolean Conversion, Before Model Training ---") # DEBUG PRINT
print(m_matchup_features_full_dataset_train_df.dtypes) # DEBUG PRINT


# --- Convert boolean columns to int8 for Model compatibility ---
bool_cols_women = w_matchup_features_full_dataset_train_df.select_dtypes(include='bool').columns # Identify boolean columns
print(f"\n--- Boolean Columns to Convert for Model (Women's): ---\n{bool_cols_men.tolist()}") # Print boolean columns

for col in bool_cols_women:
    w_matchup_features_full_dataset_train_df[col] = w_matchup_features_full_dataset_train_df[col].astype('int8') # Convert boolean to int8

print("\n--- Data Types of m_matchup_features_full_dataset_train_df AFTER Boolean Conversion, Before Model Training ---") # DEBUG PRINT
print(w_matchup_features_full_dataset_train_df.dtypes) # DEBUG PRINT


# Reconfirm there are no nulls in the datasets
print(f"Number of records in m_matchup_features_full_dataset_train_df: {len(m_matchup_features_full_dataset_train_df)}")
print(f"Number of records in w_matchup_features_full_dataset_train_df: {len(w_matchup_features_full_dataset_train_df)}")
print(f"Number of null values in m_matchup_features_full_dataset_train_df: {m_matchup_features_full_dataset_train_df.isnull().sum().sum()}")
print(f"Number of null values in w_matchup_features_full_dataset_train_df: {w_matchup_features_full_dataset_train_df.isnull().sum().sum()}")


# No nulls in dataset split into X and y
if m_matchup_features_full_dataset_train_df.isnull().sum().sum() == 0 and w_matchup_features_full_dataset_train_df.isnull().sum().sum() == 0:
  y_full_dataset_train_men = m_matchup_features_full_dataset_train_df['Team1_Win']
  feature_cols_men = [col for col in m_matchup_features_full_dataset_train_df.columns if col not in ['Team1_Win', 'Season', 'Team1ID', 'Team2ID']]
  X_full_dataset_train_men = m_matchup_features_full_dataset_train_df[feature_cols_men]

  y_full_dataset_train_women = w_matchup_features_full_dataset_train_df['Team1_Win']
  feature_cols_women = [col for col in w_matchup_features_full_dataset_train_df.columns if col not in ['Team1_Win', 'Season', 'Team1ID', 'Team2ID']]
  X_full_dataset_train_women = w_matchup_features_full_dataset_train_df[feature_cols_women]
else:
  print("Error: Null values found in dataset split into X and y")


# Convert Pandas DataFrames to cuDF DataFrames for GPU
X_full_dataset_train_men_cudf = cudf.DataFrame.from_pandas(X_full_dataset_train_men)
y_full_dataset_train_men_cudf = cudf.Series(y_full_dataset_train_men.values) # Or cudf.DataFrame.from_pandas(y_train_men.to_frame())

X_full_dataset_train_women_cudf = cudf.DataFrame.from_pandas(X_full_dataset_train_women)
y_full_dataset_train_women_cudf = cudf.Series(y_full_dataset_train_women.values)


if (m_team_season_stats_selected_features_df['Season'] == 2025).any() and (w_team_season_stats_selected_features_df['Season'] == 2025).any():
    print("\n--- 2025 Season Data Found in m_team_season_stats_selected_features_df and w_team_season_stats_selected_features_df ---\n")


# --- 4.2: Confirm dataset has 2025 data in it (Men's) and (Women's) and Build Final Prediction Datasets ---
print("\n--- 4.1.2: Confirm dataset has 2025 data in it (Men's) and (Women's) and Build Final Prediction Datasets Completed ---\n")


# prompt: Build Men's and Women's matchup datasets, using m_team_season_stats_selected_features_df and w_team_season_stats_selected_features_df, which is just the a list of 'TeamID' where 'LastD1Season' = 2025. Should be a Python  Python lists of tuples, where each tuple represents a matchup as (TeamID1, TeamID2) with TeamID1 < TeamID2.

def build_matchup_dataset(team_season_stats_df,downloaded_submission_stage_df):
    """
    Builds a matchup dataset from team season statistics.

    Args:
        team_season_stats_df: DataFrame containing team season statistics.

    Returns:
        A list of tuples, where each tuple represents a matchup (TeamID1, TeamID2) with TeamID1 < TeamID2.
    """

    # Filter for teams in the 2025 season.
    teams_2025 = team_season_stats_df[team_season_stats_df['Season'] == 2025]['TeamID'].tolist()

    # Load SampleSubmissionStage2.csv to get the required matchups
    submission_df = downloaded_submission_stage_df
    required_matchups = []
    for index, row in submission_df.iterrows():
        matchup_id = row['ID'].split('_')
        team1_id = int(matchup_id[1])
        team2_id = int(matchup_id[2])
        required_matchups.append((team1_id, team2_id))

    # Build matchups, ensuring they are in required_matchups
    matchups = []
    for team1_id, team2_id in required_matchups:
        if team1_id in teams_2025 and team2_id in teams_2025:
            matchups.append((team1_id, team2_id))

    return matchups

# Assuming m_team_season_stats_selected_features_df, 
# w_team_season_stats_selected_features_df, and data_dir are defined
men_matchups = build_matchup_dataset(m_team_season_stats_selected_features_df, sample_submission_stage_2_df)
women_matchups = build_matchup_dataset(w_team_season_stats_selected_features_df, sample_submission_stage_2_df)

print("Men's Matchups (first 5):", men_matchups[:5])
print("Women's Matchups (first 5):", women_matchups[:5])



print("\n--- Phase 4.1.1: Identify 2025 Teams Matchups Completed ---\n")


print("\n--- Phase 4.2: Extract Model Parameters for Final Model Build ---\n")


def extract_model_params(best_arm_metrics, best_params):
    # Extract model type from best_arm_metrics
    match = re.search(r"'model_type': '(\w+)'", str(best_arm_metrics))
    if match:
        model_type = match.group(1)
    else:
        print("Error: Could not extract 'model_type' from best_arm_metrics.")
        return None, None

    # Extract relevant parameters based on the model type
    params_dict = best_params
    final_model = {}
    if model_type == "logreg":
      final_model["C"] = params_dict.get("logreg_C")
      final_model["fit_intercept"] = bool(params_dict.get("logreg_fit_intercept"))
      final_model["solver"] = "qn"
      final_model["random_state"] = 42

    elif model_type == "rf":
      final_model["n_estimators"] = params_dict.get("rf_n_estimators")
      final_model["max_depth"] = params_dict.get("rf_max_depth")
      final_model["random_state"] = 42
    
    elif model_type == "xgb":
      final_model["objective"] = "binary:logistic"
      final_model["tree_method"] = "hist"
      final_model["device"] = "gpu"
      final_model["learning_rate"] = params_dict.get("xgb_learning_rate")
      final_model["max_depth"] = params_dict.get("xgb_max_depth")
      final_model["reg_alpha"] = params_dict.get("xgb_reg_alpha")
      final_model["random_state"] = 42
      if "n_estimators" not in final_model:
        final_model["n_estimators"] = 200

    elif model_type == "lgbm":
      final_model["objective"] = "binary"
      final_model["boosting_type"] = "gbdt"
      final_model["device"] = "gpu"
      final_model["learning_rate"] = params_dict.get("lgbm_learning_rate")
      final_model["num_leaves"] = params_dict.get("lgbm_num_leaves")
      final_model["reg_alpha"] = params_dict.get("lgbm_reg_alpha")
      final_model["random_state"] = 42
      if "n_estimators" not in final_model:
        final_model["n_estimators"] = 200
    else:
      print(f"Error: Unknown model type '{model_type}'.")
      return None, None
    
    return model_type, final_model


final_model_men_type, final_model_men = extract_model_params(best_arm_and_metrics_men, best_parameters_men)
final_model_women_type, final_model_women = extract_model_params(best_arm_and_metrics_women, best_parameters_women)


print("Men's Model:", final_model_men_type)
print("Men's Parameters:", final_model_men)


print("\nWomen's Model:", final_model_women_type)
print("Women's Parameters:", final_model_women)


print("\n--- Phase 4.3: Train Final Models on Entire Dataset ---\n")


if final_model_men_type == "xgb":
  final_model_men = xgb.XGBClassifier(**final_model_men)
  final_model_men.fit(X_full_dataset_train_men_cudf, y_full_dataset_train_men_cudf.values) # Train on ENTIRE men's training data (GPU)
  print("\n--- Final Men's Model (XGBoost) Training Completed ---\n")
elif final_model_men_type == "lgbm":
  final_model_men = lgb.LGBMClassifier(**final_model_men)
  final_model_men.fit(X_full_dataset_train_men.to_numpy(), y_full_dataset_train_men.values) # Train on ENTIRE men's training data (GPU)
  print("\n--- Final Men's Model (LightGBM) Training Completed ---\n")
else:
  final_model_men = cuMLLogisticRegression(**final_model_men)
  final_model_men.fit(X_full_dataset_train_men_cudf, y_full_dataset_train_men_cudf.values) # Train on ENTIRE men's training data (GPU)
  print("\n--- Final Men's Model (Logistic Regression) Training Completed ---\n")


if final_model_women_type == "xgb":
  final_model_women = xgb.XGBClassifier(**final_model_women)
  final_model_women.fit(X_full_dataset_train_women_cudf, y_full_dataset_train_women_cudf.values) # Train on ENTIRE women's training data (GPU)
  print("\n--- Final Women's Model (XGBoost) Training Completed ---\n")
elif final_model_women_type == "lgbm":
  final_model_women = lgb.LGBMClassifier(**final_model_women)
  final_model_women.fit(X_full_dataset_train_women.to_numpy(), y_full_dataset_train_women.values) # Train on ENTIRE women's training data (GPU)
  print("\n--- Final Women's Model (LightGBM) Training Completed ---\n")
else:
  final_model_women = cuMLLogisticRegression(**final_model_women)
  final_model_women.fit(X_full_dataset_train_women_cudf, y_full_dataset_train_women_cudf.values) # Train on ENTIRE women's training data (GPU)
  print("\n--- Final Women's Model (Logistic Regression) Training Completed ---\n")


print("\n--- Phase 4.3: Final Model Training Completed (Men's and Women's) ---\n")


# --- 4.4: Generate Predictions for 2025 Matchups (Men's) and (Women's) ---
print("\n--- 4.4: Generate Predictions for 2025 Matchups (Men's) and (Women's) ---\n")


# --- 4.4.1: Generate Predictions Dataset for 2025 Matchups (Men's) ---
print("\n--- 4.4.1: Generate Predictions Dataset for 2025 Matchups (Men's) ---\n")


# Assuming 'men_matchups' and 'women_matchups' are your matchup lists
# and 'm_matchup_features_full_dataset_train_df' and 'w_matchup_features_full_dataset_train_df' are your training DataFrames

# Count matchups for men
men_matchups_2025_count = len([matchup for matchup in men_matchups if matchup[0] in m_matchup_features_full_dataset_train_df[m_matchup_features_full_dataset_train_df['Season'] == 2025]['Team1ID'].values and matchup[1] in m_matchup_features_full_dataset_train_df[m_matchup_features_full_dataset_train_df['Season'] == 2025]['Team2ID'].values])
print(f"Number of men's matchups for season 2025: {men_matchups_2025_count}")

# Count matchups for women
women_matchups_2025_count = len([matchup for matchup in women_matchups if matchup[0] in w_matchup_features_full_dataset_train_df[w_matchup_features_full_dataset_train_df['Season'] == 2025]['Team1ID'].values and matchup[1] in w_matchup_features_full_dataset_train_df[w_matchup_features_full_dataset_train_df['Season'] == 2025]['Team2ID'].values])
print(f"Number of women's matchups for season 2025: {women_matchups_2025_count}")


# Prepare 2025 Men's Matchup Features (Using m_matchup_features_train_df - PREPROCESSED DATA) - CORRECTED DATA INPUTS
def prepare_prediction_features_men(men_matchups, matchup_features_train_df, val_df): # Added val_df as input
    prediction_features = []
    for matchup in men_matchups:
        season = 2025 # Prediction season is 2025
        team1_id, team2_id = matchup

        team1_features_df = matchup_features_train_df[((matchup_features_train_df['Season'] == season) & (matchup_features_train_df['Team1ID'] == team1_id))]
        team2_features_df = matchup_features_train_df[((matchup_features_train_df['Season'] == season) & (matchup_features_train_df['Team2ID'] == team2_id))]

        if team1_features_df.empty or team2_features_df.empty:
            print(f"Warning: Skipping matchup ({team1_id}, {team2_id}) due to missing data for season {season}.")
            continue

        team1_features = team1_features_df.iloc[0].to_dict()
        team2_features = team2_features_df.iloc[0].to_dict()

        game_features = {'Season': season, 'Team1ID': team1_id, 'Team2ID': team2_id}
        
        # --- Add features, handling duplicates and missing features ---
        for feature_name in val_df.columns:  # Iterate through validation dataset columns
            if feature_name not in ['Season', 'Team1ID', 'Team2ID', 'Team1_Win']:  # Skip these columns
                if feature_name.startswith('Team1_'):
                    # Get feature value from team1_features, or set to 0 if missing
                    game_features[feature_name] = team1_features.get(feature_name[6:], 0) # Remove 'Team1_' prefix
                elif feature_name.startswith('Team2_'):
                    # Get feature value from team2_features, or set to 0 if missing
                    game_features[feature_name] = team2_features.get(feature_name[6:], 0) # Remove 'Team2_' prefix
                # --- CORRECTED: Add SeedDiff features as well - They will be present in val_df.columns and handled in this loop now ---
                elif feature_name.startswith('SeedD'): # Handle SeedDiff features as well in the general loop
                    game_features[feature_name] = team1_features.get(feature_name, 0) # Or team2_features.get(feature_name, 0) - Diff features should be same for both teams in a matchup
                # --- CORRECTED: Add RankDiff and RankDiff_SAG features as well - They will be present in val_df.columns and handled in this loop now ---
                elif feature_name.startswith('Rank'): # Handle RankDiff and RankDiff_SAG features as well in the general loop
                    game_features[feature_name] = team1_features.get(feature_name, 0) # Or team2_features.get(feature_name, 0) - Diff features should be same for both teams in a matchup


        prediction_features.append(game_features)
    
    prediction_features_df = pd.DataFrame(prediction_features)
    # --- Remove duplicate columns if they exist ---
    prediction_features_df = prediction_features_df.loc[:, ~prediction_features_df.columns.duplicated()]
    return prediction_features_df


# --- 4.4.2: Generate Predictions Dataset for 2025 Matchups (Women's) ---
print("\n--- 4.4.2: Generate Predictions Dataset for 2025 Matchups (Women's) ---\n")


def prepare_prediction_features_women(women_matchups, matchup_features_train_df, val_df): # val_df is not actually used in this function in the current logic
    prediction_features = []
    for matchup in women_matchups:
        season = 2025
        team1_id, team2_id = matchup

        # --- CORRECTED: Remove column selection - Keep ALL columns from matchup_features_train_df ---
        team1_features_df = matchup_features_train_df[((matchup_features_train_df['Season'] == season) & (matchup_features_train_df['Team1ID'] == team1_id))]
        team2_features_df = matchup_features_train_df[((matchup_features_train_df['Season'] == season) & (matchup_features_train_df['Team2ID'] == team2_id))]

        if team1_features_df.empty or team2_features_df.empty:
            print(f"Warning: Skipping matchup ({team1_id}, {team2_id}) due to missing data for season {season}.")
            continue

        team1_features = team1_features_df.iloc[0].to_dict()
        team2_features = team2_features_df.iloc[0].to_dict()

        game_features = {'Season': season, 'Team1ID': team1_id, 'Team2ID': team2_id}

        # --- CORRECTED: Remove manual loop for Diff features - General loop will handle all features including Diff_Rank_POM, Diff_Seed, etc. ---
        # No manual loop for Diff_Rank_POM, SeedDiff, RankDiff_SAG needed anymore

        for feature_name in val_df.columns:  # Iterate through validation dataset columns - Assuming val_df columns represent ALL desired features
            if feature_name not in ['Season', 'Team1ID', 'Team2ID', 'Team1_Win']:  # Skip these columns
                if feature_name.startswith('Team1_'):
                    # Get feature value from team1_features, or set to 0 if missing
                    game_features[feature_name] = team1_features.get(feature_name[6:], 0)
                elif feature_name.startswith('Team2_'):
                    # Get feature value from team2_features, or set to 0 if missing
                    game_features[feature_name] = team2_features.get(feature_name[6:], 0)
                # --- CORRECTED: Add SeedDiff features as well - They will be present in val_df.columns and handled in this loop now ---
                elif feature_name.startswith('SeedD'): # Handle SeedDiff features as well in the general loop
                    game_features[feature_name] = team1_features.get(feature_name, 0) # Or team2_features.get(feature_name, 0) - Diff features should be same for both teams in a matchup
                # --- CORRECTED: Add RankDiff and RankDiff_SAG features as well - They will be present in val_df.columns and handled in this loop now ---
                elif feature_name.startswith('Rank'): # Handle RankDiff and RankDiff_SAG features as well in the general loop
                    game_features[feature_name] = team1_features.get(feature_name, 0) # Or team2_features.get(feature_name, 0) - Diff features should be same for both teams in a matchup


        prediction_features.append(game_features)

    prediction_features_df = pd.DataFrame(prediction_features)
    # --- Remove duplicate columns if they exist ---
    prediction_features_df = prediction_features_df.loc[:, ~prediction_features_df.columns.duplicated()]
    return prediction_features_df


# --- 4.4.3: Generate Predictions for 2025 Matchups (Men's) Based on Final Model Type ---
print("\n--- 4.4.2: Generate Predictions for 2025 Matchups (Men's) Based on Final Model Type ---\n")


X_pred_men_2025_pandas = prepare_prediction_features_men(men_matchups, m_matchup_features_full_dataset_train_df, X_val_men_time_cudf) # Create prediction features (Pandas)
X_pred_men_2025_cudf = cudf.DataFrame.from_pandas(X_pred_men_2025_pandas) #Convert to cuDF 


X_pred_women_2025_pandas = prepare_prediction_features_women(women_matchups, w_matchup_features_full_dataset_train_df,X_val_women_time_cudf) # Create prediction features (Pandas)
X_pred_women_2025_cudf = cudf.DataFrame.from_pandas(X_pred_women_2025_pandas) # Convert to cuDF


# Iterate through each X and confirm the columns 'RankDiff_POM', 'SeedDiff', ' RankDiff_SAG' exist
for X in [
    X_pred_women_2025_pandas, X_pred_women_2025_cudf, X_pred_men_2025_pandas, X_pred_men_2025_cudf
]:
  if 'RankDiff_POM' not in X.columns or 'SeedDiff' not in X.columns or 'RankDiff_SAG' not in X.columns:
    print("Columns 'RankDiff_POM', 'SeedDiff', and 'RankDiff_SAG' do not exist in the DataFrame.")
  else:
    print("Columns 'RankDiff_POM', 'SeedDiff', and 'RankDiff_SAG' exist in the DataFrame.")  



# Make sure the dataframes have the same columns and same order as in the training 
if final_model_men_type == "xgb":
    training_features_men = final_model_men.get_booster().feature_names
    # Ensure 'SeedDiff', 'RankDiff_SAG', and 'RankDiff_POM' are in the prediction data
    for feature in ['SeedDiff', 'RankDiff_SAG', 'RankDiff_POM']:
        if feature not in X_pred_men_2025_cudf.columns:
            X_pred_men_2025_cudf[feature] = 0  # Or any appropriate default value    
elif final_model_men_type == "lgbm":
    # Get the feature names from the training data (LightGBM)
    training_features_men = final_model_men.feature_name_
    # Ensure critical features are present in prediction data (if needed)
    for feature in ['SeedDiff', 'RankDiff_SAG', 'RankDiff_POM']:  # Adjust if needed
        if feature not in X_pred_men_2025_cudf.columns:
            X_pred_men_2025_cudf[feature] = 0  # Or appropriate default value

    # --- Ensure the prediction data has the same features as the training data ---
    # 1. Get the common features
    common_features = list(set(training_features_men).intersection(X_pred_men_2025_cudf.columns))
    # 2. Add missing features with default value (0 in this case)
    missing_features = list(set(training_features_men) - set(common_features))
    for feature in missing_features:
        X_pred_men_2025_cudf[feature] = 0  # Or any appropriate default value

# --- Reorder the columns to match training data ---
X_pred_men_2025_cudf = X_pred_men_2025_cudf[training_features_men]


# Make sure the dataframes have the same columns and same order as in the training data
# Get the feature names from the training data
if final_model_women_type == "xgb":
    training_features = final_model_women.get_booster().feature_names
    # Ensure 'SeedDiff', 'RankDiff_SAG', and 'RankDiff_POM' are in the prediction data
    for feature in ['SeedDiff', 'RankDiff_SAG', 'RankDiff_POM']:
        if feature not in X_pred_women_2025_cudf.columns:
            X_pred_women_2025_cudf[feature] = 0  # Or any appropriate default value

elif final_model_women_type == "lgbm":
    # Get the feature names from the training data (LightGBM)
    training_features_women = final_model_women.feature_name_
    # Ensure critical features are present in prediction data (if needed)
    for feature in ['SeedDiff', 'RankDiff_SAG', 'RankDiff_POM']:  # Adjust if needed
        if feature not in X_pred_women_2025_cudf.columns:
            X_pred_women_2025_cudf[feature] = 0  # Or appropriate default value

    # --- Ensure the prediction data has the same features as the training data ---
    # 1. Get the common features
    common_features = list(set(training_features_women).intersection(X_pred_women_2025_cudf.columns))
    # 2. Add missing features with default value (0 in this case)
    missing_features = list(set(training_features_women) - set(common_features))
    for feature in missing_features:
        X_pred_women_2025_cudf[feature] = 0  # Or any appropriate default value

# Reorder the prediction data columns to match the training data
X_pred_women_2025_cudf = X_pred_women_2025_cudf[training_features]


# Compare features to ensure the same datasets are being used
print("\n--- Comparing features to ensure the same datasets are being used ---\n")

def compare_features(pred_df, val_df, dataset_name):
    """Compares features between prediction and validation datasets and lists missing features."""
    pred_features = set(pred_df.columns)
    val_features = set(val_df.columns)

    print(f"\n--- Feature Comparison for {dataset_name} ---")
    print(f"Number of features in prediction dataset: {len(pred_features)}")
    print(f"Number of features in validation dataset: {len(val_features)}")

    missing_features = list(val_features - pred_features)  # Features in validation but not in prediction

    if missing_features:
        print(f"\nMissing features in prediction dataset ({len(missing_features)} total):")
        for feature in missing_features[-10:]:  # Get last 10 missing features
            print(f"- {feature}")
    else:
        print("\nNo missing features found in prediction dataset.")

# --- Compare features for men's datasets ---
compare_features(X_pred_men_2025_pandas, X_val_men_time_cudf, "Men's")

# --- Compare features for women's datasets ---
compare_features(X_pred_women_2025_pandas, X_val_women_time_cudf, "Women's")


# Determine the final model type for men's and run the prediction
if final_model_men_type == "xgb":
  print("\n--- Generating Men's Tournament Predictions (XGBoost) ---\n")
  men_preds_gpu = final_model_men.predict_proba(X_pred_men_2025_cudf)[:, 1] # Predict probabilities for men's matchups (GPU)
  men_preds = cp.asnumpy(men_preds_gpu)
  print("\n--- Men's Tournament Predictions Generated (XGBoost) ---\n")
elif final_model_men_type == "lgbm":
  print("\n--- Generating Men's Tournament Predictions (LightGBM) ---\n")
  men_preds_gpu = final_model_men.predict_proba(X_pred_men_2025_cudf.to_numpy())[:, 1] # Predict probabilities for men's matchups (GPU)
  men_preds = cp.asnumpy(men_preds_gpu)
  print("\n--- Men's Tournament Predictions Generated (LightGBM) ---\n")
else:
  print("\n--- Generating Men's Tournament Predictions (Logistic Regression) ---\n")
  men_preds_gpu = final_model_men.predict_proba(X_pred_men_2025_cudf)[:, 1] # Predict probabilities for men's matchups (GPU)
  men_preds = cp.asnumpy(men_preds_gpu)
  print("\n--- Men's Tournament Predictions Generated (Logistic Regression) ---\n")



# Determine the final model type for women's and run the prediction
if final_model_women_type == "xgb":
  print("\n--- Generating Women's Tournament Predictions (XGBoost) ---\n")
  women_preds_gpu = final_model_women.predict_proba(X_pred_women_2025_cudf)[:, 1] # Predict probabilities for women's matchups (GPU)
  women_preds = cp.asnumpy(women_preds_gpu)
  print("\n--- Women's Tournament Predictions Generated (XGBoost) ---\n")
elif final_model_women_type == "lgbm":
  print("\n--- Generating Women's Tournament Predictions (LGBM) ---\n")
  women_preds_gpu = final_model_women.predict_proba(X_pred_women_2025_cudf.to_numpy())[:, 1] # Predict probabilities for women's matchups (GPU)
  women_preds = cp.asnumpy(women_preds_gpu)
  print("\n--- Women's Tournament Predictions Generated (LGBM) ---\n")
else:
  print("\n--- Generating Women's Tournament Predictions (Logistic Regression) ---\n")
  women_preds_gpu = final_model_women.predict_proba(X_pred_women_2025_cudf)[:, 1] # Predict probabilities for women's matchups (GPU)
  women_preds = cp.asnumpy(women_preds_gpu)
  print("\n--- Women's Tournament Predictions Generated (Logistic Regression) ---\n")


# --- Added Elo Functions for missing data
def calculate_elo(
    teams, data, initial_rating=2000, k=140, alpha=None, weights=False, lowerlim=float("-inf")
    ):
    '''
    Calculate Elo ratings for each team based on match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000). 
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - alpha (float or None): Tuning parameter for the multiplier for the margin of victory. No multiplier if None.

    Returns: 
    - list: Historical ratings of the winning team (WTeam).
    - list: Historical ratings of the losing team (LTeam).
    - list: Brier score for each match (due to symmetry only for 1 team)
    '''
    
    # Dictionary to keep track of current ratings for each team
    team_dict = {}
    for team in teams:
        team_dict[team] = initial_rating
        
    # Lists to store ratings for each team in each game
    r1, r2 = [], []
    loss = []
    margin_of_victory = 1
    weight = 1

    # Iterate through the game data
    for wteam, lteam, ws, ls, w  in tqdm(zip(data.WTeamID, data.LTeamID, data.WScore, data.LScore, data.weight), total=len(data)):

        # Calculate expected outcomes based on Elo ratings
        rateW = 1 / (1 + 10 ** ((team_dict[lteam] - team_dict[wteam]) / initial_rating))
        rateL = 1 / (1 + 10 ** ((team_dict[wteam] - team_dict[lteam]) / initial_rating))
        
        if alpha:
                margin_of_victory = (ws - ls)/alpha
        if isinstance(weights, (list, np.ndarray, pd.Series)):
            weight = w

        # Update ratings for winning and losing teams
        team_dict[wteam] += w * k * margin_of_victory * (1 - rateW)
        team_dict[lteam] += w * k * margin_of_victory * (0 - rateL)

        # Ensure that ratings do not go below lower limit
        if team_dict[lteam] < lowerlim:
            team_dict[lteam] = lowerlim
            
        # Append current ratings for teams to lists
        r1.append(team_dict[wteam])
        r2.append(team_dict[lteam])
        loss.append((1-rateW)**2)
        
    return r1, r2, loss

def create_elo_data(teams, data, initial_rating=2000, k=140, alpha=None, weights=None, lowerlim=float("-inf")):
    '''
    Create a DataFrame with summary statistics of Elo ratings for teams based on historical match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000).
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - weights (array-like): Containing weights for each match.

    Returns: 
    - DataFrame: Summary statistics of Elo ratings for teams throughout a season.
    '''
    
    if isinstance(weights, (list, np.ndarray, pd.Series)):
        data['weight'] = weights
    else:
        data['weight'] = 1
    
    r1, r2, loss = calculate_elo(teams, data, initial_rating, k, alpha, weights, lowerlim)
    # Calculate loss only on tourney results
    loss = np.mean(np.array(loss)[data.tourney == 1])
    print(f"=== Brier Score: {loss:.5f} (Only  Tournaments) ===")
    
    # Concatenate arrays vertically
    seasons = np.concatenate([data.Season, data.Season])
    days = np.concatenate([data.DayNum, data.DayNum])
    teams = np.concatenate([data.WTeamID, data.LTeamID])
    tourney = np.concatenate([data.tourney, data.tourney])
    ratings = np.concatenate([r1, r2])
    # Create a DataFrame
    rating_df = pd.DataFrame({
        'Season': seasons,
        'DayNum': days,
        'TeamID': teams,
        'Rating': ratings,
        'Tourney': tourney
    })

    # Sort DataFrame and remove tournament data
    rating_df.sort_values(['TeamID', 'Season', 'DayNum'], inplace=True)
    rating_df = rating_df[rating_df['Tourney'] == 0]
    grouped = rating_df.groupby(['TeamID', 'Season'])
    results = grouped['Rating'].agg(['mean', 'median', 'std', 'min', 'max', 'last'])
    results.columns = ['Rating_Mean', 'Rating_Median', 'Rating_Std', 'Rating_Min', 'Rating_Max', 'Rating_Last']
    results['Rating_Trend'] = grouped.apply(lambda x: linregress(range(len(x)), x['Rating']).slope, include_groups=False)
    results.reset_index(inplace=True)
    
    return results, loss


m_regular_season_compact_results_df['tourney'] = 0
m_ncaa_tourney_compact_results_df['tourney'] = 1
m_regular_season_compact_results_df['weight'] = 1
m_ncaa_tourney_compact_results_df['weight'] = 0.7

data_m = pd.concat([m_regular_season_compact_results_df, m_ncaa_tourney_compact_results_df])
data_m.sort_values(['Season', 'DayNum'], inplace=True)
data_m.reset_index(inplace=True, drop=True)

elo_df_men, _ = create_elo_data(m_teams_df.TeamID, data_m, initial_rating=1200, k=125, alpha=None, weights=data_m['weight'])
elo_df_men.tail(10)


w_regular_season_compact_results_df['tourney'] = 0
w_ncaa_tourney_compact_results_df['tourney'] = 1
w_regular_season_compact_results_df['weight'] = 0.95
w_ncaa_tourney_compact_results_df['weight'] = 1

data_w = pd.concat([w_regular_season_compact_results_df, w_ncaa_tourney_compact_results_df])
data_w.sort_values(['Season', 'DayNum'], inplace=True)
data_w.reset_index(inplace=True, drop=True)

elo_df_women, _ = create_elo_data(w_teams_df.TeamID, data_w, initial_rating=1200, k=190, alpha=None, weights=data_w['weight'])
elo_df_women.tail(10)


# Split the ID into Season, T1_TeamID, and T2_TeamID
sub = sample_submission_stage_2_df.ID.str.split('_', expand=True).astype(int)
sub.columns = ["Season", "T1_TeamID", "T2_TeamID"]

# Turn elo dfs into dict
elo_dict = pd.concat([
    elo_df_women[elo_df_women.Season == 2025],
    elo_df_men[elo_df_men.Season == 2025]
]).set_index("TeamID")["Rating_Last"].to_dict()

# Calculate probabilities
sample_submission_stage_2_df.Pred = 1 / (1 + 10**((sub.T2_TeamID.map(elo_dict) - sub.T1_TeamID.map(elo_dict))/1200))


# --- Phase 4.4.4: Create Submission File ---
print("\n--- Phase 4.4.4: Create Submission File ---\n")

submission_rows = [] # List to store submission rows

# Men's Submissions
for i, matchup in enumerate(men_matchups):
    if i < len(men_preds):  # Check if index is within bounds of men_preds
        team1_id, team2_id = matchup
        pred_prob = men_preds[i] # Get prediction probability from men_preds array
        submission_rows.append({'ID': f'2025_{team1_id}_{team2_id}', 'Pred': pred_prob}) # Append men's submission row
    else:
        break  # Exit loop if index is out of bounds

# Women's Submissions
for i, matchup in enumerate(women_matchups):
    if i < len(women_preds):  # Check if index is within bounds of women_preds
        team1_id, team2_id = matchup
        pred_prob = women_preds[i] # Get prediction probability from women_preds array
        submission_rows.append({'ID': f'2025_{team1_id}_{team2_id}', 'Pred': pred_prob}) # Append women's submission row
    else:
        break  # Exit loop if index is out of bounds

submission_df = pd.DataFrame(submission_rows) # Create submission DataFrame

# append to the datasource whatever is not in there
for index, row in sample_submission_stage_2_df.iterrows():
    if row.ID not in submission_df['ID'].values:
        submission_df = pd.concat([submission_df, pd.DataFrame([row])], ignore_index=True)


submission_file_path = './submission.csv' # Define submission file path
submission_df.to_csv(submission_file_path, index=False) # Save to CSV
print(f"\n--- Submission file created successfully at: {submission_file_path} ---\n") # Print confirmation message

print("\n--- Phase 4.4: Prediction Generation and Submission Preparation Completed ---\n")

