# Standard libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pickle
import joblib
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve, IsotonicRegression
import lightgbm as lgb
import xgboost as xgb

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Clean up files from previous runs
import shutil

# Define output directories
output_dirs = [
    '/kaggle/working/processed/elo',
    '/kaggle/working/processed/features',
    '/kaggle/working/models',
    '/kaggle/working/outputs',
    '/kaggle/working/submissions'
]

# Clean and recreate directories
for directory in output_dirs:
    if os.path.exists(directory):
        print(f"Cleaning {directory}...")
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

print("Environment ready for a fresh run!")


# Define DataLoader class for Kaggle
class DataLoader:
    def __init__(self):
        # Use Kaggle's input path
        self.data_dir = '/kaggle/input/march-machine-learning-mania-2025'
        print(f"Data directory set to: {self.data_dir}")
        
    def load_regular_season_results(self, gender):
        path = os.path.join(self.data_dir, f'{gender}RegularSeasonCompactResults.csv')
        if os.path.exists(path):
            print(f"Found regular season results at: {path}")
            df = pd.read_csv(path)
            df['Source'] = 'RegularSeason'
            return df
        else:
            print(f"Warning: Could not find regular season results for {gender}.")
            return pd.DataFrame()
        
    def load_tournament_results(self, gender):
        # Use the correct file name based on the available files
        if gender == 'M':
            path = os.path.join(self.data_dir, 'MNCAATourneyCompactResults.csv')
        else:  # gender == 'W'
            path = os.path.join(self.data_dir, 'WNCAATourneyCompactResults.csv')
            
        if os.path.exists(path):
            print(f"Found tournament results at: {path}")
            df = pd.read_csv(path)
            df['Source'] = 'Tournament'
            return df
        else:
            print(f"Warning: Could not find tournament results for {gender}.")
            return pd.DataFrame()
        
    def load_teams(self, gender):
        path = os.path.join(self.data_dir, f'{gender}Teams.csv')
        if os.path.exists(path):
            print(f"Found teams data at: {path}")
            return pd.read_csv(path)
        else:
            print(f"Warning: Could not find teams data for {gender}.")
            return pd.DataFrame()
        
    def load_seeds(self, gender):
        # Use the correct file name based on the available files
        if gender == 'M':
            path = os.path.join(self.data_dir, 'MNCAATourneySeeds.csv')
        else:  # gender == 'W'
            path = os.path.join(self.data_dir, 'WNCAATourneySeeds.csv')
            
        if os.path.exists(path):
            print(f"Found seeds data at: {path}")
            return pd.read_csv(path)
        else:
            print(f"Warning: Could not find seeds data for {gender}.")
            return pd.DataFrame()
    
    def load_team_conferences(self, gender):
        # Try to load team conferences
        if gender == 'M':
            path = os.path.join(self.data_dir, 'MTeamConferences.csv')
        else:  # gender == 'W'
            path = os.path.join(self.data_dir, 'WTeamConferences.csv')
            
        if os.path.exists(path):
            print(f"Found team conferences at: {path}")
            return pd.read_csv(path)
        else:
            print(f"Warning: Could not find team conferences for {gender}.")
            return pd.DataFrame()
    
    def load_sample_submission(self):
        # Load sample submission to get the required format
        path = os.path.join(self.data_dir, 'SampleSubmissionStage1.csv')
        if os.path.exists(path):
            print(f"Found sample submission at: {path}")
            return pd.read_csv(path)
        else:
            print(f"Warning: Could not find sample submission.")
            return pd.DataFrame()
    
    # Add method to list available files in the data directory
    def list_available_files(self):
        print("Available files in Kaggle dataset:")
        for file in sorted(os.listdir(self.data_dir)):
            print(f"- {file}")

# Define EnhancedEloRater class
class EnhancedEloRater:
    def __init__(self, base_rating=1500, k_factor=20, recency_weight=0.05, 
                 home_advantage=100, tournament_weight=1.5):
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.recency_weight = recency_weight
        self.home_advantage = home_advantage
        self.tournament_weight = tournament_weight
        self.ratings = {}
        
    def reset_ratings(self):
        self.ratings = {}
        
    def get_rating(self, team_id):
        return self.ratings.get(team_id, self.base_rating)
        
    def update_ratings(self, winner_id, loser_id, winner_score, loser_score, 
                      is_tournament=0, date=None):
        # Get current ratings
        winner_rating = self.get_rating(winner_id)
        loser_rating = self.get_rating(loser_id)
        
        # Calculate expected outcome
        expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        
        # Calculate margin of victory multiplier
        score_diff = winner_score - loser_score
        mov_multiplier = np.log(max(score_diff, 1) + 1) * (2.2 / ((winner_rating - loser_rating) * 0.001 + 2.2))
        
        # Apply tournament weight
        k = self.k_factor
        if is_tournament:
            k *= self.tournament_weight
            
        # Update ratings
        rating_change = k * mov_multiplier * (1 - expected_winner)
        self.ratings[winner_id] = winner_rating + rating_change
        self.ratings[loser_id] = loser_rating - rating_change
        
    def get_ratings(self):
        return self.ratings.copy()

# Define ModelEvaluator class
class ModelEvaluator:
    def __init__(self, output_dir='/kaggle/working/outputs'):
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def brier_score(self, y_true, y_pred):
        return brier_score_loss(y_true, y_pred)
        
    def log_loss(self, y_true, y_pred):
        return log_loss(y_true, y_pred)
        
    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred > 0.5)
        
    def roc_auc(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)
    
    def evaluate_model(self, model, X, y, model_name, gender):
        # Generate predictions
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'brier_score': self.brier_score(y, y_pred),
            'log_loss': self.log_loss(y, y_pred),
            'accuracy': self.accuracy(y, y_pred),
            'roc_auc': self.roc_auc(y, y_pred)
        }
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.output_dir / f"{gender}_{model_name}_metrics.csv", index=False)
        
        # Create calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y, y_pred, n_bins=10)
        
        # Plot calibration curve
        plt.figure(figsize=(10, 8))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
        plt.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibration curve - {gender} - {model_name}")
        plt.legend()
        plt.savefig(self.output_dir / f"{gender}_{model_name}_calibration.png")
        
        return metrics, y_pred


# Set up directories for Kaggle
output_dir = '/kaggle/working'
os.makedirs(os.path.join(output_dir, 'processed/elo'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'processed/features'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'outputs'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'submissions'), exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"Output directory set to: {output_dir}")

# Initialize data loader
data_loader = DataLoader()

# List available files
data_loader.list_available_files()

# Load regular season and tournament results for both men's and women's tournaments
def load_data(gender):
    print(f"Loading {gender} tournament data...")
    regular_season = data_loader.load_regular_season_results(gender)
    tournament = data_loader.load_tournament_results(gender)
    teams = data_loader.load_teams(gender)
    seeds = data_loader.load_seeds(gender)
    conferences = data_loader.load_team_conferences(gender)
    return regular_season, tournament, teams, seeds, conferences

# Load men's data
M_regular_season, M_tournament, M_teams, M_seeds, M_conferences = load_data('M')

# Load women's data
W_regular_season, W_tournament, W_teams, W_seeds, W_conferences = load_data('W')

# Load sample submission
sample_submission = data_loader.load_sample_submission()

# Display data samples
print("\nMen's Regular Season Sample:")
display(M_regular_season.head())

print("\nMen's Tournament Sample:")
display(M_tournament.head())

print("\nMen's Teams Sample:")
display(M_teams.head())

print("\nMen's Seeds Sample:")
display(M_seeds.head())

print("\nSample Submission:")
display(sample_submission.head())


# Define a function to calculate enhanced Elo ratings
def calculate_enhanced_elo(gender, regular_season, tournament, teams):
    print(f"Calculating enhanced Elo ratings for {gender} tournament...")
    
    # Initialize enhanced Elo rater
    elo_rater = EnhancedEloRater(
        base_rating=1500,
        k_factor=20,
        recency_weight=0.05,
        home_advantage=100,
        tournament_weight=1.5
    )
    
    # Combine regular season and tournament results
    all_games = pd.concat([regular_season, tournament], ignore_index=True)
    
    # Check for date column and create DateObject for sorting
    if 'Date' in all_games.columns:
        all_games['DateObject'] = pd.to_datetime(all_games['Date'])
    else:
        # If Date column doesn't exist, use DayNum and Season as a proxy
        all_games['DateObject'] = all_games.apply(
            lambda row: pd.Timestamp(year=int(row['Season']), month=1, day=1) + 
                        pd.Timedelta(days=int(row['DayNum'])), 
            axis=1
        )
    
    # Sort games by date
    all_games = all_games.sort_values(['Season', 'DateObject'])
    
    # Calculate Elo ratings for each team and each season
    seasons = all_games['Season'].unique()
    team_ids = teams['TeamID'].unique()
    
    # Initialize ratings dictionary
    ratings = {}
    
    # Process each season
    for season in seasons:
        print(f"Processing season {season}...")
        season_games = all_games[all_games['Season'] == season]
        
        # Reset ratings at the start of each season
        elo_rater.reset_ratings()
        
        # Process each game
        for _, game in season_games.iterrows():
            # Extract game information
            team_a = game['WTeamID']
            team_b = game['LTeamID']
            score_a = game['WScore']
            score_b = game['LScore']
            is_tournament = 1 if game['Source'] == 'Tournament' else 0
            date = game['DateObject']
            
            # Update ratings
            elo_rater.update_ratings(team_a, team_b, score_a, score_b, is_tournament, date)
        
        # Store end-of-season ratings
        ratings[season] = elo_rater.get_ratings()
        
        # Save season ratings to CSV
        ratings_df = pd.DataFrame({
            'TeamID': list(ratings[season].keys()),
            'Elo': list(ratings[season].values())
        })
        ratings_df.to_csv(f"/kaggle/working/processed/elo/{gender}EnhancedEloRatings_{season}.csv", index=False)
        
        # Print average Elo rating for the season
        avg_elo = ratings_df['Elo'].mean()
        print(f"Season {season} - Average Elo rating: {avg_elo:.2f}")
        
        # Print top 5 teams by Elo rating
        top_teams = ratings_df.sort_values('Elo', ascending=False).head(5)
        print(f"Top 5 teams by Elo rating for season {season}:")
        for _, team in top_teams.iterrows():
            team_name = teams[teams['TeamID'] == team['TeamID']]['TeamName'].values[0] if len(teams[teams['TeamID'] == team['TeamID']]) > 0 else "Unknown"
            print(f"  {team_name} (ID: {team['TeamID']}): {team['Elo']:.2f}")
    
    # Combine all ratings into a single DataFrame
    all_ratings = []
    for season, season_ratings in ratings.items():
        for team_id, rating in season_ratings.items():
            all_ratings.append({
                'Season': season,
                'TeamID': team_id,
                'Elo': rating
            })
    
    all_ratings_df = pd.DataFrame(all_ratings)
    all_ratings_df.to_csv(f"/kaggle/working/processed/elo/{gender}EnhancedEloRatings.csv", index=False)
    
    # Plot Elo rating distribution for the most recent season
    latest_season = max(seasons)
    latest_ratings = all_ratings_df[all_ratings_df['Season'] == latest_season]
    
    plt.figure(figsize=(12, 6))
    sns.histplot(latest_ratings['Elo'], kde=True)
    plt.title(f"{gender} Tournament - Elo Rating Distribution (Season {latest_season})")
    plt.xlabel("Elo Rating")
    plt.ylabel("Number of Teams")
    plt.savefig(f"/kaggle/working/outputs/{gender}_elo_distribution_{latest_season}.png")
    
    return all_ratings_df

# Calculate enhanced Elo ratings for men's tournament
M_elo_ratings = calculate_enhanced_elo('M', M_regular_season, M_tournament, M_teams)

# Calculate enhanced Elo ratings for women's tournament
W_elo_ratings = calculate_enhanced_elo('W', W_regular_season, W_tournament, W_teams)

# Display sample of Elo ratings
print("\nMen's Enhanced Elo Ratings Sample:")
display(M_elo_ratings.head())

print("\nWomen's Enhanced Elo Ratings Sample:")
display(W_elo_ratings.head())


# Define a function to create enhanced features
def create_enhanced_features(gender, regular_season, tournament, teams, seeds, conferences, elo_ratings):
    print(f"Creating enhanced features for {gender} tournament...")
    
    # Combine regular season and tournament results
    all_games = pd.concat([regular_season, tournament], ignore_index=True)
    
    # Check for date column and create DateObject for sorting
    if 'Date' in all_games.columns:
        all_games['DateObject'] = pd.to_datetime(all_games['Date'])
    else:
        # If Date column doesn't exist, use DayNum and Season as a proxy
        all_games['DateObject'] = all_games.apply(
            lambda row: pd.Timestamp(year=int(row['Season']), month=1, day=1) + 
                        pd.Timedelta(days=int(row['DayNum'])), 
            axis=1
        )
    
    # Sort games by date
    all_games = all_games.sort_values(['Season', 'DateObject'])
    
    # Create team statistics
    team_stats = {}
    
    # Process each season
    seasons = all_games['Season'].unique()
    for season in seasons:
        print(f"Processing season {season} for {gender}...")
        season_games = all_games[all_games['Season'] == season]
        
        # Initialize team stats for the season
        season_team_stats = {}
        
        # Process each team
        for team_id in teams['TeamID'].unique():
            # Games where the team won
            won_games = season_games[season_games['WTeamID'] == team_id]
            # Games where the team lost
            lost_games = season_games[season_games['LTeamID'] == team_id]
            
            # Calculate total games played
            total_games = len(won_games) + len(lost_games)
            
            if total_games > 0:
                # Calculate win percentage
                win_pct = len(won_games) / total_games
                
                # Calculate points scored and allowed
                points_scored = (won_games['WScore'].sum() + lost_games['LScore'].sum()) / total_games
                points_allowed = (won_games['LScore'].sum() + lost_games['WScore'].sum()) / total_games
                
                # Calculate point differential
                point_differential = points_scored - points_allowed
                
                # Get Elo rating
                elo = elo_ratings[(elo_ratings['Season'] == season) & 
                                 (elo_ratings['TeamID'] == team_id)]['Elo'].values
                
                if len(elo) > 0:
                    elo = elo[0]
                else:
                    elo = 1500  # Default rating
                
                # Get seed if available
                if 'Seed' in seeds.columns:
                    seed_col = 'Seed'
                else:
                    # Check for alternative seed column names
                    seed_col = next((col for col in seeds.columns if 'seed' in col.lower()), None)
                
                if seed_col:
                    seed_row = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == team_id)]
                    if len(seed_row) > 0:
                        seed_str = seed_row[seed_col].values[0]
                        # Extract numeric part of seed (e.g., "W01" -> 1, "E12" -> 12)
                        seed = int(''.join(filter(str.isdigit, str(seed_str))))
                    else:
                        seed = None
                else:
                    seed = None
                
                # Calculate team form (last 10 games)
                team_games = []
                for _, game in won_games.iterrows():
                    team_games.append({
                        'Season': game['Season'],
                        'DayNum': game['DayNum'],
                        'TeamID': team_id,
                        'Won': 1
                    })
                
                for _, game in lost_games.iterrows():
                    team_games.append({
                        'Season': game['Season'],
                        'DayNum': game['DayNum'],
                        'TeamID': team_id,
                        'Won': 0
                    })
                
                team_games_df = pd.DataFrame(team_games)
                if not team_games_df.empty:
                    team_games_df = team_games_df.sort_values('DayNum')
                    last_10_games = team_games_df.tail(10)
                    form = last_10_games['Won'].mean() if len(last_10_games) > 0 else 0.5
                else:
                    form = 0.5  # Default form
                
                # Get conference information
                if not conferences.empty:
                    conf_row = conferences[(conferences['Season'] == season) & (conferences['TeamID'] == team_id)]
                    if len(conf_row) > 0:
                        conf = conf_row['ConfAbbrev'].values[0]
                    else:
                        conf = None
                else:
                    conf = None
                
                # Store team stats
                season_team_stats[team_id] = {
                    'WinPct': win_pct,
                    'PointsScored': points_scored,
                    'PointsAllowed': points_allowed,
                    'PointDifferential': point_differential,
                    'Elo': elo,
                    'Seed': seed,
                    'Form': form,
                    'Conference': conf
                }
        
        # Store season stats
        team_stats[season] = season_team_stats
    
    # Create features for all possible matchups
    features = []
    
    # Get the current season (for submission)
    current_season = max(seasons)
    
    # Process historical matchups (for training)
    print(f"Creating features for historical {gender} tournament matchups...")
    for _, game in tournament.iterrows():
        season = game['Season']
        team_a = game['WTeamID']
        team_b = game['LTeamID']
        result = 1  # Team A won
        
        # Skip if either team's stats are not available
        if team_a not in team_stats[season] or team_b not in team_stats[season]:
            continue
        
        # Get team stats
        team_a_stats = team_stats[season][team_a]
        team_b_stats = team_stats[season][team_b]
        
        # Check if teams are from the same conference
        same_conference = 0
        if team_a_stats['Conference'] and team_b_stats['Conference']:
            same_conference = 1 if team_a_stats['Conference'] == team_b_stats['Conference'] else 0
        
        # Create features
        feature = {
            'Season': season,
            'TeamA': team_a,
            'TeamB': team_b,
            'TeamA_WinPct': team_a_stats['WinPct'],
            'TeamB_WinPct': team_b_stats['WinPct'],
            'TeamA_PointsScored': team_a_stats['PointsScored'],
            'TeamB_PointsScored': team_b_stats['PointsScored'],
            'TeamA_PointsAllowed': team_a_stats['PointsAllowed'],
            'TeamB_PointsAllowed': team_b_stats['PointsAllowed'],
            'TeamA_PointDifferential': team_a_stats['PointDifferential'],
            'TeamB_PointDifferential': team_b_stats['PointDifferential'],
            'TeamA_Elo': team_a_stats['Elo'],
            'TeamB_Elo': team_b_stats['Elo'],
            'TeamA_Form': team_a_stats['Form'],
            'TeamB_Form': team_b_stats['Form'],
            'WinPctDiff': team_a_stats['WinPct'] - team_b_stats['WinPct'],
            'PointsScoredDiff': team_a_stats['PointsScored'] - team_b_stats['PointsScored'],
            'PointsAllowedDiff': team_a_stats['PointsAllowed'] - team_b_stats['PointsAllowed'],
            'PointDifferentialDiff': team_a_stats['PointDifferential'] - team_b_stats['PointDifferential'],
            'EloDiff': team_a_stats['Elo'] - team_b_stats['Elo'],
            'FormDiff': team_a_stats['Form'] - team_b_stats['Form'],
            'SameConference': same_conference,
            'Result': result
        }
        
        # Add seed difference if both teams have seeds
        if team_a_stats['Seed'] is not None and team_b_stats['Seed'] is not None:
            feature['TeamA_Seed'] = team_a_stats['Seed']
            feature['TeamB_Seed'] = team_b_stats['Seed']
            feature['SeedDiff'] = team_a_stats['Seed'] - team_b_stats['Seed']
        
        features.append(feature)
        
        # Add the reverse matchup with opposite result
        reverse_feature = feature.copy()
        reverse_feature['TeamA'] = team_b
        reverse_feature['TeamB'] = team_a
        reverse_feature['TeamA_WinPct'] = team_b_stats['WinPct']
        reverse_feature['TeamB_WinPct'] = team_a_stats['WinPct']
        reverse_feature['TeamA_PointsScored'] = team_b_stats['PointsScored']
        reverse_feature['TeamB_PointsScored'] = team_a_stats['PointsScored']
        reverse_feature['TeamA_PointsAllowed'] = team_b_stats['PointsAllowed']
        reverse_feature['TeamB_PointsAllowed'] = team_a_stats['PointsAllowed']
        reverse_feature['TeamA_PointDifferential'] = team_b_stats['PointDifferential']
        reverse_feature['TeamB_PointDifferential'] = team_a_stats['PointDifferential']
        reverse_feature['TeamA_Elo'] = team_b_stats['Elo']
        reverse_feature['TeamB_Elo'] = team_a_stats['Elo']
        reverse_feature['TeamA_Form'] = team_b_stats['Form']
        reverse_feature['TeamB_Form'] = team_a_stats['Form']
        reverse_feature['WinPctDiff'] = -feature['WinPctDiff']
        reverse_feature['PointsScoredDiff'] = -feature['PointsScoredDiff']
        reverse_feature['PointsAllowedDiff'] = -feature['PointsAllowedDiff']
        reverse_feature['PointDifferentialDiff'] = -feature['PointDifferentialDiff']
        reverse_feature['EloDiff'] = -feature['EloDiff']
        reverse_feature['FormDiff'] = -feature['FormDiff']
        reverse_feature['Result'] = 0  # Team A lost
        
        # Add seed information if available
        if 'TeamA_Seed' in feature:
            reverse_feature['TeamA_Seed'] = team_b_stats['Seed']
            reverse_feature['TeamB_Seed'] = team_a_stats['Seed']
            reverse_feature['SeedDiff'] = -feature['SeedDiff']
        
        features.append(reverse_feature)
    
    # Generate features for all possible matchups in the current season
    submission_features = generate_all_possible_matchups(gender, teams, team_stats, current_season)
    
    # Convert to DataFrames
    features_df = pd.DataFrame(features)
    submission_features_df = pd.DataFrame(submission_features)
    
    # Save features
    features_df.to_csv(f"/kaggle/working/processed/features/{gender}_enhanced_features.csv", index=False)
    submission_features_df.to_csv(f"/kaggle/working/processed/features/{gender}_enhanced_submission_features.csv", index=False)
    
    # Print feature statistics
    print(f"\n{gender} Tournament Feature Statistics:")
    print(f"Number of training examples: {len(features_df)}")
    print(f"Number of submission examples: {len(submission_features_df)}")
    
    return features_df, submission_features_df, team_stats

# Function to generate all possible matchups for submission
def generate_all_possible_matchups(gender, teams_df, team_stats, season):
    """Generate all possible matchups between teams for a given season."""
    print(f"Generating all possible matchups for {gender} tournament, season {season}...")
    
    # Get teams that have stats for the current season
    team_ids = [team_id for team_id in teams_df['TeamID'].unique() 
                if team_id in team_stats[season]]
    
    all_matchups = []
    
    # Generate all possible team pairs
    team_count = len(team_ids)
    print(f"Processing {team_count} teams, generating {team_count * (team_count - 1) // 2} matchups...")
    
    for i, team_a in enumerate(team_ids):
        if i % 50 == 0:  # Print progress every 50 teams
            print(f"Processing team {i+1}/{team_count}...")
            
        for team_b in team_ids[i+1:]:  # Ensure team_a < team_b
            # Get team stats
            team_a_stats = team_stats[season][team_a]
            team_b_stats = team_stats[season][team_b]
            
            # Check if teams are from the same conference
            same_conference = 0
            if team_a_stats['Conference'] and team_b_stats['Conference']:
                same_conference = 1 if team_a_stats['Conference'] == team_b_stats['Conference'] else 0
            
            # Create features for TeamA < TeamB (for ID format)
            feature_a_b = {
                'Season': season,
                'TeamA': min(team_a, team_b),
                'TeamB': max(team_a, team_b),
                'ID': f"{season}_{min(team_a, team_b)}_{max(team_a, team_b)}"
            }
            
            # Add team stats based on which team is A and which is B
            if team_a < team_b:
                feature_a_b.update({
                    'TeamA_WinPct': team_a_stats['WinPct'],
                    'TeamB_WinPct': team_b_stats['WinPct'],
                    'TeamA_PointsScored': team_a_stats['PointsScored'],
                    'TeamB_PointsScored': team_b_stats['PointsScored'],
                    'TeamA_PointsAllowed': team_a_stats['PointsAllowed'],
                    'TeamB_PointsAllowed': team_b_stats['PointsAllowed'],
                    'TeamA_PointDifferential': team_a_stats['PointDifferential'],
                    'TeamB_PointDifferential': team_b_stats['PointDifferential'],
                    'TeamA_Elo': team_a_stats['Elo'],
                    'TeamB_Elo': team_b_stats['Elo'],
                    'TeamA_Form': team_a_stats['Form'],
                    'TeamB_Form': team_b_stats['Form'],
                    'WinPctDiff': team_a_stats['WinPct'] - team_b_stats['WinPct'],
                    'PointsScoredDiff': team_a_stats['PointsScored'] - team_b_stats['PointsScored'],
                    'PointsAllowedDiff': team_a_stats['PointsAllowed'] - team_b_stats['PointsAllowed'],
                    'PointDifferentialDiff': team_a_stats['PointDifferential'] - team_b_stats['PointDifferential'],
                    'EloDiff': team_a_stats['Elo'] - team_b_stats['Elo'],
                    'FormDiff': team_a_stats['Form'] - team_b_stats['Form'],
                    'SameConference': same_conference
                })
            else:
                feature_a_b.update({
                    'TeamA_WinPct': team_b_stats['WinPct'],
                    'TeamB_WinPct': team_a_stats['WinPct'],
                    'TeamA_PointsScored': team_b_stats['PointsScored'],
                    'TeamB_PointsScored': team_a_stats['PointsScored'],
                    'TeamA_PointsAllowed': team_b_stats['PointsAllowed'],
                    'TeamB_PointsAllowed': team_a_stats['PointsAllowed'],
                    'TeamA_PointDifferential': team_b_stats['PointDifferential'],
                    'TeamB_PointDifferential': team_a_stats['PointDifferential'],
                    'TeamA_Elo': team_b_stats['Elo'],
                    'TeamB_Elo': team_a_stats['Elo'],
                    'TeamA_Form': team_b_stats['Form'],
                    'TeamB_Form': team_a_stats['Form'],
                    'WinPctDiff': team_b_stats['WinPct'] - team_a_stats['WinPct'],
                    'PointsScoredDiff': team_b_stats['PointsScored'] - team_a_stats['PointsScored'],
                    'PointsAllowedDiff': team_b_stats['PointsAllowed'] - team_a_stats['PointsAllowed'],
                    'PointDifferentialDiff': team_b_stats['PointDifferential'] - team_a_stats['PointDifferential'],
                    'EloDiff': team_b_stats['Elo'] - team_a_stats['Elo'],
                    'FormDiff': team_b_stats['Form'] - team_a_stats['Form'],
                    'SameConference': same_conference
                })
            
            # Add seed difference if both teams have seeds
            if team_a_stats['Seed'] is not None and team_b_stats['Seed'] is not None:
                if team_a < team_b:
                    feature_a_b['TeamA_Seed'] = team_a_stats['Seed']
                    feature_a_b['TeamB_Seed'] = team_b_stats['Seed']
                    feature_a_b['SeedDiff'] = team_a_stats['Seed'] - team_b_stats['Seed']
                else:
                    feature_a_b['TeamA_Seed'] = team_b_stats['Seed']
                    feature_a_b['TeamB_Seed'] = team_a_stats['Seed']
                    feature_a_b['SeedDiff'] = team_b_stats['Seed'] - team_a_stats['Seed']
            
            all_matchups.append(feature_a_b)
    
    return all_matchups

# Create enhanced features for men's tournament
M_features, M_submission_features, M_team_stats = create_enhanced_features('M', M_regular_season, M_tournament, M_teams, M_seeds, M_conferences, M_elo_ratings)

# Create enhanced features for women's tournament
W_features, W_submission_features, W_team_stats = create_enhanced_features('W', W_regular_season, W_tournament, W_teams, W_seeds, W_conferences, W_elo_ratings)

# Display sample of features
print("\nMen's Enhanced Features Sample:")
display(M_features.head())

print("\nMen's Submission Features Sample:")
display(M_submission_features.head())


# Define a function to train and evaluate models with improved calibration
def train_and_evaluate_models(gender, features_df, submission_features_df):
    print(f"Training and evaluating models for {gender} tournament...")
    
    # Prepare features and target
    X = features_df.drop(['Season', 'TeamA', 'TeamB', 'Result'], axis=1)
    y = features_df['Result']
    
    # Handle missing values
    X = X.fillna(0)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model evaluator
    evaluator = ModelEvaluator()
    
    # Train and evaluate Logistic Regression
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
    lr_model.fit(X_train, y_train)
    
    # Apply Isotonic Regression calibration to Logistic Regression
    print("Applying Isotonic Regression calibration to Logistic Regression...")
    lr_probs = lr_model.predict_proba(X_train)[:, 1]
    lr_isotonic = IsotonicRegression(out_of_bounds='clip')
    lr_isotonic.fit(lr_probs, y_train)
    
    # Create calibrated predictions
    lr_calibrated_probs = lr_isotonic.transform(lr_model.predict_proba(X_test)[:, 1])
    
    # Evaluate calibrated Logistic Regression
    lr_metrics, _ = evaluator.evaluate_model(lr_isotonic, lr_model.predict_proba(X_test)[:, 1], y_test, 'LogisticRegression_Calibrated', gender)
    print(f"Calibrated Logistic Regression - Brier Score: {lr_metrics['brier_score']:.6f}, Accuracy: {lr_metrics['accuracy']:.2%}")
    
    # Train and evaluate XGBoost
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Apply Isotonic Regression calibration to XGBoost
    print("Applying Isotonic Regression calibration to XGBoost...")
    xgb_probs = xgb_model.predict_proba(X_train)[:, 1]
    xgb_isotonic = IsotonicRegression(out_of_bounds='clip')
    xgb_isotonic.fit(xgb_probs, y_train)
    
    # Create calibrated predictions
    xgb_calibrated_probs = xgb_isotonic.transform(xgb_model.predict_proba(X_test)[:, 1])
    
    # Evaluate calibrated XGBoost
    xgb_metrics, _ = evaluator.evaluate_model(xgb_isotonic, xgb_model.predict_proba(X_test)[:, 1], y_test, 'XGBoost_Calibrated', gender)
    print(f"Calibrated XGBoost - Brier Score: {xgb_metrics['brier_score']:.6f}, Accuracy: {xgb_metrics['accuracy']:.2%}")
    
    # Train and evaluate LightGBM
    print("Training LightGBM model...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    
    # Apply Isotonic Regression calibration to LightGBM
    print("Applying Isotonic Regression calibration to LightGBM...")
    lgb_probs = lgb_model.predict_proba(X_train)[:, 1]
    lgb_isotonic = IsotonicRegression(out_of_bounds='clip')
    lgb_isotonic.fit(lgb_probs, y_train)
    
    # Create calibrated predictions
    lgb_calibrated_probs = lgb_isotonic.transform(lgb_model.predict_proba(X_test)[:, 1])
    
    # Evaluate calibrated LightGBM
    lgb_metrics, _ = evaluator.evaluate_model(lgb_isotonic, lgb_model.predict_proba(X_test)[:, 1], y_test, 'LightGBM_Calibrated', gender)
    print(f"Calibrated LightGBM - Brier Score: {lgb_metrics['brier_score']:.6f}, Accuracy: {lgb_metrics['accuracy']:.2%}")
    
    # Compare calibration of all models
    plt.figure(figsize=(12, 8))
    
    # Original models
    lr_fraction_of_positives, lr_mean_predicted_value = calibration_curve(y_test, lr_model.predict_proba(X_test)[:, 1], n_bins=10)
    xgb_fraction_of_positives, xgb_mean_predicted_value = calibration_curve(y_test, xgb_model.predict_proba(X_test)[:, 1], n_bins=10)
    lgb_fraction_of_positives, lgb_mean_predicted_value = calibration_curve(y_test, lgb_model.predict_proba(X_test)[:, 1], n_bins=10)
    
    # Calibrated models
    lr_cal_fraction_of_positives, lr_cal_mean_predicted_value = calibration_curve(y_test, lr_calibrated_probs, n_bins=10)
    xgb_cal_fraction_of_positives, xgb_cal_mean_predicted_value = calibration_curve(y_test, xgb_calibrated_probs, n_bins=10)
    lgb_cal_fraction_of_positives, lgb_cal_mean_predicted_value = calibration_curve(y_test, lgb_calibrated_probs, n_bins=10)
    
    # Plot calibration curves
    plt.plot(lr_mean_predicted_value, lr_fraction_of_positives, "s-", label="Logistic Regression")
    plt.plot(lr_cal_mean_predicted_value, lr_cal_fraction_of_positives, "s--", label="Logistic Regression (Calibrated)")
    plt.plot(xgb_mean_predicted_value, xgb_fraction_of_positives, "o-", label="XGBoost")
    plt.plot(xgb_cal_mean_predicted_value, xgb_cal_fraction_of_positives, "o--", label="XGBoost (Calibrated)")
    plt.plot(lgb_mean_predicted_value, lgb_fraction_of_positives, "^-", label="LightGBM")
    plt.plot(lgb_cal_mean_predicted_value, lgb_cal_fraction_of_positives, "^--", label="LightGBM (Calibrated)")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{gender} Tournament - Calibration Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/outputs/{gender}_calibration_comparison.png")
    
    # Prepare submission features
    submission_X = submission_features_df.drop(['Season', 'TeamA', 'TeamB', 'ID'], axis=1)
    submission_X = submission_X.fillna(0)
    
    # Generate predictions
    lr_submission_preds = lr_model.predict_proba(submission_X)[:, 1]
    xgb_submission_preds = xgb_model.predict_proba(submission_X)[:, 1]
    lgb_submission_preds = lgb_model.predict_proba(submission_X)[:, 1]
    
    # Apply calibration to predictions
    lr_submission_preds_calibrated = lr_isotonic.transform(lr_submission_preds)
    xgb_submission_preds_calibrated = xgb_isotonic.transform(xgb_submission_preds)
    lgb_submission_preds_calibrated = lgb_isotonic.transform(lgb_submission_preds)
    
    # Create submission DataFrames
    submission_df = submission_features_df[['ID']].copy()
    submission_df['LogisticRegression'] = lr_submission_preds_calibrated
    submission_df['XGBoost'] = xgb_submission_preds_calibrated
    submission_df['LightGBM'] = lgb_submission_preds_calibrated
    
    # Create ensemble predictions based on gender
    if gender == 'M':
        # For men's tournament, use Logistic Regression (best Brier score)
        submission_df['Ensemble'] = lr_submission_preds_calibrated
    else:
        # For women's tournament, use weighted blend (0.7*LightGBM + 0.3*XGBoost)
        submission_df['Ensemble'] = 0.7 * lgb_submission_preds_calibrated + 0.3 * xgb_submission_preds_calibrated
    
    # Save models and calibrators
    joblib.dump(lr_model, f"/kaggle/working/models/{gender}_LogisticRegression.pkl")
    joblib.dump(lr_isotonic, f"/kaggle/working/models/{gender}_LogisticRegression_Calibrator.pkl")
    joblib.dump(xgb_model, f"/kaggle/working/models/{gender}_XGBoost.pkl")
    joblib.dump(xgb_isotonic, f"/kaggle/working/models/{gender}_XGBoost_Calibrator.pkl")
    joblib.dump(lgb_model, f"/kaggle/working/models/{gender}_LightGBM.pkl")
    joblib.dump(lgb_isotonic, f"/kaggle/working/models/{gender}_LightGBM_Calibrator.pkl")
    
    # Save submission
    submission_df.to_csv(f"/kaggle/working/submissions/{gender}_submission.csv", index=False)
    
    # Feature importance analysis
    # Plot feature importance for all models
    plt.figure(figsize=(14, 10))
    
    # LightGBM feature importance
    ax1 = plt.subplot(2, 2, 1)
    lgb.plot_importance(lgb_model, ax=ax1, max_num_features=10)
    ax1.set_title(f"{gender} - LightGBM Feature Importance")
    
    # XGBoost feature importance
    ax2 = plt.subplot(2, 2, 2)
    xgb.plot_importance(xgb_model, ax=ax2, max_num_features=10)
    ax2.set_title(f"{gender} - XGBoost Feature Importance")
    
    # Logistic Regression coefficients
    ax3 = plt.subplot(2, 2, 3)
    coef = pd.Series(lr_model.coef_[0], index=X.columns)
    coef_abs = coef.abs().sort_values(ascending=False)
    top_features = coef_abs.index[:10]
    coef_sorted = coef[top_features]
    coef_sorted.plot(kind='barh', ax=ax3)
    ax3.set_title(f"{gender} - Logistic Regression Feature Importance")
    
    # Feature importance comparison
    ax4 = plt.subplot(2, 2, 4)
    
    # Get top 5 features from each model
    lgb_importance = pd.Series(lgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    lr_importance = coef_abs
    
    # Combine top features
    top_features_combined = pd.concat([
        lgb_importance.head(5).rename('LightGBM'),
        xgb_importance.head(5).rename('XGBoost'),
        lr_importance.head(5).rename('LogisticRegression')
    ], axis=1)
    
    # Fill NaN with 0
    top_features_combined = top_features_combined.fillna(0)
    
    # Normalize to 0-100 scale for comparison
    for col in top_features_combined.columns:
        if top_features_combined[col].sum() > 0:
            top_features_combined[col] = top_features_combined[col] / top_features_combined[col].max() * 100
    
    # Plot comparison
    top_features_combined.plot(kind='bar', ax=ax4)
    ax4.set_title(f"{gender} - Feature Importance Comparison")
    ax4.set_ylabel('Relative Importance (%)')
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/outputs/{gender}_feature_importance_comparison.png")
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Model': ['LogisticRegression', 'XGBoost', 'LightGBM'],
        'BrierScore': [lr_metrics['brier_score'], xgb_metrics['brier_score'], lgb_metrics['brier_score']],
        'Accuracy': [lr_metrics['accuracy'], xgb_metrics['accuracy'], lgb_metrics['accuracy']],
        'ROC_AUC': [lr_metrics['roc_auc'], xgb_metrics['roc_auc'], lgb_metrics['roc_auc']]
    })
    
    # Save metrics
    metrics_df.to_csv(f"/kaggle/working/outputs/{gender}_model_metrics.csv", index=False)
    
    return metrics_df, submission_df

# Train and evaluate models for men's tournament
M_metrics, M_submission = train_and_evaluate_models('M', M_features, M_submission_features)

# Train and evaluate models for women's tournament
W_metrics, W_submission = train_and_evaluate_models('W', W_features, W_submission_features)

# Display metrics
print("\nMen's Tournament Model Metrics:")
display(M_metrics)

print("\nWomen's Tournament Model Metrics:")
display(W_metrics)


# Generate 2025 matchups specifically
def generate_2025_matchups(M_teams, W_teams, M_team_stats, W_team_stats):
    print("Generating 2025 matchups...")
    
    # Create a mapping of team IDs to their information for 2025
    current_season = 2025
    current_season_str = str(current_season)
    
    # Create a list to store all 2025 matchups
    matchups_2025 = []
    
    # Get the most recent season with data as a proxy for 2025
    m_recent_season = max(M_team_stats.keys())
    w_recent_season = max(W_team_stats.keys())
    
    print(f"Using men's season {m_recent_season} and women's season {w_recent_season} as proxy for 2025")
    
    # Generate men's matchups for 2025
    men_count = 0
    for i, team_a in enumerate(M_teams['TeamID'].unique()):
        if team_a not in M_team_stats[m_recent_season]:
            continue
            
        for team_b in M_teams['TeamID'].unique()[i+1:]:
            if team_b not in M_team_stats[m_recent_season]:
                continue
                
            # Create matchup ID
            matchup_id = f"{current_season_str}_{team_a}_{team_b}"
            
            # Get team stats
            team_a_stats = M_team_stats[m_recent_season][team_a]
            team_b_stats = M_team_stats[m_recent_season][team_b]
            
            # Estimate probability based on Elo
            if 'Elo' in team_a_stats and 'Elo' in team_b_stats:
                elo_a = team_a_stats['Elo']
                elo_b = team_b_stats['Elo']
                prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
                prob = max(0.025, min(0.975, prob))  # Clip to valid range
            else:
                prob = 0.5
                
            matchups_2025.append({
                'ID': matchup_id,
                'Pred': prob
            })
            men_count += 1
    
    # Generate women's matchups for 2025
    women_count = 0
    for i, team_a in enumerate(W_teams['TeamID'].unique()):
        if team_a not in W_team_stats[w_recent_season]:
            continue
            
        for team_b in W_teams['TeamID'].unique()[i+1:]:
            if team_b not in W_team_stats[w_recent_season]:
                continue
                
            # Create matchup ID
            matchup_id = f"{current_season_str}_{team_a}_{team_b}"
            
            # Get team stats
            team_a_stats = W_team_stats[w_recent_season][team_a]
            team_b_stats = W_team_stats[w_recent_season][team_b]
            
            # Estimate probability based on Elo
            if 'Elo' in team_a_stats and 'Elo' in team_b_stats:
                elo_a = team_a_stats['Elo']
                elo_b = team_b_stats['Elo']
                prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
                prob = max(0.025, min(0.975, prob))  # Clip to valid range
            else:
                prob = 0.5
                
            matchups_2025.append({
                'ID': matchup_id,
                'Pred': prob
            })
            women_count += 1
    
    # Create DataFrame
    matchups_2025_df = pd.DataFrame(matchups_2025)
    
    print(f"Generated {men_count} men's and {women_count} women's matchups for 2025")
    print(f"Total 2025 matchups: {len(matchups_2025)}")
    
    return matchups_2025_df

# Create final submission with smart handling of missing matchups
def create_final_submission(M_submission, W_submission, sample_submission, M_teams, W_teams, M_seeds, W_seeds, 
                           M_elo_ratings, W_elo_ratings, M_team_stats, W_team_stats):
    print("Creating final submission with smart handling of missing matchups...")
    
    # Identify all seasons in the sample submission
    required_seasons = set()
    for match_id in sample_submission['ID']:
        season = match_id.split('_')[0]
        required_seasons.add(season)

    print(f"Submission requires predictions for these seasons: {sorted(required_seasons)}")
    
    # Create a mapping of team IDs to their information for all seasons
    team_info_by_season = {}
    
    # Process men's teams for all seasons
    for season in required_seasons:
        season_int = int(season)
        team_info_by_season[season] = {}
        
        # Get Elo ratings for this season if available
        season_elo = M_elo_ratings[M_elo_ratings['Season'] == season_int]
        
        # Get seeds for this season if available
        if season_int in M_team_stats:
            season_team_stats = M_team_stats[season_int]
        else:
            season_team_stats = {}
            
        # Get seeds for this season
        season_seeds = M_seeds[M_seeds['Season'] == season_int]
        
        # Process each team
        for team_id in M_teams['TeamID'].unique():
            team_info_by_season[season][team_id] = {'TeamName': M_teams[M_teams['TeamID'] == team_id]['TeamName'].values[0]}
            
            # Add team stats if available
            if team_id in season_team_stats:
                team_info_by_season[season][team_id].update(season_team_stats[team_id])
            
            # Add Elo rating if available
            elo_row = season_elo[season_elo['TeamID'] == team_id]
            if len(elo_row) > 0:
                team_info_by_season[season][team_id]['Elo'] = elo_row.iloc[0]['Elo']
            else:
                # Use most recent Elo if available
                recent_elo = M_elo_ratings[M_elo_ratings['TeamID'] == team_id].sort_values('Season', ascending=False)
                if len(recent_elo) > 0:
                    team_info_by_season[season][team_id]['Elo'] = recent_elo.iloc[0]['Elo']
            
            # Add seed if available
            seed_row = season_seeds[season_seeds['TeamID'] == team_id]
            if len(seed_row) > 0:
                seed_str = seed_row.iloc[0]['Seed']
                seed = int(''.join(filter(str.isdigit, str(seed_str))))
                team_info_by_season[season][team_id]['Seed'] = seed
    
    # Process women's teams similarly
    for season in required_seasons:
        season_int = int(season)
        if season not in team_info_by_season:
            team_info_by_season[season] = {}
        
        # Get Elo ratings for this season if available
        season_elo = W_elo_ratings[W_elo_ratings['Season'] == season_int]
        
        # Get seeds for this season if available
        if season_int in W_team_stats:
            season_team_stats = W_team_stats[season_int]
        else:
            season_team_stats = {}
            
        # Get seeds for this season
        season_seeds = W_seeds[W_seeds['Season'] == season_int]
        
        # Process each team
        for team_id in W_teams['TeamID'].unique():
            team_info_by_season[season][team_id] = {'TeamName': W_teams[W_teams['TeamID'] == team_id]['TeamName'].values[0]}
            
            # Add team stats if available
            if team_id in season_team_stats:
                team_info_by_season[season][team_id].update(season_team_stats[team_id])
            
            # Add Elo rating if available
            elo_row = season_elo[season_elo['TeamID'] == team_id]
            if len(elo_row) > 0:
                team_info_by_season[season][team_id]['Elo'] = elo_row.iloc[0]['Elo']
            else:
                # Use most recent Elo if available
                recent_elo = W_elo_ratings[W_elo_ratings['TeamID'] == team_id].sort_values('Season', ascending=False)
                if len(recent_elo) > 0:
                    team_info_by_season[season][team_id]['Elo'] = recent_elo.iloc[0]['Elo']
            
            # Add seed if available
            seed_row = season_seeds[season_seeds['TeamID'] == team_id]
            if len(seed_row) > 0:
                seed_str = seed_row.iloc[0]['Seed']
                seed = int(''.join(filter(str.isdigit, str(seed_str))))
                team_info_by_season[season][team_id]['Seed'] = seed
    
    # Function to estimate probability based on available information
    def estimate_probability(season, team_a_id, team_b_id):
        # Default probability
        default_prob = 0.5
        
        # Check if we have information for this season
        if season not in team_info_by_season:
            return default_prob
        
        # Check if we have information for both teams
        if team_a_id not in team_info_by_season[season] or team_b_id not in team_info_by_season[season]:
            return default_prob
        
        team_a_info = team_info_by_season[season][team_a_id]
        team_b_info = team_info_by_season[season][team_b_id]
        
        # If both teams have Elo ratings, use Elo formula
        if 'Elo' in team_a_info and 'Elo' in team_b_info:
            elo_a = team_a_info['Elo']
            elo_b = team_b_info['Elo']
            prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
            return prob
        
        # If both teams have seeds, use seed difference
        elif 'Seed' in team_a_info and 'Seed' in team_b_info:
            seed_a = team_a_info['Seed']
            seed_b = team_b_info['Seed']
            # Higher seeds (lower numbers) are better
            if seed_a < seed_b:
                # Team A is favored
                return 0.5 + min(0.4, (seed_b - seed_a) * 0.04)
            elif seed_b < seed_a:
                # Team B is favored
                return 0.5 - min(0.4, (seed_a - seed_b) * 0.04)
            else:
                # Equal seeds
                return 0.5
        
        # If no information is available, use default probability
        return default_prob
    
    # Create a mapping from ID to prediction
    pred_dict = {}
    
    # Add men's predictions
    for _, row in M_submission.iterrows():
        pred_dict[row['ID']] = row['Ensemble']
    
    # Add women's predictions
    for _, row in W_submission.iterrows():
        pred_dict[row['ID']] = row['Ensemble']
    
    # Generate 2025 matchups
    matchups_2025_df = generate_2025_matchups(M_teams, W_teams, M_team_stats, W_team_stats)
    
    # Add 2025 predictions to the dictionary
    for _, row in matchups_2025_df.iterrows():
        pred_dict[row['ID']] = row['Pred']
    
    # Create final submission matching the sample format
    final_submission = sample_submission.copy()
    final_submission['Pred'] = final_submission['ID'].map(pred_dict)
    
    # Apply smart predictions to missing matchups
    def get_smart_prediction(match_id):
        # If we already have a prediction, use it
        if match_id in pred_dict:
            return pred_dict[match_id]
            
        # Otherwise, estimate based on team information
        parts = match_id.split('_')
        if len(parts) != 3:
            return 0.5
        
        season = parts[0]
        team_a_id = int(parts[1])
        team_b_id = int(parts[2])
        
        return estimate_probability(season, team_a_id, team_b_id)
    
    # Apply the smart prediction function to missing values
    missing_mask = final_submission['Pred'].isna()
    if missing_mask.any():
        print(f"Applying smart predictions to {missing_mask.sum()} missing matchups...")
        final_submission.loc[missing_mask, 'Pred'] = final_submission.loc[missing_mask, 'ID'].apply(get_smart_prediction)

    # Add the 2025 matchups to the final submission
    print(f"Adding {len(matchups_2025_df)} matchups for 2025 to the final submission...")
    final_submission = pd.concat([final_submission, matchups_2025_df], ignore_index=True)

    # Ensure predictions are within valid range
    final_submission['Pred'] = final_submission['Pred'].clip(0.025, 0.975)
    
    # Save the final submission
    final_submission.to_csv("/kaggle/working/submissions/final_submission.csv", index=False)
    print("Created final_submission.csv with smart predictions for all required matchups.")
    
    # Check years in the submission
    all_years = [id.split('_')[0] for id in final_submission['ID']]
    years_count = pd.Series(all_years).value_counts()
    print("\nSubmission breakdown by year:")
    print(years_count)
    
    # Display sample of final submission
    print("\nFinal Submission Sample:")
    display(final_submission.head())
    
    # Calculate summary statistics for predictions
    print("\nPrediction Statistics:")
    print(f"Total predictions: {len(final_submission)}")
    print(f"Mean prediction: {final_submission['Pred'].mean():.4f}")
    print(f"Median prediction: {final_submission['Pred'].median():.4f}")
    print(f"Min prediction: {final_submission['Pred'].min():.4f}")
    print(f"Max prediction: {final_submission['Pred'].max():.4f}")
    
    # Plot prediction distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(final_submission['Pred'], bins=50, kde=True)
    plt.title('Distribution of Final Predictions')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.savefig("/kaggle/working/outputs/prediction_distribution.png")
    
    return final_submission

# Create final submission
final_submission = create_final_submission(
    M_submission, W_submission, sample_submission, 
    M_teams, W_teams, M_seeds, W_seeds, 
    M_elo_ratings, W_elo_ratings,
    M_team_stats, W_team_stats
)


# PART 8: Final Submission Verification

print("PART 8: Final Submission Verification")

# Verify the final submission
print("\nVerifying final submission...")

# Check for required seasons
required_seasons = ['2021', '2022', '2023', '2024', '2025']
submission_seasons = set([id.split('_')[0] for id in final_submission['ID']])
missing_seasons = set(required_seasons) - submission_seasons

if missing_seasons:
    print(f"WARNING: Missing predictions for seasons: {missing_seasons}")
else:
    print("All required seasons are present in the submission.")

# Check prediction range
min_pred = final_submission['Pred'].min()
max_pred = final_submission['Pred'].max()
print(f"Prediction range: {min_pred:.4f} to {max_pred:.4f}")

if min_pred < 0.025 or max_pred > 0.975:
    print("WARNING: Predictions outside valid range (0.025-0.975)")
else:
    print("All predictions within valid range (0.025-0.975)")

# Check for duplicate IDs
duplicate_ids = final_submission['ID'].duplicated().sum()
if duplicate_ids > 0:
    print(f"WARNING: Found {duplicate_ids} duplicate IDs in submission")
else:
    print("No duplicate IDs found in submission")

# Check submission size
print(f"Final submission size: {len(final_submission)} rows")

# Breakdown by gender (approximate based on ID format)
men_matchups = sum(1 for id in final_submission['ID'] if id.split('_')[1].startswith('1'))
women_matchups = len(final_submission) - men_matchups
print(f"Men's matchups (approximate): {men_matchups}")
print(f"Women's matchups (approximate): {women_matchups}")

# Breakdown by year
year_counts = pd.Series([id.split('_')[0] for id in final_submission['ID']]).value_counts().sort_index()
print("\nSubmission breakdown by year:")
print(year_counts)

# Plot prediction distribution
plt.figure(figsize=(10, 6))
sns.histplot(final_submission['Pred'], bins=50, kde=True)
plt.title('Distribution of Final Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.savefig("/kaggle/working/outputs/final_prediction_distribution.png")
plt.show()

print("\nFinal submission is ready for upload to Kaggle!")

