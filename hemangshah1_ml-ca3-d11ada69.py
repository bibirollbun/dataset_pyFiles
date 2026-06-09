import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import time
import warnings
from tqdm.notebook import tqdm
from datetime import datetime

# Machine Learning Libraries
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, brier_score_loss
import lightgbm as lgb
import xgboost as xgb
import optuna
import shap

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Suppress warnings
warnings.filterwarnings('ignore')


BASE_DIR = '/kaggle/input/march-machine-learning-mania-2025/'

# Helper function to read in the data files
def load_data(file_path):
    return pd.read_csv(os.path.join(BASE_DIR, file_path))

# Load the core data files (Men's Tournament)
m_teams = load_data('MTeams.csv')
m_seasons = load_data('MSeasons.csv')
m_tourney_seeds = load_data('MNCAATourneySeeds.csv')
m_tourney_results = load_data('MNCAATourneyCompactResults.csv')
m_regular_season = load_data('MRegularSeasonCompactResults.csv')
m_tourney_detailed = load_data('MNCAATourneyDetailedResults.csv')
m_regular_detailed = load_data('MRegularSeasonDetailedResults.csv')

# Load the core data files (Women's Tournament)
w_teams = load_data('WTeams.csv')
w_seasons = load_data('WSeasons.csv')
w_tourney_seeds = load_data('WNCAATourneySeeds.csv')
w_tourney_results = load_data('WNCAATourneyCompactResults.csv')
w_regular_season = load_data('WRegularSeasonCompactResults.csv')
w_tourney_detailed = load_data('WNCAATourneyDetailedResults.csv')
w_regular_detailed = load_data('WRegularSeasonDetailedResults.csv')

# Load supplementary files
conferences = load_data('Conferences.csv')
m_team_conferences = load_data('MTeamConferences.csv')
w_team_conferences = load_data('WTeamConferences.csv')
cities = load_data('Cities.csv')
m_game_cities = load_data('MGameCities.csv')
w_game_cities = load_data('WGameCities.csv')

print("Data loading completed.")


# Display basic information about the datasets
print("Men's Teams Dataset:")
print(m_teams.head())
print(f"Number of teams: {m_teams.shape[0]}")

print("\nWomen's Teams Dataset:")
print(w_teams.head())
print(f"Number of teams: {w_teams.shape[0]}")

print("\nMen's Tournament Seeds (recent season):")
latest_season = m_tourney_seeds['Season'].max()
print(m_tourney_seeds[m_tourney_seeds['Season'] == latest_season].head())

print("\nMen's Tournament Results (recent season):")
print(m_tourney_results[m_tourney_results['Season'] == latest_season].head())

print("\nConferences Dataset:")
print(conferences.head())


plt.style.use('fivethirtyeight')
sns.set_palette('bright')


print("Unique DayNum values in women's tournament data:", w_tourney_results['DayNum'].unique())

# Find the maximum DayNum (likely the championship game day)
max_day_women = w_tourney_results['DayNum'].max()
print(f"Maximum DayNum in women's data: {max_day_women}")

# Try using the maximum DayNum instead of hardcoding 154
w_champions = w_tourney_results[w_tourney_results['DayNum'] == max_day_women].copy()
print(f"Number of championship games found: {len(w_champions)}")

# Check the seed format
if len(w_champions) > 0:
    w_champions = w_champions.merge(w_tourney_seeds, left_on=['Season', 'WTeamID'], 
                                  right_on=['Season', 'TeamID'], how='left')
    print("Sample of merged data:")
    print(w_champions[['Season', 'WTeamID', 'Seed']].head())
    
    # Check if we have valid Seed values before extracting
    if w_champions['Seed'].notna().sum() > 0:
        # Check the format of the Seed column
        print("Sample seed values:", w_champions['Seed'].dropna().head().tolist())
        
        # Extract the seed number with more robust error handling
        w_champions['SeedNum'] = w_champions['Seed'].str.extract('(\d+)').astype(float).astype('Int64')
        
        # Proceed with plotting only if we have data
        if w_champions['SeedNum'].notna().sum() > 0:
            ax = sns.countplot(x='SeedNum', data=w_champions)
            plt.title("Distribution of NCAA Women's Tournament Champion Seeds", fontsize=16)
            plt.xlabel('Seed Number', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width()/2., p.get_height()), 
                            ha='center', va='bottom')
            
            plt.show()
        else:
            print("No valid seed numbers found after extraction")
    else:
        print("No valid seed values found after merging")
else:
    print("No championship games found in women's data")


# Define an upset as a lower seed beating a higher seed
m_upsets = m_tourney_results.copy()

# Add seed information for winning and losing teams
m_upsets = m_upsets.merge(m_tourney_seeds, left_on=['Season', 'WTeamID'], 
                          right_on=['Season', 'TeamID'], 
                          how='left', suffixes=('', '_winner'))
m_upsets = m_upsets.merge(m_tourney_seeds, left_on=['Season', 'LTeamID'], 
                          right_on=['Season', 'TeamID'], 
                          how='left', suffixes=('', '_loser'))

# Extract seed numbers
m_upsets['WSeedNum'] = m_upsets['Seed'].str.extract('(\d+)').astype(int)
m_upsets['LSeedNum'] = m_upsets['Seed_loser'].str.extract('(\d+)').astype(int)

# Define upsets (higher seed number beats lower seed number)
m_upsets['Upset'] = m_upsets['WSeedNum'] > m_upsets['LSeedNum']

# Upset percentage by round
plt.figure(figsize=(14, 7))
round_days = {
    'Round 1': [136, 137],
    'Round 2': [138, 139],
    'Sweet 16': [143, 144],
    'Elite 8': [145, 146],
    'Final 4': [152],
    'Championship': [154]
}

upset_by_round = []
for round_name, days in round_days.items():
    round_games = m_upsets[m_upsets['DayNum'].isin(days)]
    upset_count = round_games['Upset'].sum()
    total_games = len(round_games)
    upset_pct = (upset_count / total_games) * 100 if total_games > 0 else 0
    upset_by_round.append((round_name, upset_pct, upset_count, total_games))

upset_df = pd.DataFrame(upset_by_round, columns=['Round', 'Upset Percentage', 'Upset Count', 'Total Games'])
ax = sns.barplot(x='Round', y='Upset Percentage', data=upset_df)
plt.title("Upset Percentage by Tournament Round (Men's)", fontsize=16)
plt.ylabel('Upset Percentage (%)', fontsize=14)
plt.xlabel('Tournament Round', fontsize=14)
plt.xticks(rotation=45)

# Add percentage and count labels
for i, p in enumerate(ax.patches):
    count = upset_df.iloc[i]['Upset Count']
    total = upset_df.iloc[i]['Total Games']
    ax.annotate(f'{p.get_height():.1f}% ({count}/{total})', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Create a function to calculate team statistics from regular season data
def calculate_team_stats(regular_season_data, detailed_data=None):

    # Get all seasons in the dataset
    seasons = regular_season_data['Season'].unique()
    
    # Initialize a list to store dataframes for each season
    all_season_stats = []
    
    for season in tqdm(seasons, desc="Calculating team stats"):
        season_data = regular_season_data[regular_season_data['Season'] == season].copy()
        
        # Create dictionaries to store team stats
        games_played = {}
        wins = {}
        points_scored = {}
        points_allowed = {}
        home_games = {}
        away_games = {}
        neutral_games = {}
        home_wins = {}
        away_wins = {}
        neutral_wins = {}
        
        # Populate team stats
        for _, row in season_data.iterrows():
            # Winning team stats
            w_team = row['WTeamID']
            if w_team not in games_played:
                games_played[w_team] = 0
                wins[w_team] = 0
                points_scored[w_team] = 0
                points_allowed[w_team] = 0
                home_games[w_team] = 0
                away_games[w_team] = 0
                neutral_games[w_team] = 0
                home_wins[w_team] = 0
                away_wins[w_team] = 0
                neutral_wins[w_team] = 0
            
            games_played[w_team] += 1
            wins[w_team] += 1
            points_scored[w_team] += row['WScore']
            points_allowed[w_team] += row['LScore']
            
            # Track location stats
            if row['WLoc'] == 'H':
                home_games[w_team] += 1
                home_wins[w_team] += 1
            elif row['WLoc'] == 'A':
                away_games[w_team] += 1
                away_wins[w_team] += 1
            else:  # Neutral
                neutral_games[w_team] += 1
                neutral_wins[w_team] += 1
            
            # Losing team stats
            l_team = row['LTeamID']
            if l_team not in games_played:
                games_played[l_team] = 0
                wins[l_team] = 0
                points_scored[l_team] = 0
                points_allowed[l_team] = 0
                home_games[l_team] = 0
                away_games[l_team] = 0
                neutral_games[l_team] = 0
                home_wins[l_team] = 0
                away_wins[l_team] = 0
                neutral_wins[l_team] = 0
            
            games_played[l_team] += 1
            points_scored[l_team] += row['LScore']
            points_allowed[l_team] += row['WScore']
            
            # Track location stats for losing team
            if row['WLoc'] == 'A':  # If winner was away, loser was home
                home_games[l_team] += 1
            elif row['WLoc'] == 'H':  # If winner was home, loser was away
                away_games[l_team] += 1
            else:  # Neutral
                neutral_games[l_team] += 1
        
        # Create a dataframe from the dictionaries
        season_stats = pd.DataFrame({
            'TeamID': list(games_played.keys()),
            'Season': season,
            'GamesPlayed': list(games_played.values()),
            'Wins': list(wins.values()),
            'PointsScored': list(points_scored.values()),
            'PointsAllowed': list(points_allowed.values()),
            'HomeGames': list(home_games.values()),
            'AwayGames': list(away_games.values()),
            'NeutralGames': list(neutral_games.values()),
            'HomeWins': list(home_wins.values()),
            'AwayWins': list(away_wins.values()),
            'NeutralWins': list(neutral_wins.values())
        })
        
        # Calculate derived statistics
        season_stats['WinPct'] = season_stats['Wins'] / season_stats['GamesPlayed']
        season_stats['PointsPerGame'] = season_stats['PointsScored'] / season_stats['GamesPlayed']
        season_stats['PointsAllowedPerGame'] = season_stats['PointsAllowed'] / season_stats['GamesPlayed']
        season_stats['PointDifferential'] = season_stats['PointsPerGame'] - season_stats['PointsAllowedPerGame']
        
        # Home/Away/Neutral metrics
        season_stats['HomeWinPct'] = season_stats['HomeWins'] / season_stats['HomeGames'].replace(0, 1)  # Avoid div by zero
        season_stats['AwayWinPct'] = season_stats['AwayWins'] / season_stats['AwayGames'].replace(0, 1)
        season_stats['NeutralWinPct'] = season_stats['NeutralWins'] / season_stats['NeutralGames'].replace(0, 1)
        
        all_season_stats.append(season_stats)
    
    # Combine all seasons into a single dataframe
    team_stats = pd.concat(all_season_stats, ignore_index=True)
    
    return team_stats

# Calculate team stats for men's and women's data
print("Calculating men's team statistics...")
m_team_stats = calculate_team_stats(m_regular_season, m_regular_detailed)
print("Calculating women's team statistics...")
w_team_stats = calculate_team_stats(w_regular_season, w_regular_detailed)
print("Team statistics calculation completed.")

# Add team names
m_team_stats = m_team_stats.merge(m_teams[['TeamID', 'TeamName']], on='TeamID', how='left')
w_team_stats = w_team_stats.merge(w_teams[['TeamID', 'TeamName']], on='TeamID', how='left')

# Display stats for a recent season
recent_season = m_team_stats[m_team_stats['Season'] == m_team_stats['Season'].max()].sort_values(by='WinPct', ascending=False)
print("\nTop 10 men's teams by win percentage (most recent season):")
cols_to_show = ['TeamName', 'GamesPlayed', 'Wins', 'WinPct', 'PointsPerGame', 
                'PointsAllowedPerGame', 'PointDifferential']
print(recent_season[cols_to_show].head(10))


plt.figure(figsize=(12, 8))
sns.scatterplot(x='PointDifferential', y='WinPct', data=recent_season, 
                hue='TeamName', size='GamesPlayed', sizes=(50, 200), alpha=0.7)
plt.title('Win Percentage vs. Point Differential (Most Recent Season)', fontsize=16)
plt.xlabel('Point Differential (Points Per Game)', fontsize=14)
plt.ylabel('Win Percentage', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plot offensive vs. defensive metrics
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PointsAllowedPerGame', y='PointsPerGame', data=recent_season, 
                hue='TeamName', size='WinPct', sizes=(50, 200), alpha=0.7)
plt.title('Offense vs. Defense (Most Recent Season)', fontsize=16)
plt.xlabel('Points Allowed Per Game (Lower is Better)', fontsize=14)
plt.ylabel('Points Scored Per Game', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plot home vs away win percentage
plt.figure(figsize=(10, 8))
plt.scatter(recent_season['HomeWinPct'], recent_season['AwayWinPct'], 
            s=recent_season['GamesPlayed']*3, alpha=0.6)

# Add team labels for top teams
top_teams = recent_season.head(15)
for i, row in top_teams.iterrows():
    plt.annotate(row['TeamName'], 
                 (row['HomeWinPct']+0.01, row['AwayWinPct']+0.01),
                 fontsize=9)

plt.title('Home vs. Away Win Percentage (Most Recent Season)', fontsize=13)
plt.xlabel('Home Win Percentage', fontsize=12)
plt.ylabel('Away Win Percentage', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0.3, 1.05)
plt.ylim(0.3, 1.05)
# Add diagonal line
plt.plot([0.3, 1.05], [0.3, 1.05], 'k--', alpha=0.3)
plt.tight_layout()
plt.show()



seed_performance = []

for i in range(1, 17):  # Seeds 1 to 16
    # Filter teams with this seed
    seed_pattern = f'[WXYZ]{i:02d}'
    seed_teams = m_tourney_seeds[m_tourney_seeds['Seed'].str.match(seed_pattern)]
    
    # Count wins as winning team
    wins = m_tourney_results.merge(
        seed_teams, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='inner'
    ).shape[0]
    
    # Count games played (as either winning or losing team)
    games_played = (
        m_tourney_results.merge(
            seed_teams, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='inner'
        ).shape[0] +
        m_tourney_results.merge(
            seed_teams, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='inner'
        ).shape[0]
    )
    
    if games_played > 0:
        win_pct = wins / games_played
    else:
        win_pct = 0
    
    seed_performance.append({'Seed': i, 'Win_Pct': win_pct, 'Wins': wins, 'Games_Played': games_played})

seed_df = pd.DataFrame(seed_performance)

plt.figure(figsize=(14, 7))
ax = sns.barplot(x='Seed', y='Win_Pct', data=seed_df)
plt.title("Tournament Win Percentage by Seed (Men's)", fontsize=16)
plt.xlabel('Seed', fontsize=14)
plt.ylabel('Win Percentage', fontsize=14)

# Add percentage and count labels
for i, p in enumerate(ax.patches):
    wins = seed_df.iloc[i]['Wins']
    games = seed_df.iloc[i]['Games_Played']
    ax.annotate(f'{p.get_height():.2f}\n({wins}/{games})', 
                (p.get_x() + p.get_width()/2., p.get_height()+0.01), 
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Repeat for women's tournament
w_seed_performance = []

for i in range(1, 17):
    seed_pattern = f'[WXYZ]{i:02d}'
    seed_teams = w_tourney_seeds[w_tourney_seeds['Seed'].str.match(seed_pattern)]
    
    wins = w_tourney_results.merge(
        seed_teams, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='inner'
    ).shape[0]
    
    games_played = (
        w_tourney_results.merge(
            seed_teams, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='inner'
        ).shape[0] +
        w_tourney_results.merge(
            seed_teams, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='inner'
        ).shape[0]
    )
    
    if games_played > 0:
        win_pct = wins / games_played
    else:
        win_pct = 0
    
    w_seed_performance.append({'Seed': i, 'Win_Pct': win_pct, 'Wins': wins, 'Games_Played': games_played})

w_seed_df = pd.DataFrame(w_seed_performance)

plt.figure(figsize=(14, 7))
ax = sns.barplot(x='Seed', y='Win_Pct', data=w_seed_df)
plt.title("Tournament Win Percentage by Seed (Women's)", fontsize=16)
plt.xlabel('Seed', fontsize=14)
plt.ylabel('Win Percentage', fontsize=14)

for i, p in enumerate(ax.patches):
    wins = w_seed_df.iloc[i]['Wins']
    games = w_seed_df.iloc[i]['Games_Played']
    ax.annotate(f'{p.get_height():.2f}\n({wins}/{games})', 
                (p.get_x() + p.get_width()/2., p.get_height()+0.01), 
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# Define a function to add team-specific features for a matchup
def get_team_features(team_stats, team_id, season, prefix=''):
    """
    Extract features for a specific team in a specific season
    
    Parameters:
    -----------
    team_stats: DataFrame with team statistics
    team_id: TeamID to get features for
    season: Season to get features for
    prefix: Prefix to add to column names (e.g., 'Team1_' or 'Team2_')
    
    Returns:
    --------
    Dictionary of features for the team
    """
    # Get team stats for this season
    team_data = team_stats[(team_stats['TeamID'] == team_id) & 
                          (team_stats['Season'] == season)]
    
    if len(team_data) == 0:
        return {}
    
    # Basic team performance metrics
    features = {
        f'{prefix}WinPct': team_data['WinPct'].iloc[0],
        f'{prefix}PointsPerGame': team_data['PointsPerGame'].iloc[0],
        f'{prefix}PointsAllowedPerGame': team_data['PointsAllowedPerGame'].iloc[0],
        f'{prefix}PointDifferential': team_data['PointDifferential'].iloc[0],
        f'{prefix}GamesPlayed': team_data['GamesPlayed'].iloc[0],
        f'{prefix}HomeWinPct': team_data['HomeWinPct'].iloc[0],
        f'{prefix}AwayWinPct': team_data['AwayWinPct'].iloc[0],
        f'{prefix}NeutralWinPct': team_data['NeutralWinPct'].iloc[0],
    }
    
    return features

# Function to extract seed-related features
def get_seed_features(tourney_seeds, team_id, season, prefix=''):
    """
    Extract seed-related features for a team
    
    Parameters:
    -----------
    tourney_seeds: DataFrame with tournament seeds
    team_id: TeamID to get features for
    season: Season to get features for
    prefix: Prefix to add to column names
    
    Returns:
    --------
    Dictionary of seed features for the team
    """
    # Get seed for this team in this season
    team_seed_info = tourney_seeds[(tourney_seeds['TeamID'] == team_id) & 
                                  (tourney_seeds['Season'] == season)]
    
    if len(team_seed_info) == 0:
        return {f'{prefix}SeedNumber': None}
    
    # Extract numerical seed (e.g., from 'W01' get 1)
    seed_str = team_seed_info['Seed'].iloc[0]
    seed_num = int(seed_str[1:3])
    
    return {f'{prefix}SeedNumber': seed_num}

# Function to check head-to-head results in the regular season
def get_head_to_head_features(regular_season, team1_id, team2_id, season):
    """
    Get features based on head-to-head matchups between the teams
    
    Parameters:
    -----------
    regular_season: DataFrame with regular season results
    team1_id: First team ID
    team2_id: Second team ID
    season: Season to analyze
    
    Returns:
    --------
    Dictionary of head-to-head features
    """
    # Filter to this season
    season_games = regular_season[regular_season['Season'] == season]
    
    # Team1 wins against Team2
    team1_wins = season_games[
        ((season_games['WTeamID'] == team1_id) & (season_games['LTeamID'] == team2_id))
    ]
    
    # Team2 wins against Team1
    team2_wins = season_games[
        ((season_games['WTeamID'] == team2_id) & (season_games['LTeamID'] == team1_id))
    ]
    
    # Calculate features
    matchup_count = len(team1_wins) + len(team2_wins)
    
    if matchup_count > 0:
        team1_win_pct = len(team1_wins) / matchup_count
    else:
        team1_win_pct = None
    
    return {
        'MatchupCount': matchup_count,
        'Team1WinPctH2H': team1_win_pct
    }

# Create a function to generate features for tournament matchups
def create_matchup_dataset(seasons_to_use, is_mens=True, include_tourney_results=True):
    """
    Generate a dataset of features for tournament matchups
    
    Parameters:
    -----------
    seasons_to_use: List of seasons to include
    is_mens: Whether to use men's data (True) or women's data (False)
    include_tourney_results: Whether to include actual tournament results
    
    Returns:
    --------
    DataFrame with features and outcomes for matchups
    """
    # Choose appropriate datasets based on gender
    if is_mens:
        teams = m_teams
        team_stats = m_team_stats
        tourney_seeds = m_tourney_seeds
        regular_season = m_regular_season
        tourney_results = m_tourney_results
        gender_prefix = 'M'
    else:
        teams = w_teams
        team_stats = w_team_stats
        tourney_seeds = w_tourney_seeds
        regular_season = w_regular_season
        tourney_results = w_tourney_results
        gender_prefix = 'W'
    
    print(f"Creating matchup dataset for {gender_prefix} seasons: {seasons_to_use}")
    
    # Initialize list to store matchup data
    matchups = []
    
    # For each tournament season
    for season in tqdm(seasons_to_use, desc=f"{gender_prefix} Matchups"):
        # Get teams in this season's tournament
        season_teams = tourney_seeds[tourney_seeds['Season'] == season]['TeamID'].unique()
        
        # Generate all possible matchups between these teams
        for team1_id in season_teams:
            for team2_id in season_teams:
                if team1_id < team2_id:  # Ensure each matchup is only included once
                    # Create matchup ID (format used in the competition)
                    matchup_id = f"{season}_{team1_id}_{team2_id}"
                    
                                        # Get base features for each team
                    matchup_features = {}
                    
                    # Team1 features
                    team1_features = get_team_features(team_stats, team1_id, season, 'Team1_')
                    matchup_features.update(team1_features)
                    
                    # Team2 features
                    team2_features = get_team_features(team_stats, team2_id, season, 'Team2_')
                    matchup_features.update(team2_features)
                    
                    # Seed features
                    seed1_features = get_seed_features(tourney_seeds, team1_id, season, 'Team1_')
                    seed2_features = get_seed_features(tourney_seeds, team2_id, season, 'Team2_')
                    matchup_features.update(seed1_features)
                    matchup_features.update(seed2_features)
                    
                    # Head-to-head features
                    h2h_features = get_head_to_head_features(regular_season, team1_id, team2_id, season)
                    matchup_features.update(h2h_features)
                    
                    # Add team identifiers
                    matchup_features['ID'] = matchup_id
                    matchup_features['Season'] = season
                    matchup_features['Team1ID'] = team1_id
                    matchup_features['Team2ID'] = team2_id
                    
                    # If we're including tournament results, check if they played each other
                    if include_tourney_results:
                        # Check if Team1 won against Team2 in the tournament
                        team1_won = tourney_results[
                            (tourney_results['Season'] == season) & 
                            (tourney_results['WTeamID'] == team1_id) & 
                            (tourney_results['LTeamID'] == team2_id)
                        ].shape[0] > 0
                        
                        # Check if Team2 won against Team1 in the tournament
                        team2_won = tourney_results[
                            (tourney_results['Season'] == season) & 
                            (tourney_results['WTeamID'] == team2_id) & 
                            (tourney_results['LTeamID'] == team1_id)
                        ].shape[0] > 0
                        
                        if team1_won:
                            matchup_features['Result'] = 1
                        elif team2_won:
                            matchup_features['Result'] = 0
                        else:
                            matchup_features['Result'] = None
                    
                    # Add computed features
                    # Calculate differences in team metrics
                    if 'Team1_WinPct' in matchup_features and 'Team2_WinPct' in matchup_features:
                        matchup_features['WinPctDiff'] = matchup_features['Team1_WinPct'] - matchup_features['Team2_WinPct']
                    
                    if 'Team1_PointsPerGame' in matchup_features and 'Team2_PointsPerGame' in matchup_features:
                        matchup_features['ScoringDiff'] = matchup_features['Team1_PointsPerGame'] - matchup_features['Team2_PointsPerGame']
                    
                    if 'Team1_PointsAllowedPerGame' in matchup_features and 'Team2_PointsAllowedPerGame' in matchup_features:
                        matchup_features['DefenseDiff'] = matchup_features['Team1_PointsAllowedPerGame'] - matchup_features['Team2_PointsAllowedPerGame']
                    
                    if 'Team1_SeedNumber' in matchup_features and 'Team2_SeedNumber' in matchup_features:
                        if matchup_features['Team1_SeedNumber'] is not None and matchup_features['Team2_SeedNumber'] is not None:
                            matchup_features['SeedDiff'] = matchup_features['Team1_SeedNumber'] - matchup_features['Team2_SeedNumber']
                    
                    # Store this matchup
                    matchups.append(matchup_features)
    
    # Convert to DataFrame
    matchup_df = pd.DataFrame(matchups)
    
    # Add team names for easier interpretation
    matchup_df = matchup_df.merge(teams[['TeamID', 'TeamName']], 
                                  left_on='Team1ID', right_on='TeamID', how='left')
    matchup_df = matchup_df.rename(columns={'TeamName': 'Team1Name'}).drop('TeamID', axis=1)
    
    matchup_df = matchup_df.merge(teams[['TeamID', 'TeamName']], 
                                  left_on='Team2ID', right_on='TeamID', how='left')
    matchup_df = matchup_df.rename(columns={'TeamName': 'Team2Name'}).drop('TeamID', axis=1)
    
    return matchup_df


train_seasons = list(range(2003, 2021))  # 2003-2020
valid_seasons = list(range(2021, 2025))  # 2021-2024

# Create datasets
men_matchups_train = create_matchup_dataset(train_seasons, is_mens=True)
men_matchups_valid = create_matchup_dataset(valid_seasons, is_mens=True)
women_matchups_train = create_matchup_dataset(train_seasons, is_mens=False)
women_matchups_valid = create_matchup_dataset(valid_seasons, is_mens=False)

print(f"Men's training matchups: {men_matchups_train.shape}")
print(f"Men's validation matchups: {men_matchups_valid.shape}")
print(f"Women's training matchups: {women_matchups_train.shape}")
print(f"Women's validation matchups: {women_matchups_valid.shape}")

# Display sample of matchup data
print("\nSample of men's matchup data:")
men_matchups_train.sample(5).head()


def prepare_features(matchup_df):
    """
    Prepare features for modeling by handling missing values,
    selecting relevant features, and creating derived features
    
    Parameters:
    -----------
    matchup_df: DataFrame with matchup features
    
    Returns:
    --------
    X: Feature matrix
    y: Target vector (if available)
    feature_names: List of feature names
    """
    # Make a copy to avoid modifying the original
    df = matchup_df.copy()
    
    # Select relevant features
    feature_cols = [
        # Basic team performance metrics
        'Team1_WinPct', 'Team2_WinPct', 'WinPctDiff',
        'Team1_PointsPerGame', 'Team2_PointsPerGame', 'ScoringDiff',
        'Team1_PointsAllowedPerGame', 'Team2_PointsAllowedPerGame', 'DefenseDiff',
        'Team1_PointDifferential', 'Team2_PointDifferential',
        
        # Home/Away performance
        'Team1_HomeWinPct', 'Team2_HomeWinPct',
        'Team1_AwayWinPct', 'Team2_AwayWinPct',
        'Team1_NeutralWinPct', 'Team2_NeutralWinPct',
        
        # Seed information
        'Team1_SeedNumber', 'Team2_SeedNumber', 'SeedDiff',
        
        # Head-to-head information
        'MatchupCount', 'Team1WinPctH2H'
    ]
    
    # Filter to only include columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Handle missing values
    df_features = df[feature_cols].copy()
    
    # Fill missing seed numbers with high value (assume unseeded teams are worst)
    if 'Team1_SeedNumber' in df_features:
        df_features['Team1_SeedNumber'] = df_features['Team1_SeedNumber'].fillna(17)
    if 'Team2_SeedNumber' in df_features:
        df_features['Team2_SeedNumber'] = df_features['Team2_SeedNumber'].fillna(17)
    if 'SeedDiff' in df_features:
        df_features['SeedDiff'] = df_features['SeedDiff'].fillna(0)
    
    # Fill missing head-to-head stats
    if 'MatchupCount' in df_features:
        df_features['MatchupCount'] = df_features['MatchupCount'].fillna(0)
    if 'Team1WinPctH2H' in df_features:
        df_features['Team1WinPctH2H'] = df_features['Team1WinPctH2H'].fillna(0.5)  # Neutral when no history
    
    # Create additional derived features
    
    # Offensive efficiency against defensive efficiency
    if all(col in df_features for col in ['Team1_PointsPerGame', 'Team2_PointsAllowedPerGame']):
        df_features['Team1OffVsTeam2Def'] = df_features['Team1_PointsPerGame'] - df_features['Team2_PointsAllowedPerGame']
    
    if all(col in df_features for col in ['Team2_PointsPerGame', 'Team1_PointsAllowedPerGame']):
        df_features['Team2OffVsTeam1Def'] = df_features['Team2_PointsPerGame'] - df_features['Team1_PointsAllowedPerGame']
    
    # Create seed strength indicators
    if 'Team1_SeedNumber' in df_features:
        df_features['Team1SeedStrength'] = 1 / df_features['Team1_SeedNumber']
    if 'Team2_SeedNumber' in df_features:
        df_features['Team2SeedStrength'] = 1 / df_features['Team2_SeedNumber']
    
    # Get target if available
    if 'Result' in df.columns:
        y = df['Result'].copy()
    else:
        y = None
    
    # Drop any remaining NaN values
    df_features = df_features.fillna(df_features.mean())
    
    # Return features and target
    return df_features, y, df_features.columns.tolist()


men_X_train, men_y_train, men_feature_names = prepare_features(men_matchups_train)
men_X_valid, men_y_valid, _ = prepare_features(men_matchups_valid)

# Prepare women's datasets
women_X_train, women_y_train, women_feature_names = prepare_features(women_matchups_train)
women_X_valid, women_y_valid, _ = prepare_features(women_matchups_valid)

# Remove rows with missing target values
if men_y_train is not None:
    valid_train_idx = ~men_y_train.isna()
    men_X_train = men_X_train[valid_train_idx]
    men_y_train = men_y_train[valid_train_idx]

if men_y_valid is not None:
    valid_valid_idx = ~men_y_valid.isna()
    men_X_valid = men_X_valid[valid_valid_idx]
    men_y_valid = men_y_valid[valid_valid_idx]

if women_y_train is not None:
    valid_train_idx = ~women_y_train.isna()
    women_X_train = women_X_train[valid_train_idx]
    women_y_train = women_y_train[valid_train_idx]

if women_y_valid is not None:
    valid_valid_idx = ~women_y_valid.isna()
    women_X_valid = women_X_valid[valid_valid_idx]
    women_y_valid = women_y_valid[valid_valid_idx]

print(f"Men's training set: {men_X_train.shape}, target: {men_y_train.shape if men_y_train is not None else None}")
print(f"Men's validation set: {men_X_valid.shape}, target: {men_y_valid.shape if men_y_valid is not None else None}")
print(f"Women's training set: {women_X_train.shape}, target: {women_y_train.shape if women_y_train is not None else None}")
print(f"Women's validation set: {women_X_valid.shape}, target: {women_y_valid.shape if women_y_valid is not None else None}")


def train_lightgbm_model(X_train, y_train, X_valid=None, y_valid=None, feature_names=None):
    """
    Train a LightGBM model for tournament predictions
    
    Parameters:
    -----------
    X_train: Training features
    y_train: Training targets
    X_valid: Validation features (optional)
    y_valid: Validation targets (optional)
    feature_names: List of feature names (optional)
    
    Returns:
    --------
    Trained model, validation predictions, feature importances
    """
    # Define LightGBM parameters - optimized for this task
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'random_state': SEED
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    
    if X_valid is not None and y_valid is not None:
        valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names)
        valid_sets = [train_data, valid_data]
        valid_names = ['train', 'valid']
    else:
        valid_sets = [train_data]
        valid_names = ['train']
    
    # Train model with callbacks instead of early_stopping_rounds
    print("Training LightGBM model...")
    
    # Use callbacks for early stopping instead
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=valid_sets,
        valid_names=valid_names,
        num_boost_round=1000,
        callbacks=callbacks
    )
    
    # Get predictions on validation set
    if X_valid is not None:
        valid_preds = model.predict(X_valid)
    else:
        valid_preds = None
    
    # Get feature importances
    importances = model.feature_importance(importance_type='gain')
    feature_names = feature_names or [f'f{i}' for i in range(len(importances))]
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return model, valid_preds, feature_imp

# Train men's model
men_model, men_valid_preds, men_feature_imp = train_lightgbm_model(
    men_X_train, men_y_train, 
    men_X_valid, men_y_valid, 
    men_feature_names
)

# Train women's model
women_model, women_valid_preds, women_feature_imp = train_lightgbm_model(
    women_X_train, women_y_train, 
    women_X_valid, women_y_valid, 
    women_feature_names
)


def evaluate_model(y_true, y_pred, title='Model Evaluation'):
    """
    Evaluate model performance with multiple metrics
    
    Parameters:
    -----------
    y_true: True labels
    y_pred: Predicted probabilities
    title: Plot title
    
    Returns:
    --------
    Dict of metrics
    """
    # Calculate metrics
    brier = brier_score_loss(y_true, y_pred)
    log_loss_val = log_loss(y_true, y_pred)
    
    # Print metrics
    print(f"{title}:")
    print(f"Brier Score: {brier:.4f}")
    print(f"Log Loss: {log_loss_val:.4f}")
    
    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    
    # Create bins for predictions
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.minimum(bin_indices, n_bins - 1)  # Cap at max bin
    
    # Calculate actual outcomes in each bin
    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_actual = np.zeros(n_bins)
    non_zero_bins = bin_counts > 0
    bin_actual[non_zero_bins] = bin_sums[non_zero_bins] / bin_counts[non_zero_bins]
    
    # Plot calibration curve
    plt.plot(bin_centers, bin_actual, 'o-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual frequency')
    plt.title(f'Calibration Curve - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {'brier': brier, 'log_loss': log_loss_val}

# Evaluate men's model
men_metrics = evaluate_model(men_y_valid, men_valid_preds, 'Men\'s Model')

# Evaluate women's model
women_metrics = evaluate_model(women_y_valid, women_valid_preds, 'Women\'s Model')


def plot_feature_importance(feature_imp, title='Feature Importance'):
    """
    Plot feature importances
    
    Parameters:
    -----------
    feature_imp: DataFrame with feature importances
    title: Plot title
    """
    plt.figure(figsize=(12, 8))
    top_n = 20  # Show top 20 features
    top_features = feature_imp.head(top_n)
    
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} {title}', fontsize=16)
    plt.tight_layout()
    plt.show()

# Plot men's feature importances
plot_feature_importance(men_feature_imp, 'Men\'s Model Feature Importance')

# Plot women's feature importances
plot_feature_importance(women_feature_imp, 'Women\'s Model Feature Importance')


# SHAP analysis for model interpretability
def shap_analysis(model, X, feature_names):
    """
    Perform SHAP analysis on a trained model
    
    Parameters:
    -----------
    model: Trained model
    X: Feature matrix for analysis
    feature_names: List of feature names
    """
    # Sample data for SHAP analysis (for speed)
    if X.shape[0] > 500:
        X_sample = X.sample(500, random_state=SEED)
    else:
        X_sample = X
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Dependence plots for top features
    # For binary classification, we need to use shap_values[1] (the positive class)
    if isinstance(shap_values, list):
        shap_values_for_plot = shap_values[1]  # Use positive class for binary classification
    else:
        shap_values_for_plot = shap_values
    
    # Use the correct model's feature importance for the current model being analyzed
    if 'men_model' in locals() and model is men_model:
        top_features = men_feature_imp.head(3)['Feature'].values
    elif 'women_model' in locals() and model is women_model:
        top_features = women_feature_imp.head(3)['Feature'].values
    else:
        # If we can't identify which model, use a generic approach
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importance(importance_type='gain')
        }).sort_values(by='Importance', ascending=False)
        top_features = feature_imp.head(3)['Feature'].values
    
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        feature_idx = feature_names.index(feature)
        shap.dependence_plot(
            feature_idx, 
            shap_values_for_plot,  # Use the appropriate shap values
            X_sample, 
            feature_names=feature_names, 
            show=False
        )
        plt.title(f'SHAP Dependence Plot for {feature}', fontsize=16)
        plt.tight_layout()
        plt.show()

# Run SHAP analysis for men's model
print("SHAP Analysis for Men's Model:")
shap_analysis(men_model, men_X_valid, men_feature_names)

# Run SHAP analysis for women's model
print("SHAP Analysis for Women's Model:")
shap_analysis(women_model, women_X_valid, women_feature_names)


current_season = 2025

# Create datasets for 2025 matchups (without known results)
men_matchups_2025 = create_matchup_dataset([current_season], is_mens=True, include_tourney_results=False)
women_matchups_2025 = create_matchup_dataset([current_season], is_mens=False, include_tourney_results=False)

print(f"Men's 2025 possible matchups: {men_matchups_2025.shape}")
print(f"Women's 2025 possible matchups: {women_matchups_2025.shape}")

# Prepare features for prediction
men_X_2025, _, _ = prepare_features(men_matchups_2025)
women_X_2025, _, _ = prepare_features(women_matchups_2025)

# Generate predictions
men_preds_2025 = men_model.predict(men_X_2025)
women_preds_2025 = women_model.predict(women_X_2025)

# Add predictions to dataframes
men_matchups_2025['Pred'] = men_preds_2025
women_matchups_2025['Pred'] = women_preds_2025

# Create submission dataframe
men_submission = men_matchups_2025[['ID', 'Pred']].copy()
women_submission = women_matchups_2025[['ID', 'Pred']].copy()

# Combine into a single submission file
submission = pd.concat([men_submission, women_submission], ignore_index=True)

# Export submission file
submission.to_csv('submission.csv', index=False)

print(f"Generated {len(submission):,} predictions for the 2025 tournament")
print("Sample of submission file:")
print(submission.head())


print("## Model Performance Summary ##")
print(f"Men's Model - Brier Score: {men_metrics['brier']:.4f}, Log Loss: {men_metrics['log_loss']:.4f}")
print(f"Women's Model - Brier Score: {women_metrics['brier']:.4f}, Log Loss: {women_metrics['log_loss']:.4f}")

print("\n## Top Features for Men's Model ##")
print(men_feature_imp.head(10).to_string(index=False))

print("\n## Top Features for Women's Model ##")
print(women_feature_imp.head(10).to_string(index=False))

# Plot predicted upset probabilities by seed differential
plt.figure(figsize=(12, 6))

# Get seed differential and prediction data for men's matchups
seed_pred_data = men_matchups_2025[men_matchups_2025['SeedDiff'].notna()].copy()
seed_pred_data['SeedDiffBin'] = seed_pred_data['SeedDiff'].round().astype(int)

# Calculate mean prediction by seed differential
seed_pred_summary = seed_pred_data.groupby('SeedDiffBin')['Pred'].mean().reset_index()

# Plot
sns.lineplot(x='SeedDiffBin', y='Pred', data=seed_pred_summary, marker='o')
plt.axhline(y=0.5, color='r', linestyle='--', label='50/50 odds')
plt.title('Predicted Win Probability by Seed Differential (Men\'s Tournament)', fontsize=16)
plt.xlabel('Seed Differential (Team1 - Team2)', fontsize=14)
plt.ylabel('Predicted Probability of Team1 Winning', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


current_season = 2025

# Get teams in the current tournament
m_current_seeds = m_tourney_seeds[m_tourney_seeds['Season'] == current_season]
w_current_seeds = w_tourney_seeds[w_tourney_seeds['Season'] == current_season]

# Create matchups for men's tournament
m_teams_2025 = m_current_seeds['TeamID'].unique()
m_matchups_2025 = []
for team1 in m_teams_2025:
    for team2 in m_teams_2025:
        if team1 < team2:  # Avoid duplicates
            m_matchups_2025.append((current_season, team1, team2))

# Create matchups for women's tournament
w_teams_2025 = w_current_seeds['TeamID'].unique()
w_matchups_2025 = []
for team1 in w_teams_2025:
    for team2 in w_teams_2025:
        if team1 < team2:
            w_matchups_2025.append((current_season, team1, team2))

# Create dataframes
m_submit_df = pd.DataFrame(m_matchups_2025, columns=['Season', 'Team1ID', 'Team2ID'])
w_submit_df = pd.DataFrame(w_matchups_2025, columns=['Season', 'Team1ID', 'Team2ID'])

# Create full matchup datasets with team stats and seeds
print("Generating features for men's submission matchups...")
m_submit_matchups = create_matchup_dataset([current_season], is_mens=True, include_tourney_results=False)
m_submit_matchups = m_submit_matchups[m_submit_matchups['Season'] == current_season]

print("Generating features for women's submission matchups...")
w_submit_matchups = create_matchup_dataset([current_season], is_mens=False, include_tourney_results=False)
w_submit_matchups = w_submit_matchups[w_submit_matchups['Season'] == current_season]

# Generate features
m_X_submit, _, _ = prepare_features(m_submit_matchups)
w_X_submit, _, _ = prepare_features(w_submit_matchups)

# Verify data shape
print(f"Men's submission features shape: {m_X_submit.shape}")
print(f"Women's submission features shape: {w_X_submit.shape}")
print(f"Men's model feature names: {men_feature_names}")

# Make sure features match
m_X_submit = m_X_submit[men_feature_names]
w_X_submit = w_X_submit[women_feature_names]

# Generate predictions
m_submit_preds = men_model.predict(m_X_submit)
w_submit_preds = women_model.predict(w_X_submit)

# Create submission IDs
m_submit_ids = [f"M{m_submit_matchups.iloc[i]['Season']}_{m_submit_matchups.iloc[i]['Team1ID']}_{m_submit_matchups.iloc[i]['Team2ID']}" 
               for i in range(len(m_submit_matchups))]

w_submit_ids = [f"W{w_submit_matchups.iloc[i]['Season']}_{w_submit_matchups.iloc[i]['Team1ID']}_{w_submit_matchups.iloc[i]['Team2ID']}" 
               for i in range(len(w_submit_matchups))]

# Combine submissions
all_ids = m_submit_ids + w_submit_ids
all_preds = list(m_submit_preds) + list(w_submit_preds)

# Create submission dataframe
submission = pd.DataFrame({
    'ID': all_ids,
    'Pred': all_preds
})

# Ensure predictions are within valid range
submission['Pred'] = submission['Pred'].clip(0, 1)

# Save to CSV file
submission.to_csv('submission.csv', index=False)
print(f"Saved submission file with {len(submission)} predictions")
print(submission.head())

