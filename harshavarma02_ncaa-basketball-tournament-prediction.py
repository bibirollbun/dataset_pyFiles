# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import joblib


# Define file paths
data_dir = '/kaggle/input/march-machine-learning-mania-2025'
files = {
    'cities': 'Cities.csv',
    'conferences': 'Conferences.csv',
    'm_conf_tourney': 'MConferenceTourneyGames.csv',
    'm_game_cities': 'MGameCities.csv',
    'm_massey': 'MMasseyOrdinals.csv',
    'm_tourney_compact': 'MNCAATourneyCompactResults.csv',
    'm_tourney_seeds': 'MNCAATourneySeeds.csv',
    'm_teams': 'MTeams.csv',
    'w_conf_tourney': 'WConferenceTourneyGames.csv',
    'w_game_cities': 'WGameCities.csv',
    'w_tourney_compact': 'WNCAATourneyCompactResults.csv',
    'w_tourney_seeds': 'WNCAATourneySeeds.csv',
    'w_teams': 'WTeams.csv',
    'm_tourney_detailed': 'MNCAATourneyDetailedResults.csv',
    'w_tourney_detailed': 'WNCAATourneyDetailedResults.csv',
    'm_regular_detailed': 'MRegularSeasonDetailedResults.csv',
    'w_regular_detailed': 'WRegularSeasonDetailedResults.csv',
    'm_team_conferences': 'MTeamConferences.csv',
    'w_team_conferences': 'WTeamConferences.csv'
}

# Load all data into a dictionary
data = {}
for name, file in files.items():
    try:
        # Try loading with UTF-8 encoding first
        data[name] = pd.read_csv(os.path.join(data_dir, file), encoding='utf-8')
        print(f"Loaded {file} successfully with UTF-8 encoding.")
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try ISO-8859-1 encoding
            data[name] = pd.read_csv(os.path.join(data_dir, file), encoding='ISO-8859-1')
            print(f"Loaded {file} successfully with ISO-8859-1 encoding.")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            raise
    except Exception as e:
        print(f"Error loading {file}: {str(e)}")
        raise

# Validate data
required_columns = {
    'm_tourney_compact': ['Season', 'DayNum', 'WTeamID', 'LTeamID'],
    'm_tourney_seeds': ['Season', 'TeamID', 'Seed'],
    'm_teams': ['TeamID', 'TeamName'],
    'm_team_conferences': ['Season', 'TeamID', 'ConfAbbrev'],
    'w_team_conferences': ['Season', 'TeamID', 'ConfAbbrev']
}

for df_name, cols in required_columns.items():
    if df_name in data:
        missing_cols = set(cols) - set(data[df_name].columns)
        if missing_cols:
            raise ValueError(f"Missing columns in {df_name}: {missing_cols}")

# Print head of each CSV for verification
for name, df in data.items():
    print(f"Head of {name}.csv:")
    display(df.head())
    print("=" * 50)

# Process team data
m_teams = data['m_teams'][data['m_teams']['LastD1Season'] >= 2025].copy()
m_teams['Gender'] = 'M'
w_teams = data['w_teams'].copy()
w_teams['Gender'] = 'W'

print("\nğŸ�€ Team Statistics:")
print("-" * 50)
print(f"Total Active Teams: {len(m_teams) + len(w_teams):,}")
print(f"Men's Teams: {len(m_teams):,}")
print(f"Women's Teams: {len(w_teams):,}")


def calculate_possession_stats(df):
    """Calculate advanced metrics with proper team perspective"""
    # Create team-centric view by melting winners and losers
    winners = df.rename(columns={
        'WTeamID': 'TeamID',
        'LTeamID': 'OpponentID',
        'WScore': 'Score',
        'LScore': 'Opponent_Score',
        'WFGM': 'FGM',
        'WFGA': 'FGA',
        'WFGM3': 'FGM3',
        'WFTA': 'FTA',
        'WFTM': 'FTM',
        'WOR': 'OR',
        'WDR': 'DR',
        'WTO': 'TO',
        'WStl': 'Stl',
        'WBlk': 'Blk',
        'WPF': 'PF'
    }).assign(win=1)

    losers = df.rename(columns={
        'LTeamID': 'TeamID',
        'WTeamID': 'OpponentID',
        'LScore': 'Score',
        'WScore': 'Opponent_Score',
        'LFGM': 'FGM',
        'LFGA': 'FGA',
        'LFGM3': 'FGM3',
        'LFTA': 'FTA',
        'LFTM': 'FTM',
        'LOR': 'OR',
        'LDR': 'DR',
        'LTO': 'TO',
        'LStl': 'Stl',
        'LBlk': 'Blk',
        'LPF': 'PF'
    }).assign(win=0)

    # Combine and calculate metrics
    team_games = pd.concat([winners, losers], ignore_index=True)
    
    # Calculate possessions and efficiency metrics
    team_games['Possessions'] = 0.96 * (
        team_games['FGA'] + 
        team_games['TO'] + 
        0.44 * team_games['FTA'] - 
        team_games['OR']
    ).replace(0, 1)
    
    team_games['Offensive_Efficiency'] = team_games['Score'] / team_games['Possessions'] * 100
    team_games['Defensive_Efficiency'] = team_games['Opponent_Score'] / team_games['Possessions'] * 100
    team_games['eFG%'] = (team_games['FGM'] + 0.5 * team_games['FGM3']) / team_games['FGA'].replace(0, 1)
    team_games['TO_Rate'] = team_games['TO'] / team_games['Possessions'] * 100
    
    return team_games

def feature_engineering(data):
    def process_gender(gender):
        features = []
        prefix = 'm' if gender == 'M' else 'w'
        
        # Load and prepare data
        regular = data[f'{prefix}_regular_detailed']
        tourney = data[f'{prefix}_tourney_detailed']
        conferences = data[f'{prefix}_team_conferences']
        
        # Combine and process games
        detailed_results = pd.concat([regular, tourney])
        team_games = calculate_possession_stats(detailed_results)
        team_games['Gender'] = gender
        
        # Process each season
        seasons = team_games['Season'].unique()
        # FIX: Remove {season} from description
        for season in tqdm(seasons, desc=f'Processing {gender} seasons'):
            # Filter season data
            season_data = team_games[team_games['Season'] == season]
            
            # Initialize Elo system
            elo = MarginAwareElo()
            
            # Calculate Elo ratings
            for _, row in season_data.iterrows():
                if row['win'] == 1:
                    elo.update_ratings(
                        row['TeamID'], 
                        row['OpponentID'], 
                        row['Score'] - row['Opponent_Score'], 
                        season
                    )
            
            # Get final Elo ratings
            elo_ratings = {
                team: rating 
                for (team, s), rating in elo.rating.items() 
                if s == season
            }
            
            # Calculate team statistics
            team_stats = season_data.groupby('TeamID').agg(
                Offensive_Efficiency_mean=('Offensive_Efficiency', 'mean'),
                Offensive_Efficiency_std=('Offensive_Efficiency', 'std'),
                Defensive_Efficiency_mean=('Defensive_Efficiency', 'mean'),
                Defensive_Efficiency_std=('Defensive_Efficiency', 'std'),
                eFG_mean=('eFG%', 'mean'),
                TO_Rate_mean=('TO_Rate', 'mean'),
                Score_mean=('Score', 'mean'),
                Score_max=('Score', 'max'),
                Opponent_Score_mean=('Opponent_Score', 'mean'),
                WinPct=('win', 'mean')
            ).reset_index()
            
            # Add Elo ratings
            team_stats['Elo'] = team_stats['TeamID'].map(elo_ratings)
            
            # Add conference strength
            conf_strength = conferences[conferences['Season'] == season]
            team_stats = team_stats.merge(
                conf_strength[['TeamID', 'ConfAbbrev']], 
                on='TeamID', 
                how='left'
            )
            
            # Add recent performance (last 5 games)
            last5 = season_data.groupby('TeamID').tail(5)
            last5_stats = last5.groupby('TeamID').agg(
                Last5_OffEff=('Offensive_Efficiency', 'mean'),
                Last5_DefEff=('Defensive_Efficiency', 'mean')
            ).reset_index()
            
            team_stats = team_stats.merge(last5_stats, on='TeamID', how='left')
            
            # Add season and gender
            team_stats['Season'] = season
            team_stats['Gender'] = gender
            
            features.append(team_stats)
        
        return pd.concat(features)
    
    # Process both genders
    men_features = process_gender('M')
    women_features = process_gender('W')
    
    # Combine and normalize
    full_features = pd.concat([men_features, women_features])
    
    # Season-specific normalization
    numeric_cols = [c for c in full_features.columns if c not in ['TeamID', 'Season', 'Gender', 'ConfAbbrev']]
    full_features[numeric_cols] = full_features.groupby('Season')[numeric_cols].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    
    return full_features


class MarginAwareElo:
    # Add the class definition from previous answer here
    def __init__(self, base_rating=1500, k=32, margin_factor=1/20):
        self.rating = {}
        self.base = base_rating
        self.k = k
        self.margin_factor = margin_factor
        
    def update_ratings(self, team1, team2, score_diff, season):
        r1 = self.rating.get((team1, season), self.base)
        r2 = self.rating.get((team2, season), self.base)
        
        expected = 1 / (1 + 10 ** ((r2 - r1) / 400))
        actual = 0.5 + 0.5 * np.tanh(score_diff * self.margin_factor)
        
        delta = self.k * (actual - expected)
        self.rating[(team1, season)] = r1 + delta
        self.rating[(team2, season)] = r2 - delta

# Execute feature engineering pipeline
team_features = feature_engineering(data)

# Save the features
team_features.to_csv('advanced_features.csv', index=False)

# Preview the features
print("Generated features shape:", team_features.shape)
display(team_features.head())


def process_seeds(seeds_df, conferences_df):
    """Convert seed strings to numerical values with conference data"""
    # Merge seeds with conference information
    seeds_df = seeds_df.merge(
        conferences_df[['Season', 'TeamID', 'ConfAbbrev']],
        on=['Season', 'TeamID'],
        how='left'
    )
    
    # Handle missing conferences
    seeds_df['ConfAbbrev'] = seeds_df['ConfAbbrev'].fillna('UNK')
    
    def seed_to_float(seed):
        cleaned = seed[1:] if seed[0].isalpha() else seed
        digits = ''.join(filter(str.isdigit, cleaned))
        return int(digits) if digits else 99
    
    seeds_df['SeedValue'] = seeds_df['Seed'].apply(seed_to_float)
    
    # Conference-based normalization
    conference_avg = seeds_df.groupby(['Season', 'ConfAbbrev'])['SeedValue'].mean().reset_index()
    conference_avg.columns = ['Season', 'ConfAbbrev', 'ConfSeedAvg']
    
    seeds_df = seeds_df.merge(conference_avg, on=['Season', 'ConfAbbrev'])
    seeds_df['AdjSeed'] = seeds_df['SeedValue'] - seeds_df['ConfSeedAvg']
    
    # Keep Gender column in output
    return seeds_df[['Season', 'TeamID', 'SeedValue', 'AdjSeed', 'Gender']]

def get_massey_rankings(massey_df, tournament_days):
    """Calculate consensus Massey rankings before tournament start"""
    # Get latest rankings before tournament
    massey_features = []
    for (season, gender), day in tournament_days.items():
        prefix = 'M' if gender == 'M' else 'W'
        current_massey = massey_df[massey_df['Season'] == int(season)]
        
        # Filter to latest ranking before tournament
        current_massey = current_massey[current_massey['RankingDayNum'] <= day]
        latest_massey = current_massey.sort_values(['Season', 'TeamID', 'RankingDayNum']) \
                                    .groupby(['Season', 'TeamID']).last().reset_index()
        
        # Calculate consensus ranking across systems
        consensus = latest_massey.groupby(['Season', 'TeamID'])['OrdinalRank'] \
                               .agg(['mean', 'std', 'min', 'max']) \
                               .reset_index()
        consensus.columns = ['Season', 'TeamID', 
                           f'{prefix}_MasseyMean', f'{prefix}_MasseyStd',
                           f'{prefix}_MasseyMin', f'{prefix}_MasseyMax']
        
        massey_features.append(consensus)
    
    return pd.concat(massey_features)


def create_training_data(data, features_df):
    """Create final training dataset with unique competitive features"""
    # 1. Process tournament games
    tournament_games = pd.concat([
        data['m_tourney_compact'].assign(Gender='M'),
        data['w_tourney_compact'].assign(Gender='W')
    ])
    
    # Create Team1/Team2 pairs
    tournament_pairs = []
    for _, row in tqdm(tournament_games.iterrows(), total=len(tournament_games)):
        team1 = min(row['WTeamID'], row['LTeamID'])
        team2 = max(row['WTeamID'], row['LTeamID'])
        label = 1 if team1 == row['WTeamID'] else 0
        
        tournament_pairs.append({
            'Season': row['Season'],
            'Gender': row['Gender'],
            'Team1': team1,
            'Team2': team2,
            'Label': label,
            'DayNum': row['DayNum']
        })
    
    pair_df = pd.DataFrame(tournament_pairs)
    
    # 2. Add seed features
    seeds = pd.concat([
        process_seeds(
            data['m_tourney_seeds'].assign(Gender='M'),
            data['m_team_conferences']
        ),
        process_seeds(
            data['w_tourney_seeds'].assign(Gender='W'),
            data['w_team_conferences']
        )
    ])
    
    pair_df = pair_df.merge(
        seeds.rename(columns={
            'TeamID': 'Team1',
            'SeedValue': 'Team1_Seed',
            'AdjSeed': 'Team1_AdjSeed'
        }),
        on=['Season', 'Gender', 'Team1']
    ).merge(
        seeds.rename(columns={
            'TeamID': 'Team2',
            'SeedValue': 'Team2_Seed',
            'AdjSeed': 'Team2_AdjSeed'
        }),
        on=['Season', 'Gender', 'Team2']
    )
    
    # 3. Add Massey rankings
    tournament_days = pair_df.groupby(['Season', 'Gender'])['DayNum'].min().to_dict()
    massey_features = get_massey_rankings(data['m_massey'], tournament_days)
    
    pair_df = pair_df.merge(
        massey_features.rename(columns={
            'TeamID': 'Team1',
            'M_MasseyMean': 'Team1_MasseyMean',
            'M_MasseyStd': 'Team1_MasseyStd',
            'M_MasseyMin': 'Team1_MasseyMin',
            'M_MasseyMax': 'Team1_MasseyMax'
        }),
        on=['Season', 'Team1'],
        how='left'
    ).merge(
        massey_features.rename(columns={
            'TeamID': 'Team2',
            'M_MasseyMean': 'Team2_MasseyMean',
            'M_MasseyStd': 'Team2_MasseyStd',
            'M_MasseyMin': 'Team2_MasseyMin',
            'M_MasseyMax': 'Team2_MasseyMax'
        }),
        on=['Season', 'Team2'],
        how='left'
    )
    
    # 4. Merge team features (corrected version)
    # Separate identifier columns and feature columns
    identifiers = features_df[['Season', 'Gender', 'TeamID']]
    features = features_df.drop(columns=['Season', 'Gender', 'TeamID'])
    
    # Add prefix to feature columns
    team1_features = features.add_prefix('Team1_')
    team1_df = pd.concat([identifiers.rename(columns={'TeamID': 'Team1'}), team1_features], axis=1)
    
    team2_features = features.add_prefix('Team2_')
    team2_df = pd.concat([identifiers.rename(columns={'TeamID': 'Team2'}), team2_features], axis=1)
    
    # Merge features
    pair_df = pair_df.merge(
        team1_df,
        on=['Season', 'Gender', 'Team1']
    ).merge(
        team2_df,
        on=['Season', 'Gender', 'Team2']
    )
    
    # 5. Create difference features
    pair_df['SeedDiff'] = pair_df['Team1_Seed'] - pair_df['Team2_Seed']
    pair_df['EloDiff'] = pair_df['Team1_Elo'] - pair_df['Team2_Elo']
    pair_df['MasseyDiff'] = pair_df['Team1_MasseyMean'] - pair_df['Team2_MasseyMean']
    
    # 6. Add historical matchup win ratio
    historical = pd.concat([
        data['m_regular_detailed'].assign(Gender='M'),
        data['w_regular_detailed'].assign(Gender='W')
    ])
    
    def get_head_to_head(row):
        mask = (
            (historical['Season'] < row['Season']) &
            ((historical['WTeamID'] == row['Team1']) & (historical['LTeamID'] == row['Team2'])) |
            ((historical['WTeamID'] == row['Team2']) & (historical['LTeamID'] == row['Team1']))
        )
        matchups = historical[mask]
        total = len(matchups)
        wins = sum((matchups['WTeamID'] == row['Team1']) & (matchups['LTeamID'] == row['Team2']))
        return wins / total if total > 0 else 0.5
    
    pair_df['HistoricalWinRate'] = pair_df.apply(get_head_to_head, axis=1)
    
    # 7. Final feature selection
    features = [
        'Season', 'Gender', 'Label',
        'SeedDiff', 'EloDiff', 'MasseyDiff', 'HistoricalWinRate',
        'Team1_Offensive_Efficiency_mean', 'Team2_Offensive_Efficiency_mean',
        'Team1_Defensive_Efficiency_mean', 'Team2_Defensive_Efficiency_mean',
        'Team1_Last5_OffEff', 'Team2_Last5_OffEff',
        'Team1_Last5_DefEff', 'Team2_Last5_DefEff',
        'Team1_WinPct', 'Team2_WinPct'
    ]
    
    return pair_df[features].dropna()

# Execute pipeline
team_features = pd.read_csv('advanced_features.csv')
final_train_df = create_training_data(data, team_features)
final_train_df.to_csv('training_data.csv', index=False)
print("Final training data shape:", final_train_df.shape)
display(final_train_df.head())


# Load the training data
train_df = pd.read_csv('./training_data.csv')

# Feature Selection
features = [
    'SeedDiff', 'EloDiff', 'MasseyDiff', 'HistoricalWinRate',
    'Team1_Offensive_Efficiency_mean', 'Team2_Offensive_Efficiency_mean',
    'Team1_Defensive_Efficiency_mean', 'Team2_Defensive_Efficiency_mean',
    'Team1_Last5_OffEff', 'Team2_Last5_OffEff',
    'Team1_Last5_DefEff', 'Team2_Last5_DefEff',
    'Team1_WinPct', 'Team2_WinPct'
]

# Target variable
target = 'Label'

# Temporal Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Hyperparameter Space for XGBoost
space = {
    'max_depth': hp.choice('max_depth', range(3, 10)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'min_child_weight': hp.choice('min_child_weight', range(1, 6)),
    'n_estimators': hp.choice('n_estimators', range(100, 500))
}

# Objective Function for Hyperparameter Optimization
def objective(params):
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=10,
        **params
    )
    
    # Temporal Cross-Validation
    log_loss_scores = []
    for train_idx, val_idx in tscv.split(train_df):
        X_train, X_val = train_df.iloc[train_idx][features], train_df.iloc[val_idx][features]
        y_train, y_val = train_df.iloc[train_idx][target], train_df.iloc[val_idx][target]
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict_proba(X_val)[:, 1]
        log_loss_scores.append(log_loss(y_val, y_pred))
    
    # Return average log loss
    return {'loss': np.mean(log_loss_scores), 'status': STATUS_OK}

# Bayesian Optimization
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
print("Best parameters found:", best_params)

# Train Final Model with Best Parameters
final_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=10,  # Moved here
    **best_params
)

# Fit on Full Training Data with a Validation Set
# Here we can use a simple split for validation
train_idx = int(len(train_df) * 0.8)  # 80% for training
X_train, X_val = train_df[features][:train_idx], train_df[features][train_idx:]
y_train, y_val = train_df[target][:train_idx], train_df[target][train_idx:]

final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Feature Importance Analysis
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Save the Model
joblib.dump(final_model, './xgboost_model.pkl')
print("Model saved as 'xgboost_model.pkl'.")

# Predict on Validation Data (Example)
val_predictions = final_model.predict_proba(X_val)[:, 1]
print("Validation Log Loss:", log_loss(y_val, val_predictions))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Predict on Validation Data
val_predictions = final_model.predict(X_val)  # Class predictions (0 or 1)

# Calculate Metrics
accuracy = accuracy_score(y_val, val_predictions)
precision = precision_score(y_val, val_predictions)
recall = recall_score(y_val, val_predictions)
f1 = f1_score(y_val, val_predictions)
conf_matrix = confusion_matrix(y_val, val_predictions)

# Print Metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:\n", conf_matrix)


# Load model
model = joblib.load('./xgboost_model.pkl')

# Extract feature importance
feature_importance = pd.DataFrame({
    'Feature': model.feature_names_in_,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


import scipy
# 1. Load the advanced features for 2025 (precomputed from your regular season data)
features_df = pd.read_csv('./advanced_features.csv')
features_2025 = features_df[features_df['Season'] == 2025].copy()

# If your features file does not include tournament-specific fields (e.g., seeds, Elo, Massey)
# you can fill these with a default value (or compute them separately when data becomes available)
extra_cols = ['Seed', 'AdjSeed', 'Elo', 'MasseyMean']
for col in extra_cols:
    if col not in features_2025.columns:
        features_2025[col] = 0

# ---------------------------
# 2. Generate All Possible Pairs
# ---------------------------
pairs = []
for gender in ['M', 'W']:
    # Filter teams by gender
    teams = features_2025[features_2025['Gender'] == gender]
    # Get unique sorted team IDs (assumes TeamID is numeric; adjust sorting if necessary)
    team_ids = sorted(teams['TeamID'].unique())
    # Create all possible pairs such that Team1 < Team2
    for i in range(len(team_ids)):
        for j in range(i+1, len(team_ids)):
            pairs.append({
                'Season': 2025,
                'Gender': gender,
                'Team1': team_ids[i],
                'Team2': team_ids[j]
            })

pairs_df = pd.DataFrame(pairs)

# ---------------------------
# 3. Merge Team Features for Each Pair
# ---------------------------
# Prepare features for Team1
team1_features = features_2025.copy().add_prefix('Team1_')
team1_features.rename(columns={
    'Team1_Season': 'Season',
    'Team1_Gender': 'Gender',
    'Team1_TeamID': 'Team1'
}, inplace=True)

# Prepare features for Team2
team2_features = features_2025.copy().add_prefix('Team2_')
team2_features.rename(columns={
    'Team2_Season': 'Season',
    'Team2_Gender': 'Gender',
    'Team2_TeamID': 'Team2'
}, inplace=True)

# Merge features onto pairs
pairs_df = pairs_df.merge(team1_features, on=['Season', 'Gender', 'Team1'], how='left') \
                   .merge(team2_features, on=['Season', 'Gender', 'Team2'], how='left')

# ---------------------------
# 4. Compute Difference Features to Match Training Data
# ---------------------------
# For seed features, Elo, and Massey rankings, fill with default values if needed.
for col in ['Team1_Seed', 'Team2_Seed', 'Team1_Elo', 'Team2_Elo', 
            'Team1_MasseyMean', 'Team2_MasseyMean']:
    if col not in pairs_df.columns:
        pairs_df[col] = 0
    else:
        pairs_df[col] = pairs_df[col].fillna(0)

# Create difference features
pairs_df['SeedDiff'] = pairs_df['Team1_Seed'] - pairs_df['Team2_Seed']
pairs_df['EloDiff'] = pairs_df['Team1_Elo'] - pairs_df['Team2_Elo']
pairs_df['MasseyDiff'] = pairs_df['Team1_MasseyMean'] - pairs_df['Team2_MasseyMean']

# For historical win rate and conference power differences, use default values (if not computed)
pairs_df['HistoricalWinRate'] = 0.5
# Note: 'ConfPowerDiff' is omitted because it was not used in training

# ---------------------------
# 5. Prepare Final Feature Set
# ---------------------------
# List the prediction features exactly in the order used during training.
pred_features = [
    'SeedDiff', 'EloDiff', 'MasseyDiff', 'HistoricalWinRate',
    'Team1_Offensive_Efficiency_mean', 'Team2_Offensive_Efficiency_mean',
    'Team1_Defensive_Efficiency_mean', 'Team2_Defensive_Efficiency_mean',
    'Team1_Last5_OffEff', 'Team2_Last5_OffEff',
    'Team1_Last5_DefEff', 'Team2_Last5_DefEff',
    'Team1_WinPct', 'Team2_WinPct'
]

# Ensure all required columns are present.
X_pred = pairs_df[pred_features]

# ---------------------------
# 6. Load the Trained Model and Predict
# ---------------------------
final_model = joblib.load('./xgboost_model.pkl')
# Predict probability that Team1 wins
pairs_df['Pred'] = final_model.predict_proba(X_pred)[:, 1]

# ---------------------------
# 7. Adjust Extreme Probabilities (Logistic Transformation)
# ---------------------------
def adjust_probabilities(preds, factor=0.2):
    return scipy.special.expit(factor * (preds - 0.5))  # Keeps probabilities closer to 0.5

pairs_df['Pred'] = adjust_probabilities(pairs_df['Pred'])

# ---------------------------
# 8. Create Submission File
# ---------------------------
# Format the submission ID as "2025_Team1_Team2"
pairs_df['ID'] = pairs_df.apply(lambda row: f"{row['Season']}_{row['Team1']}_{row['Team2']}", axis=1)
submission = pairs_df[['ID', 'Pred']]

# Save submission file
submission.to_csv('./submission_2025.csv', index=False)
print("Submission file saved as 'submission_2025.csv'.")


import pandas as pd
import joblib
import scipy

# 1. Load the advanced features for 2025 (precomputed from your regular season data)
features_df = pd.read_csv('./advanced_features.csv')
features_2025 = features_df[features_df['Season'] == 2025].copy()

# If your features file does not include tournament-specific fields (e.g., seeds, Elo, Massey)
# you can fill these with a default value (or compute them separately when data becomes available)
extra_cols = ['Seed', 'AdjSeed', 'Elo', 'MasseyMean']
for col in extra_cols:
    if col not in features_2025.columns:
        features_2025[col] = 0

# ---------------------------
# 2. Generate All Possible Pairs
# ---------------------------
pairs = []
for gender in ['M', 'W']:
    # Filter teams by gender
    teams = features_2025[features_2025['Gender'] == gender]
    # Get unique sorted team IDs (assumes TeamID is numeric; adjust sorting if necessary)
    team_ids = sorted(teams['TeamID'].unique())
    # Create all possible pairs such that Team1 < Team2
    for i in range(len(team_ids)):
        for j in range(i + 1, len(team_ids)):
            pairs.append({
                'Season': 2025,
                'Gender': gender,
                'Team1': team_ids[i],
                'Team2': team_ids[j]
            })

pairs_df = pd.DataFrame(pairs)

# ---------------------------
# 3. Merge Team Features for Each Pair
# ---------------------------
# Prepare features for Team1
team1_features = features_2025.copy().add_prefix('Team1_')
team1_features.rename(columns={
    'Team1_Season': 'Season',
    'Team1_Gender': 'Gender',
    'Team1_TeamID': 'Team1'
}, inplace=True)

# Prepare features for Team2
team2_features = features_2025.copy().add_prefix('Team2_')
team2_features.rename(columns={
    'Team2_Season': 'Season',
    'Team2_Gender': 'Gender',
    'Team2_TeamID': 'Team2'
}, inplace=True)

# Merge features onto pairs
pairs_df = pairs_df.merge(team1_features, on=['Season', 'Gender', 'Team1'], how='left') \
                   .merge(team2_features, on=['Season', 'Gender', 'Team2'], how='left')

# ---------------------------
# 4. Compute Difference Features to Match Training Data
# ---------------------------
# For seed features, Elo, and Massey rankings, fill with default values if needed.
for col in ['Team1_Seed', 'Team2_Seed', 'Team1_Elo', 'Team2_Elo', 
            'Team1_MasseyMean', 'Team2_MasseyMean']:
    if col not in pairs_df.columns:
        pairs_df[col] = 0
    else:
        pairs_df[col] = pairs_df[col].fillna(0)

# Create difference features
pairs_df['SeedDiff'] = pairs_df['Team1_Seed'] - pairs_df['Team2_Seed']
pairs_df['EloDiff'] = pairs_df['Team1_Elo'] - pairs_df['Team2_Elo']
pairs_df['MasseyDiff'] = pairs_df['Team1_MasseyMean'] - pairs_df['Team2_MasseyMean']

# For historical win rate and conference power differences, use default values (if not computed)
pairs_df['HistoricalWinRate'] = 0.5
# Note: 'ConfPowerDiff' is omitted because it was not used in training

# ---------------------------
# 5. Prepare Final Feature Set
# ---------------------------
# List the prediction features exactly in the order used during training.
pred_features = [
    'SeedDiff', 'EloDiff', 'MasseyDiff', 'HistoricalWinRate',
    'Team1_Offensive_Efficiency_mean', 'Team2_Offensive_Efficiency_mean',
    'Team1_Defensive_Efficiency_mean', 'Team2_Defensive_Efficiency_mean',
    'Team1_Last5_OffEff', 'Team2_Last5_OffEff',
    'Team1_Last5_DefEff', 'Team2_Last5_DefEff',
    'Team1_WinPct', 'Team2_WinPct'
]

# Ensure all required columns are present.
X_pred = pairs_df[pred_features]

# ---------------------------
# 6. Load the Trained Model and Predict
# ---------------------------
final_model = joblib.load('./xgboost_model.pkl')
# Predict probability that Team1 wins
pairs_df['Pred'] = final_model.predict_proba(X_pred)[:, 1]

# ---------------------------
# 7. Adjust Probabilities to a Reasonable Range
# ---------------------------
def scale_and_clip_probabilities(preds, lower_bound=0.35, upper_bound=0.75):
    """
    Scale the predicted probabilities to fit within the specified bounds.
    """
    # Scale the probabilities to the range [0, 1]
    scaled_preds = (preds - preds.min()) / (preds.max() - preds.min())
    
    # Scale to the desired range [lower_bound, upper_bound]
    adjusted_preds = lower_bound + (upper_bound - lower_bound) * scaled_preds
    
    # Clip the probabilities to ensure they are within the bounds
    adjusted_preds = np.clip(adjusted_preds, lower_bound, upper_bound)
    
    return adjusted_preds

# Adjust the probabilities to the desired range
pairs_df['Pred'] = scale_and_clip_probabilities(pairs_df['Pred'])

# ---------------------------
# 8. Create Submission File
# ---------------------------
# Format the submission ID as "2025_Team1_Team2"
pairs_df['ID'] = pairs_df.apply(lambda row: f"{row['Season']}_{row['Team1']}_{row['Team2']}", axis=1)
submission = pairs_df[['ID', 'Pred']]

# Save submission file
submission.to_csv('./submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")


#overfiting evaluation
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_csv('./submission.csv')

# Plot histogram of prediction probabilities
plt.figure(figsize=(8, 5))
plt.hist(df['Pred'], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.show()

