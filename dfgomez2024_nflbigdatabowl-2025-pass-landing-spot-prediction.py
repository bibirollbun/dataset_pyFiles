import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

warnings.filterwarnings('ignore')

# Load CSV files into DataFrames
def load_data(base_files, tracking_weeks):
    dfs = {}
    for file in base_files:
        dfs[f'{file}_df'] = pd.read_csv(f'/kaggle/input/nfl-big-data-bowl-2025/{file}.csv')
    
    for week in tracking_weeks:
        df_name = f'tracking_week_{week}_df'
        dfs[df_name] = pd.read_csv(f'/kaggle/input/nfl-big-data-bowl-2025/tracking_week_{week}.csv')
        # Process the last record as before
        dfs[f'{df_name}_first_last_register'] = (
            dfs[df_name]
            .sort_values(by=["gameId", "playId", "nflId", "frameType", "time"])
            .groupby(["gameId", "playId", "nflId", "frameType"])
            .tail(1)
            .reset_index()
        )
    return dfs

# Merge plays and games DataFrames
def merge_plays_and_games(dfs):
    plays_df = pd.merge(
        dfs["plays_df"],
        dfs["games_df"][["gameId", "homeTeamAbbr", "visitorTeamAbbr", "week"]],
        on="gameId",
        how="left"
    )
    return plays_df

# Calculate yardline and target coordinates
def calculate_yardline_and_targets(plays_df):
    plays_df['is_home_offense'] = plays_df['possessionTeam'] == plays_df['homeTeamAbbr']
    plays_df['yardlineSide'] = plays_df['yardlineSide'] == plays_df['possessionTeam']

    plays_df['yardlineNumber'] = np.where(
        plays_df['yardlineSide'],
        plays_df['yardlineNumber'],
        100 - plays_df['yardlineNumber']
    )

    plays_df['yardlineNumber_v2'] = plays_df['yardlineNumber']

    plays_df['targetY'] = np.where(
        plays_df['yardlineSide'],
        plays_df['targetY'],
        53.3 - plays_df['targetY']
    )

    plays_df['targetX'] = np.where(
        (plays_df['yardlineNumber'] == plays_df['absoluteYardlineNumber'] - 10),
        plays_df['targetX'],
        120 - plays_df['targetX']
    )
    return plays_df

# Calculate score difference and offensive advantage
def calculate_score_difference(plays_df):
    plays_df['score_difference_offense'] = np.where(
        plays_df['is_home_offense'],
        plays_df['preSnapHomeScore'] - plays_df['preSnapVisitorScore'],  # Home offense
        plays_df['preSnapVisitorScore'] - plays_df['preSnapHomeScore']   # Visitor offense
    )

    plays_df['offense_advantage'] = np.select(
        [
            plays_df['score_difference_offense'] > 8,
            plays_df['score_difference_offense'] < -8
        ],
        [
            'advantage',
            'disadvantage'
        ],
        default='neutral'
    )
    return plays_df

# Get yards from player play DataFrame
def get_yards_from_player_play(dfs):
    player_play_df = dfs["player_play_df"]
    yards_df = player_play_df[
        (player_play_df['rushingYards'] > 0) | 
        (player_play_df['passingYards'] > 0)
    ][['gameId', 'playId', 'rushingYards', 'passingYards']]

    yards_df = yards_df.groupby(['gameId', 'playId']).agg({
        'passingYards': 'sum',
        'rushingYards': 'sum'
    }).reset_index()
    return yards_df

# Prepare plays DataFrame for historical analysis
def prepare_plays_history(plays_df, yards_df):
    plays_df = pd.merge(
        plays_df,
        yards_df,
        on=['gameId', 'playId'],
        how='left'
    )

    plays_history = plays_df[["passResult", "week", "possessionTeam", "defensiveTeam", "offense_advantage", "rushingYards", "passingYards"]].copy()
    plays_history['rushResult'] = plays_history['passResult']
    plays_history['sackResult'] = plays_history['passResult']
    plays_history['week'] = pd.to_numeric(plays_history['week'])
    return plays_history

# Create combinations of teams and weeks
def create_combinations(plays_history):
    all_teams = pd.concat([plays_history['possessionTeam'], plays_history['defensiveTeam']]).unique()
    all_weeks = range(1, plays_history['week'].max() + 1)
    all_advantages = ['advantage', 'disadvantage', 'neutral']

    combinations_offense = pd.DataFrame([
        {'possessionTeam': team, 'offense_advantage': adv, 'week': week}
        for team in all_teams
        for week in all_weeks
        for adv in all_advantages
    ])

    combinations_defense = pd.DataFrame([
        {'defensiveTeam': team, 'offense_advantage': adv, 'week': week}
        for team in all_teams
        for week in all_weeks
        for adv in all_advantages
    ])

    return combinations_offense, combinations_defense

# Calculate weekly stats for offense and defense
def calculate_weekly_stats(plays_history, combinations_offense, combinations_defense):
    # Offensive stats
    team_weekly_stats = plays_history.groupby(['possessionTeam', 'offense_advantage', 'week']).agg({
        'sackResult': lambda x: sum(x == 'S'),
        'passResult': lambda x: sum(x.isin(['C', 'I', 'IN'])),
        'rushResult': lambda x: sum(x.isna()),
        'passingYards': 'sum',
        'rushingYards': 'sum'
    }).reset_index()

    team_weekly_stats = pd.merge(
        combinations_offense,
        team_weekly_stats,
        on=['possessionTeam', 'week', 'offense_advantage'],
        how='left'
    ).fillna(0)

    team_weekly_stats.columns = ['possessionTeam', 'offense_advantage', 'week', 'sacks', 'pass_attempts', 'rush_attempts', 'passingYards', 'rushingYards']

    # Defensive stats
    defense_weekly_stats = plays_history.groupby(['defensiveTeam', 'offense_advantage', 'week']).agg({
        'sackResult': lambda x: sum(x == 'S'),
        'passResult': lambda x: sum(x.isin(['C', 'I', 'IN'])),
        'rushResult': lambda x: sum(x.isna()),
        'passingYards': 'sum',
        'rushingYards': 'sum'
    }).reset_index()

    defense_weekly_stats = pd.merge(
        combinations_defense,
        defense_weekly_stats,
        on=['defensiveTeam', 'week', 'offense_advantage'],
        how='left'
    ).fillna(0)

    defense_weekly_stats.columns = ['defensiveTeam', 'offense_advantage', 'week', 'sacks', 'pass_attempts', 'rush_attempts', 'passingYards', 'rushingYards']
    
    return team_weekly_stats, defense_weekly_stats

# Calculate cumulative stats
def calculate_cumulative_stats(team_weekly_stats, defense_weekly_stats):
    # Cumulative stats for offense
    team_cumulative_stats = team_weekly_stats.sort_values(['possessionTeam', 'offense_advantage', 'week'])
    team_cumulative_stats['cum_sacks'] = team_cumulative_stats.groupby(['possessionTeam', 'offense_advantage'])['sacks'].cumsum()
    team_cumulative_stats['cum_pass_attempts'] = team_cumulative_stats.groupby(['possessionTeam', 'offense_advantage'])['pass_attempts'].cumsum()
    team_cumulative_stats['cum_rush_attempts'] = team_cumulative_stats.groupby(['possessionTeam', 'offense_advantage'])['rush_attempts'].cumsum()
    team_cumulative_stats['cum_passing_yards'] = team_cumulative_stats.groupby(['possessionTeam', 'offense_advantage'])['passingYards'].cumsum()
    team_cumulative_stats['cum_rushing_yards'] = team_cumulative_stats.groupby(['possessionTeam', 'offense_advantage'])['rushingYards'].cumsum()
    team_cumulative_stats['avg_yards_per_pass'] = team_cumulative_stats['cum_passing_yards'] / team_cumulative_stats['cum_pass_attempts'].replace(0, 1)
    team_cumulative_stats['avg_yards_per_rush'] = team_cumulative_stats['cum_rushing_yards'] / team_cumulative_stats['cum_rush_attempts'].replace(0, 1)
    team_cumulative_stats['pct_pass_attempts'] = team_cumulative_stats["cum_pass_attempts"] / (team_cumulative_stats["cum_pass_attempts"] + team_cumulative_stats['cum_rush_attempts']).replace(0, 1)
    team_cumulative_stats['pct_sacks'] = team_cumulative_stats["cum_sacks"] / (team_cumulative_stats["cum_pass_attempts"] + team_cumulative_stats['cum_sacks']).replace(0, 1)

    # Prepare for merge: need stats up to the previous week
    team_cumulative_stats['week'] = team_cumulative_stats['week'] + 1

    # Cumulative stats for defense
    defense_cumulative_stats = defense_weekly_stats.sort_values(['defensiveTeam', 'offense_advantage', 'week'])
    defense_cumulative_stats['cum_sacks'] = defense_cumulative_stats.groupby(['defensiveTeam', 'offense_advantage'])['sacks'].cumsum()
    defense_cumulative_stats['cum_pass_attempts'] = defense_cumulative_stats.groupby(['defensiveTeam', 'offense_advantage'])['pass_attempts'].cumsum()
    defense_cumulative_stats['cum_rush_attempts'] = defense_cumulative_stats.groupby(['defensiveTeam', 'offense_advantage'])['rush_attempts'].cumsum()
    defense_cumulative_stats['cum_passing_yards'] = defense_cumulative_stats.groupby(['defensiveTeam', 'offense_advantage'])['passingYards'].cumsum()
    defense_cumulative_stats['cum_rushing_yards'] = defense_cumulative_stats.groupby(['defensiveTeam', 'offense_advantage'])['rushingYards'].cumsum()
    defense_cumulative_stats['defense_avg_yards_per_pass'] = defense_cumulative_stats['cum_passing_yards'] / defense_cumulative_stats['cum_pass_attempts'].replace(0, 1)
    defense_cumulative_stats['defense_avg_yards_per_rush'] = defense_cumulative_stats['cum_rushing_yards'] / defense_cumulative_stats['cum_rush_attempts'].replace(0, 1)
    defense_cumulative_stats['defense_pct_pass_attempts'] = defense_cumulative_stats["cum_pass_attempts"] / (defense_cumulative_stats["cum_pass_attempts"] + defense_cumulative_stats['cum_rush_attempts']).replace(0, 1)
    defense_cumulative_stats['defense_pct_sacks'] = defense_cumulative_stats["cum_sacks"] / (defense_cumulative_stats["cum_pass_attempts"] + defense_cumulative_stats["cum_sacks"]).replace(0, 1)

    # Prepare for merge: need stats up to the previous week
    defense_cumulative_stats['week'] = defense_cumulative_stats['week'] + 1

    return team_cumulative_stats, defense_cumulative_stats

# Merge final DataFrame
def merge_final_data(plays_df, team_cumulative_stats, defense_cumulative_stats):
    final_df = plays_df[(plays_df["passResult"].isin(["C", "I", "IN"])) & (plays_df["yardsGained"] > 0)]
    
    final_df = pd.merge(
        final_df,
        team_cumulative_stats[['possessionTeam', 'week', 'offense_advantage', 'avg_yards_per_pass', 'avg_yards_per_rush', 'pct_pass_attempts', 'pct_sacks']],
        on=['possessionTeam', 'week', 'offense_advantage'],
        how='left'
    )

    final_df = pd.merge(
        final_df,
        defense_cumulative_stats[['defensiveTeam', 'week', 'offense_advantage', 'defense_avg_yards_per_pass', 'defense_avg_yards_per_rush', 'defense_pct_pass_attempts', 'defense_pct_sacks']],
        on=['defensiveTeam', 'week', 'offense_advantage'],
        how='left'
    )

    # Transform win probabilities based on offensive/defensive team
    final_df['preSnapOffenseTeamWinProbability'] = np.where(
        final_df['is_home_offense'],
        final_df['preSnapHomeTeamWinProbability'],     # Home offense
        final_df['preSnapVisitorTeamWinProbability']   # Visitor offense
    )

    final_df['preSnapDefenseTeamWinProbability'] = np.where(
        final_df['is_home_offense'],
        final_df['preSnapVisitorTeamWinProbability'],  # Home offense, visitor defense
        final_df['preSnapHomeTeamWinProbability']      # Visitor offense, home defense
    )

    # Optionally drop original columns if no longer needed
    final_df.drop(['preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability'], axis=1, inplace=True)

    return final_df

# Convert game clock to seconds
def convert_to_seconds(game_clock, quarter):
    minutes, seconds = map(int, game_clock.split(':'))
    return (4 - quarter) * 15 * 60 + (minutes * 60 + seconds)

# Prepare heat map DataFrame
def prepare_heat_map(final_df):
    final_heat_map_df = final_df.loc[
        (~final_df["targetX"].isnull()),
        ["gameId", "playId", "gameClock", "quarter", "down", "yardsToGo", "preSnapHomeScore", "preSnapVisitorScore",
         "yardlineNumber", "yardlineNumber_v2", "score_difference_offense", "offense_advantage", "week", "pct_pass_attempts",
         "defense_pct_pass_attempts", 'avg_yards_per_pass', 'avg_yards_per_rush',
         'defense_avg_yards_per_rush', 'defense_avg_yards_per_pass', 'pct_sacks', 'defense_pct_sacks',
         "preSnapOffenseTeamWinProbability", "preSnapDefenseTeamWinProbability", "offenseFormation", "receiverAlignment",
         "targetX", "targetY"]
    ]

    final_heat_map_df['gameClock_original'] = final_heat_map_df['gameClock']
    final_heat_map_df['gameClock'] = final_heat_map_df.apply(lambda x: convert_to_seconds(x.gameClock, x.quarter), axis=1)

    return final_heat_map_df

# Create hexagons
def get_hex_df(hex_radius):
    hex_height = np.sqrt(3) * hex_radius
    
    hexagons = []
    centers = []
    x_ = []
    y_ = []
    
    for x in np.arange(0, 120, hex_height):
        for j, y in enumerate(np.arange(0, 53.3 + hex_height, hex_radius*1.5)):
            offset = j%2 * np.sqrt(3) * hex_radius/2
            
            vertices = [(x + offset + hex_radius * np.sin(np.pi/3 * i +(np.pi)),
                     y + hex_radius * np.cos(np.pi/3 * i)) for i in range(6)]
        
            polygon = Polygon(vertices)
            hexagons.append(polygon)
            centers.append(x + offset)
            x_.append(x + offset)
            y_.append(y)
    
    hexagon_df = pd.DataFrame({
        'hexagon': hexagons,
        'center': centers,
        'x': x_,
        'y': y_
    })

    return hexagon_df

# Merge final df and the hexagons
def merge_hexagons(final_heat_map_df,hexagon_df,training_weeks,test_weeks):
    gdf_points = gpd.GeoDataFrame(
        final_heat_map_df, 
        geometry=[Point(x, y) for x, y in zip(final_heat_map_df['targetX'], final_heat_map_df['targetY'])]
    )
    
    # Para hexagon_df
    gdf_hexagons = gpd.GeoDataFrame(hexagon_df, geometry='hexagon')
    
    # Realizar el join espacial
    final_heat_map_with_hex = gpd.sjoin(
        gdf_points, 
        gdf_hexagons, 
        how='left', 
        predicate='within'
    )

    final_heat_map_with_hex = final_heat_map_with_hex[final_heat_map_with_hex.week.isin(training_weeks+test_weeks)].reset_index(drop = True)
    final_heat_map_with_hex = final_heat_map_with_hex[~final_heat_map_with_hex.index_right.isnull()].reset_index(drop = True)

    final_heat_map_with_hex_last = final_heat_map_with_hex[final_heat_map_with_hex.week.isin(test_weeks)].copy()
    final_heat_map_with_hex_model = final_heat_map_with_hex[final_heat_map_with_hex.week.isin(training_weeks)].copy()

    return final_heat_map_with_hex, final_heat_map_with_hex_last, final_heat_map_with_hex_model

# Identify categorical and numerical variables
def identify_columns(final_heat_map_with_hex):
    categorical_columns = final_heat_map_with_hex.select_dtypes(include=['object', 'category']).columns
    numerical_columns = final_heat_map_with_hex.select_dtypes(include=['int64', 'float64']).columns
    bool_columns = final_heat_map_with_hex.select_dtypes(include=['bool']).columns

    # Exclude certain columns from features
    columns_to_exclude = ['yardlineNumber_v2', 'center', 'x', 'y', "gameId", "playId", "targetX", "targetY", "geometry", "week", "gameClock_original"]
    target_column = 'index_right'

    categorical_features = [col for col in categorical_columns if col not in columns_to_exclude and col != target_column]
    numerical_features = [col for col in numerical_columns if col not in columns_to_exclude and col != target_column]
    bool_features = [col for col in bool_columns if col not in columns_to_exclude and col != target_column]

    return categorical_features, numerical_features, bool_features

# Prepare DataFrame for the model
def prepare_model_data(final_heat_map_with_hex_model, categorical_features, numerical_features, bool_features):
    X = final_heat_map_with_hex_model[categorical_features + numerical_features + bool_features + ['yardlineNumber_v2']].copy()
    y = final_heat_map_with_hex_model['index_right']

    # Convert categorical variables to dummies
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Normalize numerical variables
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return X, y, scaler


# Train probabilistic model
def train_probabilistic_model(hexagon_df, X_, y_, category, verbose, oversampling_ratio=0.5):
    # Train a probabilistic model for a category with oversampling
    X = X_[((hexagon_df.loc[category].center - (X_.yardlineNumber_v2 + 10)) <= 50) & 
            ((hexagon_df.loc[category].center - (X_.yardlineNumber_v2 + 10)) >= -10)].copy()
    y = y_.loc[X.index].copy()

    X.drop(['yardlineNumber_v2'], axis=1, inplace=True)

    # Create binary variable for the current category
    y_binary = (y == category).astype(int)

    # Check class distribution
    class_counts = Counter(y_binary)
    if verbose:
        print(f"Original class distribution for category {category}:")
        print(class_counts)

    try:
        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, 
            test_size=0.2, 
            random_state=42,
            stratify=y_binary
        )

        # Apply SMOTE for oversampling the minority class
        if min(Counter(y_train).values()) >= 5:  # SMOTE needs at least 5 samples
            sampling_strategy = min(oversampling_ratio, 1.0)
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
                k_neighbors=min(5, min(Counter(y_train).values()) - 1)
            )
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            if verbose:
                print("Distribution after SMOTE:")
                print(Counter(y_train_balanced))
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            if verbose:
                print("SMOTE could not be applied - too few samples")

        # Train the model
        model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=10,      # Maximum depth of the tree
            random_state=42
        )

        model.fit(X_train_balanced, y_train_balanced)

        # Evaluate the model
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        brier = brier_score_loss(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)

        if verbose:
            print(f"\nCalibration metrics:")
            print(f"Brier Score: {brier:.4f}")
            print(f"Log Loss: {logloss:.4f}")

        return {
            'model': model,
            'brier_score': brier,
            'log_loss': logloss
        }

    except Exception as e:
        if verbose:
            print(f"Error in the model for category {category}: {str(e)}")
        return None

# Create probabilistic models for all categories
def create_probabilistic_models(hexagon_df, X, y, verbose):
    X_scaled = X

    # Get unique categories
    categories = y.unique()
    if verbose:
        print(f"Total number of categories: {len(categories)}")

    # Dictionary to store models
    models = {}

    # Train a model for each category
    for category in categories:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing category: {category}")

        result = train_probabilistic_model(hexagon_df, X_scaled, y, category, verbose)

        if result is not None:
            models[category] = result

    return models

# Predict probabilities for new data
def predict_probabilities(df, hexagon_df, models, categorical_features, numerical_features, bool_features, final_features, scaler):
    # Predict probabilities for all categories
    X_new = df[categorical_features + numerical_features + bool_features + ['yardlineNumber_v2']].copy()
    X_new = pd.get_dummies(X_new, columns=categorical_features, drop_first=True)
    missing_columns = [name for name in final_features if name not in X_new.columns]
    for col in missing_columns:
        X_new[col] = False

    extra_columns = [name for name in X_new.columns if name not in final_features]
    X_new.drop(extra_columns, axis=1, inplace=True)
    X_new = X_new[final_features]

    X_new[numerical_features] = scaler.fit_transform(X_new[numerical_features])
    X_new_v2 = X_new.copy()
    X_new_v2.drop(['yardlineNumber_v2'], axis=1, inplace=True)

    # Dictionary to store probabilities
    probabilities = {}

    # Predict for each category
    for category, model_info in models.items():
        model = model_info['model']
        prob = model.predict_proba(X_new_v2)[:, 1]

        flag = list(((hexagon_df.loc[category].center - (X_new.yardlineNumber_v2 + 10)) <= 50) & 
                     ((hexagon_df.loc[category].center - (X_new.yardlineNumber_v2 + 10)) >= -10))

        prob = [num if boolean else 0 for boolean, num in zip(flag, prob)]

        probabilities[category] = prob

    # Convert to DataFrame
    prob_df = pd.DataFrame(probabilities)

    # Normalize probabilities to sum to 1 per row
    prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)

    return prob_df, X_new

# Calculate position match percentage
def calculate_position_match_percentage(list1, list2):
    # Calculate the percentage of matches between two lists in the same position.
    min_length = min(len(list1), len(list2))

    if min_length == 0:
        return 0.0  # Avoid division by zero if both lists are empty

    matches = sum(1 for i in range(min_length) if list1[i] == list2[i])
    match_percentage = (matches / min_length) * 100
    return match_percentage

# Get maximum predictions
def get_max_predictions(prob_df):
    max_values = prob_df.max(axis=1)
    max_categories = prob_df.idxmax(axis=1)

    results = pd.DataFrame({
        'predicted_category': max_categories,
        'probability': max_values
    })

    return results

# Get final score
def get_score(final_heat_map_with_hex_last, prob_df, threshold):
    score = 0
    
    for i in range(0,len(final_heat_map_with_hex_last)):
        final_category = final_heat_map_with_hex_last.iloc[i].index_right
        
        prob_categories = pd.DataFrame(prob_df.iloc[i].sort_values(ascending=False)).reset_index()
        prob_categories.columns = ['category','prob']
        prob_categories['cum_prob'] = prob_categories['prob'].cumsum()
        
        threshold_row = prob_categories[prob_categories['cum_prob'] > threshold].index.min()
        
        if threshold_row is not None:
            threshold_categories = list(prob_categories.loc[:threshold_row].category)
        else:
            threshold_categories = list(prob_categories.category)
    
        if final_category in threshold_categories:
            score = score + 1

    percentage = score*100/len(final_heat_map_with_hex_last)
    print(f"Final score: {percentage:.2f}%")

    return percentage

# Create football field visualization
def create_football_field(hex_df, play_number, hex_radius, df_heat_map, linenumbers=True, endzones=True, highlight_line=False, highlight_line_number=50, highlighted_name='Line of Scrimmage', fifty_is_los=False, figsize=(12, 6.33), ball_yardline=None):
    # Function that plots the football field for viewing plays
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3, linewidth=0.1, edgecolor='r', facecolor='blue', alpha=0.2, zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='blue', alpha=0.2, zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name), color='yellow')

    # Plot the ball as a brown oval
    if ball_yardline is not None:
        ball_line = df_heat_map.iloc[play_number].yardlineNumber + 10
        football = patches.Ellipse((ball_line, 26.65), width=2, height=1, edgecolor='brown', facecolor='brown', zorder=5)
        ax.add_patch(football)
        plt.text(ball_line + 2, 50, '<- Snap', color='brown')

        football = patches.Ellipse((df_heat_map.iloc[play_number].targetX, df_heat_map.iloc[play_number].targetY), width=2, height=1, edgecolor='lightblue', facecolor='lightblue', zorder=5)
        ax.add_patch(football)

    hex_height = np.sqrt(3) * hex_radius

    # List to store hexagon polygons
    hexagons = []
    for index, row in hex_df.iterrows():
        if row.prob > 0.1:
            hexagon = patches.RegularPolygon(
                (row.x, row.y), numVertices=6, radius=hex_radius,
                edgecolor='red', facecolor='none', zorder=4, color="orange", alpha=row.prob / max(hex_df.prob)
            )
            ax.add_patch(hexagon)
        else:
            hexagon = patches.RegularPolygon(
                (row.x, row.y), numVertices=6, radius=hex_radius,
                edgecolor='red', facecolor='none', zorder=4, color="orange", alpha=row.prob / 0.1
            )
            ax.add_patch(hexagon)

    return fig, ax

# Main execution flow
if __name__ == "__main__":
    # Define base files and weeks
    base_files = ['games', 'player_play', 'players', 'plays']
    tracking_weeks = range(1, 10)
    hex_radius = 6
    training_weeks = [2,3,4,5,6,7,8]
    test_weeks = [9]
    threshold = 0.7
    verbose = False

    # Load data
    dfs = load_data(base_files, tracking_weeks)

    # Merge plays and games
    plays_df = merge_plays_and_games(dfs)

    # Calculate yardline and target coordinates
    plays_df = calculate_yardline_and_targets(plays_df)

    # Calculate score difference and offensive advantage
    plays_df = calculate_score_difference(plays_df)

    # Get yards from player play DataFrame
    yards_df = get_yards_from_player_play(dfs)

    # Prepare plays DataFrame for historical analysis
    plays_history = prepare_plays_history(plays_df, yards_df)

    # Create combinations of teams and weeks
    combinations_offense, combinations_defense = create_combinations(plays_history)

    # Calculate weekly stats for offense and defense
    team_weekly_stats, defense_weekly_stats = calculate_weekly_stats(plays_history, combinations_offense, combinations_defense)

    # Calculate cumulative stats
    team_cumulative_stats, defense_cumulative_stats = calculate_cumulative_stats(team_weekly_stats, defense_weekly_stats)

    # Merge final DataFrame
    final_df = merge_final_data(plays_df, team_cumulative_stats, defense_cumulative_stats)

    # Prepare heat map DataFrame
    final_heat_map_df = prepare_heat_map(final_df)

    # Get hexagons
    hexagon_df = get_hex_df(hex_radius)

    # Merge final DataFrame and hexagons
    final_heat_map_with_hex, final_heat_map_with_hex_last, final_heat_map_with_hex_model = merge_hexagons(final_heat_map_df, hexagon_df, training_weeks, test_weeks)
    
    # Identify categorical and numerical variables
    categorical_features, numerical_features, bool_features = identify_columns(final_heat_map_with_hex)

    # Prepare DataFrame for the model
    X, y, scaler = prepare_model_data(final_heat_map_with_hex_model, categorical_features, numerical_features, bool_features)

    # Create probabilistic models
    models = create_probabilistic_models(hexagon_df, X, y, verbose)

    # # Ex. 1
    # final_heat_map_with_hex_last = final_heat_map_with_hex_last[
    #         (final_heat_map_with_hex_last.down.isin([3])) &
    #         (final_heat_map_with_hex_last.quarter.isin([4])) &
    #         (final_heat_map_with_hex_last.offense_advantage.isin(['advantage','disadvantage'])) &
    #         (final_heat_map_with_hex_last.yardlineNumber >= 35) & 
    #         (final_heat_map_with_hex_last.yardlineNumber <= 45) 
    #     ].copy()
    
    # # Ex. 2
    # final_heat_map_with_hex_last = final_heat_map_with_hex_last[
    #         (final_heat_map_with_hex_last.down.isin([2,3])) &
    #         (final_heat_map_with_hex_last.quarter.isin([1,4])) &
    #         (final_heat_map_with_hex_last.offense_advantage.isin(['disadvantage'])) &
    #         (final_heat_map_with_hex_last.yardlineNumber >= 20) & 
    #         (final_heat_map_with_hex_last.yardlineNumber <= 30) 
    #     ].copy()
    
    # Predict probabilities
    prob_df_result, X_test = predict_probabilities(final_heat_map_with_hex_last, hexagon_df, models, categorical_features, numerical_features, bool_features, X.columns, scaler)

    # Get maximum predictions
    predictions = get_max_predictions(prob_df_result)

    # Get final score
    final_score = get_score(final_heat_map_with_hex_last, prob_df_result, threshold)
    
    # Visualization of the football field
    for play_number in range(0, 10):
        print('--------------------------------------------')
        print('Play number: %s'%(play_number))
        print(final_heat_map_with_hex_last.iloc[play_number])
        hex_df = pd.merge(hexagon_df, pd.DataFrame(prob_df_result.iloc[play_number].transpose()), left_index=True, right_index=True, how='left')
        hex_df.drop(['center'], axis=1, inplace=True)
        hex_df.columns = ['POLYGON', 'x', 'y', 'prob']
        hex_df.fillna(0, inplace=True)
        print(f"Likelihood: {hex_df.loc[final_heat_map_with_hex_last.iloc[play_number].index_right].prob * 100:.2f}%")
        print(f"Max likelihood: {max(hex_df.prob) * 100:.2f}%")
        fig, ax = create_football_field(hex_df, play_number, hex_radius, final_heat_map_with_hex_last, ball_yardline=35)
        # plt.savefig(str(final_heat_map_with_hex_last.iloc[play_number].gameId) + '_' + str(final_heat_map_with_hex_last.iloc[play_number].playId) + '.png', format='png')
        plt.show()

