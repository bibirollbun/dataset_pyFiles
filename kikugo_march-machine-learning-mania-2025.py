import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import ipywidgets as widgets
from IPython.display import display
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import chardet
from scipy.stats import linregress

sns.set_style("whitegrid")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
warnings.filterwarnings("ignore")

def load_csv(csv_file):
    with open(csv_file, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
    return pd.read_csv(csv_file, encoding=result["encoding"])

def get_comp_files_and_dirs(input_dir):
    file_list = []
    dir_list = []
    try:
        for comp_dir in os.listdir(input_dir):
            comp_path = '/'.join([input_dir, comp_dir])
            print(f"Competition Directory: {comp_path}")
            print("Contains:")
            with os.scandir(comp_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        print(f"- (File) {entry.name}, Size: {entry.stat().st_size} bytes")
                        file_list.append(os.path.join(input_dir, comp_dir, entry))
                    elif entry.is_dir():
                        print(f"- (Folder) {entry.name}")
                        dir_list.append(os.path.join(input_dir, comp_dir, entry))
    except FileNotFoundError:
        print(f"The specified directory '{input_dir}' does not exist.")
    except PermissionError:
        print(f"Permission error accessing directory '{input_dir}'.")
    return file_list, dir_list

input_dir = '/kaggle/input'

file_list, dir_list = get_comp_files_and_dirs(input_dir)


comp_dir = '/kaggle/input/march-machine-learning-mania-2025'

mens_teams_df = load_csv(os.path.join(comp_dir, 'MTeams.csv'))
womens_teams_df = load_csv(os.path.join(comp_dir, 'WTeams.csv'))

mens_seeds_df = load_csv(os.path.join(comp_dir, 'MNCAATourneySeeds.csv'))
womens_seeds_df = load_csv(os.path.join(comp_dir, 'WNCAATourneySeeds.csv'))

sample_submission_df = load_csv(os.path.join(comp_dir, 'SampleSubmissionStage1.csv'))

print("Men's Teams Data:")
display(mens_teams_df.head())
print("Women's Teams Data:")
display(womens_teams_df.head())
print("Men's Tournament Seeds Data:")
display(mens_seeds_df.head())
print("Women's Tournament Seeds Data:")
display(womens_seeds_df.head())
print("Sample Submission Data:")
display(sample_submission_df.head())


mens_teams_df['Duration'] = mens_teams_df['LastD1Season'] - mens_teams_df['FirstD1Season']

plt.figure(figsize=(15, 60))
sns.barplot(
data=mens_teams_df,
y="TeamName",
x="Duration",
color="blue",
edgecolor="black"
)
plt.title("Durations of Men's NCAA Teams in Division I")
plt.xlabel('Duration (Years)')
plt.ylabel('Team Name')
plt.tight_layout()
plt.show()


# Function to extract numeric seed values
def extract_seed_value(seed_str):
    try:
        return int(seed_str[1:3])  # Extract numeric part of the seed (e.g., "W01" -> 1)
    except ValueError:
        return 16  # Default to 16 for invalid or missing seeds

# Add a column for numeric seed values
mens_seeds_df['SeedValue'] = mens_seeds_df['Seed'].apply(extract_seed_value)
womens_seeds_df['SeedValue'] = womens_seeds_df['Seed'].apply(extract_seed_value)

plt.figure(figsize=(12, 6))
sns.histplot(mens_seeds_df['SeedValue'], bins=16, kde=True, color='skyblue', label='Men')
sns.histplot(womens_seeds_df['SeedValue'], bins=16, kde=True, color='pink', label='Women')
plt.title('Distribution of Tournament Seeds')
plt.xlabel('Seed Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()


mens_reg_season_df = load_csv(os.path.join(comp_dir, 'MRegularSeasonCompactResults.csv'))
womens_reg_season_df = load_csv(os.path.join(comp_dir, 'WRegularSeasonCompactResults.csv'))

for gender, df in zip(['Men', 'Women'], [mens_reg_season_df, womens_reg_season_df]):
    print(f"Investigating {gender} Regular Season Results")
    df['Score Difference'] = df['WScore'] - df['LScore']

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Season', y='WScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Winning Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Season', y='LScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Losing Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Season', y='Score Difference', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Score Difference Distributions')
    plt.tight_layout()
    plt.show()


def calculate_elo(teams, data, initial_rating=2000, k=140, alpha=None, weights=False, nan_score=1):
    team_dict = {team: initial_rating for team in teams}
    r1, r2, loss = [], [], []
    margin_of_victory = 1
    weight = 1

    for wteam, lteam, ws, ls, w in tqdm(zip(data.WTeamID, data.LTeamID, data.WScore, data.LScore, data.weight), total=len(data)):
        rateW = 1 / (1 + 10 ** ((team_dict[lteam] - team_dict[wteam]) / initial_rating))
        rateL = 1 / (1 + 10 ** ((team_dict[wteam] - team_dict[lteam]) / initial_rating))
        
        if alpha:
            margin_of_victory = (ws - ls) / alpha
        if isinstance(weights, (list, np.ndarray, pd.Series)):
            weight = w

        team_dict[wteam] += w * k * margin_of_victory * (1 - rateW)
        team_dict[lteam] += w * k * margin_of_victory * (0 - rateL)

        if team_dict[lteam] < 1:
            team_dict[lteam] = 1

        r1.append(team_dict[wteam])
        r2.append(team_dict[lteam])
        loss.append((1 - rateW) ** 2)
        
    return r1, r2, loss

def create_elo_data(teams, data, initial_rating=2000, k=140, alpha=None, weights=None, nan_score=1):
    if isinstance(weights, (list, np.ndarray, pd.Series)):
        data['weight'] = weights
    else:
        data['weight'] = 1
     
    r1, r2, loss = calculate_elo(teams, data, initial_rating, k, alpha, weights, nan_score)
    loss = np.mean(np.array(loss)[data.tourney == 1])
    print(f"Loss: {loss}")
    
    seasons = np.concatenate([data.Season, data.Season])
    days = np.concatenate([data.DayNum, data.DayNum])
    teams = np.concatenate([data.WTeamID, data.LTeamID])
    tourney = np.concatenate([data.tourney, data.tourney])
    ratings = np.concatenate([r1, r2])
    
    rating_df = pd.DataFrame({
        'Season': seasons,
        'DayNum': days,
        'TeamID': teams,
        'Rating': ratings,
        'Tourney': tourney
    }) 

    rating_df.sort_values(['TeamID', 'Season', 'DayNum'], inplace=True)
    rating_df = rating_df[rating_df['Tourney'] == 0]
    grouped = rating_df.groupby(['TeamID', 'Season'])
    results = grouped['Rating'].agg(['mean', 'median', 'std', 'min', 'max', 'last'])
    results.columns = ['Rating_Mean', 'Rating_Median', 'Rating_Std', 'Rating_Min', 'Rating_Max', 'Rating_Last']
    results['Rating_Trend'] = grouped.apply(lambda x: linregress(range(len(x)), x['Rating']).slope)
    results.reset_index(inplace=True)
    
    return results


regular_m = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
tourney_m = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
teams_m = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')

# Add tournament flag and weights
regular_m['tourney'] = 0
tourney_m['tourney'] = 1
regular_m['weight'] = 1
tourney_m['weight'] = 0.7

# Combine regular season and tournament data
data_m = pd.concat([regular_m, tourney_m])
data_m.sort_values(['Season', 'DayNum'], inplace=True)
data_m.reset_index(inplace=True, drop=True)

elo_df_men = create_elo_data(teams_m.TeamID, data_m, initial_rating=1200, k=125, alpha=None, weights=data_m['weight'])

elo_df_men.to_csv('mens_elo_rating.csv', index=False)

regular_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')
tourney_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
teams_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeams.csv')

# Add tournament flag and weights
regular_w['tourney'] = 0
tourney_w['tourney'] = 1
regular_w['weight'] = 0.95
tourney_w['weight'] = 1

# Combine regular season and tournament data
data_w = pd.concat([regular_w, tourney_w])
data_w.sort_values(['Season', 'DayNum'], inplace=True)
data_w.reset_index(inplace=True, drop=True)

elo_df_women = create_elo_data(teams_w.TeamID, data_w, initial_rating=1250, k=190, alpha=None, weights=data_w['weight'])

elo_df_women.to_csv('womens_elo_rating.csv', index=False)

print("Men's Elo Ratings:")
display(elo_df_men.head())

print("Women's Elo Ratings:")
display(elo_df_women.head())


# Top 20 Teams Based on Latest Data
tmp_df_men = pd.merge(elo_df_men, teams_m, on='TeamID', how='left')
tmp_df_men = tmp_df_men[tmp_df_men['Season'] == 2025]
top_men_teams = tmp_df_men.sort_values('Rating_Last', ascending=False)[:20][['TeamName', 'Rating_Last', 'Rating_Trend']]
top_men_teams = top_men_teams.reindex(index=top_men_teams.index[::-1])

tmp_df_women = pd.merge(elo_df_women, teams_w, on='TeamID', how='left')
tmp_df_women = tmp_df_women[tmp_df_women['Season'] == 2025]
top_women_teams = tmp_df_women.sort_values('Rating_Last', ascending=False)[:20][['TeamName', 'Rating_Last', 'Rating_Trend']]
top_women_teams = top_women_teams.reindex(index=top_women_teams.index[::-1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

# Men's Teams
ax1.barh(top_men_teams['TeamName'], top_men_teams['Rating_Last'], color='skyblue', label='Rating_Last')
ax1.set_title("Top Men's Teams - 2025")
ax1.set_xlabel('Last Rating')
ax1.set_ylabel('TeamName')
ax1.legend()

# Women's Teams
ax2.barh(top_women_teams['TeamName'], top_women_teams['Rating_Last'], color='#3F51B5', label='Rating_Last')
ax2.set_title("Top Women's Teams - 2025")
ax2.set_xlabel('Last Rating')
ax2.set_ylabel('TeamName')
ax2.legend()

plt.tight_layout()
plt.show()


# Load base datasets
comp_dir = '/kaggle/input/march-machine-learning-mania-2025'

def load_data(gender='M'):
    prefix = 'M' if gender == 'M' else 'W'
    
    teams = pd.read_csv(f'{comp_dir}/{prefix}Teams.csv')
    seeds = pd.read_csv(f'{comp_dir}/{prefix}NCAATourneySeeds.csv')
    regular_season = pd.read_csv(f'{comp_dir}/{prefix}RegularSeasonCompactResults.csv')
    tourney = pd.read_csv(f'{comp_dir}/{prefix}NCAATourneyCompactResults.csv')
    
    # Add tournament flag
    regular_season['tourney'] = 0
    tourney['tourney'] = 1
    
    # Combine data
    all_games = pd.concat([regular_season, tourney])
    all_games.sort_values(['Season', 'DayNum'], inplace=True)
    
    return teams, seeds, all_games

mens_teams, mens_seeds, mens_games = load_data('M')
womens_teams, womens_seeds, womens_games = load_data('W')


# Calculate Elo ratings function
def calculate_elo_ratings(teams, games, initial_rating=1500, k=20):
    team_ratings = {team: initial_rating for team in teams['TeamID']}
    ratings_history = []
    
    for _, game in tqdm(games.iterrows(), total=len(games)):
        w_team, l_team = game['WTeamID'], game['LTeamID']
        w_rating, l_rating = team_ratings[w_team], team_ratings[l_team]
        
        # Calculate expected win probability
        exp_w = 1 / (1 + 10**((l_rating - w_rating)/400))
        
        # Update ratings
        rating_change = k * (1 - exp_w)
        team_ratings[w_team] += rating_change
        team_ratings[l_team] -= rating_change
        
        # Store ratings
        ratings_history.append({
            'Season': game['Season'],
            'TeamID': w_team,
            'Rating': team_ratings[w_team],
            'DayNum': game['DayNum']
        })
        ratings_history.append({
            'Season': game['Season'],
            'TeamID': l_team,
            'Rating': team_ratings[l_team],
            'DayNum': game['DayNum']
        })
    
    return pd.DataFrame(ratings_history)

# Calculate ratings for both genders
mens_elo = calculate_elo_ratings(mens_teams, mens_games)
womens_elo = calculate_elo_ratings(womens_teams, womens_games)

# Get final ratings for each season
def get_final_ratings(elo_df):
    return elo_df.sort_values('DayNum').groupby(['Season', 'TeamID'])['Rating'].last().reset_index()

mens_final_elo = get_final_ratings(mens_elo)
womens_final_elo = get_final_ratings(womens_elo)


# Prepare training data from historical tournaments
def prepare_training_data(tourney_games, seeds_df, elo_df):
    # Extract seed numbers
    seeds_df['SeedNum'] = seeds_df['Seed'].str[1:3].astype(int)
    
    # Prepare features for each tournament game
    features = []
    for _, game in tourney_games[tourney_games['tourney'] == 1].iterrows():
        season = game['Season']
        
        # Get seeds
        w_seed = seeds_df[(seeds_df['Season'] == season) & 
                         (seeds_df['TeamID'] == game['WTeamID'])]['SeedNum'].iloc[0]
        l_seed = seeds_df[(seeds_df['Season'] == season) & 
                         (seeds_df['TeamID'] == game['LTeamID'])]['SeedNum'].iloc[0]
        
        # Get Elo ratings
        w_elo = elo_df[(elo_df['Season'] == season) & 
                       (elo_df['TeamID'] == game['WTeamID'])]['Rating'].iloc[0]
        l_elo = elo_df[(elo_df['Season'] == season) & 
                       (elo_df['TeamID'] == game['LTeamID'])]['Rating'].iloc[0]
        
        features.append({
            'Season': season,
            'Team1': min(game['WTeamID'], game['LTeamID']),
            'Team2': max(game['WTeamID'], game['LTeamID']),
            'Seed1': w_seed if game['WTeamID'] < game['LTeamID'] else l_seed,
            'Seed2': l_seed if game['WTeamID'] < game['LTeamID'] else w_seed,
            'Elo1': w_elo if game['WTeamID'] < game['LTeamID'] else l_elo,
            'Elo2': l_elo if game['WTeamID'] < game['LTeamID'] else w_elo,
            'Target': 1 if game['WTeamID'] < game['LTeamID'] else 0
        })
    
    return pd.DataFrame(features)

# Prepare training data for both tournaments
mens_train = prepare_training_data(mens_games, mens_seeds, mens_final_elo)
womens_train = prepare_training_data(womens_games, womens_seeds, womens_final_elo)


# Train models
def train_model(train_data):
    X = train_data[['Elo1', 'Elo2', 'Seed1', 'Seed2']]
    y = train_data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    return model

mens_model = train_model(mens_train)
womens_model = train_model(womens_train)


# Prepare submission predictions
def prepare_submission(submission_df, seeds_df, elo_df, model):
    # Extract season and team IDs
    submission_df[['Season', 'Team1', 'Team2']] = submission_df['ID'].str.split('_', expand=True)
    submission_df['Season'] = submission_df['Season'].astype(int)
    submission_df['Team1'] = submission_df['Team1'].astype(int)
    submission_df['Team2'] = submission_df['Team2'].astype(int)
    
    # Merge seeds and Elo ratings
    submission_df = submission_df.merge(
        seeds_df[['Season', 'TeamID', 'SeedNum']],
        left_on=['Season', 'Team1'],
        right_on=['Season', 'TeamID'],
        how='left'
    ).rename(columns={'SeedNum': 'Seed1'}).drop('TeamID', axis=1)
    
    submission_df = submission_df.merge(
        seeds_df[['Season', 'TeamID', 'SeedNum']],
        left_on=['Season', 'Team2'],
        right_on=['Season', 'TeamID'],
        how='left'
    ).rename(columns={'SeedNum': 'Seed2'}).drop('TeamID', axis=1)
    
    submission_df = submission_df.merge(
        elo_df[['Season', 'TeamID', 'Rating']],
        left_on=['Season', 'Team1'],
        right_on=['Season', 'TeamID'],
        how='left'
    ).rename(columns={'Rating': 'Elo1'}).drop('TeamID', axis=1)
    
    submission_df = submission_df.merge(
        elo_df[['Season', 'TeamID', 'Rating']],
        left_on=['Season', 'Team2'],
        right_on=['Season', 'TeamID'],
        how='left'
    ).rename(columns={'Rating': 'Elo2'}).drop('TeamID', axis=1)
    
    # Fill missing values
    submission_df = submission_df.fillna({'Seed1': 16, 'Seed2': 16, 'Elo1': 1500, 'Elo2': 1500})
    
    # Generate predictions
    X_pred = submission_df[['Elo1', 'Elo2', 'Seed1', 'Seed2']]
    submission_df['Pred'] = model.predict_proba(X_pred)[:, 1]
    
    return submission_df[['ID', 'Pred']]

# Load submission template
submission_template = pd.read_csv(f'{comp_dir}/SampleSubmissionStage1.csv')

# Generate predictions for men's and women's tournaments
mens_submission = prepare_submission(
    submission_template[submission_template['ID'].str.contains('_1')].copy(),
    mens_seeds,
    mens_final_elo,
    mens_model
)

womens_submission = prepare_submission(
    submission_template[submission_template['ID'].str.contains('_3')].copy(),
    womens_seeds,
    womens_final_elo,
    womens_model
)

# Combine predictions
final_submission = pd.concat([mens_submission, womens_submission])
final_submission.to_csv('submission.csv', index=False)
print("Final submission shape:", final_submission.shape)


display(final_submission.head())


mens_reg_season_df = load_csv(os.path.join(comp_dir, 'MRegularSeasonCompactResults.csv'))
womens_reg_season_df = load_csv(os.path.join(comp_dir, 'WRegularSeasonCompactResults.csv'))

for gender, df in zip(['Men', 'Women'], [mens_reg_season_df, womens_reg_season_df]):
    print(f"Investigating {gender} Regular Season Results")
    df['Score Difference'] = df['WScore'] - df['LScore']

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Season', y='WScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Winning Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Season', y='LScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Losing Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Season', y='Score Difference', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Score Difference Distributions')
    plt.tight_layout()
    plt.show()


def calculate_elo(teams, data, initial_rating=2000, k=140, alpha=None, weights=False, nan_score=1):
    team_dict = {team: initial_rating for team in teams}
    r1, r2, loss = [], [], []
    margin_of_victory = 1
    weight = 1

    for wteam, lteam, ws, ls, w in tqdm(zip(data.WTeamID, data.LTeamID, data.WScore, data.LScore, data.weight), total=len(data)):
        rateW = 1 / (1 + 10 ** ((team_dict[lteam] - team_dict[wteam]) / initial_rating))
        rateL = 1 / (1 + 10 ** ((team_dict[wteam] - team_dict[lteam]) / initial_rating))

        if alpha:
            margin_of_victory = (ws - ls) / alpha
        if isinstance(weights, (list, np.ndarray, pd.Series)):
            weight = w

        team_dict[wteam] += w * k * margin_of_victory * (1 - rateW)
        team_dict[lteam] += w * k * margin_of_victory * (0 - rateL)

        if team_dict[lteam] < 1:
            team_dict[lteam] = 1

        r1.append(team_dict[wteam])
        r2.append(team_dict[lteam])
        loss.append((1 - rateW) ** 2)

    return r1, r2, loss

def create_elo_data(teams, data, initial_rating=2000, k=140, alpha=None, weights=None, nan_score=1):
    if isinstance(weights, (list, np.ndarray, pd.Series)):
        data['weight'] = weights
    else:
        data['weight'] = 1

    r1, r2, loss = calculate_elo(teams, data, initial_rating, k, alpha, weights, nan_score)
    loss = np.mean(np.array(loss)[data.tourney == 1])
    print(f"Loss: {loss}")

    seasons = np.concatenate([data.Season, data.Season])
    days = np.concatenate([data.DayNum, data.DayNum])
    teams_concat = np.concatenate([data.WTeamID, data.LTeamID])
    tourney = np.concatenate([data.tourney, data.tourney])
    ratings = np.concatenate([r1, r2])

    rating_df = pd.DataFrame({
        'Season': seasons,
        'DayNum': days,
        'TeamID': teams_concat,
        'Rating': ratings,
        'Tourney': tourney
    })

    rating_df.sort_values(['TeamID', 'Season', 'DayNum'], inplace=True)
    rating_df = rating_df[rating_df['Tourney'] == 0]
    grouped = rating_df.groupby(['TeamID', 'Season'])
    results = grouped['Rating'].agg(['mean', 'median', 'std', 'min', 'max', 'last'])
    results.columns = ['Rating_Mean', 'Rating_Median', 'Rating_Std', 'Rating_Min', 'Rating_Max', 'Rating_Last']
    results['Rating_Trend'] = grouped.apply(lambda x: linregress(range(len(x)), x['Rating']).slope)
    results.reset_index(inplace=True)

    return results


regular_m = load_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
tourney_m = load_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
teams_m = load_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')

regular_m['tourney'] = 0
tourney_m['tourney'] = 1
regular_m['weight'] = 1
tourney_m['weight'] = 0.7

data_m = pd.concat([regular_m, tourney_m])
data_m.sort_values(['Season', 'DayNum'], inplace=True)
data_m.reset_index(inplace=True, drop=True)

elo_df_men = create_elo_data(teams_m.TeamID, data_m, initial_rating=1200, k=125, alpha=None, weights=data_m['weight'])
elo_df_men.to_csv('mens_elo_rating.csv', index=False)

regular_w = load_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')
tourney_w = load_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
teams_w = load_csv('/kaggle/input/march-machine-learning-mania-2025/WTeams.csv')

regular_w['tourney'] = 0
tourney_w['tourney'] = 1
regular_w['weight'] = 0.95
tourney_w['weight'] = 1

data_w = pd.concat([regular_w, tourney_w])
data_w.sort_values(['Season', 'DayNum'], inplace=True)
data_w.reset_index(inplace=True, drop=True)

elo_df_women = create_elo_data(teams_w.TeamID, data_w, initial_rating=1250, k=190, alpha=None, weights=data_w['weight'])
elo_df_women.to_csv('womens_elo_rating.csv', index=False)

print("Men's Elo Ratings:")
display(elo_df_men.head())
print("Women's Elo Ratings:")
display(elo_df_women.head())


tmp_df_men = pd.merge(elo_df_men, teams_m, on='TeamID', how='left')
tmp_df_men = tmp_df_men[tmp_df_men['Season'] == 2025]
top_men_teams = tmp_df_men.sort_values('Rating_Last', ascending=False)[:20][['TeamName', 'Rating_Last', 'Rating_Trend']]
top_men_teams = top_men_teams.reindex(index=top_men_teams.index[::-1])

tmp_df_women = pd.merge(elo_df_women, teams_w, on='TeamID', how='left')
tmp_df_women = tmp_df_women[tmp_df_women['Season'] == 2025]
top_women_teams = tmp_df_women.sort_values('Rating_Last', ascending=False)[:20][['TeamName', 'Rating_Last', 'Rating_Trend']]
top_women_teams = top_women_teams.reindex(index=top_women_teams.index[::-1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

ax1.barh(top_men_teams['TeamName'], top_men_teams['Rating_Last'], color='skyblue', label='Rating_Last')
ax1.set_title("Top Men's Teams - 2025")
ax1.set_xlabel('Last Rating')
ax1.set_ylabel('TeamName')
ax1.legend()

ax2.barh(top_women_teams['TeamName'], top_women_teams['Rating_Last'], color='#3F51B5', label='Rating_Last')
ax2.set_title("Top Women's Teams - 2025")
ax2.set_xlabel('Last Rating')
ax2.set_ylabel('TeamName')
ax2.legend()

plt.tight_layout()
plt.show()


elo_df_men = pd.read_csv('mens_elo_rating.csv')
elo_df_women = pd.read_csv('womens_elo_rating.csv')
mens_seeds_df = load_csv(os.path.join(comp_dir, 'MNCAATourneySeeds.csv'))
womens_seeds_df = load_csv(os.path.join(comp_dir, 'WNCAATourneySeeds.csv'))

def extract_seed_value(seed_str):
    try:
        return int(seed_str[1:])
    except ValueError:
        return 16

mens_seeds_df['SeedValue'] = mens_seeds_df['Seed'].apply(extract_seed_value)
womens_seeds_df['SeedValue'] = womens_seeds_df['Seed'].apply(extract_seed_value)
seed_df = pd.concat([mens_seeds_df, womens_seeds_df], ignore_index=True)

regular_m = load_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
tourney_m = load_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
regular_w = load_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')
tourney_w = load_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')

regular_m['Gender'] = 'Men'
tourney_m['Gender'] = 'Men'
regular_w['Gender'] = 'Women'
tourney_w['Gender'] = 'Women'

game_results_df = pd.concat([regular_m, tourney_m, regular_w, tourney_w], ignore_index=True)

game_results_df['Team1'] = np.minimum(game_results_df['WTeamID'], game_results_df['LTeamID'])
game_results_df['Team2'] = np.maximum(game_results_df['WTeamID'], game_results_df['LTeamID'])
game_results_df['Team1Win'] = (game_results_df['Team1'] == game_results_df['WTeamID']).astype(int)

augmented_data = pd.DataFrame()

for gender in ['Men', 'Women']:
    gender_elo_df = elo_df_men if gender == 'Men' else elo_df_women
    gender_seed_df = mens_seeds_df if gender == 'Men' else womens_seeds_df
    gender_results_df = game_results_df[game_results_df['Gender'] == gender].copy()

    gender_results_df = pd.merge(gender_results_df, gender_elo_df[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo1'}).drop(columns=['TeamID'])
    gender_results_df = pd.merge(gender_results_df, gender_elo_df[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo2'}).drop(columns=['TeamID'])
    gender_results_df['EloDiff'] = gender_results_df['Elo1'] - gender_results_df['Elo2']

    gender_results_df = pd.merge(gender_results_df, gender_seed_df[['Season', 'TeamID', 'SeedValue']], left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedValue': 'Seed1'}).drop(columns=['TeamID'])
    gender_results_df = pd.merge(gender_results_df, gender_seed_df[['Season', 'TeamID', 'SeedValue']], left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedValue': 'Seed2'}).drop(columns=['TeamID'])
    gender_results_df[['Seed1', 'Seed2']] = gender_results_df[['Seed1', 'Seed2']].fillna(16)
    gender_results_df['SeedDiff'] = gender_results_df['Seed1'] - gender_results_df['Seed2']

    gender_results_df['Target'] = gender_results_df['Team1Win']

    augmented_data = pd.concat([augmented_data, gender_results_df], ignore_index=True)

augmented_data.dropna(subset=['EloDiff', 'SeedDiff', 'Target'], inplace=True)
X_augmented = augmented_data[['EloDiff', 'SeedDiff']]
y_augmented = augmented_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

print("Missing values in X_train after imputation:", np.isnan(X_train_imputed).sum())
print("Missing values in X_test after imputation:", np.isnan(X_test_imputed).sum())
print("Class distribution in y_train after augmentation:")
print(y_train.value_counts())

model = LogisticRegression(random_state=42)
model.fit(X_train_imputed, y_train)

y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
brier_score = brier_score_loss(y_test, y_pred_proba)
print(f"Brier Score on Test Set: {brier_score}")


submission_df = load_csv(os.path.join(comp_dir, 'SampleSubmissionStage1.csv'))

def extract_game_info(id_str):
    parts = id_str.split('_')
    season = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])

    if teamID1 < 3000:
        gender = 'Men'
    else:
        gender = 'Women'
    return season, teamID1, teamID2, gender

submission_df[['Season', 'TeamID1', 'TeamID2', 'Gender']] = submission_df['ID'].apply(lambda x: pd.Series(extract_game_info(x)))

submission_df = pd.merge(submission_df, elo_df_men[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo1'}).drop(columns=['TeamID'])
submission_df = pd.merge(submission_df, elo_df_men[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo2'}).drop(columns=['TeamID'])

submission_df = pd.merge(submission_df, elo_df_women[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo1_w'}).drop(columns=['TeamID'])
submission_df = pd.merge(submission_df, elo_df_women[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo2_w'}).drop(columns=['TeamID'])

submission_df['Elo1'] = submission_df.apply(lambda row: row['Elo1'] if row['Gender'] == 'Men' else row['Elo1_w'], axis=1)
submission_df['Elo2'] = submission_df.apply(lambda row: row['Elo2'] if row['Gender'] == 'Men' else row['Elo2_w'], axis=1)
submission_df['EloDiff'] = submission_df['Elo1'] - submission_df['Elo2']

submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']], left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']], left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

submission_df[['SeedValue1', 'SeedValue2']] = submission_df[['SeedValue1', 'SeedValue2']].fillna(16)
submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']

submission_features = submission_df[['EloDiff', 'SeedDiff']]
submission_features_imputed = imputer.transform(submission_features)
submission_df['Pred'] = model.predict_proba(submission_features_imputed)[:, 1]

submission_df[['ID', 'Pred']].to_csv('/kaggle/working/submission.csv', index=False)

print("Submission File Preview:")
display(submission_df[['ID', 'Pred']].head())


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import ipywidgets as widgets
from IPython.display import display
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import chardet
from scipy.stats import linregress
import glob
import re
from sklearn.preprocessing import StandardScaler # ADD THIS LINE

sns.set_style("whitegrid")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
warnings.filterwarnings("ignore")

def load_csv(csv_file):
    with open(csv_file, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
    return pd.read_csv(csv_file, encoding=result["encoding"])

comp_dir = '/kaggle/input/march-machine-learning-mania-2025'

mens_teams_df = load_csv(os.path.join(comp_dir, 'MTeams.csv'))
womens_teams_df = load_csv(os.path.join(comp_dir, 'WTeams.csv'))

mens_seeds_df = load_csv(os.path.join(comp_dir, 'MNCAATourneySeeds.csv'))
womens_seeds_df = load_csv(os.path.join(comp_dir, 'WNCAATourneySeeds.csv'))

sample_submission_df = load_csv(os.path.join(comp_dir, 'SampleSubmissionStage1.csv'))

elo_df_men = pd.read_csv('mens_elo_rating.csv')
elo_df_women = pd.read_csv('womens_elo_rating.csv')


path = '/kaggle/input/march-machine-learning-mania-2025/**'
data = {p.split('/')[-1].split('.')[0] : pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}

season_dresults_m = data['MRegularSeasonDetailedResults']
tourney_dresults_m = data['MNCAATourneyDetailedResults']
season_dresults_w = data['WRegularSeasonDetailedResults']
tourney_dresults_w = data['WNCAATourneyDetailedResults']

season_dresults = pd.concat([season_dresults_m, season_dresults_w])
tourney_dresults = pd.concat([tourney_dresults_m, tourney_dresults_w])

games_detailed = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games_detailed.reset_index(drop=True, inplace=True)

games_detailed['ID'] = games_detailed.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games_detailed['IDTeams'] = games_detailed.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games_detailed['Team1'] = games_detailed.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games_detailed['Team2'] = games_detailed.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games_detailed['Target'] = (games_detailed['WTeamID'] == games_detailed['Team1']).astype(int) # ADDED TARGET VARIABLE HERE


c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']

gb = games_detailed.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
gb.columns = [''.join(c) + '_c_score' for c in gb.columns]


submission_df = load_csv(os.path.join(comp_dir, 'SampleSubmissionStage1.csv'))

def extract_game_info(id_str):
    parts = id_str.split('_')
    season = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])

    if teamID1 < 3000:
        gender = 'Men'
    else:
        gender = 'Women'
    return season, teamID1, teamID2, gender

submission_df[['Season', 'TeamID1', 'TeamID2', 'Gender']] = submission_df['ID'].apply(lambda x: pd.Series(extract_game_info(x)))
submission_df['IDTeams'] = submission_df.apply(lambda r: '_'.join(map(str, sorted([r['TeamID1'], r['TeamID2']]))), axis=1)

submission_df = pd.merge(submission_df, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')


submission_df = pd.merge(submission_df, elo_df_men[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo1'}).drop(columns=['TeamID'])
submission_df = pd.merge(submission_df, elo_df_men[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo2'}).drop(columns=['TeamID'])

submission_df = pd.merge(submission_df, elo_df_women[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo1_w'}).drop(columns=['TeamID'])
submission_df = pd.merge(submission_df, elo_df_women[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo2_w'}).drop(columns=['TeamID'])

submission_df['Elo1'] = submission_df.apply(lambda row: row['Elo1'] if row['Gender'] == 'Men' else row['Elo1_w'], axis=1)
submission_df['Elo2'] = submission_df.apply(lambda row: row['Elo2'] if row['Gender'] == 'Men' else row['Elo2_w'], axis=1)
submission_df['EloDiff'] = submission_df['Elo1'] - submission_df['Elo2']

def extract_seed_value(seed_str):
    try:
        return int(re.search(r'\d+', seed_str).group(0))
    except:
        return 16

mens_seeds_df['SeedValue'] = mens_seeds_df['Seed'].apply(extract_seed_value)
womens_seeds_df['SeedValue'] = womens_seeds_df['Seed'].apply(extract_seed_value)
seed_df = pd.concat([mens_seeds_df, womens_seeds_df], ignore_index=True)

submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']], left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']], left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

submission_df[['SeedValue1', 'SeedValue2']] = submission_df[['SeedValue1', 'SeedValue2']].fillna(16)
submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']


# Feature Selection
feature_cols = ['EloDiff', 'SeedDiff'] + [col for col in submission_df.columns if col.endswith('_c_score')]
X = submission_df[feature_cols]
#y = submission_df['Pred']
y = np.random.randint(0, 2, X.shape[0]) # Generate synthetic binary labels

# Imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

X_train, X_val, y_train, y_val = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)


# Initialize and train Logistic Regression model
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train, y_train)

y_pred_proba_val = model.predict_proba(X_val)[:, 1]

brier_score_val = brier_score_loss(y_val, y_pred_proba_val)
print(f"Validation Brier Score (Logistic Regression): {brier_score_val}")


# Feature Engineering for games_detailed (Elo, Seed, detailed stats)
games_detailed = pd.merge(games_detailed, elo_df_men[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo1'}).drop(columns=['TeamID'])
games_detailed = pd.merge(games_detailed, elo_df_men[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo2'}).drop(columns=['TeamID'])
games_detailed = pd.merge(games_detailed, elo_df_women[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo1_w'}).drop(columns=['TeamID'])
games_detailed = pd.merge(games_detailed, elo_df_women[['Season', 'TeamID', 'Rating_Last']], left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'Rating_Last': 'Elo2_w'}).drop(columns=['TeamID'])

games_detailed['Elo1'] = games_detailed.apply(lambda row: row['Elo1'] if row['Team1'] < 3000 else row['Elo1_w'], axis=1) # Assuming TeamIDs < 3000 are Men's
games_detailed['Elo2'] = games_detailed.apply(lambda row: row['Elo2'] if row['Team2'] < 3000 else row['Elo2_w'], axis=1) # Assuming TeamIDs < 3000 are Men's
games_detailed['EloDiff'] = games_detailed['Elo1'] - games_detailed['Elo2']

games_detailed = pd.merge(games_detailed, seed_df[['Season', 'TeamID', 'SeedValue']], left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])
games_detailed = pd.merge(games_detailed, seed_df[['Season', 'TeamID', 'SeedValue']], left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

games_detailed[['SeedValue1', 'SeedValue2']] = games_detailed[['SeedValue1', 'SeedValue2']].fillna(16)
games_detailed['SeedDiff'] = games_detailed['SeedValue1'] - games_detailed['SeedValue2']

games_detailed = pd.merge(games_detailed, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')


feature_cols_game = ['EloDiff', 'SeedDiff'] + [col for col in games_detailed.columns if col.endswith('_c_score')] # Feature columns for game data
X_game = games_detailed[feature_cols_game] # Features from games_detailed
y_game = games_detailed['Target'] # Target variable from games_detailed

imputer_game = SimpleImputer(strategy='mean') # Separate imputer for game data
X_game_imputed = imputer_game.fit_transform(X_game)

scaler_game = StandardScaler() # Separate scaler for game data
X_game_scaled = scaler_game.fit_transform(X_game_imputed)
X_game_scaled_df = pd.DataFrame(X_game_scaled, columns=feature_cols_game)

X_train_game, X_val_game, y_train_game, y_val_game = train_test_split(X_game_scaled_df, y_game, test_size=0.2, random_state=42)

print("Shape of X_train_game:", X_train_game.shape)
print("Shape of X_val_game:", X_val_game.shape)
print("Shape of y_train_game:", y_train_game.shape)
print("Shape of y_val_game:", y_val_game.shape)

model_game = LogisticRegression(random_state=42, solver='liblinear')
model_game.fit(X_train_game, y_train_game)

y_pred_proba_val_game = model_game.predict_proba(X_val_game)[:, 1]

brier_score_val_game = brier_score_loss(y_val_game, y_pred_proba_val_game)
print(f"Validation Brier Score (Logistic Regression on game data): {brier_score_val_game}")


final_model_game = LogisticRegression(random_state=42, solver='liblinear')
final_model_game.fit(X_game_scaled_df, y_game)


X_submission = submission_df[feature_cols_game]

X_submission_imputed = imputer_game.transform(X_submission)

X_submission_scaled = scaler_game.transform(X_submission_imputed)
X_submission_scaled_df = pd.DataFrame(X_submission_scaled, columns=feature_cols_game)


submission_pred_proba = final_model_game.predict_proba(X_submission_scaled_df)[:, 1]
submission_df['Pred'] = submission_pred_proba


submission_final_df = submission_df[['ID', 'Pred']]
submission_final_df['Pred'] = np.clip(submission_final_df['Pred'], 0.001, 0.999)
submission_final_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created.")
print(submission_final_df.head())

