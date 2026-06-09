import chardet
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
import os
import optuna
import warnings
from sklearn import metrics  
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.isotonic import IsotonicRegression


w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')


seed_df.head()


submission_df.head()


def extract_game_info(id_str):
    # Extract year and team_ids
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

def extract_seed_value(seed_str):
    # Extract seed value
    try:
        return int(seed_str[1:])
    # Set seed to 16 for unselected teams and errors
    except ValueError:
        return 16

# Reformat the data
submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(extract_game_info).tolist()
seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

# Merge seed information for TeamID1
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed information for TeamID2
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])


# View files

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
        print(f"The specified directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Permission error accessing directory '{directory}'.")
    return file_list, dir_list
    
input_dir = '/kaggle/input'
file_list, dir_list = get_comp_files_and_dirs(input_dir)


comp_dir = '/kaggle/input/march-machine-learning-mania-2025'
data_section_1_mens_list = [
    'MTeams.csv',
    'MSeasons.csv',
    'MNCAATourneySeeds.csv',
    'MRegularSeasonCompactResults.csv',
    'MNCAATourneyCompactResults.csv'
]
data_section_1_womens_list = [
    'WTeams.csv',
    'WSeasons.csv',
    'WNCAATourneySeeds.csv',
    'WRegularSeasonCompactResults.csv',
    'WNCAATourneyCompactResults.csv',
]
sample_submission = 'SampleSubmissionStage1.csv'
data_section_2_mens_list = [
    'MRegularSeasonDetailedResults.csv',
    'MNCAATourneyDetailedResults.csv',
]
data_section_2_womens_list = [
    'WRegularSeasonDetailedResults.csv',
    'WNCAATourneyDetailedResults.csv',
]
data_section_3_mens_list = [
    'Cities.csv',
    'MGameCities.csv',
]
data_section_3_womens_list = [
    'Cities.csv',
    'WGameCities.csv',
]
data_section_4_list = [
    'MMasseyOrdinals.csv'
]
data_section_5_list = [
    'MTeamCoaches.csv'
]


# Tools
def load_csv(csv_file):
    with open(csv_file, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
    
    # print(result)  # Check detected encoding
    df = pd.read_csv(csv_file, encoding=result["encoding"])
    return df


mens_df = load_csv(os.path.join(comp_dir, 'MTeams.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WTeams.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())


# Find unique values in Team Names
series1 = mens_df['TeamName']
series2 = womens_df['TeamName']
unique_in_series1 = series1[~series1.isin(series2)]

# Values in series2 but not in series1
unique_in_series2 = series2[~series2.isin(series1)]

# Combine results
unique_values = pd.concat([unique_in_series1, unique_in_series2])

print(unique_values)


mens_df = load_csv(os.path.join(comp_dir, 'MNCAATourneySeeds.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WNCAATourneySeeds.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())


mens_df = load_csv(os.path.join(comp_dir, 'MRegularSeasonCompactResults.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WRegularSeasonCompactResults.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())


# Settings
sns.set_style("whitegrid")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
warnings.filterwarnings("ignore")


# Plot the durations of each team
for gender in ['Mens']:
    print(f"Investigating {gender} Team")
    df = load_csv(os.path.join(comp_dir, 'MTeams.csv'))

    # Calculate the width of the bar
    df['Widths'] = df['LastD1Season'] - df['FirstD1Season']
    

    # # Plot bars
    fig, ax = plt.subplots(figsize=(15,60))
    sns.barplot(
        data=df,
        y="TeamName",        # Y-axis (categorical variable)
        x="Widths",          # X-axis (bar length)
        hue=None,            # No grouping
        orient="h",          # Horizontal bars
        color="blue",        # Bar color
        edgecolor="black",   # Border color
        ax=ax                # Use the existing axis
    )
    # Offset each bar by the start
    for i, (start, width) in enumerate(zip(df["FirstD1Season"], df["Widths"])):
        ax.patches[i].set_x(start)  # Shift bar to start position
    ax.set_xlim(df["FirstD1Season"].min(), df["LastD1Season"].max())  # Fit all bars correctly
    # Show x-axis at the top as well
    ax.xaxis.set_ticks_position("both")  # Show ticks on both top and bottom
    ax.xaxis.set_label_position("top")   # Move x-axis label to the top
    ax.tick_params(axis="x", which="both", labeltop=True, labelbottom=True)  # Show tick labels at the top
    ax.spines["top"].set_visible(True)   # Show the top spine (border)
    ax.set_xlabel("")
    plt.title('Durations of Mens NCAA Teams')
    plt.show()


# Look at distributions of winning and losing scores across seasons
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    df['Score Difference']  = df['WScore'] - df['LScore']

    plt.figure(figsize=(12,8))
    sns.boxplot(x='Season', y='WScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Winning Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    sns.boxplot(x='Season', y='LScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Losing Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    sns.boxplot(x='Season', y='Score Difference', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Score Difference Distributions')
    plt.tight_layout()
    plt.show()


path = '/kaggle/input/march-machine-learning-mania-2025/**'
data = {p.split('/')[-1].split('.')[0] : pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}


teams = pd.concat([data['MTeams'], data['WTeams']])
teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])
teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
teams_spelling.columns = ['TeamID', 'TeamNameCount']
teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
del teams_spelling


season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])
seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
gcities = pd.concat([data['MGameCities'], data['WGameCities']])
seasons = pd.concat([data['MSeasons'], data['WSeasons']])

seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}
cities = data['Cities']
sub = data['SampleSubmissionStage2']
del data


season_cresults['ST'] = 'S'
season_dresults['ST'] = 'S'
tourney_cresults['ST'] = 'T'
tourney_dresults['ST'] = 'T'
#games = pd.concat((season_cresults, tourney_cresults), axis=0, ignore_index=True)
games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed'] 
games = games.fillna(-1)

c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
gb.columns = [''.join(c) + '_c_score' for c in gb.columns]

games = games[games['ST']=='T']

sub['WLoc'] = 3
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['Season'].astype(int)
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed'] 
sub = sub.fillna(-1)

games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc'] + c_score_col]


imputer = SimpleImputer(strategy='mean')  
scaler = StandardScaler()


X = games[col].fillna(-1)
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

y = games['Pred']


params = {'n_estimators': 296, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 20}
clf = ensemble.RandomForestRegressor(**params) #linear_model.LinearRegression()
clf.fit(X_scaled, y)
pred = clf.predict(X_scaled).clip(0.0001,0.9999)
print('Log Loss:', metrics.log_loss(games['Pred'], pred))


X = sub[col].fillna(-1)
X_imputed = imputer.transform(X)
X_scaled = scaler.transform(X_imputed)
preds = clf.predict(X_scaled).clip(0.0001,0.9999)
        
# Optionally, apply the same isotonic calibration as above (using training fit)
ir = IsotonicRegression(out_of_bounds='clip')
        
# Refit calibration on training predictions for consistency:
X_train = imputer.fit_transform(games[col].fillna(-1))
X_train_scaled = scaler.fit_transform(X_train)
train_preds = clf.predict(X_train_scaled).clip(0.0001, 0.9999)
ir.fit(train_preds, games['Pred'])
preds_cal = ir.transform(preds)


sub['Pred'] = preds_cal
sub[['ID', 'Pred']].to_csv('submission.csv', index=False)

