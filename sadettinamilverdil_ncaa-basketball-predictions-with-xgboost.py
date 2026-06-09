import numpy as np
import pandas as pd

from sklearn import *
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")


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
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])
tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
slots = pd.concat([data['MNCAATourneySlots'], data['WNCAATourneySlots']])
seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
seeds_dict = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}
gcities = pd.concat([data['MGameCities'], data['WGameCities']])
seasons = pd.concat([data['MSeasons'], data['WSeasons']])


seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}
cities = data['Cities']
sub = data['SampleSubmissionStage1']
del data

season_cresults['ST'] = 'S'
season_dresults['ST'] = 'S'
tourney_cresults['ST'] = 'T'
tourney_dresults['ST'] = 'T'

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


tourney_dresults['WSeed'] = tourney_dresults.apply(lambda r: seeds_dict.get(f"{r['Season']}_{r['WTeamID']}", np.nan), axis=1)
tourney_dresults['LSeed'] = tourney_dresults.apply(lambda r: seeds_dict.get(f"{r['Season']}_{r['LTeamID']}", np.nan), axis=1)
tourney_dresults['Seed_Diff'] = tourney_dresults['WSeed'] - tourney_dresults['LSeed']

seed_win_counts = tourney_dresults.groupby(['WSeed', 'LSeed']).size().reset_index(name='WinCount')
plt.figure(figsize=(12, 8))
heatmap_data = seed_win_counts.pivot(index='LSeed', columns='WSeed', values='WinCount')
sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".0f", linewidths=0.5)

plt.title("Win Distribution by Seed Matchups")
plt.xlabel("Winning Seed")
plt.ylabel("Losing Seed")
plt.show()


plt.figure(figsize=(12,6))
sns.histplot(season_dresults['WScore'], bins=30, kde=True, color='blue', label='Winning Score')
sns.histplot(season_dresults['LScore'], bins=30, kde=True, color='red', label='Losing Score')
plt.legend()
plt.title("Score Distribution of Winning & Losing Teams")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()


seed_win_rate = tourney_dresults.groupby('WSeed').size() / tourney_dresults.groupby('WSeed').size().sum() * 100
plt.figure(figsize=(12, 6))
sns.barplot(x=seed_win_rate.index, y=seed_win_rate.values, palette="viridis")
plt.title("Win Rate by Seed (%)")
plt.xlabel("Seed Number")
plt.ylabel("Win Percentage")
plt.show()


plt.figure(figsize=(12, 6))
sns.scatterplot(x=tourney_dresults['Seed_Diff'], y=tourney_dresults['WSeed'], alpha=0.6)
plt.title("Seed Difference vs. Winning Seed")
plt.xlabel("Seed Difference (Winning - Losing)")
plt.ylabel("Winning Seed")
plt.axvline(0, linestyle="--", color="red", alpha=0.7)  # Highlight equal seeds
plt.show()


tourney_dresults['Upset'] = (tourney_dresults['Seed_Diff'] > 0).astype(int)  # 1 if lower seed wins
upset_rate_per_year = tourney_dresults.groupby('Season')['Upset'].mean() * 100
plt.figure(figsize=(12, 6))
sns.lineplot(x=upset_rate_per_year.index, y=upset_rate_per_year.values, marker="o", color="darkred")
plt.title("Upset Rate Over the Years (Lower Seed Wins)")
plt.xlabel("Season")
plt.ylabel("Upset Percentage (%)")
plt.show()


plt.figure(figsize=(12, 6))
tourney_dresults_melted = tourney_dresults.melt(id_vars=['Season'], value_vars=['WScore', 'LScore'], 
                                                var_name="Winner/Loser", value_name="Score")
sns.boxplot(data=tourney_dresults_melted, x='Season', y='Score', hue="Winner/Loser", 
            palette={"WScore": "blue", "LScore": "red"})  
plt.title("Winning & Losing Score Distribution Over Years")
plt.xlabel("Season")
plt.ylabel("Score")
plt.xticks(rotation=90)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ["Winning Score", "Losing Score"], title="Score Type")
plt.show()


tourney_dresults['Score_Diff'] = tourney_dresults['WScore'] - tourney_dresults['LScore']
score_diff_trend = tourney_dresults.groupby('Season')['Score_Diff'].mean()
plt.figure(figsize=(12, 6))
sns.lineplot(x=score_diff_trend.index, y=score_diff_trend.values, marker="o", color="purple")
plt.title("Average Winning Score Difference Over Years")
plt.xlabel("Season")
plt.ylabel("Winning Margin")
plt.show()


tourney_dresults = tourney_dresults.merge(teams, left_on='WTeamID', right_on='TeamID', how='left')
tourney_dresults = tourney_dresults.rename(columns={'TeamName': 'WinningTeam'})
tourney_dresults = tourney_dresults.merge(teams, left_on='LTeamID', right_on='TeamID', how='left')
tourney_dresults = tourney_dresults.rename(columns={'TeamName': 'LosingTeam'})
team_win_counts = tourney_dresults['WinningTeam'].value_counts().head(15)
plt.figure(figsize=(12, 6))
sns.barplot(x=team_win_counts.values, y=team_win_counts.index, palette="magma")
plt.title("Top 15 Most Successful Teams (Based on Tournament Wins)")
plt.xlabel("Number of Wins")
plt.ylabel("Team Name")
plt.show()


cinderella_teams = tourney_dresults[tourney_dresults['WSeed'] >= 10]['WinningTeam'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=cinderella_teams.values, y=cinderella_teams.index, palette="Blues_r")
plt.title("Cinderella Stories - Most Wins by Low Seeded Teams")
plt.xlabel("Number of Wins")
plt.ylabel("Team Name")
plt.show()


imputer = SimpleImputer(strategy='mean')  
scaler = StandardScaler()

X = games[col].fillna(-1)
missing_cols = set(col) - set(sub.columns)
for c in missing_cols:
    sub[c] = 0

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

xgb = XGBRegressor(n_estimators=5000,device="gpu", learning_rate=0.03, max_depth=6, random_state=42)
xgb.fit(X_scaled, games['Pred'])

pred = xgb.predict(X_scaled).clip(0.001, 0.999)

print(f'Log Loss: {log_loss(games["Pred"], pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(games["Pred"], pred)}')
print(f'Brier Score: {brier_score_loss(games["Pred"], pred)}')

cv_scores = cross_val_score(xgb, X_scaled, games['Pred'], cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validated MSE: {-cv_scores.mean()}')


sub_X = sub[col].fillna(-1)
sub_X_imputed = imputer.transform(sub_X)
sub_X_scaled = scaler.transform(sub_X_imputed)

sub['Pred'] = xgb.predict(sub_X_scaled).clip(0.001, 0.999)
sub[['ID', 'Pred']].to_csv('submission.csv', index=False)

