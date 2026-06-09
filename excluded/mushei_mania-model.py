import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import optuna
from sklearn.model_selection import KFold
from sklearn.base import clone
import glob
import warnings 

warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


Path = "/kaggle/input/march-machine-learning-mania-2025/*.csv"
data = {x.split('/')[-1].split('.')[0] : pd.read_csv(x, encoding="latin-1") for x in glob.glob(Path)}


MTeams = data['MTeams']
WTeams = data['WTeams']

MTeamSpellings = data['MTeamSpellings']
WTeamSpellings = data['WTeamSpellings']

MRegularSeasonCompactResults = data['MRegularSeasonCompactResults']
WRegularSeasonCompactResults = data['WRegularSeasonCompactResults']

MRegularSeasonDetailedResults = data['MRegularSeasonDetailedResults']
WRegularSeasonDetailedResults = data['WRegularSeasonDetailedResults']

MNCAATourneyCompactResults = data['MNCAATourneyCompactResults']
WNCAATourneyCompactResults = data['WNCAATourneyCompactResults']

MNCAATourneyDetailedResults = data['MNCAATourneyDetailedResults']
WNCAATourneyDetailedResults = data['WNCAATourneyDetailedResults']

MGameCities = data['MGameCities']
WGameCities = data['WGameCities']

MSeasons = data['MSeasons']
WSeasons = data['WSeasons']

MNCAATourneySeeds = data['MNCAATourneySeeds']
WNCAATourneySeeds = data['WNCAATourneySeeds']

Cities = data['Cities']
SampleSub = data['SampleSubmissionStage2']

del data


Teams = pd.concat([MTeams, WTeams])
TeamSpelling = pd.concat([MTeamSpellings, WTeamSpellings])
TeamSpelling = TeamSpelling.groupby(by="TeamID", as_index=False)['TeamNameSpelling'].count()
TeamSpelling.columns = ['TeamID', 'TeamNameCount']
Teams = pd.merge(Teams, TeamSpelling, how='left', on=['TeamID'])
del TeamSpelling

SeasonCompactResults = pd.concat([MRegularSeasonCompactResults, WRegularSeasonCompactResults])
SeasonDetailedResults = pd.concat([MRegularSeasonDetailedResults, WRegularSeasonDetailedResults])

TourneyCompactResults = pd.concat([MNCAATourneyCompactResults, WNCAATourneyCompactResults])
TourneyDetailedResults = pd.concat([MNCAATourneyDetailedResults, WNCAATourneyDetailedResults])

GameCities = pd.concat([MGameCities, WGameCities])
Seasons = pd.concat([MSeasons,WSeasons])
Seeds = pd.concat([MNCAATourneySeeds, WNCAATourneySeeds])

Seeds = {'_'.join(map(str, [int (x1), x2])): int(v[1:3])  for x1, v, x2 in Seeds[['Season', 'Seed', 'TeamID']].values}

SeasonCompactResults['ST'] = 'S'
SeasonDetailedResults['ST'] = 'S'
TourneyCompactResults['ST'] = 'T'
TourneyDetailedResults['ST'] = 'T'

Games = pd.concat((SeasonDetailedResults,TourneyDetailedResults), axis=0, ignore_index=True)
Games.reset_index(drop=True, inplace=True)
Games['WLoc'] = Games['WLoc'].map({'H': 0, 'A': 1, 'N': 2})


Games['ID'] = Games.apply(lambda x: '_'.join(map(str, [x['Season']]+sorted([x['WTeamID'], x['LTeamID']]))), axis=1)
Games['IDTeams'] = Games.apply(lambda x: '_'.join(map(str, sorted([x['WTeamID'], x['LTeamID']]))), axis=1)
Games['Team1'] = Games.apply(lambda x:  sorted([x['WTeamID'], x['LTeamID']])[0], axis=1)
Games['Team2'] = Games.apply(lambda x: sorted([x['WTeamID'], x['LTeamID']])[1], axis=1)
Games['IDTeam1'] = Games.apply(lambda x: '_'.join(map(str, [x['Season'], x['Team1']])), axis=1)
Games['IDTeam2'] = Games.apply(lambda x: '_'.join(map(str, [x['Season'], x['Team2']])), axis=1)

Games['Team1Seed'] = Games['IDTeam1'].map(Seeds).fillna(0)
Games['Team2Seed'] = Games['IDTeam2'].map(Seeds).fillna(0)

# Games['Score_Difference'] = Games['WScore'] - Games['LScore']
Games['Pred'] = Games.apply(lambda x: 1.0 if sorted([x['WTeamID'], x['LTeamID']])[0] == x['WTeamID'] else 0.0 , axis=1)
# Games['ScoreDifferenceNorm'] = Games.apply(lambda x: x['Score_Difference'] * -1 if x['Pred'] == 0.0 
#                                            else x['Score_Difference'], axis=1)
Games['SeedDifference'] = Games['Team1Seed'] - Games['Team2Seed']

columns = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF']
agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
gb = Games.groupby(by=('IDTeams')).agg({x: agg for x in columns }).reset_index()
gb.columns = [''.join(x) + '_score' for x in gb.columns]
Games = Games[Games['ST'] == 'T']


SampleSub['WLoc'] = 3
SampleSub['Season'] = SampleSub['ID'].map(lambda x: x.split('_')[0])
SampleSub['Season'] = SampleSub['Season'].astype(int)
SampleSub['Team1'] = SampleSub['ID'].map(lambda x: x.split('_')[1])
SampleSub['Team2'] = SampleSub['ID'].map(lambda x: x.split('_')[2])
SampleSub['IDTeams'] = SampleSub.apply(lambda x: '_'.join(map(str, [x['Team1'], x['Team2']])), axis=1)
SampleSub['IDTeam1'] = SampleSub.apply(lambda x: '_'.join(map(str, [x['Season'], x['Team1']])), axis=1)
SampleSub['IDTeam2'] = SampleSub.apply(lambda x: '_'.join(map(str, [x['Season'], x['Team2']])), axis=1)
SampleSub['Team1Seed'] = SampleSub['IDTeam1'].map(Seeds).fillna(0)
SampleSub['Team2Seed'] = SampleSub['IDTeam2'].map(Seeds).fillna(0)
SampleSub['SeedDifference'] = SampleSub['Team1Seed'] - SampleSub['Team2Seed']


Games = pd.merge(Games,gb, how='left', left_on='IDTeams', right_on='IDTeams_score')
SampleSub = pd.merge(SampleSub, gb, how='left', left_on='IDTeams', right_on='IDTeams_score')
Games = Games.fillna(-1)
SampleSub = SampleSub.fillna(-1)


cols = [c for c in Games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
                                             'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm',
                                             'WLoc'] + columns]

train = Games[cols]
X = train.drop(columns=['Pred'], errors='ignore')
Y = Games['Pred']
test = SampleSub[cols] 


Scaler = StandardScaler()
X_scaled = Scaler.fit_transform(X)
X_test = Scaler.transform(test)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        # Fixed parameters
        'criterion': 'squared_error',
        'bootstrap': True,
        'verbose': 0,
        'random_state': 42
    }

    RandomForestModel = RandomForestRegressor(**params)
    Error = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kfold.split(X_scaled, Y):
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

        model = clone(RandomForestModel)
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        error = mean_squared_error(Y_test, pred)
        Error.append(error)

    return np.mean(Error)


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)
# print(study.best_params)


params = {'n_estimators': 1438, 'max_depth': 11,
'min_samples_split': 8, 'min_samples_leaf': 27, 'max_features': 0.6005630225311747}

Model = RandomForestRegressor(random_state=42, **params)
Model.fit(X_scaled, Y)
pred = Model.predict(X_test)

Result = pd.DataFrame({
    'ID': SampleSub['ID'],
    'Pred': pred
})
Result.to_csv('Predictions.csv', index=False)
print("Submission File Saved Successfully")

