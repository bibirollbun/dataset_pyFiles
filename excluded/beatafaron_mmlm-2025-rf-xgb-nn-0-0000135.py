
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import numpy as np
import pandas as pd
from sklearn import *
import glob

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


seedsS = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])
slots = pd.concat([data['MNCAATourneySlots'], data['WNCAATourneySlots']])
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


games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

games['Team1Seed'] = games['IDTeam1'].map(seeds)#.fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds)#.fillna(0)

games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed'] 
games['Points'] = round(0.5 + (0.033 * games['SeedDiff']),2)*100


games = games[games['Season']>2022]
seedsS = seedsS[seedsS['Season']==2025]


mat = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
mat['IDTeams'] = mat.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
mat = mat[mat['Season']==2025]
mat.drop('Season', axis = 1, inplace=True)


sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['Season'].astype(int)
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
sub['Team1Seed'] = sub['IDTeam1'].map(seeds)#.fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds)#.fillna(0)
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed'] 
sub['Points'] = round(0.5 + (0.033 * sub['SeedDiff']),2)*100

sub = pd.merge(sub,mat, how='left', left_on='IDTeams', right_on='IDTeams')


c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF','Points']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
gb.columns = [''.join(c) + '_c_score' for c in gb.columns]

games = games[games['ST']=='T']

games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')


col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred','Pred1','Points', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc','IDTeams_c_score'] + c_score_col]


#Selected features with importance above a threshold = mean
#feature selecting Code two cells below

col = ['Team1Seed', 'Team2Seed', 'SeedDiff', 'WFGMsum_c_score',
       'WFGMmean_c_score', 'WFGMmedian_c_score', 'WFGMmax_c_score',
       'WFGMmin_c_score', 'WFGMstd_c_score', 'WFGAsum_c_score',
       'WFGAmean_c_score', 'WFGAmedian_c_score', 'WFGAmax_c_score',
       'WFGAstd_c_score', 'WFGA3max_c_score', 'WFTMmedian_c_score',
       'WFTMmax_c_score', 'WFTMstd_c_score', 'WFTAsum_c_score',
       'WFTAmedian_c_score', 'WFTAmax_c_score', 'WFTAmin_c_score',
       'WFTAstd_c_score', 'WORsum_c_score', 'WORmin_c_score',
       'WDRmean_c_score', 'WDRmedian_c_score', 'WDRmax_c_score',
       'WDRmin_c_score', 'WDRstd_c_score', 'WAstsum_c_score',
       'WAstmean_c_score', 'WAstmedian_c_score', 'WAstmax_c_score',
       'WAstmin_c_score', 'WAststd_c_score', 'WTOstd_c_score',
       'WStlmean_c_score', 'WStlmax_c_score', 'WStlstd_c_score',
       'WBlkstd_c_score', 'LFGMmean_c_score', 'LFGMmedian_c_score',
       'LFGMstd_c_score', 'LFGAmax_c_score', 'LFGAstd_c_score',
       'LFGM3std_c_score', 'LFGA3mean_c_score', 'LFGA3median_c_score',
       'LFGA3std_c_score', 'LFTMsum_c_score', 'LFTMmean_c_score',
       'LFTMmedian_c_score', 'LFTMstd_c_score', 'LFTAmean_c_score',
       'LFTAmedian_c_score', 'LFTAmin_c_score', 'LFTAstd_c_score',
       'LDRmean_c_score', 'LDRmedian_c_score', 'LDRmin_c_score',
       'LDRstd_c_score', 'LAstsum_c_score', 'LAstmedian_c_score',
       'LAstmax_c_score', 'LAstmin_c_score', 'LAststd_c_score',
       'LTOmin_c_score', 'LTOstd_c_score', 'LStlmean_c_score',
       'LStlmax_c_score', 'LStlmin_c_score', 'LStlstd_c_score',
       'LBlkmean_c_score', 'LPFsum_c_score', 'LPFmedian_c_score',
       'LPFmax_c_score', 'Pointssum_c_score', 'Pointsmean_c_score',
       'Pointsmedian_c_score', 'Pointsmax_c_score', 'Pointsmin_c_score',
       'Pointsstd_c_score']


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import KNNImputer

# Initialize KNN Imputer with k=5 (can be tuned)
imputer = KNNImputer(n_neighbors=5) 
scaler = StandardScaler()

X = games[col]
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

sub_X = sub[col]
sub_X_imputed = imputer.transform(sub_X)
sub_X_scaled = scaler.transform(sub_X_imputed)



y_train= games['Pred']
X_train_scaled = X_scaled


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
# Train Random Forest for feature selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Select features with importance above a threshold
model = SelectFromModel(rf, prefit=True, threshold="mean")
X_train_selected = model.transform(X_train_scaled)
#X_test_selected = model.transform(X_test_scaled)

print(f"Original features: {X.shape[1]}, Selected features: {X_train_selected.shape[1]}")

# Get the mask of selected features
selected_features_mask = model.get_support()

# Get the names of the selected features
selected_features_names = np.array(X.columns)[selected_features_mask]
selected_features_names


reg = RandomForestRegressor(n_estimators=500, random_state=42)
reg.fit(X_scaled, games['Pred'])

pred = reg.predict(X_scaled).clip(0.001, 0.999)

print(f'Log Loss: {log_loss(games["Pred"], pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(games["Pred"], pred)}')
print(f'Brier Score: {brier_score_loss(games["Pred"], pred)}')


sub['Pred'] = reg.predict(sub_X_scaled).clip(0.001, 0.999)
sub[['ID', 'Pred']].to_csv('submission_rf.csv', index=False)




from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=6, random_state=42)
xgb.fit(X_scaled, games['Pred'])

pred = xgb.predict(X_scaled).clip(0.001, 0.999)

print(f'Log Loss: {log_loss(games["Pred"], pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(games["Pred"], pred)}')
print(f'Brier Score: {brier_score_loss(games["Pred"], pred)}')

sub['Pred'] = xgb.predict(sub_X_scaled).clip(0.001, 0.999)
sub[['ID', 'Pred']].to_csv('submission_xgb.csv', index=False)



from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=10, random_state=42)
xgb.fit(X_scaled, games['Pred'])

pred = xgb.predict(X_scaled).clip(0.001, 0.999)

print(f'Log Loss: {log_loss(games["Pred"], pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(games["Pred"], pred)}')
print(f'Brier Score: {brier_score_loss(games["Pred"], pred)}')

sub['Pred'] = xgb.predict(sub_X_scaled).clip(0.001, 0.999)
sub[['ID', 'Pred']].to_csv('submission_xgb2.csv', index=False)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss

# Step 1: Define the Neural Network Model
model = keras.Sequential([
    layers.Input(shape=(X_scaled.shape[1],)),  # Input layer matching feature count
    layers.Dense(128, activation='relu'),      # Hidden Layer 1
    layers.Dense(64, activation='relu'),       # Hidden Layer 2
    layers.Dense(32, activation='relu'),       # Hidden Layer 3
    layers.Dense(1, activation='sigmoid')      # Output layer with sigmoid activation for probability output
])

# Step 2: Compile the Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Learning rate matches XGBoost setup
    loss='binary_crossentropy',  # Equivalent to log loss for probability outputs
    metrics=['mae']
)

# Step 3: Train the Model
model.fit(X_scaled, games['Pred'], epochs=50, batch_size=32, verbose=1)

# Step 4: Make Predictions and Clip
pred = model.predict(X_scaled).clip(0.001, 0.999)

# Step 5: Evaluate Model
print(f'Log Loss: {log_loss(games["Pred"], pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(games["Pred"], pred)}')
print(f'Brier Score: {brier_score_loss(games["Pred"], pred)}')

# Step 6: Make Submission Predictions
sub['Pred'] = model.predict(sub_X_scaled).clip(0.001, 0.999)
sub[['ID', 'Pred']].to_csv('submission_nn.csv', index=False)


