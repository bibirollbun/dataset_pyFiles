import numpy as np
import pandas as pd

import xgboost as xgb
import pickle
import os


mncaaseeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
wncaaseeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')


mncaaseeds['Zone'] = mncaaseeds['Seed'].str[0]
mncaaseeds['Seed'] = np.int64(mncaaseeds['Seed'].str[1:3])


wncaaseeds['Zone'] = wncaaseeds['Seed'].str[0]
wncaaseeds['Seed'] = np.int64(wncaaseeds['Seed'].str[1:3])


mseason = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv')
wseason = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WSeasons.csv')


# Merge on 'Season'
merged_df = mncaaseeds.merge(mseason, on="Season", how="left")

# Use apply with loc to dynamically select the correct region
merged_df["Region"] = merged_df.apply(lambda row: row["Region" + row["Zone"]], axis=1)

# Select required columns
m_seed_region = merged_df[["Season", "Seed", "TeamID", "Region"]]


# Merge on 'Season'
merged_df = wncaaseeds.merge(wseason, on="Season", how="left")

# Use apply with loc to dynamically select the correct region
merged_df["Region"] = merged_df.apply(lambda row: row["Region" + row["Zone"]], axis=1)

# Select required columns
w_seed_region = merged_df[["Season", "Seed", "TeamID", "Region"]]


seed_region = pd.concat([m_seed_region, w_seed_region])


mrsdr = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
wrsdr = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv')


rsdr = pd.concat([mrsdr,wrsdr])


rsdr.drop(['DayNum','WLoc'], axis= 1, inplace = True)


df_win = rsdr.copy()
df_lose = rsdr.copy()

STATS = ["mean","std","count","nunique","median","min","max","skew"]

# Rename columns for winning team stats
df_win = df_win.rename(columns={
    'WTeamID': 'TeamID'
}).drop(columns=['LTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
                 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'])

df_win_avg = df_win.groupby(['Season', 'TeamID']).agg(STATS).reset_index()
df_win_avg.columns = [f'{col}_{stat}' if col not in ['TeamID','Season'] else col for col, stat in df_win_avg.columns]

# Rename columns for losing team stats
df_lose = df_lose.rename(columns={
    'LTeamID': 'TeamID'
}).drop(columns=['WTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
                 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'])

df_lose_avg = df_lose.groupby(['Season', 'TeamID']).agg(STATS).reset_index()
df_lose_avg.columns = [f'{col}_{stat}' if col not in ['TeamID','Season'] else col for col, stat in df_lose_avg.columns]

# Merge win and loss stats on Season and TeamID
df_season_stats = pd.merge(df_win_avg, df_lose_avg, on=['Season', 'TeamID'], how='outer')

# Sort by TeamID
df_season_stats = df_season_stats.sort_values(by=['TeamID','Season']).reset_index(drop=True)

df_season_stats = df_season_stats.fillna(0)

# Rename all columns except the first two
df_season_stats.columns = df_season_stats.columns[:2].tolist() + [col+'_season' for col in df_season_stats.columns[2:]]


df_season_stats.head()


mntdr = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')
wntdr = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyDetailedResults.csv')


ntdr = pd.concat([mntdr, wntdr])


ntdr.drop(['DayNum','WLoc'], axis= 1, inplace = True)


df_win = ntdr.copy()
df_lose = ntdr.copy()

STATS = ["mean","std","count","nunique","median","min","max","skew"]

# Rename columns for winning team stats
df_win = df_win.rename(columns={
    'WTeamID': 'TeamID'
}).drop(columns=['LTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
                 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'])

df_win_avg = df_win.groupby(['Season', 'TeamID']).agg(STATS).reset_index()
df_win_avg.columns = [f'{col}_{stat}' if col not in ['TeamID','Season'] else col for col, stat in df_win_avg.columns]

# Rename columns for losing team stats
df_lose = df_lose.rename(columns={
    'LTeamID': 'TeamID'
}).drop(columns=['WTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
                 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'])

df_lose_avg = df_lose.groupby(['Season', 'TeamID']).agg(STATS).reset_index()
df_lose_avg.columns = [f'{col}_{stat}' if col not in ['TeamID','Season'] else col for col, stat in df_lose_avg.columns]

# Merge win and loss stats on Season and TeamID
df_ncaa_stats = pd.merge(df_win_avg, df_lose_avg, on=['Season', 'TeamID'], how='outer')

# Sort by TeamID
df_ncaa_stats = df_ncaa_stats.sort_values(by=['TeamID','Season']).reset_index(drop=True)

df_ncaa_stats = df_ncaa_stats.fillna(0)

# Rename all columns except the first two
df_ncaa_stats.columns = df_ncaa_stats.columns[:2].tolist() + [col+'_ncaa' for col in df_ncaa_stats.columns[2:]]


df_ncaa_stats.head()


mctg = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MConferenceTourneyGames.csv')
wctg = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WConferenceTourneyGames.csv')
ctg = pd.concat([mctg, wctg])


win_count = ctg.groupby(['Season', 'WTeamID','ConfAbbrev']).size().reset_index(name='ConfWinCount')
win_count.rename(columns={'WTeamID':'TeamID'}, inplace= True)
lose_count = ctg.groupby(['Season', 'LTeamID','ConfAbbrev']).size().reset_index(name='ConfLoseCount')
lose_count.rename(columns={'LTeamID':'TeamID'}, inplace = True)
conf_count = win_count.merge(lose_count, on=['Season','TeamID','ConfAbbrev'], how='outer').fillna(0)
conf_count.head()


ncaa_diff = ntdr[['Season', 'WTeamID']]
ncaa_diff = ncaa_diff.rename(columns={'WTeamID':'TeamID'})
ncaa_diff['ScoreDiff'] = ntdr['WScore'] - ntdr['LScore']
ncaa_diff['Season'] = ncaa_diff['Season'] + 1# so that it merges with the next year data
conf_te = conf_count[['Season','TeamID', 'ConfAbbrev']].merge(ncaa_diff, on =['Season', 'TeamID'], how='left').fillna(0)
conf_te = conf_te.drop('TeamID', axis =1)
STATS = ["mean","std","count","nunique","median","min","max","skew"]
conf_te = conf_te.groupby(['Season','ConfAbbrev']).agg(STATS).reset_index()
conf_te.columns = [f'{col}_{stat}' if col not in ['ConfAbbrev','Season'] else col for col, stat in conf_te.columns]
#conf_count[[column for column in conf_te.columns if column not in ['Season','ConfAbbrev']]] = conf_te[[column for column in conf_te.columns if column not in ['Season','ConfAbbrev']]] #decreasing accuracy
conf_count.head()


mstcr = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyCompactResults.csv')
wstcr = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WSecondaryTourneyCompactResults.csv')


stcr = pd.concat([mstcr, wstcr])
stcr.head()
stcr[(stcr['Season'] == 2024)]
STATS = ["mean","std","count","nunique","median","min","max","skew"]
sec_tour_win = stcr[['Season','WTeamID', 'WScore']].groupby(['Season','WTeamID']).mean().reset_index().fillna(0)
sec_tour_win = sec_tour_win.rename(columns={'WTeamID':'TeamID'})
sec_tour_lose = stcr[['Season','LTeamID', 'LScore']].groupby(['Season','LTeamID']).mean().reset_index().fillna(0)
sec_tour_lose = sec_tour_lose.rename(columns={'LTeamID':'TeamID'})

sec_tour = sec_tour_win.merge(sec_tour_lose, on = ['Season','TeamID'], how='outer').fillna(0)

sec_tour.columns = [f'{col}_sec' if col not in ['TeamID','Season'] else col for col in sec_tour.columns]
sec_tour.head()


coach = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeamCoaches.csv')
coach = coach.drop('CoachName',axis=1)
STATS = ["mean","std","count","nunique","median","min","max","skew"]
coach = coach.groupby(['Season','TeamID']).agg(STATS).reset_index().fillna(0)
coach.columns = [f'{col}_{stat}' if col not in ['TeamID','Season'] else col for col, stat in coach.columns]
coach.head()


team_season = seed_region.merge(df_season_stats, on=['Season', 'TeamID'], how='outer')
team_season = team_season.merge(df_ncaa_stats, on=['Season', 'TeamID'], how='outer')
team_season = team_season.merge(conf_count, on = ['Season', 'TeamID'], how = 'left')
team_season = team_season.merge(sec_tour, on = ['Season', 'TeamID'], how = 'left')
team_season = team_season.merge(coach, on = ['Season', 'TeamID'], how = 'left')
team_season = team_season.fillna(0)
#team_season = team_season.merge(rank_stats, on=['Season', 'TeamID'], how='left') decreases accuracy
#team_season = team_season.apply(lambda x: x.fillna(x.max()) if x.dtype != 'O' else x)
team_season = team_season.sort_values(by=['TeamID','Season']).reset_index(drop=True)
team_season.head()


# Make a copy to prevent modifying the original dataframe
df = team_season.copy()

# Current season's regular-season performance (shift `Season` back by 1 year)
df_curr_season = df.copy()
df_curr_season = df_curr_season[[col for col in df.columns if not col.endswith('_ncaa')]]
df_curr_season['Prev_Season'] = df_curr_season['Season']-1

# Previous season's regular-season performance (shift `Season` back by 1 year)
df_prev_season = df.copy()
df_prev_season = df_prev_season[['Season', 'TeamID', 'Seed', 'Region', 'ConfWinCount', 'ConfLoseCount', 'ConfAbbrev'] + [col for col in df.columns if col.endswith('_season')]]
df_prev_season = df_prev_season.rename(columns=lambda x: x.replace('_season', '_prev_season') if '_season' in x else x)
df_prev_season.rename(columns={"Season": "Prev_Season","Seed":"Prev_Seed", "Region":"Prev_Region", "ConfWinCount":"Prev_ConfWinCount", "ConfLoseCount":"Prev_ConfLoseCount",
                              "ConfAbbrev":"Prev_ConfAbbrev"},inplace=True)

# Previous season's NCAA performance (shift `Season` back by 1 year)
df_prev_ncaa = df.copy()
df_prev_ncaa = df_prev_ncaa[['Season', 'TeamID'] + [col for col in df.columns if col.endswith('_ncaa')]]
df_prev_ncaa = df_prev_ncaa.rename(columns=lambda x: x.replace('_ncaa', '_prev_ncaa') if '_ncaa' in x else x)
df_prev_ncaa.rename(columns={"Season": "Prev_Season"},inplace=True)

# Previous season's NCAA performance (shift `Season` back by 1 year)
df_prev_sec = df.copy()
df_prev_sec = df_prev_sec[['Season', 'TeamID'] + [col for col in df.columns if col.endswith('_sec')]]
df_prev_sec = df_prev_sec.rename(columns=lambda x: x.replace('_sec', '_prev_sec') if '_sec' in x else x)
df_prev_sec.rename(columns={"Season": "Prev_Season"},inplace=True)

# Merge all three dataframes
df_final = df_curr_season.merge(df_prev_season, on=['Prev_Season', 'TeamID'], how='left').merge(df_prev_ncaa, on=['Prev_Season', 'TeamID'], how='left')
df_final = df_final.merge(df_prev_sec, on=['Prev_Season', 'TeamID'], how='left')
df_final.drop('Prev_Season',axis=1,inplace = True)
df_final = df_final.fillna(0)

df_final['Region_Change'] = np.float64(df_final['Region']== df_final['Prev_Region'])
df_final.drop('Prev_Region',axis=1,inplace = True)
df_final['Seed_Change'] = np.float64(df_final['Seed'] - df_final['Prev_Seed'])

df_final['ConfWinVsLose'] = df_final['ConfWinCount'] - df_final['ConfLoseCount']
df_final['ConfWinVsPrevWin'] = df_final['ConfWinCount'] - df_final['Prev_ConfWinCount']

df_final['ConfAbbrev_Change'] = np.float64(df_final['ConfAbbrev']== df_final['Prev_ConfAbbrev'])
df_final.drop('ConfAbbrev',axis=1,inplace = True)
df_final.drop('Prev_ConfAbbrev',axis=1,inplace = True)

# Sorting for readability
df_final = df_final.sort_values(by=['TeamID', 'Season']).reset_index(drop=True)

# Display result
df_final.head()


df_final.drop('Region',axis=1,inplace = True)


mncaa_compact = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
wncaa_compact = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')


ncaa_compact = pd.concat([mncaa_compact, wncaa_compact], axis=0, ignore_index=True)
ncaa_compact.head()


ncaa_compact['WinDiff'] = ncaa_compact['WScore'] - ncaa_compact['LScore']
ncaa_compact['LoseDiff'] = ncaa_compact['LScore'] - ncaa_compact['WScore']


past_ncaa = ncaa_compact[['Season', 'WTeamID', 'LTeamID', 'WinDiff']]

df_train = past_ncaa.merge(df_final, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
df_train = df_train.merge(df_final, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=('_team1', '_team2'))
# Step 3: Compute feature differences using pd.concat() for better performance
features = [col for col in df_final.columns if col not in ['Season', 'TeamID']]
df_diff = pd.concat(
    [(df_train[f"{feature}_team1"] - df_train[f"{feature}_team2"]).rename(f"{feature}_diff") for feature in features],
    axis=1
)
df_train = pd.concat([df_train, df_diff], axis=1)

# Step 6: Keep only required columns for training
X_cols = [col for col in df_train.columns if '_diff' in col]  # Feature differences
X_train1 = df_train[X_cols]
y_train1 = df_train['WinDiff']


past_ncaa = ncaa_compact[['Season', 'WTeamID', 'LTeamID', 'LoseDiff']]

df_train = past_ncaa.merge(df_final, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
df_train = df_train.merge(df_final, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=('_team1', '_team2'))
# Step 3: Compute feature differences using pd.concat() for better performance
features = [col for col in df_final.columns if col not in ['Season', 'TeamID']]
df_diff = pd.concat(
    [(df_train[f"{feature}_team2"] - df_train[f"{feature}_team1"]).rename(f"{feature}_diff") for feature in features],
    axis=1
)
df_train = pd.concat([df_train, df_diff], axis=1)

# Step 6: Keep only required columns for training
X_cols = [col for col in df_train.columns if '_diff' in col]  # Feature differences
X_train2 = df_train[X_cols]
y_train2 = df_train['LoseDiff']


X = pd.concat([X_train1, X_train2])
y = pd.concat([y_train1, y_train2])


# Define save path
save_path = "xgb_checkpoint4.json"

# Callback to save the model periodically
class SaveModelCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        if epoch % 10000 == 0:  # Save every 1000 iterations
            model.save_model(save_path)
            print(f"Checkpoint saved at iteration {epoch}")
        return False

# Define the model
model = xgb.XGBRegressor(
    device="cuda",
    max_depth=6,  
    colsample_bynode=0.3, 
    subsample=0.8,  
    n_estimators=400_000,  
    learning_rate=0.01,  
    enable_categorical=True,
    min_child_weight=10,
    #early_stopping_rounds=500,
)

# Load model if exists
if os.path.exists(save_path):
    print("Resuming from checkpoint...")
    model.load_model(save_path)

# Train the model with callback
model.fit(X, y, eval_set=[(X, y)], verbose=0, callbacks=[SaveModelCallback()])

model.save_model(save_path)
print(f"Checkpoint saved after stopping")


# Define the model
model = xgb.XGBRegressor(
    device="cuda",
    max_depth=6,  
    colsample_bynode=0.3, 
    subsample=0.8,  
    n_estimators=400_000,  
    learning_rate=0.01,  
    enable_categorical=True,
    min_child_weight=10,
    #early_stopping_rounds=500,
)

# Load model if exists
if os.path.exists(save_path):
    print("Resuming from checkpoint...")
    model.load_model(save_path)
    
y_pred = model.predict(X)


from sklearn.metrics import mean_squared_error

# Assuming y_train contains actual outcomes (0 or 1)
# Assuming y_pred contains predicted probabilities
mse = mean_squared_error(y, y_pred)

print(f"Mean Square Error: {mse:.4f}")


sample_submission = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')
sample_submission.head()


sample_submission[['Season', 'Team1', 'Team2']] = sample_submission['ID'].str.split('_', expand=True).astype(int)


print(df_final[['Season', 'TeamID']].duplicated().sum())



for chunk in np.array_split(sample_submission,5):
    print(chunk.shape)
    df_pred = chunk.merge(df_final, right_on = ['Season','TeamID'], left_on = ['Season', 'Team1'], how='left')
    df_pred = df_pred.merge(df_final, right_on = ['Season', 'TeamID'], left_on = ['Season', 'Team2'], how = 'left', suffixes = ('_team1', '_team2'))
    df_pred = df_pred.fillna(0)
    features = [feature for feature in df_final.columns if feature not in ['Season', 'TeamID']]
    df_diff = pd.concat(
        [(df_pred[f'{feature}_team1']-df_pred[f'{feature}_team2']).rename(f'{feature}_diff') for feature in features],
        axis = 1, 
    )
    pred = model.predict(df_diff)
    sample_submission.loc[chunk.index[0]:chunk.index[-1],'Pred'] = 1 / (1 + np.exp(-0.1 * pred))


sample_submission.head()


sample_submission[['ID','Pred']].to_csv('march_mania_submission2.csv', index=False)
print('Submission created.')




