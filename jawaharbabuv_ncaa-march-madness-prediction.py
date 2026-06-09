#import require libraries
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import zipfile




import zipfile
import os

input_dir = "/kaggle/input/march-machine-learning-mania-2025"

# List files in the input directory
files = os.listdir(input_dir)

print(files)



import pandas as pd
import glob
import os

csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

# Create a dictionary to store DataFrames
dfs = {}

# Read CSV files with different encoding
for file in csv_files:
    file_name = os.path.basename(file)  # Extract filename
    try:
        df = pd.read_csv(file, encoding="utf-8")  # Try UTF-8 first
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding="ISO-8859-1")  # Try different encoding
    dfs[file_name] = df  # Store in dictionary with filename as key

print("All CSV files loaded successfully!")


for name, df in dfs.items():
    print(f"Loaded {name} with {df.shape[0]} rows and {df.shape[1]} columns")



for name, df in dfs.items():
    print(f"DataFrame for {name}:")
    print(df.head())  # Print first 5 rows of each DataFrame
    print("\n" + "="*50 + "\n")  # Separator


for name, df in dfs.items():
    print(df.isnull().any())
    print("\n" + "="*50 + "\n")  # Separator



for name, df in dfs.items():
    df.drop_duplicates(inplace=True)
    
print("duplicated values removed")


mseason, mteams, mncaatourseed, mncaatouresult, mregularseasonresult, mgamecity, mteamcoach, mordinals, mconfergames, msecondaryresult, mncaatourslots, mncaaseedroundslot= dfs['MSeasons.csv'],dfs['MTeams.csv'], dfs['MNCAATourneySeeds.csv'], dfs['MNCAATourneyDetailedResults.csv'], dfs['MRegularSeasonDetailedResults.csv'],dfs['MGameCities.csv'], dfs['MTeamCoaches.csv'], dfs['MMasseyOrdinals.csv'], dfs['MConferenceTourneyGames.csv'], dfs['MSecondaryTourneyCompactResults.csv'],dfs['MNCAATourneySlots.csv'], dfs['MNCAATourneySeedRoundSlots.csv']



mncaatouresult.describe()


mconfergames.describe()


mregularseasonresult.describe()


msecondaryresult.describe()


# Merging dataset
mncaatouresult['Gametype'] = 'MNCAAtour'
mregularseasonresult['Gametype'] = 'MRegularseason'
msecondaryresult['Gametype'] = 'msecondary'
merge_df = pd.concat([mncaatouresult,mregularseasonresult])
merge_df['Outcome'] = 1

# Create a reversed version of the data, so that each game appears twice.

merge_reverse = merge_df.copy()
features_to_swap = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
for feature in features_to_swap:
    merge_reverse[['W' + feature, 'L' + feature]] = merge_reverse[['L' + feature, 'W' + feature]]
    
merge_reverse['Outcome'] = 0

df = pd.concat([merge_df,merge_reverse])
df


wteam_epoints = df.groupby(['Season','DayNum','WTeamID']).apply(lambda x: (x['WFGM'].sum() + 0.5 * x['WFGM3'].sum()) / x['WFGA'].sum() if x['WFGA'].sum() > 0 else 0).round(1)*100
wteam_epoints = wteam_epoints.reset_index(name = 'efieldgoalpercent')
print('NCAA tour winning team effective field goal percentage:','\n', wteam_epoints)


lteam_epoints = df.groupby(['Season','DayNum','LTeamID']).apply(lambda x: (x['LFGM'].sum() + 0.5 * x['LFGM3'].sum()) / x['LFGA'].sum() if x['LFGA'].sum() > 0 else 0).round(1)*100
lteam_epoints = lteam_epoints.reset_index(name = 'efieldgoalpercent')
print('NCAA tour losing team effective field goal percentage:','\n', lteam_epoints)


df['WFieldGoalPercentage'] = np.where(df['WFGA'] != 0, df['WFGM'] / df['WFGA'], 0)
df['WThreePointPercentage'] = np.where(df['WFGA3'] != 0, df['WFGM3'] / df['WFGA3'], 0)
df['WFreeThrowPercentage'] = np.where(df['WFTA'] != 0, df['WFTM'] / df['WFTA'], 0)
df['WEffectiveFieldGoalPercentage'] = np.where(df['WFGA'] != 0, (df['WFGM'] + 0.5 * df['WFGM3']) / df['WFGA'], 0)
df['WTrueShootingPercentage'] = np.where((2 * (df['WFGA'] + 0.475 * df['WFTA'])) != 0, df['WScore'] / (2 * (df['WFGA'] + 0.475 * df['WFTA'])), 0)
df['WTotalReboundRatio'] = np.where((df['LOR'] + df['LDR'] + df['WOR'] + df['WDR']) != 0,
                                     (df['WOR'] + df['WDR']) / (df['LOR'] + df['LDR'] + df['WOR'] + df['WDR']),0)
df['WAssistTurnoverRatio'] = np.where(df['WTO'] != 0, df['WAst'] / df['WTO'], 0)
df['WStealTurnoverRatio'] = np.where(df['WTO'] != 0, df['WStl'] / df['WTO'], 0)


df['LFieldGoalPercentage'] = np.where(df['LFGA'] != 0, df['LFGM'] / df['LFGA'], 0)
df['LThreePointPercentage'] = np.where(df['LFGA3'] != 0, df['LFGM3'] / df['LFGA3'], 0)
df['LFreeThrowPercentage'] = np.where(df['LFTA'] != 0, df['LFTM'] / df['LFTA'], 0)
df['LEffectiveFieldGoalPercentage'] = np.where(df['LFGA'] != 0, (df['LFGM'] + 0.5 * df['LFGM3']) / df['LFGA'], 0)
df['LTrueShootingPercentage'] = np.where((2 * (df['LFGA'] + 0.475 * df['LFTA'])) != 0, df['LScore'] / (2 * (df['LFGA'] + 0.475 * df['LFTA'])), 0)
df['LTotalReboundRatio'] = np.where((df['WOR'] + df['WDR'] + df['LOR'] + df['LDR']) != 0,
                                     (df['LOR'] + df['LDR']) / (df['WOR'] + df['WDR'] + df['LOR'] + df['LDR']),0)
df['LAssistTurnoverRatio'] = np.where(df['LTO'] != 0, df['LAst'] / df['LTO'], 0)
df['LStealTurnoverRatio'] = np.where(df['LTO'] != 0, df['LStl'] / df['LTO'], 0)

df['FieldGoalPercentage_diff'] = df['WFieldGoalPercentage'] - (df['LFGM'] / df['LFGA'])
df['ThreePointPercentage_diff'] = df['WThreePointPercentage'] - (df['LFGM3'] / df['LFGA3'])
df['FreeThrowPercentage_diff'] = df['WFreeThrowPercentage'] - (df['LFTM'] / df['LFTA'])
df['EffectiveFieldGoalPercentage_diff'] = df['WEffectiveFieldGoalPercentage'] - ((df['LFGM'] + 0.5 * df['LFGM3']) / df['LFGA'])
df['TrueShootingPercentage_diff'] = df['WTrueShootingPercentage'] - (df['LScore'] / (2 * (df['LFGA'] + 0.475 * df['LFTA'])))

df = df.drop('FreeThrowPercentage_diff', axis = 'columns')

dummies = pd.get_dummies(df.Gametype)
df1 = pd.concat([df, dummies], axis = 'columns')
df_new = df1.drop(['Gametype','WLoc'], axis = 'columns')
df_new


from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

X = df_new.drop(['Outcome'], axis = 'columns')
y = df_new['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf_model = RandomForestClassifier(n_estimators = 100, oob_score = True, warm_start = True, random_state = 42)
rf_model.fit(X_train, y_train)

y_predict = rf_model.predict(X_train)

y_predict1 = rf_model.predict(X_test)
print("Training set accuracy score: ", accuracy_score(y_train,y_predict))
print("Test set accuracy score: ", accuracy_score(y_test,y_predict1))


ncaa_new = dfs['SampleSubmissionStage2.csv']
ncaa_new[['Season', 'TeamID1', 'TeamID2']] = ncaa_new['ID'].str.split('_', expand = True)
ncaa_new = ncaa_new.drop(['ID','Pred'], axis = 'columns')
ncaa_new['TeamID1'] = ncaa_new['TeamID1'].astype(int)
ncaa_new['TeamID2'] = ncaa_new['TeamID2'].astype(int)
ncaa_new['Season'] = ncaa_new['Season'].astype(int)
ncaa_new = ncaa_new[(ncaa_new['TeamID1'] >= 1000) & (ncaa_new['TeamID1'] <= 1999) & (ncaa_new['TeamID2'] >= 1000) & (ncaa_new['TeamID2'] <= 1999)]

team_avg_stats = df_new.groupby(['Season', 'WTeamID']).mean().reset_index()
ncaa = pd.read_csv("../input/mteam-avg/team_avg1.csv")

ncaa_predict = rf_model.predict(ncaa)
#print("2025 NCAA new set of data accuracy score: ", accuracy_score(y_train,ncaa_predict))
probabilities = rf_model.predict_proba(ncaa)[:,1]

mncaa_result = pd.DataFrame({
    'ID': ncaa['Season'].astype(str) + '_' + ncaa['WTeamID'].astype(str) + '_' + ncaa['LTeamID'].astype(str),
    'Pred': probabilities
})

mncaa_result


wseason, wteams, wncaatourseed, wncaatouresult, wregularseasonresult, wgamecity,wconfergames, wsecondaryresult, wncaatourslots,= dfs['WSeasons.csv'],dfs['WTeams.csv'], dfs['WNCAATourneySeeds.csv'], dfs['WNCAATourneyDetailedResults.csv'], dfs['WRegularSeasonDetailedResults.csv'],dfs['WGameCities.csv'],dfs['WConferenceTourneyGames.csv'], dfs['WSecondaryTourneyCompactResults.csv'],dfs['WNCAATourneySlots.csv']



wncaatouresult.describe()


wregularseasonresult.describe()


wconfergames.describe()


wsecondaryresult.describe()


# merging datasets
wncaatouresult['Gametype'] = 'WNCAAtour'
wregularseasonresult['Gametype'] = 'WRegularseason'
wsecondaryresult['Gametype'] = 'Wsecondary'
wmerge_df = pd.concat([wncaatouresult,wregularseasonresult])
wmerge_df['Outcome'] = 1

# Create a reversed version of the data, so that each game appears twice.

wmerge_reverse = wmerge_df.copy()
wfeatures_to_swap = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
for feature in wfeatures_to_swap:
    wmerge_reverse[['W' + feature, 'L' + feature]] = wmerge_reverse[['L' + feature, 'W' + feature]]
    
wmerge_reverse['Outcome'] = 0

df_w = pd.concat([wmerge_df,wmerge_reverse])
df_w = df_w.drop('WLoc', axis = 'columns')
df_w.info()


df_w['WFieldGoalPercentage'] = np.where(df_w['WFGA'] != 0, df_w['WFGM'] / df_w['WFGA'], 0)
df_w['WThreePointPercentage'] = np.where(df_w['WFGA3'] != 0, df_w['WFGM3'] / df_w['WFGA3'], 0)
df_w['WFreeThrowPercentage'] = np.where(df_w['WFTA'] != 0, df_w['WFTM'] / df_w['WFTA'], 0)
df_w['WEffectiveFieldGoalPercentage'] = np.where(df_w['WFGA'] != 0, (df_w['WFGM'] + 0.5 * df_w['WFGM3']) / df_w['WFGA'], 0)
#df_w['WTrueShootingPercentage'] = np.where((2 * (df_w['WFGA'] + 0.475 * df_w['WFTA'])) != 0, df_w['WScore'] / (2 * (df_w['WFGA'] + 0.475 * df['WFTA'])), 0)
df_w['WTotalReboundRatio'] = np.where((df_w['LOR'] + df_w['LDR'] + df_w['WOR'] + df_w['WDR']) != 0,
                                     (df_w['WOR'] + df_w['WDR']) / (df_w['LOR'] + df_w['LDR'] + df_w['WOR'] + df_w['WDR']),0)
df_w['WAssistTurnoverRatio'] = np.where(df_w['WTO'] != 0, df_w['WAst'] / df_w['WTO'], 0)
df_w['WStealTurnoverRatio'] = np.where(df_w['WTO'] != 0, df_w['WStl'] / df_w['WTO'], 0)


df_w['LFieldGoalPercentage'] = np.where(df_w['LFGA'] != 0, df_w['LFGM'] / df_w['LFGA'], 0)
df_w['LThreePointPercentage'] = np.where(df_w['LFGA3'] != 0, df_w['LFGM3'] / df_w['LFGA3'], 0)
df_w['LFreeThrowPercentage'] = np.where(df_w['LFTA'] != 0, df_w['LFTM'] / df_w['LFTA'], 0)
df_w['LEffectiveFieldGoalPercentage'] = np.where(df_w['LFGA'] != 0, (df_w['LFGM'] + 0.5 * df_w['LFGM3']) / df_w['LFGA'], 0)
df_w['LTrueShootingPercentage'] = np.where((2 * (df_w['LFGA'] + 0.475 * df_w['LFTA'])) != 0, df_w['LScore'] / (2 * (df_w['LFGA'] + 0.475 * df_w['LFTA'])), 0)
df_w['LTotalReboundRatio'] = np.where((df_w['WOR'] + df_w['WDR'] + df_w['LOR'] + df_w['LDR']) != 0,
                                     (df_w['LOR'] + df_w['LDR']) / (df_w['WOR'] + df_w['WDR'] + df_w['LOR'] + df_w['LDR']),0)
df_w['LAssistTurnoverRatio'] = np.where(df_w['LTO'] != 0, df_w['LAst'] / df_w['LTO'], 0)
df_w['LStealTurnoverRatio'] = np.where(df_w['LTO'] != 0, df_w['LStl'] / df_w['LTO'], 0)

df_w['FieldGoalPercentage_diff'] = df_w['WFieldGoalPercentage'] - (df_w['LFGM'] / df_w['LFGA'])
df_w['ThreePointPercentage_diff'] = df_w['WThreePointPercentage'] - (df_w['LFGM3'] / df_w['LFGA3'])
df_w['FreeThrowPercentage_diff'] = df_w['WFreeThrowPercentage'] - (df_w['LFTM'] / df_w['LFTA'])
df_w['EffectiveFieldGoalPercentage_diff'] = df_w['WEffectiveFieldGoalPercentage'] - ((df_w['LFGM'] + 0.5 * df_w['LFGM3']) / df_w['LFGA'])
#df_w['TrueShootingPercentage_diff'] = df_w['WTrueShootingPercentage'] - (df_w['LScore'] / (2 * (df_w['LFGA'] + 0.475 * df_w['LFTA'])))

#df = df.drop('FreeThrowPercentage_diff', axis = 'columns')

dummies1 = pd.get_dummies(df_w.Gametype)
df_w1 = pd.concat([df_w, dummies1], axis = 'columns')
w_df = df_w1.drop('Gametype', axis = 'columns')
w_df = w_df.dropna()
w_df


X_w = w_df.drop(['Outcome'], axis = 'columns')
y_w = w_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X_w, y_w, test_size = 0.2, random_state = 42)

rf_model_w = RandomForestClassifier(n_estimators = 100, oob_score = True, warm_start = True, random_state = 42)
rf_model_w.fit(X_train, y_train)

y_predict_w = rf_model_w.predict(X_train)

y_predict_w1 = rf_model_w.predict(X_test)
print("Training set accuracy score: ", accuracy_score(y_train,y_predict_w))
print("Test set accuracy score: ", accuracy_score(y_test,y_predict_w1))



w_ncaa_new = dfs['SampleSubmissionStage2.csv']
w_ncaa_new[['Season', 'WTeamID', 'LTeamID']] = w_ncaa_new['ID'].str.split('_', expand = True)
w_ncaa_new = w_ncaa_new.drop(['ID','Pred'], axis = 'columns')
w_ncaa_new['WTeamID'] = w_ncaa_new['WTeamID'].astype(int)
w_ncaa_new['LTeamID'] = w_ncaa_new['LTeamID'].astype(int)
w_ncaa_new['Season'] = w_ncaa_new['Season'].astype(int)
w_ncaa_new = w_ncaa_new[(w_ncaa_new['WTeamID'] >= 3000) & (w_ncaa_new['WTeamID'] <= 3999) & (w_ncaa_new['LTeamID'] >= 3000) & (w_ncaa_new['LTeamID'] <= 3999)]

w_team_avg_stats = w_df.groupby(['Season', 'WTeamID']).mean().reset_index()

ncaa_w = pd.read_csv("../input/wstats-avg/w_ncaa.csv")

wncaa_predict = rf_model_w.predict(ncaa_w)
w_probabilities = rf_model_w.predict_proba(ncaa_w)[:,1]

wncaa_result = pd.DataFrame({
    'ID': ncaa_w['Season'].astype(str) + '_' + ncaa_w['WTeamID'].astype(str) + '_' + ncaa_w['LTeamID'].astype(str),
    'Pred': w_probabilities
})

wncaa_result


ncaa_final_prediction = pd.concat([mncaa_result,wncaa_result])
print("NCAA final prediction for both Men and Women")
ncaa_final_prediction

