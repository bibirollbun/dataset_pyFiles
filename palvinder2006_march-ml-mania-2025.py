import os
import numpy as np
from numpy import array, mean
import pandas as pd
from pandas import DataFrame
import xgboost as xgb 
from matplotlib import pyplot as plt
import seaborn as sns
import math

from plotly import tools
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px

import sklearn
from collections import Counter
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


PATH = "/kaggle/input/march-machine-learning-mania-2025"

csv_files = []
for file in os.listdir(PATH):
    if file.endswith('.csv'):
        csv_files.append(file)


# Sort files and print total count
csv_files = sorted(csv_files)
print("The number of CSV files in the competition data:", len(csv_files))


# Display all columns when showing DataFrames
pd.set_option('display.max_columns', None)

# Read and display each CSV
for cnt, filename in enumerate(csv_files, 1):
    print("=" * 50)
    try:
        df = pd.read_csv(os.path.join(PATH, filename))
    except:
        df = pd.read_csv(os.path.join(PATH, filename), encoding='cp1252')
    print(f"{cnt}", filename, df.shape)
    display(df.head(4))


df_teams = pd.read_csv(os.path.join(PATH, "MTeams.csv"))
df_teams['SeasonPlayed'] = df_teams['LastD1Season'] - df_teams['FirstD1Season']
print(f'Shape: {df_teams.shape}')
df_teams.head()


df_seasons = pd.read_csv(os.path.join(PATH, "MSeasons.csv"))

print(f'Shape: {df_seasons.shape}')
df_seasons.head()


# Load tournament results (men's + women's)
tourney_results = pd.concat([
    pd.read_csv(os.path.join(PATH, "MNCAATourneyDetailedResults.csv")),
    pd.read_csv(os.path.join(PATH, "WNCAATourneyDetailedResults.csv")),
], ignore_index=True)

# Load seed info (men's + women's)
seeds = pd.concat([
    pd.read_csv(os.path.join(PATH, "MNCAATourneySeeds.csv")),
    pd.read_csv(os.path.join(PATH, "WNCAATourneySeeds.csv")),
], ignore_index=True)

# Load regular season details results (men's + women's)
regular_results = pd.concat([
    pd.read_csv(os.path.join(PATH, "MRegularSeasonDetailedResults.csv")),
    pd.read_csv(os.path.join(PATH, "WRegularSeasonDetailedResults.csv")),
], ignore_index=True)

# Load regular season compact results (men's + women's)
df_season_results = pd.concat([
    pd.read_csv(os.path.join(PATH, "MRegularSeasonCompactResults.csv")),
    pd.read_csv(os.path.join(PATH, "WRegularSeasonCompactResults.csv")),
], ignore_index=True)

# Load regular tourney compact results (men's + women's)
df_tourney_results = pd.concat([
    pd.read_csv(os.path.join(PATH, "WNCAATourneyCompactResults.csv")),
    pd.read_csv(os.path.join(PATH, "MNCAATourneyCompactResults.csv")),
], ignore_index=True)


print(f'Shape: {regular_results.shape}')
regular_results.head()


df_season_results.head()


df_season_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)

print(f'Shape: {df_season_results.shape}')
df_season_results.head()


df_tourney_results.head()


df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
print(f'Shape: {df_tourney_results.shape}')

df_tourney_results.head()


#WinRatio
total_wins = df_season_results.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"WTeamID": "TeamID", "DayNum": "NumWins"})
total_losses = df_season_results.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"LTeamID": "TeamID", "DayNum": "NumLosses"})
df_wr = total_wins.merge(total_losses, how = 'left', on = ['Season','TeamID'])
df_wr['WinRatio'] = (df_wr.iloc[:,2]/(df_wr.iloc[:,2]+df_wr.iloc[:,3]))

df_seasons = total_wins.reset_index()[['Season','TeamID']]

df_seasons = df_seasons.merge(df_wr, how = 'left', on = ['Season','TeamID'])[['Season', 'TeamID', 'WinRatio']]


#AVG score

sum_WScore = (df_season_results.groupby(['Season', 'WTeamID']).sum())
sum_WScore = sum_WScore.reset_index()[['Season','WTeamID', 'WScore']].rename(columns={"WTeamID": "TeamID"})

sum_LScore = (df_season_results.groupby(['Season', 'LTeamID']).sum())
sum_LScore = sum_LScore.reset_index()[['Season','LTeamID', 'LScore']].rename(columns={"LTeamID": "TeamID"})

Lgames_played = (df_season_results.groupby(['Season', 'LTeamID']).count())
Lgames_played = Lgames_played.reset_index()[['Season','LTeamID', 'LScore']].rename(columns={"LTeamID": "TeamID", "LScore" : 'gamesL'})

Wgames_played = (df_season_results.groupby(['Season', 'WTeamID']).count())
Wgames_played = Wgames_played.reset_index()[['Season','WTeamID', 'WScore']].rename(columns={"WTeamID": "TeamID", "WScore" : 'gamesW'})

sum_score = sum_LScore.merge(sum_WScore, how = 'left', on = ['Season','TeamID'])
sum_score = sum_score.merge(Lgames_played, how = 'left', on = ['Season','TeamID'])
sum_score = sum_score.merge(Wgames_played, how = 'left', on = ['Season','TeamID'])

sum_score = sum_score.fillna(0)
sum_score['AvgScore'] = (sum_score.iloc[:,2]+sum_score.iloc[:,3])/(sum_score.iloc[:,4]+sum_score.iloc[:,5])


#Avg Score Diff
sumVsScoreW = (df_season_results.groupby(['Season', 'WTeamID']).sum())
sumVsScoreW = sumVsScoreW.reset_index()[['Season','WTeamID', 'LScore']].rename(columns={"WTeamID": "TeamID",'LScore':'WOppScore'})

sumVsScoreL = (df_season_results.groupby(['Season', 'LTeamID']).sum())
sumVsScoreL = sumVsScoreL.reset_index()[['Season','LTeamID', 'WScore']].rename(columns={"LTeamID": "TeamID", 'WScore':'LOppScore'})


sum_score = sum_score.merge(sumVsScoreW, how = 'left', on = ['Season','TeamID'])
sum_score = sum_score.merge(sumVsScoreL, how = 'left', on = ['Season','TeamID'])

sum_score
sum_score['AvgDiffScore'] = (sum_score.loc[:,'WScore']-
                             sum_score.loc[:,'WOppScore']+sum_score.loc[:,'LScore']
                             -sum_score.loc[:,'LOppScore'])/(sum_score.loc[:,'gamesW']+sum_score.loc[:,'gamesL'])



sum_score = sum_score.reset_index()[['Season','TeamID', 'AvgScore', 'AvgDiffScore']]
df_seasons = df_seasons.merge(sum_score, on = ['Season', 'TeamID'])


print(f'Shape: {df_seasons.shape}')
df_seasons.head()



df_vis = df_seasons.groupby(['TeamID'])['WinRatio'].mean().reset_index()

playoff_wins = df_tourney_results.groupby(['WTeamID']).count()
playoff_losses = df_tourney_results.groupby(['LTeamID']).count().reset_index().rename(columns={"LTeamID": "TeamID"})

playoff_wins = playoff_wins.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"WTeamID": "TeamID", "DayNum": "PlayoffWins"})
playoff_wins = playoff_wins.merge(playoff_losses, how = 'left', on = (['TeamID']))
playoff_wins['PlayoffWR'] = (playoff_wins.iloc[:,2]/(playoff_wins.iloc[:,2]+playoff_wins.iloc[:,3]))


df_vis = df_vis.merge(playoff_wins, on = (['TeamID']), how = 'left')


df_vis = df_vis.loc[df_vis['WinRatio'].nlargest(20).index]
trace1 = go.Bar(

                y = df_vis.WinRatio,
                name = "WinRatio",
                marker = dict(color = 'rgba(0, 123, 255, 0.6)',
                             line=dict(color='rgb(0,0,0)',width=1.1)),
                )
# create trace2 
trace2 = go.Bar(

                y = df_vis.PlayoffWR,
                name = "PlayoffWR",
                marker = dict(color = 'rgba(40, 167, 69, 0.6)',
                              line=dict(color='rgb(0,0,0)',width=1.1)),
                )

data = [trace1, trace2]
layout = go.Layout(barmode = "group", title = 'Average WinRatio during Season vs Playoffs',xaxis_title="Teams with best seasonal WinRatio",
    yaxis_title="Ratio")
fig = go.Figure(data = data, layout = layout)
fig.show()


teams = df_seasons.TeamID.unique()
seasons = regular_results.Season.unique()
all_stats = []

for season in seasons:
    for team in teams:
        df_temp = regular_results[
            (regular_results['Season'] == season) & 
            ((regular_results['WTeamID'] == team) | (regular_results['LTeamID'] == team))
        ].tail(14).reset_index(drop=True)

        WPts, LPts, Pos, OE, DE, eFG, ATR = [], [], [], [], [], [], []

        if len(df_temp) != 0:
            for i in range(len(df_temp)):
                if df_temp.loc[i, 'WTeamID'] == team:
                    prefix = 'W'
                else:
                    prefix = 'L'

                WPts.append(df_temp.loc[i, 'WScore'])
                LPts.append(df_temp.loc[i, 'LScore'])

                possessions = df_temp.loc[i, prefix + 'FGA'] - df_temp.loc[i, prefix + 'OR'] + df_temp.loc[i, prefix + 'TO'] + (0.475 * df_temp.loc[i, prefix + 'FTA'])
                Pos.append(possessions)

                if prefix == 'W':
                    OE.append(WPts[-1] / possessions)
                    DE.append(LPts[-1] / possessions)
                else:
                    OE.append(LPts[-1] / possessions)
                    DE.append(WPts[-1] / possessions)

                eFG.append((df_temp.loc[i, prefix + 'FGM'] + 0.5 * df_temp.loc[i, prefix + 'FGM3']) / df_temp.loc[i, prefix + 'FGA'])
                ATR.append(df_temp.loc[i, prefix + 'Ast'] / df_temp.loc[i, prefix + 'TO'])

            all_stats.append({
                "Season": season,
                "TeamID": team,
                "AvgPos": sum(Pos) / len(Pos),
                "AvgOE": sum(OE) / len(OE),
                "AvgDE": sum(DE) / len(DE),
                "Avg_eFG": sum(eFG) / len(eFG),
                "AvgATR": sum(ATR) / len(ATR)
            })

df_stats = pd.DataFrame(all_stats)


print(f'Shape: {df_stats.shape}')
df_stats.head()


df_seasons = df_seasons.merge(df_stats, how = 'right',on = ['Season', 'TeamID'])


df_Mconf = pd.read_csv(os.path.join(PATH, "MTeamConferences.csv"))
df_Wconf = pd.read_csv(os.path.join(PATH, "WTeamConferences.csv"))

df_Mconf = df_Mconf.rename(columns = {"TeamID" : "WTeamID"})
df_Wconf = df_Wconf.rename(columns = {"TeamID" : "WTeamID"})
df_Wconf['ConfAbbrev'] = 'w_' + df_Wconf['ConfAbbrev'].astype(str)
df_conf = pd.concat([df_Mconf, df_Wconf], ignore_index = True)
df_conf_rank = df_tourney_results.merge(df_conf, how = 'left', on = ['WTeamID', 'Season'])


df_seasons = df_seasons.merge(df_conf, how = 'left', left_on = ['Season','TeamID'], right_on = ['Season', 'WTeamID'])


df = df_conf_rank.groupby(['Season', 'ConfAbbrev', 'WTeamID']).count().reset_index()
df = df[['Season', 'ConfAbbrev', 'DayNum']].rename(columns={"DayNum": "PlayoffWins"})


# Load conference abbreviations and duplicate with 'w_' prefix
confs = pd.read_csv(os.path.join(PATH, "Conferences.csv"))["ConfAbbrev"]
w_confs = pd.Series(["w_" + conf for conf in confs])
confs = pd.concat([confs, w_confs], ignore_index=True)

seasons = df_tourney_results.Season.unique()

conf_rank_list = []
conf_seed_list = []

for season in seasons:
    next_season = season + 1  # don't overwrite 'season' in loop
    for conf in confs:
        df_rank = df_seasons[(df_seasons["Season"] == next_season) & (df_seasons["ConfAbbrev"] == conf)]
        df_rank = df_rank.sort_values(by=['WinRatio'], ascending=False).reset_index(drop=True)

        conf_seed_list.append(df_rank.loc[:, ['Season', 'TeamID']])  # save for later concat

        # Compute score over last 3 seasons
        score = 0
        for i in range(1, 4):
            temp = df[(df["Season"] == next_season - i) & (df["ConfAbbrev"] == conf)]
            if not temp.empty:
                score += temp.PlayoffWins.max()

        conf_rank_list.append({
            "Season": next_season,
            "ConfAbbrev": conf,
            "ConfScore": score
        })

# Create final DataFrames
conf_rank = pd.DataFrame(conf_rank_list)
conf_seed = pd.concat(conf_seed_list, ignore_index=True)
conf_seed = conf_seed.reset_index().rename(columns={"index": "ConfSeed"})



df_seasons = df_seasons.merge(conf_rank, how = 'left', on = ['Season', 'ConfAbbrev'])
df_seasons = df_seasons.merge(conf_seed, how = 'left', on = ['Season', 'TeamID'])

df_seasons = df_seasons.fillna(0).drop(['ConfAbbrev', 'WTeamID'], axis=1)


features = df_seasons.iloc[:,2:].columns.values.tolist()
df_seasons.replace([np.inf, -np.inf], np.nan, inplace=True)
df_seasons = df_seasons.dropna()
df_seasons.head()


#Vis of conf score

vis_conf = df_seasons.groupby(['TeamID'])['ConfScore'].mean().reset_index()
vis_conf.iloc[:,1] = vis_conf.iloc[:,1]/max(vis_conf.iloc[:,1])

df_vis = df_vis.merge(vis_conf, on = (['TeamID']), how = 'left')
df_vis.fillna(0, inplace = True)


trace1 = go.Bar(

                y = df_vis.WinRatio,
                name = "WinRatio",
                marker = dict(color = 'rgba(0, 123, 255, 0.6)',
                             line=dict(color='rgb(0,0,0)',width=1.1)),
                )
# create trace2 
trace2 = go.Bar(

                y = df_vis.PlayoffWR,
                name = "PlayoffWR",
                marker = dict(color = 'rgba(255, 153, 51, 0.6)',
                              line=dict(color='rgb(0,0,0)',width=1.1)),
                )
trace3 = go.Bar(

                y = df_vis.ConfScore,
                name = "ConferenceScore",
                marker = dict(color = 'rgba(153, 102, 255, 0.6)',
                              line=dict(color='rgb(0,0,0)',width=0.7)),
                )
data = [trace1, trace2, trace3]
layout = go.Layout(barmode = "group", title = 'Histogram of WinRatios and Conference Score',xaxis_title="Teams with best seasonal WinRatio",
    yaxis_title="Ratio")
fig = go.Figure(data = data, layout = layout)
fig.show()


df_seasons[features] = MinMaxScaler().fit_transform(df_seasons[features])

#Add features for winning team with prefix W_
df_train = df_tourney_results.merge(df_seasons, how = 'right', left_on = ['Season', 'WTeamID'],
                    right_on = ['Season', 'TeamID']).rename(columns={'WinRatio': 'A_WinRatio',
                            'AvgScore': 'A_AvgScore',
                            'AvgDiffScore': 'A_DiffScore',
                            'ConfScore': 'A_ConfScore',
                            'AvgPos': 'A_AvgPos',
                            'AvgOE': 'A_AvgOE',
                            'AvgDE': 'A_AvgDE',
                            'Avg_eFG': 'A_Avg_eFG',
                            'AvgATR': 'A_AvgATR',
                            'WTeamID': 'A_TeamID',
                            'ConfSeed': 'A_ConfSeed',
                            'WScore': 'A_Score'}).drop(columns='TeamID', axis=1)

#Add features for losing team with prefix L_
df_train = df_train.merge(df_seasons, how = 'left', left_on = ['Season', 'LTeamID'],
                    right_on = ['Season', 'TeamID']).rename(columns={'WinRatio': 'B_WinRatio',
                            'AvgScore': 'B_AvgScore',
                            'AvgDiffScore': 'B_DiffScore',
                            'ConfScore': 'B_ConfScore',
                            'AvgPos': 'B_AvgPos',
                            'AvgOE': 'B_AvgOE',
                            'AvgDE': 'B_AvgDE',
                            'Avg_eFG': 'B_Avg_eFG',
                            'AvgATR': 'B_AvgATR',
                            'LTeamID': 'B_TeamID',
                            'ConfSeed': 'B_ConfSeed',
                            'LScore': 'B_Score'}).drop(columns='TeamID', axis=1)
df_train = df_train.dropna()

print(f'Shape: {df_train.shape}')


df_copy = df_train.rename(columns={'B_WinRatio': 'A_WinRatio',
                                   'B_AvgScore': 'A_AvgScore',
                                   'B_DiffScore': 'A_DiffScore',
                                   'B_ConfScore': 'A_ConfScore',
                                   'B_AvgPos': 'A_AvgPos',
                                   'B_AvgOE': 'A_AvgOE',
                                   'B_AvgDE': 'A_AvgDE',
                                   'B_Avg_eFG': 'A_Avg_eFG',
                                   'B_AvgATR': 'A_AvgATR',
                                   'B_TeamID': 'A_TeamID',
                                   'B_Score': 'A_Score',
                                   'B_ConfSeed': 'A_ConfSeed',
                                   'A_WinRatio': 'B_WinRatio',                                                                                
                                   'A_Score': 'B_Score',
                                   'A_AvgScore': 'B_AvgScore',
                                   'A_DiffScore': 'B_DiffScore',
                                   'A_ConfScore': 'B_ConfScore',
                                   'A_AvgPos': 'B_AvgPos',
                                   'A_AvgOE': 'B_AvgOE',
                                   'A_AvgDE': 'B_AvgDE',
                                   'A_Avg_eFG': 'B_Avg_eFG',
                                   'A_AvgATR': 'B_AvgATR',
                                   'A_TeamID': 'B_TeamID',
                                   'A_ConfSeed': 'B_ConfSeed'
                                  })

df_train = pd.concat([df_train, df_copy]).sort_values(by=['Season', 'DayNum'])


df_train['ConfDiff'] = df_train['A_ConfScore'] - df_train['B_ConfScore']
df_train['ScoreDiff'] = df_train['A_DiffScore'] - df_train['B_DiffScore']
df_train['WRDiff'] = df_train['A_WinRatio'] - df_train['B_WinRatio']
df_train['SeedDiff'] = df_train['A_ConfSeed'] - df_train['B_ConfSeed']


df_train['OEDiff'] = (df_train['A_AvgOE']-df_train['B_AvgOE'])
df_train['DEDiff'] = (df_train['A_AvgDE']-df_train['B_AvgDE'])
df_train['ATRDiff'] = (df_train['A_AvgATR']-df_train['B_AvgATR'])
df_train['PosDiff'] = (df_train['A_AvgPos']-df_train['B_AvgPos'])
df_train['ScoreDiff'] = (df_train['A_DiffScore']-df_train['B_DiffScore'])
df_train['eFGDiff'] = (df_train['A_Avg_eFG']-df_train['B_Avg_eFG'])

df_train = df_train.drop(columns=['A_AvgOE', 'B_AvgOE','A_AvgDE','B_AvgDE','A_AvgATR','B_AvgATR',
                                  'A_AvgPos', 'B_AvgPos', 'B_AvgPos', 'A_AvgPos', 'B_Avg_eFG', 'A_Avg_eFG'])
features = df_train.iloc[:,6:].columns.values.tolist()

#Labels
df_train['MatchDiff'] = (df_train['A_Score']-df_train['B_Score'])
df_train['Target'] = ((df_train['A_Score']-df_train['B_Score']) > 0).astype(int)


#Vis features

fig = ff.create_scatterplotmatrix(df_train.iloc[0:10000:25,[26,16,17,18,19,20,21]], index='Target', diag='box', size=2, height=900, width=900, colormap='Bluered')
fig.show()


mode = 'xgb'
seasons = df_train['Season'].unique()
seasons.sort()
cvs = []


for season in seasons[5:]:

    print(f'\nValidating on season {season}')

    X = df_train[df_train['Season'] < season-1].reset_index(drop=True).copy()
    X_val = df_train[df_train['Season'] == season].reset_index(drop=True).copy()




    if mode == "xgb":

        model = xgb.XGBClassifier(booster = 'gbtree',eval_metric = 'mae', colsample_bytree = 0.7, min_child_weight = 40,
                                     gamma = 10, max_depth = 3, num_parallel_tree = 3, eta = 0.1)
    else:
        model = LogisticRegression(C=0.5)


    model.fit(X[features], X["Target"])

    pred = model.predict_proba(X_val[features])[:, 1]
    #pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = np.clip(pred, 0, 1)
    score = ((X_val['Target'].values - pred) ** 2).mean()
    cvs.append(score)


    print(f'\t -> Scored {score:.3f}')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(pred, X_val['MatchDiff'].values, s=5)
    plt.title('Prediction vs Score Diff')
    plt.grid(True)
    plt.plot([0.5,0.5,0.5],[-50,0,50], "r--")
    plt.plot([-0.1,0,1.1],[0,0,0], "r--")

    plt.subplot(1, 2, 2)
    sns.histplot(pred, bins=25)
    plt.title('Predictions probability')
    plt.show()

print(f'\n Local CV is {np.mean(cvs):.3f}')
if mode == 'xgb':
    #plt_feats = plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    #plt_feats.title('hej')
    import plotly.express as px
    fig = px.bar(x=range(len(model.feature_importances_)), y=model.feature_importances_, labels={'x':'Features', 'y':'Importance'}, title = 'Feature Importance')
    fig.show()
    print('Features:', features)


model = xgb.XGBClassifier(booster = 'gbtree')

params = {
        'min_child_weight': [1, 5, 10, 20, 30, 40, 50],
        'gamma': [0.5, 1, 1.5, 2, 5, 10],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'eta' : [0.02, 0.05, 0.1, 0.15]
        }


folds = 5
param_comb = 25

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 11)

random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='neg_brier_score', n_jobs=4, cv=folds, verbose=1, random_state=111 )

random_search.fit(df_train[features],df_train['Target'])


print('\n Best estimator:')
print(random_search.best_estimator_)

print('\n Best hyperparameters:')
print(random_search.best_params_)


df_test1 = pd.read_csv(os.path.join(PATH, "SampleSubmissionStage1.csv"))
df_test2 = pd.read_csv(os.path.join(PATH, "SampleSubmissionStage2.csv"))

df_test1['Season'] = df_test1['ID'].apply(lambda x: int(x.split('_')[0]))
df_test1['TeamIdA'] = df_test1['ID'].apply(lambda x: int(x.split('_')[1]))
df_test1['TeamIdB'] = df_test1['ID'].apply(lambda x: int(x.split('_')[2]))

df_test2['Season'] = df_test2['ID'].apply(lambda x: int(x.split('_')[0]))
df_test2['TeamIdA'] = df_test2['ID'].apply(lambda x: int(x.split('_')[1]))
df_test2['TeamIdB'] = df_test2['ID'].apply(lambda x: int(x.split('_')[2]))


#Add features for winning team with prefix A_
df_test1 = df_test1.merge(df_seasons, how = 'left', left_on = ['Season', 'TeamIdA'],
                        right_on = ['Season', 'TeamID']).rename(columns={'WinRatio': 'A_WinRatio',
                                            'AvgScore': 'A_AvgScore',
                                            'AvgDiffScore': 'A_DiffScore',
                                            'ConfScore': 'A_ConfScore',
                                            'AvgPos': 'A_AvgPos',
                                            'AvgOE': 'A_AvgOE',
                                            'AvgDE': 'A_AvgDE',
                                            'Avg_eFG': 'A_Avg_eFG',
                                            'AvgATR': 'A_AvgATR',
                                            'TeamIdA': 'A_TeamID',
                                            'WScore': 'A_Score',
                                            'ConfScore': 'A_ConfScore',
                                            'ConfSeed': 'A_ConfSeed'                                                                       
                                                              }).drop(columns='TeamID', axis=1)

#Add features for losing team with prefix B_

df_test1 = df_test1.merge(df_seasons, how = 'left', left_on = ['Season', 'TeamIdB'],
                                    right_on = ['Season', 'TeamID']).rename(columns={'WinRatio': 'B_WinRatio',
                                             'AvgScore': 'B_AvgScore',
                                             'AvgDiffScore': 'B_DiffScore',
                                             'ConfScore': 'B_ConfScore',
                                             'AvgPos': 'B_AvgPos',
                                             'AvgOE': 'B_AvgOE',
                                             'AvgDE': 'B_AvgDE',
                                             'Avg_eFG': 'B_Avg_eFG',
                                             'AvgATR': 'B_AvgATR',
                                             'TeamIdB': 'B_TeamID',
                                             'LScore': 'B_Score',
                                             'ConfScore': 'B_ConfScore',
                                             'ConfSeed': 'B_ConfSeed'}).drop(columns='TeamID', axis=1)
#df_test1 = df_train.dropna()


df_test1['ConfDiff'] = df_test1['A_ConfScore'] - df_test1['B_ConfScore']
df_test1['ScoreDiff'] = df_test1['A_DiffScore'] - df_test1['B_DiffScore']
df_test1['WRDiff'] = df_test1['A_WinRatio'] - df_test1['B_WinRatio']
df_test1['SeedDiff'] = df_test1['A_ConfSeed'] - df_test1['B_ConfSeed']


df_test1 = df_test1.fillna(0)


df_test1['OEDiff'] = (df_test1['A_AvgOE']-df_test1['B_AvgOE'])
df_test1['DEDiff'] = (df_test1['A_AvgDE']-df_test1['B_AvgDE'])
df_test1['ATRDiff'] = (df_test1['A_AvgATR']-df_test1['B_AvgATR'])
df_test1['PosDiff'] = (df_test1['A_AvgPos']-df_test1['B_AvgPos'])
df_test1['ScoreDiff'] = (df_test1['A_DiffScore']-df_test1['B_DiffScore'])
df_test1['eFGDiff'] = (df_test1['A_Avg_eFG']-df_test1['B_Avg_eFG'])

df_test1 = df_test1.drop(columns=['A_AvgOE', 'B_AvgOE','A_AvgDE','B_AvgDE','A_AvgATR','B_AvgATR',
                                  'A_AvgPos', 'B_AvgPos', 'B_AvgPos', 'A_AvgPos', 'B_Avg_eFG', 'A_Avg_eFG'])


model = random_search.best_estimator_
model.fit(df_train[features], df_train["Target"])


pred = model.predict_proba(df_test1[features])[:, 1]
pred = (pred - pred.min()) / (pred.max() - pred.min())

sns.histplot(pred, bins=25)

plt.title('Predictions probability')
plt.show()


# Define the path to save the submission file
submission_path = '/kaggle/working/submission1.csv'

# Read sample submission and update predictions
submission = pd.read_csv(os.path.join(PATH, 'SampleSubmissionStage1.csv'))
submission['Pred'] = pred

# Save submission to desired location
submission.to_csv(submission_path, index=False)




