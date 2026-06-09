#As I say in the notebook, I've tried different things so some of the imports may not be necessary anymore because I ended up not using some code that used them 
import os
import re
import math
import sklearn
import statistics
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns

from collections import Counter
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.decomposition import *
from functools import partial

DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/'

def treat_seed(seed):
    return int(re.sub("[^0-9]", "", seed))

df_rs_results = pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv")
df_seeds = pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv")
df_teams = pd.concat([df_rs_results[['Season','WTeamID']].rename(columns={'WTeamID':'TeamID'}),df_rs_results[['Season','LTeamID']].rename(columns={'LTeamID':'TeamID'})], axis=0).groupby(['Season','TeamID']).count().reset_index()[['Season','TeamID']]
df_teams = df_seeds.merge(df_teams, how="right").fillna('W20')
df_teams['Seed'] = df_teams['Seed'].apply(treat_seed)




    df_massey_ordinals = pd.read_csv(DATA_PATH + "MMasseyOrdinals.csv")
    df_massey_pom = df_massey_ordinals.query('RankingDayNum == 133 and SystemName == "POM"').reset_index()[['Season','TeamID','OrdinalRank']].rename(columns={"OrdinalRank":"Rank"})
    df_massey_mor = df_massey_ordinals.query('RankingDayNum == 133 and SystemName == "MOR"').reset_index()[['Season','TeamID','OrdinalRank']].rename(columns={"OrdinalRank":"Rank"})
    df_massey_rank = pd.concat([df_massey_pom, df_massey_mor], axis=0, sort=False).groupby(['Season', 'TeamID']).mean().reset_index()
    df_teams = df_teams.merge(df_massey_rank, how="left")


# Computing teams stats for the RS
    
def win_value(x):
    home_advg = 3 if x['WLoc'] == 'H' else -3 if x['WLoc'] == 'A' else 0
    if ((x['NumOT'] > 0) & (x['WLoc'] == 'H')) | (x['WScore'] - x['LScore'] - home_advg <= -1):
        return -0.5
    elif ((x['NumOT'] > 0) & (x['WLoc'] != 'H')) | (x['WScore'] - x['LScore'] - home_advg == 1):
        return 0.5
    elif x['WScore'] - x['LScore'] - home_advg == 0:
        return 0
    else:
        return math.log(x['WScore'] - x['LScore'] - home_advg)
        
df_rs_results['WinValue'] = df_rs_results.apply(win_value, axis=1)   


df_rs_results['Games'] = 1 
df_team_stats_w = df_rs_results.groupby(['Season','WTeamID']).sum().reset_index()[['Season','Games','WTeamID','WScore','LScore','WFGA','WFTM','WFTA','WOR','WTO','WinValue']].rename(columns={'WTeamID':'TeamID','WScore':'Score','LScore':'OScore','WFGA':'FGA','WFTM':'FTM','WFTA':'FTA','WOR':'OR','WTO':'TO'})
df_team_stats_l = df_rs_results.groupby(['Season','LTeamID']).sum().reset_index()[['Season','Games','LTeamID','WScore','LScore','LFGA','LFTM','LFTA','LOR','LTO','WinValue']].rename(columns={'LTeamID':'TeamID','LScore':'Score','WScore':'OScore','LFGA':'FGA','LFTM':'FTM','LFTA':'FTA','LOR':'OR','LTO':'TO'})
df_team_stats_l['WinValue'] = - df_team_stats_l['WinValue']
df_team_stats = pd.concat([df_team_stats_w, df_team_stats_l], axis=0, sort=False).groupby(['Season', 'TeamID']).sum().reset_index()
df_team_stats['Poss'] = 0.96*(df_team_stats['FGA']+df_team_stats['TO']+(0.44*df_team_stats['FTA'])-df_team_stats['OR'])
df_team_stats['ORTG'] = df_team_stats['Score']/df_team_stats['Poss']*100 #points scored per 100 possessions
df_team_stats['DRTG'] = df_team_stats['OScore']/df_team_stats['Poss']*100 #points scored per 100 possessions
df_team_stats['NETRTG'] = df_team_stats['ORTG'] - df_team_stats['DRTG']
df_team_stats['FTRate'] = df_team_stats['FTM']/df_team_stats['FGA']
df_team_stats['WinValue'] = df_team_stats['WinValue']/df_team_stats['Games']
df_team_stats = df_team_stats.drop(['Score', 'OScore', 'FGA', 'TO','FTM','FTA','OR','Games','Poss'], axis=1)


#Adjusting WinValue on opponent strength

df_rs_opp_results_won = df_rs_results.merge(df_team_stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=['', 'OppAvg'])[['Season','WTeamID','WinValueOppAvg','WinValue','WScore','LScore','WOR','WFGA','WFTA','WTO','NETRTG']].rename(columns = {"WTeamID":"TeamID",'WScore':'Score','LScore':'OScore','WFGA':'FGA','WFTA':'FTA','WOR':'OR','WTO':'TO'})
df_rs_opp_results_lost = df_rs_results.merge(df_team_stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], suffixes=['', 'OppAvg'])[['Season','LTeamID', 'WinValueOppAvg', 'WinValue','WScore','LScore','LOR','LFGA','LFTA','LTO','ORTG','NETRTG']].rename(columns = {"LTeamID":"TeamID",'LScore':'Score','WScore':'OScore','LFGA':'FGA','LFTA':'FTA','LOR':'OR','LTO':'TO'})
df_rs_opp_results_lost['WinValue'] = - df_rs_opp_results_lost['WinValue']
df_rs_opp_results = pd.concat([df_rs_opp_results_won, df_rs_opp_results_lost], axis=0, sort=False)
df_rs_opp_results['AdjWinValue'] = df_rs_opp_results['WinValue'] + df_rs_opp_results['WinValueOppAvg']
df_rs_opp_results['Poss'] = 0.96*(df_rs_opp_results['FGA']+df_rs_opp_results['TO']+(0.44*df_rs_opp_results['FTA'])-df_rs_opp_results['OR'])
df_rs_opp_results['GameORTG'] = df_rs_opp_results['Score']/df_rs_opp_results['Poss']*100
df_rs_opp_results['GameDRTG'] = df_rs_opp_results['OScore']/df_rs_opp_results['Poss']*100
df_rs_opp_results['AdjNETRTG'] = df_rs_opp_results['GameORTG'] - df_rs_opp_results['GameDRTG'] - df_rs_opp_results['NETRTG']
df_rs_opp_results = df_rs_opp_results.groupby(['Season', 'TeamID']).mean().reset_index()[['Season','TeamID', 'AdjWinValue', 'AdjNETRTG']]
df_team_stats = df_team_stats.merge(df_rs_opp_results, how="left")

#Computing teams' consistency
    
df_rs_own_results_lost = df_rs_results.merge(df_team_stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=['', 'Avg'])[['Season','TeamID','WTeamID','NETRTG','WinValue','WLoc']]
df_rs_own_results_lost = df_rs_own_results_lost.merge(df_team_stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], suffixes=['', '_Opp'])[['Season','TeamID','WTeamID','NETRTG','NETRTG_Opp','WinValue','WLoc']]
df_rs_own_results_lost['HomeAdvg'] = 0
df_rs_own_results_lost.loc[df_rs_own_results_lost['WLoc'] == 'A','HomeAdvg'] = 3
df_rs_own_results_lost.loc[df_rs_own_results_lost['WLoc'] == 'H','HomeAdvg'] = -3
df_rs_own_results_won = df_rs_results.merge(df_team_stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], suffixes=['', 'Avg'])[['Season','TeamID','LTeamID', 'AdjWinValue','NETRTG','WinValue','WLoc']]
df_rs_own_results_won = df_rs_own_results_won.merge(df_team_stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=['', '_Opp'])[['Season','TeamID','LTeamID', 'AdjWinValue','NETRTG','NETRTG_Opp','WinValue','WLoc']]
df_rs_own_results_won['HomeAdvg'] = 0
df_rs_own_results_won.loc[df_rs_own_results_won['WLoc'] == 'H','HomeAdvg'] = 3
df_rs_own_results_won.loc[df_rs_own_results_won['WLoc'] == 'A','HomeAdvg'] = -3
df_rs_own_results_lost['WinValue'] = - df_rs_own_results_lost['WinValue']
df_rs_own_results = pd.concat([df_rs_own_results_won, df_rs_own_results_lost], axis=0, sort=False)
df_rs_own_results['WVStDev'] = statistics.stdev(df_rs_own_results['WinValue'].values)
df_rs_own_results['WVExp'] = (df_rs_own_results['NETRTG'] - df_rs_own_results['NETRTG_Opp'])*0.7 - df_rs_own_results['HomeAdvg'] 
df_rs_own_results['WVExp'] = df_rs_own_results['WVExp'].apply(lambda x: -math.log(-x) if x < -2 else -0.5 if x < 0 else 0 if x==0 else 0.5 if x <= 2 else math.log(x))
df_rs_own_results['WVConsToExp'] = abs(df_rs_own_results['WinValue'] - df_rs_own_results['WVExp'])/df_rs_own_results['WVStDev']
df_rs_own_results = df_rs_own_results[['Season','TeamID', 'WVConsToExp']].groupby(['Season', 'TeamID']).mean().reset_index()
df_team_stats = df_team_stats.merge(df_rs_own_results, how="left")
df_teams = df_teams.merge(df_team_stats, how="left")


df = pd.read_csv(DATA_PATH + "MNCAATourneyCompactResults.csv")
df_submission = pd.read_csv(DATA_PATH + "SampleSubmissionStage2.csv")
df_test = pd.DataFrame({'Season':df_submission['ID'].apply(lambda x: int(x[0:4])), 'WTeamID':df_submission['ID'].apply(lambda x: int(x[5:9])),'LTeamID':df_submission['ID'].apply(lambda x: int(x[10:14])), 'WScore' : 0, 'LScore' : 0})
df = pd.concat([df[df['Season'] < 2020], df[df['Season'] > 2022], df_test], axis=0, sort=False)
df_matchups_won = df_teams.merge(df, left_on=['Season', 'TeamID'], right_on=['Season', 'WTeamID']).drop(['WTeamID','DayNum','NumOT','WLoc'], axis=1)
df_matchups_won = df_matchups_won.merge(df_teams, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID']).drop(['LTeamID'], axis=1)
df_matchups_won['Result'] = 1
df_matchups_won['Score_DIFF'] = df_matchups_won['WScore'] - df_matchups_won['LScore']
df_matchups_lost = df_teams.merge(df, left_on=['Season', 'TeamID'], right_on=['Season', 'LTeamID']).drop(['LTeamID','DayNum','NumOT','WLoc'], axis=1)
df_matchups_lost = df_matchups_lost.merge(df_teams, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']).drop(['WTeamID'], axis=1)
df_matchups_lost['Result'] = 0
df_matchups_lost['Score_DIFF'] = df_matchups_lost['LScore'] - df_matchups_lost['WScore']
df_matchups = pd.concat([df_matchups_won, df_matchups_lost], axis=0, sort=False)
df_matchups['ID'] = df_matchups['Season'].apply(str) + '_' + df_matchups['TeamID_x'].apply(str) + '_' + df_matchups['TeamID_y'].apply(str)
df_matchups['Seed_DIFF'] = df_matchups['Seed_x'] - df_matchups['Seed_y']
df_matchups['AdjWinValue_DIFF'] = df_matchups['AdjWinValue_x'] - df_matchups['AdjWinValue_y']
df_matchups['Rank_DIFF'] = df_matchups['Rank_x'] - df_matchups['Rank_y']
df_matchups['FTRate_DIFF'] = df_matchups['FTRate_x'] - df_matchups['FTRate_y']
df_matchups['WVConsToExp_DIFF'] = df_matchups['WVConsToExp_x'] - df_matchups['WVConsToExp_y']


def kfold(df, keys, features, df_test_=None, verbose=0):
    
    keys_values = df.loc[:, keys].reset_index(drop=True)
    x = df.loc[:, features].values
    y = df.loc[:,['Score_DIFF', 'Result']].reset_index(drop=True)
    
    x = StandardScaler().fit_transform(x)
    
    df = pd.DataFrame(data = x, columns = features)
    df = pd.concat([keys_values, df, y], axis=1)

    seasons = df['Season'].unique()
    cvs = []
    pred_tests = []
    target = 'Score_DIFF'
    
    for season in seasons[1:]:
        if verbose:
            print(f'\nValidating on season {season}')
        
        df_train = df[df['Season'] != season].reset_index(drop=True).copy()
        df_val = df[df['Season'] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy() if df_test_ is not None else None
        
        model = BayesianRidge(n_iter=1000)
        model.fit(df_train[features], df_train[target])
        pred = model.predict(df_val[features])
        pred = margin_to_proba(pred, 0.688,1.188, 39)
      
        if df_test is not None:
            test_x = df_test.loc[:, features].values
            test_x = StandardScaler().fit_transform(test_x)
            df_test = pd.DataFrame(data = test_x, columns = features)
            pred_test = model.predict(df_test)
            pred_test = margin_to_proba(pred_test, 0.688,1.188, 39)   
            pred_tests.append(pred_test)
        
        score = ((df_val['Result'].values - pred) ** 2).mean()
        cvs.append(score)

        if verbose:
            print(f'\t -> Scored {score:.6f}')
    
    print(f'\n Local CV is {np.mean(cvs):.6f} with stdev {statistics.stdev(cvs):.6f}')
    
    return pred_tests


def margin_to_proba(x, exp, scaler_factor, max_sure):
    def filter_sign(x, exp):
        if x > 0:
            return math.pow(x, exp)
        elif x < 0:
            return -math.pow(abs(x), exp)
        else:
            return 0
    
    scaler = math.pow(max_sure, exp)*scaler_factor
    y = np.vectorize(filter_sign)(x, exp)/scaler + 0.5

    return np.clip(y, 0, 1)


keys_m = ['Season', 'TeamID_x','TeamID_y']
features_m = ['Seed_DIFF','Rank_DIFF','AdjWinValue_DIFF','WVConsToExp_DIFF', 'FTRate_DIFF']
#keys_w = ['Season', 'TeamID_x','TeamID_y']
#features_w = ['Seed_DIFF','AdjWinValue_DIFF','WVConsToExp_DIFF', 'FTRate_DIFF']

df_test_m = df_matchups.loc[(df_matchups['Season'] == 2025) & (df_matchups['TeamID_x'] < df_matchups['TeamID_y'])]
#df_test_w = df_matchups_w.loc[(df_matchups_w['Season'] == 2025) & (df_matchups_w['TeamID_x'] < df_matchups_w['TeamID_y'])]
pred_tests_m = kfold(df = df_matchups.loc[df_matchups['Season'] < 2025], df_test_ = df_test_m, keys = keys_m, features = features_m, verbose=1)
#pred_tests_w = kfold(df = df_matchups_w.loc[df_matchups_w['Season'] < 2025], df_test_ = df_test_w, keys = keys_w, features = features_w, verbose=1)
pred_test_m = np.mean(pred_tests_m, 0)
#pred_test_w = np.mean(pred_tests_w, 0)
df_test_m['Pred'] = pred_test_m
#df_test_w['Pred'] = pred_test_w
#df_test = pd.concat([df_test_m, df_test_w], axis=0, sort=False)

df_test_m[['ID','Pred']].sort_values(by='ID').to_csv('Submission.csv', index=False)


