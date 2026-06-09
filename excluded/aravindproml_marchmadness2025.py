import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, ElasticNet as ENC
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, ElasticNet as ENR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, classification_report, r2_score, mean_squared_error, brier_score_loss
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from tensorflow.keras import layers, optimizers
import tensorflow as tf
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.mixture import GaussianMixture
import os
import pyarrow.parquet as pa
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
import polars as pl
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
# from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer


import pandas as pd
dfreg = pd.concat([
                pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv'),
                pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv')
            ], ignore_index=True)
dftourney = pd.concat([
                pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv'),
                pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyDetailedResults.csv')
             ], ignore_index=True)

# TODO: Check if using secondary tourney helps because we aren't using tourney scores anyways
# dfsec = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyCompactResults.csv')
# dfsec['Type'] = 'Secondary'
# dfsec = dfsec.drop(columns='SecondaryTourney')

cities = pd.concat([
            pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MGameCities.csv'),
            pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WGameCities.csv')
        ], ignore_index=True)


dfmatches = dftourney.merge(cities[['Season', 'DayNum', 'WTeamID', 'LTeamID', 'CityID']], on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], how='left')
dfmatches[['Team1', 'Team2', 'Win', 'MLoc']] = dfmatches.apply(lambda match: pd.Series({'Team1': min(match['WTeamID'], match['LTeamID']), 
                                                                                'Team2': max(match['WTeamID'], match['LTeamID']), 
                                                                                'Win': 1 if match['WTeamID']<match['LTeamID'] else 0,
                                                                                'MLoc':  1 if (match['WLoc'] == 'H' and match['WTeamID'] < match['LTeamID']) or 
                                                                                          (match['WLoc'] == 'A' and match['WTeamID'] > match['LTeamID']) 
                                                                                     else -1 if (match['WLoc'] == 'A' and match['WTeamID'] < match['LTeamID']) or 
                                                                                              (match['WLoc'] == 'H' and match['WTeamID'] > match['LTeamID']) 
                                                                                     else 0
                                                                               }), axis=1)
mainfeat = ['Season', 'DayNum','MLoc','CityID','Team1', 'Team2', 'Win']

dfmain = dfmatches[mainfeat]
dfswap = dfmain.rename(columns={'Team1':"Team2", "Team2":"Team1"})
dfswap["Win"] = 1-dfswap["Win"]
dfswap["MLoc"] = -dfswap["MLoc"]

dfmatches = pd.concat([dfswap,dfmain], ignore_index=True).sort_values(['Season','DayNum'], ignore_index=True)


dfreg['WFG%'] = dfreg['WFGM']/dfreg['WFGA']
dfreg['LFG%'] = dfreg['LFGM']/dfreg['LFGA']
dfreg['WFG3%'] = dfreg['WFGM3']/dfreg['WFGA3']
dfreg['LFG3%'] = dfreg['LFGM3']/dfreg['LFGA3']
dfreg['WFT%'] = dfreg['WFTM']/dfreg['WFTA']
dfreg['LFT%'] = dfreg['LFTM']/dfreg['LFTA']
dfreg["WScoreDiff"] = dfreg['WScore']-dfreg['LScore']
dfreg["LScoreDiff"] = dfreg['LScore']-dfreg['WScore']

# regular_results['WEFFG'] = regular_results['WFGM'] / regular_results['WFGA']
# regular_results['WEFFG3'] = regular_results['WFGM3'] / regular_results['WFGA3']
dfreg['WDARE'] = dfreg['WFGM3'] / dfreg['WFGM']
dfreg['WTOQUETOQUE'] = dfreg['WAst'] / dfreg['WFGM']

# regular_results['LEFFG'] = regular_results['LFGM'] / regular_results['LFGA']
# regular_results['LEFFG3'] = regular_results['LFGM3'] / regular_results['LFGA3']
dfreg['LDARE'] = dfreg['LFGM3'] / dfreg['LFGM']
dfreg['LTOQUETOQUE'] = dfreg['LAst'] / dfreg['LFGM']

seasonstats = ['TOQUETOQUE','DARE','Score','FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR','Ast','TO','Stl','Blk','PF','FG%','FG3%','FT%', "ScoreDiff"]
# seasonstats = ['Score','FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR','Ast','TO','Stl','Blk','PF']

df_win = dfreg[["Season","WTeamID"]+[f"W{i}" for i in seasonstats]].rename(columns={"WTeamID":"TeamID"}|{f"W{i}":i for i in seasonstats})
df_win['Win']=1

df_loss = dfreg[["Season","LTeamID"]+[f"L{i}" for i in seasonstats]].rename(columns={"LTeamID":"TeamID"}|{f"L{i}":i for i in seasonstats})
df_loss["Win"] = 0

df_reg_stats = pd.concat([df_win, df_loss])
df_reg_stats = df_reg_stats.groupby(["Season","TeamID"]).mean().reset_index().rename(columns={"Win":"SeasonWins"}|{i:f"Season{i}" for i in seasonstats})

dfmatches = dfmatches.merge(df_reg_stats, left_on=['Team1',"Season"], right_on=["TeamID","Season"]).rename(columns={"SeasonWins":"Team1SeasonWins"}|{f"Season{i}":f"Team1Season{i}" for i in seasonstats}).drop(columns="TeamID")
dfmatches = dfmatches.merge(df_reg_stats, left_on=['Team2',"Season"], right_on=["TeamID","Season"]).rename(columns={"SeasonWins":"Team2SeasonWins"}|{f"Season{i}":f"Team2Season{i}" for i in seasonstats}).drop(columns="TeamID")


dfmatches


# Add the stats of last 14 days of the season 132-14=118
dftemp = dfreg[dfreg['DayNum']>118]

seasonstats = ['DARE', 'Score','FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR','Ast','TO','Stl','Blk','PF','FG%','FG3%','FT%', "ScoreDiff"]
# seasonstats = ['Score','FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR','Ast','TO','Stl','Blk','PF']

df_win = dftemp[["Season","WTeamID"]+[f"W{i}" for i in seasonstats]].rename(columns={"WTeamID":"TeamID"}|{f"W{i}":i for i in seasonstats})
df_win['Win']=1

df_loss = dftemp[["Season","LTeamID"]+[f"L{i}" for i in seasonstats]].rename(columns={"LTeamID":"TeamID"}|{f"L{i}":i for i in seasonstats})
df_loss["Win"] = 0

df_last_stats = pd.concat([df_win, df_loss])
df_last_stats = df_last_stats.groupby(["Season","TeamID"]).mean().reset_index().rename(columns={"Win":"Last14DaysWins"}|{i:f"Last14Days{i}" for i in seasonstats})

dfmatches = dfmatches.merge(df_last_stats, left_on=['Team1',"Season"], right_on=["TeamID","Season"]).rename(columns={"Last14DaysWins":"Team1Last14DaysWins"}|{f"Last14Days{i}":f"Team1Last14Days{i}" for i in seasonstats}).drop(columns="TeamID")
dfmatches = dfmatches.merge(df_last_stats, left_on=['Team2',"Season"], right_on=["TeamID","Season"]).rename(columns={"Last14DaysWins":"Team2Last14DaysWins"}|{f"Last14Days{i}":f"Team2Last14Days{i}" for i in seasonstats}).drop(columns="TeamID")


pd.set_option('display.max_columns', None)
dfmatches.head()


# # Previous Tourney Stats
# seasonstats = ['Score','FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR','Ast','TO','Stl','Blk','PF']

# df_win = dftourney[["Season","WTeamID"]+[f"W{i}" for i in seasonstats]].rename(columns={"WTeamID":"TeamID"}|{f"W{i}":i for i in seasonstats})
# df_win['Win']=1

# df_loss = dftourney[["Season","LTeamID"]+[f"L{i}" for i in seasonstats]].rename(columns={"LTeamID":"TeamID"}|{f"L{i}":i for i in seasonstats})
# df_loss["Win"] = 0

# df_stats = pd.concat([df_win, df_loss])
# df_stats = df_stats.groupby(["Season","TeamID"]).mean().reset_index().rename(columns={"Win":"TourneyWins"}|{i:f"Tourney{i}" for i in seasonstats})
# df_stats['Season'] += 1

# dfmatches = dfmatches.merge(df_stats, left_on=["Season","Team1"], right_on=["Season","TeamID"]).rename(columns={"TourneyWins":"Team1TourneyWins"}|{f"Tourney{i}":f"Team1Tourney{i}" for i in seasonstats}).drop(columns=['TeamID'])
# dfmatches = dfmatches.merge(df_stats, left_on=["Season","Team2"], right_on=["Season","TeamID"]).rename(columns={"TourneyWins":"Team2TourneyWins"}|{f"Tourney{i}":f"Team2Tourney{i}" for i in seasonstats}).drop(columns=['TeamID'])
# dfmatches.head()


# Add seeds
dfseeds = pd.concat([
                pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv'),
                pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
            ])
dfseeds['Seedval'] = dfseeds['Seed'].apply(lambda x: int(x[1:3]))

dfmatches = dfmatches.merge(dfseeds[["Season","TeamID","Seedval"]], left_on=["Season","Team1"], right_on=['Season','TeamID']).rename(columns={"Seedval":"Team1Seedval"}).drop(columns=["TeamID"])
dfmatches = dfmatches.merge(dfseeds[["Season","TeamID","Seedval"]], left_on=["Season","Team2"], right_on=['Season','TeamID']).rename(columns={"Seedval":"Team2Seedval"}).drop(columns=["TeamID"])
dfmatches.head()


# Add team quality
from concurrent.futures import ThreadPoolExecutor

def teamquality(season, df):
    print(season)
    formula = 'Win~-1+Team1+Team2'
    
    glm = sm.GLM.from_formula(formula=formula, 
                              data=df[df['Season']==season],
                              family=sm.families.Binomial()).fit()
    
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    quality['quality'] = (quality['quality']-quality['quality'].mean())/quality['quality'].std()
    # display(quality)
    #quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('Team1')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[6:10]).astype(int)
    # quality.to_csv(f"teamquality{season}.csv")
    return quality

dfregteam = dfreg[['Season','WTeamID',"LTeamID"]].rename(columns={"WTeamID":"Team1","LTeamID":"Team2"})
dfregteam['Win']=1
dfswap = dfregteam.rename(columns={"WTeamID":"Team2","LTeamID":"Team1"})
dfswap['Win'] = 0
dfregteam = pd.concat([dfregteam, dfswap]).astype({"Team1":'category',"Team2":"category"})
dfregteam['Win'] = (dfregteam['Win']-dfregteam['Win'].mean())/dfregteam['Win'].std()

dfregwteam = dfregteam[(dfregteam['Team1'].astype(int)>=3000)&(dfregteam['Team2'].astype(int)>=3000)].reset_index()
dfregwteam['Team1'] = dfregwteam['Team1'].cat.remove_unused_categories()
dfregwteam['Team2'] = dfregwteam['Team2'].cat.remove_unused_categories()

dfregmteam = dfregteam[(dfregteam['Team1'].astype(int)<3000)&(dfregteam['Team2'].astype(int)<3000)].reset_index()
dfregmteam['Team1'] = dfregmteam['Team1'].cat.remove_unused_categories()
dfregmteam['Team2'] = dfregmteam['Team2'].cat.remove_unused_categories()


# for season in range(2003, 2026):
#     team_quality(season)
# seasons = range(2003, 2026)
# with ThreadPoolExecutor() as executor:
#     executor.map(team_quality, seasons)
team_quality_m = pd.concat([teamquality(season, dfregmteam) for season in range(2010, 2026)])
team_quality_w = pd.concat([teamquality(season, dfregwteam) for season in range(2010, 2026)])
team_quality = pd.concat([team_quality_m, team_quality_w])


team_quality.to_csv('combquality.csv')


dfmatches = dfmatches.merge(team_quality, left_on=['Season','Team1'], right_on=['Season','TeamID']).rename(columns={'quality':'Team1quality'}).drop(columns=['TeamID'])
dfmatches = dfmatches.merge(team_quality, left_on=['Season','Team2'], right_on=['Season','TeamID']).rename(columns={'quality':'Team2quality'}).drop(columns=['TeamID'])


# Add gap features
for i in seasonstats+['Wins',"ScoreDiff"]:
    dfmatches[f"Season{i}Gap"] = dfmatches[f"Team1Season{i}"]-dfmatches[f"Team2Season{i}"]
    dfmatches[f"Last14Days{i}Gap"] = dfmatches[f"Team1Last14Days{i}"]-dfmatches[f"Team2Last14Days{i}"]
    # dfmatches[f"Season{i}Ratio"] = dfmatches[f"Team1Season{i}"]/dfmatches[f"Team2Season{i}"]
    # dfmatches[f"Last14Days{i}Ratio"] = dfmatches[f"Team1Last14Days{i}"]/dfmatches[f"Team2Last14Days{i}"]
dfmatches['qualitydiff'] = dfmatches['Team1quality']-dfmatches['Team2quality']
dfmatches['seeddiff'] = dfmatches['Team1Seedval']-dfmatches['Team2Seedval']


dfmmas = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv')
for sys in ['MAS', 'POM']:
    # .reset_index(ignore_index=True)
    dfmatches = dfmatches.merge(dfmmas.query(f"SystemName=='{sys}' and RankingDayNum == 133").drop(columns=['RankingDayNum','SystemName']), left_on=['Season','Team1'], right_on=['Season','TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'OrdinalRank':f'Team1{sys}Rank'})
    dfmatches = dfmatches.merge(dfmmas.query(f"SystemName=='{sys}' and RankingDayNum == 133").drop(columns=['RankingDayNum','SystemName']), left_on=['Season','Team2'], right_on=['Season','TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'OrdinalRank':f'Team2{sys}Rank'}).fillna(-1)
    dfmatches[f'{sys}RankDiff'] = dfmatches[f'Team1{sys}Rank']-dfmatches[f'Team2{sys}Rank']


dfmatches


dfmatches['CityID'] = dfmatches['CityID'].fillna(-1).astype('int')
dfmatches = dfmatches.astype({col:'category' for col in ['CityID', "Team1", "Team2", 'MLoc']})


# out = dfmatches[featurestrain+['Win']].corr()[['Win']].abs().sort_values(by='Win')
# bestfeatures = list(out[out['Win']>=0.1].index.drop('Win'))


from sklearn.ensemble import StackingClassifier

# def cauchy_loss(labels, preds, sigma=1.0):
#     """
#     Cauchy loss function for XGBoost.
    
#     Args:
#         preds: Predictions from the model.
#         dtrain: DMatrix of training data.
#         sigma: Scale parameter, controls robustness.
    
#     Returns:
#         grad: Gradient.
#         hess: Hessian (second derivative).
#     """
#     residuals = preds - labels
    
#     # Cauchy loss gradient: ∂L/∂r = 2 * r / (sigma^2 + r^2)
#     grad = 2 * residuals / (sigma**2 + residuals**2)
    
#     # Cauchy loss hessian: ∂²L/∂r² = 2 * (sigma^2 - r^2) / (sigma^2 + r^2)^2
#     hess = 2 * (sigma**2 - residuals**2) / (sigma**2 + residuals**2)**2
    
#     return grad, hess
    
featurestrain = [
    'Season', 
    # 'Team2', 'Team1',
    'Team1SeasonScore', 'Team1SeasonFGM', 'Team1SeasonFGA', 'Team1SeasonFGM3', 'Team1SeasonFGA3', 'Team1SeasonFTM', 'Team1SeasonFTA', 'Team1SeasonOR', 'Team1SeasonDR', 'Team1SeasonAst', 'Team1SeasonTO', 'Team1SeasonStl', 'Team1SeasonBlk', 'Team1SeasonPF', 'Team1SeasonFG%', 'Team1SeasonFG3%', 'Team1SeasonFT%', 'Team1SeasonScoreDiff', 'Team1SeasonWins', 'Team1Last14DaysScore', 'Team1Last14DaysFGM', 'Team1Last14DaysFGA', 'Team1Last14DaysFGM3', 'Team1Last14DaysFGA3', 'Team1Last14DaysFTM', 'Team1Last14DaysFTA', 'Team1Last14DaysOR', 'Team1Last14DaysDR', 'Team1Last14DaysAst', 'Team1Last14DaysTO', 'Team1Last14DaysStl', 'Team1Last14DaysBlk', 'Team1Last14DaysPF', 'Team1Last14DaysFG%', 'Team1Last14DaysFG3%', 'Team1Last14DaysFT%', 'Team1Last14DaysScoreDiff', 'Team1Last14DaysWins','Team1Seedval', 'Team1quality', 
    'Team1MASRank', 'Team1POMRank', 
    'Team2SeasonScore', 'Team2SeasonFGM', 'Team2SeasonFGA', 'Team2SeasonFGM3', 'Team2SeasonFGA3', 'Team2SeasonFTM', 'Team2SeasonFTA', 'Team2SeasonOR', 'Team2SeasonDR', 'Team2SeasonAst', 'Team2SeasonTO', 'Team2SeasonStl', 'Team2SeasonBlk', 'Team2SeasonPF', 'Team2SeasonFG%', 'Team2SeasonFG3%', 'Team2SeasonFT%', 'Team2SeasonScoreDiff', 'Team2SeasonWins',  'Team2Last14DaysScore', 'Team2Last14DaysFGM', 'Team2Last14DaysFGA', 'Team2Last14DaysFGM3', 'Team2Last14DaysFGA3', 'Team2Last14DaysFTM', 'Team2Last14DaysFTA', 'Team2Last14DaysOR', 'Team2Last14DaysDR', 'Team2Last14DaysAst', 'Team2Last14DaysTO', 'Team2Last14DaysStl', 'Team2Last14DaysBlk', 'Team2Last14DaysPF', 'Team2Last14DaysFG%', 'Team2Last14DaysFG3%', 'Team2Last14DaysFT%', 'Team2Last14DaysScoreDiff', 'Team2Last14DaysWins', 'Team2Seedval',  'Team2quality', 
    'Team2MASRank', 'Team2POMRank',
    'SeasonScoreGap',  'SeasonFGMGap',  'SeasonFGAGap', 'SeasonFGM3Gap',  'SeasonFGA3Gap',  'SeasonFTMGap',  'SeasonFTAGap', 'SeasonORGap',  'SeasonDRGap',  'SeasonAstGap',  'SeasonTOGap',  'SeasonStlGap',  'SeasonBlkGap',  'SeasonPFGap',  'SeasonFG%Gap', 'SeasonFG3%Gap',  'SeasonFT%Gap',  'SeasonScoreDiffGap',  'SeasonWinsGap',  
    'Last14DaysScoreGap', 'Last14DaysFGMGap', 'Last14DaysFGAGap', 'Last14DaysFGM3Gap', 'Last14DaysFGA3Gap', 'Last14DaysFTMGap', 'Last14DaysFTAGap', 'Last14DaysORGap','Last14DaysDRGap', 'Last14DaysTOGap','Last14DaysAstGap', 'Last14DaysStlGap','Last14DaysBlkGap', 'Last14DaysPFGap', 'Last14DaysFG%Gap', 'Last14DaysFG3%Gap', 'Last14DaysFT%Gap',  'Last14DaysWinsGap',
    'Last14DaysScoreDiffGap',
    # 'SeasonScoreRatio',  'SeasonFGMRatio',  'SeasonFGARatio', 'SeasonFGM3Ratio',  'SeasonFGA3Ratio',  'SeasonFTMRatio',  'SeasonFTARatio', 'SeasonORRatio',  'SeasonDRRatio',  'SeasonAstRatio',  'SeasonTORatio',  'SeasonStlRatio',  'SeasonBlkRatio',  'SeasonPFRatio',  'SeasonFG%Ratio', 'SeasonFG3%Ratio',  'SeasonFT%Ratio',  'SeasonScoreDiffRatio',  
    # 'Last14DaysScoreRatio', 'Last14DaysFGMRatio', 'Last14DaysFGARatio', 'Last14DaysFGM3Ratio', 'Last14DaysFGA3Ratio', 'Last14DaysFTMRatio', 'Last14DaysFTARatio', 'Last14DaysORRatio','Last14DaysDRRatio', 'Last14DaysTORatio','Last14DaysAstRatio', 'Last14DaysStlRatio','Last14DaysBlkRatio', 'Last14DaysPFRatio', 'Last14DaysFG%Ratio', 'Last14DaysFG3%Ratio', 'Last14DaysFT%Ratio', 
    'seeddiff',  'qualitydiff', 
    'MASRankDiff', 'POMRankDiff'
]
xgbparam = {
    # "alpha":1, # Def 0
    # "lambda":2, # Default 1
    "max_depth":3, # Def 6
    # "eta":0.1, 
    # "objective":"binary:hinge",  # Ensures sigmoid output
    # "eval_metric":"logloss",
    "enable_categorical":True,
    # "gpu_id":0  # Specify GPU ID
}
lgbmparam={
    "objective":"binary",  # Ensures sigmoid output
    "metric":"binary_logloss",
    "verbose": -1,
    "max_depth":3,
    #  "device":"gpu",
    #  "max_bin": 155,
    # "gpu_platform_id": 0,  # Select GPU platform (optional)
    # "gpu_device_id": 0  # Specify GPU ID (optional)
}

tot=0
score = 0
for year in range(2021, 2026):
    dftrain = dfmatches[dfmatches["Season"]<year]
    dftest = dfmatches[dfmatches["Season"]==year]

    if(len(dftest)==0):
        continue
    tot+=1

    param = {"enable_categorical":True,'eval_metric': 'mae', 'booster': 'gbtree', 'eta': 0.05, 'subsample': 0.35, 'colsample_bytree': 0.7, 'num_parallel_tree': 3, 'min_child_weight': 40, 'gamma': 10, 'max_depth': 3,
             # "objective":cauchyobj
            }
    # param = {"enable_categorical":True, 'lambda': 0.03214602684300908, 'alpha': 9.7328013646586, 'max_depth': 14, 'learning_rate': 0.014054728192394284, 'n_estimators': 227, 'min_child_weight': 9, 'subsample': 0.568262286515593, 'colsample_bytree': 0.9228116097459054, 'gamma': 0.5465106408450069}
    # base_regressors = [
    #     ('xgb', XGBClassifier(**param)),
    #     ("lgbm", LGBMClassifier(max_depth=3, verbose=-1)),
    #     ("rf", RandomForestClassifier()),
    #     # ("cat", CatBoostClassifier(verbose=False)),
    #     ("gbdt", GradientBoostingClassifier())
    # ]
    
    # Define the meta-regressor
    # meta_regressor = XGBRegressor(objective="binary:logistic",  enable_categorical=True)
    # meta_regressor = LinearRegression()
    # meta_regressor = LogisticRegression()
    # meta_regressor = XGBClassifier()
    
    # Create the StackingRegressor
    # model = StackingClassifier(
    #     estimators=base_regressors,
    #     final_estimator=meta_regressor,
    #     passthrough=False
    # )

    model = XGBRegressor(
        **param, 
         objective="binary:logistic"
        )
    # model = LGBMRegressor(**lgbmparam)
    # model = CatBoostRegressor(verbose=False, cat_features=['CityID',"MLoc",'Team2', 'Team1'])
    # model = GradientBoostingRegressor()
    # model = RandomForestRegressor()
    # model = XGBRegressor(**xgbparam)
    # model = AdaBoostRegressor()
    
    # model = XGBClassifier(**param)
    # model = LGBMClassifier(**lgbmparam)
    # model = LGBMClassifier(max_depth=2, verbose=-1)
    # model = CatBoostClassifier(verbose=False, cat_features=['CityID','Team2', 'Team1','MLoc'])
    # model = RandomForestClassifier()
    # model = GradientBoostingClassifier()
    # model = AdaBoostClassifier()
    
    model.fit(dftrain[featurestrain], dftrain['Win'])
    
    print(f"{year}: Train:{brier_score_loss(dftrain['Win'], model.predict(dftrain[featurestrain]).clip(0, 1))} Test:{brier_score_loss(dftest['Win'], model.predict(dftest[featurestrain]).clip(0, 1))}")
    score += brier_score_loss(dftest['Win'], model.predict(dftest[featurestrain]).clip(0, 1))
    # print(f"{year}: Train:{brier_score_loss(dftrain['Win'], model.predict_proba(dftrain[featurestrain])[:,1])} Test:{brier_score_loss(dftest['Win'], model.predict_proba(dftest[featurestrain])[:,1])}")
    # score += brier_score_loss(dftest['Win'], model.predict_proba(dftest[featurestrain])[:,1])
print(score/tot)


# Shap feature importance
import shap

dfshap = dftrain.copy()

# Convert categorical columns to numerical codes
for col in dfshap.select_dtypes(include=['category']).columns:
    dfshap[col] = dfshap[col].cat.codes
dfshap = dfshap.astype({col:'category' for col in ['CityID', 'MLoc', 'CityID', "Team1", "Team2"]})
# dfshap.dtypes

explainer = shap.Explainer(model,dftrain[featurestrain])
shap_values = explainer(dftrain[featurestrain], check_additivity=False)
shap.summary_plot(shap_values, dftrain[featurestrain])


from sklearn.ensemble import StackingClassifier

# def cauchy_loss(labels, preds, sigma=1.0):
#     """
#     Cauchy loss function for XGBoost.
    
#     Args:
#         preds: Predictions from the model.
#         dtrain: DMatrix of training data.
#         sigma: Scale parameter, controls robustness.
    
#     Returns:
#         grad: Gradient.
#         hess: Hessian (second derivative).
#     """
#     residuals = preds - labels
    
#     # Cauchy loss gradient: ∂L/∂r = 2 * r / (sigma^2 + r^2)
#     grad = 2 * residuals / (sigma**2 + residuals**2)
    
#     # Cauchy loss hessian: ∂²L/∂r² = 2 * (sigma^2 - r^2) / (sigma^2 + r^2)^2
#     hess = 2 * (sigma**2 - residuals**2) / (sigma**2 + residuals**2)**2
    
#     return grad, hess
    
featurestrain = [
    'Season', 
    # 'Team2', 'Team1',
    'Team1SeasonScore', 'Team1SeasonFGM', 'Team1SeasonFGA', 'Team1SeasonFGM3', 'Team1SeasonFGA3', 'Team1SeasonFTM', 'Team1SeasonFTA', 'Team1SeasonOR', 'Team1SeasonDR', 'Team1SeasonAst', 'Team1SeasonTO', 'Team1SeasonStl', 'Team1SeasonBlk', 'Team1SeasonPF', 'Team1SeasonFG%', 'Team1SeasonFG3%', 'Team1SeasonFT%', 'Team1SeasonScoreDiff', 'Team1SeasonWins', 'Team1Last14DaysScore', 'Team1Last14DaysFGM', 'Team1Last14DaysFGA', 'Team1Last14DaysFGM3', 'Team1Last14DaysFGA3', 'Team1Last14DaysFTM', 'Team1Last14DaysFTA', 'Team1Last14DaysOR', 'Team1Last14DaysDR', 'Team1Last14DaysAst', 'Team1Last14DaysTO', 'Team1Last14DaysStl', 'Team1Last14DaysBlk', 'Team1Last14DaysPF', 'Team1Last14DaysFG%', 'Team1Last14DaysFG3%', 'Team1Last14DaysFT%', 'Team1Last14DaysScoreDiff', 'Team1Last14DaysWins','Team1Seedval', 'Team1quality', 
    'Team1MASRank', 'Team1POMRank', 
    'Team2SeasonScore', 'Team2SeasonFGM', 'Team2SeasonFGA', 'Team2SeasonFGM3', 'Team2SeasonFGA3', 'Team2SeasonFTM', 'Team2SeasonFTA', 'Team2SeasonOR', 'Team2SeasonDR', 'Team2SeasonAst', 'Team2SeasonTO', 'Team2SeasonStl', 'Team2SeasonBlk', 'Team2SeasonPF', 'Team2SeasonFG%', 'Team2SeasonFG3%', 'Team2SeasonFT%', 'Team2SeasonScoreDiff', 'Team2SeasonWins',  'Team2Last14DaysScore', 'Team2Last14DaysFGM', 'Team2Last14DaysFGA', 'Team2Last14DaysFGM3', 'Team2Last14DaysFGA3', 'Team2Last14DaysFTM', 'Team2Last14DaysFTA', 'Team2Last14DaysOR', 'Team2Last14DaysDR', 'Team2Last14DaysAst', 'Team2Last14DaysTO', 'Team2Last14DaysStl', 'Team2Last14DaysBlk', 'Team2Last14DaysPF', 'Team2Last14DaysFG%', 'Team2Last14DaysFG3%', 'Team2Last14DaysFT%', 'Team2Last14DaysScoreDiff', 'Team2Last14DaysWins', 'Team2Seedval',  'Team2quality', 
    'Team2MASRank', 'Team2POMRank',
    'SeasonScoreGap',  'SeasonFGMGap',  'SeasonFGAGap', 'SeasonFGM3Gap',  'SeasonFGA3Gap',  'SeasonFTMGap',  'SeasonFTAGap', 'SeasonORGap',  'SeasonDRGap',  'SeasonAstGap',  'SeasonTOGap',  'SeasonStlGap',  'SeasonBlkGap',  'SeasonPFGap',  'SeasonFG%Gap', 'SeasonFG3%Gap',  'SeasonFT%Gap',  'SeasonScoreDiffGap',  'SeasonWinsGap',  
    'Last14DaysScoreGap', 'Last14DaysFGMGap', 'Last14DaysFGAGap', 'Last14DaysFGM3Gap', 'Last14DaysFGA3Gap', 'Last14DaysFTMGap', 'Last14DaysFTAGap', 'Last14DaysORGap','Last14DaysDRGap', 'Last14DaysTOGap','Last14DaysAstGap', 'Last14DaysStlGap','Last14DaysBlkGap', 'Last14DaysPFGap', 'Last14DaysFG%Gap', 'Last14DaysFG3%Gap', 'Last14DaysFT%Gap',  'Last14DaysWinsGap',
    'Last14DaysScoreDiffGap',
    # 'SeasonScoreRatio',  'SeasonFGMRatio',  'SeasonFGARatio', 'SeasonFGM3Ratio',  'SeasonFGA3Ratio',  'SeasonFTMRatio',  'SeasonFTARatio', 'SeasonORRatio',  'SeasonDRRatio',  'SeasonAstRatio',  'SeasonTORatio',  'SeasonStlRatio',  'SeasonBlkRatio',  'SeasonPFRatio',  'SeasonFG%Ratio', 'SeasonFG3%Ratio',  'SeasonFT%Ratio',  'SeasonScoreDiffRatio',  
    # 'Last14DaysScoreRatio', 'Last14DaysFGMRatio', 'Last14DaysFGARatio', 'Last14DaysFGM3Ratio', 'Last14DaysFGA3Ratio', 'Last14DaysFTMRatio', 'Last14DaysFTARatio', 'Last14DaysORRatio','Last14DaysDRRatio', 'Last14DaysTORatio','Last14DaysAstRatio', 'Last14DaysStlRatio','Last14DaysBlkRatio', 'Last14DaysPFRatio', 'Last14DaysFG%Ratio', 'Last14DaysFG3%Ratio', 'Last14DaysFT%Ratio', 
    'seeddiff',  'qualitydiff', 
    'MASRankDiff', 'POMRankDiff'
]
xgbparam = {
    # "alpha":1, # Def 0
    # "lambda":2, # Default 1
    "max_depth":3, # Def 6
    # "eta":0.1, 
    # "objective":"binary:hinge",  # Ensures sigmoid output
    # "eval_metric":"logloss",
    "enable_categorical":True,
    # "gpu_id":0  # Specify GPU ID
}
lgbmparam={
    "objective":"binary",  # Ensures sigmoid output
    "metric":"binary_logloss",
    "verbose": -1,
    "max_depth":3,
    #  "device":"gpu",
    #  "max_bin": 155,
    # "gpu_platform_id": 0,  # Select GPU platform (optional)
    # "gpu_device_id": 0  # Specify GPU ID (optional)
}

tot=0
score = 0
for year in range(2021, 2026):
    dftrain = dfmatches[dfmatches["Season"]<year]
    dftest = dfmatches[dfmatches["Season"]==year]

    if(len(dftest)==0):
        continue
    tot+=1

    param = {"enable_categorical":True,'eval_metric': 'mae', 'booster': 'gbtree', 'eta': 0.05, 'subsample': 0.35, 'colsample_bytree': 0.7, 'num_parallel_tree': 3, 'min_child_weight': 40, 'gamma': 10, 'max_depth': 3,
             # "objective":cauchyobj
            }
    # param = {"enable_categorical":True, 'lambda': 0.03214602684300908, 'alpha': 9.7328013646586, 'max_depth': 14, 'learning_rate': 0.014054728192394284, 'n_estimators': 227, 'min_child_weight': 9, 'subsample': 0.568262286515593, 'colsample_bytree': 0.9228116097459054, 'gamma': 0.5465106408450069}
    base_regressors = [
        ('xgb', XGBRegressor(**param)),
        ("lgbm", LGBMRegressor(max_depth=3, verbose=-1)),
        ("rf", RandomForestRegressor()),
        # ("cat", CatBoostClassifier(verbose=False)),
        ("gbdt", GradientBoostingRegressor())
    ]
    
    # Define the meta-regressor
    # meta_regressor = XGBRegressor(objective="binary:logistic",  enable_categorical=True)
    meta_regressor = LinearRegression()
    # meta_regressor = LogisticRegression()
    # meta_regressor = XGBClassifier()
    
    # Create the StackingRegressor
    model = StackingRegressor(
        estimators=base_regressors,
        final_estimator=meta_regressor,
        passthrough=False
    )

    # model = XGBRegressor(
    #     **param, 
    #      objective="binary:logistic"
    #     )
    # model = LGBMRegressor(**lgbmparam)
    # model = CatBoostRegressor(verbose=False, cat_features=['CityID',"MLoc",'Team2', 'Team1'])
    # model = GradientBoostingRegressor()
    # model = RandomForestRegressor()
    # model = XGBRegressor(**xgbparam)
    # model = AdaBoostRegressor()
    
    # model = XGBClassifier(**param)
    # model = LGBMClassifier(**lgbmparam)
    # model = LGBMClassifier(max_depth=2, verbose=-1)
    # model = CatBoostClassifier(verbose=False, cat_features=['CityID','Team2', 'Team1','MLoc'])
    # model = RandomForestClassifier()
    # model = GradientBoostingClassifier()
    # model = AdaBoostClassifier()
    
    model.fit(dftrain[featurestrain], dftrain['Win'])
    
    print(f"{year}: Train:{brier_score_loss(dftrain['Win'], model.predict(dftrain[featurestrain]).clip(0, 1))} Test:{brier_score_loss(dftest['Win'], model.predict(dftest[featurestrain]).clip(0, 1))}")
    score += brier_score_loss(dftest['Win'], model.predict(dftest[featurestrain]).clip(0, 1))
    # print(f"{year}: Train:{brier_score_loss(dftrain['Win'], model.predict_proba(dftrain[featurestrain])[:,1])} Test:{brier_score_loss(dftest['Win'], model.predict_proba(dftest[featurestrain])[:,1])}")
    # score += brier_score_loss(dftest['Win'], model.predict_proba(dftest[featurestrain])[:,1])
print(score/tot)


# from sklearn.ensemble import StackingClassifier

# def cauchy_loss(labels, preds, sigma=1.0):
#     """
#     Cauchy loss function for XGBoost.
    
#     Args:
#         preds: Predictions from the model.
#         dtrain: DMatrix of training data.
#         sigma: Scale parameter, controls robustness.
    
#     Returns:
#         grad: Gradient.
#         hess: Hessian (second derivative).
#     """
#     residuals = preds - labels
    
#     # Cauchy loss gradient: ∂L/∂r = 2 * r / (sigma^2 + r^2)
#     grad = 2 * residuals / (sigma**2 + residuals**2)
    
#     # Cauchy loss hessian: ∂²L/∂r² = 2 * (sigma^2 - r^2) / (sigma^2 + r^2)^2
#     hess = 2 * (sigma**2 - residuals**2) / (sigma**2 + residuals**2)**2
    
#     return grad, hess
    
# featurestrainm = [
#     'Season', 
#     'CityID', 'MLoc',
#     'Team2', 'Team1',
#     'Team1SeasonScore', 'Team1SeasonFGM', 'Team1SeasonFGA', 'Team1SeasonFGM3', 'Team1SeasonFGA3', 'Team1SeasonFTM', 'Team1SeasonFTA', 'Team1SeasonOR', 'Team1SeasonDR', 'Team1SeasonAst', 'Team1SeasonTO', 'Team1SeasonStl', 'Team1SeasonBlk', 'Team1SeasonPF', 'Team1SeasonFG%', 'Team1SeasonFG3%', 'Team1SeasonFT%', 'Team1SeasonScoreDiff', 'Team1SeasonWins', 'Team1Last14DaysScore', 'Team1Last14DaysFGM', 'Team1Last14DaysFGA', 'Team1Last14DaysFGM3', 'Team1Last14DaysFGA3', 'Team1Last14DaysFTM', 'Team1Last14DaysFTA', 'Team1Last14DaysOR', 'Team1Last14DaysDR', 'Team1Last14DaysAst', 'Team1Last14DaysTO', 'Team1Last14DaysStl', 'Team1Last14DaysBlk', 'Team1Last14DaysPF', 'Team1Last14DaysFG%', 'Team1Last14DaysFG3%', 'Team1Last14DaysFT%', 'Team1Last14DaysScoreDiff', 'Team1Last14DaysWins','Team1Seedval', 'Team1quality', 
#     'Team1MASRank', 'Team1POMRank', 
#     'Team2SeasonScore', 'Team2SeasonFGM', 'Team2SeasonFGA', 'Team2SeasonFGM3', 'Team2SeasonFGA3', 'Team2SeasonFTM', 'Team2SeasonFTA', 'Team2SeasonOR', 'Team2SeasonDR', 'Team2SeasonAst', 'Team2SeasonTO', 'Team2SeasonStl', 'Team2SeasonBlk', 'Team2SeasonPF', 'Team2SeasonFG%', 'Team2SeasonFG3%', 'Team2SeasonFT%', 'Team2SeasonScoreDiff', 'Team2SeasonWins',  'Team2Last14DaysScore', 'Team2Last14DaysFGM', 'Team2Last14DaysFGA', 'Team2Last14DaysFGM3', 'Team2Last14DaysFGA3', 'Team2Last14DaysFTM', 'Team2Last14DaysFTA', 'Team2Last14DaysOR', 'Team2Last14DaysDR', 'Team2Last14DaysAst', 'Team2Last14DaysTO', 'Team2Last14DaysStl', 'Team2Last14DaysBlk', 'Team2Last14DaysPF', 'Team2Last14DaysFG%', 'Team2Last14DaysFG3%', 'Team2Last14DaysFT%', 'Team2Last14DaysScoreDiff', 'Team2Last14DaysWins', 'Team2Seedval',  'Team2quality', 
#     'Team2MASRank', 'Team2POMRank',
#     'SeasonScoreGap',  'SeasonFGMGap',  'SeasonFGAGap', 'SeasonFGM3Gap',  'SeasonFGA3Gap',  'SeasonFTMGap',  'SeasonFTAGap', 'SeasonORGap',  'SeasonDRGap',  'SeasonAstGap',  'SeasonTOGap',  'SeasonStlGap',  'SeasonBlkGap',  'SeasonPFGap',  'SeasonFG%Gap', 'SeasonFG3%Gap',  'SeasonFT%Gap',  'SeasonScoreDiffGap',  'SeasonWinsGap',  
#     'Last14DaysScoreGap', 'Last14DaysFGMGap', 'Last14DaysFGAGap', 'Last14DaysFGM3Gap', 'Last14DaysFGA3Gap', 'Last14DaysFTMGap', 'Last14DaysFTAGap', 'Last14DaysORGap','Last14DaysDRGap', 'Last14DaysTOGap','Last14DaysAstGap', 'Last14DaysStlGap','Last14DaysBlkGap', 'Last14DaysPFGap', 'Last14DaysFG%Gap', 'Last14DaysFG3%Gap', 'Last14DaysFT%Gap',  'Last14DaysWinsGap',
#     'Last14DaysScoreDiffGap',
#     # 'SeasonScoreRatio',  'SeasonFGMRatio',  'SeasonFGARatio', 'SeasonFGM3Ratio',  'SeasonFGA3Ratio',  'SeasonFTMRatio',  'SeasonFTARatio', 'SeasonORRatio',  'SeasonDRRatio',  'SeasonAstRatio',  'SeasonTORatio',  'SeasonStlRatio',  'SeasonBlkRatio',  'SeasonPFRatio',  'SeasonFG%Ratio', 'SeasonFG3%Ratio',  'SeasonFT%Ratio',  'SeasonScoreDiffRatio',  
#     # 'Last14DaysScoreRatio', 'Last14DaysFGMRatio', 'Last14DaysFGARatio', 'Last14DaysFGM3Ratio', 'Last14DaysFGA3Ratio', 'Last14DaysFTMRatio', 'Last14DaysFTARatio', 'Last14DaysORRatio','Last14DaysDRRatio', 'Last14DaysTORatio','Last14DaysAstRatio', 'Last14DaysStlRatio','Last14DaysBlkRatio', 'Last14DaysPFRatio', 'Last14DaysFG%Ratio', 'Last14DaysFG3%Ratio', 'Last14DaysFT%Ratio', 
#     'seeddiff',  'qualitydiff', 
#     'MASRankDiff', 'POMRankDiff'
# ]
# featurestrainw = [
#     'Season', 
#     'CityID', 'MLoc',
#     'Team2', 'Team1',
#     'Team1SeasonScore', 'Team1SeasonFGM', 'Team1SeasonFGA', 'Team1SeasonFGM3', 'Team1SeasonFGA3', 'Team1SeasonFTM', 'Team1SeasonFTA', 'Team1SeasonOR', 'Team1SeasonDR', 'Team1SeasonAst', 'Team1SeasonTO', 'Team1SeasonStl', 'Team1SeasonBlk', 'Team1SeasonPF', 'Team1SeasonFG%', 'Team1SeasonFG3%', 'Team1SeasonFT%', 'Team1SeasonScoreDiff', 'Team1SeasonWins', 'Team1Last14DaysScore', 'Team1Last14DaysFGM', 'Team1Last14DaysFGA', 'Team1Last14DaysFGM3', 'Team1Last14DaysFGA3', 'Team1Last14DaysFTM', 'Team1Last14DaysFTA', 'Team1Last14DaysOR', 'Team1Last14DaysDR', 'Team1Last14DaysAst', 'Team1Last14DaysTO', 'Team1Last14DaysStl', 'Team1Last14DaysBlk', 'Team1Last14DaysPF', 'Team1Last14DaysFG%', 'Team1Last14DaysFG3%', 'Team1Last14DaysFT%', 'Team1Last14DaysScoreDiff', 'Team1Last14DaysWins','Team1Seedval', 'Team1quality', 
#     'Team1MASRank', 'Team1POMRank', 
#     'Team2SeasonScore', 'Team2SeasonFGM', 'Team2SeasonFGA', 'Team2SeasonFGM3', 'Team2SeasonFGA3', 'Team2SeasonFTM', 'Team2SeasonFTA', 'Team2SeasonOR', 'Team2SeasonDR', 'Team2SeasonAst', 'Team2SeasonTO', 'Team2SeasonStl', 'Team2SeasonBlk', 'Team2SeasonPF', 'Team2SeasonFG%', 'Team2SeasonFG3%', 'Team2SeasonFT%', 'Team2SeasonScoreDiff', 'Team2SeasonWins',  'Team2Last14DaysScore', 'Team2Last14DaysFGM', 'Team2Last14DaysFGA', 'Team2Last14DaysFGM3', 'Team2Last14DaysFGA3', 'Team2Last14DaysFTM', 'Team2Last14DaysFTA', 'Team2Last14DaysOR', 'Team2Last14DaysDR', 'Team2Last14DaysAst', 'Team2Last14DaysTO', 'Team2Last14DaysStl', 'Team2Last14DaysBlk', 'Team2Last14DaysPF', 'Team2Last14DaysFG%', 'Team2Last14DaysFG3%', 'Team2Last14DaysFT%', 'Team2Last14DaysScoreDiff', 'Team2Last14DaysWins', 'Team2Seedval',  'Team2quality', 
#     'Team2MASRank', 'Team2POMRank',
#     'SeasonScoreGap',  'SeasonFGMGap',  'SeasonFGAGap', 'SeasonFGM3Gap',  'SeasonFGA3Gap',  'SeasonFTMGap',  'SeasonFTAGap', 'SeasonORGap',  'SeasonDRGap',  'SeasonAstGap',  'SeasonTOGap',  'SeasonStlGap',  'SeasonBlkGap',  'SeasonPFGap',  'SeasonFG%Gap', 'SeasonFG3%Gap',  'SeasonFT%Gap',  'SeasonScoreDiffGap',  'SeasonWinsGap',  
#     'Last14DaysScoreGap', 'Last14DaysFGMGap', 'Last14DaysFGAGap', 'Last14DaysFGM3Gap', 'Last14DaysFGA3Gap', 'Last14DaysFTMGap', 'Last14DaysFTAGap', 'Last14DaysORGap','Last14DaysDRGap', 'Last14DaysTOGap','Last14DaysAstGap', 'Last14DaysStlGap','Last14DaysBlkGap', 'Last14DaysPFGap', 'Last14DaysFG%Gap', 'Last14DaysFG3%Gap', 'Last14DaysFT%Gap',  'Last14DaysWinsGap',
#     'Last14DaysScoreDiffGap',
#     # 'SeasonScoreRatio',  'SeasonFGMRatio',  'SeasonFGARatio', 'SeasonFGM3Ratio',  'SeasonFGA3Ratio',  'SeasonFTMRatio',  'SeasonFTARatio', 'SeasonORRatio',  'SeasonDRRatio',  'SeasonAstRatio',  'SeasonTORatio',  'SeasonStlRatio',  'SeasonBlkRatio',  'SeasonPFRatio',  'SeasonFG%Ratio', 'SeasonFG3%Ratio',  'SeasonFT%Ratio',  'SeasonScoreDiffRatio',  
#     # 'Last14DaysScoreRatio', 'Last14DaysFGMRatio', 'Last14DaysFGARatio', 'Last14DaysFGM3Ratio', 'Last14DaysFGA3Ratio', 'Last14DaysFTMRatio', 'Last14DaysFTARatio', 'Last14DaysORRatio','Last14DaysDRRatio', 'Last14DaysTORatio','Last14DaysAstRatio', 'Last14DaysStlRatio','Last14DaysBlkRatio', 'Last14DaysPFRatio', 'Last14DaysFG%Ratio', 'Last14DaysFG3%Ratio', 'Last14DaysFT%Ratio', 
#     'seeddiff',  'qualitydiff', 
#     # 'MASRankDiff', 'POMRankDiff'
# ]
# xgbparam = {
#     # "alpha":1, # Def 0
#     # "lambda":2, # Default 1
#     "max_depth":3, # Def 6
#     # "eta":0.1, 
#     # "objective":"binary:hinge",  # Ensures sigmoid output
#     # "eval_metric":"logloss",
#     "enable_categorical":True,
#     # "gpu_id":0  # Specify GPU ID
# }
# lgbmparam={
#     "objective":"binary",  # Ensures sigmoid output
#     "metric":"binary_logloss",
#     "verbose": -1,
#     "max_depth":3,
#     #  "device":"gpu",
#     #  "max_bin": 155,
#     # "gpu_platform_id": 0,  # Select GPU platform (optional)
#     # "gpu_device_id": 0  # Specify GPU ID (optional)
# }

# tot=0
# score = 0

# for year in range(2018, 2026):
#     dftrainm = dfmatches[(dfmatches["Season"]<year)&(dfmatches['Team1'].astype(int)<3000)]
#     dftrainw = dfmatches[(dfmatches["Season"]<year)&(dfmatches['Team1'].astype(int)>=3000)]
    
#     dftestm = dfmatches[(dfmatches["Season"]==year)&(dfmatches['Team1'].astype(int)<3000)]
#     dftestw = dfmatches[(dfmatches["Season"]==year)&(dfmatches['Team1'].astype(int)>=3000)]

#     if(len(dftestm)==0 and len(dftestw)==0):
#         continue
#     tot+=1

#     param = {"enable_categorical":True,'eval_metric': 'mae', 'booster': 'gbtree', 'eta': 0.05, 'subsample': 0.35, 'colsample_bytree': 0.7, 'num_parallel_tree': 3, 'min_child_weight': 40, 'gamma': 10, 'max_depth': 3, 'silent': 1,
#              # "objective":cauchyobj
#             }
#     base_regressors = [
#         ('xgb', XGBRegressor(**param)),
#         ("lgbm", LGBMRegressor(max_depth=3, verbose=-1)),
#         ("rf", RandomForestRegressor()),
#         # ("cat", CatBoostClassifier(verbose=False)),
#         ("gbdt", GradientBoostingRegressor())
#     ]
    
#     # Define the meta-regressor
#     # meta_regressor = XGBRegressor(objective="binary:logistic",  enable_categorical=True)
#     meta_regressor = LinearRegression()
#     # meta_regressor = LogisticRegression()
#     # meta_regressor = XGBClassifier(max_depth=3)
    
#     # Create the StackingRegressor
#     model = StackingRegressor(
#         estimators=base_regressors,
#         final_estimator=meta_regressor,
#         passthrough=False
#     )

#     # model1 = XGBRegressor(**param, objective="binary:logistic")
#     # model2 = XGBRegressor(**param, objective="binary:logistic")
    
#     # model1 = GradientBoostingRegressor()
#     # model2 = GradientBoostingRegressor()
    
#     model1 = LGBMRegressor(**lgbmparam)
#     model2 = LGBMRegressor(**lgbmparam)
    
#     # model = LGBMRegressor(**lgbmparam)
#     # model = CatBoostRegressor(verbose=False, cat_features=['CityID',"MLoc",'Team2', 'Team1'])
#     # model = GradientBoostingRegressor()
#     # model = RandomForestRegressor()
#     # model = XGBRegressor(**xgbparam)
#     # model = AdaBoostRegressor()
    
#     # model = XGBClassifier(**param)
#     # model = LGBMClassifier(**lgbmparam)
#     # model = LGBMClassifier(max_depth=2, verbose=-1)
#     # model = CatBoostClassifier(verbose=False, cat_features=['CityID','Team2', 'Team1','MLoc'])
#     # model = RandomForestClassifier()
#     # model = GradientBoostingClassifier()
#     # model = AdaBoostClassifier()
    
#     model1.fit(dftrainm[featurestrainm], dftrainm['Win'])
#     model2.fit(dftrainw[featurestrainw], dftrainw['Win'])

#     trainwins = np.concatenate([dftrainm['Win'], dftrainw['Win']])
#     trainpred = np.concatenate([model1.predict(dftrainm[featurestrainm]), model2.predict(dftrainw[featurestrainw])]).clip(0.01, 0.99)
    
#     testwins = np.concatenate([dftestm['Win'], dftestw['Win']])
#     testpred = np.concatenate([model1.predict(dftestm[featurestrainm]), model2.predict(dftestw[featurestrainw])]).clip(0.01, 0.99)
    
#     print(f"{year}: Train:{brier_score_loss(trainwins, trainpred)} Test:{brier_score_loss(testwins, testpred)}")
#     score += brier_score_loss(testwins, testpred)
#     # print(f"{year}: Train:{brier_score_loss(dftrain['Win'], model.predict_proba(dftrain[featurestrain])[:,1])} Test:{brier_score_loss(dftest['Win'], model.predict_proba(dftest[featurestrain])[:,1])}")
#     # score += brier_score_loss(dftest['Win'], model.predict_proba(dftest[featurestrain])[:,1])
# print(score/tot)


# import optuna

# def objective(trial):
#     # Hyperparameter search space
#     # params = {
#     #     "n_estimators": trial.suggest_int('n_estimators', 50, 300),
#     #     "max_depth" : trial.suggest_int('max_depth', 2, 20),
#     #     "min_samples_split" : trial.suggest_int('min_samples_split', 2, 10),
#     #     "min_samples_leaf" : trial.suggest_int('min_samples_leaf', 1, 10),
#     #     "max_features" : trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
#     # }

#     params = {
#         "enable_categorical":True,
#         "objective": "binary:logistic",  # For binary classification
#         "eval_metric": "logloss",
#         "booster": "gbtree",
#         "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
#         "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#     }
    
#     tot=0
#     score = 0
#     for year in range(2019, 2026):
#         dftrain = dfmatches[dfmatches["Season"]<year]
#         dftest = dfmatches[dfmatches["Season"]==year]
    
#         if(len(dftest)==0):
#             continue
#         tot+=1
    
#         # base_regressors = [
#         #     # ('xgb', XGBClassifier(**xgbparam)),
#         #     ("lgbm", LGBMClassifier(max_depth=3)),
#         #     ("rf", RandomForestClassifier()),
#         #     # ("cat", CatBoostClassifier(verbose=False)),
#         #     ("gbdt", GradientBoostingClassifier())
#         # ]
        
#         # Define the meta-regressor
#         # meta_regressor = XGBRegressor(objective="binary:logistic",  enable_categorical=True)
#         # meta_regressor = LinearRegression()
#         # meta_regressor = LogisticRegression()
#         # meta_regressor = XGBClassifier(max_depth=3)
        
#         # Create the StackingRegressor
#         # model = StackingClassifier(
#         #     estimators=base_regressors,
#         #     final_estimator=meta_regressor,
#         #     passthrough=False
#         # )
#         # param = {'eval_metric': 'mae', 'booster': 'gbtree', 'eta': 0.05, 'subsample': 0.35, 'colsample_bytree': 0.7, 'num_parallel_tree': 3, 'min_child_weight': 40, 'gamma': 10, 'max_depth': 3, 'silent': 1}
#         # model = XGBClassifier(**xgbparam)
#         # model = XGBRegressor(**xgbparam)
#         # model = LGBMRegressor(**lgbmparam)
#         # model = LGBMClassifier(max_depth=2, verbose=-1)
#         # model = RandomForestRegressor(**params)
#         model = XGBClassifier(**params)
#         # model = CatBoostClassifier(verbose=False)
#         # model = GradientBoostingClassifier()
#         model.fit(dftrain[featurestrain], dftrain['Win'])
    
#         # print(f"{year}: Train:{brier_score_loss(dftrain['Win'], model.predict_proba(dftrain[featurestrain])[:,1])} Test:{brier_score_loss(dftest['Win'], model.predict_proba(dftest[featurestrain])[:,1])}")
#         score += brier_score_loss(dftest['Win'], model.predict_proba(dftest[featurestrain])[:,1])
#     return(score/tot)

# # Create study and optimize
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# # Best parameters and accuracy
# print("Best trial:", study.best_trial.params)
# print("Best accuracy:", study.best_value)


model.fit(dftrain[featurestrain], dftrain['Win'])


sub = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')
sub = pd.json_normalize(sub.apply(lambda x: {"ID":x['ID'], "Season":int(x['ID'][:4]), "Team1":int(x['ID'][5:9]), "Team2":int(x['ID'][10:14])}, axis=1))

# Add regular season stats
dfsub = sub.merge(df_reg_stats, left_on=['Team1',"Season"], right_on=["TeamID","Season"]).rename(columns={"SeasonWins":"Team1SeasonWins"}|{f"Season{i}":f"Team1Season{i}" for i in seasonstats}).drop(columns="TeamID")
dfsub = dfsub.merge(df_reg_stats, left_on=['Team2',"Season"], right_on=["TeamID","Season"]).rename(columns={"SeasonWins":"Team2SeasonWins"}|{f"Season{i}":f"Team2Season{i}" for i in seasonstats}).drop(columns="TeamID")
# display(dfsub)

# Add last 14 days
dfsub = dfsub.merge(df_last_stats, left_on=['Team1',"Season"], right_on=["TeamID","Season"], how='left').rename(columns={"Last14DaysWins":"Team1Last14DaysWins"}|{f"Last14Days{i}":f"Team1Last14Days{i}" for i in seasonstats}).drop(columns="TeamID").fillna(0)
dfsub = dfsub.merge(df_last_stats, left_on=['Team2',"Season"], right_on=["TeamID","Season"], how='left').rename(columns={"Last14DaysWins":"Team2Last14DaysWins"}|{f"Last14Days{i}":f"Team2Last14Days{i}" for i in seasonstats}).drop(columns="TeamID").fillna(0)
# display(dfsub)

# Team quality
dfsub = dfsub.merge(team_quality, left_on=['Season','Team1'], right_on=['Season','TeamID']).rename(columns={'quality':'Team1quality'}).drop(columns=['TeamID'])
dfsub = dfsub.merge(team_quality, left_on=['Season','Team2'], right_on=['Season','TeamID']).rename(columns={'quality':'Team2quality'}).drop(columns=['TeamID'])
# display(dfsub)

# Add seeds
dfsub = dfsub.merge(dfseeds[["Season","TeamID","Seedval"]], left_on=["Season","Team1"], right_on=['Season','TeamID']).rename(columns={"Seedval":"Team1Seedval"}).drop(columns=["TeamID"])
dfsub = dfsub.merge(dfseeds[["Season","TeamID","Seedval"]], left_on=["Season","Team2"], right_on=['Season','TeamID']).rename(columns={"Seedval":"Team2Seedval"}).drop(columns=["TeamID"])
# display(dfsub)

# Add gap features
for i in seasonstats+['Wins',"ScoreDiff"]:
    dfsub[f"Season{i}Gap"] = dfsub[f"Team1Season{i}"]-dfsub[f"Team2Season{i}"]
    dfsub[f"Last14Days{i}Gap"] = dfsub[f"Team1Last14Days{i}"]-dfsub[f"Team2Last14Days{i}"]
    dfsub[f"Season{i}Ratio"] = dfsub[f"Team1Season{i}"]/dfsub[f"Team2Season{i}"]
    dfsub[f"Last14Days{i}Ratio"] = dfsub[f"Team1Last14Days{i}"]/dfsub[f"Team2Last14Days{i}"]
dfsub['qualitydiff'] = dfsub['Team1quality']-dfsub['Team2quality']
dfsub['seeddiff'] = dfsub['Team1Seedval']-dfsub['Team2Seedval']
# display(dfsub)

for sys in ['MAS', 'POM']:
    # .reset_index(ignore_index=True)
    dfsub = dfsub.merge(dfmmas.query(f"SystemName=='{sys}' and RankingDayNum == 133").drop(columns=['RankingDayNum','SystemName']), left_on=['Season','Team1'], right_on=['Season','TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'OrdinalRank':f'Team1{sys}Rank'})
    dfsub = dfsub.merge(dfmmas.query(f"SystemName=='{sys}' and RankingDayNum == 133").drop(columns=['RankingDayNum','SystemName']), left_on=['Season','Team2'], right_on=['Season','TeamID'], how='left').drop(columns=['TeamID']).rename(columns={'OrdinalRank':f'Team2{sys}Rank'}).fillna(-1)
    dfsub[f'{sys}RankDiff'] = dfsub[f'Team1{sys}Rank']-dfsub[f'Team2{sys}Rank']

dfsub['Pred'] = model.predict(dfsub[featurestrain])
final = sub.merge(dfsub[['ID','Pred']], on=['ID'], how='left')[['ID','Pred']]

final.fillna(0.5).to_csv('sub.csv', index=False)

