# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


PREVIOUS_SEASONS_MEN = True
PREVIOUS_SEASONS_WOMEN  = True 
USE_GPU = True # Turn on GPU P100 if USE_GPU=True
USE_ADDITONAL_COLUMN = False
USE_SIMPLEFLAG = False
USE_YEAR_WINFLAG = False
USE_DAYS_WINFLAG = True


import os

import matplotlib.pyplot as plt
import statsmodels.api as sm
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn import preprocessing
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold
from tqdm import tqdm
pd.set_option("display.max_column", 200)
pd.set_option("display.max_rows", 200)
# print(os.listdir("../input"))
xgb.__version__ # I used '1.2.0-SNAPSHOT'



DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025/"
tourney_results = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv"),
], ignore_index=True)

seeds = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv"),
], ignore_index=True)
#  Data Inquiry for 2025
if 2025 not in seeds["Season"].unique():
    print("Provisionally processed as of 2024, when 2025 seed data does not exist")
    
    # 2024 Season Data copy
    seeds_2025 = seeds[seeds["Season"] == 2024].copy()
    seeds_2025["Season"] = 2025  # Change Seasn 2025
    
    # Combine with source data
    seeds = pd.concat([seeds, seeds_2025], ignore_index=True)
    
    print(" Added 2025 Season Data.")
else:
    print(" Already exist 2025 Season Data")
    

print(seeds.tail(10)) 
regular_results = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WRegularSeasonDetailedResults.csv"),
], ignore_index=True)


def prepare_data(df_data, use_additional_column=False):
    df = df_data.copy()
    df.rename(columns={'WLoc': 'location'}, inplace=True)
    
    # Create Data for swap
    dfswap = df[[
        'Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'location', 'NumOT', 
        'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
        'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'
    ]]

    #  additional_columns list
    additional_cols = ['WEFFG', 'WEFFG3', 'WDARE', 'WTOQUETOQUE',
                       'LEFFG', 'LEFFG3', 'LDARE', 'LTOQUETOQUE']
    
    #  Whether additional data columns are reflected
    if use_additional_column:
        df = df.assign(**{col: df[col.split('_')[1]] / df[col.split('_')[1] + 'A'] for col in additional_cols})
        dfswap = dfswap.assign(**{col: dfswap[col.split('_')[1]] / dfswap[col.split('_')[1] + 'A'] for col in additional_cols})

    #  Win Team(W) â†’ T1, LoseTeam(L) â†’ T2
    df.columns = df.columns.str.replace('W', 'T1_')
    df.columns = df.columns.str.replace('L', 'T2_')

    #  dfswap(Lose team ->1, Win Team ->T2)
    dfswap.columns = dfswap.columns.str.replace('L', 'T1_')
    dfswap.columns = dfswap.columns.str.replace('W', 'T2_')

    # combined Swap Data
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    
    #  Converting Home Team to Numbers: N â†’ 0, H â†’ 1, A â†’ -1
    output.loc[output.location == 'N', 'location'] = '0'
    output.loc[output.location == 'H', 'location'] = '1'
    output.loc[output.location == 'A', 'location'] = '-1'
    output['location'] = output['location'].astype(int)

    output['PointDiff'] = output['T1_Score'] - output['T2_Score']


    if use_additional_column:
        output['T1_EFFG'] = output['T1_FGM'] / output['T1_FGA']
        output['T1_EFFG3'] = output['T1_FGM3'] / output['T1_FGA3']
        output['T1_DARE'] = output['T1_FGM3'] / output['T1_FGM']
        output['T1_TOQUETOQUE'] = output['T1_Ast'] / output['T1_FGM']
        
        output['T2_EFFG'] = output['T2_FGM'] / output['T2_FGA']
        output['T2_EFFG3'] = output['T2_FGM3'] / output['T2_FGA3']
        output['T2_DARE'] = output['T2_FGM3'] / output['T2_FGM']
        output['T2_TOQUETOQUE'] = output['T2_Ast'] / output['T2_FGM']

        # if additional results == Nan then fill in 0 
        cols_to_fill = ['T1_EFFG','T1_EFFG3','T1_DARE','T1_TOQUETOQUE',
                        'T2_EFFG','T2_EFFG3','T2_DARE','T2_TOQUETOQUE']
        output[cols_to_fill] = output[cols_to_fill].fillna(0.0)

    return output




#Additional Parameter
'''
EFFG â†’ "Team's effective field goal percentage"
EFFG3 â†’ "Team's three-point shooting accuracy"
DARE â†’ "Percentage of three-point shots among all made field goals"
TOQUETOQUE â†’ "Assist-to-field-goal ratio"
'''
if USE_ADDITONAL_COLUMN == True:
    regular_results['WEFFG'] = regular_results['WFGM'] / regular_results['WFGA']
    regular_results['WEFFG3'] = regular_results['WFGM3'] / regular_results['WFGA3']
    regular_results['WDARE'] = regular_results['WFGM3'] / regular_results['WFGM']
    regular_results['WTOQUETOQUE'] = regular_results['WAst'] / regular_results['WFGM']
    
    regular_results['LEFFG'] = regular_results['LFGM'] / regular_results['LFGA']
    regular_results['LEFFG3'] = regular_results['LFGM3'] / regular_results['LFGA3']
    regular_results['LDARE'] = regular_results['LFGM3'] / regular_results['LFGM']
    regular_results['LTOQUETOQUE'] = regular_results['LAst'] / regular_results['LFGM']
    
    
    tourney_results['WEFFG'] = tourney_results['WFGM'] / tourney_results['WFGA']
    tourney_results['WEFFG3'] = tourney_results['WFGM3'] / tourney_results['WFGA3']
    tourney_results['WDARE'] = tourney_results['WFGM3'] / tourney_results['WFGM']
    tourney_results['WTOQUETOQUE'] = tourney_results['WAst'] / tourney_results['WFGM']
    
    tourney_results['LEFFG'] = tourney_results['LFGM'] / tourney_results['LFGA']
    tourney_results['LEFFG3'] = tourney_results['LFGM3'] / tourney_results['LFGA3']
    tourney_results['LDARE'] = tourney_results['LFGM3'] / tourney_results['LFGM']
    tourney_results['LTOQUETOQUE'] = tourney_results['LAst'] / tourney_results['LFGM']
    cols_with_ratios = [
    'WEFFG','WEFFG3','WDARE','WTOQUETOQUE',
    'LEFFG','LEFFG3','LDARE','LTOQUETOQUE'
    ]
    



regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)


base_cols = [
    'T1_Score', 'T2_Score',
    'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3',
    'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF',
    'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3',
    'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',
    'PointDiff'
]

additional_cols = [
    'T1_EFFG', 'T1_EFFG3', 'T1_DARE', 'T1_TOQUETOQUE',
    'T2_EFFG', 'T2_EFFG3', 'T2_DARE', 'T2_TOQUETOQUE'
]

if USE_ADDITONAL_COLUMN:
    boxscore_cols = base_cols + additional_cols
else:
    boxscore_cols = base_cols
if USE_SIMPLEFLAG:
    boxscore_cols = ['T1_Score', 'T2_Score', 'PointDiff']


season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg([np.mean]).reset_index()
season_statistics.columns = [f"{col[0]}_mean" if col[1] == "mean" else col[0] for col in season_statistics.columns]
season_statistics.head(3)



def get_mean_of_3_season(val_2ps, val_1ps, val_0ps, 
                         weight_2=1, weight_1=2, weight_0=3, 
                         degree_weight=1.0
                         ):
    weight_2 = weight_2**degree_weight
    weight_1 = weight_1**degree_weight
    weight_0 = weight_0**degree_weight
    if val_2ps == 0  and val_1ps == 0:
        return val_0ps
    elif val_2ps == 0:
        return (val_1ps*weight_2 + val_0ps*weight_1)/(weight_2 + weight_1)
    else:
        sum_of_values = val_2ps*weight_2 + val_1ps*weight_1 + val_0ps*weight_0
        return sum_of_values/(weight_2 + weight_1 + weight_0)

def get_3_feature(df_team, feature):
    value_2_seasons_ago = 0
    value_1_season_ago = 0
    value_0_season_ago = 0
    for _, val in df_team.iterrows():
        value_2_seasons_ago = value_1_season_ago
        value_1_season_ago = value_0_season_ago
        value_0_season_ago = val[feature]
    return value_2_seasons_ago, value_1_season_ago, value_0_season_ago

def write_mean_of_3_seasons(df, features, degree_weight=1.0):
    df_copy = df.copy()
    suffix = "_mn3s"
    

    for ft in features:
        df_copy[ft + suffix] = 0.0  

    for idx, val in tqdm(df_copy.iterrows(), total=len(df_copy)):
        team = val.T1_TeamID
        season = val.Season
        df_team = df_copy[(df_copy.T1_TeamID == team)&
                          (df_copy.Season <= season)&
                          (df_copy.Season > season-3)]
        
        for ft in features:
            val_2ps, val_1ps, val_0ps = get_3_feature(df_team, ft)
            ft_mean_3 = get_mean_of_3_season(val_2ps, val_1ps, val_0ps, degree_weight=degree_weight)
            

            df_copy.loc[idx, ft + suffix] = float(ft_mean_3)  

    return df_copy



#Make two copies of the data
if PREVIOUS_SEASONS_MEN:
    features_for_calc = ["T1_Score_mean", "T1_FGA_mean",  "T1_FGA3_mean"]
    season_statistics_with_3_seas = write_mean_of_3_seasons(
        season_statistics, features_for_calc, degree_weight=1.0
    )
    season_statistics_T1 = season_statistics_with_3_seas.copy()
    season_statistics_T2 = season_statistics_with_3_seas.copy()
else:
    season_statistics_T1 = season_statistics.copy()
    season_statistics_T2 = season_statistics.copy()

season_statistics_T1[1000:1003]


season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T1.columns)]
season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T2.columns)]
season_statistics_T1.columns.values[0] = "Season"
season_statistics_T2.columns.values[0] = "Season"

# We don't have the box score statistics in the prediction bank. So drop it.
tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score']]
season_statistics_T1.head(3)


last14days_stats_T1 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff']>0,1,0)
last14days_stats_T1 = last14days_stats_T1.groupby(['Season','T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

last14days_stats_T2 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff']<0,1,0)
last14days_stats_T2 = last14days_stats_T2.groupby(['Season','T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')

#  T1 íŒ€ì�˜ ì •ê·œì‹œì¦Œ ì „ì²´ ìŠ¹ë¥  ê³„ì‚°
season_win_stats_T1 = regular_data.copy()
season_win_stats_T1['win'] = np.where(season_win_stats_T1['PointDiff'] > 0, 1, 0)
season_win_stats_T1 = season_win_stats_T1.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_season')

#  T2 íŒ€ì�˜ ì •ê·œì‹œì¦Œ ì „ì²´ ìŠ¹ë¥  ê³„ì‚°
season_win_stats_T2 = regular_data.copy()
season_win_stats_T2['win'] = np.where(season_win_stats_T2['PointDiff'] < 0, 1, 0)
season_win_stats_T2 = season_win_stats_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_season')



season_win_stats_T2


tourney_data = pd.merge(tourney_data, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')
if USE_YEAR_WINFLAG:
    tourney_data = pd.merge(tourney_data, season_win_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, season_win_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')






#Create all possible matches for the last seeded teams, stat allocation.

tourney_data = pd.merge(tourney_data, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')

regular_season_effects = regular_data[['Season','T1_TeamID','T2_TeamID','PointDiff']].copy()
regular_season_effects['T1_TeamID'] = regular_season_effects['T1_TeamID'].astype(str)
regular_season_effects['T2_TeamID'] = regular_season_effects['T2_TeamID'].astype(str)
regular_season_effects['win'] = np.where(regular_season_effects['PointDiff']>0,1,0)
march_madness = pd.merge(seeds[['Season','TeamID']],seeds[['Season','TeamID']],on='Season')
march_madness.columns = ['Season', 'T1_TeamID', 'T2_TeamID']
march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
regular_season_effects = pd.merge(regular_season_effects, march_madness, on = ['Season','T1_TeamID','T2_TeamID'])
regular_season_effects.shape


print(march_madness.head(30))  # win ì»¬ëŸ¼ì�´ ì�ˆëŠ”ì§€ í™•ì�¸



regular_season_effects.tail(5)


def normalize_column(values):
    themean = np.mean(values)
    thestd = np.std(values)
    norm = (values - themean)/(thestd) 
    return(pd.DataFrame(norm))


def team_quality(season):
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    #
    glm = sm.GLM.from_formula(formula=formula, 
                              data=regular_season_effects.loc[regular_season_effects.Season==season,:], 
                              family=sm.families.Binomial()).fit()
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    quality['quality'] = normalize_column(quality['quality'])
    quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)

    return quality

# This is metric to measure the team's strength, in this case, this is a logistic regression and we
# the coefficients
glm_quality = pd.concat([team_quality(2010),
                         team_quality(2011),
                         team_quality(2012),
                         team_quality(2013),
                         team_quality(2014),
                         team_quality(2015),
                         team_quality(2016),
                         team_quality(2017),
                         team_quality(2018),
                         team_quality(2019),
                         team_quality(2021),
                         team_quality(2022),
                         team_quality(2023),
                         team_quality(2024),
                         team_quality(2025)
                        ]).reset_index(drop=True)

# Eliminating outliers
threshold = glm_quality['quality'].quantile(0.99)
glm_quality['quality'] = glm_quality['quality'].where(glm_quality['quality'] <= threshold, np.nan)
glm_quality['quality'].isnull().sum()  # NaN ê°’ì�´ ëª‡ ê°œ ì�ˆëŠ”ì§€ í™•ì�¸


import numpy as np
import pandas as pd
import statsmodels.api as sm

def normalize_column(values):
    themean = np.mean(values)
    thestd = np.std(values)
    norm = (values - themean)/(thestd)
    return pd.DataFrame(norm)

def team_quality_regularized(season, alpha=0.1, l1_wt=0.0):
    """
    ì •ê·œí™”(Regularization)ë�œ ë¡œì§€ìŠ¤í‹± íšŒê·€ë¥¼ ì‚¬ìš©í•´ team_qualityë¥¼ ì¶”ì •.
    
    Parameters
    ----------
    season : int
        ë¶„ì„�í•  ì‹œì¦Œ
    alpha : float
        ê·œì œ ì„¸ê¸°(í�´ìˆ˜ë¡� ê·œì œê°€ ê°•í•¨)
    l1_wt : float
        L1(=1)ì™€ L2(=0) í˜¼í•©ë¹„ìœ¨ (Elastic Net)
        - 0.0 â†’ L2(Ridge)
        - 1.0 â†’ L1(Lasso)
        - 0.5 â†’ L1,L2 í˜¼í•©(Elastic Net)
    """
    formula = 'win ~ -1 + C(T1_TeamID) + C(T2_TeamID)'
    
    # 1) GLM ëª¨ë�¸ ìƒ�ì„± (Binomial = ë¡œì§€ìŠ¤í‹±)
    model = sm.GLM.from_formula(
        formula=formula, 
        data=regular_season_effects.loc[regular_season_effects.Season==season, :],
        family=sm.families.Binomial()
    )
    
    # 2) ì •ê·œí™”ë�œ í•™ìŠµ: fit() ëŒ€ì‹  fit_regularized() ì‚¬ìš©
    #    alpha=0.1 ë“± ì›�í•˜ëŠ” ê°’ ì¡°ì •
    #    l1_wt=0.0ì�´ë©´ Ridge(L2), 1.0ì�´ë©´ Lasso(L1)
    res = model.fit_regularized(alpha=alpha, L1_wt=l1_wt)
    
    # 3) ê³„ìˆ˜(íŒ€ ì „ë ¥) ì¶”ì¶œ
    quality = pd.DataFrame({
        'TeamID': res.params.index,
        'quality': res.params.values
    }).reset_index(drop=True)
    quality['Season'] = season

    # 4) ì‚¬í›„ì²˜ë¦¬ (z-score + exp) â†’ ê¸°ì¡´ ì½”ë“œì™€ ë�™ì�¼
    #    ë‹¤ë§Œ statsmodelsê°€ param ì�´ë¦„ì�„ "C(T1_TeamID)[T.XXXX]"ë¡œ ë§Œë“¤ë¯€ë¡œ
    #    T1_ íŒŒì‹± ì‹œ ì£¼ì�˜
    quality['quality'] = normalize_column(quality['quality'])
    quality['quality'] = np.exp(quality['quality'])

    # 5) T1 íŒ€ë§Œ í•„í„°ë§�
    #    "C(T1_TeamID)[T.XXXX]" í˜•íƒœì�¸ì§€ í™•ì�¸ í›„ íŒŒì‹±
    mask_t1 = (
        quality['TeamID'].str.contains(r'T1_TeamID') |
        quality['TeamID'].str.contains(r'C\(T1_TeamID\)')
    )

    quality = quality.loc[mask_t1].reset_index(drop=True)
    
    # 6) TeamID ë¬¸ì��ì—´ì—�ì„œ ì‹¤ì œ íŒ€ ë²ˆí˜¸ë§Œ ì¶”ì¶œ
    #    ì •ê·œí‘œí˜„ì‹�ìœ¼ë¡œ ì•ˆì „í•˜ê²Œ íŒŒì‹±
    import re
    def extract_id(x):
        # ì˜ˆ: "C(T1_TeamID)[T.3124]" â†’ "3124"
        match = re.findall(r'\[T\.(\d+)\]', x)
        if match:
            return int(match[0])
        # ì˜ˆ: "T1_TeamID[3124]" â†’ "3124"
        match_brackets = re.findall(r'\[(\d+)\]', x)
        if match_brackets:
            return int(match_brackets[0])
        return np.nan  # fallback
    
    quality['TeamID'] = quality['TeamID'].apply(extract_id).astype('Int64')
    
    print(quality['quality'].mean(), quality['quality'].std())
    return quality
# This is metric to measure the team's strength, in this case, this is a logistic regression and we
# the coefficients
glm_quality = pd.concat([team_quality_regularized(2010),
                         team_quality_regularized(2011),
                         team_quality_regularized(2012),
                         team_quality_regularized(2013),
                         team_quality_regularized(2014),
                         team_quality_regularized(2015),
                         team_quality_regularized(2016),
                         team_quality_regularized(2017),
                         team_quality_regularized(2018),
                         team_quality_regularized(2019),
                         team_quality_regularized(2021),
                         team_quality_regularized(2022),
                         team_quality_regularized(2023),
                         team_quality_regularized(2024),
                         team_quality_regularized(2025)
                        ]).reset_index(drop=True)

# Eliminating outliers
threshold = glm_quality['quality'].quantile(0.99)
glm_quality['quality'] = glm_quality['quality'].where(glm_quality['quality'] <= threshold, np.nan)
glm_quality['quality'].isnull().sum()  # NaN ê°’ì�´ ëª‡ ê°œ ì�ˆëŠ”ì§€ í™•ì�¸





glm_quality_T1 = glm_quality.copy()
glm_quality_T2 = glm_quality.copy()
glm_quality_T1.columns = ['T1_TeamID','T1_quality','Season']
glm_quality_T2.columns = ['T2_TeamID','T2_quality','Season']

tourney_data = pd.merge(tourney_data, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')

tourney_data.head()
tourney_data['T1_quality'] = tourney_data['T1_quality'].fillna(0.2)
tourney_data['T2_quality'] = tourney_data['T2_quality'].fillna(0.2)
tourney_data.T2_quality.isnull().sum()

seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds.head()


tourney_data.tail(3)


print(glm_quality[glm_quality['quality'] > 5])



seeds_T1 = seeds[['Season','TeamID','seed']].copy()
seeds_T2 = seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed']

tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

#Optional but not relevant
tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]

if PREVIOUS_SEASONS_MEN:
    features_for_calc = ["T1_quality", "T2_quality", "T1_seed"]
    tourney_data_with_3_seas = write_mean_of_3_seasons(tourney_data, features_for_calc, degree_weight=1.0)
    tourney_data = tourney_data_with_3_seas.copy()

tourney_data[1000:1002]


# The descriptive feature is the score, not the winner
# y = tourney_data['T1_Score'] - tourney_data['T2_Score']
# y.describe()

if PREVIOUS_SEASONS_MEN:
    features = list(season_statistics_T1.columns[2:999]) + \
        list(season_statistics_T2.columns[2:999]) + \
        list(seeds_T1.columns[2:999]) + \
        list(seeds_T2.columns[2:999]) + \
        ["Seed_diff"] + ["T1_quality","T2_quality"] +\
        ["T1_quality_mn3s", "T2_quality_mn3s", "T1_seed_mn3s"]
else:
    features = list(season_statistics_T1.columns[2:999]) + \
        list(season_statistics_T2.columns[2:999]) + \
        list(seeds_T1.columns[2:999]) + \
        list(seeds_T2.columns[2:999]) + \
        ["Seed_diff"] + ["T1_quality","T2_quality"] 

print(len(features))



tourney_data[features]


sub = pd.read_csv('../input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')

sub["Season"] = sub["ID"].apply(lambda x: x[0:4]).astype(int)
sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)
sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)
sub.shape




sub = pd.merge(sub, season_statistics_T1, on = ['Season', 'T1_TeamID'])
sub = pd.merge(sub, season_statistics_T2, on = ['Season', 'T2_TeamID'])
print(sub.shape)
sub = pd.merge(sub, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left') 
sub = pd.merge(sub, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')
print(sub.shape)
sub = pd.merge(sub, seeds_T1, on = ['Season', 'T1_TeamID'])
sub = pd.merge(sub, seeds_T2, on = ['Season', 'T2_TeamID'])
print(sub.shape)
sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]
print(sub.shape)
sub.head(3)


print(sub.T2_quality.isnull().sum())
sub['T1_quality'].fillna(0.2, inplace = True)
sub['T2_quality'].fillna(0.2, inplace = True)
print(sub.T2_quality.isnull().sum())

if PREVIOUS_SEASONS_MEN:
    features_for_calc = ["T1_quality", "T2_quality", "T1_seed"]
    sub = write_mean_of_3_seasons(sub, features_for_calc, degree_weight=1.0)
    sub = sub.copy()
sub.head(3)


y = tourney_data['T1_Score'] - tourney_data['T2_Score']
y.describe()


features = list(season_statistics_T1.columns[2:999]) + \
    list(season_statistics_T2.columns[2:999]) + \
    list(seeds_T1.columns[2:999]) + \
    list(seeds_T2.columns[2:999]) + \
    ["Seed_diff"] + ["T1_quality","T2_quality"]
if USE_DAYS_WINFLAG:
    features = features + list(last14days_stats_T1.columns[2:999]) + \
    list(last14days_stats_T2.columns[2:999])
    
if USE_YEAR_WINFLAG:
    features = features + list(season_win_stats_T1.columns[2:999]) + \
    list(season_win_stats_T2.columns[2:999])
    
len(features)


print(len(season_statistics_T1.columns))  # ì „ì²´ ì»¬ëŸ¼ ê°œìˆ˜ í™•ì�¸
print(season_statistics_T1.columns[:10])  # ì²˜ì�Œ 10ê°œ ì»¬ëŸ¼ ì�´ë¦„ í™•ì�¸



print(tourney_data.columns)  # 'T1_win_ratio_14d', 'T2_win_ratio_14d' í�¬í•¨ ì—¬ë¶€ í™•ì�¸



X = tourney_data[features].values
dtrain = xgb.DMatrix(X, label = y)


def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000 
    x =  preds-labels    
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess


param = {} 
param['eval_metric'] = 'mae' #Mean absolute Error
#When you try to reduce the difference between the actual value and the predicted value
param['booster'] = 'gbtree' #boosting method
param['eta'] = 0.05  # 
param['subsample'] = 0.35
param['colsample_bytree'] = 0.7
param['num_parallel_tree'] = 10  #
param['min_child_weight'] = 40
param['gamma'] = 10
param['max_depth'] = 3
param['verbosity'] = 1  

#  GPU Setting On/Off
if USE_GPU:
    param.update({
        'tree_method': 'hist',
        'device' : 'cuda'  # GPU Seeting
    })


import torch
if USE_GPU:
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Current Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))



# import optuna
# import xgboost as xgb

# def objective(trial):
#     param_opt = {
#         'eta': trial.suggest_float('eta', 0.005, 0.05, log=True),  # âœ… ìˆ˜ì •
#         'subsample': trial.suggest_float('subsample', 0.3, 0.8),  # âœ… ìˆ˜ì •
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # âœ… ìˆ˜ì •
#         'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
#         'gamma': trial.suggest_float('gamma', 0, 20),  # âœ… ìˆ˜ì •
#         'max_depth': trial.suggest_int('max_depth', 3, 7),
#         'eval_metric': 'mae'  # âœ… MAE ëª…ì‹œ
#     }


#     # âœ… í•™ìŠµ ë�°ì�´í„° ì •ì�˜ (X, yê°€ ì‚¬ì „ì—� ì •ì�˜ë�˜ì–´ ì�ˆì–´ì•¼ í•¨)
#     dtrain = xgb.DMatrix(X, label=y)

#     # âœ… `metrics=['mae']`ë¥¼ ëª…ì‹œí•´ì„œ ì œëŒ€ë¡œ ë�œ ê²°ê³¼ê°€ ë‚˜ì˜¤ë�„ë¡� ì„¤ì •
#     cv_result = xgb.cv(
#         param_opt, dtrain, num_boost_round=1000, nfold=5, 
#         early_stopping_rounds=25, metrics=['mae'], as_pandas=True
#     )

#     # âœ… 'test-mae-mean' ì»¬ëŸ¼ í™•ì�¸ í›„ ì²˜ë¦¬
#     if 'test-mae-mean' not in cv_result:
#         print("âš ï¸� 'test-mae-mean' ì»¬ëŸ¼ì�´ ì—†ì�Œ! XGBoost ê²°ê³¼ í™•ì�¸ í•„ìš”!")
#         return float('inf')  # Optunaê°€ ì�´ trialì�„ ë²„ë¦¬ë�„ë¡� ì²˜ë¦¬

#     return cv_result['test-mae-mean'].min()

# # âœ… Optuna ì‹¤í–‰
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=20)

# # âœ… ìµœì �ì�˜ íŒŒë�¼ë¯¸í„° ì �ìš©
# param.update(study.best_params)
# print("ğŸ”¥ ìµœì �í™”ë�œ íŒŒë�¼ë¯¸í„° ì �ìš© ì™„ë£Œ!")
# print(param)



xgb_cv = []
repeat_cv = 10 # recommend 10

for i in range(repeat_cv): 
    print(f"Fold repeater {i}")
    xgb_cv.append(
        xgb.cv(
          params = param,
          dtrain = dtrain,
          obj = cauchyobj,
          num_boost_round = 3000,
          folds = KFold(n_splits = 5, shuffle = True, random_state = i),
          early_stopping_rounds = 25,
          verbose_eval = 50
        )
    )


iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]
# Return index with minimal value
val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]

#interation_counts => for num_boost_round setting
iteration_counts, val_mae





oof_preds = [] #T1_score-T2_score score diff
for i in range(repeat_cv):
    print(f"Fold repeater {i}")
    preds = y.copy()
    kfold = KFold(n_splits = 5, shuffle = True, random_state = i)    
    for train_index, val_index in kfold.split(X,y):
        dtrain_i = xgb.DMatrix(X[train_index], label = y[train_index])
        dval_i = xgb.DMatrix(X[val_index], label = y[val_index])  
        model = xgb.train(
              params = param,
              dtrain = dtrain_i,
              num_boost_round = iteration_counts[i],
              verbose_eval = 50
        )
        preds[val_index] = model.predict(dval_i)
    oof_preds.append(np.clip(preds,-30,30))




#oof_preds->ì˜ˆì¸¡ë�œ ì �ìˆ˜ ì°¨ì�´, np.where(y>0,1,0)-> ì‹¤ì œ ê²½ê¸° ê²°ê³¼
#ì �ìˆ˜ì°¨ì�´ì—� ì�˜í•œ ìŠ¹ë¥  ê·¸ë�˜í”„ ì‹œê°�í™”
plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0)})
plot_df["pred_int"] = plot_df["pred"].astype(int)
plot_df = plot_df.groupby('pred_int')['label'].mean().reset_index(name='average_win_pct')

plt.figure()
plt.plot(plot_df.pred_int,plot_df.average_win_pct)


spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0))) #ì �ìˆ˜ì°¨ì�´ì™€ ì‹¤ì œ ìŠ¹íŒ¨ ë¬¶ì�Œ
    dat = sorted(dat, key = lambda x: x[0]) #ì˜ˆì¸¡ë�œ ì �ìˆ˜ ì°¨ì�´ë¥¼ ê¸°ì¤€ìœ¼ë¡œ ì •ë ¬ / Splineì�˜ ê²½ìš° ì •ë ¬ë�˜ì–´ ì�ˆì–´ì•¼ í•™ìŠµì�´ ì�˜ë�¨.
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
        
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    
    print(f"logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 



plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0), "spline":spline_model[0](oof_preds[0])})
plot_df["pred_int"] = (plot_df["pred"]).astype(int)
plot_df = plot_df.groupby('pred_int')[['spline', 'label']].mean().reset_index()

plt.figure()
plt.plot(plot_df.pred_int,plot_df.spline)
plt.plot(plot_df.pred_int,plot_df.label)



# spline_model = []

# for i in range(repeat_cv):
#     dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
#     dat = sorted(dat, key = lambda x: x[0])
#     datdict = {}
#     for k in range(len(dat)):
#         datdict[dat[k][0]]= dat[k][1]
#     spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
#     spline_fit = spline_model[i](oof_preds[i])
#     spline_fit = np.clip(spline_fit,0.025,0.975)
    
#     print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}")





# spline_model = []

# for i in range(repeat_cv):
#     dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
#     dat = sorted(dat, key = lambda x: x[0])
#     datdict = {}
#     for k in range(len(dat)):
#         datdict[dat[k][0]]= dat[k][1]
#     spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
#     spline_fit = spline_model[i](oof_preds[i])
#     spline_fit = np.clip(spline_fit,0.025,0.975)
#     spline_fit[(tourney_data.T1_seed==1) & (tourney_data.T2_seed==16)] = 1.0
#     spline_fit[(tourney_data.T1_seed==2) & (tourney_data.T2_seed==15)] = 1.0
#     spline_fit[(tourney_data.T1_seed==3) & (tourney_data.T2_seed==14)] = 1.0
#     spline_fit[(tourney_data.T1_seed==4) & (tourney_data.T2_seed==13)] = 1.0
#     spline_fit[(tourney_data.T1_seed==16) & (tourney_data.T2_seed==1)] = 0.0
#     spline_fit[(tourney_data.T1_seed==15) & (tourney_data.T2_seed==2)] = 0.0
#     spline_fit[(tourney_data.T1_seed==14) & (tourney_data.T2_seed==3)] = 0.0
#     spline_fit[(tourney_data.T1_seed==13) & (tourney_data.T2_seed==4)] = 0.0
    
#     print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 


# #looking for upsets
# pd.concat(
#     [tourney_data[(tourney_data.T1_seed==1) & (tourney_data.T2_seed==16) & (tourney_data.T1_Score < tourney_data.T2_Score)],
#      tourney_data[(tourney_data.T1_seed==2) & (tourney_data.T2_seed==15) & (tourney_data.T1_Score < tourney_data.T2_Score)],
#      tourney_data[(tourney_data.T1_seed==3) & (tourney_data.T2_seed==14) & (tourney_data.T1_Score < tourney_data.T2_Score)],
#      tourney_data[(tourney_data.T1_seed==4) & (tourney_data.T2_seed==13) & (tourney_data.T1_Score < tourney_data.T2_Score)],
#      tourney_data[(tourney_data.T1_seed==16) & (tourney_data.T2_seed==1) & (tourney_data.T1_Score > tourney_data.T2_Score)],
#      tourney_data[(tourney_data.T1_seed==15) & (tourney_data.T2_seed==2) & (tourney_data.T1_Score > tourney_data.T2_Score)],
#      tourney_data[(tourney_data.T1_seed==14) & (tourney_data.T2_seed==3) & (tourney_data.T1_Score > tourney_data.T2_Score)],
#      tourney_data[(tourney_data.T1_seed==13) & (tourney_data.T2_seed==4) & (tourney_data.T1_Score > tourney_data.T2_Score)]]
# )   


# spline_model = []

# for i in range(repeat_cv):
#     dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
#     dat = sorted(dat, key = lambda x: x[0])
#     datdict = {}
#     for k in range(len(dat)):
#         datdict[dat[k][0]]= dat[k][1]
#     spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
#     spline_fit = spline_model[i](oof_preds[i])
#     spline_fit = np.clip(spline_fit,0.025,0.975)
#     spline_fit[(tourney_data.T1_seed==1) & (tourney_data.T2_seed==16) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
#     spline_fit[(tourney_data.T1_seed==2) & (tourney_data.T2_seed==15) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
#     spline_fit[(tourney_data.T1_seed==3) & (tourney_data.T2_seed==14) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
#     spline_fit[(tourney_data.T1_seed==4) & (tourney_data.T2_seed==13) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
#     spline_fit[(tourney_data.T1_seed==16) & (tourney_data.T2_seed==1) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
#     spline_fit[(tourney_data.T1_seed==15) & (tourney_data.T2_seed==2) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
#     spline_fit[(tourney_data.T1_seed==14) & (tourney_data.T2_seed==3) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
#     spline_fit[(tourney_data.T1_seed==13) & (tourney_data.T2_seed==4) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    
#     print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 


val_cv = []
spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    spline_fit = np.clip(spline_fit,0.025,0.975)
    spline_fit[(tourney_data.T1_seed==1) & (tourney_data.T2_seed==16) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==2) & (tourney_data.T2_seed==15) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==3) & (tourney_data.T2_seed==14) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==4) & (tourney_data.T2_seed==13) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==16) & (tourney_data.T2_seed==1) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    spline_fit[(tourney_data.T1_seed==15) & (tourney_data.T2_seed==2) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    spline_fit[(tourney_data.T1_seed==14) & (tourney_data.T2_seed==3) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    spline_fit[(tourney_data.T1_seed==13) & (tourney_data.T2_seed==4) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    
    val_cv.append(pd.DataFrame({"y":np.where(y>0,1,0), "pred":spline_fit, "season":tourney_data.Season}))
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 
    
val_cv = pd.concat(val_cv)
val_cv.groupby('season').apply(lambda x: log_loss(x.y, x.pred))


"""
sub["Season"] = 2018
sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)
sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)
sub.head()
"""
sub = pd.read_csv(DATA_PATH + "SampleSubmissionStage2.csv")
sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
sub["T1_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[1]))
sub["T2_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[2]))
sub.head()


sub = pd.merge(sub, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')

sub = pd.merge(sub, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')
sub = pd.merge(sub, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')
sub = pd.merge(sub, season_win_stats_T1, on=['Season', 'T1_TeamID'], how='left')
sub = pd.merge(sub, season_win_stats_T2, on=['Season', 'T2_TeamID'], how='left')
sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]

sub.head()





Xsub = sub[features].values
dtest = xgb.DMatrix(Xsub)


sub_models = []
for i in range(repeat_cv):
    print(f"Fold repeater {i}")
    sub_models.append(
        xgb.train(
          params = param,
          dtrain = dtrain,
          num_boost_round = int(iteration_counts[i] * 1.05),
          verbose_eval = 50
        )
    )


import pandas as pd

# âœ… ê¸°ì¡´ ScoreDiff ìœ ì§€ (ë³€ë�™ ì—†ì�Œ)
predicted_score_diff = np.mean(
    [sub_models[i].predict(dtest) for i in range(repeat_cv)], axis=0
)

# âœ… ê¸°ì¡´ ì œì¶œë�œ Pred ê°’ì�„ ìœ ì§€í•œ ì±„ ScoreDiff ì¶”ê°€
sub["ScoreDiff"] = predicted_score_diff  

# âœ… íŠ¹ì • ID(ì˜ˆ: 2025_3106_3163) ê²€ìƒ‰í•˜ì—¬ Predì™€ ScoreDiff ë¹„êµ�
target_id = "2025_3106_3163"
analysis_df = sub[sub["ID"] == target_id]

analysis_df


sub_preds = []
for i in range(repeat_cv):
    # sub_preds.append(np.clip(spline_model[i](np.clip(sub_models[i].predict(dtest),-30,30)),0.025,0.975))
    #sub_preds.append(spline_model[i](sub_models[i].predict(dtest)))
    sub_preds.append(np.clip(spline_model[i](np.clip(sub_models[i].predict(dtest), -30, 30)), 0, 1))

sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)
sub.loc[sub["ScoreDiff"] >= 23, "Pred"] = 1.0
sub.loc[sub["ScoreDiff"] <= -23, "Pred"] = 0.0
sub.loc[(sub.T1_seed==1) & (sub.T2_seed==16), 'Pred'] = 1.0
sub.loc[(sub.T1_seed==16) & (sub.T2_seed==1), 'Pred'] = 0.0

"""
sub.loc[(sub.T1_seed==1) & (sub.T2_seed==16), 'Pred'] = 1.0
sub.loc[(sub.T1_seed==2) & (sub.T2_seed==15), 'Pred'] = 1.0
sub.loc[(sub.T1_seed==3) & (sub.T2_seed==14), 'Pred'] = 1.0
sub.loc[(sub.T1_seed==4) & (sub.T2_seed==13), 'Pred'] = 1.0
sub.loc[(sub.T1_seed==16) & (sub.T2_seed==1), 'Pred'] = 0.0
sub.loc[(sub.T1_seed==15) & (sub.T2_seed==2), 'Pred'] = 0.0
sub.loc[(sub.T1_seed==14) & (sub.T2_seed==3), 'Pred'] = 0.0
sub.loc[(sub.T1_seed==13) & (sub.T2_seed==4), 'Pred'] = 0.0
"""
sub[['ID','Pred']].to_csv("real_final_overriding_no_clip.csv", index = None)


filtered_df = sub[sub["ScoreDiff"].between(-23.3, -23)]
filtered_df[['Pred', 'ScoreDiff']]


print(sub["ScoreDiff"].sort_values().head(20))  # ê°€ì�¥ ì�‘ì�€ ê°’ 20ê°œ ì¶œë ¥
print(sub["ScoreDiff"].sort_values().tail(20))  # ê°€ì�¥ í�° ê°’ 20ê°œ ì¶œë ¥









import matplotlib.pyplot as plt

# ScoreDiff í�ˆìŠ¤í† ê·¸ë�¨
plt.figure(figsize=(6, 3))
plt.hist(sub["ScoreDiff"], bins=50, edgecolor="black")
plt.title("Distribution of ScoreDiff")
plt.xlabel("ScoreDiff")
plt.ylabel("Frequency")
plt.show()

# Pred í�ˆìŠ¤í† ê·¸ë�¨
plt.figure(figsize=(6, 3))
plt.hist(sub["Pred"], bins=50, edgecolor="black")
plt.title("Distribution of Pred")
plt.xlabel("Pred")
plt.ylabel("Frequency")
plt.show()






sub[sub['ID']=="2025_3106_3163"]





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1ï¸�âƒ£ ëª¨ë�¸ì�´ ì˜ˆì¸¡í•œ ì �ìˆ˜ ì°¨ì�´ê°’ê³¼ ë³€í™˜ë�œ í™•ë¥ ê°’ ê°€ì ¸ì˜¤ê¸°
predicted_score_diff = np.mean([np.clip(sub_models[i].predict(dtest), -50, 50) for i in range(repeat_cv)], axis=0)
predicted_win_prob = np.mean([spline_model[i](predicted_score_diff) for i in range(repeat_cv)], axis=0)

# 2ï¸�âƒ£ ë�°ì�´í„°ë¥¼ ì •ë ¬í•˜ì—¬ ê·¸ë�˜í”„ë¥¼ ê·¸ë¦´ ìˆ˜ ì�ˆë�„ë¡� ë³€í™˜
plot_df = pd.DataFrame({"ScoreDiff": predicted_score_diff, "WinProb": predicted_win_prob})
plot_df = plot_df.sort_values("ScoreDiff")  # Xì¶•(ì �ìˆ˜ ì°¨ì�´)ì�„ ì •ë ¬

# 3ï¸�âƒ£ ê·¸ë�˜í”„ ê·¸ë¦¬ê¸°
plt.figure(figsize=(8, 5))

# ìŠ¤í”Œë�¼ì�¸ ë³€í™˜ë�œ í™•ë¥ 
plt.plot(plot_df["ScoreDiff"], plot_df["WinProb"], label="Spline Fitted Probabilities", color='blue')

# ëª¨ë�¸ì�´ ì˜ˆì¸¡í•œ ì›�ë³¸ ì �ìˆ˜ ì°¨ì�´ & í™•ë¥ 
plt.scatter(plot_df["ScoreDiff"], plot_df["WinProb"], alpha=0.3, color='orange', label="Model Predictions")

# ê¸°ì¤€ì„  ì¶”ê°€
plt.axhline(0.5, linestyle="--", color="gray", alpha=0.5)  # 50% í™•ë¥  ê¸°ì¤€ì„ 
plt.axvline(0, linestyle="--", color="gray", alpha=0.5)  # ì �ìˆ˜ ì°¨ì�´ 0 ê¸°ì¤€ì„ 

plt.xlabel("Predicted Score Difference")
plt.ylabel("Win Probability")
plt.title("Sorted Score Difference vs. Win Probability")
plt.legend()
plt.show()



import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(predicted_score_diff, bins=50, edgecolor="black")
plt.xlabel("Predicted Score Difference")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Score Differences")
plt.show()


import pandas as pd

# ëª¨ë�¸ì�´ ì˜ˆì¸¡í•œ ì �ìˆ˜ ì°¨ì�´ì™€ ë³€í™˜ë�œ í™•ë¥  ê°€ì ¸ì˜¤ê¸°
predicted_score_diff = np.mean([np.clip(sub_models[i].predict(dtest), -30, 30) for i in range(repeat_cv)], axis=0)
predicted_win_prob = np.mean([spline_model[i](predicted_score_diff) for i in range(repeat_cv)], axis=0)

# ë�°ì�´í„°í”„ë ˆì�„ìœ¼ë¡œ ì •ë¦¬
predictions_df = pd.DataFrame({"ScoreDiff": predicted_score_diff, "WinProb": predicted_win_prob})

# ì •ë ¬ë�œ ë�°ì�´í„° ì¶œë ¥
predictions_df = predictions_df.sort_values("ScoreDiff")  # ì �ìˆ˜ ì°¨ì�´ ê¸°ì¤€ìœ¼ë¡œ ì •ë ¬
predictions_df





def filter_win_prob(predictions_df, min_diff, max_diff):
    """
    íŠ¹ì • ì �ìˆ˜ ì°¨ì�´ ë²”ìœ„ì—� í•´ë‹¹í•˜ëŠ” ìŠ¹ë¦¬ í™•ë¥  ì¡°íšŒ í•¨ìˆ˜
    :param predictions_df: ì �ìˆ˜ ì°¨ì�´ì™€ ìŠ¹ë¦¬ í™•ë¥  ë�°ì�´í„°í”„ë ˆì�„
    :param min_diff: ìµœì†Œ ì �ìˆ˜ ì°¨ì�´
    :param max_diff: ìµœëŒ€ ì �ìˆ˜ ì°¨ì�´
    :return: í•´ë‹¹ ë²”ìœ„ ë‚´ì�˜ ë�°ì�´í„° ì¶œë ¥
    """
    filtered_df = predictions_df[(predictions_df["ScoreDiff"] >= min_diff) & (predictions_df["ScoreDiff"] <= max_diff)]
    
    # ì¡°íšŒë�œ ë�°ì�´í„°ê°€ ë§�ìœ¼ë©´ ìƒ�ìœ„ 10ê°œë§Œ ì¶œë ¥
    if len(filtered_df) > 10:
        print(f"ì¡°íšŒë�œ ë�°ì�´í„°ê°€ {len(filtered_df)}ê°œ ì�ˆìŠµë‹ˆë‹¤. ìƒ�ìœ„ 10ê°œë§Œ ì¶œë ¥í•©ë‹ˆë‹¤.\n")
        display(filtered_df.head(10))
    else:
        display(filtered_df)

# âœ… ì˜ˆì œ: -5 ~ 5 ì‚¬ì�´ì�˜ ì �ìˆ˜ ì°¨ì�´ì—� ëŒ€í•œ ìŠ¹ë¦¬ í™•ë¥  ì¡°íšŒ
filter_win_prob(predictions_df, 23.30, 24.3)



# ë‘� ê°’ì�´ ë�™ì�¼í•œì§€ ë¹„êµ�
comparison_df = pd.DataFrame({
    "ID": sub["ID"],  # ì œì¶œ íŒŒì�¼ì�˜ ID
    "Pred_in_submission": sub["Pred"],  # ìµœì¢… ì œì¶œê°’
    "Pred_in_analysis": predictions_df["WinProb"]  # ìœ„ì—�ì„œ ë¶„ì„�í•œ ìŠ¹ë¦¬ í™•ë¥ 
})

# ì°¨ì�´ê°€ ì�ˆëŠ” í–‰ë§Œ í•„í„°ë§�
diff_df = comparison_df[comparison_df["Pred_in_submission"] != comparison_df["Pred_in_analysis"]]

diff_df


# ì˜ˆì¸¡ í™•ë¥  ë¶„í�¬ ì‹œê°�í™”
plt.figure(figsize=(8, 5))
plt.hist(sub["Pred"], bins=30, edgecolor="black", alpha=0.7)
plt.axvline(sub["Pred"].mean(), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {sub['Pred'].mean():.3f}")
plt.xlabel("Predicted Win Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Win Probabilities (Final Submission)")
plt.legend()
plt.grid(True)
plt.show()



win_counts_2025 = regular_season_effects.loc[regular_season_effects['Season']==2025, 'win'].value_counts()
print(win_counts_2025)
regular_season_effects.query("Season == 2025").groupby("T1_TeamID")['win'].mean().sort_values()



team_quality(2025)


for i in range(repeat_cv):
    # Step 1: sub_modelsê°€ ì˜ˆì¸¡í•œ ì �ìˆ˜ ì°¨ì�´ ê°€ì ¸ì˜¤ê¸°
    predicted_score_diff = np.clip(sub_models[i].predict(dtest), -30, 30)
    
    # Step 2: Spline ë³€í™˜ì�„ ì �ìš©í•˜ì—¬ í™•ë¥  ê°’ìœ¼ë¡œ ë³€í™˜
    predicted_win_prob = spline_model[i](predicted_score_diff)

    # Step 3: íŠ¹ì • ì �ìˆ˜ ì°¨ì�´ ì�´ìƒ�ë§Œ í•„í„°ë§� (ì˜ˆ: 20ì � ì�´ìƒ�)
    mask = predicted_score_diff > 20
    
    print(f"\n=== Fold {i} ===")
    print("Predicted Score Differences (20+):", predicted_score_diff[mask])  # ì˜ˆì¸¡ë�œ ì �ìˆ˜ ì°¨ì�´
    print("Converted Win Probabilities (20+):", predicted_win_prob[mask])  # ë³€í™˜ë�œ í™•ë¥  ê°’


# âœ… 2024ë…„ ë�°ì�´í„°ë§Œ ìƒˆë¡­ê²Œ ì¸¡ì •í•˜ê¸° ìœ„í•´ ë³€ìˆ˜ëª…ì�„ ë‹¤ë¥´ê²Œ ì„¤ì •

# 1ï¸�âƒ£ í•™ìŠµ ë�°ì�´í„°ì—�ì„œ 2024ë…„ ë�°ì�´í„°ë¥¼ ì œì™¸ (ìƒˆë¡œìš´ ë³€ìˆ˜ ì‚¬ìš©)
train_data_eval = tourney_data[tourney_data['Season'] < 2024]  # 2024ë…„ ì œì™¸
X_train_eval = train_data_eval[features].values
y_train_eval = train_data_eval['T1_Score'] - train_data_eval['T2_Score']

dtrain_eval = xgb.DMatrix(X_train_eval, label=y_train_eval)

# 2ï¸�âƒ£ 2024ë…„ í† ë„ˆë¨¼íŠ¸ ë�°ì�´í„°ë§Œ ì‚¬ìš©í•˜ì—¬ í�‰ê°€
df_2024_eval = tourney_data[tourney_data['Season'] == 2024]  # 2024ë…„ í† ë„ˆë¨¼íŠ¸ ê²½ê¸°ë§Œ
X_2024_eval = df_2024_eval[features].values
dtest_2024_eval = xgb.DMatrix(X_2024_eval)

# 3ï¸�âƒ£ ëª¨ë�¸ í›ˆë ¨ (ìƒˆë¡œìš´ ëª¨ë�¸ ë¦¬ìŠ¤íŠ¸ ì‚¬ìš©)
sub_models_eval = []
for i in range(repeat_cv):
    model_eval = xgb.train(params=param, dtrain=dtrain_eval, num_boost_round=iteration_counts[i])
    sub_models_eval.append(model_eval)

# 4ï¸�âƒ£ ì˜ˆì¸¡ ë°� Brier Score í�‰ê°€
preds_list_eval = []
for i in range(repeat_cv):
    margin_2024_eval = sub_models_eval[i].predict(dtest_2024_eval)
    margin_2024_eval = np.clip(margin_2024_eval, -30, 30)
    prob_2024_eval = spline_model[i](margin_2024_eval)
    prob_2024_eval = np.clip(prob_2024_eval, 0.0, 1)
    preds_list_eval.append(prob_2024_eval)

final_prob_2024_eval = np.mean(preds_list_eval, axis=0)

y_2024_eval = (df_2024_eval['T1_Score'] > df_2024_eval['T2_Score']).astype(int)
brier_2024_eval = np.mean((final_prob_2024_eval - y_2024_eval)**2)

# 5ï¸�âƒ£ ê²°ê³¼ ì¶œë ¥
print("Brier Score for 2024 (re-evaluated):", brier_2024_eval)


















# import optuna
# import xgboost as xgb

# def objective(trial):
#     param_opt = {
#         'eta': trial.suggest_float('eta', 0.01, 0.1, log=True),  
#         'subsample': trial.suggest_float('subsample', 0.5, 0.8),  
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),  
#         'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
#         'gamma': trial.suggest_float('gamma', 2, 5),  
#         'max_depth': trial.suggest_int('max_depth', 3, 6),
#         'num_parallel_tree': 1,  
#         'eval_metric': 'mae'
#     }

#     dtrain = xgb.DMatrix(X_train_eval_2023, label=y_train_eval_2023)

#     cv_result = xgb.cv(
#         param_opt, dtrain, num_boost_round=1000, nfold=5, 
#         early_stopping_rounds=25, metrics=['mae'], as_pandas=True
#     )

#     return cv_result['test-mae-mean'].min()

# # âœ… Optuna ì‹¤í–‰ (Brier Scoreë¥¼ ìµœì �í™”í•˜ëŠ” ë°©í–¥ìœ¼ë¡œ)
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)  # 50ë²ˆ íƒ�ìƒ‰ ìˆ˜í–‰

# # âœ… ìµœì �ì�˜ íŒŒë�¼ë¯¸í„° ì �ìš©
# param.update(study.best_params)
# print("ğŸ”¥ ìµœì �í™”ë�œ íŒŒë�¼ë¯¸í„° ì �ìš© ì™„ë£Œ!")
# print(param)



param = {} 
# param['objective'] = 'reg:linear'  # í˜„ì�¬ ì‚¬ìš©í•˜ì§€ ì•Šì�Œ (íšŒê·€ ë¬¸ì œ ì•„ë‹˜)
param['eval_metric'] = 'mae'
param['booster'] = 'gbtree'
param['eta'] = 0.05  # í•™ìŠµë¥  (ê¸°ë³¸ 0.02 ê¶Œì�¥)
param['subsample'] = 0.35
param['colsample_bytree'] = 0.7
param['num_parallel_tree'] = 10  # ë³‘ë ¬ íŠ¸ë¦¬ ê°œìˆ˜ (10 ì¶”ì²œ)
param['min_child_weight'] = 40
param['gamma'] = 10
param['max_depth'] = 3
param['verbosity'] = 1  # ìµœì‹  XGBoostì—�ì„œëŠ” ì‚¬ìš© ì•ˆ í•¨ (ê²½ê³  ë°œìƒ� ê°€ëŠ¥)

# âœ… GPU í™œì„±í™” ì—¬ë¶€ í™•ì�¸ í›„ ì„¤ì • ì �ìš©
if USE_GPU:
    param.update({
        'tree_method': 'hist',
        'device' : 'cuda'  # GPU ì‚¬ìš©í•˜ì—¬ í•™ìŠµ ê°€ì†�í™”
    })

# âœ… íŒŒë�¼ë¯¸í„° ì¶œë ¥ (GPU í™œì„±í™” ì—¬ë¶€ í™•ì�¸)
print(param)


# âœ… 2023ë…„ ê¸°ì¤€ìœ¼ë¡œ Brier Score ì¸¡ì • (2023 ëŒ€íšŒ ë°©ì‹� ì �ìš©)

# 1ï¸�âƒ£ í•™ìŠµ ë�°ì�´í„°ì—�ì„œ 2023ë…„ ë�°ì�´í„°ë¥¼ ì œì™¸í•˜ê³  í•™ìŠµ
train_data_eval_2023 = tourney_data[tourney_data['Season'] < 2023]  # 2023ë…„ ì œì™¸
X_train_eval_2023 = train_data_eval_2023[features].values
y_train_eval_2023 = train_data_eval_2023['T1_Score'] - train_data_eval_2023['T2_Score']

dtrain_eval_2023 = xgb.DMatrix(X_train_eval_2023, label=y_train_eval_2023)

# 2ï¸�âƒ£ 2023ë…„ í† ë„ˆë¨¼íŠ¸ ë�°ì�´í„°ë§Œ ì‚¬ìš©í•˜ì—¬ í�‰ê°€
df_2023_eval = tourney_data[tourney_data['Season'] == 2023]  # 2023ë…„ í† ë„ˆë¨¼íŠ¸ ê²½ê¸°ë§Œ
X_2023_eval = df_2023_eval[features].values
dtest_2023_eval = xgb.DMatrix(X_2023_eval)

# 3ï¸�âƒ£ ëª¨ë�¸ í›ˆë ¨ (ìƒˆë¡œìš´ ëª¨ë�¸ ë¦¬ìŠ¤íŠ¸ ì‚¬ìš©)
sub_models_eval_2023 = []
for i in range(repeat_cv):
    model_eval_2023 = xgb.train(params=param, dtrain=dtrain_eval_2023, num_boost_round=iteration_counts[i])
    sub_models_eval_2023.append(model_eval_2023)

# 4ï¸�âƒ£ ì˜ˆì¸¡ ë°� Brier Score í�‰ê°€
preds_list_eval_2023 = []
for i in range(repeat_cv):
    margin_2023_eval = sub_models_eval_2023[i].predict(dtest_2023_eval)
    margin_2023_eval = np.clip(margin_2023_eval, -30, 30)
    prob_2023_eval = spline_model[i](margin_2023_eval)
    prob_2023_eval = np.clip(prob_2023_eval, 0.0, 1)
    preds_list_eval_2023.append(prob_2023_eval)

final_prob_2023_eval = np.mean(preds_list_eval_2023, axis=0)

y_2023_eval = (df_2023_eval['T1_Score'] > df_2023_eval['T2_Score']).astype(int)
final_prob_2023_eval[df_2023_eval["T1_Score"] - df_2023_eval["T2_Score"] >= 23] = 1.0
final_prob_2023_eval[df_2023_eval["T1_Score"] - df_2023_eval["T2_Score"] <= -23] = 0.0
final_prob_2023_eval[(df_2023_eval["T1_seed"] == 1) & (df_2023_eval["T2_seed"] == 16)] = 1.0
final_prob_2023_eval[(df_2023_eval["T1_seed"] == 16) & (df_2023_eval["T2_seed"] == 1)] = 0.0

brier_2023_eval = np.mean((final_prob_2023_eval - y_2023_eval)**2)

# 5ï¸�âƒ£ ê²°ê³¼ ì¶œë ¥
print("Brier Score for 2023 (re-evaluated, 2023 Kaggle Style):", brier_2023_eval)



0.17904065546131856
Brier Score for 2023 (re-evaluated, 2023 Kaggle Style): 0.17902283435384325 #clipì œì™¸



import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# âœ… Feature Importance ê°€ì ¸ì˜¤ê¸°
importance = model.get_score(importance_type='weight')

# âœ… Feature ì�´ë¦„ ë§¤í•‘ (numpy ë°°ì—´ ëŒ€ë¹„ ëŒ€ì�‘)
feature_names = features  # Xê°€ numpy ë°°ì—´ì�´ë�¼ë©´ ì§�ì ‘ ë¦¬ìŠ¤íŠ¸ ì§€ì • í•„ìš”
importance_named = {feature_names[int(k[1:])]: v for k, v in importance.items()}

# âœ… ì¤‘ìš”ë�„ ë‚®ì�€ ìˆœìœ¼ë¡œ ì •ë ¬
sorted_importance = sorted(importance_named.items(), key=lambda x: x[1])

# âœ… í•˜ìœ„ 20ê°œ Featureë§Œ í‘œì‹œ
bottom_n = 25
sorted_importance = sorted_importance[:bottom_n]

# âœ… ì‹œê°�í™”
plt.figure(figsize=(12, 8))  # ê·¸ë�˜í”„ í�¬ê¸° ì¦�ê°€
plt.barh([x[0] for x in sorted_importance], [x[1] for x in sorted_importance])
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance (Bottom 20 Features)', fontsize=16)
plt.yticks(fontsize=12)  # Yì¶• ê¸€ì�� í�¬ê¸° í‚¤ìš°ê¸°
plt.show()



feature_name = "T1_EFFG_mean"  # ì¡°íšŒí•˜ê³  ì‹¶ì�€ Feature ì�´ë¦„
importance_value = importance_named.get(feature_name, "Not Found")
print(f"Feature '{feature_name}' Importance Score:", importance_value)



print("Stored Feature Names:", list(importance_named.keys()))



import seaborn as sns
import matplotlib.pyplot as plt

# âœ… Feature ìƒ�ê´€ê´€ê³„ ê³„ì‚°
corr_matrix = X_train.corr()

# âœ… Heatmap ê·¸ë¦¬ê¸°
plt.figure(figsize=(12, 10))  # ê·¸ë�˜í”„ í�¬ê¸° ì„¤ì •
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

# âœ… ì œëª© ì„¤ì •
plt.title("Feature Correlation Matrix", fontsize=16)
plt.show()


