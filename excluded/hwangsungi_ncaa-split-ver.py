


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


PREVIOUS_SEASONS_MEN = False
PREVIOUS_SEASONS_WOMEN  = False 
USE_GPU = True # Turn on GPU P100 if USE_GPU=True
USE_ADDITONAL_COLUMN = False
USE_SIMPLEFLAG = False
USE_YEAR_WINFLAG = False
USE_DAYS_WINFLAG = True


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


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


import os

DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025/"
tourney_results = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv")
], ignore_index=True)

seeds = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv")
], ignore_index=True)
#  2025 ë�°ì�´í„°ê°€ ì�ˆëŠ”ì§€ í™•ì�¸
if 2025 not in seeds["Season"].unique():
    print(" 2025 ì‹œì¦Œ ë�°ì�´í„°ê°€ ì—†ìœ¼ë¯€ë¡œ 2024 ë�°ì�´í„°ë¥¼ ë³µì‚¬í•˜ì—¬ ìƒ�ì„±í•©ë‹ˆë‹¤.")
    
    # 2024 ì‹œì¦Œ ë�°ì�´í„° ë³µì‚¬
    seeds_2025 = seeds[seeds["Season"] == 2024].copy()
    seeds_2025["Season"] = 2025  # ì‹œì¦Œì�„ 2025ë¡œ ë³€ê²½
    
    # ì›�ë³¸ ë�°ì�´í„°ì™€ í•©ì¹˜ê¸°
    seeds = pd.concat([seeds, seeds_2025], ignore_index=True)
    
    print(" 2025 ì‹œì¦Œ ë�°ì�´í„°ê°€ ì„±ê³µì �ìœ¼ë¡œ ì¶”ê°€ë�˜ì—ˆìŠµë‹ˆë‹¤.")
else:
    print(" 2025 ì‹œì¦Œ ë�°ì�´í„°ê°€ ì�´ë¯¸ ì¡´ì�¬í•©ë‹ˆë‹¤.")
    
#  ìµœì¢… ë�°ì�´í„° í™•ì�¸
print(seeds.tail(10))  # ë§ˆì§€ë§‰ 10ê°œ ë�°ì�´í„° í™•ì�¸
regular_results = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv")
], ignore_index=True)


def prepare_data(df_data, use_additional_column=False):
    df = df_data.copy()
    df.rename(columns={'WLoc': 'location'}, inplace=True)
    
    # ìŠ¤ì™‘ìš© ë�°ì�´í„°í”„ë ˆì�„ ìƒ�ì„± (Lâ†’T1, Wâ†’T2 ë³€í™˜)
    dfswap = df[[
        'Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'location', 'NumOT', 
        'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
        'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'
    ]]

    #  ì¶”ê°€ ì»¬ëŸ¼ ë¦¬ìŠ¤íŠ¸
    additional_cols = ['WEFFG', 'WEFFG3', 'WDARE', 'WTOQUETOQUE',
                       'LEFFG', 'LEFFG3', 'LDARE', 'LTOQUETOQUE']
    
    #  ì¶”ê°€ ì¹¼ëŸ¼ì�„ ë°˜ì˜�í• ì§€ ì—¬ë¶€ í™•ì�¸
    if use_additional_column:
        df = df.assign(**{col: df[col.split('_')[1]] / df[col.split('_')[1] + 'A'] for col in additional_cols})
        dfswap = dfswap.assign(**{col: dfswap[col.split('_')[1]] / dfswap[col.split('_')[1] + 'A'] for col in additional_cols})

    #  ì»¬ëŸ¼ëª… ë³€í™˜: ìŠ¹ë¦¬íŒ€(W) â†’ T1, íŒ¨ë°°íŒ€(L) â†’ T2
    df.columns = df.columns.str.replace('W', 'T1_')
    df.columns = df.columns.str.replace('L', 'T2_')

    #  dfswap(íŒ¨ë°°íŒ€ì�„ T1ìœ¼ë¡œ, ìŠ¹ë¦¬íŒ€ì�„ T2ë¡œ ë³€ê²½)
    dfswap.columns = dfswap.columns.str.replace('L', 'T1_')
    dfswap.columns = dfswap.columns.str.replace('W', 'T2_')

    #  í•œ ê²½ê¸°(W/L)ë¥¼ ë‘� ê°œì�˜ T1/T2 í˜•íƒœë¡œ ë³€í™˜
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    
    #  ê²½ê¸° ì�¥ì†Œ ë³€í™˜: N â†’ 0, H â†’ 1, A â†’ -1
    output.loc[output.location == 'N', 'location'] = '0'
    output.loc[output.location == 'H', 'location'] = '1'
    output.loc[output.location == 'A', 'location'] = '-1'
    output['location'] = output['location'].astype(int)

    #  ì �ìˆ˜ ì°¨ì�´ ê³„ì‚° (íƒ€ê²Ÿ ë³€ìˆ˜ë¡œ í™œìš© ê°€ëŠ¥)
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']

    #  ì¶”ê°€ íŒŒìƒ� ë³€ìˆ˜ ìƒ�ì„± (use_additional_column í™œì„±í™” ì‹œ)
    if use_additional_column:
        output['T1_EFFG'] = output['T1_FGM'] / output['T1_FGA']
        output['T1_EFFG3'] = output['T1_FGM3'] / output['T1_FGA3']
        output['T1_DARE'] = output['T1_FGM3'] / output['T1_FGM']
        output['T1_TOQUETOQUE'] = output['T1_Ast'] / output['T1_FGM']
        
        output['T2_EFFG'] = output['T2_FGM'] / output['T2_FGA']
        output['T2_EFFG3'] = output['T2_FGM3'] / output['T2_FGA3']
        output['T2_DARE'] = output['T2_FGM3'] / output['T2_FGM']
        output['T2_TOQUETOQUE'] = output['T2_Ast'] / output['T2_FGM']

        # ë¶„ëª¨ 0ì�¼ ê²½ìš° NaN ë°œìƒ� â†’ 0ìœ¼ë¡œ ëŒ€ì²´
        cols_to_fill = ['T1_EFFG','T1_EFFG3','T1_DARE','T1_TOQUETOQUE',
                        'T2_EFFG','T2_EFFG3','T2_DARE','T2_TOQUETOQUE']
        output[cols_to_fill] = output[cols_to_fill].fillna(0.0)

    return output



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


season_statistics.shape


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
    
    # 1ï¸�âƒ£ ì´ˆê¸°ê°’ì�„ 0.0 (float)ìœ¼ë¡œ ì„¤ì •í•˜ì—¬ ì��ë�™ float ë³€í™˜
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
            
            # 2ï¸�âƒ£ float ë³€í™˜ í›„ í• ë‹¹
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


season_statistics_T1.shape


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

# âœ… T2 íŒ€ì�˜ ì •ê·œì‹œì¦Œ ì „ì²´ ìŠ¹ë¥  ê³„ì‚°
season_win_stats_T2 = regular_data.copy()
season_win_stats_T2['win'] = np.where(season_win_stats_T2['PointDiff'] < 0, 1, 0)
season_win_stats_T2 = season_win_stats_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_season')



print(season_statistics_T1.shape)  # season_statistics_T1ì�˜ í�¬ê¸° í™•ì�¸
print(season_statistics_T2.shape)  # season_statistics_T2ì�˜ í�¬ê¸° í™•ì�¸



tourney_data = pd.merge(tourney_data, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')
if USE_YEAR_WINFLAG:
    tourney_data = pd.merge(tourney_data, season_win_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, season_win_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')



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


def normalize_column(values):
    themean = np.mean(values)
    thestd = np.std(values)
    norm = (values - themean)/(thestd) 
    return(pd.DataFrame(norm))

def team_quality(season):
    formula = 'win~-1+T1_TeamID+T2_TeamID'
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
    print(quality['quality'].mean(), quality['quality'].std())
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

# ìƒ�ìœ„ 1% ê°’ ê³„ì‚°
threshold = glm_quality['quality'].quantile(0.99)

# thresholdë¥¼ ì´ˆê³¼í•˜ëŠ” ê°’ë“¤ì�„ NaNìœ¼ë¡œ ì„¤ì •
glm_quality['quality'] = glm_quality['quality'].where(glm_quality['quality'] <= threshold, np.nan)

# ê²°ê³¼ í™•ì�¸
glm_quality['quality'].isnull().sum()  # NaN ê°’ì�´ ëª‡ ê°œ ì�ˆëŠ”ì§€ í™•ì�¸


print(glm_quality[glm_quality['quality'] > 5])
print(glm_quality['quality'].isnull().sum())  # NaNì�˜ ê°œìˆ˜ í™•ì�¸
print(glm_quality['quality'].isnull().mean())  # NaNì�˜ ë¹„ìœ¨ í™•ì�¸


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


import pandas as pd

# âœ… íŠ¹ì • ì‹œë“œ(1, 2, 3) vs íŠ¹ì • ì‹œë“œ(16, 15, 14) í•„í„°ë§�
filtered_df = tourney_data[
    (tourney_data["T1_seed"].isin([2])) & 
    (tourney_data["T2_seed"].isin([15]))
]

# âœ… ì‹¤ì œ ìŠ¹íŒ¨ ì—¬ë¶€ ê³„ì‚° (T1ì�´ ì�´ê²¼ìœ¼ë©´ 1, ì¡Œìœ¼ë©´ 0)
filtered_df["Win"] = (filtered_df["T1_Score"] > filtered_df["T2_Score"]).astype(int)

# âœ… ì—°ë�„ë³„ & (T1_seed, T2_seed) ë³„ ìŠ¹ë¥  ê³„ì‚°
win_rate_by_matchup = (
    filtered_df.groupby(["Season", "T1_seed", "T2_seed"])["Win"]
    .mean()
    .reset_index()
    .rename(columns={"Win": "WinRate"})
)

# âœ… ê²°ê³¼ ì¶œë ¥
win_rate_by_matchup



MMassey = pd.read_csv(DATA_PATH + "MMasseyOrdinals.csv")
MMassey.tail(5)


# ì‹œì¦Œë³„ë¡œ RankingDayNum == 128ì�´ ì�ˆëŠ”ì§€ í™•ì�¸
season_has_128 = MMassey.groupby("Season")["RankingDayNum"].apply(lambda x: 128 in x.values)

# ê²°ê³¼ ì¶œë ¥
print(season_has_128)



# ì‹œì¦Œë³„ë¡œ 128ì�´ ëª‡ ë²ˆ ë“±ì�¥í•˜ëŠ”ì§€ í™•ì�¸
season_128_counts = MMassey[MMassey["RankingDayNum"] == 128].groupby("Season").size()

# ê²°ê³¼ ì¶œë ¥
print(season_128_counts)
season_128_counts = MMassey[MMassey["RankingDayNum"] == 128].groupby("Season").size()


# TeamIDë³„, Seasonë³„ë¡œ DayNumì�´ 128ì�¸ ë�°ì�´í„° ê°œìˆ˜ í™•ì�¸
team_season_daynum_counts = MMassey[MMassey["RankingDayNum"] == 128].groupby(["TeamID", "Season"]).size()

# ê²°ê³¼ ì¶œë ¥
print(team_season_daynum_counts)



team_id = 1101  # ì›�í•˜ëŠ” TeamIDë¡œ ë³€ê²½
season = 2024   # ì›�í•˜ëŠ” ì‹œì¦Œ

# íŠ¹ì • íŒ€, íŠ¹ì • ì‹œì¦Œì—�ì„œ DayNum == 128ì�¸ ë�°ì�´í„° í•„í„°ë§�
team_128_data = MMassey[(MMassey["TeamID"] == team_id) & 
                         (MMassey["Season"] == season) & 
                         (MMassey["RankingDayNum"] == 128)]

# ê²°ê³¼ í™•ì�¸
print(team_128_data)



# ê°� ì‹œì¦Œ, íŒ€ë³„ë¡œ RankingDayNum == 128ì�˜ í�‰ê·  ë�­í‚¹ì�„ ê³„ì‚°
MMassey_avg = MMassey[MMassey["RankingDayNum"] == 128].groupby(["TeamID", "Season"], as_index=False)["OrdinalRank"].mean()

# ì»¬ëŸ¼ëª… ë³€ê²½ (í�‰ê·  ë�­í‚¹ ê°’ì�´ë�€ ì�˜ë¯¸ë¥¼ ëª…í™•í•˜ê²Œ í•˜ê¸° ìœ„í•´)
MMassey_avg.rename(columns={"OrdinalRank": "AvgOrdinalRank_128"}, inplace=True)

# ê²°ê³¼ í™•ì�¸
print(MMassey_avg.head())
rankings_T1 = MMassey_avg.rename(columns={"TeamID": "T1_TeamID", "AvgOrdinalRank_128": "T1_AvgOrdinalRank_128"})
rankings_T2 = MMassey_avg.rename(columns={"TeamID": "T2_TeamID", "AvgOrdinalRank_128": "T2_AvgOrdinalRank_128"})
tourney_data = pd.merge(tourney_data, rankings_T1, on=["Season", "T1_TeamID"], how="left")

# T2 íŒ€ì�˜ í�‰ê·  ë�­í‚¹ì�„ tourney_dataì—� ë³‘í•©
tourney_data = pd.merge(tourney_data, rankings_T2, on=["Season", "T2_TeamID"], how="left")


tourney_data.shape


# ë�°ì�´í„° ë¡œë“œ
sub = pd.read_csv('../input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')

# ë¨¼ì € T1_TeamID, T2_TeamID, Seasonì�„ ìƒ�ì„±í•´ì•¼ í•¨
sub["Season"] = sub["ID"].apply(lambda x: x[:4]).astype(int)
sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)
sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)

print(len(sub))
# ë‚¨ì�� íŒ€ë“¤ë�¼ë¦¬ ê²½ê¸°ë§Œ í�¬í•¨ (T1_TeamID, T2_TeamIDê°€ 2000 ë¯¸ë§Œ)
sub_men = sub[(sub["T1_TeamID"] < 2000) & (sub["T2_TeamID"] < 2000)].copy()  # ğŸ”¥ .copy() ì¶”ê°€
print(len(sub_men))
print(sub_men.shape)  # í•„í„°ë§� í›„ ë�°ì�´í„° í�¬ê¸° í™•ì�¸

print(sub_men.shape)  # í•„í„°ë§� í›„ ë�°ì�´í„° í�¬ê¸° í™•ì�¸

# ë‚¨ì�� íŒ€ë“¤ë�¼ë¦¬ ê²½ê¸°ë§Œ í�¬í•¨ (T1_TeamID, T2_TeamIDê°€ 2000 ë¯¸ë§Œ)
print(sub_men.shape)  # í•„í„°ë§� í›„ ë�°ì�´í„° í�¬ê¸° í™•ì�¸

sub_men["Season"] = sub_men["ID"].apply(lambda x: x[0:4]).astype(int)
sub_men["T1_TeamID"] = sub_men["ID"].apply(lambda x: x[5:9]).astype(int)
sub_men["T2_TeamID"] = sub_men["ID"].apply(lambda x: x[10:14]).astype(int)
print(len(sub_men))
sub_men = pd.merge(sub_men, season_statistics_T1, on = ['Season', 'T1_TeamID'], how='left')
sub_men = pd.merge(sub_men, season_statistics_T2, on = ['Season', 'T2_TeamID'], how='left')
print(len(sub_men))
sub_men = pd.merge(sub_men, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left') 
sub_men = pd.merge(sub_men, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')
print(len(sub_men))
sub_men = pd.merge(sub_men, seeds_T1, on = ['Season', 'T1_TeamID'], how='left')
sub_men = pd.merge(sub_men, seeds_T2, on = ['Season', 'T2_TeamID'], how='left')
sub_men["Seed_diff"] = sub_men["T1_seed"] - sub_men["T2_seed"]
print(len(sub_men))
sub_men = pd.merge(sub_men, rankings_T1, on=["Season", "T1_TeamID"], how="left")
sub_men = pd.merge(sub_men, rankings_T2, on=["Season", "T2_TeamID"], how="left")

print(len(sub_men))
print(season_statistics_T2.shape) 


print(sub_men.T2_quality.isnull().sum())
sub_men["T1_quality"] = sub_men["T1_quality"].fillna(0.2)
sub_men["T2_quality"] = sub_men["T2_quality"].fillna(0.2)
print(sub_men.T2_quality.isnull().sum())

if PREVIOUS_SEASONS_MEN:
    features_for_calc = ["T1_quality", "T2_quality", "T1_seed"]
    sub_men = write_mean_of_3_seasons(sub, features_for_calc, degree_weight=1.0)
    sub_men = sub_men.copy()
sub.head(3)


y = tourney_data['T1_Score'] - tourney_data['T2_Score']
y.describe()


features = list(season_statistics_T1.columns[2:999]) + \
    list(season_statistics_T2.columns[2:999]) + \
    list(seeds_T1.columns[2:999]) + \
    list(seeds_T2.columns[2:999]) + \
    ["Seed_diff"] + ["T1_quality","T2_quality"] + \
    ["T1_AvgOrdinalRank_128", "T2_AvgOrdinalRank_128"]  # ğŸš€ Massey Ordinals ì¶”ê°€

if USE_DAYS_WINFLAG:
    features = features + list(last14days_stats_T1.columns[2:999]) + \
    list(last14days_stats_T2.columns[2:999])



if USE_YEAR_WINFLAG:
    features = features + list(season_win_stats_T1.columns[2:999]) + \
    list(season_win_stats_T2.columns[2:999])
    
len(features)


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
val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
iteration_counts, val_mae


oof_preds = [] #T1_score-T2_scoreì�˜ ì �ìˆ˜ì°¨ì�´
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


# âœ… ìµœê·¼ 14ì�¼ ê²½ê¸° í†µê³„ ë�°ì�´í„°ë¥¼ T1, T2ì—� ëŒ€í•´ ë³‘í•©
sub_men = pd.merge(sub_men, last14days_stats_T1, on=["Season", "T1_TeamID"], how="left")
sub_men = pd.merge(sub_men, last14days_stats_T2, on=["Season", "T2_TeamID"], how="left")

Xsub = sub_men[features].values
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


# ì˜ˆì¸¡í•œ ì �ìˆ˜ ì°¨ì�´ë¥¼ ScoreDiff ì»¬ëŸ¼ì—� ì¶”ê°€
sub_men["ScoreDiff"] = np.mean([sub_models[i].predict(dtest) for i in range(repeat_cv)], axis=0)



sub_preds = []
for i in range(repeat_cv):
    sub_preds.append(np.clip(spline_model[i](np.clip(sub_models[i].predict(dtest),-30,30)),0.0,1.0))
#  ì �ìˆ˜ ì°¨ì�´ì—� ë”°ë¥¸ í™•ë¥  ë³´ì •
sub_men.loc[sub_men["ScoreDiff"] >= 23, "Pred"] = 1.0
sub_men.loc[sub_men["ScoreDiff"] <= -23, "Pred"] = 0.0
sub_men.loc[(sub_men.T1_seed == 1) & (sub_men.T2_seed == 16), "Pred"] = 1.0
sub_men.loc[(sub_men.T1_seed == 16) & (sub_men.T2_seed == 1), "Pred"] = 0.0
sub_men["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)

sub_men[['ID','Pred']].to_csv("submission_men.csv", index = None)


# ì˜ˆì¸¡ í™•ë¥  ë¶„í�¬ ì‹œê°�í™”
plt.figure(figsize=(8, 5))
plt.hist(sub_men["Pred"], bins=30, edgecolor="black", alpha=0.7)
plt.axvline(sub_men["Pred"].mean(), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {sub_men['Pred'].mean():.3f}")
plt.xlabel("Predicted Win Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Win Probabilities (Final Submission)")
plt.legend()
plt.grid(True)
plt.show()



import os

DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025/"
tourney_results = pd.concat([
    pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv")
], ignore_index=True)

seeds = pd.concat([
    pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv")
], ignore_index=True)
#  2025 ë�°ì�´í„°ê°€ ì�ˆëŠ”ì§€ í™•ì�¸
if 2025 not in seeds["Season"].unique():
    print(" 2025 ì‹œì¦Œ ë�°ì�´í„°ê°€ ì—†ìœ¼ë¯€ë¡œ 2024 ë�°ì�´í„°ë¥¼ ë³µì‚¬í•˜ì—¬ ìƒ�ì„±í•©ë‹ˆë‹¤.")
    
    # 2024 ì‹œì¦Œ ë�°ì�´í„° ë³µì‚¬
    seeds_2025 = seeds[seeds["Season"] == 2024].copy()
    seeds_2025["Season"] = 2025  # ì‹œì¦Œì�„ 2025ë¡œ ë³€ê²½
    
    # ì›�ë³¸ ë�°ì�´í„°ì™€ í•©ì¹˜ê¸°
    seeds = pd.concat([seeds, seeds_2025], ignore_index=True)
    
    print(" 2025 ì‹œì¦Œ ë�°ì�´í„°ê°€ ì„±ê³µì �ìœ¼ë¡œ ì¶”ê°€ë�˜ì—ˆìŠµë‹ˆë‹¤.")
else:
    print(" 2025 ì‹œì¦Œ ë�°ì�´í„°ê°€ ì�´ë¯¸ ì¡´ì�¬í•©ë‹ˆë‹¤.")
    
#  ìµœì¢… ë�°ì�´í„° í™•ì�¸
print(seeds.tail(10))  # ë§ˆì§€ë§‰ 10ê°œ ë�°ì�´í„° í™•ì�¸
regular_results = pd.concat([
    pd.read_csv(DATA_PATH + "WRegularSeasonDetailedResults.csv")
], ignore_index=True)


def prepare_data(df_data, use_additional_column=False):
    df = df_data.copy()
    df.rename(columns={'WLoc': 'location'}, inplace=True)
    
    # ìŠ¤ì™‘ìš© ë�°ì�´í„°í”„ë ˆì�„ ìƒ�ì„± (Lâ†’T1, Wâ†’T2 ë³€í™˜)
    dfswap = df[[
        'Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'location', 'NumOT', 
        'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
        'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'
    ]]

    #  ì¶”ê°€ ì»¬ëŸ¼ ë¦¬ìŠ¤íŠ¸
    additional_cols = ['WEFFG', 'WEFFG3', 'WDARE', 'WTOQUETOQUE',
                       'LEFFG', 'LEFFG3', 'LDARE', 'LTOQUETOQUE']
    
    #  ì¶”ê°€ ì¹¼ëŸ¼ì�„ ë°˜ì˜�í• ì§€ ì—¬ë¶€ í™•ì�¸
    if use_additional_column:
        df = df.assign(**{col: df[col.split('_')[1]] / df[col.split('_')[1] + 'A'] for col in additional_cols})
        dfswap = dfswap.assign(**{col: dfswap[col.split('_')[1]] / dfswap[col.split('_')[1] + 'A'] for col in additional_cols})

    #  ì»¬ëŸ¼ëª… ë³€í™˜: ìŠ¹ë¦¬íŒ€(W) â†’ T1, íŒ¨ë°°íŒ€(L) â†’ T2
    df.columns = df.columns.str.replace('W', 'T1_')
    df.columns = df.columns.str.replace('L', 'T2_')

    #  dfswap(íŒ¨ë°°íŒ€ì�„ T1ìœ¼ë¡œ, ìŠ¹ë¦¬íŒ€ì�„ T2ë¡œ ë³€ê²½)
    dfswap.columns = dfswap.columns.str.replace('L', 'T1_')
    dfswap.columns = dfswap.columns.str.replace('W', 'T2_')

    #  í•œ ê²½ê¸°(W/L)ë¥¼ ë‘� ê°œì�˜ T1/T2 í˜•íƒœë¡œ ë³€í™˜
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    
    #  ê²½ê¸° ì�¥ì†Œ ë³€í™˜: N â†’ 0, H â†’ 1, A â†’ -1
    output.loc[output.location == 'N', 'location'] = '0'
    output.loc[output.location == 'H', 'location'] = '1'
    output.loc[output.location == 'A', 'location'] = '-1'
    output['location'] = output['location'].astype(int)

    #  ì �ìˆ˜ ì°¨ì�´ ê³„ì‚° (íƒ€ê²Ÿ ë³€ìˆ˜ë¡œ í™œìš© ê°€ëŠ¥)
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']

    #  ì¶”ê°€ íŒŒìƒ� ë³€ìˆ˜ ìƒ�ì„± (use_additional_column í™œì„±í™” ì‹œ)
    if use_additional_column:
        output['T1_EFFG'] = output['T1_FGM'] / output['T1_FGA']
        output['T1_EFFG3'] = output['T1_FGM3'] / output['T1_FGA3']
        output['T1_DARE'] = output['T1_FGM3'] / output['T1_FGM']
        output['T1_TOQUETOQUE'] = output['T1_Ast'] / output['T1_FGM']
        
        output['T2_EFFG'] = output['T2_FGM'] / output['T2_FGA']
        output['T2_EFFG3'] = output['T2_FGM3'] / output['T2_FGA3']
        output['T2_DARE'] = output['T2_FGM3'] / output['T2_FGM']
        output['T2_TOQUETOQUE'] = output['T2_Ast'] / output['T2_FGM']

        # ë¶„ëª¨ 0ì�¼ ê²½ìš° NaN ë°œìƒ� â†’ 0ìœ¼ë¡œ ëŒ€ì²´
        cols_to_fill = ['T1_EFFG','T1_EFFG3','T1_DARE','T1_TOQUETOQUE',
                        'T2_EFFG','T2_EFFG3','T2_DARE','T2_TOQUETOQUE']
        output[cols_to_fill] = output[cols_to_fill].fillna(0.0)

    return output



regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)


season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg([np.mean]).reset_index()
season_statistics.columns = [f"{col[0]}_mean" if col[1] == "mean" else col[0] for col in season_statistics.columns]
season_statistics.head(3)


#Make two copies of the data
if PREVIOUS_SEASONS_WOMEN:
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

# âœ… T2 íŒ€ì�˜ ì •ê·œì‹œì¦Œ ì „ì²´ ìŠ¹ë¥  ê³„ì‚°
season_win_stats_T2 = regular_data.copy()
season_win_stats_T2['win'] = np.where(season_win_stats_T2['PointDiff'] < 0, 1, 0)
season_win_stats_T2 = season_win_stats_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_season')



tourney_data = pd.merge(tourney_data, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')
if USE_YEAR_WINFLAG:
    tourney_data = pd.merge(tourney_data, season_win_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, season_win_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')



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

# ìƒ�ìœ„ 1% ê°’ ê³„ì‚°
threshold = glm_quality['quality'].quantile(0.99)

# thresholdë¥¼ ì´ˆê³¼í•˜ëŠ” ê°’ë“¤ì�„ NaNìœ¼ë¡œ ì„¤ì •
glm_quality['quality'] = glm_quality['quality'].where(glm_quality['quality'] <= threshold, np.nan)

# ê²°ê³¼ í™•ì�¸
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


# ë�°ì�´í„° ë¡œë“œ
sub = pd.read_csv('../input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')

# ë¨¼ì € T1_TeamID, T2_TeamID, Seasonì�„ ìƒ�ì„±í•´ì•¼ í•¨
sub["Season"] = sub["ID"].apply(lambda x: x[:4]).astype(int)
sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)
sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)

# ë‚¨ì�� íŒ€ë“¤ë�¼ë¦¬ ê²½ê¸°ë§Œ í�¬í•¨ (T1_TeamID, T2_TeamIDê°€ 2000 ë¯¸ë§Œ)
sub_women = sub[(sub["T1_TeamID"] >= 2000) & (sub["T2_TeamID"] >= 2000)].copy()  # ğŸ”¥ .copy() ì¶”ê°€

print(sub_women.shape)  # í•„í„°ë§� í›„ ë�°ì�´í„° í�¬ê¸° í™•ì�¸

print(sub_women.shape)  # í•„í„°ë§� í›„ ë�°ì�´í„° í�¬ê¸° í™•ì�¸

# ë‚¨ì�� íŒ€ë“¤ë�¼ë¦¬ ê²½ê¸°ë§Œ í�¬í•¨ (T1_TeamID, T2_TeamIDê°€ 2000 ë¯¸ë§Œ)
print(sub_women.shape)  # í•„í„°ë§� í›„ ë�°ì�´í„° í�¬ê¸° í™•ì�¸

sub_women["Season"] = sub_women["ID"].apply(lambda x: x[0:4]).astype(int)
sub_women["T1_TeamID"] = sub_women["ID"].apply(lambda x: x[5:9]).astype(int)
sub_women["T2_TeamID"] = sub_women["ID"].apply(lambda x: x[10:14]).astype(int)
sub_women = pd.merge(sub_women, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub_women = pd.merge(sub_women, season_statistics_T2, on = ['Season', 'T2_TeamID'], how='left')
print(sub_women.shape)
sub_women = pd.merge(sub_women, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left') 
sub_women = pd.merge(sub_women, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')
print(sub_women.shape)
sub_women = pd.merge(sub_women, seeds_T1, on = ['Season', 'T1_TeamID'], how='left')
sub_women = pd.merge(sub_women, seeds_T2, on = ['Season', 'T2_TeamID'], how='left')
print(sub_women.shape)
sub_women["Seed_diff"] = sub_women["T1_seed"] - sub_men["T2_seed"]
print(sub_women.shape)
sub_women = pd.merge(sub_women, rankings_T1, on=["Season", "T1_TeamID"], how="left")
sub_women = pd.merge(sub_women, rankings_T2, on=["Season", "T2_TeamID"], how="left")




print(sub_women.T2_quality.isnull().sum())
sub_women["T1_quality"] = sub_women["T1_quality"].fillna(0.2)
sub_women["T2_quality"] = sub_women["T2_quality"].fillna(0.2)
print(sub_women.T2_quality.isnull().sum())

if PREVIOUS_SEASONS_WOMEN:
    features_for_calc = ["T1_quality", "T2_quality", "T1_seed"]
    sub_women = write_mean_of_3_seasons(sub, features_for_calc, degree_weight=1.0)
    sub_women = sub_women.copy()
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


X = tourney_data[features].values
dtrain = xgb.DMatrix(X, label = y)


 tourney_data[features]


import pandas as pd

# âœ… íŠ¹ì • ì‹œë“œ(1, 2, 3) vs íŠ¹ì • ì‹œë“œ(16, 15, 14) í•„í„°ë§�
filtered_df = tourney_data[
    (tourney_data["T1_seed"].isin([4])) & 
    (tourney_data["T2_seed"].isin([13]))
]

# âœ… ì‹¤ì œ ìŠ¹íŒ¨ ì—¬ë¶€ ê³„ì‚° (T1ì�´ ì�´ê²¼ìœ¼ë©´ 1, ì¡Œìœ¼ë©´ 0)
filtered_df["Win"] = (filtered_df["T1_Score"] > filtered_df["T2_Score"]).astype(int)

# âœ… ì—°ë�„ë³„ & (T1_seed, T2_seed) ë³„ ìŠ¹ë¥  ê³„ì‚°
win_rate_by_matchup = (
    filtered_df.groupby(["Season", "T1_seed", "T2_seed"])["Win"]
    .mean()
    .reset_index()
    .rename(columns={"Win": "WinRate"})
)

# âœ… ê²°ê³¼ ì¶œë ¥
win_rate_by_matchup



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
val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
iteration_counts, val_mae


oof_preds = [] #T1_score-T2_scoreì�˜ ì �ìˆ˜ì°¨ì�´
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


# âœ… ìµœê·¼ 14ì�¼ ê²½ê¸° í†µê³„ ë�°ì�´í„°ë¥¼ T1, T2ì—� ëŒ€í•´ ë³‘í•©
sub_women = pd.merge(sub_women, last14days_stats_T1, on=["Season", "T1_TeamID"], how="left")
sub_women = pd.merge(sub_women, last14days_stats_T2, on=["Season", "T2_TeamID"], how="left")

Xsub = sub_women[features].values
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


sub_women["ScoreDiff"] = np.mean([sub_models[i].predict(dtest) for i in range(repeat_cv)], axis=0)



sub_preds = []
for i in range(repeat_cv):
    sub_preds.append(np.clip(spline_model[i](np.clip(sub_models[i].predict(dtest),-30,30)),0.025,0.975))
    
sub_women["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)
sub_women.loc[sub_women["ScoreDiff"] >= 23, "Pred"] = 1.0
sub_women.loc[sub_women["ScoreDiff"] <= -23, "Pred"] = 0.0

# âœ… ì‹œë“œ ê¸°ë°˜ ë³´ì •
sub_women.loc[(sub_women.T1_seed == 1) & (sub_women.T2_seed == 16), "Pred"] = 1.0
sub_women.loc[(sub_women.T1_seed == 16) & (sub_women.T2_seed == 1), "Pred"] = 0.0

sub_women.loc[(sub_women.T1_seed == 2) & (sub_women.T2_seed == 15), "Pred"] = 1.0
sub_women.loc[(sub_women.T1_seed == 15) & (sub_women.T2_seed == 2), "Pred"] = 0.0

sub_women.loc[(sub_women.T1_seed == 3) & (sub_women.T2_seed == 14), "Pred"] = 1.0
sub_women.loc[(sub_women.T1_seed == 14) & (sub_women.T2_seed == 3), "Pred"] = 0.0
sub_women[['ID','Pred']].to_csv("submission_women.csv", index = None)


# ì˜ˆì¸¡ í™•ë¥  ë¶„í�¬ ì‹œê°�í™”
plt.figure(figsize=(8, 5))
plt.hist(sub_women["Pred"], bins=30, edgecolor="black", alpha=0.7)
plt.axvline(sub_women["Pred"].mean(), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {sub_men['Pred'].mean():.3f}")
plt.xlabel("Predicted Win Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Win Probabilities (Final Submission)")
plt.legend()
plt.grid(True)
plt.show()



import pandas as pd

# CSV íŒŒì�¼ ë¶ˆëŸ¬ì˜¤ê¸°
sub_men = pd.read_csv("submission_men.csv")
sub_women = pd.read_csv("submission_women.csv")

# ë�°ì�´í„° í•©ì¹˜ê¸° (ë°‘ìœ¼ë¡œ ì¶”ê°€)
sub_final = pd.concat([sub_men, sub_women], ignore_index=True)

# ìµœì¢… CSV ì €ì�¥
sub_final.to_csv("submission_final_real.csv", index=False)



print(len(sub_final)) 
print(len(sub_men))
print(len(sub_women))


