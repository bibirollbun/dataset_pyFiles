import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
import random
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import ParameterSampler


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',999)

data_dir = "../input/march-machine-learning-mania-2025"

M_regular_results = pd.read_csv(f'{data_dir}/WRegularSeasonDetailedResults.csv')
M_tourney_results = pd.read_csv(f'{data_dir}/WNCAATourneyDetailedResults.csv')
M_seeds = pd.read_csv(f'{data_dir}/WNCAATourneySeeds.csv')

season = 2010
regular_results = M_regular_results.loc[M_regular_results['Season'] >= season]
tourney_results = M_tourney_results.loc[M_tourney_results['Season'] >= season]
seeds = M_seeds.loc[M_seeds['Season'] >= season]

def prepare(df):
    # 根据加时赛数量对数据做调整
    adjot = (40 + 5 * df['NumOT']) / 40
    adjcols = ["LScore", "WScore", 
               "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
               "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]
    for col in adjcols:
        df[col] = df[col] / adjot

    dfswap = df.copy()
    # 交换主客场信息
    dfswap['WLoc'] = dfswap['WLoc'].replace({'H': 'A', 'A': 'H'})
    # 修改表头
    df.rename(columns={'WLoc':'loc'},inplace=True)
    dfswap.rename(columns={'WLoc':'loc'},inplace=True)
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]
    # 合并数据
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    output['win'] = (output['PointDiff'] > 0) * 1

    return output

regular_data = prepare(regular_results)
tourney_data = prepare(tourney_results)

seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds_T1 = seeds[['Season', 'TeamID', 'seed']].copy()
seeds_T2 = seeds[['Season', 'TeamID', 'seed']].copy()
seeds_T1.columns = ['Season', 'T1_TeamID', 'T1_seed']
seeds_T2.columns = ['Season', 'T2_TeamID', 'T2_seed']

tourney_data = tourney_data[['Season', 'T1_TeamID', 'T2_TeamID', 'PointDiff','win']]
tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')
tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]

T1_adjust_cols = ["T1_Score", "T1_FGM", "T1_FTM", "T1_FGM3", "T1_OR", "T1_DR", "T1_Ast", "T1_Stl", "T1_Blk"]
T2_adjust_cols = ["T2_Score", "T2_FGM", "T2_FTM", "T2_FGM3", "T2_OR", "T2_DR", "T2_Ast", "T2_Stl", "T2_Blk"]


import statsmodels.api as sm
import tqdm

regular_data["ST1"] = regular_data.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T1_TeamID"])), axis=1)
regular_data["ST2"] = regular_data.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T2_TeamID"])), axis=1)
seeds_T1["ST1"] = seeds_T1.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T1_TeamID"])), axis=1)
seeds_T2["ST2"] = seeds_T2.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T2_TeamID"])), axis=1)

# collect tourney teams
st = set(seeds_T1["ST1"]) | set(seeds_T2["ST2"])
# append non-tourney teams which were able to beat tourney team at least once
st = st | set(regular_data.loc[(regular_data["T1_Score"] > regular_data["T2_Score"]) & 
                               (regular_data["ST2"].isin(st)), "ST1"])

def team_quality(season):
    # mixed effects: fixed intercept=0, random slope
    formula = "PointDiff~-1+T1_TeamID+T2_TeamID"
    glm = sm.GLM.from_formula(
        formula=formula,
        data=dt.loc[(dt["Season"] == season),:],
        family=sm.families.Gaussian(),
    ).fit()
    
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ["TeamID", "quality"]
    quality["quality"] = quality["quality"]
    quality["Season"] = season
    quality = quality.loc[quality.TeamID.str.contains("T1_")].reset_index(drop=True)
    quality["TeamID"] = quality["TeamID"].apply(lambda x: x[10:14]).astype(int)
    return quality


glm_quality = []

dt = regular_data.loc[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)]
dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"
seasons = sorted(set(seeds["Season"]))
for s in tqdm.tqdm(seasons, unit="season"):
        glm_quality.append(team_quality(s))

glm_quality = pd.concat(glm_quality).reset_index(drop=True)

glm_quality_T1 = glm_quality.copy()
glm_quality_T2 = glm_quality.copy()
glm_quality_T1.columns = ["T1_TeamID", "T1_quality", "Season"]
glm_quality_T2.columns = ["T2_TeamID", "T2_quality", "Season"]
tourney_data = pd.merge(tourney_data, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["diff_quality"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]


# 使用主客场信息来调整数据
regular_data.loc[regular_data['loc'] == 'H', T1_adjust_cols] *= 1
regular_data.loc[regular_data['loc'] == 'A', T2_adjust_cols] *= 1

# 创建新的特征
regular_data['T1_FG'] = regular_data['T1_FGM'] / regular_data['T1_FGA']
regular_data['T1_FG3'] = regular_data['T1_FGM3'] / regular_data['T1_FGA3']
regular_data['T2_FG'] = regular_data['T2_FGM'] / regular_data['T2_FGA']
regular_data['T2_FG3'] = regular_data['T2_FGM3'] / regular_data['T2_FGA3']
regular_data['T1_OTH'] = regular_data['T1_Stl'] + regular_data['T1_Blk'] + regular_data['T1_FGM'] - regular_data['T1_FGA'] + regular_data['T1_FGM3'] - regular_data['T1_FGA3'] + regular_data['T1_OR'] + regular_data['T1_DR'] - regular_data['T1_TO'] + regular_data['T1_FTM'] - regular_data['T1_FTA']
regular_data['T2_OTH'] = regular_data['T2_Stl'] + regular_data['T2_Blk'] + regular_data['T2_FGM'] - regular_data['T2_FGA'] + regular_data['T2_FGM3'] - regular_data['T2_FGA3'] + regular_data['T2_OR'] + regular_data['T2_DR'] - regular_data['T2_TO'] + regular_data['T2_FTM'] - regular_data['T2_FTA']
regular_data['T1_PER'] = regular_data['T1_Score'] + regular_data['T1_Ast'] + regular_data['T1_Stl'] + regular_data['T1_Blk'] + regular_data['T1_FGM'] - regular_data['T1_FGA'] + regular_data['T1_FGM3'] - regular_data['T1_FGA3'] + regular_data['T1_OR'] + regular_data['T1_DR'] - regular_data['T1_TO'] + regular_data['T1_FTM'] - regular_data['T1_FTA']
regular_data['T2_PER'] = regular_data['T2_Score'] + regular_data['T2_Ast'] + regular_data['T2_Stl'] + regular_data['T2_Blk'] + regular_data['T2_FGM'] - regular_data['T2_FGA'] + regular_data['T2_FGM3'] - regular_data['T2_FGA3'] + regular_data['T2_OR'] + regular_data['T2_DR'] - regular_data['T2_TO'] + regular_data['T2_FTM'] - regular_data['T2_FTA']

boxcols = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF", "T1_FG","T1_FG3",
    "T1_OTH", "T1_PER",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF", "T2_FG","T2_FG3",
    "T2_OTH", "T2_PER",
    "PointDiff",
]

ss = regular_data.groupby(["Season", "T1_TeamID"])[boxcols].agg("mean").reset_index()

ss_T1 = ss.copy()
ss_T1.columns = ["T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(ss_T1.columns)]
ss_T1 = ss_T1.rename({"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}, axis=1)
ss_T2 = ss.copy()
ss_T2.columns = ["T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(ss_T2.columns)]
ss_T2 = ss_T2.rename({"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}, axis=1)

tourney_data = pd.merge(tourney_data, ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, ss_T2, on=["Season", "T2_TeamID"], how="left")


# 新的特征：elo评分
def update_elo(winner_elo,loser_elo):
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1-expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def expected_result(elo_a, elo_b):
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))

base_elo = 1500
elo_width = 400
k_factor = 64 # 可调

elos = []
for season in sorted(set(seeds["Season"])):
    ss = regular_data.loc[regular_data["Season"] == season]
    ss = ss.loc[ss["win"] == 1].reset_index(drop=True)
    teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
    elo = dict(zip(teams, [base_elo] * len(teams)))
    for i in range(ss.shape[0]):
        w_team, l_team = ss.loc[i, "T1_TeamID"], ss.loc[i, "T2_TeamID"]
        w_elo, l_elo = elo[w_team], elo[l_team]
        w_elo_new, l_elo_new = update_elo(w_elo, l_elo)
        elo[w_team] = w_elo_new
        elo[l_team] = l_elo_new
    elo = pd.DataFrame.from_dict(elo, orient="index").reset_index()
    elo = elo.rename({"index": "TeamID", 0: "elo"}, axis=1)
    elo["Season"] = season
    elos.append(elo)
elos = pd.concat(elos)

elos_T1 = elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
elos_T2 = elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)
tourney_data = pd.merge(tourney_data, elos_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, elos_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]


features1 = [
    "Seed_diff",
    "T1_quality",
    "T2_quality",
    "diff_quality",
    "T1_elo",
    "T2_elo",
    "elo_diff",
    "T1_avg_PointDiff",
    "T2_avg_PointDiff",
    "T1_avg_Score",
    "T1_avg_FG",
    "T1_avg_FGM",
    "T1_avg_Ast",
    "T1_avg_Blk",
    "T1_avg_OTH",
    "T1_avg_opponent_Score",
    "T1_avg_opponent_FG",
    "T1_avg_opponent_FGM",
    "T1_avg_opponent_Ast",
    "T1_avg_opponent_Blk",
    "T1_avg_opponent_OTH",
    "T2_avg_Score",
    "T2_avg_FG",
    "T2_avg_FGM",
    "T2_avg_Ast",
    "T2_avg_Blk",
    "T2_avg_OTH",
    "T2_avg_opponent_Score",
    "T2_avg_opponent_FG",
    "T2_avg_opponent_FGM",
    "T2_avg_opponent_Ast",
    "T2_avg_opponent_Blk",
    "T2_avg_opponent_OTH",
]


# from tqdm import tqdm
# # 参数分布
# param_dist = {
#     "eta": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
#     "max_depth": [3, 4, 5, 6, 7],
#     "subsample": [0.5, 0.6, 0.7],
#     "colsample_bynode": [0.7, 0.8, 0.9],
#     "min_child_weight": [3, 4, 5, 6 ,7],
#     "num_parallel_tree": [1, 2, 3, 4, 5],
#     "num_rounds": [500, 700,  900],  # 加入num_rounds参数
#     "max_bin": [16, 32, 64, 128],  # 加入max_bin参数
#     "lambda": [0.1, 0.5, 1.0, 3.0, 5.0],  # 加入lambda参数
#     "alpha": [0.1, 0.5, 1.0, 3.0, 5.0]  # 加入alpha参数
# }

# # 随机采样次数
# n_iter = 500

# # 初始化 MAE 列表
# mae_lst = []

# # 随机调参
# for param in tqdm(ParameterSampler(param_dist, n_iter=n_iter, random_state=42), total=n_iter):
#     models = {}
#     oof_mae = []
    
#     # 留一法交叉验证
#     for oof_season in set(tourney_data.Season):
#         x_train = tourney_data.loc[tourney_data["Season"] != oof_season, features1].values
#         y_train = tourney_data.loc[tourney_data["Season"] != oof_season, "PointDiff"].values
#         x_val = tourney_data.loc[tourney_data["Season"] == oof_season, features1].values
#         y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values
        
#         dtrain = xgb.DMatrix(x_train, label=y_train)
#         dval = xgb.DMatrix(x_val, label=y_val)
        
#         # 训练模型，加入早停机制
#         model = xgb.train(
#             params=param,
#             dtrain=dtrain,
#             num_boost_round=param['num_rounds'],  # 使用当前参数中的num_rounds
#             evals=[(dval, 'eval')],  # 加入验证集
#             early_stopping_rounds=50,  # 早停轮数
#             maximize=False,  # 是否最大化评估指标
#             verbose_eval=False
#         )
        
#         # 预测
#         preds = model.predict(dval)
        
#         # 计算 MAE
#         mae = mean_absolute_error(y_val, preds)
#         oof_mae.append(mae)
    
#     # 计算平均 MAE
#     weights = np.arange(len(oof_mae)) + 1
#     avg_mae = np.sum(oof_mae * weights) / np.sum(weights)
#     mae_lst.append(avg_mae)

# # 找到最佳参数
# best_param_idx = np.argmin(mae_lst)
# best_param = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))[best_param_idx]
# print(f"Best Parameters: {best_param}, Best Average MAE: {mae_lst[best_param_idx]}")

