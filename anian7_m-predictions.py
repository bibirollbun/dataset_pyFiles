import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
import random
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import tqdm
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import ParameterSampler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',999)

data_dir = "../input/march-machine-learning-mania-2025"

M_regular_results = pd.read_csv(f'{data_dir}/MRegularSeasonDetailedResults.csv')
M_tourney_results = pd.read_csv(f'{data_dir}/MNCAATourneyDetailedResults.csv')
M_seeds = pd.read_csv(f'{data_dir}/MNCAATourneySeeds.csv')
suggestions = pd.read_csv(f'{data_dir}/MMasseyOrdinals.csv')

season = 2003
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


from tqdm import tqdm
# 确保数据类型正确
suggestions['Season'] = suggestions['Season'].astype(int)
suggestions['RankingDayNum'] = suggestions['RankingDayNum'].astype(int)
suggestions['TeamID'] = suggestions['TeamID'].astype(int)
suggestions['OrdinalRank'] = suggestions['OrdinalRank'].astype(int)

# 1. 计算每支队伍在每家机构的排名随时间变化的斜率
slope_data = []
for (season, team_id, system_name), group in tqdm(suggestions.groupby(['Season', 'TeamID', 'SystemName']), desc="Calculating slopes"):
    X = group['RankingDayNum'].values.reshape(-1, 1)
    y = group['OrdinalRank'].values
    if len(X) > 1:
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
    else:
        slope = 0  # 如果只有一个时间点，斜率为0
    slope_data.append({
        'Season': season,
        'TeamID': team_id,
        'SystemName': system_name,
        'Slope': slope
    })

slope_df = pd.DataFrame(slope_data)

# 2. 计算每支队伍的平均斜率（动量指标）
momentum_data = []
for (season, team_id), group in tqdm(slope_df.groupby(['Season', 'TeamID']), desc="Calculating momentum"):
    avg_slope = group['Slope'].mean()
    momentum_data.append({
        'Season': season,
        'TeamID': team_id,
        'Momentum': avg_slope
    })

momentum_df = pd.DataFrame(momentum_data)

# 3. 计算每支队伍在每家机构最后一个时间点的排名
last_rank_data = []
for (season, team_id, system_name), group in tqdm(suggestions.groupby(['Season', 'TeamID', 'SystemName']), desc="Calculating last ranks"):
    last_rank = group.sort_values('RankingDayNum', ascending=False).iloc[0]['OrdinalRank']
    last_rank_data.append({
        'Season': season,
        'TeamID': team_id,
        'SystemName': system_name,
        'LastRank': last_rank
    })

last_rank_df = pd.DataFrame(last_rank_data)

# 4. 计算每支队伍的平均排名（实力指标）
strength_data = []
for (season, team_id), group in tqdm(last_rank_df.groupby(['Season', 'TeamID']), desc="Calculating strength"):
    avg_rank = group['LastRank'].mean()
    strength_data.append({
        'Season': season,
        'TeamID': team_id,
        'Strength': avg_rank
    })

strength_df = pd.DataFrame(strength_data)

# 5. 合并动量和实力指标
team_metrics = pd.merge(momentum_df, strength_df, on=['Season', 'TeamID'])

# 6. 创建最终表格，包含 Team1_ID 和 Team2_ID
team1_df = team_metrics.rename(columns={'TeamID': 'T1_TeamID'})
team2_df = team_metrics.rename(columns={'TeamID': 'T2_TeamID'})


team1_df = team1_df.rename(columns={'Momentum':'T1_Momentum','Strength':'T1_Strength'})
team2_df = team2_df.rename(columns={'Momentum':'T2_Momentum','Strength':'T2_Strength'})
tourney_data = pd.merge(tourney_data,team1_df,on=['Season','T1_TeamID'],how='left')
tourney_data = pd.merge(tourney_data,team2_df,on=['Season','T2_TeamID'],how='left')


tourney_data2 = tourney_data.copy()


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

tourney_data2 = pd.merge(tourney_data2, ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data2 = pd.merge(tourney_data2, ss_T2, on=["Season", "T2_TeamID"], how="left")


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
k_factor = 16 # 可调

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
tourney_data2 = pd.merge(tourney_data2, elos_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data2 = pd.merge(tourney_data2, elos_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data2["elo_diff"] = tourney_data2["T1_elo"] - tourney_data2["T2_elo"]


features1 = [
    "T1_Momentum",
    "T1_Strength",
    "T2_Momentum",
    "T2_Strength",
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
features2 = [
    "T1_Momentum",
    "T1_Strength",
    "T2_Momentum",
    "T2_Strength",
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
    # "T1_avg_opponent_Score",
    # "T1_avg_opponent_FG",
    # "T1_avg_opponent_FGM",
    # "T1_avg_opponent_Ast",
    # "T1_avg_opponent_Blk",
    # "T1_avg_opponent_OTH",
    "T2_avg_Score",
    "T2_avg_FG",
    "T2_avg_FGM",
    "T2_avg_Ast",
    "T2_avg_Blk",
    "T2_avg_OTH",
    # "T2_avg_opponent_Score",
    # "T2_avg_opponent_FG",
    # "T2_avg_opponent_FGM",
    # "T2_avg_opponent_Ast",
    # "T2_avg_opponent_Blk",
    # "T2_avg_opponent_OTH",
]
features3 = ["T1_Momentum","T1_Strength","T2_Momentum","T2_Strength","Seed_diff","T1_quality","T2_quality","T1_elo","T2_elo","T1_avg_PER","T2_avg_PER","T1_avg_opponent_PER","T2_avg_opponent_PER"]


def get_xgboost_avg_score(features):
    param = {}
    param["objective"] = "reg:squarederror"
    param["booster"] = "gbtree"
    param["eta"] = 0.1
    param["subsample"] = 0.6
    param["colsample_bynode"] = 0.8
    param["num_parallel_tree"] = 2
    param["min_child_weight"] = 4
    param["max_depth"] = 4
    param["tree_method"] = "hist"
    param['grow_policy'] = 'lossguide'
    param["max_bin"] = 32
    num_rounds = 700

    models = {}
    oof_mae = []
    oof_preds = []
    oof_targets = []
    oof_ss = []

    for oof_season in set(tourney_data.Season):
        x_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, features].values
        y_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, "PointDiff"].values
        x_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, features].values
        y_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, "PointDiff"].values
        s_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, "Season"].values
    
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        models[oof_season] = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round = num_rounds,        
        )
        preds = models[oof_season].predict(dval)
        oof_mae.append(mean_absolute_error(y_val, preds))
        oof_preds += list(preds)
        oof_targets += list(y_val)
        oof_ss += list(s_val)
        weights = np.arange(len(oof_mae)) + 1
    print(f"average mae: {np.sum(oof_mae * weights) / np.sum(weights)}")


get_xgboost_avg_score(features1)


get_xgboost_avg_score(features2)


get_xgboost_avg_score(features3)


# 参数分布
param_dist = {
    "eta": [0.01,0.015,0.02,0.025,0.03],
    "max_depth": [3, 4, 5],
    "subsample": [0.65, 0.7, 0.75],
    "colsample_bynode": [0.65, 0.7, 0.75],
    "min_child_weight": [ 5, 6 ,7],
    "num_parallel_tree": [5 ,6, 7],
    "num_rounds": [850, 900 , 950],  # 加入num_rounds参数
    "max_bin": [16, 32, 64,],  # 加入max_bin参数
    "lambda": [1.0, 1.5, 2.0, 2.5],  # 加入lambda参数
    "alpha": [0.1, 0.5, 1.0, 1.5, 2.0]  # 加入alpha参数
}

# 随机采样次数
n_iter = 250

# 初始化 MAE 列表
mae_lst = []

# 随机调参
for param in tqdm(ParameterSampler(param_dist, n_iter=n_iter, random_state=42), total=n_iter):
    models = {}
    oof_mae = []
    
    # 留一法交叉验证
    for oof_season in set(tourney_data.Season):
        x_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, features1].values
        y_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, "PointDiff"].values
        x_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, features1].values
        y_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, "PointDiff"].values
        
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        
        # 训练模型，加入早停机制
        model = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=param['num_rounds'],  # 使用当前参数中的num_rounds
            evals=[(dval, 'eval')],  # 加入验证集
            early_stopping_rounds=50,  # 早停轮数
            maximize=False,  # 是否最大化评估指标
            verbose_eval=False
        )
        
        # 预测
        preds = model.predict(dval)
        
        # 计算 MAE
        mae = mean_absolute_error(y_val, preds)
        oof_mae.append(mae)
    
    # 计算平均 MAE
    weights = np.arange(len(oof_mae)) + 1
    avg_mae = np.sum(oof_mae * weights) / np.sum(weights)
    mae_lst.append(avg_mae)

# 找到最佳参数
best_param_idx = np.argmin(mae_lst)
best_param = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))[best_param_idx]
print(f"Best Parameters: {best_param}, Best Average MAE: {mae_lst[best_param_idx]}")


# 定义参数
param = {}
param["objective"] = "reg:squarederror"
param["booster"] = "gbtree"
param["eta"] = 0.025
param["subsample"] = 0.65
param["colsample_bynode"] = 0.7
param["num_parallel_tree"] = 5
param["min_child_weight"] = 5
param["max_depth"] = 3
param["tree_method"] = "hist"
param['grow_policy'] = 'lossguide'
param["max_bin"] = 32
param["lambda"] = 1.5
param["alpha"] = 0.1
num_rounds = 900

# 初始化模型字典和评估指标列表
models = {}
oof_mae = []
oof_preds = []
oof_targets = []
oof_ss = []

# 留一法交叉验证
for oof_season in set(tourney_data2.Season):
    # 准备训练和验证数据
    x_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, features1].values
    y_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, "PointDiff"].values
    x_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, features1].values
    y_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, "PointDiff"].values
    s_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, "Season"].values
    
    # 创建 DMatrix
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    
    # 训练模型，加入早停机制
    models[oof_season] = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, 'eval')],  # 加入验证集
        early_stopping_rounds=50,  # 早停轮数
        maximize=False,  # 是否最大化评估指标
        verbose_eval=False  # 禁用训练过程中的打印信息
    )
    
    # 预测
    preds = models[oof_season].predict(dval)
    
    # 计算 MAE
    mae = mean_absolute_error(y_val, preds)
    oof_mae.append(mae)
    
    # 保存预测结果和目标值
    oof_preds += list(preds)
    oof_targets += list(y_val)
    oof_ss += list(s_val)

# 计算加权平均 MAE
weights = np.arange(len(oof_mae)) + 1
avg_mae = np.sum(oof_mae * weights) / np.sum(weights)
print(f"average mae: {avg_mae}")


# 创建数据框
df = pd.DataFrame(
    {"Season": oof_ss, "pred": oof_preds, "label": [(t > 0) * 1 for t in oof_targets]}
)
df["pred_pointdiff"] = df["pred"].astype(int)

t = 40

# 获取所有赛季的唯一值，并排序
seasons = sorted(set(tourney_data2.Season))

# 用于存储每个验证赛季的 Brier 值
brier_scores = []

# 留一法交叉验证
for val_season in seasons:
    # 训练集：所有非验证赛季的数据
    train_df = df[df["Season"] != val_season]
    # 验证集：当前验证赛季的数据
    val_df = df[df["Season"] == val_season]
    
    # 准备训练数据
    train_dat = list(zip(train_df["pred"], np.array(train_df["label"])))
    train_dat = sorted(train_dat, key=lambda x: x[0])
    train_pred, train_label = list(zip(*train_dat))
    
    # 平滑样条拟合
    spline_model = UnivariateSpline(np.clip(train_pred, -t, t), train_label, k=5)
    
    # 对验证集进行预测
    val_pred = val_df["pred"].values
    val_label = val_df["label"].values
    val_spline_fit = np.clip(spline_model(np.clip(val_pred, -t, t)), 0, 1)
    
    # 计算 Brier 分数
    brier = brier_score_loss(val_label, val_spline_fit)
    brier_scores.append(brier)
    # print(f"Validation Season {val_season}: Brier Score = {np.round(brier, 5)}")

# 计算平均 Brier 值
weights = np.arange(len(brier_scores)) + 1
avg_brier = np.sum(brier_scores * weights) / np.sum(weights)
print(f"Average Brier Score across all seasons: {np.round(avg_brier, 5)}")


def get_feature_score(k,m):
    tourney_data2 = tourney_data.copy()
    # 使用主客场信息来调整数据
    regular_data.loc[regular_data['loc'] == 'H', T1_adjust_cols] *= k
    regular_data.loc[regular_data['loc'] == 'A', T2_adjust_cols] *= k

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

    tourney_data2 = pd.merge(tourney_data2, ss_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data2 = pd.merge(tourney_data2, ss_T2, on=["Season", "T2_TeamID"], how="left")
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
    k_factor = m # 可调

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
    tourney_data2 = pd.merge(tourney_data2, elos_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data2 = pd.merge(tourney_data2, elos_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data2["elo_diff"] = tourney_data2["T1_elo"] - tourney_data2["T2_elo"]

    features1 = [
    "T1_Momentum",
    "T1_Strength",
    "T2_Momentum",
    "T2_Strength",
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
    # 定义参数
    param = {}
    param["objective"] = "reg:squarederror"
    param["booster"] = "gbtree"
    param["eta"] = 0.025
    param["subsample"] = 0.65
    param["colsample_bynode"] = 0.7
    param["num_parallel_tree"] = 5
    param["min_child_weight"] = 5
    param["max_depth"] = 3
    param["tree_method"] = "hist"
    param['grow_policy'] = 'lossguide'
    param["max_bin"] = 32
    param["lambda"] = 1.5
    param["alpha"] = 0.1
    num_rounds = 900

    # 初始化模型字典和评估指标列表
    models = {}
    oof_mae = []
    oof_preds = []
    oof_targets = []
    oof_ss = []

    # 留一法交叉验证
    for oof_season in set(tourney_data2.Season):
        # 准备训练和验证数据
        x_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, features1].values
        y_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, "PointDiff"].values
        x_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, features1].values
        y_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, "PointDiff"].values
        s_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, "Season"].values
        
        # 创建 DMatrix
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        
        # 训练模型，加入早停机制
        models[oof_season] = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            evals=[(dval, 'eval')],  # 加入验证集
            early_stopping_rounds=50,  # 早停轮数
            maximize=False,  # 是否最大化评估指标
            verbose_eval=False  # 禁用训练过程中的打印信息
        )
        
        # 预测
        preds = models[oof_season].predict(dval)
        
        # 计算 MAE
        mae = mean_absolute_error(y_val, preds)
        oof_mae.append(mae)
        
        # 保存预测结果和目标值
        oof_preds += list(preds)
        oof_targets += list(y_val)
        oof_ss += list(s_val)

    # 计算加权平均 MAE
    weights = np.arange(len(oof_mae)) + 1
    avg_mae = np.sum(oof_mae * weights) / np.sum(weights)

    # 创建数据框
    df = pd.DataFrame(
        {"Season": oof_ss, "pred": oof_preds, "label": [(t > 0) * 1 for t in oof_targets]}
    )
    df["pred_pointdiff"] = df["pred"].astype(int)

    t = 30

    # 获取所有赛季的唯一值，并排序
    seasons = sorted(set(tourney_data2.Season))

    # 用于存储每个验证赛季的 Brier 值
    brier_scores = []

    # 留一法交叉验证
    for val_season in seasons:
        # 训练集：所有非验证赛季的数据
        train_df = df[df["Season"] != val_season]
        # 验证集：当前验证赛季的数据
        val_df = df[df["Season"] == val_season]
        
        # 准备训练数据
        train_dat = list(zip(train_df["pred"], np.array(train_df["label"])))
        train_dat = sorted(train_dat, key=lambda x: x[0])
        train_pred, train_label = list(zip(*train_dat))
        
        # 平滑样条拟合
        spline_model = UnivariateSpline(np.clip(train_pred, -t, t), train_label, k=5)
        
        # 对验证集进行预测
        val_pred = val_df["pred"].values
        val_label = val_df["label"].values
        val_spline_fit = np.clip(spline_model(np.clip(val_pred, -t, t)), 0, 1)
        
        # 计算 Brier 分数
        brier = brier_score_loss(val_label, val_spline_fit)
        brier_scores.append(brier)
        # print(f"Validation Season {val_season}: Brier Score = {np.round(brier, 5)}")

    # 计算平均 Brier 值
    weights = np.arange(len(brier_scores)) + 1
    avg_brier = np.sum(brier_scores * weights) / np.sum(weights)

    print(f'k={k},m={m},mae={avg_mae},brier={avg_brier}')




get_feature_score(0.95,8)


for k in tqdm([0.92, 0.94, 0.96, 0.98], desc="Outer Loop"):
    for m in tqdm([16, 32, 64, 128], desc="Inner Loop", leave=False):
        get_feature_score(k, m)


# 逻辑回归

from sklearn.linear_model import LogisticRegression

# 创建数据框
df = pd.DataFrame(
    {"Season": oof_ss, "pred": oof_preds, "label": [(t > 0) * 1 for t in oof_targets]}
)

# 获取所有赛季的唯一值，并排序
seasons = sorted(set(tourney_data2.Season))

# 用于存储每个验证赛季的 Brier 值
brier_scores = []

# 留一法交叉验证
for val_season in seasons:
    # 训练集：所有非验证赛季的数据
    train_df = df[df["Season"] != val_season]
    # 验证集：当前验证赛季的数据
    val_df = df[df["Season"] == val_season]
    
    # 准备训练数据
    train_pred = train_df["pred"].values.reshape(-1, 1)
    train_label = train_df["label"].values
    
    # 训练逻辑回归模型
    logistic_model = LogisticRegression()
    logistic_model.fit(train_pred, train_label)
    
    # 对验证集进行预测
    val_pred = val_df["pred"].values.reshape(-1, 1)
    val_label = val_df["label"].values
    val_logistic_fit = logistic_model.predict_proba(val_pred)[:, 1]
    
    # 计算 Brier 分数
    brier = brier_score_loss(val_label, val_logistic_fit)
    brier_scores.append(brier)
    # print(f"Validation Season {val_season}: Brier Score = {np.round(brier, 5)}")

# 计算平均 Brier 值
weights = np.arange(len(brier_scores)) + 1
avg_brier = np.sum(brier_scores * weights) / np.sum(weights)
print(f"Average Brier Score across all seasons: {np.round(avg_brier, 5)}")


# 直接归一化

# 创建数据框
df = pd.DataFrame(
    {"Season": oof_ss, "pred": oof_preds, "label": [(t > 0) * 1 for t in oof_targets]}
)

# 获取所有赛季的唯一值，并排序
seasons = sorted(set(tourney_data2.Season))

# 用于存储每个验证赛季的 Brier 值
brier_scores = []

# 留一法交叉验证
for val_season in seasons:
    # 训练集：所有非验证赛季的数据
    train_df = df[df["Season"] != val_season]
    # 验证集：当前验证赛季的数据
    val_df = df[df["Season"] == val_season]
    
    # 准备训练数据
    train_pred = train_df["pred"].values
    train_label = train_df["label"].values
    
    # 归一化预测分差到 [0, 1] 区间
    min_pred = train_pred.min()
    max_pred = train_pred.max()
    normalized_train_pred = (train_pred - min_pred) / (max_pred - min_pred)
    
    # 对验证集进行预测
    val_pred = val_df["pred"].values
    normalized_val_pred = (val_pred - min_pred) / (max_pred - min_pred)
    
    # 裁剪归一化后的值，确保其在 [0, 1] 范围内
    normalized_val_pred = np.clip(normalized_val_pred, 0, 1)
    
    # 计算 Brier 分数
    brier = brier_score_loss(val_df["label"].values, normalized_val_pred)
    brier_scores.append(brier)
    # print(f"Validation Season {val_season}: Brier Score = {np.round(brier, 5)}")

# 计算平均 Brier 值
weights = np.arange(len(brier_scores)) + 1
avg_brier = np.sum(brier_scores * weights) / np.sum(weights)
print(f"Average Brier Score across all seasons: {np.round(avg_brier, 5)}")


# 概率校准

from sklearn.calibration import CalibratedClassifierCV

# 创建数据框
df = pd.DataFrame(
    {"Season": oof_ss, "pred": oof_preds, "label": [(t > 0) * 1 for t in oof_targets]}
)

# 获取所有赛季的唯一值，并排序
seasons = sorted(set(tourney_data2.Season))

# 用于存储每个验证赛季的 Brier 值
brier_scores = []

# 留一法交叉验证
for val_season in seasons:
    # 训练集：所有非验证赛季的数据
    train_df = df[df["Season"] != val_season]
    # 验证集：当前验证赛季的数据
    val_df = df[df["Season"] == val_season]
    
    # 准备训练数据
    train_pred = train_df["pred"].values.reshape(-1, 1)
    train_label = train_df["label"].values
    
    # 训练概率校准模型
    calibrated_model = CalibratedClassifierCV(method='isotonic')
    calibrated_model.fit(train_pred, train_label)
    
    # 对验证集进行预测
    val_pred = val_df["pred"].values.reshape(-1, 1)
    val_label = val_df["label"].values
    val_calibrated_fit = calibrated_model.predict_proba(val_pred)[:, 1]
    
    # 计算 Brier 分数
    brier = brier_score_loss(val_label, val_calibrated_fit)
    brier_scores.append(brier)
    # print(f"Validation Season {val_season}: Brier Score = {np.round(brier, 5)}")

# 计算平均 Brier 值
weights = np.arange(len(brier_scores)) + 1
avg_brier = np.sum(brier_scores * weights) / np.sum(weights)
print(f"Average Brier Score across all seasons: {np.round(avg_brier, 5)}")


# 定义参数
param = {}
param["objective"] = "reg:squarederror"
param["booster"] = "gbtree"
param["eta"] = 0.025
param["subsample"] = 0.65
param["colsample_bynode"] = 0.7
param["num_parallel_tree"] = 5
param["min_child_weight"] = 5
param["max_depth"] = 3
param["tree_method"] = "hist"
param['grow_policy'] = 'lossguide'
param["max_bin"] = 32
param["lambda"] = 1.5
param["alpha"] = 0.1
num_rounds = 900

# 初始化模型字典和评估指标列表
models = {}
oof_mae = []
oof_preds = []
oof_targets = []
oof_ss = []

# 留一法交叉验证
for oof_season in set(tourney_data2.Season):
    # 准备训练和验证数据
    x_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, features1].values
    y_train = tourney_data2.loc[tourney_data2["Season"] != oof_season, "PointDiff"].values
    x_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, features1].values
    y_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, "PointDiff"].values
    s_val = tourney_data2.loc[tourney_data2["Season"] == oof_season, "Season"].values
    
    # 创建 DMatrix
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    
    # 训练模型，加入早停机制
    models[oof_season] = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, 'eval')],  # 加入验证集
        early_stopping_rounds=50,  # 早停轮数
        maximize=False,  # 是否最大化评估指标
        verbose_eval=False  # 禁用训练过程中的打印信息
    )
    
    # 预测
    preds = models[oof_season].predict(dval)
    
    # 计算 MAE
    mae = mean_absolute_error(y_val, preds)
    oof_mae.append(mae)
    
    # 保存预测结果和目标值
    oof_preds += list(preds)
    oof_targets += list(y_val)
    oof_ss += list(s_val)

# 计算加权平均 MAE
weights = np.arange(len(oof_mae)) + 1
avg_mae = np.sum(oof_mae * weights) / np.sum(weights)
print(f"average mae: {avg_mae}")

# 创建数据框
df = pd.DataFrame(
    {"Season": oof_ss, "pred": oof_preds, "label": [(t > 0) * 1 for t in oof_targets]}
)
df["pred_pointdiff"] = df["pred"].astype(int)

t = 30

# 获取所有赛季的唯一值，并排序
seasons = sorted(set(tourney_data2.Season))

# 用于存储每个验证赛季的 Brier 值
brier_scores = []

# 留一法交叉验证
for val_season in seasons:
    # 训练集：所有非验证赛季的数据
    train_df = df[df["Season"] != val_season]
    # 验证集：当前验证赛季的数据
    val_df = df[df["Season"] == val_season]
    
    # 准备训练数据
    train_dat = list(zip(train_df["pred"], np.array(train_df["label"])))
    train_dat = sorted(train_dat, key=lambda x: x[0])
    train_pred, train_label = list(zip(*train_dat))
    
    # 平滑样条拟合
    spline_model = UnivariateSpline(np.clip(train_pred, -t, t), train_label, k=5)
    
    # 对验证集进行预测
    val_pred = val_df["pred"].values
    val_label = val_df["label"].values
    val_spline_fit = np.clip(spline_model(np.clip(val_pred, -t, t)), 0, 1)
    
    # 计算 Brier 分数
    brier = brier_score_loss(val_label, val_spline_fit)
    brier_scores.append(brier)
    # print(f"Validation Season {val_season}: Brier Score = {np.round(brier, 5)}")

# 计算平均 Brier 值
weights = np.arange(len(brier_scores)) + 1
avg_brier = np.sum(brier_scores * weights) / np.sum(weights)
print(f"Average Brier Score across all seasons: {np.round(avg_brier, 5)}")


X = pd.read_csv(f"{data_dir}/SampleSubmissionStage2.csv")
X = X[X['ID'].str[5] == '1']
# construct dataframe for submission
X['Season'] = X['ID'].apply(lambda t: int(t.split('_')[0]))
X['T1_TeamID'] = X['ID'].apply(lambda t: int(t.split('_')[1]))
X['T2_TeamID'] = X['ID'].apply(lambda t: int(t.split('_')[2]))
X = pd.merge(X, ss_T1, on = ['Season', 'T1_TeamID'], how = 'left')
X = pd.merge(X, ss_T2, on = ['Season', 'T2_TeamID'], how = 'left')
X = pd.merge(X, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
X = pd.merge(X, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')
X = pd.merge(X, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, elos_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, elos_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, team1_df, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, team2_df, on=["Season", "T2_TeamID"], how="left")
X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]
X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
X["diff_quality"] = X["T1_quality"] - X["T2_quality"]


# 获取所有唯一的赛季年份并排序
seasons = sorted(set(tourney_data2.Season))

# 计算每个赛季的权重，权重与赛季年份成正比
season_weights = np.arange(1, len(seasons) + 1)
season_weights = season_weights / season_weights.sum()  # 归一化权重

preds = []
weights = []

for i, oof_season in enumerate(seasons):
    dtest = xgb.DMatrix(X[features1].values)
    margin_preds = models[oof_season].predict(dtest) * 1.0
    probs = np.clip(spline_model(np.clip(margin_preds, -t, t)), 0.01, 0.99)
    preds.append(probs)
    weights.append(season_weights[i])

# 将预测结果和权重转换为 NumPy 数组
preds = np.array(preds)
weights = np.array(weights)

# 计算加权平均预测
weighted_preds = np.average(preds, axis=0, weights=weights)

# 将加权平均预测结果存储到 X 表中
X['Pred'] = weighted_preds



pd.pivot_table(data = X, index='T1_seed', columns='T2_seed', values='Pred', aggfunc='mean').style.bar(color='#5fba7d', vmin=0, vmax=1)


X = X[['ID','Pred']]


X.to_csv('m_predictions.csv',index=None)




