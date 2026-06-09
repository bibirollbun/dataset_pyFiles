import numpy as np
import pandas as pd
import glob
path = '/kaggle/input/march-machine-learning-mania-2025/**'
#key是只取文件名
#值是读取文件内容
#glob.glob() 函数根据给定的路径模式查找文件，这里 path 表示的是要查找所有 CSV 文件。
data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}


seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])


season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])

season_cresults.drop(['NumOT', 'WLoc'], axis=1, inplace=True)


season_cresults


season_cresults['ScoreGap'] = season_cresults['WScore'] - season_cresults['LScore']


season_cresults.head()


num_win = season_cresults.groupby(['Season', 'WTeamID']).count()
num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"})
num_win


num_loss = season_cresults.groupby(['Season', 'LTeamID']).count()
num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"})


gap_win = season_cresults.groupby(['Season', 'WTeamID']).mean().reset_index()
gap_win = gap_win[['Season', 'WTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"})
gap_win.head()


gap_loss = season_cresults.groupby(['Season', 'LTeamID']).mean().reset_index()
gap_loss = gap_loss[['Season', 'LTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"})


df_features_season_w = season_cresults.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
df_features_season_l = season_cresults.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})


#这是本次数据处理的核心，因为它为后面的数据填充奠基
df_features_season = pd.concat([df_features_season_w, df_features_season_l], axis=0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)
df_features_season.head()


df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_loss, on=['Season', 'TeamID'], how='left')


df_features_season.head()


df_features_season.fillna(0, inplace=True)  


df_features_season['WinRatio'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])


df_features_season['GapAvg'] = (
    (df_features_season['NumWins'] * df_features_season['GapWins'] - 
    df_features_season['NumLosses'] * df_features_season['GapLosses'])
    / (df_features_season['NumWins'] + df_features_season['NumLosses'])
)


df_features_season.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)


df_features_season = df_features_season[df_features_season['Season']>= 2003]


df_features_season


season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
season_dresults.head()


win_season_shoot = season_dresults.groupby(['Season', 'WTeamID'])[['WScore','WFGA','WFTA']].sum().reset_index()
win_season_shoot['WTS'] = round(win_season_shoot['WScore'] / (2*(win_season_shoot['WFGA'] + 0.44 * win_season_shoot['WFTA'])),4)
win_season_shoot.head()


lose_season_shoot = season_dresults.groupby(['Season', 'LTeamID'])[['LScore','LFGA','LFTA']].sum().reset_index()
lose_season_shoot['LTS'] = round(lose_season_shoot['LScore'] / (2*(lose_season_shoot['LFGA'] + 0.44 * lose_season_shoot['LFTA'])),4)
lose_season_shoot.head()


# 提取 win_season_shoot 中需要的列（Season, WTeamID 和 WTS）
win_season_shoot_filtered = win_season_shoot[['Season', 'WTeamID', 'WTS']]

# 合并 df_features_season 和 win_season_shoot_filtered，匹配 Season 和 TeamID
df_features_season = df_features_season.merge(win_season_shoot_filtered, left_on=['Season', 'TeamID'], right_on=['Season', 'WTeamID'],how = 'left').drop(columns = 'WTeamID',axis = 1)

df_features_season.head()


df_features_season['WTS'] = df_features_season['WTS'].fillna(0)


# 提取 win_season_shoot 中需要的列（Season, WTeamID 和 WTS）
lose_season_shoot_filtered = lose_season_shoot[['Season', 'LTeamID', 'LTS']]

# 合并 df_features_season 和 win_season_shoot_filtered，匹配 Season 和 TeamID
df_features_season = df_features_season.merge(lose_season_shoot_filtered, left_on=['Season', 'TeamID'], right_on=['Season', 'LTeamID'],how = 'left').drop(columns = 'LTeamID',axis = 1)

df_features_season.head()


df_features_season['LTS'] = df_features_season['LTS'].fillna(0)


# ## 计算对手的得分
# opp_scoreW = season_dresults.groupby(['Season','WTeamID'])[['LScore','LFGA','LFTA','LOR','LTO']].sum().reset_index()
# opp_scoreW.head()


# opp_scoreW['WDE'] = opp_scoreW['LScore'] / (opp_scoreW['LFGA'] - opp_scoreW['LOR'] + opp_scoreW['LTO'] + (0.44 * opp_scoreW['LFTA']))
# opp_scoreW.head()


# opp_scoreL = season_dresults.groupby(['Season','LTeamID'])[['WScore','WFGA','WFTA','WOR','WTO']].sum().reset_index()
# opp_scoreL.head()


# opp_scoreL['LDE'] = opp_scoreL['WScore'] / (opp_scoreL['WFGA'] - opp_scoreL['WOR'] + opp_scoreL['WTO'] + (0.44 * opp_scoreL['WFTA']))
# opp_scoreL.head()


# win_season_de_filtered = opp_scoreW[['Season', 'WTeamID', 'WDE']]
# lose_season_de_filtered = opp_scoreL[['Season', 'LTeamID', 'LDE']]


# df_features_season = df_features_season.merge(win_season_de_filtered, left_on=['Season', 'TeamID'], right_on=['Season', 'WTeamID'],how = 'left').drop(columns = 'WTeamID',axis = 1)

# df_features_season.head()


# df_features_season = df_features_season.merge(lose_season_de_filtered, left_on=['Season', 'TeamID'], right_on=['Season', 'LTeamID'],how = 'left').drop(columns = 'LTeamID',axis = 1)

# df_features_season.head()


# ast = season_dresults.groupby(['Season','WTeamID'])[['WAst','WFGA','WFTA','WOR','WTO']].sum().reset_index()
# ast.head()


# ast['Wast'] = ast['WAst'] / (ast['WFGA'] - ast['WOR'] + ast['WTO'] + (0.44 * ast['WFTA']))
# ast.head()


# last = season_dresults.groupby(['Season','LTeamID'])[['LAst','LFGA','LFTA','LOR','LTO']].sum().reset_index()
# last.head()


# last['Last'] = last['LAst'] / (last['LFGA'] - last['LOR'] + last['LTO'] + (0.44 * last['LFTA']))
# last.head()


# win_season_ast_filtered = ast[['Season', 'WTeamID', 'Wast']]
# lose_season_ast_filtered = last[['Season', 'LTeamID', 'Last']]


# df_features_season = df_features_season.merge(win_season_ast_filtered, left_on=['Season', 'TeamID'], right_on=['Season', 'WTeamID'],how = 'left').drop(columns = 'WTeamID',axis = 1)

# df_features_season.head()


# df_features_season = df_features_season.merge(lose_season_ast_filtered, left_on=['Season', 'TeamID'], right_on=['Season', 'LTeamID'],how = 'left').drop(columns = 'LTeamID',axis = 1)

# df_features_season.head()


season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
season_dresults.head()


win_season_detailed = win_season_mean = season_dresults.groupby(['Season', 'WTeamID']).apply(lambda x: x.loc[:, 'WFGM':].mean()).reset_index()


win_season_detailed = win_season_detailed[['Season'] + [col for col in win_season_detailed.columns if col.startswith('W')]]


win_season_detailed.head()


lose_season_mean = season_dresults.groupby(['Season', 'LTeamID']).apply(lambda x: x.loc[:, 'LFGM':].mean()).reset_index()


lose_season_mean.head()


df_features_season = df_features_season.merge(win_season_detailed, left_on=['Season', 'TeamID'], right_on=['Season', 'WTeamID'],how = 'left').drop(columns = 'WTeamID',axis = 1)
df_features_season.head()


df_features_season = df_features_season.merge(lose_season_mean, left_on=['Season', 'TeamID'], right_on=['Season', 'LTeamID'],how = 'left').drop(columns = 'LTeamID',axis = 1)
df_features_season.head()


# rank = data['MMasseyOrdinals']
# rank


# rank_mean = rank.groupby(['Season','TeamID'])[['OrdinalRank']].agg('mean').reset_index()
# rank_mean.head()


# df_features_season = df_features_season.merge(rank_mean,on=['Season', 'TeamID'], how='left')


# df_features_season.head()


tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])


tourney_cresults.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
tourney_cresults


#有作者在这里只取2016年之后的数据，原因未知
df = tourney_cresults.copy()
df = df[df['Season'] >= 2003].reset_index(drop=True)

df


df = pd.merge(
    df, 
    seeds, 
    how='left', 
    left_on=['Season', 'WTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})


df = pd.merge(
    df, 
    seeds, 
    how='left', 
    left_on=['Season', 'LTeamID'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})


df['SeedW'] = df['SeedW'].str.extract('(\d+)').astype(int)
df['SeedL'] = df['SeedL'].str.extract('(\d+)').astype(int)


df_features_season.head()


df= pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsW',
    'NumLosses': 'NumLossesW',
    'GapWins': 'GapWinsW',
    'GapLosses': 'GapLossesW',
    'WinRatio': 'WinRatioW',
    'GapAvg': 'GapAvgW',
}).drop(columns='TeamID', axis=1)


df.columns


# cols_to_drop = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR',
#                 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF','LDE']
cols_to_drop = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR',
                'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
# 删除这些列
df = df.drop(columns=cols_to_drop,axis = 1)


# df = df.drop('Last',axis = 1)


df = pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsL',
    'NumLosses': 'NumLossesL',
    'GapWins': 'GapWinsL',
    'GapLosses': 'GapLossesL',
    'WinRatio': 'WinRatioL',
    'GapAvg': 'GapAvgL',
}).drop(columns='TeamID', axis=1)


# df = df.drop('Wast_y',axis = 1)


cols_to_drop = ['WFGM_y', 'WFGA_y', 'WFGM3_y', 'WFGA3_y', 'WFTM_y', 'WFTA_y',
                'WOR_y', 'WDR_y', 'WAst_y', 'WTO_y', 'WStl_y', 'WBlk_y', 'WPF_y']

# 删除这些列
df = df.drop(columns=cols_to_drop,axis = 1)


df.columns


df = df.rename(columns={'WTS_x':'TSA','LTS_y':'TSB'}).drop(['LTS_x','WTS_y'],axis = 1 )

df.head()


df = df.rename(columns=lambda x: x.replace('_x', '') if x.endswith('_x') else x)

df.columns


# df = df.drop(['WDE_y'],axis = 1)


# df = df.rename(columns={
#     'OrdinalRank_x': 'OrdinalRankA',
#     'OrdinalRank_y': 'OrdinalRankB'
# })



# 重命名胜方的列并生成胜方数据框
win_df = df.rename(columns={
    "WTeamID": "TeamIdA", 
    "WScore": "ScoreA", 
    "LTeamID": "TeamIdB", 
    "LScore": "ScoreB",
    "SeedW": "SeedA", 
    "WinRatioW": "WinRatioA", 
    "GapAvgW": "GapAvgA",
    "SeedL": "SeedB", 
    "WinRatioL": "WinRatioB", 
    "GapAvgL": "GapAvgB"
})

# 重命名负方的列并生成负方数据框
lose_df = df.rename(columns={
    "WTeamID": "TeamIdB", 
    "WScore": "ScoreB", 
    "LTeamID": "TeamIdA", 
    "LScore": "ScoreA",
    "SeedW": "SeedB", 
    "WinRatioW": "WinRatioB", 
    "GapAvgW": "GapAvgB",
    "SeedL": "SeedA", 
    "WinRatioL": "WinRatioA", 
    "GapAvgL": "GapAvgA"
})

# 合并胜方和负方的数据框
df = pd.concat([win_df, lose_df], ignore_index=True)

df.head()


df.columns


# 手动计算差值
df['SeedDiff'] = df['SeedA'] - df['SeedB']
df['WinRatioDiff'] = df['WinRatioA'] - df['WinRatioB']
df['GapAvgDiff'] = df['GapAvgA'] - df['GapAvgB']
# df['RankDiff'] = df['OrdinalRank'] - df['OrdinalRankB']


df_test = data['SampleSubmissionStage2']


df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))


df_test.head()


df_test = pd.merge(
    df_test,
    seeds,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'}).fillna('W01')


df_test = pd.merge(
    df_test, 
    seeds, 
    how='left', 
    left_on=['Season', 'TeamIdB'], 
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'}).fillna('W01')


df_test


df_test['SeedA'] = df_test['SeedA'].str.extract('(\d+)').astype(int)
df_test['SeedB'] = df_test['SeedB'].str.extract('(\d+)').astype(int)


df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsA',
    'NumLosses': 'NumLossesA',
    'GapWins': 'GapWinsA',
    'GapLosses': 'GapLossesA',
    'WinRatio': 'WinRatioA',
    'GapAvg': 'GapAvgA',
}).drop(columns='TeamID', axis=1)


df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsB',
    'NumLosses': 'NumLossesB',
    'GapWins': 'GapWinsB',
    'GapLosses': 'GapLossesB',
    'WinRatio': 'WinRatioB',
    'GapAvg': 'GapAvgB',
}).drop(columns='TeamID', axis=1)


df_test.head(39)


df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
df['WinA'] = (df['ScoreDiff'] > 0).astype(int)


df.head()


df.columns


train_data = df[(df['Season'] >= 2016) & (df['Season'] <= 2024)]

# 筛选出 2021 到 2024 年的测试数据
test_data = df[(df['Season'] >= 2021) & (df['Season'] <= 2024)]
test_data1 = df[df['Season'] == 2024]
test_data2 = df[df['Season'] == 2022]
test_data3 = df[df['Season'] == 2023]



train_data[train_data.isnull()]


season_cresults[(season_cresults['Season'] == 2003) & (season_cresults['DayNum'] == 138)]


 # train_data1 = train_data[train_data['TeamIdA'].astype(str).str.startswith('1')]


# features = [
#     "SeedA", "SeedB", 'WinRatioA', 'GapAvgA', 'WinRatioB', 'GapAvgB', 'SeedDiff', 'WinRatioDiff', 'GapAvgDiff'
# ]

# features = [
#     "SeedA", "SeedB", 'WinRatioA', 'GapAvgA', 'WinRatioB', 'GapAvgB', 'SeedDiff', 'WinRatioDiff', 'GapAvgDiff','TSA','TSB'
# ]

# features = [
#     'SeedA', 'SeedB', 'WinRatioA', 'GapAvgA', 'TSA', 'WFGM', 'WFGA', 'WFGM3',
#     'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
#     'WinRatioB', 'GapAvgB', 'TSB', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM',
#     'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'SeedDiff',
#     'WinRatioDiff', 'GapAvgDiff','WDE','LDE','Wast','Last'
# ]

features = [
    'SeedA', 'SeedB', 'WinRatioA', 'GapAvgA', 'TSA', 'WFGM', 'WFGA', 'WFGM3',
    'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
    'WinRatioB', 'GapAvgB', 'TSB', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM',
    'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'SeedDiff',
    'WinRatioDiff', 'GapAvgDiff'
]


selected = train_data[features]
y = train_data['WinA']


from sklearn.model_selection import GridSearchCV, train_test_split
x_train , x_test , y_train , y_test = train_test_split(selected, y, test_size=0.2, random_state=42) # 80% 训练，20% 测试


import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score  # 导入二分类指标
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 特征标准化
    ('lgbm', lgb.LGBMClassifier(objective='binary', random_state=42))  # LGBM 二分类模型
])


# param_grid = {
#     'lgbm__num_leaves': [63, 127], # 减少范围
#     'lgbm__learning_rate': [0.01, 0.1], # 减少范围
#     'lgbm__n_estimators': [200, 300], # 减少范围
#     'lgbm__max_depth': [5, 8], # 减少范围
#     'lgbm__min_child_samples': [30, 40], # 减少范围
# }




from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, brier_score_loss

# 创建 Brier Score Loss 的 scorer
brier_scorer = make_scorer(brier_score_loss, greater_is_better=False)

# 使用 GridSearchCV 进行超参数搜索
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=brier_scorer, verbose=1)
grid_search.fit(x_train, y_train)


best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# 打印最佳参数
print("Best Parameters:", best_params)


# 使用最佳模型进行预测
y_pred = best_model.predict(x_test)
y_pred_proba = best_model.predict_proba(x_test)[:, 1]  # 获取正例的概率


brier_loss = brier_score_loss(y_test, y_pred_proba)  # 计算 brier_score_loss
print("Brier Score Loss:", brier_loss)


a = test_data[features]
b = test_data['WinA']


a1 = test_data1[features]
b1 = test_data1['WinA']


y_pred = best_model.predict(a)
y_pred_proba = best_model.predict_proba(a)[:, 1]
brier_loss = brier_score_loss(b, y_pred_proba) 
print("Brier Score Loss:", brier_loss)


from sklearn.svm import SVC



from sklearn.impute import SimpleImputer
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),  # 用均值填充缺失值
    ('svm', SVC(probability=True))  # 使用SVC模型，开启probability选项以输出概率
])


param_grid = {
    'svm__C': [0.01,0.05,0.1, 1],  # C的不同值
    'svm__kernel': ['linear', 'rbf'],  # 核函数类型
    'svm__gamma': ['scale', 'auto', 0.1, 1],  # gamma值
    'svm__degree': [3, 5],  # 仅适用于多项式核
}



from tqdm.auto import tqdm
# 使用GridSearchCV进行超参数搜索
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=brier_scorer, verbose=1)
grid_search.fit(x_train, y_train)


best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# 打印最佳参数
print("Best Parameters:", best_params)


# 使用最佳模型进行预测
y_pred = best_model.predict(x_test)
y_pred_proba = best_model.predict_proba(x_test)[:, 1]  # 获取正例的概率


brier_loss = brier_score_loss(y_test, y_pred_proba)  # 计算 brier_score_loss
print("Brier Score Loss:", brier_loss)


y_pred = best_model.predict(a)
y_pred_proba = best_model.predict_proba(a)[:, 1]
brier_loss = brier_score_loss(b, y_pred_proba) 
print("Brier Score Loss:", brier_loss)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# param_dist = {
#     'n_estimators': [50, 100, 150, 200, 300],  # 树的数量
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率
#     'max_depth': [3, 5, 7, 10, -1],  # 树的最大深度
#     'num_leaves': [20, 30, 50, 100],  # 叶子节点数
#     'subsample': [0.7, 0.8, 0.9, 1.0],  # 训练样本的比例
#     'colsample_bytree': [0.7, 0.8, 1.0],  # 特征采样比例
#     'min_child_samples': [5, 10, 20],  # 每个叶子节点的最小样本数
#     'reg_alpha': [0, 0.1, 0.5, 1.0],  # L1正则化
#     'reg_lambda': [0, 0.1, 0.5, 1.0]  # L2正则化
# }

param_grid_lgbm = {
    'n_estimators': [100, 200, 500],                # 迭代次数（决策树数量）
    'learning_rate': [0.01, 0.05, 0.1],             # 学习率
    'max_depth': [5, 7, 8],                         # 树的最大深度
    'num_leaves': [31, 50, 100],                     # 树的叶子节点数目
    'min_child_samples': [20, 50, 100],              # 每个叶子节点的最小样本数
    'subsample': [0.7, 0.8, 0.9],                    # 子样本的比例
    'colsample_bytree': [0.7, 0.8, 0.9],             # 每棵树使用的特征的比例
    'reg_alpha': [0, 0.1, 1],                        # L1 正则化系数
    'reg_lambda': [0, 0.1, 1],                       # L2 正则化系数
}

def rescale(features, df_train, df_val, df_test):
    scaler = StandardScaler()
    
    # Fit and transform the training set
    df_train[features] = scaler.fit_transform(df_train[features])
    
    # Only transform the validation and test sets based on the training set scaling
    df_val[features] = scaler.transform(df_val[features])
    df_test[features] = scaler.transform(df_test[features])
    
    return df_train, df_val, df_test

def kfold(df, df_test_=None, plot=False, verbose=0, mode="reg"):
    seasons = df['Season'].unique()
    cvs = []
    pred_tests = []
    target = "ScoreDiff" if mode == "reg" else "WinA"
    
    for season in seasons[1:]:
        if verbose:
            print(f'\nValidating on season {season}')
        
        df_train = df[df['Season'] < season].reset_index(drop=True).copy()
        df_val = df[df['Season'] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()
        
        df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)
        
        if mode == "reg":
            model = ElasticNet(alpha=1, l1_ratio=0.5)
        else:
            model = RandomizedSearchCV(
                estimator=LGBMClassifier(objective='binary', 
                                       boosting_type='dart',
                                       metric='binary_logloss',
                                       random_state=42,
                                       importance_type='gain'),  # 使用 LGBMClassifier
                param_distributions=param_grid_lgbm,        # 使用 LGBM 的参数网格
                n_iter=100,                                 # 随机搜索的次数
                cv=5,                                       # 交叉验证次数
                scoring='neg_log_loss',                     # 使用对数损失评分
                verbose=1,                                  # 输出详细信息
                random_state=42,                            # 随机种子
                n_jobs=-1                                   # 使用所有可用的 CPU
            )
        
        model.fit(df_train[features], df_train[target])

        best_lgbm_model = model.best_estimator_  # 获取最佳模型
        best_lgbm_model.fit(df_train[features], df_train[target])
        if mode == "reg":
            pred = best_lgbm_model.predict(df_val[features])
        else:
            pred = best_lgbm_model.predict_proba(df_val[features])[:, 1]
        
        if df_test is not None:
            if mode == "reg":
                pred_test = best_lgbm_model.predict(df_test[features])
                pred_test = (pred_test - pred_test.min()) / (pred_test.max() - pred_test.min())
            else:
                pred_test = best_lgbm_model.predict_proba(df_test[features])[:, 1]
            
            pred_tests.append(pred_test)
        
        if plot:
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(pred, df_val['ScoreDiff'].values, s=5)
            plt.title('Prediction vs Score Diff')
            plt.grid(True)
            plt.subplot(1, 2, 2)
            sns.histplot(pred, bins=20)
            plt.title('Predictions probability repartition')
            plt.show()

        # pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred = np.clip(pred, 0, 1)

        brier_score = ((df_val['WinA'].values - pred) ** 2).mean()
        cvs.append(brier_score)

        if verbose:
            print(f'\t -> Scored {brier_score:.3f}')
        
    print(f'\n Local CV is {np.mean(cvs):.3f}')
    
    return pred_tests


from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


bayes_params = {
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 10),
                'num_leaves': Integer(20, 100),
                'min_child_samples': Integer(5, 50),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0),
                'reg_alpha': Real(1e-8, 10.0, prior='log-uniform'),
                'reg_lambda': Real(1e-8, 10.0, prior='log-uniform'),
                'min_split_gain': Real(1e-8, 1.0, prior='log-uniform'),
                'n_estimators': Integer(50, 300)
            }




def kfold(df, df_test_=None, plot=False, verbose=0, mode="reg"):
    seasons = df['Season'].unique()
    cvs = []
    pred_tests = []
    target = "ScoreDiff" if mode == "reg" else "WinA"
    
    for season in seasons[1:]:
        if verbose:
            print(f'\nValidating on season {season}')
        
        df_train = df[df['Season'] < season].reset_index(drop=True).copy()
        df_val = df[df['Season'] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()
        
        df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)
        
        if mode == "reg":
            model = ElasticNet(alpha=1, l1_ratio=0.5)
        else:
            
            model = BayesSearchCV(
                estimator=LGBMClassifier(
                    objective='binary', 
                    boosting_type='dart',
                    metric='binary_logloss',
                    random_state=42,
                    importance_type='gain'
                ),
                search_spaces=bayes_params,  # 使用正确定义的skopt参数空间
                n_iter=100,
                cv=5,
                scoring='neg_log_loss',
                verbose=1,
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(df_train[features], df_train[target])
        best_params = model.best_params_
        print("最佳参数:", best_params)
        best_model = LGBMClassifier(
                objective='binary', 
                boosting_type='dart',
                metric='binary_logloss',
                random_state=42,
                importance_type='gain',
                **best_params  # 使用找到的最佳参数
            )
            
            # 在训练集上重新拟合最佳模型
        best_model.fit(df_train[features], df_train[target])

        if mode == "reg":
            best_model = model
            pred = best_model.predict(df_val[features])
        else:
            best_model = model.best_estimator_  # 获取最佳模型
            pred = best_model.predict_proba(df_val[features])[:, 1]
        
        if df_test is not None:
            if mode == "reg":
                pred_test = best_model.predict(df_test[features])
                pred_test = (pred_test - pred_test.min()) / (pred_test.max() - pred_test.min())
            else:
                pred_test = best_model.predict_proba(df_test[features])[:, 1]
            
            pred_tests.append(pred_test)
        
        if plot:
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(pred, df_val['ScoreDiff'].values, s=5)
            plt.title('Prediction vs Score Diff')
            plt.grid(True)
            plt.subplot(1, 2, 2)
            sns.histplot(pred, bins=20)
            plt.title('Predictions probability repartition')
            plt.show()

        # pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred = np.clip(pred, 0, 1)

        brier_score = ((df_val['WinA'].values - pred) ** 2).mean()
        cvs.append(brier_score)

        if verbose:
            print(f'\t -> Scored {brier_score:.3f}')
        
    print(f'\n Local CV is {np.mean(cvs):.3f}')
    
    return pred_tests


pred_tests = kfold(train_data, test_data1, plot=False, verbose=1, mode="cls")


test_data1.isna().sum()


print(train_data['Season'].unique())  # 查看目标变量唯一值


