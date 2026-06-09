import numpy as np
import pandas as pd
import glob



path = '/kaggle/input/march-machine-learning-mania-2025/**'


#key是只取文件名
#值是读取文件内容
#glob.glob() 函数根据给定的路径模式查找文件，这里 path 表示的是要查找所有 CSV 文件。
data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}


#合并男女的队伍名称，concat() 默认是按行（纵向）拼接，如果列名不同，会自动返回nan
teams = pd.concat([data['MTeams'], data['WTeams']])


teams.head()


teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])


teams_spelling.head()


teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()


teams_spelling.head()


#修改列名
teams_spelling.columns = ['TeamID','TeamNameCount']


teams_spelling.head()


teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])


teams['FirstD1Season'] = teams['FirstD1Season'].fillna(0)
teams['LastD1Season'] = teams['LastD1Season'].fillna(0)



teams['season_diff'] = teams['LastD1Season'] - teams['FirstD1Season']
teams['season_diff'] = teams['season_diff'].astype(int)
teams.head()


season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])

season_cresults.head(10)
#赛季、日期编号、胜队 ID、胜分、负队 ID、负分、主场位置和加时赛次数


season_cresults[season_cresults['NumOT'] != 0]


season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
season_dresults.head()


tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
tourney_cresults.head()#常规赛的最后一天也是132日


tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])
tourney_dresults.head()


seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
seeds.head(20)


gcities = pd.concat([data['MGameCities'], data['WGameCities']])
gcities.head()


seasons = pd.concat([data['MSeasons'], data['WSeasons']])
seasons.head()


# seeds = {'_'.join(map(str, [int(k1), k2])): int(v[1:3]) 
#          for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}



cities = data['Cities']
sub = data['SampleSubmissionStage1']


cities.head()


sub.head()


season_cresults['ST'] = 'S'
season_dresults['ST'] = 'S'
tourney_cresults['ST'] = 'T'
tourney_dresults['ST'] = 'T'


games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map({'A':1,'H':2,'N':3})


games.info()


#对每行就行处理，将胜者id，负者id进行排序合并，row只是一个参数
games['Match'] = games.apply(lambda row: "_".join(map(str, sorted([row['WTeamID'], row['LTeamID']]))) , axis = 1)


#只用到一列，因此要对games['Match']一列进行操作即可
games["Team1"] = games["Match"].apply(lambda x: int(x.split("_")[0]))
games["Team2"] = games["Match"].apply(lambda x: int(x.split("_")[1]))


#因为要用到dataframe的两列，因此lambda中的x取games中的任意行
games['label'] = games.apply(lambda x:1 if x['WTeamID'] == x['Team1'] else 0 ,axis = 1) 


games['type'] = games['ST'].map({'S':0 , 'T':1})


games.head(20)


games['ScoreMargin'] = np.where(games['WTeamID'] < games['LTeamID'],games['WScore'] - games['LScore'],games['LScore'] - games['WScore'])


clean_datav1 = games[['Season' , 'DayNum' , 'WTeamID' , 'LTeamID','ST' , 'label']]


#新建得分差异

clean_datav1 = pd.merge(clean_datav1, games[['Season' , 'DayNum' , 'WTeamID' , 'LTeamID', 'ScoreMargin']], left_on=['Season' , 'DayNum' , 'WTeamID' , 'LTeamID'], right_on=['Season' , 'DayNum' , 'WTeamID' , 'LTeamID'], how='left')




clean_datav1['season_type'] = clean_datav1['ST'].map({'S':0 ,'T':1})


#合并大小id
clean_datav1['final_ID'] = clean_datav1.apply(lambda row: f"{np.minimum(row['WTeamID'], row['LTeamID'])}_{np.maximum(row['WTeamID'], row['LTeamID'])}", axis=1)


wteamid_index = clean_datav1.columns.get_loc('WTeamID')
clean_datav1.insert(wteamid_index, 'final_ID', clean_datav1.pop('final_ID'))


clean_datav1.head()


total_games = pd.concat([
    clean_datav1.groupby(['Season', 'WTeamID']).size(),
    clean_datav1.groupby(['Season', 'LTeamID']).size()
]).reset_index()
total_games.columns = ['Season', 'TeamID', 'TotalGames']  # 统一列名
total_games = total_games.groupby(['Season', 'TeamID'])['TotalGames'].sum().reset_index()


total_games.head()


wins = clean_datav1.groupby(['Season', 'WTeamID']).size().reset_index(name='Wins')
wins.rename(columns={'WTeamID': 'TeamID'}, inplace=True) # 重命名 WTeamID 为 TeamID


team_stats = pd.merge(total_games, wins, on=['Season', 'TeamID'], how='left').fillna(0)
team_stats['WinRate'] = round(team_stats['Wins'] / team_stats['TotalGames'],4)
team_stats.head()


clean_datav1 = pd.merge(clean_datav1, team_stats[['Season', 'TeamID', 'WinRate']],
                        left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
clean_datav1 = clean_datav1.rename(columns={'WinRate': 'WTeam_WinRate'}).drop('TeamID', axis=1)

clean_datav1 = pd.merge(clean_datav1, team_stats[['Season', 'TeamID', 'WinRate']],
                        left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
clean_datav1 = clean_datav1.rename(columns={'WinRate': 'LTeam_WinRate'}).drop('TeamID', axis=1)


clean_datav1['SWinRate'] = np.where(clean_datav1['WTeamID'] < clean_datav1['LTeamID'],
                                      clean_datav1['WTeam_WinRate'] ,
                                      clean_datav1['LTeam_WinRate'] )


clean_datav1['LWinRate'] = np.where(clean_datav1['WTeamID'] < clean_datav1['LTeamID'],
                                      clean_datav1['LTeam_WinRate'] ,
                                      clean_datav1['WTeam_WinRate'] )


clean_datav1['WinRateDiff'] = np.where(clean_datav1['WTeamID'] < clean_datav1['LTeamID'],
                                      clean_datav1['WTeam_WinRate'] - clean_datav1['LTeam_WinRate'],
                                      clean_datav1['LTeam_WinRate'] - clean_datav1['WTeam_WinRate'])


clean_datav1.head(20)


clean_datav1['WLoc'] = games['WLoc']


# 2. 创建辅助列，用于存储交换后的主客场信息
clean_datav1['WLoc_adjusted'] = clean_datav1['WLoc']  # 初始值与 WLoc 相同
clean_datav1['LLoc_adjusted'] = 0 #先都填充0


# 3. 根据条件交换主客场
condition = (clean_datav1['WTeamID'] > clean_datav1['LTeamID']) & (clean_datav1['WLoc'] != 3) #这里不能用and

clean_datav1.loc[condition, 'LLoc_adjusted'] = clean_datav1.loc[condition, 'WLoc'].map({1:2, 2:1})
clean_datav1.loc[condition, 'WLoc_adjusted'] = clean_datav1.loc[condition, 'LLoc_adjusted']
clean_datav1.loc[~condition, 'LLoc_adjusted'] = clean_datav1.loc[~condition, 'WLoc'].map({1:2, 2:1, 3:3})


clean_datav1.drop('WLoc', axis=1, inplace=True)
clean_datav1.drop('LLoc_adjusted', axis=1, inplace=True)


clean_datav1 = clean_datav1.rename(columns={'WLoc_adjusted': 'WLoc'})


clean_datav1 = pd.merge(clean_datav1, season_cresults[['Season', 'DayNum', 'WTeamID','LTeamID', 'NumOT']],
                        on=['Season', 'DayNum','WTeamID','LTeamID'], how='left')


clean_datav1['Overtime'] = (clean_datav1['NumOT'] > 0).astype(int)


#清洗缺失值
clean_datav1['Overtime'] = clean_datav1['Overtime'].fillna(0)


clean_datav1.drop('NumOT', axis=1, inplace=True)


# games['WFG_Rate'] = round(games['WFGM'] /games['WFGA'] ,4)
# games['LFG_Rate'] = round(games['LFGM'] /games['LFGA'] ,4)

#计算真实投篮命中率=总得分 / (2 × (投篮出手数 + 0.44 × 罚球出手数))
games['WFG_Rate'] = round(games['WScore'] / (2 *games['WFGA'] + 0.44 *games['WFTA']) ,4)
games['LFG_Rate'] = round(games['LScore'] / (2 *games['LFGA'] + 0.44 *games['LFTA']) ,4)


clean_datav1 = pd.merge(clean_datav1, games[['Season', 'DayNum', 'WTeamID','LTeamID', 'WFG_Rate']],
                        on=['Season', 'DayNum','WTeamID','LTeamID'], how='left')


clean_datav1 = pd.merge(clean_datav1, games[['Season', 'DayNum', 'WTeamID','LTeamID', 'LFG_Rate']],
                        on=['Season', 'DayNum','WTeamID','LTeamID'], how='left')


#SWFG_Rate是小id，另一个是大id
clean_datav1['SWFG_Rate'] = clean_datav1.apply(lambda x: x['WFG_Rate']  if x['WTeamID'] < x['LTeamID'] else x['LFG_Rate'] ,axis = 1)
clean_datav1['LWFG_Rate'] = clean_datav1.apply(lambda x: x['LFG_Rate']  if x['WTeamID'] < x['LTeamID'] else x['WFG_Rate'] ,axis = 1)


clean_datav1.drop('WFG_Rate', axis=1, inplace=True)
clean_datav1.drop('LFG_Rate', axis=1, inplace=True)


#计算每回合数，防守效率以对方为衡量标准，数值越低防守越好
games['WPossessions'] = games['WFGA'] - games['WOR'] + games['WTO'] + (0.44 * games['WFTA'])
games['LPossessions'] = games['LFGA'] - games['LOR'] + games['LTO'] + (0.44 * games['LFTA'])

#球队B在面对球队A时，每100个回合的得分
def calculate_drtg(row):
    if row['WTeamID'] < row['LTeamID']:  # 获胜队伍ID小于失败队伍ID
        return (row['LScore'] / row['LPossessions']) * 100 if row['LPossessions'] !=0 else 0
    else:
        return (row['WScore'] / row['WPossessions']) * 100 if row['WPossessions'] !=0 else 0

games['DRtg'] = games.apply(calculate_drtg, axis=1)


clean_datav1 = pd.merge(clean_datav1, games[['Season', 'DayNum', 'WTeamID','LTeamID', 'DRtg']],
                        on=['Season', 'DayNum','WTeamID','LTeamID'], how='left')


games_detailed = games.copy()


cols_to_process = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'Score']

for col in cols_to_process:
    games_detailed[f'first_{col}'] = np.where(games_detailed['WTeamID'] < games_detailed['LTeamID'], games_detailed[f'W{col}'], games_detailed[f'L{col}'])
    games_detailed[f'final_{col}'] = np.where(games_detailed['WTeamID'] < games_detailed['LTeamID'], games_detailed[f'L{col}'], games_detailed[f'W{col}'])


data = games_detailed[['Season', 'DayNum', 'WTeamID', 'LTeamID'] + 
                      [col for col in games_detailed.columns if 'first_' in col or 'final_' in col]].copy() #复制一份，防止修改原始数据


clean_datav1 = pd.merge(clean_datav1, data,
                        on=['Season', 'DayNum','WTeamID','LTeamID'], how='left')


#计算每回合数
games['WPossessions'] = games['WFGA'] - games['WOR'] + games['WTO'] + (0.44 * games['WFTA'])
games['LPossessions'] = games['LFGA'] - games['LOR'] + games['LTO'] + (0.44 * games['LFTA'])

def calculate_drtg(row):
    if row['WTeamID'] < row['LTeamID']:  # 获胜队伍ID小于失败队伍ID
        return (row['WAst'] / row['WPossessions']) * 100 if row['WAst'] !=0 else 0
    else:
        return (row['LAst'] / row['LPossessions']) * 100 if row['LAst'] !=0 else 0

games['SAST'] = games.apply(calculate_drtg, axis=1)


#计算每回合数
games['WPossessions'] = games['WFGA'] - games['WOR'] + games['WTO'] + (0.44 * games['WFTA'])
games['LPossessions'] = games['LFGA'] - games['LOR'] + games['LTO'] + (0.44 * games['LFTA'])

def calculate_drtg(row):
    if row['WTeamID'] < row['LTeamID']:  # 获胜队伍ID小于失败队伍ID
        return (row['LAst'] / row['LPossessions']) * 100 if row['LAst'] !=0 else 0
    else:
        return (row['WAst'] / row['WPossessions']) * 100 if row['WAst'] !=0 else 0

games['LAST'] = games.apply(calculate_drtg, axis=1)


clean_datav1 = pd.merge(clean_datav1, games[['Season', 'DayNum', 'WTeamID','LTeamID', 'SAST']],
                        on=['Season', 'DayNum','WTeamID','LTeamID'], how='left')

clean_datav1 = pd.merge(clean_datav1, games[['Season', 'DayNum', 'WTeamID','LTeamID', 'LAST']],
                        on=['Season', 'DayNum','WTeamID','LTeamID'], how='left')


#计算每回合数
games['WPossessions'] = games['WFGA'] - games['WOR'] + games['WTO'] + (0.44 * games['WFTA'])
games['LPossessions'] = games['LFGA'] - games['LOR'] + games['LTO'] + (0.44 * games['LFTA'])

def calculate_drtg(row):
    if row['WTeamID'] < row['LTeamID']:  # 获胜队伍ID小于失败队伍ID
        return (row['WTO'] / row['WPossessions']) * 100 if row['WTO'] !=0 else 0
    else:
        return (row['LTO'] / row['LPossessions']) * 100 if row['LTO'] !=0 else 0

games['STO'] = games.apply(calculate_drtg, axis=1)


#计算每回合数
games['WPossessions'] = games['WFGA'] - games['WOR'] + games['WTO'] + (0.44 * games['WFTA'])
games['LPossessions'] = games['LFGA'] - games['LOR'] + games['LTO'] + (0.44 * games['LFTA'])

def calculate_drtg(row):
    if row['WTeamID'] < row['LTeamID']:  # 获胜队伍ID小于失败队伍ID
        return (row['LTO'] / row['LPossessions']) * 100 if row['LTO'] !=0 else 0
    else:
        return (row['WTO'] / row['WPossessions']) * 100 if row['WTO'] !=0 else 0

games['LTO1'] = games.apply(calculate_drtg, axis=1)


clean_datav1 = pd.merge(clean_datav1, games[['Season', 'DayNum', 'WTeamID','LTeamID', 'STO']],
                        on=['Season', 'DayNum','WTeamID','LTeamID'], how='left')

clean_datav1 = pd.merge(clean_datav1, games[['Season', 'DayNum', 'WTeamID','LTeamID', 'LTO1']],
                        on=['Season', 'DayNum','WTeamID','LTeamID'], how='left')


seeds['Seed1'] = seeds['Seed'].str.extract('(\d+)').astype(int)




clean_datav1 = pd.merge(clean_datav1, seeds[['Season', 'TeamID', 'Seed1']], left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left').drop(columns='TeamID')

clean_datav1 = pd.merge(clean_datav1, seeds[['Season', 'TeamID', 'Seed1']], left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left').drop(columns='TeamID')


#nan值填充当赛季最大值+1
def count_unique_teams(df, season):
    """计算一个赛季中独特的队伍数量。"""
    teams_w = df[df['Season'] == season]['WTeamID'].unique()
    teams_l = df[df['Season'] == season]['LTeamID'].unique()
    all_teams = pd.concat([pd.Series(teams_w), pd.Series(teams_l)]).unique()
    return len(all_teams)

season_team_counts = {
    season: count_unique_teams(clean_datav1, season)
    for season in clean_datav1['Season'].unique()
}

# 2. 使用队伍数量 + 1 填充 NaN
for col in ['Seed1_x', 'Seed1_y']:
    clean_datav1[col] = clean_datav1.apply(
        lambda row: season_team_counts[row['Season']] + 1 if pd.isna(row[col]) else row[col],
        axis=1
    )

# for col in ['Seed1_x', 'Seed1_y']: 
#     clean_datav1[col] = clean_datav1.groupby('Season')[col].transform(lambda x: x.fillna(x.max() + 1))


clean_datav1['Sseed'] = clean_datav1.apply(lambda x: x['Seed1_x']  if x['WTeamID'] < x['LTeamID'] else x['Seed1_y'] ,axis = 1)
clean_datav1['Lseed'] = clean_datav1.apply(lambda x: x['Seed1_y']  if x['WTeamID'] < x['LTeamID'] else x['Seed1_x'] ,axis = 1)


clean_datav1.drop(['Seed1_x' , 'Seed1_y'],axis = 1,inplace = True)


clean_datav1['seed_diff'] = clean_datav1['Sseed'] - clean_datav1['Lseed']


clean_datav1.head(20)


# 1.  创建获胜方的数据框 (win_df) -  每行代表一场获胜的比赛，为获胜球队标记 'Win' = 1
win_df = games[['Season', 'DayNum', 'WTeamID']].copy() # 复制需要的列
win_df.rename(columns={'WTeamID': 'TeamID'}, inplace=True) # 将 'WTeamID' 列重命名为 'TeamID'，方便后续合并
win_df['Win'] = 1  #  对于获胜球队，'Win' 列标记为 1 (代表胜利)
win_df.head()


# 2. 创建失败方的数据框 (lose_df) - 每行代表一场失败的比赛，为失败球队标记 'Win' = 0 (或者根据 label 决定)
lose_df = games[['Season', 'DayNum', 'LTeamID']].copy() # 复制需要的列
lose_df.rename(columns={'LTeamID': 'TeamID'}, inplace=True) # 将 'LTeamID' 列重命名为 'TeamID'，与 win_df 统一
lose_df['Win'] = 0
lose_df.head()


# 3. 合并获胜方 (win_df) 和失败方 (lose_df) 的数据框
team_games_df = pd.concat([win_df, lose_df], ignore_index=True) 
team_games_df.head()


team_games_df.sort_values(by=['Season', 'TeamID', 'DayNum'], inplace=True)


team_games_df['CumulativeWins'] = team_games_df.groupby(['Season', 'TeamID'])['Win'].cumsum()
team_games_df['CumulativeGames'] = team_games_df.groupby(['Season', 'TeamID']).cumcount() + 1
team_games_df['CumulativeWinRate'] = team_games_df['CumulativeWins'] / team_games_df['CumulativeGames']



team_games_df['ratio'] = pd.NA

# 使用 shift() 获取上一行的 CumulativeWinRate
previous_win_rate = team_games_df['CumulativeWinRate'].shift(1)

# 使用 numpy.where()  进行条件判断和赋值
# 条件1: CumulativeGames == 1， 则 胜率变化 = 0.5
condition_games_1 = team_games_df['CumulativeGames'] == 1
team_games_df['ratio'] = np.where(condition_games_1, 0.5, team_games_df['ratio'])

# 条件2: 其他情况 (CumulativeGames != 1)， 则 胜率变化 = 上一行的 CumulativeWinRate
condition_others = ~condition_games_1 #  ~ 表示逻辑非，即 CumulativeGames != 1
team_games_df['ratio'] = np.where(condition_others, previous_win_rate, team_games_df['ratio'])


# 1. 准备 team_games_df
team_games_ratio_w = team_games_df[['Season', 'DayNum', 'TeamID', 'ratio']].copy()
team_games_ratio_w.rename(columns={'TeamID': 'WTeamID'}, inplace=True)

team_games_ratio_l = team_games_df[['Season', 'DayNum', 'TeamID', 'ratio']].copy()
team_games_ratio_l.rename(columns={'TeamID': 'LTeamID'}, inplace=True)
# 2. 合并 (WTeamID)
clean_datav1 = pd.merge(clean_datav1, team_games_ratio_w,
                        on=['Season', 'DayNum', 'WTeamID'],
                        how='left')
clean_datav1.rename(columns={'ratio': 'W_ratio'}, inplace=True)

# 3. 合并 (LTeamID)
clean_datav1 = pd.merge(clean_datav1, team_games_ratio_l,
                        on=['Season', 'DayNum', 'LTeamID'],
                        how='left')
clean_datav1.rename(columns={'ratio': 'L_ratio'}, inplace=True)



clean_datav1.head()


clean_datav1['s_ratio'] = np.where(clean_datav1['WTeamID'] < clean_datav1['LTeamID'],
                                  clean_datav1['W_ratio'],
                                  clean_datav1['L_ratio'])

clean_datav1['l_ratio'] = np.where(clean_datav1['WTeamID'] < clean_datav1['LTeamID'],
                                  clean_datav1['L_ratio'],
                                  clean_datav1['W_ratio'])

# 5. 删除 W_ratio 和 L_ratio
clean_datav1.drop(columns=['W_ratio', 'L_ratio'], inplace=True)



# 将 's_ratio' 列转换为 float 类型，如果遇到无法转换的值，将其转换为 NaN
clean_datav1['s_ratio'] = pd.to_numeric(clean_datav1['s_ratio'], errors='coerce')

# 将 'l_ratio' 列转换为 float 类型，同样处理转换错误
clean_datav1['l_ratio'] = pd.to_numeric(clean_datav1['l_ratio'], errors='coerce')


clean_datav1.drop(['LTeam_WinRate','WTeam_WinRate'],axis = 1 ,inplace = True)


y = clean_datav1['label']


# 获取 'label' 列的索引位置
label_column_index = clean_datav1.columns.get_loc('label')

# 选择 'label' 列之后的所有列作为特征变量 x
# 使用 iloc[:, ...]  按位置索引来选择，从 label_column_index + 1 开始到最后一列
x = clean_datav1.iloc[:, (label_column_index + 1):].copy()
x.drop('ScoreMargin',axis = 1 , inplace = True)


import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # 80% 训练，20% 测试


model = xgb.XGBClassifier(objective='binary:logistic', #  或者 'multi:softmax' 如果是多分类
                          use_label_encoder=False,  #  避免警告，新版 XGBoost 需要设置
                          eval_metric='logloss', #  或者 'mlogloss' 如果是多分类,  或者 'rmse' 如果是回归
                          random_state=42)





model.fit(X_train, y_train)


importance = model.feature_importances_
feature_names = X_train.columns # 获取特征名，假设 x 是 DataFrame



feature_importance_df = pd.DataFrame({'Feature': feature_names,
                                       'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n特征重要性 (基于 Gain):")
print(feature_importance_df)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('特征') #  中文标签
plt.ylabel('重要性 (Gain)') #  中文标签
plt.title('XGBoost 特征重要性') #  中文标题
plt.tight_layout()
plt.show()


selected_features = feature_importance_df[feature_importance_df['Importance'] > 0.01]['Feature'].tolist()
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


X_train_selected.head()




