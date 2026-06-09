import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os


# path_root = '.\\march-machine-learning-mania-2025'
path_root = '/kaggle/input/march-machine-learning-mania-2025'


mteams = pd.read_csv(os.path.join(path_root,'MTeams.csv'))          # 'TeamID', 'TeamName', 'FirstD1Season', 'LastD1Season'
mseasons = pd.read_csv(os.path.join(path_root,'MSeasons.csv'))
regular = pd.read_csv(os.path.join(path_root,'MRegularSeasonCompactResults.csv'))   # Season  DayNum  WTeamID  WScore  LTeamID  LScore WLoc  NumOT
tourney = pd.read_csv(os.path.join(path_root,'MNCAATourneyCompactResults.csv'))
seeds = pd.read_csv(os.path.join(path_root,'MNCAATourneySeeds.csv'))

# data shape
print(f"Teams: {mteams.shape}, Seasons: {mseasons.shape}")
print(f"Regular Season Games: {regular.shape}, Tourney Games: {tourney.shape}")

# data columns
print("\nTeams Columns:", mteams.columns.tolist())
print("Regular Season Head:\n", regular.head(3))


# 球队参赛年限分析
plt.figure(figsize=(10,6))
mteams['LastD1Season'].hist(bins=30)
plt.title('Distribution of Team Last D1 Season')
plt.xlabel('Year')

# 比赛得分分布
plt.figure(figsize=(12,5))
plt.subplot(121)
regular['WScore'].plot(kind='hist', bins=50, title='Winning Scores')
plt.subplot(122)
regular['LScore'].plot(kind='hist', bins=50, title='Losing Scores')

# 种子排名分析
seeds['SeedNum'] = seeds['Seed'].str.extract('(\d+)').astype(int)
plt.figure(figsize=(10,6))
seeds['SeedNum'].value_counts().sort_index().plot(kind='bar')
plt.title('Seed Number Distribution')


# 计算分差
regular['ScoreDiff'] = regular['WScore'] - regular['LScore']
tourney['ScoreDiff'] = tourney['WScore'] - tourney['LScore']

# 分差分布
plt.figure(figsize=(12,5))
sns.histplot(data=regular, x='ScoreDiff', bins=50, kde=True)
plt.title('Regular Season Score Difference Distribution')

# 加时赛分析
ot_games = regular[regular['NumOT'] > 0]
print(f"OT Games Percentage: {len(ot_games)/len(regular):.2%}")

# 主客场分析
loc_counts = regular['WLoc'].value_counts()
plt.pie(loc_counts, labels=loc_counts.index, autopct='%1.1f%%')
plt.title('Winning Team Location Distribution')


# 按赛季聚合数据
regular_by_season = regular.groupby('Season').agg({
    'WScore': 'mean',
    'LScore': 'mean',
    'ScoreDiff': 'mean'
}).reset_index()

# 绘制趋势线
plt.figure(figsize=(12,6))
plt.plot(regular_by_season['Season'], regular_by_season['WScore'], label='Winning Score')
plt.plot(regular_by_season['Season'], regular_by_season['LScore'], label='Losing Score')
plt.legend()
plt.title('Average Scores Over Time')


# 合并种子数据与锦标赛结果
tourney_results = pd.merge(
    tourney,
    seeds.rename(columns={'TeamID':'WTeamID', 'Seed':'WSeed'}),
    on=['Season', 'WTeamID']
).merge(
    seeds.rename(columns={'TeamID':'LTeamID', 'Seed':'LSeed'}),
    on=['Season', 'LTeamID'],
    suffixes=('_W', '_L')
)

# 种子差与分差关系
tourney_results['SeedDiff'] = tourney_results['SeedNum_W'] - tourney_results['SeedNum_L']
sns.lmplot(data=tourney_results, x='SeedDiff', y='ScoreDiff', height=6)
plt.title('Seed Difference vs Score Difference')

