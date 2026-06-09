import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler


cluster = pd.read_parquet('/kaggle/input/static-cluster-result')


cluster.shape


cluster.describe()


cluster.columns


cluster['Cluster'].value_counts().sort_index()


user_cols = ['PerformanceTier', 'CumulativePoints', 'AvgPointsPerAchv',
             'DaysSinceLastAchv', 'DaysSinceSignup', 'CurrentRanking', 'HighestRanking',
             'TotalGold', 'TotalSilver', 'TotalBronze']


user_pivot = cluster.groupby('Cluster')[user_cols].mean().sort_values(by='Cluster', ascending=True)
user_pivot


data_cols = ['TotalDataset', 'TotalDatasetViews', 'TotalDatasetDownloads', 'TotalDatasetVotes', 'TotalDatasetTagCount', 'DatasetMedalCount',
             'DatasetPerDay']

data_pivot = cluster.groupby('Cluster')[data_cols].mean().sort_values(by='Cluster', ascending=True)
data_pivot


palette = sns.color_palette("Set2", n_colors=len(user_pivot))

# 지표별로 한 장씩 그리기
for col in data_cols:
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=data_pivot.index,
        y=data_pivot[col].values,
        palette=palette
    )
    plt.title(col, fontsize=11)
    plt.xlabel('Cluster')
    plt.ylabel('Mean')
    plt.tight_layout()
    plt.show()


comp_cols = ['TotalCompetitions', 'AvgSubmissionPerComp', 'CompMedalCount', 'CompPerDay']

comp_pivot = cluster.groupby('Cluster')[comp_cols].mean().sort_values(by='Cluster', ascending=True)
comp_pivot


palette = sns.color_palette("Set2", n_colors=len(user_pivot))

# 지표별로 한 장씩 그리기
for col in comp_cols:
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=comp_pivot.index,
        y=comp_pivot[col].values,
        palette=palette
    )
    plt.title(col, fontsize=11)
    plt.xlabel('Cluster')
    plt.ylabel('Mean')
    plt.tight_layout()
    plt.show()


kernel_cols = ['TotalKernels', 'TotalKernelVotes', 'TotalKernelComments', 'KernelMedalCount', 'KernelPerDay']

kernel_pivot = cluster.groupby('Cluster')[kernel_cols].mean().sort_values(by='Cluster', ascending=True)
kernel_pivot


palette = sns.color_palette("Set2", n_colors=len(user_pivot))

# 지표별로 한 장씩 그리기
for col in kernel_cols:
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=kernel_pivot.index,
        y=kernel_pivot[col].values,
        palette=palette
    )
    plt.title(col, fontsize=11)
    plt.xlabel('Cluster')
    plt.ylabel('Mean')
    plt.tight_layout()
    plt.show()


forum_cols = ['TotalForumPosts', 'TotalForumViews', 'TotalForumScore', 'ForumMedalCount', 'ForumPerDay']

forum_pivot = cluster.groupby('Cluster')[forum_cols].mean().sort_values(by='Cluster', ascending=True)
forum_pivot


palette = sns.color_palette("Set2", n_colors=len(user_pivot))

# 지표별로 한 장씩 그리기
for col in forum_cols:
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=forum_pivot.index,
        y=forum_pivot[col].values,
        palette=palette
    )
    plt.title(col, fontsize=11)
    plt.xlabel('Cluster')
    plt.ylabel('Mean')
    plt.tight_layout()
    plt.show()


# MedalRate 지표만 추출
focus_cols = ['DatasetPerDay', 'CompPerDay', 'KernelPerDay', 'ForumPerDay']

# 클러스터별 평균
focus_pivot = (
    cluster
      .groupby('Cluster')[focus_cols]
      .mean()
      .sort_index()
)

# 비중 계산 (row-wise normalize)
focus_ratio = focus_pivot.div(focus_pivot.sum(axis=1), axis=0)

# Set2 팔레트 (활동 항목 수만큼)
palette = sns.color_palette("PuBu", n_colors=len(focus_cols))

# 스택차트
plt.figure(figsize=(7,5))
bottom = None

for i, col in enumerate(focus_cols):
    plt.bar(focus_ratio.index, focus_ratio[col], 
            bottom=bottom, 
            label=col, 
            color=palette[i])
    if bottom is None:
        bottom = focus_ratio[col]
    else:
        bottom += focus_ratio[col]

plt.ylabel('Activity Proportion')
plt.xlabel('Cluster')
plt.title('Cluster Activity Composition (PerDay)')
plt.legend(loc='center left', bbox_to_anchor=(1,0.9))
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.xticks(focus_ratio.index)
plt.show()



# MedalRate 지표만 추출
focus_cols = ['CompMedalRate', 'KernelMedalRate', 'ForumMedalRate', 'DatasetMedalRate']

# 클러스터별 평균
focus_pivot = (
    cluster
      .groupby('Cluster')[focus_cols]
      .mean()
      .sort_index()
)

# Set2 팔레트 (클러스터 수만큼)
palette = sns.color_palette("Set2", n_colors=focus_pivot.shape[0])

# 라인그래프
plt.figure(figsize=(14, 4))
for i, (cluster_idx, row) in enumerate(focus_pivot.iterrows()):
    plt.plot(focus_cols, row.values, marker='o', label=f'Cluster {cluster_idx}', color=palette[i])

plt.ylabel('Medal Rate Mean')
plt.grid(True, linestyle='--', alpha=0.4 )
plt.tight_layout()
plt.show()



focus_pivot

