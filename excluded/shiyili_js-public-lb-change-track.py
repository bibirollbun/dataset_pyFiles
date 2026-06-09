import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os


csv_list = glob("/kaggle/input/jane-street-public-leaderboard/*.csv")

df_list = []

for x in csv_list:
    _df = pd.read_csv(x)
    _df['Timestamp'] = pd.to_datetime(
        x.split("/")[-1].split(".")[-2][-19:], 
        format="%Y-%m-%dT%H_%M_%S"
        )
    df_list.append(_df)

df_all = pd.concat(df_list)


df_all.head()


df_all = df_all.sort_values(by=['TeamName', 'Timestamp'])
df_all['RankChange'] = df_all.groupby(['TeamName'])['Rank'].diff()

fig, ax = plt.subplots()
ax.hist(df_all['RankChange'].dropna().values, bins=100)
ax.set_title("Jane Street Competition Rank Change")
ax.set_xlabel("Rank Change")
ax.set_ylabel("Counts")
plt.show()


%matplotlib inline 

df_all = df_all.sort_values(by=['TeamName', 'Timestamp'])

timestamps = df_all['Timestamp'].unique()

first_ts, second_ts = timestamps[-2], timestamps[-1]
df_first = df_all[df_all['Timestamp'] == first_ts][['TeamName', 'Rank']]
df_second = df_all[df_all['Timestamp'] == second_ts][['TeamName', 'Rank']]

df_first = df_first.rename(columns={'Rank': 'first_rank'})
df_second = df_second.rename(columns={'Rank': 'second_rank'})

df_merge = pd.merge(df_first, df_second, on='TeamName', how='inner')

plt.figure(figsize=(8, 6))
plt.scatter(df_merge['first_rank'], df_merge['second_rank'], alpha=0.7)
plt.xlabel(f"Rank at {first_ts.strftime('%Y-%m-%d %H:%M:%S')}")
plt.ylabel(f"Rank at {second_ts.strftime('%Y-%m-%d %H:%M:%S')}")
plt.title("Scatter Plot: Rank at First vs Second Timestamp")
plt.grid(True)
plt.show()


df_rank = df_merge.sort_values('second_rank').reset_index(drop=True)
df_rank = df_rank[df_rank['second_rank']<=100]
df_rank['rank_diff'] = -(df_rank['second_rank'] - df_rank['first_rank'])

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 22))  # 根据用户数调整高度
plt.barh(df_rank['TeamName'], df_rank['rank_diff'], color='skyblue')
plt.xlabel('Rank Difference (Second - First)')
plt.ylabel('Team Name')
plt.title('Rank Change between Two Timestamps')
plt.grid( linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()
plt.show()




