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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import math


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

train.head(10)
# test.head(10)


print(train.columns)
# print(test.columns)

# つづりが違うので修正
train = train.rename(columns={'Temparature': 'Temperature'})
test = test.rename(columns={'Temparature': 'Temperature'})

# 目的変数と分離 ついでにidも
train_x = train.drop(["Fertilizer Name", "id"], axis = 1)
train_y = train["Fertilizer Name"]
test_x = test.drop(["id"], axis = 1)

train_x.head(10)


# 何が入ってるのか
soil_types = train["Soil Type"].unique()
crop_types = train["Crop Type"].unique()
fertilizer_name = train_y.unique()
print(soil_types)
print(crop_types)
print(fertilizer_name)

# 一応確認だけ
soil_types = test["Soil Type"].unique()
crop_types = test["Crop Type"].unique()
print(soil_types)
print(crop_types)




train.describe()


test.describe()


# df = train.sample(frac=0.0001, random_state=42)

# sns.pairplot(
#     df,
#     vars=['Temperature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous'],
#     hue='Fertilizer Name',     # カテゴリ列を指定するだけで色分けしてくれる
#     palette='tab10',
#     corner=True,
#     plot_kws={'s':60, 'alpha':0.8}
# )
# plt.show()


# ヒストグラム
train_x.hist(bins=100, color="blue", grid=True, label='pandas')
plt.show()

# 欠損値
na_train = train_x.isnull().sum().loc[lambda s: s>0].sort_values(ascending=False)
print(f"\nTrain:\t{train_x.shape[0]} samples")
print(na_train)

na_test = test.isnull().sum().loc[lambda s: s>0].sort_values(ascending=False)
print(f"\nTest:\t{test.shape[0]} samples")
print(na_test)


target = "Fertilizer Name"


# 目的変数の割合
sns.countplot(x=target, data=train)


# soilの割合
sns.countplot(x="Soil Type", data=train)


# cropの割合
sns.countplot(x="Crop Type", data=train)


fertilizer_names = train_y.unique()
cmap = plt.colormaps['tab10']
print(fertilizer_names)

# Temparature

# 温度ごとに肥料名の出現頻度を計算
count_data = train.groupby(['Temperature', 'Fertilizer Name']).size().unstack(fill_value=0)

# # 重ねてプロット（Fertilizer Nameごとに色分け）
# count_data.plot(kind='bar', stacked=False, figsize=(10, 6), width=0.8)

# # ラベル設定
# plt.xlabel('Temperature')
# plt.ylabel('Count of Fertilizer Name')
# plt.title('Count of Fertilizer Name by Temperature')

# # 凡例設定
# plt.legend(title='Fertilizer Name')

# # グラフ表示
# plt.tight_layout()
# plt.show()

# プロット
plt.figure(figsize=(10, 6))

# 各肥料名ごとにプロット
for i, fertilizer in enumerate(count_data.columns):
    heights = count_data[fertilizer].values
    # 点をプロット
    plt.scatter(count_data.index, heights, label=fertilizer, color=f"C{i}", s=50)
    # 点を線でつなげる
    plt.plot(count_data.index, heights, color=f"C{i}", zorder=5)

# ラベル設定
plt.xlabel('Temperature')
plt.ylabel('Count of Fertilizer Name')
plt.title('Count of Fertilizer Name by Temperature')

# 凡例設定
plt.legend(title='Fertilizer Name')

# グラフ表示
plt.tight_layout()
plt.show()





# Humidity
count_data = train.groupby(['Humidity', 'Fertilizer Name']).size().unstack(fill_value=0)

# プロット
plt.figure(figsize=(10, 6))

# 各肥料名ごとにプロット
for i, fertilizer in enumerate(count_data.columns):
    heights = count_data[fertilizer].values
    # 点をプロット
    plt.scatter(count_data.index, heights, label=fertilizer, color=f"C{i}", s=50)
    # 点を線でつなげる
    plt.plot(count_data.index, heights, color=f"C{i}", zorder=5)

# ラベル設定
plt.xlabel('Humidity')
plt.ylabel('Count of Fertilizer Name')
plt.title('Count of Fertilizer Name by Humidity')

# 凡例設定
plt.legend(title='Fertilizer Name')

# グラフ表示
plt.tight_layout()
plt.show()


# temperatureとHumidityでみてみる

# まとめて表示
# fertilizers =train['Fertilizer Name'].unique()
# n = len(fertilizers)
# cols = 3
# rows = math.ceil(n/cols)

# fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4), 
#                          sharex=True, sharey=True)

# for ax, fert in zip(axes.flat, fertilizers):
#     # 該当肥料データを集計
#     sub = train[train['Fertilizer Name'] == fert]
#     pivot = (
#         sub
#         .groupby(['Temperature','Humidity'])
#         .size()
#         .unstack(fill_value=0)
#     )
#     sns.heatmap(pivot, ax=ax, cmap='YlOrRd', cbar=False)
#     ax.set_title(fert)
#     ax.set_xlabel('Humidity')
#     ax.set_ylabel('Temperature')

# # カラーバーを全体でひとつだけ描画
# cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# sns.heatmap(pivot, cmap='YlOrRd', cbar=True, ax=axes.flat[-1], cbar_ax=cax, 
#             cbar_kws={'label':'Count'})

# # 余ったサブプロットを消す
# for ax in axes.flat[n:]:
#     fig.delaxes(ax)

# plt.tight_layout(rect=[0,0,0.9,1])
# plt.show()

# 個別に表示
fertilizers = train['Fertilizer Name'].unique()

for fert in fertilizers:
    # 該当肥料データを抽出
    sub = train[train['Fertilizer Name'] == fert]
    
    # 温度×湿度ごとに出現回数を集計
    pivot = (
        sub
        .groupby(['Temperature','Humidity'])
        .size()
        .unstack(fill_value=0)
    )
    
    # 図を新規作成
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap='YlOrRd')
    plt.title(f'Heatmap of "{fert}" Counts\nby Temperature & Humidity')
    plt.xlabel('Humidity')
    plt.ylabel('Temperature')
    plt.tight_layout()
    plt.show()


fertilizers = train['Fertilizer Name'].unique()

labels = ['Q1','Q2','Q3','Q4']  
df = train.copy()
tem_edges = df['Temperature'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
df['Tem_bin'] = pd.cut(
    df['Temperature'],   # 元の750000行
    bins=tem_edges,          # 境界は5個なので4ビンに分かれる
    labels=labels,
    include_lowest=True
)

hum_edges = df['Humidity'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
df['Hum_bin'] = pd.cut(df['Humidity'], bins=hum_edges,
                          labels=labels, include_lowest=True)

# df['Hum_bin'] = pd.cut(df['Humidity'], bins=10)
# df['Tem_bin'] = pd.cut(df['Temperature'], bins=10)

for fert in fertilizers:
    # 該当肥料データを抽出
    sub = df[df['Fertilizer Name'] == fert]
    
    # 温度×湿度ごとに出現回数を集計
    pivot = (
        sub
        .groupby(['Tem_bin','Hum_bin'], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    
    # 図を新規作成
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap='YlOrRd')
    plt.title(f'Heatmap of "{fert}" Counts\nby Temperature & Humidity')
    plt.xlabel('Tem_bin')
    plt.ylabel('Hum_bin')
    plt.tight_layout()
    plt.show()


# fertilizers = train['Fertilizer Name'].unique()

# labels = ['Q1','Q2','Q3','Q4', 'Q5','Q6','Q7','Q8']  
# df = train.copy()
# tem_edges = df['Temperature'].quantile([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]).values
# df['Tem_bin'] = pd.cut(
#     df['Temperature'],   # 元の750000行
#     bins=tem_edges,          # 境界は5個なので4ビンに分かれる
#     labels=labels,
#     include_lowest=True
# )

# hum_edges = df['Humidity'].quantile([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]).values
# df['Hum_bin'] = pd.cut(df['Humidity'], bins=hum_edges,
#                           labels=labels, include_lowest=True)

# df['Hum_bin'] = pd.cut(df['Humidity'], bins=10)
# df['Tem_bin'] = pd.cut(df['Temperature'], bins=10)

# for fert in fertilizers:
#     # 該当肥料データを抽出
#     sub = df[df['Fertilizer Name'] == fert]
    
#     # 温度×湿度ごとに出現回数を集計
#     pivot = (
#         sub
#         .groupby(['Tem_bin','Hum_bin'], observed=True)
#         .size()
#         .unstack(fill_value=0)
#     )
    
#     # 図を新規作成
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(pivot, cmap='YlOrRd')
#     plt.title(f'Heatmap of "{fert}" Counts\nby Temperature & Humidity')
#     plt.xlabel('Tem_bin')
#     plt.ylabel('Hum_bin')
#     plt.tight_layout()
#     plt.show()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ここからビンニング以降のみ
df = train[['Temperature', 'Humidity', 'Fertilizer Name']].copy()

probs = np.linspace(0, 1, 9)
temp_edges = df['Temperature'].quantile(probs).values
hum_edges = df['Humidity'].quantile(probs).values
tlabels = [f'T{i}' for i in range(1, 9)]
hlabels = [f'H{i}' for i in range(1, 9)]

df['Tem_bin'] = pd.cut(df['Temperature'], bins=temp_edges, labels=tlabels, include_lowest=True)
df['Hum_bin'] = pd.cut(df['Humidity'], bins=hum_edges, labels=hlabels, include_lowest=True)

fertilizers = df['Fertilizer Name'].unique()
pivots = {}
for fert in fertilizers:
    sub = df[df['Fertilizer Name'] == fert]
    pivot = sub.groupby(['Tem_bin', 'Hum_bin'], observed=True).size().unstack(fill_value=0)
    pivots[fert] = pivot

all_values = np.concatenate([pivot.values.flatten() for pivot in pivots.values()])
vmin, vmax = all_values.min(), all_values.max()

for fert, pivot in pivots.items():
    rel = pivot.div(pivot.sum(axis=1), axis=0)
    plt.figure(figsize=(7, 6))
    sns.heatmap(rel, cmap='YlOrRd', vmin=0, vmax=rel.values.max(), annot=True, fmt='.4f',
                cbar_kws={'label': 'Proportion'})
    plt.title(f'Heatmap of "{fert}" Counts\nby Temperature & Humidity')
    plt.xlabel('Humidity bin')
    plt.ylabel('Temperature bin')
    plt.tight_layout()
    plt.show()




peaks = {}
for fert, rel in pivots.items():
    # rel: 各肥料の相対頻度ヒートマップ (DataFrame)
    idx = rel.values.argmax()
    i, j = np.unravel_index(idx, rel.shape)
    peaks[fert] = (rel.index[i], rel.columns[j], rel.values[i,j])
print(peaks)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pivots: {fert: pivot(count)} から rels: {fert: rel(proportion)} を作ってある前提
rels = {fert: pivot.div(pivot.sum(axis=1), axis=0)
        for fert, pivot in pivots.items()}

# 1) 平均相対頻度マップを作成
mean_rel = sum(rels.values()) / len(rels)

# 2) 差分ヒートマップを描画
for fert, rel in rels.items():
    diff = rel - mean_rel
    
    plt.figure(figsize=(7,6))
    sns.heatmap(
        diff,
        cmap='RdBu_r',       # 正負を赤青で見やすく
        center=0,            # 差分ゼロを中立色
        annot=True,
        fmt='.4f',
        cbar_kws={'label': 'Difference from Mean'}
    )
    plt.title(f'Difference Heatmap for "{fert}"\n(rel - mean_rel)')
    plt.xlabel('Humidity bin')
    plt.ylabel('Temperature bin')
    plt.tight_layout()
    plt.show()



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ここからビンニング以降のみ
df = train[['Nitrogen', 'Potassium', 'Fertilizer Name']].copy()

probs = np.linspace(0, 1, 5)
n_edges = df['Nitrogen'].quantile(probs).values
k_edges = df['Potassium'].quantile(probs).values
nlabels = [f'N{i}' for i in range(1, 5)]
klabels = [f'K{i}' for i in range(1, 5)]

df['N_bin'] = pd.cut(df['Nitrogen'], bins=n_edges, labels=nlabels, include_lowest=True)
df['K_bin'] = pd.cut(df['Potassium'], bins=k_edges, labels=klabels, include_lowest=True)

fertilizers = df['Fertilizer Name'].unique()
pivots = {}
for fert in fertilizers:
    sub = df[df['Fertilizer Name'] == fert]
    pivot = sub.groupby(['N_bin', 'K_bin'], observed=True).size().unstack(fill_value=0)
    pivots[fert] = pivot

all_values = np.concatenate([pivot.values.flatten() for pivot in pivots.values()])
vmin, vmax = all_values.min(), all_values.max()

for fert, pivot in pivots.items():
    rel = pivot.div(pivot.sum(axis=1), axis=0)
    plt.figure(figsize=(7, 6))
    sns.heatmap(rel, cmap='YlOrRd', vmin=0, vmax=rel.values.max(), annot=True, fmt='.4f',
                cbar_kws={'label': 'Proportion'})
    plt.title(f'Heatmap of "{fert}" Counts\nby Temperature & Humidity')
    plt.xlabel('N bin')
    plt.ylabel('Temperature bin')
    plt.tight_layout()
    plt.show()

peaks = {}
for fert, rel in pivots.items():
    # rel: 各肥料の相対頻度ヒートマップ (DataFrame)
    idx = rel.values.argmax()
    i, j = np.unravel_index(idx, rel.shape)
    peaks[fert] = (rel.index[i], rel.columns[j], rel.values[i,j])
print(peaks)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pivots: {fert: pivot(count)} から rels: {fert: rel(proportion)} を作ってある前提
rels = {fert: pivot.div(pivot.sum(axis=1), axis=0)
        for fert, pivot in pivots.items()}

# 1) 平均相対頻度マップを作成
mean_rel = sum(rels.values()) / len(rels)

# 2) 差分ヒートマップを描画
for fert, rel in rels.items():
    diff = rel - mean_rel
    
    plt.figure(figsize=(7,6))
    sns.heatmap(
        diff,
        cmap='RdBu_r',       # 正負を赤青で見やすく
        center=0,            # 差分ゼロを中立色
        annot=True,
        fmt='.4f',
        cbar_kws={'label': 'Difference from Mean'}
    )
    plt.title(f'Difference Heatmap for "{fert}"\n(rel - mean_rel)')
    plt.xlabel('K bin')
    plt.ylabel('N bin')
    plt.tight_layout()
    plt.show()



# Moisture
count_data = train.groupby(['Moisture', 'Fertilizer Name']).size().unstack(fill_value=0)

# プロット
plt.figure(figsize=(10, 6))

# 各肥料名ごとにプロット
for i, fertilizer in enumerate(count_data.columns):
    heights = count_data[fertilizer].values
    # 点をプロット
    plt.scatter(count_data.index, heights, label=fertilizer, color=f"C{i}", s=50)
    # 点を線でつなげる
    plt.plot(count_data.index, heights, color=f"C{i}", zorder=5)

# ラベル設定
plt.xlabel('Humidity')
plt.ylabel('Count of Fertilizer Name')
plt.title('Count of Fertilizer Name by Moisuture')

# 凡例設定
plt.legend(title='Fertilizer Name')

# グラフ表示
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ① Moisture をビン分け
#    例として 5％刻みのビンに分ける
bins = np.arange(train['Moisture'].min(), train['Moisture'].max() + 5, 5)
train['Moist_bin'] = pd.cut(train['Moisture'], bins=bins)

# ② ビンごと × 肥料名 でカウント集計
count_data = (
    train
    .groupby(['Moist_bin', 'Fertilizer Name'])
    .size()
    .unstack(fill_value=0)
)

# ③ 各ビンの「中点」を計算して X 軸に使う
#    IntervalIndex の mid 値を取り出す
bin_mid = count_data.index.map(lambda iv: iv.mid)

# ④ プロット
plt.figure(figsize=(12, 6))

for i, fert in enumerate(count_data.columns):
    y = count_data[fert].values
    color = f"C{i}"
    # 散布図＋線
    plt.scatter(bin_mid, y, label=fert, color=color, s=50, alpha=0.8)
    plt.plot(bin_mid, y, color=color, alpha=0.7)

# ⑤ 軸・タイトル・凡例
plt.xlabel('Moisture bin (midpoint)')
plt.ylabel('Count of Fertilizer Name')
plt.title('Count of Fertilizer Name by Moisture (binned)')
plt.xticks(bin_mid, [str(iv) for iv in count_data.index], rotation=45)
plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.03, 1), loc='upper left')
plt.tight_layout()
plt.show()


import seaborn as sns
df = train[['Moisture','Fertilizer Name']]
sns.boxplot(data=df, x='Fertilizer Name', y='Moisture')
plt.xticks(rotation=45)
plt.show()

sns.heatmap(count_data.T, cmap='YlGnBu', annot=False)


# # まずプロット用の DataFrame を作り直し
# pct_plot = pct.copy()
# pct_plot.index = bin_mid       # Index を数値の中央値に置き換え
# pct_plot.index.name = 'Moisture'

# # あとは普通にプロット
# ax = pct_plot.plot(
#     kind='line',
#     marker='o',
#     figsize=(12,6),
#     alpha=0.8
# )
# ax.set_xlabel('Moisture')
# ax.set_ylabel('Proportion of Fertilizer')
# ax.set_title('Proportion of Each Fertilizer by Moisture Bin')
# ax.legend(title='Fertilizer Name', bbox_to_anchor=(1.02,1), loc='upper left')
# plt.tight_layout()
# plt.show()



# Soil Type
ct = pd.crosstab(train['Fertilizer Name'], train['Soil Type'])

plt.figure(figsize=(8,6))
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Soil Type')
plt.ylabel('Fertilizer Name')
plt.title('Heatmap of Fertilizer vs Soil Type')
plt.tight_layout()
plt.show()




# Crop Type
ct = pd.crosstab(train['Fertilizer Name'], train['Crop Type'])

plt.figure(figsize=(8,6))
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Crop Type')
plt.ylabel('Fertilizer Name')
plt.title('Heatmap of Fertilizer vs Crop Type')
plt.tight_layout()
plt.show()


fertilizers = train['Fertilizer Name'].unique()

for fert in fertilizers:
    # 該当肥料データを抽出
    sub = train[train['Fertilizer Name'] == fert]
    
    # 温度×湿度ごとに出現回数を集計
    pivot = (
        sub
        .groupby(['Crop Type','Soil Type'])
        .size()
        .unstack(fill_value=0)
    )
    
    # 図を新規作成
    plt.figure(figsize=(8, 7))
    sns.heatmap(pivot, cmap='YlOrRd')
    plt.title(f'Heatmap of "{fert}" Counts\nby Temperature & Humidity')
    plt.xlabel('Crop Type')
    plt.ylabel('Soil Type')
    plt.tight_layout()
    plt.show()


# Nitrogen
count_data = train.groupby(['Nitrogen', 'Fertilizer Name']).size().unstack(fill_value=0)

# プロット
plt.figure(figsize=(10, 6))

# 各肥料名ごとにプロット
for i, fertilizer in enumerate(count_data.columns):
    heights = count_data[fertilizer].values
    # 点をプロット
    plt.scatter(count_data.index, heights, label=fertilizer, color=f"C{i}", s=50)
    # 点を線でつなげる
    plt.plot(count_data.index, heights, color=f"C{i}", zorder=5)

# ラベル設定
plt.xlabel('Nitrogen')
plt.ylabel('Count of Fertilizer Name')
plt.title('Count of Fertilizer Name by Nitrogen')

# 凡例設定
plt.legend(title='Fertilizer Name')

# グラフ表示
plt.tight_layout()
plt.show()


# Phosphorous
count_data = train.groupby(['Phosphorous', 'Fertilizer Name']).size().unstack(fill_value=0)

# プロット
plt.figure(figsize=(10, 6))

# 各肥料名ごとにプロット
for i, fertilizer in enumerate(count_data.columns):
    heights = count_data[fertilizer].values
    # 点をプロット
    plt.scatter(count_data.index, heights, label=fertilizer, color=f"C{i}", s=50)
    # 点を線でつなげる
    plt.plot(count_data.index, heights, color=f"C{i}", zorder=5)

# ラベル設定
plt.xlabel('Phosphorous')
plt.ylabel('Count of Fertilizer Name')
plt.title('Count of Fertilizer Name by Phosphorous')

# 凡例設定
plt.legend(title='Fertilizer Name')

# グラフ表示
plt.tight_layout()
plt.show()


# Potassium
count_data = train.groupby(['Potassium', 'Fertilizer Name']).size().unstack(fill_value=0)

# プロット
plt.figure(figsize=(10, 6))

# 各肥料名ごとにプロット
for i, fertilizer in enumerate(count_data.columns):
    heights = count_data[fertilizer].values
    # 点をプロット
    plt.scatter(count_data.index, heights, label=fertilizer, color=f"C{i}", s=50)
    # 点を線でつなげる
    plt.plot(count_data.index, heights, color=f"C{i}", zorder=5)

# ラベル設定
plt.xlabel('Potassium')
plt.ylabel('Count of Fertilizer Name')
plt.title('Count of Fertilizer Name by Potassium')

# 凡例設定
plt.legend(title='Fertilizer Name')

# グラフ表示
plt.tight_layout()
plt.show()


# Nitrogen vs Phosphorous vs Potassium

fertilizers = train['Fertilizer Name'].unique()

df = train.copy()
df['N_bin'] = pd.cut(df['Nitrogen'], bins=10)
df['P_bin'] = pd.cut(df['Phosphorous'], bins=10)
df['K_bin'] = pd.cut(df['Potassium'], bins=10)

for fert in fertilizers:
    # 該当肥料データを抽出
    sub = df[df['Fertilizer Name'] == fert]
    
    # 温度×湿度ごとに出現回数を集計
    pivot = (
        sub
        .groupby(['N_bin','P_bin'], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    
    # 図を新規作成
    plt.figure(figsize=(8, 7))
    sns.heatmap(pivot, cmap='YlOrRd')
    plt.title(f'Heatmap of "{fert}" Counts\nby Temperature & Humidity')
    plt.xlabel('N_bin')
    plt.ylabel('P_bin')
    plt.tight_layout()
    plt.show()



for fert in fertilizers:
    # 該当肥料データを抽出
    sub = df[df['Fertilizer Name'] == fert]
    
    # 温度×湿度ごとに出現回数を集計
    pivot = (
        sub
        .groupby(['K_bin','P_bin'], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    
    # 図を新規作成
    plt.figure(figsize=(8, 7))
    sns.heatmap(pivot, cmap='YlOrRd')
    plt.title(f'Heatmap of "{fert}" Counts\nby Temperature & Humidity')
    plt.xlabel('K_bin')
    plt.ylabel('P_bin')
    plt.tight_layout()
    plt.show()


for fert in fertilizers:
    # 該当肥料データを抽出
    sub = df[df['Fertilizer Name'] == fert]
    
    # 温度×湿度ごとに出現回数を集計
    pivot = (
        sub
        .groupby(['N_bin','K_bin'], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    
    # 図を新規作成
    plt.figure(figsize=(8, 7))
    sns.heatmap(pivot, cmap='YlOrRd')
    plt.title(f'Heatmap of "{fert}" Counts\nby Temperature & Humidity')
    plt.xlabel('N_bin')
    plt.ylabel('K_bin')
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import umap


# df = train.sample(frac=0.1, random_state=42)
# le = LabelEncoder()
# labels = le.fit_transform(df['Fertilizer Name'])
# X = df[['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']].values

# 標準化
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# PCAによる次元削減
# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X)

# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.7)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA of Features')
# cbar = plt.colorbar(ticks=range(len(le.classes_)))
# cbar.set_label('Fertilizer')
# cbar.set_ticks(range(len(le.classes_)))
# cbar.set_ticklabels(le.classes_)
# plt.tight_layout()
# plt.show()

# reducer = umap.UMAP(n_components=2, random_state=42)
# X_umap = reducer.fit_transform(X)

# # 4) プロット
# plt.figure(figsize=(8,6))
# scatter = plt.scatter(X_umap[:,0], X_umap[:,1],
#                       c=labels, cmap='tab10', alpha=0.7)
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')
# plt.title('UMAP Projection of NPK & Env Features')

# # カラーバーに肥料名を表示
# cbar = plt.colorbar(scatter, ticks=range(len(le.classes_)))
# cbar.set_ticklabels(le.classes_)
# cbar.set_label('Fertilizer Name')

# plt.tight_layout()
# plt.show()



# 1. 小麦かつ赤土でフィルタリング
df_sub = train[(train['Crop Type'] == 'Wheat') & (train['Soil Type'] == 'Red')]

# 2. 窒素の値を肥料ごとにカウント
# count_data = (
#     df_sub
#     .groupby(['Fertilizer Name', 'Nitrogen'])
#     .size()
#     .unstack(fill_value=0)
# )

# 3. プロット
# plt.figure(figsize=(12, 6))
# count_data.T.plot(kind='bar', width=0.8)
# plt.xlabel('Nitrogen Level')
# plt.ylabel('Count of Records')
# plt.title('Count of Nitrogen Levels by Fertilizer (Wheat & Red Soil)')
# plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.02, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# Nitrogen
count_data = df_sub.groupby(['Nitrogen', 'Fertilizer Name']).size().unstack(fill_value=0)

# プロット
plt.figure(figsize=(10, 6))

# 各肥料名ごとにプロット
for i, fertilizer in enumerate(count_data.columns):
    heights = count_data[fertilizer].values
    # 点をプロット
    plt.scatter(count_data.index, heights, label=fertilizer, color=f"C{i}", s=50)
    # 点を線でつなげる
    plt.plot(count_data.index, heights, color=f"C{i}", zorder=5)

# ラベル設定
plt.xlabel('Nitrogen')
plt.ylabel('Count of Records')
plt.title('Count of Nitrogen Levels by Fertilizer (Wheat & Red Soil)')

# 凡例設定
plt.legend(title='Fertilizer Name')

# グラフ表示
plt.tight_layout()
plt.show()

# Potassium
count_data = df_sub.groupby(['Potassium', 'Fertilizer Name']).size().unstack(fill_value=0)

# プロット
plt.figure(figsize=(10, 6))

# 各肥料名ごとにプロット
for i, fertilizer in enumerate(count_data.columns):
    heights = count_data[fertilizer].values
    # 点をプロット
    plt.scatter(count_data.index, heights, label=fertilizer, color=f"C{i}", s=50)
    # 点を線でつなげる
    plt.plot(count_data.index, heights, color=f"C{i}", zorder=5)

# ラベル設定
plt.xlabel('Potassium')
plt.ylabel('Count of Records')
plt.title('Count of Potassium Levels by Fertilizer (Wheat & Red Soil)')

# 凡例設定
plt.legend(title='Fertilizer Name')

# グラフ表示
plt.tight_layout()
plt.show()

# Phosphorous
count_data = df_sub.groupby(['Phosphorous', 'Fertilizer Name']).size().unstack(fill_value=0)

# プロット
plt.figure(figsize=(10, 6))

# 各肥料名ごとにプロット
for i, fertilizer in enumerate(count_data.columns):
    heights = count_data[fertilizer].values
    # 点をプロット
    plt.scatter(count_data.index, heights, label=fertilizer, color=f"C{i}", s=50)
    # 点を線でつなげる
    plt.plot(count_data.index, heights, color=f"C{i}", zorder=5)

# ラベル設定
plt.xlabel('Phosphorous')
plt.ylabel('Count of Records')
plt.title('Count of Phosphorous Levels by Fertilizer (Wheat & Red Soil)')

# 凡例設定
plt.legend(title='Fertilizer Name')

# グラフ表示
plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Wheat & Red Soil だけに絞った df_sub を前提に
plt.figure(figsize=(10,6))
sns.boxplot(
    data=df_sub,
    x='Fertilizer Name',
    y='Nitrogen',
    order=sorted(df_sub['Fertilizer Name'].unique())
)
plt.xticks(rotation=45)
plt.title('Distribution of Nitrogen by Fertilizer (Wheat & Red Soil)')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
for fert in df_sub['Fertilizer Name'].unique():
    sns.kdeplot(
        df_sub.loc[df_sub['Fertilizer Name']==fert, 'Nitrogen'],
        label=fert,
        fill=False,  # True にすると塗りつぶし
        alpha=0.7
    )
plt.legend(title='Fertilizer')
plt.title('KDE of Nitrogen by Fertilizer (Wheat & Red Soil)')
plt.show()


import pandas as pd
stats = df_sub.groupby('Fertilizer Name')['Nitrogen'].agg(['mean','std']).sort_values('mean')

plt.figure(figsize=(8,5))
plt.errorbar(
    x=stats.index, 
    y=stats['mean'], 
    yerr=stats['std'], 
    fmt='o', 
    capsize=5
)
plt.xticks(rotation=45)
plt.ylabel('Nitrogen')
plt.title('Mean±Std of Nitrogen by Fertilizer')
plt.tight_layout()
plt.show()


# 1) データの再読み込み（ファイル名・パスは適宜変更）
import pandas as pd
# train = pd.read_csv('train.csv')  

# 2) multivariate 可視化コード
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# a) ペアプロット（サンプリングして高速化）
df_sample = train[['Nitrogen','Phosphorous','Potassium','Fertilizer Name']] \
               .sample(n=2000, random_state=42)
sns.pairplot(
    df_sample,
    vars=['Nitrogen','Phosphorous','Potassium'],
    hue='Fertilizer Name',
    diag_kind='kde',
    plot_kws={'alpha':0.6, 's':30}
)
plt.suptitle('Pairplot of N, P, K by Fertilizer (sampled)', y=1.02)
plt.show()

# b) PCA で2次元投影
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(train[['Nitrogen','Phosphorous','Potassium']])

df_pca = pd.DataFrame(coords, columns=['PC1','PC2'], index=train.index)
df_pca['Fertilizer Name'] = train['Fertilizer Name']

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_pca.sample(n=2000, random_state=42),
    x='PC1', y='PC2',
    hue='Fertilizer Name',
    alpha=0.6,
    s=40
)
plt.title('PCA Projection of NPK Features')
plt.legend(title='Fertilizer', bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ① N, P, K とラベルを準備（サンプリングで高速化）
df_np = train[['Nitrogen','Phosphorous','Potassium','Fertilizer Name']].sample(5000, random_state=42)
X = df_np[['Nitrogen','Phosphorous','Potassium']].values
y = df_np['Fertilizer Name'].values

# ② LDA で 2 次元に削減
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# ③ 散布図をプロット
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_lda[:,0], y=X_lda[:,1],
    hue=y,
    palette='tab10',
    alpha=0.7,
    s=40
)
plt.title('LDA Projection of NPK (Wheat & Red Soil)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(title='Fertilizer', bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout()
plt.show()



from mpl_toolkits.mplot3d import Axes3D

df_np = train[['Nitrogen','Phosphorous','Potassium','Fertilizer Name']] \
            .sample(3000, random_state=42)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

for fert, grp in df_np.groupby('Fertilizer Name'):
    ax.scatter(
        grp['Nitrogen'], grp['Phosphorous'], grp['Potassium'],
        label=fert, alpha=0.6, s=20
    )

ax.set_xlabel('Nitrogen')
ax.set_ylabel('Phosphorous')
ax.set_zlabel('Potassium')
ax.set_title('3D Scatter of NPK by Fertilizer')
ax.legend(bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout()
plt.show()



from sklearn.preprocessing   import LabelEncoder

# 一旦無視ラベルエンコーディングだけしておく
le = LabelEncoder()
train_x['Soil Type'] = le.fit_transform(train_x['Soil Type'])
train_x['Crop Type'] = le.fit_transform(train_x['Crop Type'])
test_x['Soil Type'] = le.fit_transform(test_x['Soil Type'])
test_x['Crop Type'] = le.fit_transform(test_x['Crop Type'])

# Soilをone-hotに
# soil_cols = pd.get_dummies(train_x['Soil Type'], prefix='Soil').columns
# soil_dummies = pd.get_dummies(train_x['Soil Type'], prefix='Soil').astype(int)
# train_x = pd.concat([train_x.drop(columns=['Soil Type']), soil_dummies], axis=1)

# test_soil = pd.get_dummies(test_x['Soil Type'], prefix='Soil').astype(int)
# test_soil = test_soil.reindex(columns=soil_cols, fill_value=0)
# test_x = pd.concat([test_x.drop(columns=['Soil Type']), test_soil], axis=1)
# train_x.head(10)

# Cropをone-hotに
# crop_cols = pd.get_dummies(train_x['Crop Type'], prefix='Crop').columns
# crop_dummies = pd.get_dummies(train_x['Crop Type'], prefix='Crop').astype(int)
# train_x = pd.concat([train_x.drop(columns=['Crop Type']), crop_dummies], axis=1)
# test_crop = pd.get_dummies(test_x['Crop Type'], prefix='Crop').astype(int)
# test_crop = test_crop.reindex(columns=crop_cols, fill_value=0)
# test_x = pd.concat([test_x.drop(columns=['Crop Type']), test_crop], axis=1)
# train_x.head(10)




import umap
import matplotlib.pyplot as plt

# 特徴量の選定
feats = ['Nitrogen','Phosphorous','Potassium',
         'Temperature','Humidity','Moisture',
         'Soil_TE','Crop_TE','SC_TE']
X = train_x.sample(5000, random_state=42)[feats]
y = train.loc[X.index, 'Fertilizer Name']

# UMAP で埋め込み
proj = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(proj[:,0], proj[:,1], c=y.map({n:i for i,n in enumerate(y.unique())}),
            cmap='tab10', s=10, alpha=0.7)
plt.title('UMAP of Full Feature Set')
plt.show()



train_x.head(10)


test_x.head(10)


from sklearn.model_selection import train_test_split
import xgboost as xgb


# 分割
X_tr, X_val, y_tr, y_val = train_test_split(
    train_x, train_y, random_state=71
)

# 目的変数もラベルエンコーディング
le_y = LabelEncoder()
y_tr = le_y.fit_transform(y_tr)
y_val = le_y.transform(y_val)

# 一旦XGBoost
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le_y.classes_),
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_tr, y_tr)


# 各レコードごとのaverage precisionを計算する関数
# K = 3
# def apk(y_i_true, y_i_pred):
#     # y_predがK以下の長さで、要素がすべて異なることが必要
#     assert (len(y_i_pred) <= K)
#     assert (len(np.unique(y_i_pred)) == len(y_i_pred))

#     sum_precision = 0.0
#     num_hits = 0.0

#     for i, p in enumerate(y_i_pred):
#         if p in y_i_true:
#             num_hits += 1
#             precision = num_hits / (i + 1)
#             sum_precision += precision

#     return sum_precision / min(len(y_i_true), K)

# def mapk(y_true, y_pred):
#     return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])

# print(le_y.classes_)

# proba = model.predict_proba(X_val)
# print(proba[0])

# top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
# print(top3_idx[0])

# y_pred = top3_idx.tolist() 
# print(y_pred[0])

# y_true = [[lbl] for lbl in y_val]
# print(y_true[0])

# val_score = mapk(y_true, y_pred)
# print(f"Hold-out Validation MAP@3: {val_score:.5f}")

# X_tr_proba = model.predict_proba(X_tr)
# X_tr_top3_idx = np.argsort(X_tr_proba, axis=1)[:, -3:][:, ::-1]
# y_tr_pred = X_tr_top3_idx.tolist() 
# y_tr_true = [[lbl] for lbl in y_tr]
# tr_score = mapk(y_tr_true, y_tr_pred)
# print(f"Hold-out Train MAP@3: {tr_score:.5f}")

# proba = model.predict_proba(X_val)
# top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
# y_pred = top3_idx.tolist() 
# y_true = [[lbl] for lbl in y_val]
# val_score = mapk(y_true, y_pred)
# print(f"Hold-out Validation MAP@3: {val_score:.5f}")


# proba  = model.predict_proba(test_x)
# top3 = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
# pred_labels = le_y.inverse_transform(top3.flatten()).reshape(top3.shape)

# submission = pd.DataFrame({
#     'id': test['id'],  # test.csv に id カラムがある想定
#     'Fertilizer Name': [' '.join(row) for row in pred_labels]
# })

# submission.to_csv('submission.csv', index=False)
# print("create csv")

