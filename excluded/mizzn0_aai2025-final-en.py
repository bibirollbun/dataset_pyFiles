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

# わかりにくいので修正
train = train.rename(columns={'Nitrogen': 'N'})
test = test.rename(columns={'Nitrogen': 'N'})
train = train.rename(columns={'Potassium': 'K'})
test = test.rename(columns={'Potassium': 'K'})
train = train.rename(columns={'Phosphorous': 'P'})
test = test.rename(columns={'Phosphorous': 'P'})

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




from sklearn.preprocessing   import LabelEncoder
# 目的変数もラベルエンコーディング
le_y = LabelEncoder()
train_y = le_y.fit_transform(train_y)


from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# # SoilとCropの組み合わせ　x
# train_x['Soil_Crop'] = train_x['Soil Type'] + '_' + train_x['Crop Type']
# test_x['Soil_Crop'] = test_x['Soil Type'] + '_' + test_x['Crop Type']

# をone-hot x
# sc_cols = pd.get_dummies(train_x['Soil_Crop'], prefix='Soil_Crop').columns
# sc_dummies = pd.get_dummies(train_x['Soil_Crop'], prefix='Soil_Crop').astype(int)
# train_x = pd.concat([train_x.drop(columns=['Soil_Crop']), sc_dummies], axis=1)
# test_sc = pd.get_dummies(test_x['Soil_Crop'], prefix='Soil_Crop').astype(int)
# test_sc = test_sc.reindex(columns=sc_cols, fill_value=0)
# test_x = pd.concat([test_x.drop(columns=['Soil_Crop']), test_sc], axis=1)

# Soilをone-hotに
soil_cols = pd.get_dummies(train_x['Soil Type'], prefix='Soil').columns
soil_dummies = pd.get_dummies(train_x['Soil Type'], prefix='Soil').astype(int)
train_x = pd.concat([train_x, soil_dummies], axis=1)

test_soil = pd.get_dummies(test_x['Soil Type'], prefix='Soil').astype(int)
test_soil = test_soil.reindex(columns=soil_cols, fill_value=0)
test_x = pd.concat([test_x, test_soil], axis=1)
# train_x.head(10)

# Cropをone-hotに
crop_cols = pd.get_dummies(train_x['Crop Type'], prefix='Crop').columns
crop_dummies = pd.get_dummies(train_x['Crop Type'], prefix='Crop').astype(int)
train_x = pd.concat([train_x, crop_dummies], axis=1)

test_crop = pd.get_dummies(test_x['Crop Type'], prefix='Crop').astype(int)
test_crop = test_crop.reindex(columns=crop_cols, fill_value=0)
test_x = pd.concat([test_x, test_crop], axis=1)
# train_x.head(10)

# tem*hum x
# train_x['tem_hum'] = train_x['Temperature'] * train_x['Humidity']
# test_x['tem_hum'] = test_x['Temperature'] * test_x['Humidity']

# tem*mois 
# train_x['tem_moi'] = train_x['Temperature'] * train_x['Moisture']
# test_x['tem_moi'] = test_x['Temperature'] * test_x['Moisture']

# SoilとCropの組み合わせ
train_x['Soil_Crop'] = train_x['Soil Type'] + '_' + train_x['Crop Type']
test_x['Soil_Crop'] = test_x['Soil Type'] + '_' + test_x['Crop Type']


df = train_x.copy()
df['target'] = train_y
global_mean = df['target'].mean()

# Soilをtarget encoding　x
# df['Soil_TE'] = np.nan

# # StratifiedKFold で Out-of-Fold を作成 x
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# for train_idx, valid_idx in skf.split(df, df['target']):
#     # 訓練フォールドでカテゴリごとの平均を計算
#     mapping = df.iloc[train_idx].groupby('Soil Type')['target'].mean()
#     # 検証フォールドにマッピング
#     df.loc[valid_idx, 'Soil_TE'] = df.loc[valid_idx, 'Soil Type'].map(mapping)
# df['Soil_TE'] = df['Soil_TE'].fillna(global_mean)
# train_x['Soil_TE'] = df['Soil_TE']
# soil_mapping_full = df.groupby('Soil Type')['target'].mean()
# test_x['Soil_TE'] = test_x['Soil Type'].map(soil_mapping_full).fillna(global_mean)

# Cropをtarget encoding x
# df['Crop_TE'] = np.nan
# # StratifiedKFold で Out-of-Fold を作成
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# for train_idx, valid_idx in skf.split(df, df['target']):
#     # 訓練フォールドでカテゴリごとの平均を計算
#     mapping = df.iloc[train_idx].groupby('Crop Type')['target'].mean()
#     # 検証フォールドにマッピング
#     df.loc[valid_idx, 'Crop_TE'] = df.loc[valid_idx, 'Crop Type'].map(mapping)
# df['Crop_TE'] = df['Crop_TE'].fillna(global_mean)
# train_x['Crop_TE'] = df['Crop_TE']
# crop_mapping_full = df.groupby('Crop Type')['target'].mean()
# test_x['Crop_TE'] = test_x['Crop Type'].map(crop_mapping_full).fillna(global_mean)

# Soil_Cropをtarget encoding
df['SC_TE'] = np.nan
# StratifiedKFold で Out-of-Fold を作成
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_idx, valid_idx in skf.split(df, df['target']):
    # 訓練フォールドでカテゴリごとの平均を計算
    mapping = df.iloc[train_idx].groupby('Soil_Crop')['target'].mean()
    # 検証フォールドにマッピング
    df.loc[valid_idx, 'SC_TE'] = df.loc[valid_idx, 'Soil_Crop'].map(mapping)
df['SC_TE'] = df['SC_TE'].fillna(global_mean)
train_x['SC_TE'] = df['SC_TE']
sc_mapping_full = df.groupby('Soil_Crop')['target'].mean()
test_x['SC_TE'] = test_x['Soil_Crop'].map(sc_mapping_full).fillna(global_mean)


# 窒素リンカリウムの総量x
# train_x['NPK_sum'] = train_x['Nitrogen'] + train_x['Phosphorous'] + train_x['Potassium']
# test_x['NPK_sum'] = test_x['Nitrogen'] + test_x['Phosphorous'] + test_x['Potassium']

# 割合x
# train_x['N/sum'] = train_x['Nitrogen'] / train_x['NPK_sum']
# test_x['N/sum'] = test_x['Nitrogen'] / test_x['NPK_sum']

# train_x['P/sum'] = train_x['Phosphorous'] / train_x['NPK_sum']
# test_x['P/sum'] = test_x['Phosphorous'] / test_x['NPK_sum']

# train_x['K/sum'] = train_x['Potassium'] / train_x['NPK_sum']
# test_x['K/sum'] = test_x['Potassium'] / test_x['NPK_sum']

# train_x = train_x.drop(["Nitrogen", "Phosphorous", "Potassium"], axis = 1)
# test_x = test_x.drop(["Nitrogen", "Phosphorous", "Potassium"], axis = 1)

# moistをbin分けx
# bins = np.arange(train_x['Moisture'].min(), train_x['Moisture'].max() + 5, 5)
# train_x['Moist_bin'] = pd.cut(train_x['Moisture'], bins=bins, include_lowest=True)
# train_x['Moist_bin'] = train_x['Moist_bin'].cat.codes
# test_x['Moist_bin'] = pd.cut(test_x['Moisture'], bins=bins, include_lowest=True)
# test_x['Moist_bin'] = test_x['Moist_bin'].cat.codes

# moist_ohe = pd.get_dummies(train_x['Moist_bin'], prefix='Moist').astype(int)
# train_x = pd.concat([train_x, moist_ohe], axis=1)

# test 側で同じ bins/dummies を reindex して揃えるx
# test_ohe = pd.get_dummies(test_x['Moist_bin'], prefix='Moist')astype(int)
# test_ohe = test_ohe.reindex(columns=moist_ohe.columns, fill_value=0)
# test_x = pd.concat([test_x, test_ohe], axis=1)

# train_x = train_x.drop(columns=['Moisture', 'Moist_bin'])
# test_x  = test_x .drop(columns=['Moisture', 'Moist_bin'])

# bins = np.arange(train_x['Moisture'].min(), train_x['Moisture'].max() + 5, 5)
# train_x['Moist_bin'] = pd.cut(train_x['Moisture'], bins=bins, include_lowest=True)
# train_x['Moist_bin'] = train_x['Moist_bin'].cat.codes
# test_x['Moist_bin'] = pd.cut(test_x['Moisture'], bins=bins, include_lowest=True)
# test_x['Moist_bin'] = test_x['Moist_bin'].cat.codes

# train_x = train_x.drop(columns=['Moisture'])
# test_x  = test_x .drop(columns=['Moisture'])

# 食べる（使う）部分の分類
important_part_map = {
    'Wheat': 'fruit',
    'Paddy': 'fruit',
    'Barley': 'fruit',
    'Millets': 'fruit',
    'Maize': 'fruit',
    'Ground Nuts': 'seed',
    'Oil seeds': 'seed',
    'Pulses': 'seed',
    'Cotton': 'seed',
    'Sugarcane': 'stem',
    'Tobacco': 'leaf',
}
train_x['Important_Part'] = train_x['Crop Type'].map(important_part_map)
test_x ['Important_Part'] = test_x ['Crop Type'].map(important_part_map)

le_part = LabelEncoder()
train_x['Important_Part'] = le_part.fit_transform(train_x['Important_Part'])
test_x ['Important_Part'] = le_part.transform(test_x ['Important_Part'])

ip_cols = pd.get_dummies(train_x['Important_Part'], prefix='IP').columns
ip_dummies = pd.get_dummies(train_x['Important_Part'], prefix='IP').astype(int)
train_x = pd.concat([train_x, ip_dummies], axis=1)

test_ip = pd.get_dummies(test_x['Important_Part'], prefix='IP').astype(int)
test_ip = test_ip.reindex(columns=ip_cols, fill_value=0)
test_x = pd.concat([test_x, test_ip], axis=1)

train_x = train_x.drop(["Soil Type", "Crop Type", "Soil_Crop", 'Important_Part'], axis = 1)
test_x = test_x.drop(["Soil Type", "Crop Type", "Soil_Crop", 'Important_Part'], axis = 1)

# NPKだけでやってみる
# train_x = train_x.drop(['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',], axis = 1)
# test_x = test_x.drop(['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',], axis = 1)

# # 温度と湿度を8分位数でbinにしたグリッドをつくる
# # 8分位数で分割
probs = np.linspace(0, 1, 9)
temp_edges = train_x['Temperature'].quantile(probs).values
hum_edges = train_x['Humidity'].quantile(probs).values
tlabels = [f'T{i}' for i in range(1, 9)]
hlabels = [f'H{i}' for i in range(1, 9)]
# # binにする
train_x['Tem_bin'] = pd.cut(train_x['Temperature'], bins=temp_edges, labels=tlabels, include_lowest=True)
train_x['Hum_bin'] = pd.cut(train_x['Humidity'], bins=hum_edges, labels=hlabels, include_lowest=True)
test_x['Tem_bin'] = pd.cut(test_x['Temperature'], bins=temp_edges, labels=tlabels, include_lowest=True)
test_x['Hum_bin']  = pd.cut(test_x['Humidity'], bins=hum_edges,  labels=hlabels, include_lowest=True)

# くっつけ
train_x['grid'] = train_x['Tem_bin'].astype(str) + '_' + train_x['Hum_bin'].astype(str)
test_x ['grid'] = test_x ['Tem_bin'].astype(str) + '_' + test_x ['Hum_bin'].astype(str)

# one-hotにする
region_ohe = pd.get_dummies(train_x['grid'], prefix='grid').astype(int)
train_x = pd.concat([train_x, region_ohe], axis=1)

region_ohe_test = pd.get_dummies(test_x['grid'], prefix='grid').astype(int)
# test 側は train と同じ列順に揃える
region_ohe_test = region_ohe_test.reindex(columns=region_ohe.columns, fill_value=0)
test_x = pd.concat([test_x, region_ohe_test], axis=1)

train_x = train_x.drop(['Tem_bin', 'Hum_bin', 'grid'], axis = 1)
test_x = test_x.drop(['Tem_bin', 'Hum_bin', 'grid'], axis = 1)

# 上位10個
top_ten_columns = ['grid_T5_H7',
                  'grid_T4_H8',
                  'grid_T8_H8',
                  'grid_T6_H4',
                  'grid_T5_H2']
                  # 'grid_T8_H6',
                  # 'grid_T8_H3',
                  # 'grid_T5_H4',
                  # 'grid_T8_H5',
                  # 'grid_T6_H7']

all_grid = ['grid_T1_H1',
            'grid_T1_H2',
            'grid_T1_H3',
            'grid_T1_H4',
            'grid_T1_H5',
            'grid_T1_H6',
            'grid_T1_H7',
            'grid_T1_H8',
            'grid_T2_H1',
            'grid_T2_H2',
            'grid_T2_H3',
            'grid_T2_H4',
            'grid_T2_H5',
            'grid_T2_H6',
            'grid_T2_H7',
            'grid_T2_H8',
            'grid_T3_H1',
            'grid_T3_H2',
            'grid_T3_H3',
            'grid_T3_H4',
            'grid_T3_H5',
            'grid_T3_H6',
            'grid_T3_H7',
            'grid_T3_H8',
            'grid_T4_H1',
            'grid_T4_H2',
            'grid_T4_H3',
            'grid_T4_H4',
            'grid_T4_H5',
            'grid_T4_H6',
            'grid_T4_H7',
            'grid_T4_H8',
            'grid_T5_H1',
            'grid_T5_H2',
            'grid_T5_H3',
            'grid_T5_H4',
            'grid_T5_H5',
            'grid_T5_H6',
            'grid_T5_H7',
            'grid_T5_H8',
            'grid_T6_H1',
            'grid_T6_H2',
            'grid_T6_H3',
            'grid_T6_H4',
            'grid_T6_H5',
            'grid_T6_H6',
            'grid_T6_H7',
            'grid_T6_H8',
            'grid_T7_H1',
            'grid_T7_H2',
            'grid_T7_H3',
            'grid_T7_H4',
            'grid_T7_H5',
            'grid_T7_H6',
            'grid_T7_H7',
            'grid_T7_H8',
            'grid_T8_H1',
            'grid_T8_H2',
            'grid_T8_H3',
            'grid_T8_H4',
            'grid_T8_H5',
            'grid_T8_H6',
            'grid_T8_H7',
            'grid_T8_H8']

drop_grid = list(set(all_grid) - set(top_ten_columns))
print(drop_grid)

train_x = train_x.drop(drop_grid, axis = 1)
test_x = test_x.drop(drop_grid, axis = 1)

df2 = train_x.copy()
df3 = test_x.copy()
df2['Fertilizer Name'] = train_y 

# NKPを四分位でbin
probs = np.linspace(0, 1, 5)
n_edges = df2['N'].quantile(probs).values
k_edges = df2['K'].quantile(probs).values
p_edges = df2['P'].quantile(probs).values
n_labels = [f'N{i}' for i in range(1, 5)]
k_labels = [f'K{i}' for i in range(1, 5)]
p_labels = [f'P{i}' for i in range(1, 5)]

df2['N_bin'] = pd.cut(df2['N'], bins=n_edges, labels=n_labels, include_lowest=True)
df2['K_bin'] = pd.cut(df2['K'], bins=k_edges, labels=k_labels, include_lowest=True)
df2['P_bin'] = pd.cut(df2['P'], bins=p_edges, labels=p_labels, include_lowest=True)

df3['N_bin'] = pd.cut(df3['N'], bins=n_edges, labels=n_labels, include_lowest=True)
df3['K_bin'] = pd.cut(df3['K'], bins=k_edges, labels=k_labels, include_lowest=True)
df3['P_bin'] = pd.cut(df3['P'], bins=p_edges, labels=p_labels, include_lowest=True)

# 相対頻度にする
nk_rels = {}
pk_rels = {}
for fert in df2['Fertilizer Name'].unique():
    sub = df2[df2['Fertilizer Name'] == fert]
    
    nk_pivot = sub.groupby(['N_bin','K_bin'], observed=True).size().unstack(fill_value=0)
    nk_rel = nk_pivot.div(nk_pivot.sum(axis=1), axis=0)  # 各 N_bin 行で割合計算
    nk_rels[fert] = nk_rel

    
    pk_pivot = sub.groupby(['P_bin','K_bin'], observed=True).size().unstack(fill_value=0)
    pk_rel = pk_pivot.div(pk_pivot.sum(axis=1), axis=0)  # 各 N_bin 行で割合計算
    pk_rels[fert] = pk_rel
    
nk_mean_rel = sum(nk_rels.values()) / len(nk_rels)
pk_mean_rel = sum(pk_rels.values()) / len(pk_rels)

df2['GlobalNKRel'] = df2.apply(
    lambda r: nk_mean_rel.loc[r['N_bin'], r['K_bin']], axis=1)
df2['GlobalPKRel'] = df2.apply(
    lambda r: pk_mean_rel.loc[r['P_bin'], r['K_bin']], axis=1)

df3['GlobalNKRel'] = df3.apply(
    lambda r: nk_mean_rel.loc[r['N_bin'], r['K_bin']], axis=1)
df3['GlobalPKRel'] = df3.apply(
    lambda r: pk_mean_rel.loc[r['P_bin'], r['K_bin']], axis=1)


train_x['GlobalNKRel'] = df2['GlobalNKRel']
train_x['GlobalPKRel'] = df2['GlobalPKRel']
test_x['GlobalNKRel'] = df3['GlobalNKRel']
test_x['GlobalPKRel'] = df3['GlobalPKRel']

# train_x = train_x.drop(['N', 'K', 'P'], axis = 1)
# test_x = test_x.drop(['N', 'K', 'P'], axis = 1)

# 温度と湿度も
# probs = np.linspace(0, 1, 9)
# tem_edges = df2['Temperature'].quantile(probs).values
# hum_edges = df2['Humidity'].quantile(probs).values
# moi_edges = df2['Moisture'].quantile(probs).values
# t_labels = [f'T{i}' for i in range(1, 9)]
# h_labels = [f'H{i}' for i in range(1, 9)]
# m_labels = [f'M{i}' for i in range(1, 9)]

# df2['Tem_bin'] = pd.cut(df2['Temperature'], bins=tem_edges, labels=t_labels, include_lowest=True)
# df2['Hum_bin'] = pd.cut(df2['Humidity'], bins=hum_edges, labels=h_labels, include_lowest=True)
# df2['Moi_bin'] = pd.cut(df2['Moisture'], bins=moi_edges, labels=m_labels, include_lowest=True)

# df3['Tem_bin'] = pd.cut(df3['Temperature'], bins=tem_edges, labels=t_labels, include_lowest=True)
# df3['Hum_bin'] = pd.cut(df3['Humidity'], bins=hum_edges, labels=h_labels, include_lowest=True)
# df3['Moi_bin'] = pd.cut(df3['Moisture'], bins=moi_edges, labels=m_labels, include_lowest=True)

# 相対頻度にする
# th_rels = {}
# tm_rels = {}
# for fert in df2['Fertilizer Name'].unique():
#     sub = df2[df2['Fertilizer Name'] == fert]
    
#     th_pivot = sub.groupby(['Tem_bin','Hum_bin'], observed=True).size().unstack(fill_value=0)
#     th_rel = th_pivot.div(th_pivot.sum(axis=1), axis=0)  # 各 N_bin 行で割合計算
#     th_rels[fert] = th_rel

    
#     tm_pivot = sub.groupby(['Tem_bin','Moi_bin'], observed=True).size().unstack(fill_value=0)
#     tm_rel = tm_pivot.div(tm_pivot.sum(axis=1), axis=0)  # 各 N_bin 行で割合計算
#     tm_rels[fert] = tm_rel
    
# th_mean_rel = sum(th_rels.values()) / len(th_rels)
# tm_mean_rel = sum(tm_rels.values()) / len(tm_rels)

# df2['GlobalTHRel'] = df2.apply(
#     lambda r: th_mean_rel.loc[r['Tem_bin'], r['Hum_bin']], axis=1)
# df2['GlobalTMRel'] = df2.apply(
#     lambda r: tm_mean_rel.loc[r['Tem_bin'], r['Moi_bin']], axis=1)

# df3['GlobalTHRel'] = df3.apply(
#     lambda r: th_mean_rel.loc[r['Tem_bin'], r['Hum_bin']], axis=1)
# df3['GlobalTMRel'] = df3.apply(
#     lambda r: tm_mean_rel.loc[r['Tem_bin'], r['Moi_bin']], axis=1)


# train_x['GlobalTHRel'] = df2['GlobalTHRel']
# train_x['GlobalTMRel'] = df2['GlobalTMRel']
# test_x['GlobalTHRel'] = df3['GlobalTHRel']
# test_x['GlobalTMRel'] = df3['GlobalTMRel']





train_x.head(10)


test_x.head(10)


train_x.columns


test_x.columns


from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# 分割
X_tr, X_val, y_tr, y_val = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
)

# 一旦XGBoost
model = xgb.XGBClassifier(
    max_depth=8,          #デフォルト6
    learning_rate=0.02,    #デフォルト0.3
    n_estimators=1000,     #デフォルト100
    reg_alpha=1,          # L1 正則化
    reg_lambda=10,         # L2 正則化
    objective='multi:softprob',
    num_class=len(le_y.classes_),
    random_state=42,
    subsample=0.7,
    colsample_bytree=0.7,
    eval_metric='mlogloss',
    early_stopping_rounds=30
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# LGBM
# model = lgb.LGBMClassifier(
#     objective='multiclass',
#     num_class=le_y.classes_.shape[0],
#     learning_rate=0.1,
#     n_estimators=1000,
#     num_leaves=31,
# )
# model.fit(
#     X_tr, y_tr,
#     eval_set=[(X_val, y_val)],
#     eval_metric='multi_logloss',
#     callbacks=[lgb.early_stopping(stopping_rounds=50)],
# )

#  randomForest
# model = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=10,
#     random_state=42,
#     n_jobs=-1
# )
# model.fit(X_tr, y_tr)

# ロジスティック回帰
# model = LogisticRegression(
#     multi_class='ovr',
#     solver='saga',
#     C=1.0,
#     max_iter=1000,
#     n_jobs=-1
# )
# model.fit(X_tr, y_tr)


imp = pd.Series(model.feature_importances_, index=X_tr.columns)
imp.sort_values().plot(kind='barh', figsize=(6,10))
plt.title('Feature Importances (XGBoost)')
plt.tight_layout()
plt.show()


# from sklearn.feature_selection import mutual_info_classif

# mi = mutual_info_classif(X_tr, y_tr, discrete_features=True)
# pd.Series(mi, index=X_tr.columns).sort_values(ascending=False) \
#     .plot.barh(figsize=(6,10))
# plt.title('Mutual Information Feature Ranking')
# plt.tight_layout()
# plt.show()


# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.preprocessing       import StandardScaler

# # 数値特徴のみを標準化して LDA
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_tr)  # X_full: one-hot も含めた全特徴量
# lda = LinearDiscriminantAnalysis(n_components=1)
# lda.fit(X_scaled, y_tr)
# coefs = pd.Series(lda.coef_[0], index=X_tr.columns)

# coefs.abs().sort_values().plot(kind='barh', figsize=(6,10))
# plt.title('LDA Coefficients (Absolute)')
# plt.tight_layout()
# plt.show()



# train_x.columns


# import umap
# import matplotlib.pyplot as plt

# # 特徴量の選定
# feats = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium',
#        'Phosphorous', 'Soil_Black', 'Soil_Clayey', 'Soil_Loamy', 'Soil_Red',
#        'Soil_Sandy', 'Crop_Barley', 'Crop_Cotton', 'Crop_Ground Nuts',
#        'Crop_Maize', 'Crop_Millets', 'Crop_Oil seeds', 'Crop_Paddy',
#        'Crop_Pulses', 'Crop_Sugarcane', 'Crop_Tobacco', 'Crop_Wheat', 'SC_TE',
#        'IP_fruit', 'IP_leaf', 'IP_seed', 'IP_stem']
# X = train_x.sample(5000, random_state=42)[feats]
# y = train.loc[X.index, 'Fertilizer Name']

# # UMAP で埋め込み
# proj = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

# plt.figure(figsize=(8,6))
# plt.scatter(proj[:,0], proj[:,1], c=y.map({n:i for i,n in enumerate(y.unique())}),
#             cmap='tab10', s=10, alpha=0.7)
# plt.title('UMAP of Full Feature Set')
# plt.show()



# 各レコードごとのaverage precisionを計算する関数
K = 3
def apk(y_i_true, y_i_pred):
    # y_predがK以下の長さで、要素がすべて異なることが必要
    assert (len(y_i_pred) <= K)
    assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision

    return sum_precision / min(len(y_i_true), K)

def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])

print(le_y.classes_)

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

X_tr_proba = model.predict_proba(X_tr)
X_tr_top3_idx = np.argsort(X_tr_proba, axis=1)[:, -3:][:, ::-1]
y_tr_pred = X_tr_top3_idx.tolist() 
y_tr_true = [[lbl] for lbl in y_tr]
tr_score = mapk(y_tr_true, y_tr_pred)
print(f"Hold-out Train MAP@3: {tr_score:.5f}")

proba = model.predict_proba(X_val)
top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
y_pred = top3_idx.tolist() 
y_true = [[lbl] for lbl in y_val]
val_score = mapk(y_true, y_pred)
print(f"Hold-out Validation MAP@3: {val_score:.5f}")


proba  = model.predict_proba(test_x)
top3 = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
pred_labels = le_y.inverse_transform(top3.flatten()).reshape(top3.shape)

submission = pd.DataFrame({
    'id': test['id'],  # test.csv に id カラムがある想定
    'Fertilizer Name': [' '.join(row) for row in pred_labels]
})

submission.to_csv('submission.csv', index=False)
print("create csv")




