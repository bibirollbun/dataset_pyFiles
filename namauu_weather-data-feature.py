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


import time
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder


air_reserve = pd.read_csv("/kaggle/input/recruit-restaurant-visitor-forecasting/air_reserve.csv.zip").rename(columns={'air_store_id':'store_id'})
hpg_reserve = pd.read_csv("/kaggle/input/recruit-restaurant-visitor-forecasting/hpg_reserve.csv.zip").rename(columns={'hpg_store_id':'store_id'})
air_store = pd.read_csv("/kaggle/input/recruit-restaurant-visitor-forecasting/air_store_info.csv.zip").rename(columns={'air_store_id':'store_id'})
hpg_store = pd.read_csv("/kaggle/input/recruit-restaurant-visitor-forecasting/hpg_store_info.csv.zip").rename(columns={'hpg_store_id':'store_id'})
air_visit = pd.read_csv("/kaggle/input/recruit-restaurant-visitor-forecasting/air_visit_data.csv.zip").rename(columns={'air_store_id':'store_id'})
store_id_map = pd.read_csv('/kaggle/input/recruit-restaurant-visitor-forecasting/store_id_relation.csv.zip').set_index('hpg_store_id', drop=False)
date_info = pd.read_csv("/kaggle/input/recruit-restaurant-visitor-forecasting/date_info.csv.zip").rename(columns={'calendar_date': 'visit_date'}).drop('day_of_week',axis=1)
submission = pd.read_csv('/kaggle/input/recruit-restaurant-visitor-forecasting/sample_submission.csv.zip')
submission['visit_date'] = submission['id'].str[-10:]
submission['store_id'] = submission['id'].str[:-11]
air_reserve['visit_date'] = air_reserve['visit_datetime'].str[:10]# 2016-01-13の形式にする
air_reserve['reserve_date'] = air_reserve['reserve_datetime'].str[:10]# 2016-01-13の形式にする
air_reserve['dow'] = pd.to_datetime(air_reserve['visit_date']).dt.dayofweek# 曜日追加
air_visit['id'] = air_visit['store_id'] + '_' + air_visit['visit_date']# サブミッションと同じ形式（storeid_YYYY-MM-DD）に揃える。

# hpg_reserveのstore_id列にあって、store_id_mapのhpg_store_id列にあるデータの数
(hpg_reserve['store_id'].isin(store_id_map['hpg_store_id'])).sum(),len(store_id_map),len(hpg_reserve)



# ⑦ hpg 系 ID を air 系 ID に揃える
hpg_reserve['store_id'] = hpg_reserve['store_id'].map(store_id_map['air_store_id']).fillna(hpg_reserve['store_id'])
hpg_store['store_id'] = hpg_store['store_id'].map(store_id_map['air_store_id']).fillna(hpg_store['store_id'])
hpg_store.rename(columns={'hpg_genre_name':'air_genre_name',
                          'hpg_area_name':'air_area_name'},inplace=True)
# ⑧ 訪問データとサブミッションの結合
air_visit["train_test"]="train"
submission["train_test"]="test"
data = pd.concat([air_visit, submission]).copy()
data['dow'] = pd.to_datetime(data['visit_date']).dt.dayofweek# 曜日

# ⑩ エリア名の特徴量作成."Tokyo Minato Roppongi" → "Tokyo"
air_store['air_area_name0'] = air_store['air_area_name'].apply(lambda x: x.split(' ')[0])
air_store



# ⑪ カテゴリ変数の Label Encoding. 機械学習モデル（例えば LightGBM）に投入できるように整数化。
lbl = LabelEncoder()
air_store['air_genre_name'] = lbl.fit_transform(air_store['air_genre_name'])
air_store['air_area_name0'] = lbl.fit_transform(air_store['air_area_name0'])
# ⑬ 店舗情報と日付情報を結合
data = data.merge(air_store, on='store_id', how='left')

# ⑨ 日付情報に新しい休日フラグを作成。土日もしくはholiday_flg==1
date_info['holiday_flg2'] = pd.to_datetime(date_info['visit_date']).dt.dayofweek
date_info['holiday_flg2'] = ((date_info['holiday_flg2']>4) | (date_info['holiday_flg']==1)).astype(int)
data = data.merge(date_info[['visit_date','holiday_flg','holiday_flg2']],
                  on='visit_date', how='left')
data["date"] = pd.to_datetime(data["visit_date"])
data["year"]   = data["date"].dt.year
data["month"]  = data["date"].dt.month
data["day"]    = data["date"].dt.day
data = data.drop(columns="date")
# ⑫ 目的変数（訪問者数）のログ変換
data['visitors'] = np.log1p(data['visitors'])

print(data.shape,submission.shape,air_visit.shape)

data.head()


air_station_distances = pd.read_csv("/kaggle/input/rrv-weather-data/air_station_distances.csv")
air_store_info_with_nearest_active_station = pd.read_csv("/kaggle/input/rrv-weather-data/air_store_info_with_nearest_active_station.csv")
hpg_store_info_with_nearest_active_station = pd.read_csv("/kaggle/input/rrv-weather-data/hpg_store_info_with_nearest_active_station.csv")
feature_manifest = pd.read_csv("/kaggle/input/rrv-weather-data/feature_manifest.csv")
nearby_active_stations = pd.read_csv("/kaggle/input/rrv-weather-data/nearby_active_stations.csv")
weather_stations = pd.read_csv("/kaggle/input/rrv-weather-data/weather_stations.csv")


# data の store_id とair_store_info_with_nearest_active_station の air_store_id が一致する行を突き合わせ、一致した行の station_id を data 側へ付与
merged = data.merge(
    air_store_info_with_nearest_active_station[['air_store_id', 'station_id']],
    left_on='store_id',
    right_on='air_store_id',
    how='left'
)

# 取得した station_id を data に反映したい場合
data['station_id'] = merged['station_id']
data


# 天気データの読み出し
import pandas as pd
import os
from tqdm import tqdm

base_path = "/kaggle/input/rrv-weather-data/1-1-16_5-31-17_Weather/1-1-16_5-31-17_Weather"

# マージ結果をためる
merged_list = []
group_list = data.groupby("station_id")
for sid, group in tqdm(group_list,total=len(group_list)):
    csv_path = os.path.join(base_path, f"{sid}.csv")
    
    if not os.path.exists(csv_path):
        print(f"Missing file: {sid}")
        merged_list.append(group)  # そのまま保持
        continue
    
    # CSV 読み込み
    weather = pd.read_csv(csv_path)
    
    # 日付型を揃える（必須）
    group = group.copy()
    group["visit_date"] = pd.to_datetime(group["visit_date"])
    weather["calendar_date"] = pd.to_datetime(weather["calendar_date"])
    
    # station_id & 日付で内部結合
    merged = group.merge(
        weather,
        left_on="visit_date",
        right_on="calendar_date",
        how="left"
    )
    merged_list.append(merged)

# 結合
data_merged = pd.concat(merged_list, ignore_index=True)
data_merged


data_merged.info()


# snow系、deepest_snowfall、total_snowfallは削除
data_merged = data_merged.drop(columns=['deepest_snowfall', 'total_snowfall'])
data_merged.info()


# 相対的な日付を作成
min_date = data_merged["visit_date"].min()
data_merged["relative_day"] = (data_merged["visit_date"] - min_date).dt.days
data_merged


# 都道府県、市町村だけ取り出す
data_merged[["prefecture", "city"]] = data_merged["air_area_name"].str.split(" ", expand=True)[[0, 1]]
data_merged["prefecture"] = data_merged["prefecture"].str.lower().str.split("-").str[0].str.replace("ō", "o")

data_merged[["air_area_name","prefecture", "city","visit_date"]]


# # prefecture × city × visit_date でグループ化し、グループごとのデータ数のヒストグラム
# import matplotlib.pyplot as plt

# # グループごとのデータ数
# group_sizes = data_filled.groupby(["prefecture", "city", "visit_date"]).size()

# # ヒストグラム描画
# plt.figure(figsize=(8,5))
# plt.hist(group_sizes, bins=30, color='skyblue', edgecolor='black')
# plt.xlabel("Number of rows in group")
# plt.ylabel("Number of groups")
# plt.title("Distribution of rows per (prefecture, city, visit_date) group")
# plt.show()


# # グループ数が最大のグループを可視化したい。
# # グループごとのデータ数
# group_sizes = data_filled.groupby(["prefecture", "city", "visit_date"]).size()

# # 最大のグループを特定
# max_group = group_sizes.idxmax()  # tuple: (prefecture, city, visit_date)
# print("最大グループ:", max_group)
# print("行数:", group_sizes.max())

# # 最大グループのデータ抽出
# max_group_data = data_filled[
#     (data_filled["prefecture"] == max_group[0]) &
#     (data_filled["city"] == max_group[1]) &
#     (data_filled["visit_date"] == max_group[2])
# ]
# max_group_data[["visit_date","prefecture","city","store_id"]]


# # 各列のユニーク値
# for col in ["visit_date","prefecture","city","store_id"]:
#     unique_vals = max_group_data[col].unique()
#     print(f"{col} のユニーク値 ({len(unique_vals)} 個): {unique_vals}")


# import pandas as pd

# # 対象列
# cols = [
#     "avg_temperature", "high_temperature", "low_temperature",
#     "precipitation", "hours_sunlight", "solar_radiation",
#     "avg_wind_speed", "avg_vapor_pressure", "avg_local_pressure",
#     "avg_humidity", "avg_sea_pressure", "cloud_cover"
# ]

# # グループ化して集計
# grouped = data_filled.groupby(["visit_date","prefecture","city"]).agg(
#     {col: lambda x: x.notnull().sum() for col in cols}  # nullでない数
# )

# # グループサイズ（行数）を追加
# grouped["group_size"] = data_filled.groupby(["visit_date","prefecture","city"]).size()

# # インデックスを列に戻す
# grouped = grouped.reset_index()

# # 確認
# grouped




# # grouped データフレームを使用
# # チェック対象
# check_cols = cols + ["group_size"]

# # フィルタリング関数
# def row_filter(row):
#     # 各colについて、group_sizeと異なりかつ0でないかを判定
#     mask = (row[cols] != row["group_size"]) & (row[cols] != 0)
#     # 1列でもTrueなら行を残す
#     return mask.any()

# # フィルタリング
# filtered = grouped[grouped.apply(row_filter, axis=1)]

# filtered


# # prefecture × city × visit_date でグループ化し、そのグループ内の平均で気象データの欠損を補完
# data_filled = data_merged.copy()

# cols = [
#     "avg_temperature", "high_temperature", "low_temperature",
#     "precipitation", "hours_sunlight", "solar_radiation",
#     "avg_wind_speed", "avg_vapor_pressure", "avg_local_pressure",
#     "avg_humidity", "avg_sea_pressure", "cloud_cover"
# ]

# data_filled[cols] = data_filled.groupby(
#     ["prefecture", "city", "visit_date"]
# )[cols].transform(lambda x: x.fillna(x.mean()))
# data_filled.info()


import pandas as pd
import glob
import os

data_filled = data_merged.copy()
# 欠損補完したい列
cols = [
    "avg_temperature", "high_temperature", "low_temperature",
    "precipitation", "hours_sunlight", "solar_radiation",
    "avg_wind_speed", "avg_vapor_pressure", "avg_local_pressure",
    "avg_humidity", "avg_sea_pressure", "cloud_cover"
]

# 気象 CSV があるディレクトリ
weather_dir = "/kaggle/input/rrv-weather-data/1-1-16_5-31-17_Weather/1-1-16_5-31-17_Weather"

# 都道府県ごとに気象データをまとめる辞書
pref_weather_mean = {}

# すべての CSV ファイルを取得
csv_files = glob.glob(os.path.join(weather_dir, "*.csv"))

# --- 1) 都道府県ごとに関連 CSV を読み込み・日付平均を作る -------------------------

for prefecture in data_merged["prefecture"].unique():

    # ファイル名に都道府県名（小文字）を含むものをフィルタ
    pref_name_l = prefecture.lower().replace("_", "").replace("-", "")
    
    matched_files = [
        f for f in csv_files 
        if pref_name_l in os.path.basename(f).lower().split("__")[0].split("_")[0]
    ]

    if not matched_files:
        print(prefecture,"skip")
        continue

    dfs = []
    for f in matched_files:
        df_temp = pd.read_csv(f)
        
        # calendar_date を datetime 化
        df_temp["calendar_date"] = pd.to_datetime(df_temp["calendar_date"])
        
        # 必要な列だけ残す
        needed_cols = ["calendar_date"] + cols
        df_temp = df_temp[needed_cols]

        dfs.append(df_temp)

    # すべて結合 → 日付ごとに平均をとる
    df_pref_all = pd.concat(dfs, ignore_index=True)
    df_pref_mean = df_pref_all.groupby("calendar_date")[cols].mean().reset_index()

    pref_weather_mean[prefecture] = df_pref_mean





# 3:30かかる

# --- 2) data_merged の visit_date と prefecture に対応する平均値を lookup する -------
from tqdm import tqdm
# 補完
for idx, row in tqdm(data_filled.iterrows(),total=len(data_filled)):
    pref = row["prefecture"]
    date = row["visit_date"]

    # その都道府県の平均データがない場合はスキップ
    if pref not in pref_weather_mean:
        continue

    df_pref = pref_weather_mean[pref]  # calendar_date と cols を持つ平均表

    # 同日データを取得
    match = df_pref[df_pref["calendar_date"] == date]

    if match.empty:
        continue

    # 補填値を辞書化
    fill_values = match.iloc[0].to_dict()

    # 欠損部分だけ補完
    for c in cols:
        if pd.isna(data_filled.at[idx, c]):
            data_filled.at[idx, c] = fill_values[c]
    

# 完了
data_filled.info()



data_filled


data_filled = data_filled.drop(columns=["solar_radiation","calendar_date"])


data_filled.to_csv("/kaggle/working/dataset.csv", index=False)

