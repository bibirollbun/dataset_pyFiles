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


import pandas as pd  
  
# データの読み込み  
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')  
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')  
sub = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')  

#日付情報の追加
train["datetime"] = pd.to_datetime(train["datetime"])
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour

test["datetime"] = pd.to_datetime(test["datetime"])
test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour


#datetimeを削除
train = train.drop("datetime", axis = 1)

# データの型を確認  
print("Train Data Types:\n", train.dtypes)  
print("Test Data Types:\n", test.dtypes)  
  
# 欠損値の確認  
print("Missing Values in Train Data:\n", train.isnull().sum())  
print("Missing Values in Test Data:\n", test.isnull().sum())  
  
# 基礎集計の実行  
summary = train.describe()  
  
# 相関行列の計算  
correlation_matrix = train.corr()  
  
# 結果の表示  
print("Summary Statistics:\n", summary)  
print("Correlation Matrix:\n", correlation_matrix)


import pandas as pd  
import matplotlib.pyplot as plt  
  
# 'train'は既に定義されているデータフレームと仮定します  
# 基礎集計  
summary = train.describe()  
  
# 可視化  
plt.figure(figsize=(12, 8))  
  
# 気温のヒストグラム  
plt.subplot(2, 2, 1)  
plt.hist(train['temp'], bins='auto', color='yellow', alpha=0.7)  
plt.title('Temperature Distribution')  
plt.xlabel('Temperature')  
plt.ylabel('Frequency')  
plt.grid(axis='y', alpha=0.75)  
  
# 湿度のヒストグラム  
plt.subplot(2, 2, 2)  
plt.hist(train['humidity'], bins='auto', color='blue', alpha=0.7)  
plt.title('Humidity Distribution')  
plt.xlabel('Humidity')  
plt.ylabel('Frequency')  
plt.grid(axis='y', alpha=0.75)  
  
# 登録ユーザーのヒストグラム  
plt.subplot(2, 2, 3)  
plt.hist(train['registered'], bins='auto', color='pink', alpha=0.7)  
plt.title('Registered Users Distribution')  
plt.xlabel('Registered Users')  
plt.ylabel('Frequency')  
plt.grid(axis='y', alpha=0.75)  
  
# 非登録ユーザーのヒストグラム  
plt.subplot(2, 2, 4)  
plt.hist(train['casual'], bins='auto', color='green', alpha=0.7)  
plt.title('Casual Users Distribution')  
plt.xlabel('Casual Users')  
plt.ylabel('Frequency')  
plt.grid(axis='y', alpha=0.75)  
  
plt.tight_layout()  
plt.show()  
  
# 結果の表示  
print(summary)  


import pandas as pd    
import matplotlib.pyplot as plt    
    
# 'train'は既に定義されているデータフレームと仮定します    
# 'hour' 列でグループ化し、数値列の平均を計算    
hourly_counts = train.groupby('hour')[['casual', 'registered']].mean()    
    
# スタック棒グラフを描画    
fig, ax = plt.subplots(figsize=(12, 8))    
hourly_counts.plot(    
    kind='bar',    
    stacked=True,    
    ax=ax,    
    color=['lightcoral', 'royalblue'],  # 色を変更    
    alpha=0.8    
)    
ax.set_title('Average User Counts by Hour', fontsize=16)  # タイトルのフォントサイズを変更    
ax.set_xlabel('Hour of the Day', fontsize=14)  # X軸ラベルのフォントサイズを変更    
ax.set_ylabel('Average User Count', fontsize=14)  # Y軸ラベルのフォントサイズを変更    
ax.legend(['Casual Users', 'Registered Users'], fontsize=12)  # 凡例のフォントサイズを変更    
ax.grid(axis='y', linestyle='--', alpha=0.7)  # Y軸にグリッドを追加    
plt.xticks(rotation=0)  # X軸のラベルを水平に    
plt.tight_layout()  # レイアウトを調整    
plt.show() 


import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
  
# 天気と人数のクロス集計  
weather_cross = train.groupby('weather')[['casual', 'registered']].sum()  
  
# 季節と人数のクロス集計  
season_cross = train.groupby('season')[['casual', 'registered']].sum()  
  
# 天気と人数のクロス集計を可視化  
plt.figure(figsize=(12, 8))  
weather_cross.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], alpha=0.8)  
plt.title('User Counts by Weather', fontsize=16)  
plt.xlabel('Weather', fontsize=14)  
plt.ylabel('User Count', fontsize=14)  
plt.xticks(rotation=0)  # X軸ラベルを水平に  
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Y軸にグリッドを追加  
plt.legend(['Casual Users', 'Registered Users'], fontsize=12)  
plt.tight_layout()  # レイアウトを調整  
plt.show()  
  
# 季節と人数のクロス集計を可視化  
plt.figure(figsize=(12, 8))  
season_cross.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], alpha=0.8)  
plt.title('User Counts by Season', fontsize=16)  
plt.xlabel('Season', fontsize=14)  
plt.ylabel('User Count', fontsize=14)  
plt.xticks(rotation=0)  # X軸ラベルを水平に  
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Y軸にグリッドを追加  
plt.legend(['Casual Users', 'Registered Users'], fontsize=12)  
plt.tight_layout()  # レイアウトを調整  
plt.show()  
  
# クロス集計結果の表示  
print("Weather and User Counts Cross Tabulation:")  
print(weather_cross)  
print("\nSeason and User Counts Cross Tabulation:")  
print(season_cross) 


import seaborn as sns
# データセットの最初の数行を表示
print(train.head())

# データセットの概要統計量を表示
print(train.describe())

# 欠損値の確認
print(train.isnull().sum())

# 目的変数（count）の分布を表示
plt.figure(figsize=(10, 6))
sns.histplot(train['count'], kde=True)
plt.title('Distribution of Count')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.show()

"""
# 時間ごとのcountの分布を表示
plt.figure(figsize=(15, 8))
train['datetime'] = pd.to_datetime(train['datetime'])
train.set_index('datetime')['count'].plot()
plt.title('Count Over Time')
plt.xlabel('DateTime')
plt.ylabel('Count')
plt.show()
"""

# カテゴリカル変数とcountの関係を確認するためのボックスプロット
plt.figure(figsize=(15, 8))
sns.boxplot(x='season', y='count', data=train)
plt.title('Box Plot of Count by Season')
plt.xlabel('Season')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(15, 8))
sns.boxplot(x='weather', y='count', data=train)
plt.title('Box Plot of Count by Weather')
plt.xlabel('Weather')
plt.ylabel('Count')
plt.show()

# 数値変数とcountの関係を確認するための散布図
plt.figure(figsize=(15, 8))
sns.scatterplot(x='temp', y='count', data=train)
plt.title('Scatter Plot of Count by Temperature')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(15, 8))
sns.scatterplot(x='humidity', y='count', data=train)
plt.title('Scatter Plot of Count by Humidity')
plt.xlabel('Humidity')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(15, 8))
sns.scatterplot(x='windspeed', y='count', data=train)
plt.title('Scatter Plot of Count by Windspeed')
plt.xlabel('Windspeed')
plt.ylabel('Count')
plt.show()

# 相関行列の表示
corr_matrix = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()



pivot_table = train.pivot_table(values='count', index='season', aggfunc='mean')
print(pivot_table)


#import pandas as pd
#import numpy as np
from sklearn.ensemble import RandomForestRegressor

test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour

# 特徴量と目的変数の選択
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'day', 'hour']
X_train = train[features]
y_train = train['count']
X_test = test[features]


# 初期化とトレーニング
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



# テストデータで予測
test['count'] = model.predict(X_test)

# 提出ファイルの作成
submission = test[['datetime', 'count']]
submission.to_csv('submission.csv', index=False)

#とりあえず提出
pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')  


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 特徴量と目的変数の選択
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'day', 'hour']
X = train[features]
y = train['count']

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#初期化とトレーニング
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



# テストデータで予測
y_pred = model.predict(X_test)

# 評価指標の計算
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')


""""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# 特徴量と目的変数の選択
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'day', 'hour', 'day_of_week', 'is_holiday_eve', 'temp_diff', 'monthly_avg']
X_train = train[features]
y_train_casual = train['casual']
y_train_registered = train['registered']

# データの分割（casualユーザー）
X_train_casual_split, X_val_casual_split, y_train_casual_split, y_val_casual_split = train_test_split(X_train, y_train_casual, test_size=0.2, random_state=42)

# ランダムフォレストモデルの構築（casualユーザー）
model_casual_split = RandomForestRegressor(n_estimators=100, random_state=42)
model_casual_split.fit(X_train_casual_split, y_train_casual_split)

# 予測と評価（casualユーザー）
y_pred_casual_split = model_casual_split.predict(X_val_casual_split)
rmse_casual_split = np.sqrt(mean_squared_error(y_val_casual_split, y_pred_casual_split))
print(f'Root Mean Squared Error (Casual Users - Train-Test Split): {rmse_casual_split}')

# データの分割（registeredユーザー）
X_train_registered_split, X_val_registered_split, y_train_registered_split, y_val_registered_split = train_test_split(X_train, y_train_registered, test_size=0.2, random_state=42)

# ランダムフォレストモデルの構築（registeredユーザー）
model_registered_split = RandomForestRegressor(n_estimators=100, random_state=42)
model_registered_split.fit(X_train_registered_split, y_train_registered_split)

# 予測と評価（registeredユーザー）
y_pred_registered_split = model_registered_split.predict(X_val_registered_split)
rmse_registered_split = np.sqrt(mean_squared_error(y_val_registered_split, y_pred_registered_split))
print(f'Root Mean Squared Error (Registered Users - Train-Test Split): {rmse_registered_split}')
"""



import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error

#
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    df["temp_squared"] = df["temp"] ** 2
    df["humidity_squared"] = df["humidity"] ** 2
    df["hour_workingday"] = df["hour"] * df["workingday"]
    df.drop("datetime", axis=1, inplace=True)
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# 目的変数の対数変換（RMSLE対策）
train["count"] = np.log1p(train["count"])

# 特徴量と目的変数の分離
X = train.drop(["count", "casual", "registered"], axis=1)
y = train["count"]

# 学習・検証データに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル定義
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, random_state=42)

# クロスバリデーション
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_score = np.sqrt(-cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_log_error').mean())
xgb_cv_score = np.sqrt(-cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_log_error').mean())

print(f"Random Forest CV RMSLE: {rf_cv_score:.4f}")
print(f"XGBoost CV RMSLE: {xgb_cv_score:.4f}")

# モデル学習
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# 検証データで予測
rf_preds_val = np.expm1(rf_model.predict(X_val))
xgb_preds_val = np.expm1(xgb_model.predict(X_val))
y_val_exp = np.expm1(y_val)

# RMSLE評価
rf_rmsle = np.sqrt(mean_squared_log_error(y_val_exp, rf_preds_val))
xgb_rmsle = np.sqrt(mean_squared_log_error(y_val_exp, xgb_preds_val))

print(f"Random Forest Validation RMSLE: {rf_rmsle:.4f}")
print(f"XGBoost Validation RMSLE: {xgb_rmsle:.4f}")

# テストデータで予測し、平均をとる
rf_test_preds = np.expm1(rf_model.predict(test))
xgb_test_preds = np.expm1(xgb_model.predict(test))
submission["count"] = (rf_test_preds + xgb_test_preds) / 2

# 提出ファイル保存
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv が作成されました！")



def feature_engineering(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    df["temp_squared"] = df["temp"] ** 2
    df["humidity_squared"] = df["humidity"] ** 2
    df["hour_workingday"] = df["hour"] * df["workingday"]
    df.drop("datetime", axis=1, inplace=True)
    return df


