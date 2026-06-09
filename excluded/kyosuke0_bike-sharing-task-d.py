import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------
# データ読み込み
# ----------------------------------------------
DATA_DIR = "/kaggle/input/bike-sharing-demand"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")


# ----------------------------------------------
# 前処理
# ----------------------------------------------
def create_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    
    # 気温差
    df['temp_diff'] = abs(df['temp'] - df['atemp'])

    #通勤時間帯（7〜9時・17〜19時）をフラグとして持たせる
    df['rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # 時間帯を数値として分類（hour_band）
    # 夜間(21〜23, 0〜4)は 3
    df['hour_band'] = -1
    # 早朝（5-8）
    df.loc[df['hour'].between(5, 8), 'hour_band'] = 0
    # 日中（9-16）
    df.loc[df['hour'].between(9, 16), 'hour_band'] = 1
    # 夕方（17-20）
    df.loc[df['hour'].between(17, 20), 'hour_band'] = 2
    # 夜間（21-23 or 0-4）
    df.loc[(df['hour'] >= 21) | (df['hour'] <= 4), 'hour_band'] = 3

    # 天気と時間帯の交互作用
    df['weather_x_hour_band'] = df['weather'].astype(str) + "_" + df['hour_band'].astype(str)
    df['weather_x_hour_band'] = df['weather_x_hour_band'].astype('category').cat.codes

    # 平日と時間帯の交互作用
    df['hour_x_weekday'] = df['hour'].astype(str) + "_" + df['weekday'].astype(str)
    df['hour_x_weekday'] = df['hour_x_weekday'].astype('category').cat.codes

    # 季節と時間帯の交互作用
    df['season_x_hour_band'] = df['season'].astype(str) + "_" + df['hour_band'].astype(str)
    df['season_x_hour_band'] = df['season_x_hour_band'].astype('category').cat.codes

    # 平日と天気の交互作用
    df['weather_x_weekday'] = df['weather'].astype(str) + "_" + df['weekday'].astype(str)
    df['weather_x_weekday'] = df['weather_x_weekday'].astype('category').cat.codes

    # 天候と気温のギャップ（異常気象検出）
    df['weather_temp_gap'] = abs(df['weather'] - df['temp'].round())

    return df

# 特徴量作成
train = create_features(train)
test = create_features(test)

# 対象カラムの選択
drop_cols = ['datetime', 'casual', 'registered', 'count']
feature_cols = [col for col in train.columns if col not in drop_cols]

X = train[feature_cols]
y = np.log1p(train['count'])  # log1p変換

# 検証用データ分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

sc_X = MinMaxScaler()
# スケーラー：trainのみfit。validとtestはtransformだけ
X_train = sc_X.fit_transform(X_train)
X_valid = sc_X.transform(X_valid)
X_submit = sc_X.transform(test[feature_cols])
print(feature_cols)


# ----------------------------------------------
# モデル訓練
# ----------------------------------------------
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# 検証データで予測：評価用
rf_prediction = rf.predict(X_valid)
preds_valid = np.expm1(np.maximum(0, rf_prediction))    # 逆log1pして0未満クリップ
y_valid_real = np.expm1(y_valid)

# 評価指標（RMSLE/Kaggle本番指標に最も近い）
rmsle = np.sqrt(metrics.mean_squared_log_error(y_valid_real, preds_valid))
print(f'RMSLE (Validation): {rmsle:.5f}')

# 可視化:
plt.scatter(y_valid_real, preds_valid, alpha=0.4)
plt.xlabel("Actual count")
plt.ylabel("Predicted count")
plt.title("Validation: Actual vs Predicted")
plt.show()

# ----------------------------------------------
# テストデータ（Kaggle提出用）で予測
# ----------------------------------------------
rf_test_pred = rf.predict(X_submit)
rf_test_pred = np.expm1(np.maximum(0, rf_test_pred))  # 逆log1p＋0未満は0

submission = pd.read_csv(f"{DATA_DIR}/sampleSubmission.csv")
submission["count"] = rf_test_pred
submission.to_csv("submission.csv", index=False)

print("提出ファイルが作成されました: submission.csv")

# ----------------------------------------------
# 特徴量重要度
# ----------------------------------------------
feature_importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("特徴量の重要度（降順）:")
print(feature_importance)

# 上位20個を表示（必要に応じて調整）
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.values[:20], y=feature_importance.index[:20], palette="viridis")

plt.title("Feature Importance (Top 20)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

