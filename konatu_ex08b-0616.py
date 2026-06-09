# -----------------------------------------------------------
# 学習データ，テストデータの読み込みと準備（このセルは変更不可）
# -----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 乱数シードがある関数を呼ぶときはこの値で統一する
rseed = 71

# 学習データ、テストデータの読み込み
train = pd.read_csv('../input/demand-forecasting-kernels-only/train.csv')
test = pd.read_csv('../input/demand-forecasting-kernels-only/test.csv')

# 学習データから目的変数を分離
train_x = train.drop(['sales'], axis=1)
train_y = train['sales']

# テストデータからid列を分離
test_x = test.drop(['id'], axis=1)
test_id = test['id'] # submit用に温存

print(train_x.isnull().sum())
print(test_x.isnull().sum())

# 学習データにおける各特徴量の要約統計量・ヒストグラム・欠損(True: 欠損値あり)
print(train_x.describe())
print(train_x.isnull().any())
train_x.hist(bins=100, color="blue", grid=True, label='pandas')
plt.show()

# テストデータにおける各特徴量の要約統計量・ヒストグラム・欠損(True: 欠損値あり)
print(test_x.describe())
print(test_x.isnull().any())
test_x.hist(bins=100, color="blue", grid=True, label='pandas')
plt.show()


# -----------------------------------------------------------
# (A) 変数 store の前処理
# -----------------------------------------------------------
col = 'store'

# ベースラインモデル：ラベルエンコーディング
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x[col] = le.transform(train_x[col].fillna(method='ffill'))
test_x[col] = le.transform(test_x[col].fillna(method='ffill'))


# -----------------------------------------------------------
# (B) 変数 item の前処理
# -----------------------------------------------------------
col = 'item'

# ベースライン：ラベルエンコーディング
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x[col] = le.transform(train_x[col].fillna(method='ffill'))
test_x[col] = le.transform(test_x[col].fillna(method='ffill'))


import pandas as pd
import numpy as np

# -----------------------------------------------------------
# (C) 変数 date の前処理を行う関数
# -----------------------------------------------------------
def extract_date_features(time_series):
    """
    日付や時刻情報を持つ Series から、機械学習で使える特徴量を抽出する。
    """
    
    # -----------------------------
    # 1) 文字列を datetime64 型に変換
    # -----------------------------
    # errors='coerce' は変換できない値を NaT (Not a Time) にします
    #dt = pd.to_datetime(time_series, errors='coerce')
    dt = time_series
    # -----------------------------
    # 2) 月 -> 季節ラベル(文字列) を返す内部関数
    #    ※季節の定義は要件に合わせて変更してください。
    #      (例: 6月は一般的に夏に含まれます)
    # -----------------------------
    def get_season_label(month):
        if month in [3, 4, 5, 6]: # 春: 3, 4, 5月
            return '春'
        elif month in [7, 8, 9]: # 夏: 6, 7, 8月
            return '夏'
        elif month in [9, 10, 11]: # 秋: 9, 10, 11月
            return '秋'
        else: # 冬: 12, 1, 2月
            return '冬'

    label_map = {'冬': 0, '春': 1, '夏': 2, '秋': 3}

    # -----------------------------
    # 3) .dt アクセサで各要素を抽出
    # -----------------------------
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    hour = dt.dt.hour
    dayofweek = dt.dt.dayofweek
    dayofyear = dt.dt.dayofyear
    
    season_label = dt.dt.month.apply(get_season_label)
    season_encoded = season_label.map(label_map)


    # -----------------------------
    # 4) 新機能：週末フラグと祝日フラグを追加
    # -----------------------------
    # 週末フラグ (土曜日=5, 日曜日=6)
    is_weekend = dt.dt.dayofweek.isin([5, 6]).astype(int)

    # -----------------------------
    # 5) 各種 sin/cos エンコーディング
    # -----------------------------
    # 【修正点】'day' の代わりに 'dayofyear' を使用
    day_sin = np.sin(2 * np.pi * dayofyear / 365.0)
    day_cos = np.cos(2 * np.pi * dayofyear / 365.0)

    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    dayofweek_sin = np.sin(2 * np.pi * dayofweek / 7.0)
    dayofweek_cos = np.cos(2 * np.pi * dayofweek / 7.0)


    # -----------------------------
    # 6) 結果を DataFrame にまとめて返却
    # -----------------------------
    return pd.DataFrame({
        'season_encoded': season_encoded,
        #'is_weekend': is_weekend,
        #'day_sin': day_sin,
        #'day_cos': day_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        #'hour_sin': hour_sin,
        #'hour_cos': hour_cos,
        # 必要に応じて有効化
        'dayofweek': dayofweek,
        'year': year,
        'month':month,
        'day':day,
        'dayofweek_sin':dayofweek_sin,
        'dayofweek_cos':dayofweek_cos,
        
    })

col = 'date' # 対象の列名

# 【修正点】元のDataFrameの日付型を変換
train[col] = pd.to_datetime(train[col])
test[col]  = pd.to_datetime(test[col])


# 関数を呼び出して特徴量を生成
train_features = extract_date_features(train[col])
test_features = extract_date_features(test[col])

# 元のDataFrameと結合
train_x = pd.concat([train_x.drop([col], axis=1), train_features], axis=1)
test_x = pd.concat([test_x.drop([col], axis=1), test_features], axis=1)


# 1. 特定の店舗における特定の曜日の平均販売数を訓練データから計算
tmp_x = pd.concat([train_x, train_y], axis=1)
store_day_avg_sales = tmp_x.groupby(['store', 'dayofweek'])['sales'].mean().reset_index()
store_day_avg_sales.rename(columns={'sales': 'avg_sales_store_day'}, inplace=True)

# 訓練データとテストデータに結合
train_x = pd.merge(train_x, store_day_avg_sales, on=['store', 'dayofweek'], how='left')
test_x = pd.merge(test_x, store_day_avg_sales, on=['store', 'dayofweek'], how='left')

# 2. 特定の商品の特定の月の平均販売数を訓練データから計算
item_month_avg_sales = tmp_x.groupby(['item', 'month'])['sales'].mean().reset_index()
item_month_avg_sales.rename(columns={'sales': 'avg_sales_item_month'}, inplace=True)

# 訓練データとテストデータに結合
train_x = pd.merge(train_x, item_month_avg_sales, on=['item', 'month'], how='left')
test_x = pd.merge(test_x, item_month_avg_sales, on=['item', 'month'], how='left')


import seaborn as sns

# train_xとtrain_yを列方向(axis=1)に連結します
combined_df = pd.concat([train_x, train_y], axis=1)

# 相関行列を計算します
# 数値データのみを対象とするため、エラーを無視するオプションを追加
corr = combined_df.corr(numeric_only=True)

# ヒートマップを描画
plt.figure(figsize=(10, 8)) # サイズを調整
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("heatmap")
plt.show()

corr = combined_df.corr()
corr_thres = 0
correlation_with_price = corr['sales'].abs()
significant_columns = correlation_with_price[correlation_with_price > corr_thres].index
significant_columns = significant_columns[significant_columns != 'sales']

train_x = train_x[significant_columns].copy()
test_x = test_x[significant_columns].copy()


# -----------------------------------------------------------
# XGBRegressorの学習・推論（このセルは変更不可）
# -----------------------------------------------------------
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 回帰用GBDTの学習
model = XGBRegressor(objective="reg:squarederror", random_state=rseed)
model.fit(train_x, train_y)

# 学習データ予測 & RMSE算出（参考値）
train_pred = model.predict(train_x)
rmse = np.sqrt(mean_squared_error(train_y, train_pred))
print(f"Train RMSE: {rmse:.5f}")

# テスト予測 & 提出ファイル作成
test_pred = model.predict(test_x)
submission = pd.DataFrame({'id': test_id, 'sales': test_pred})
submission.to_csv("submission.csv", index=False)

