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


# =============================================================
# (A) 変数 store の前処理
# =============================================================
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

col = 'store'



# Target Encoding
def target_encoding_cv(train_x, train_y, test_x, col, n_splits=6, random_state=71):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_encoded = np.zeros(len(train_x))
    
    # CV-based Target Encoding for train
    for train_idx, val_idx in kf.split(train_x):
        X_train_fold = train_x.iloc[train_idx]
        y_train_fold = train_y.iloc[train_idx]
        X_val_fold = train_x.iloc[val_idx]
        
        # 各foldでTarget Encodingの統計を計算
        target_mean = y_train_fold.groupby(X_train_fold[col]).mean()
        global_mean = y_train_fold.mean()
        
        # validation setに適用
        train_encoded[val_idx] = X_val_fold[col].map(target_mean).fillna(global_mean)
    
    # For test data, use all training data
    target_mean_full = train_y.groupby(train_x[col]).mean()
    global_mean_full = train_y.mean()
    test_encoded = test_x[col].map(target_mean_full).fillna(global_mean_full)
    
    return train_encoded, test_encoded

train_x[f'{col}_target'], test_x[f'{col}_target'] = target_encoding_cv(
    train_x, train_y, test_x, col, random_state=rseed)



# =============================================================
# (B) 変数 item の前処理
# =============================================================
col = 'item'

# Target Encoding
train_x[f'{col}_target'], test_x[f'{col}_target'] = target_encoding_cv(
    train_x, train_y, test_x, col, random_state=rseed)


# =============================================================
# 組み合わせ特徴量
# =============================================================

# Store × Item 組み合わせ特徴量
train_x['store_item_combo'] = train_x['store'].astype(str) + '_' + train_x['item'].astype(str)
test_x['store_item_combo'] = test_x['store'].astype(str) + '_' + test_x['item'].astype(str)


# Store×Item組み合わせのTarget Encoding
train_x['store_item_target'], test_x['store_item_target'] = target_encoding_cv(
    train_x, train_y, test_x, 'store_item_combo', random_state=rseed)


# =============================================================
# (C) 変数 date の前処理 
# =============================================================
col = 'date'


# 日付文字列をPandasの日時型(datetime: dt)に変換
train[col] = pd.to_datetime(train[col])
test[col] = pd.to_datetime(test[col])

# 基本的な日付特徴量
train_x['year'] = train[col].dt.year
train_x['month'] = train[col].dt.month
train_x['day'] = train[col].dt.day

test_x['year'] = test[col].dt.year
test_x['month'] = test[col].dt.month
test_x['day'] = test[col].dt.day

# 月と曜日の周期性
train_x['month_sin'] = np.sin(2 * np.pi * train[col].dt.month / 12)
train_x['month_cos'] = np.cos(2 * np.pi * train[col].dt.month / 12)
train_x['dow_sin'] = np.sin(2 * np.pi * train[col].dt.dayofweek / 7)
train_x['dow_cos'] = np.cos(2 * np.pi * train[col].dt.dayofweek / 7)

test_x['month_sin'] = np.sin(2 * np.pi * test[col].dt.month / 12)
test_x['month_cos'] = np.cos(2 * np.pi * test[col].dt.month / 12)
test_x['dow_sin'] = np.sin(2 * np.pi * test[col].dt.dayofweek / 7)
test_x['dow_cos'] = np.cos(2 * np.pi * test[col].dt.dayofweek / 7)

# 週末・月初月末フラグ
train_x['is_weekend'] = (train[col].dt.dayofweek >= 5).astype(int)
train_x['is_month_start'] = train[col].dt.is_month_start.astype(int)
train_x['is_month_end'] = train[col].dt.is_month_end.astype(int)

test_x['is_weekend'] = (test[col].dt.dayofweek >= 5).astype(int)
test_x['is_month_start'] = test[col].dt.is_month_start.astype(int)
test_x['is_month_end'] = test[col].dt.is_month_end.astype(int)


# 最終処理・確認

cols_to_drop=['date', 'store_item_combo']


# 上記リストに該当する列を train_x / test_x 両方からまとめて削除
#    errors="ignore" により、存在しない場合でもエラーにならない
train_x = train_x.drop(cols_to_drop, axis=1, errors="ignore")
test_x  = test_x.drop(cols_to_drop, axis=1, errors="ignore")


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

