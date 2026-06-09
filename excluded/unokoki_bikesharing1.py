import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# --- 1. データ読み込み ---
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test  = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

# --- 2. 前処理関数 ---
def preprocess_data(df):
    # datetime を datetime 型に変換し、時間／日／月／年／曜日 を抽出
    df['datetime']  = pd.to_datetime(df['datetime'])
    df['hour']      = df['datetime'].dt.hour
    df['day']       = df['datetime'].dt.day
    df['month']     = df['datetime'].dt.month
    df['year']      = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek
    # 週末フラグ・通勤時間フラグ
    df['is_weekend']  = (df['dayofweek'] >= 5).astype(int)
    df['is_workhour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    return df

train = preprocess_data(train)
test  = preprocess_data(test)

# --- 3. 使用特徴量（ベースライン＋日時派生） ---
feature_cols = [
    'hour','day','month','year','dayofweek','is_weekend','is_workhour',
    'temp','atemp','humidity','windspeed',
    'weather','season','holiday','workingday'
]

# --- 4. 欠損補完（-1 埋め）＆ LabelEncoding ---
cat_cols = ['weather','season','holiday','workingday']
for df in (train, test):
    for c in feature_cols:
        df[c] = df[c].fillna(-1)
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[c], test[c]], axis=0))
    train[c] = le.transform(train[c])
    test[c]  = le.transform(test[c])

# --- 5. 学習データ準備（対数変換） ---
X = train[feature_cols]
y = np.log1p(train['count'])   # log1p for RMSLE

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=71
)

# --- 6. モデル定義＆学習 ---
model = XGBRegressor(
    random_state=71,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

# --- 7. 検証スコア計算 ---
val_pred = model.predict(X_val)
val_pred = np.expm1(val_pred)              # 逆変換
val_pred = np.clip(val_pred, 0, None)      # 非負制約
val_true = np.expm1(y_val)
rmsle = np.sqrt(mean_squared_log_error(val_true, val_pred))
print(f"Validation RMSLE: {rmsle:.5f}")

# --- 8. 本番予測＆提出ファイル作成 ---
test_pred = model.predict(test[feature_cols])
test_pred = np.expm1(test_pred)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    'datetime': test['datetime'],
    'count':    test_pred
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created.")


