import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder

# データ読み込み
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')



# datetime の分解
def preprocess_datetime(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    return df

train = preprocess_datetime(train)
test = preprocess_datetime(test)

# windspeed == 0 を欠損とみなして補完（中央値）
train['windspeed'] = train['windspeed'].replace(0, np.nan)
test['windspeed'] = test['windspeed'].replace(0, np.nan)
train['windspeed'] = train['windspeed'].fillna(train['windspeed'].median())
test['windspeed'] = test['windspeed'].fillna(train['windspeed'].median())


# 交差特徴量
for df in [train, test]:
    df['hour_workingday'] = df['hour'].astype(str) + '_' + df['workingday'].astype(str)
    df['temp_atemp_diff'] = df['atemp'] - df['temp']

    le = LabelEncoder()
    df['hour_workingday'] = le.fit_transform(df['hour_workingday'])

# 特徴量一覧（必須: weather, workingday）
features = [
    'season', 'holiday', 'workingday', 'weather',
    'temp', 'atemp', 'humidity', 'windspeed',
    'year', 'month', 'day', 'hour', 'weekday', 'is_weekend',
    'hour_workingday', 'temp_atemp_diff'
]

# 目的変数（対数変換）
train['casual_log'] = np.log1p(train['casual'])
train['registered_log'] = np.log1p(train['registered'])

X = train[features]
X_test = test[features]


def train_predict_xgb(X, y, X_test, name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pred_test = np.zeros(X_test.shape[0])
    scores = []

    for tr_idx, va_idx in kf.split(X):
        tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        dtest = xgb.DMatrix(X_test)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
        }

        model = xgb.train(params, dtrain, num_boost_round=1000,
                          evals=[(dtrain, 'train'), (dvalid, 'valid')],
                          early_stopping_rounds=50, verbose_eval=False)

        va_pred = np.expm1(model.predict(dvalid))
        va_true = np.expm1(va_y)
        score = np.sqrt(mean_squared_log_error(va_true.clip(0, None), va_pred.clip(0, None)))
        scores.append(score)

        pred_test += model.predict(dtest) / kf.n_splits

    print(f"{name} RMSLE mean: {np.mean(scores):.5f}")
    return pred_test



# =======================
# 2. モデル学習
# =======================
pred_casual = train_predict_xgb(X, train['casual_log'], X_test, 'Casual')
pred_registered = train_predict_xgb(X, train['registered_log'], X_test, 'Registered')

# =======================
# 3. 予測 & 提出
# =======================
final_pred = np.expm1(pred_casual) + np.expm1(pred_registered)
final_pred = final_pred.clip(0, None)
submission = pd.DataFrame({'datetime': test['datetime'], 'count': final_pred})
submission.to_csv('submission.csv', index=False)
print("Submission saved as submission.csv")


