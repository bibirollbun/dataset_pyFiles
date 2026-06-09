# ■■ 0) 必要ライブラリのインポート ■■
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


# ■■ 1) データ読み込み & インデックス設定 ■■
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
# 既に DatetimeIndex なら不要ですが念のため
if not isinstance(train.index, pd.DatetimeIndex):
    train.index = pd.to_datetime(train['timestamp'])
train = train.sort_index()



# ■■ 2) Fold 列の作成（6 期間に分けた時系列 CV 用） ■■
train['Fold'] = 0
train.loc['2023-03-01':'2023-05-01','Fold'] = 1
train.loc['2023-05-01':'2023-07-01','Fold'] = 2
train.loc['2023-07-01':'2023-09-01','Fold'] = 3
train.loc['2023-09-01':'2023-11-01','Fold'] = 4
train.loc['2023-11-01':'2024-01-01','Fold'] = 5
train.loc['2024-01-01':'2024-03-01','Fold'] = 6
print(train['Fold'].value_counts().sort_index())


# ■■ 3) 短期モメンタム & EMA 差分の計算 ■■

# --- 既存 --- 
train['momentum_1m'] = train['label'] - train['label'].shift(1)
train['momentum_5m'] = train['label'] - train['label'].shift(5)
train['ema_5']  = train['label'].ewm(span=5,  adjust=False).mean()
train['ema_10'] = train['label'].ewm(span=10, adjust=False).mean()
train['ema_diff'] = train['ema_5'] - train['ema_10']
# 欠損（先頭数行）をゼロ埋め
train[['momentum_1m','momentum_5m','ema_diff']] = \
    train[['momentum_1m','momentum_5m','ema_diff']].fillna(0)

# ←ここまで既存のコードです。以下を追加します。↓

# ■■ 追加：マルチタイムスケール・モメンタム ■■
for span in [15, 30, 60]:
    col = f'momentum_{span}m'
    train[col] = train['label'] - train['label'].shift(span)

# 欠損を 0 埋め
mom_cols = [f'momentum_{span}m' for span in [15,30,60]]
train[mom_cols] = train[mom_cols].fillna(0)

# 追加分の動作確認
print("追加したモメンタム列：", mom_cols)



# ■■ 4) 特徴量リスト・目的変数の定義 ■■
# ■■ 4) 特徴量リストに追加 ■■
features = ['momentum_1m','momentum_5m','ema_diff'] + mom_cols

target   = 'label'
fold_col = 'Fold'
print("使用特徴量：", features)


# ■■ 5) 特徴量リスト・目的変数の定義（マルチタイムスケール・モメンタムを追加） ■■
mom_cols = [f'momentum_{span}m' for span in [1,5,15,30,60]]
features = mom_cols + ['ema_diff']
target   = 'label'
fold_col = 'Fold'
print("使用特徴量：", features)



# ■■ 6) 標準化（LightGBM は必須ではありませんが安定化のため） ■■
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])



# ■■ 7) 時系列 CV + LightGBM 学習・評価 ■■
import numpy as np
from scipy.stats import pearsonr

def pearsonr_score(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

lgb_params = {
    "objective":     "regression",
    "metric":        "mae",
    "learning_rate": 0.05,
    "num_leaves":    32,
    "verbosity":     -1,
}

cv_scores = []
for f in sorted(train[fold_col].unique()):
    is_tr = train[fold_col] != f
    is_va = train[fold_col] == f

    dtr = lgb.Dataset(train.loc[is_tr, features], label=train.loc[is_tr, target])
    dva = lgb.Dataset(train.loc[is_va, features], label=train.loc[is_va, target], reference=dtr)

    model = lgb.train(
        lgb_params,
        dtr,
        num_boost_round=200,
        valid_sets=[dva],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ]
    )

    preds = model.predict(train.loc[is_va, features])
    score = pearsonr_score(train.loc[is_va, target], preds)
    print(f"Fold {f} Pearson: {score:.4f}")
    cv_scores.append(score)

print("CV mean Pearson:", np.mean(cv_scores))



# ■■■ Hyperparameter Optimization with Optuna (suppressing intermediate logs) ■■■

# 0) 必要ライブラリのインポート
import optuna
import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optunaの情報ログを抑制

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr

# ——————————————————————————————————————————————
# ★ 事前準備 ★
# 以下は既に実行済みの前提です：
#   ・train: pandas.DataFrame（DatetimeIndex, 'label' 列あり）
#   ・features: モデルに使う特徴量リスト（例 ['momentum_1m','momentum_5m','momentum_15m', … ,'ema_diff']）
#   ・target = 'label'
# ——————————————————————————————————————————————

def objective(trial):
    # 1) 探索するパラメータ空間
    params = {
        "objective":        "regression",
        "metric":           "mae",
        "learning_rate":    trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "num_leaves":       trial.suggest_int("num_leaves", 16, 256),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 1000),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
        "verbosity":        -1,
    }

    tss = TimeSeriesSplit(n_splits=3)
    pearson_scores = []

    # 2) 時系列CV（3分割）
    for tr_idx, va_idx in tss.split(train):
        dtr = lgb.Dataset(train.iloc[tr_idx][features], label=train.iloc[tr_idx][target])
        dva = lgb.Dataset(train.iloc[va_idx][features], label=train.iloc[va_idx][target], reference=dtr)

        # 3) モデル学習（コールバックで早期停止・ログ抑制）
        gbm = lgb.train(
            params,
            dtr,
            num_boost_round=200,
            valid_sets=[dva],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=0)   # 学習ログを出さない
            ]
        )

        preds = gbm.predict(train.iloc[va_idx][features])
        pearson_scores.append(pearsonr(train.iloc[va_idx][target], preds)[0])

    # 4) 各foldの平均Pearson相関を返す
    return np.mean(pearson_scores)

# 5) Optunaによる最適化実行
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 6) 最終結果表示
print("▶ Best parameters:", study.best_params)
print(f"▶ Best CV mean Pearson: {study.best_value:.4f}")



import pandas as pd

# 1) テストデータ読み込み
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
test.index = pd.to_datetime(test.index)  # インデックスが既に datetime の場合は不要

# 2) 特徴量作成（訓練時と同じ処理を）
for delta in [1,5,15,30,60]:
    test[f'momentum_{delta}m'] = test['label'].diff(periods=delta)  # テストは label=0 ダミーなので price列があればそちら
# ※ 実際には price 列がないので、momentum は submission 用に別途渡された “price” を使う想定です。
#    Kaggle の test.parquet には price 情報がないため、本番予測用には price 列を外部で用意するか、
#    事前に price 相当のラベルを生成する必要があります。

# ここでは訓練データ同様、momentum と ema_diff を作れる前提で示します。
test['ema5']  = test['label'].ewm(span=5).mean()
test['ema10'] = test['label'].ewm(span=10).mean()
test['ema_diff'] = test['ema5'] - test['ema10']

# 3) 特徴量リスト
features = ['momentum_1m','momentum_5m','momentum_15m','momentum_30m','momentum_60m','ema_diff']

# 4) 予測
preds = model.predict(test[features])  # model はチューニング後の LightGBM

# 5) 提出用 DataFrame 作成
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submission['prediction'] = preds

# 6) CSV 出力
submission.to_csv('submission.csv', index=False)
print("submission.csv を作成しました。")


