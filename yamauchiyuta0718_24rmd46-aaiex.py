# 必要なライブラリをインポート
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error


# データの読み込み
train_df = pd.read_csv("../input/playground-series-s4e4/train.csv")
test_df = pd.read_csv("../input/playground-series-s4e4/test.csv")

# 提出用にIDを保持
test_ids = test_df['id']
# 学習に不要なid列の削除
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


# --- 特徴量エンジニアリング ---
def create_features(df):
    # データフレームから新しい特徴量を作成
    # 物理的にありえないHeight=0を中央値で補完
    if 0 in df['Height'].values:
        median_height = df['Height'][df['Height'] > 0].median()
        df['Height'] = df['Height'].replace(0, median_height)

    # 比率の特徴量を作成
    df['length_dia_ratio'] = df['Length'] / df['Diameter']
    df['length_height_ratio'] = df['Length'] / df['Height']
    # アワビの密度に関連する可能性のある特徴量
    # df['bmi'] = df['Whole weight'] / (df['Height']**2)
    df['water_loss'] = df['Whole weight'] - df['Whole weight.1'] - df['Whole weight.2'] - df['Shell weight']
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

# 'Sex'列はカテゴリカル変数なので、数値変数に変換する
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
test_df['Sex'] = le.transform(test_df['Sex'])


# 目的変数と説明変数に分割
X = train_df.drop('Rings', axis=1)
y = train_df['Rings']
X_test = test_df

# 目的変数を対数変換
y_log = np.log1p(y)


# --- LightGBMモデルと交差検証の導入 ---
# LightGBMのハイパーパラメータ
params = {
    'objective': 'regression_l1', # MAE（L1損失）を目的関数とする
    'metric': 'rmse',             # RMSE（二乗平均平方根誤差）で評価
    'n_estimators': 2000,         # 木の数
    'learning_rate': 0.01,        # 学習率
    'feature_fraction': 0.8,      # 各木で使う特徴量の割合
    'bagging_fraction': 0.8,      # 各木で使うデータの割合
    'bagging_freq': 1,
    'lambda_l1': 0.1,             # L1正則化
    'lambda_l2': 0.1,             # L2正則化
    'num_leaves': 31,             # 葉の数
    'verbose': -1,                # ログ出力を抑制
    'n_jobs': -1,                 # 全てのCPUコアを使用
    'seed': 42,
    'boosting_type': 'gbdt',      # 勾配ブースティング決定木
}


# 5分割交差検証の準備
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))       # Out-of-Fold予測を格納する配列
test_preds = np.zeros(len(X_test)) # テストデータへの予測を格納する配列

# 交差検証ループ
for fold, (train_index, val_index) in enumerate(kf.split(X, y_log)):
    print(f"========== Fold {fold+1} ==========")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_log, y_val_log = y_log.iloc[train_index], y_log.iloc[val_index]

    # モデルの定義と学習
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train_log,
              eval_set=[(X_val, y_val_log)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)]) # 100ラウンド改善がなければ停止

    # バリデーションデータとテストデータへの予測
    oof_preds[val_index] = model.predict(X_val)
    test_preds += model.predict(X_test) / N_SPLITS

# OOF予測の評価 (Root Mean Squared Logarithmic Error)
# 負の値を避けるためにクリッピングを行う
oof_preds[oof_preds < 0] = 0
rmsle = np.sqrt(mean_squared_log_error(y, np.expm1(oof_preds)))
print(f"\nOverall OOF RMSLE: {rmsle:.5f}")


# --- 予測値の逆変換と提出ファイルの作成 ---

# 予測値を元のスケールに戻す (exp(x) - 1)
final_preds = np.expm1(test_preds)

# Rings（年齢）は正の整数なので、後処理を行い、負の予測値が万が一あれば0にする
final_preds[final_preds < 0] = 0
# 四捨五入して整数にする
final_preds = np.round(final_preds).astype(int)


# 提出用データフレームを作成
submission = pd.DataFrame({'id': test_ids, 'Rings': final_preds})
submission.to_csv('submission_improved.csv', index=False)

print("\n提出ファイル 'submission_improved.csv' を作成しました。")
print(submission.head())

