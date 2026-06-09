import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder # LabelEncoderを使用している
from sklearn.model_selection import StratifiedKFold # StratifiedKFoldを使用している
from sklearn.metrics import log_loss
# StandardScaler は XGBoost 単体では通常不要だが、NNやロジスティック回帰のベースラインモデルを比較する際に使用されていた
# 今回はXGBoost単体に特化するため、必要であればコメントアウトまたは削除
# from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

DATA_DIR = "/kaggle/input/otto-group-product-classification-challenge"
try:
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv", decimal=',')
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv", decimal=',')
except FileNotFoundError:
    print("Kaggle環境のパスでファイルが見つかりませんでした。ローカルパスを試行します。") #
    train_df = pd.read_csv('train.csv', decimal=',') #
    test_df = pd.read_csv('test.csv', decimal=',') #

train_id = train_df['id'] #
test_id = test_df['id'] #

X_train_raw = train_df.drop(['id', 'target'], axis=1) #
y_str = train_df['target'] #
X_test_raw = test_df.drop(['id'], axis=1) #

le = LabelEncoder() #
y_train_encoded = le.fit_transform(y_str) #

print("データ読み込み・初期前処理完了。") #
print(f"元の訓練データ形状: {X_train_raw.shape}, テストデータ形状: {X_test_raw.shape}") #

# --- 特徴量エンジニアリング (XGBoostに有効なものに絞る) ---
print("\n特徴量エンジニアリング中...")

# 行ごとの統計量（row_sumとrow_meanはexlast-25rmd44-4.ipynbで採用されている）
X_train_fe = X_train_raw.copy()
X_test_fe = X_test_raw.copy()

X_train_fe['row_sum'] = X_train_fe.sum(axis=1) #
X_test_fe['row_sum'] = X_test_fe.sum(axis=1) #

X_train_fe['row_mean'] = X_train_fe.mean(axis=1) #
X_test_fe['row_mean'] = X_test_fe.mean(axis=1) #

#X_train_fe['row_std'] = X_train_fe.std(axis=1) # ★追加
#X_test_fe['row_std'] = X_test_fe.std(axis=1) # ★追加

#X_train_fe['row_zeros'] = (X_train_fe == 0).sum(axis=1) # ★追加
#X_test_fe['row_zeros'] = (X_test_fe == 0).sum(axis=1) # ★追加

# --- 新たに追加する行統計量 ---

# 1. 行の中央値
#X_train_fe['row_median'] = X_train_fe.median(axis=1)
#X_test_fe['row_median'] = X_test_fe.median(axis=1)

# 2. 行の最大値
#X_train_fe['row_max_val'] = X_train_fe.max(axis=1)
#X_test_fe['row_max_val'] = X_test_fe.max(axis=1)

# 4. 行の分散 (stdの2乗)
#X_train_fe['row_var'] = X_train_fe.var(axis=1)
#X_test_fe['row_var'] = X_test_fe.var(axis=1)

# 5. 行の非ゼロ要素の数
#X_train_fe['row_non_zeros'] = (X_train_fe != 0).sum(axis=1)
#X_test_fe['row_non_zeros'] = (X_test_fe != 0).sum(axis=1)

# 6. 行の歪度 / 尖度
# これらの計算はデータサイズによっては時間がかかることがあります。
# 特にNaNが含まれる場合やデータ型に注意が必要ですが、Ottoのデータは数値のみで欠損なしなので問題ないでしょう。
#X_train_fe['row_skew'] = X_train_fe.skew(axis=1)
#X_test_fe['row_skew'] = X_test_fe.skew(axis=1)

#X_train_fe['row_kurt'] = X_train_fe.kurt(axis=1)
#X_test_fe['row_kurt'] = X_test_fe.kurt(axis=1)

# 7. 行の四分位数 / 四分位範囲
X_train_fe['row_q1'] = X_train_fe.quantile(0.25, axis=1)
X_test_fe['row_q1'] = X_test_fe.quantile(0.25, axis=1)

#X_train_fe['row_q3'] = X_train_fe.quantile(0.75, axis=1)
#X_test_fe['row_q3'] = X_test_fe.quantile(0.75, axis=1)

#X_train_fe['row_iqr'] = X_train_fe['row_q3'] - X_train_fe['row_q1']
#X_test_fe['row_iqr'] = X_test_fe['row_q3'] - X_test_fe['row_q1']

print("特徴量エンジニアリング完了。")
print(f"加工後の訓練データ形状: {X_train_fe.shape}, 加工後のテストデータ形状: {X_test_fe.shape}")

# --- スケーリングはXGBoost単体では通常行わない ---
# exlast-25rmd44-4.ipynbではStandardScalerが適用されているが、
# これはXGBoostには通常不要で、NNなど他のモデルとの併用を考慮した名残かもしれない。
# XGBoost単体で性能を追求するなら、この部分はコメントアウトまたは削除を推奨。
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_final = scaler.fit_transform(X_train_fe)
# X_test_final = scaler.transform(X_test_fe)
# print("\nStandardScalerを適用しました。")

# スケーリングしない場合、そのままPandas DataFrameをDMatrixに渡す
X_train_final = X_train_fe
X_test_final = X_test_fe

# XGBoost向けDMatrixの準備
dtrain_full = xgb.DMatrix(X_train_final, label=y_train_encoded) #
dtest_full = xgb.DMatrix(X_test_final) #

# --- XGBoost 単体モデルの学習と予測 ---
print("\n--- XGBoost 単体モデルの学習と予測中... ---")

# XGBoostモデルのパラメータ (最適化を試みる)
xgb_params = {
    'objective': 'multi:softprob', #
    'num_class': 9, #
    'eval_metric': 'mlogloss', #
    'seed': 71, #
    'nthread': -1, #

    # ★ 最適化対象のハイパーパラメータ
    'eta': 0.01, # 学習率を少し低めに、かつnum_boost_roundを増やすのが良いバランス
    'max_depth': 8, # ツリーの深さ
    'subsample': 0.8, # 各ツリーで使うデータの割合
    'colsample_bytree': 0.8, # 各ツリーで使う特徴量の割合
    'min_child_weight': 1, # 葉ノードに必要となるインスタンスの最小総重み
    'gamma': 0.1, # 損失減少の最小閾値
    'lambda': 1, # L2正則化項
    'alpha': 0 # L1正則化項
}

num_boost_round = 4500 # ★ イテレーション数を大幅に増やす (early_stoppingと併用)
early_stopping_rounds = 200 # ★ 早い段階で停止しないように

xgb_single_model = xgb.train(
    xgb_params,
    dtrain_full,
    num_boost_round=num_boost_round,
    evals=[(dtrain_full, 'train')], # 訓練セットのみの評価 (時間がない場合)
    # 検証セットがあればさらに良い:
    # X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(X_train_final, y_train_encoded, test_size=0.1, random_state=42, stratify=y_train_encoded)
    # evals=[(xgb.DMatrix(X_train_part, label=y_train_part), 'train'), (xgb.DMatrix(X_val_part, label=y_val_part), 'valid')],
    # early_stopping_rounds=early_stopping_rounds,
    verbose_eval=100 # 100ラウンドごとに進捗を表示
)
# --- 重要度の低い特徴量の出力 ---

# 1. 特徴量重要度を辞書形式で取得 (importance_type='gain'を指定)
#feature_importances = xgb_single_model.get_score(importance_type='gain')

# 2. 辞書をDataFrameに変換してソート
# 特徴量名がDMatrixの内部名 (f0, f1, ...) になっている可能性があるので、
# 元の列名とDMatrixの列名のマッピングが必要です。
# DMatrixにDataFrameを直接渡した場合は、列名がそのまま使われます。
#importance_df = pd.DataFrame({
#    'feature': list(feature_importances.keys()),
#    'importance': list(feature_importances.values())
#})

# 重要度で昇順にソート（低い順）
#importance_df = importance_df.sort_values(by='importance', ascending=True)

#print("\n--- 重要度の低い特徴量 (上位10件) ---")
#print(importance_df.head(10))

# 3. 必要であれば、重要度が特定の閾値以下の特徴量を抽出
# 例: 重要度が0.1未満の特徴量を抽出
# low_importance_features = importance_df[importance_df['importance'] < 0.1]['feature'].tolist()
# print("\n重要度が0.1未満の特徴量:", low_importance_features)

# 重要度が0の特徴量（全く使われなかった特徴量）
#zero_importance_features = importance_df[importance_df['importance'] == 0]['feature'].tolist()
#print("\n重要度が0の特徴量:", zero_importance_features)

# 予測
# 早期停止があった場合、best_iterationを使う
# if xgb_single_model.best_iteration is not None:
#     xgb_single_preds = xgb_single_model.predict(dtest_full, iteration_range=(0, xgb_single_model.best_iteration + 1))
# else:
xgb_single_preds = xgb_single_model.predict(dtest_full) #


# 提出ファイル作成
submission_xgb_single = pd.DataFrame(xgb_single_preds, columns=[f'Class_{i+1}' for i in range(9)]) #
submission_xgb_single.insert(0, 'id', test_id) #
submission_xgb_single.to_csv('submission_best_single_xgb.csv', index=False) # ファイル名を変更
print("\nXGBoost 最適化単体モデルの提出ファイル 'submission_best_single_xgb.csv' を作成しました。") #

