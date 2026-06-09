import pandas as pd
import numpy as np
import os
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tqdm.auto import tqdm
import warnings
warnings.simplefilter('ignore')

# --- データ読み込み ---
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# --- 特徴量エンジニアリング (元のコードをそのまま使用) ---
def bin_column(df, column, bins, bin_names=None):
    if bin_names is None:
        bin_names = [f'{b:.1f}_to_{b_next:.1f}' for b, b_next in zip(bins[:-1], bins[1:])]
    df[column + '_binned'] = pd.cut(df[column], bins=bins, labels=bin_names, include_lowest=True)
    return df

bins_dict = {
    'RhythmScore': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'VocalContent': [0.025, 0.1, 0.15, 0.2],
    'AcousticQuality': [0.01, 0.2, 0.4, 0.6, 0.8, 1.0],
    'InstrumentalScore': [0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
    'LivePerformanceLikelihood': [0.05, 0.2, 0.4],
    'MoodScore': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'Energy': [0, 0.2, 0.4, 0.6, 0.8, 1.0]
}

for col, bins in bins_dict.items():
    train = bin_column(train, col, bins)
    test = bin_column(test, col, bins)

for df in [train, test]:
    df['VocalContent_log'] = np.log(df['VocalContent'])
    df['AcousticQuality_log'] = np.log(df['AcousticQuality'])
    df['InstrumentalScore_log'] = np.log(df['InstrumentalScore'])
    df['LivePerformanceLikelihood_log'] = np.log(df['LivePerformanceLikelihood'])

numerical_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
                      'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                      'TrackDurationMs', 'Energy']

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)

# --- 目的変数の中心化 ---
BeatsPerMinute_global_avg = train['BeatsPerMinute'].mean()
X = train.drop(columns=["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"] - BeatsPerMinute_global_avg
X_test = test.drop(columns=["id"])

# --- XGBoostモデルの学習 ---
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

print(f"\n{'='*10} XGBoost Training {'='*10}")
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    start = time.time()
    
    model = XGBRegressor(
        device="cuda",
        max_depth=5,
        colsample_bytree=0.9,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.08,
        gamma=10.0,
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True
    )
    
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    
    oof_preds[valid_idx] = model.predict(x_valid)
    test_preds += model.predict(X_test.copy()) / FOLDS
    
    rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_idx]))
    print(f"XGB Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Training time: {time.time() - start:.1f} sec")

full_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\n--- XGB Final CV RMSE (before blending): {full_rmse:.4f} ---")


# --- alphaを変化させて複数のsubmissionファイルを作成 ---
print(f"\n{'='*10} Generating Submissions for different alpha values {'='*10}")

# alphaを0.1から0.9まで0.1刻みでループ
for alpha in np.arange(0.1, 1.0, 0.1):
    alpha = round(alpha, 1) # 浮動小数点誤差を丸める
    
    # 1. 予測値をalphaで縮小（平均値に寄せる）
    blended_oof = oof_preds * alpha
    blended_test = test_preds * alpha
    
    # 2. そのalphaでのCVスコアを計算・表示
    cv_rmse_alpha = np.sqrt(mean_squared_error(y, blended_oof))
    print(f"Alpha = {alpha:.1f} -> CV RMSE: {cv_rmse_alpha:.4f}")

    # 3. 提出用データを作成
    # 予測値に全体の平均値を足し戻す
    final_preds = blended_test + BeatsPerMinute_global_avg
    
    # clip処理
    final_preds_clipped = np.clip(final_preds, 46.718, 206.037)
    
    # 4. submissionファイルに書き出し
    submission["BeatsPerMinute"] = final_preds_clipped
    
    # ファイル名を指定して保存
    file_name = f"submission_alpha_{alpha:.1f}.csv"
    submission.to_csv(file_name, index=False)
    print(f"-> Saved {file_name}")

print("\nAll submission files have been generated.")




