# ===========================================
# ライブラリ
# ===========================================
import os, warnings, math
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import RidgeCV

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

# ===========================================
# 0. データ読み込み & 列名整理
# ===========================================
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {
        "Whole weight":   "WholeWeight",
        "Whole weight.1": "ShuckedWeight",
        "Whole weight.2": "VisceraWeight",
        "Shell weight":   "ShellWeight",
    }
    return df.rename(columns=rename_map)

train = load_and_clean("/kaggle/input/playground-series-s4e4/train.csv")
test  = load_and_clean("/kaggle/input/playground-series-s4e4/test.csv")

TARGET = "Rings"
ID_COL = "id"

# ===========================================
# 1. 特徴量エンジニアリング
# ===========================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-5
    df = df.copy()

    # 体積・面積
    df["Volume"] = df["Length"] * df["Diameter"] * df["Height"]
    df["Area"]   = math.pi * (df["Diameter"] / 2.0) ** 2

    # 比率
    df["Height_to_Diameter"] = df["Height"] / (df["Diameter"] + eps)
    df["Shell_to_Length"]    = df["ShellWeight"] / (df["Length"] + eps)
    df["Weight_per_Volume"]  = df["WholeWeight"] / (df["Volume"] + eps)
    df["Shucked_to_Whole"]   = df["ShuckedWeight"] / (df["WholeWeight"] + eps)
    df["Viscera_to_Whole"]   = df["VisceraWeight"] / (df["WholeWeight"] + eps)
    df["Shell_to_Whole"]     = df["ShellWeight"] / (df["WholeWeight"] + eps)

    return df

train_fe = add_features(train)
test_fe  = add_features(test)

# ===========================================
# 2. 前処理パイプライン
# ===========================================
categorical_cols = ["Sex"]
numerical_cols   = [
    c for c in train_fe.columns
    if c not in categorical_cols + [TARGET, ID_COL]
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("scaler", StandardScaler()),
        ]), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ===========================================
# 3. ベースモデル
# ===========================================
base_models = {
    "lgb": lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.03, random_state=42,
    ),
    "xgb": XGBRegressor(
        n_estimators=1200, learning_rate=0.03, 
        objective="reg:squaredlogerror",  
        random_state=42,
    ),
    "cat": CatBoostRegressor(
        learning_rate=0.03, iterations=1200,
        loss_function="RMSE", random_seed=42, verbose=False,
    ),
    
}

# ===========================================
# 4. KFold CV & OOF 予測 (log スケール)
# ===========================================
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)

oof_log_preds  = np.zeros((train_fe.shape[0], len(base_models)))
test_log_preds = np.zeros((test_fe.shape[0],  len(base_models)))
rmsle_list     = []

for m_idx, (name, model) in enumerate(base_models.items()):
    print(f"===== Training base model: {name} =====")
    fold_oof_log  = np.zeros(train_fe.shape[0])
    fold_test_log = np.zeros(test_fe.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_fe)):
        X_tr = train_fe.iloc[tr_idx].drop(columns=[TARGET])
        y_tr = np.log1p(train_fe.iloc[tr_idx][TARGET])   # ---- log1p ----
        X_va = train_fe.iloc[va_idx].drop(columns=[TARGET])
        y_va = train_fe.iloc[va_idx][TARGET]            # ---- 原スケール ----

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_tr, y_tr)

        # 予測（log スケール）
        pred_va_log = pipe.predict(X_va)
        fold_oof_log[va_idx] = pred_va_log
        fold_test_log       += pipe.predict(test_fe) / NFOLD

        # RMSLE を原スケールで評価
        pred_va = np.expm1(pred_va_log)
        pred_va = np.clip(pred_va, 0, None)
        rmsle = math.sqrt(mean_squared_log_error(y_va, pred_va))
        print(f"  Fold{fold+1}: RMSLE = {rmsle:.5f}")

    oof_log_preds[:, m_idx]  = fold_oof_log
    test_log_preds[:, m_idx] = fold_test_log

    # OOF RMSLE
    oof_pred = np.expm1(fold_oof_log)
    oof_pred = np.clip(oof_pred, 0, None)
    rmsle = math.sqrt(mean_squared_log_error(train_fe[TARGET], oof_pred))
    rmsle_list.append(rmsle)
    print(f"=> {name} OOF RMSLE: {rmsle:.5f}\n")

print("Base model OOF RMSLEs:",
      {k: f"{v:.5f}" for k, v in zip(base_models.keys(), rmsle_list)})

# ===========================================
# 5. スタッキング (RidgeCV) も log スケール
# ===========================================
meta_model = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)
meta_model.fit(oof_log_preds, np.log1p(train_fe[TARGET]))

final_oof_log = meta_model.predict(oof_log_preds)
final_pred_log = meta_model.predict(test_log_preds)

# OOF RMSLE
final_oof = np.expm1(final_oof_log)
final_oof = np.clip(final_oof, 0, None)
final_rmsle = math.sqrt(mean_squared_log_error(train_fe[TARGET], final_oof))
print(f"\n==== Stacked Model OOF RMSLE: {final_rmsle:.5f} ====\n")

# ===========================================
# 6. 提出ファイル
# ===========================================
final_test_pred = np.expm1(final_pred_log)
final_test_pred = np.clip(final_test_pred, 0, None)

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: final_test_pred
})
submission.to_csv("submission.csv", index=False)

