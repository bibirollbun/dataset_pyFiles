import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.base import clone

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s4e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e4/test.csv")

# Preprocessing
train_y = np.log1p(train["Rings"])
train_x = train.drop(columns=["id", "Rings"])
test_x = test.drop(columns=["id"])

# Encode categorical
le = LabelEncoder()
train_x["Sex"] = le.fit_transform(train_x["Sex"].astype(str))
test_x["Sex"] = le.transform(test_x["Sex"].astype(str))

# Fill missing
median_height = train_x.loc[train_x["Height"] > 0, "Height"].median()
train_x.loc[train_x["Height"] == 0, "Height"] = median_height
test_x.loc[test_x["Height"] == 0, "Height"] = median_height

# Feature engineering
for df in [train_x, test_x]:
    df["volume"] = df["Length"] * df["Diameter"] * df["Height"]
    #df["density"] = df["Whole weight"] / (df["volume"] + 1e-9)  # ← 追加
    #df["shell_thickness"] = df["Diameter"] - df["Height"]       # ← 追加
    df["area"] = df["Length"] * df["Diameter"]
    df["shucked/whole"] = df["Whole weight.1"] / (df["Whole weight"] + 1e-9)
    df["viscera/whole"] = df["Whole weight.2"] / (df["Whole weight"] + 1e-9)
    #df["shell/whole"] = df["Shell weight"] / (df["Whole weight"] + 1e-9)
    #df["height/diameter"] = df["Height"] / (df["Diameter"] + 1e-9)
    #df["diameter/length"] = df["Diameter"] / (df["Length"] + 1e-9)

# Models
base_models = [
    XGBRegressor(n_estimators=500, random_state=42),
    LGBMRegressor(n_estimators=500, random_state=42),
    CatBoostRegressor(n_estimators=500, verbose=0, random_state=42)
]

# KFold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test_x))
oof_preds = np.zeros(len(train_x))

for model in base_models:
    fold_preds = np.zeros(len(test_x))
    oof = np.zeros(len(train_x))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        X_tr, X_va = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        y_tr, y_va = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)

        oof[va_idx] = m.predict(X_va)
        fold_preds += m.predict(test_x) / kf.n_splits

    test_preds += fold_preds / len(base_models)
    oof_preds += oof / len(base_models)

# Evaluate
rmse = np.sqrt(mean_squared_error(train_y, oof_preds))
print(f"OOF RMSE (log scale): {rmse:.5f}")

# Export
# モデル定義（Poisson）
poisson_model = LGBMRegressor(
    objective="poisson",
    n_estimators=1000,
    random_state=42
)

# 学習
poisson_model.fit(train_x, train_y)

# 予測
poisson_preds = poisson_model.predict(test_x)

# 提出ファイル作成
submission = pd.DataFrame({
    "id": test["id"],
    "Rings": np.expm1(poisson_preds)
})
submission.to_csv("submission_lgb_poisson.csv", index=False)

# 平均アンサンブル
ensemble_preds = (test_preds + poisson_preds) / 2

submission = pd.DataFrame({
    "id": test["id"],
    "Rings": np.expm1(ensemble_preds)
})
submission.to_csv("submission_ensemble_avg.csv", index=False)


