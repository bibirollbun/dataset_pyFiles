import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import optuna

# =======================
# 1. VERİYİ YÜKLE
# =======================
train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# =======================
# 2. FEATURE ENGINEERING
# =======================
def feature_engineering(df):
    df = df.copy()
    
    # Duration dönüşümleri
    df["LogDuration"] = np.log1p(df["TrackDurationMs"])
    df["SqrtDuration"] = np.sqrt(df["TrackDurationMs"])
    
    # Çapraz etkileşimler
    df["Rhythm_Energy"] = df["RhythmScore"] * df["Energy"]
    df["Mood_Energy"] = df["MoodScore"] * df["Energy"]
    df["Loudness_Energy"] = df["AudioLoudness"] * df["Energy"]
    df["Rhythm_Mood"] = df["RhythmScore"] * df["MoodScore"]
    df["Rhythm_Loudness"] = df["RhythmScore"] * df["AudioLoudness"]
    df["Mood_Loudness"] = df["MoodScore"] * df["AudioLoudness"]
    
    # Oranlar
    df["Rhythm_Mood_Ratio"] = df["RhythmScore"] / (df["MoodScore"] + 1e-3)
    df["Energy_Duration_Ratio"] = df["Energy"] / (df["TrackDurationMs"] + 1e-3)
    
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=["id", "BeatsPerMinute"])
y = train_df["BeatsPerMinute"]
X_test = test_df.drop(columns=["id"])

# =======================
# 3. OPTUNA AMAÇ FONKSİYONU
# =======================
def objective(trial):
    # Model parametreleri
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 600),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.1),
        "max_depth": trial.suggest_int("xgb_max_depth", 5, 9),
        "subsample": trial.suggest_float("xgb_subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.7, 1.0),
        "random_state": 42,
        "n_jobs": -1
    }

    lgb_params = {
        "n_estimators": trial.suggest_int("lgb_n_estimators", 300, 600),
        "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.1),
        "max_depth": trial.suggest_int("lgb_max_depth", -1, 12),
        "subsample": trial.suggest_float("lgb_subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("lgb_colsample", 0.7, 1.0),
        "random_state": 42,
        "n_jobs": -1
    }

    cat_params = {
        "iterations": trial.suggest_int("cat_iterations", 300, 600),
        "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.1),
        "depth": trial.suggest_int("cat_depth", 5, 9),
        "subsample": trial.suggest_float("cat_subsample", 0.7, 1.0),
        "colsample_bylevel": trial.suggest_float("cat_colsample", 0.7, 1.0),
        "random_state": 42,
        "verbose": 0
    }

    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 200, 500),
        "max_depth": trial.suggest_int("rf_max_depth", 5, 15),
        "random_state": 42,
        "n_jobs": -1
    }

    et_params = {
        "n_estimators": trial.suggest_int("et_n_estimators", 200, 500),
        "max_depth": trial.suggest_int("et_max_depth", 5, 15),
        "random_state": 42,
        "n_jobs": -1
    }

    hgb_params = {
        "max_iter": trial.suggest_int("hgb_max_iter", 200, 500),
        "max_depth": trial.suggest_int("hgb_max_depth", 5, 15),
        "learning_rate": trial.suggest_float("hgb_lr", 0.01, 0.1),
        "random_state": 42
    }

    # Ağırlıklar
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = trial.suggest_float("w2", 0.0, 1.0)
    w3 = trial.suggest_float("w3", 0.0, 1.0)
    w4 = trial.suggest_float("w4", 0.0, 1.0)
    w5 = trial.suggest_float("w5", 0.0, 1.0)
    w6 = trial.suggest_float("w6", 0.0, 1.0)
    total = w1 + w2 + w3 + w4 + w5 + w6
    if total == 0:
        return float("inf")
    w1, w2, w3, w4, w5, w6 = [w/total for w in [w1, w2, w3, w4, w5, w6]]

    # K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgb_model = XGBRegressor(**xgb_params)
        lgb_model = LGBMRegressor(**lgb_params)
        cat_model = CatBoostRegressor(**cat_params)
        rf_model = RandomForestRegressor(**rf_params)
        et_model = ExtraTreesRegressor(**et_params)
        hgb_model = HistGradientBoostingRegressor(**hgb_params)

        xgb_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)
        cat_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        et_model.fit(X_train, y_train)
        hgb_model.fit(X_train, y_train)

        xgb_pred = xgb_model.predict(X_val)
        lgb_pred = lgb_model.predict(X_val)
        cat_pred = cat_model.predict(X_val)
        rf_pred = rf_model.predict(X_val)
        et_pred = et_model.predict(X_val)
        hgb_pred = hgb_model.predict(X_val)

        blended = (
            w1 * xgb_pred +
            w2 * lgb_pred +
            w3 * cat_pred +
            w4 * rf_pred +
            w5 * et_pred +
            w6 * hgb_pred
        )
        rmse = mean_squared_error(y_val, blended, squared=False)
        rmses.append(rmse)

    return np.mean(rmses)

# =======================
# 4. OPTUNA ÇALIŞTIR
# =======================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # süreye göre artırabilirsin

print("En iyi parametreler:", study.best_params)
print("En iyi RMSE:", study.best_value)

# =======================
# 5. TÜM VERİ İLE EĞİTİM
# =======================
params = study.best_params
w1, w2, w3, w4, w5, w6 = [params[f"w{i}"] for i in range(1,7)]
total = w1 + w2 + w3 + w4 + w5 + w6
w1, w2, w3, w4, w5, w6 = [w/total for w in [w1, w2, w3, w4, w5, w6]]

xgb_model = XGBRegressor(
    n_estimators=params["xgb_n_estimators"],
    learning_rate=params["xgb_lr"],
    max_depth=params["xgb_max_depth"],
    subsample=params["xgb_subsample"],
    colsample_bytree=params["xgb_colsample"],
    random_state=42, n_jobs=-1
)

lgb_model = LGBMRegressor(
    n_estimators=params["lgb_n_estimators"],
    learning_rate=params["lgb_lr"],
    max_depth=params["lgb_max_depth"],
    subsample=params["lgb_subsample"],
    colsample_bytree=params["lgb_colsample"],
    random_state=42, n_jobs=-1
)

cat_model = CatBoostRegressor(
    iterations=params["cat_iterations"],
    learning_rate=params["cat_lr"],
    depth=params["cat_depth"],
    subsample=params["cat_subsample"],
    colsample_bylevel=params["cat_colsample"],
    random_state=42, verbose=0
)

rf_model = RandomForestRegressor(
    n_estimators=params["rf_n_estimators"],
    max_depth=params["rf_max_depth"],
    random_state=42, n_jobs=-1
)

et_model = ExtraTreesRegressor(
    n_estimators=params["et_n_estimators"],
    max_depth=params["et_max_depth"],
    random_state=42, n_jobs=-1
)

hgb_model = HistGradientBoostingRegressor(
    max_iter=params["hgb_max_iter"],
    max_depth=params["hgb_max_depth"],
    learning_rate=params["hgb_lr"],
    random_state=42
)

xgb_model.fit(X, y)
lgb_model.fit(X, y)
cat_model.fit(X, y)
rf_model.fit(X, y)
et_model.fit(X, y)
hgb_model.fit(X, y)

xgb_pred = xgb_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
cat_pred = cat_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
et_pred = et_model.predict(X_test)
hgb_pred = hgb_model.predict(X_test)

final_pred = (
    w1 * xgb_pred +
    w2 * lgb_pred +
    w3 * cat_pred +
    w4 * rf_pred +
    w5 * et_pred +
    w6 * hgb_pred
)

# =======================
# 6. SUBMISSION
# =======================
submission = sample_submission.copy()
submission["BeatsPerMinute"] = final_pred
submission.to_csv("submission.csv", index=False)

print("submission.csv kaydedildi ✅")


