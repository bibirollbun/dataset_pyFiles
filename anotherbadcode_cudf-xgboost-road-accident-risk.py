import json
import numpy as np
import pandas as pd
import cudf
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from skopt import gp_minimize
from skopt.space import Real, Integer

import warnings
warnings.filterwarnings("ignore")

CATEGORICAL_COLS = ["road_type", "lighting", "weather", "time_of_day"]
BOOLEAN_COLS = ["road_signs_present", "public_road", "holiday", "school_season"]
NUMERIC_COLS = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]

train = cudf.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = cudf.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

SEED=42; N_ITER=50


train_pd = train.to_pandas()
train_pd = train_pd.sample(frac=0.2)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.histplot(train_pd["accident_risk"], bins=40, kde=True, ax=axes[0,0], color="royalblue")
axes[0,0].set_title("Distribution of Accident Risk", fontsize=14)
axes[0,0].set_xlabel("Accident Risk")
axes[0,0].set_ylabel("Frequency")

sns.boxplot(x="road_type", y="accident_risk", data=train_pd, ax=axes[0,1], palette="Set2")
axes[0,1].set_title("Accident Risk by Road Type", fontsize=14)
axes[0,1].set_xlabel("Road Type")
axes[0,1].set_ylabel("Accident Risk")
axes[0,1].tick_params(axis='x', rotation=20)

sns.scatterplot(x="speed_limit", y="accident_risk", data=train_pd,
                hue="weather", alpha=0.6, ax=axes[1,0], palette="coolwarm")
axes[1,0].set_title("Speed Limit vs Accident Risk", fontsize=14)
axes[1,0].set_xlabel("Speed Limit")
axes[1,0].set_ylabel("Accident Risk")

sns.violinplot(x="time_of_day", y="accident_risk", data=train_pd, ax=axes[1,1], palette="muted")
axes[1,1].set_title("Accident Risk by Time of Day", fontsize=14)
axes[1,1].set_xlabel("Time of Day")
axes[1,1].set_ylabel("Accident Risk")
axes[1,1].tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.show()


def cyclical_time_of_day(s):
    mapping = {
        "night": 2, "late_night": 3, "early_morning": 6, "morning": 9,
        "noon": 12, "afternoon": 15, "evening": 19, "late_evening": 21,
    }

    s_low = s.astype(str).str.lower()
    parsed = s_low.str.extract(r"([0-9]{1,2})").iloc[:, 0]
    parsed = parsed.astype("float32")
    mapped = s_low.map(mapping).astype("float32")
    hours = parsed.fillna(mapped).fillna(12.0)

    rad = 2 * cp.pi * (hours % 24) / 24.0
    return cp.sin(rad), cp.cos(rad)

def engineer(df):
    g = df.copy(deep=True)
    for c in BOOLEAN_COLS:
        if c in g.columns: g[c] = g[c].astype("int8")
    for c in NUMERIC_COLS:
        if c in g.columns: g[c] = g[c].astype("float32")

    if "time_of_day" in g.columns:
        s, c = cyclical_time_of_day(g["time_of_day"])
        g["tod_sin"], g["tod_cos"] = s, c

    if {"curvature","speed_limit"}.issubset(g.columns):
        g["curv_speed"] = g["curvature"] * g["speed_limit"]
    if {"num_lanes","speed_limit"}.issubset(g.columns):
        g["lanes_speed"] = g["num_lanes"] * g["speed_limit"]

    if "num_reported_accidents" in g.columns and "speed_limit" in g.columns:
        denom = g["speed_limit"].replace(0, cp.nan)
        g["accidents_per_speed"] = (g["num_reported_accidents"] / denom).fillna(0)

    for c in CATEGORICAL_COLS:
        if c in g.columns:
            g[c] = g[c].astype("category").cat.codes.astype("int32")


    keep = [c for c in (NUMERIC_COLS+BOOLEAN_COLS+
                        ["tod_sin","tod_cos","curv_speed","lanes_speed","accidents_per_speed"]+
                        CATEGORICAL_COLS) if c in g.columns]
    return g[keep]

def align_categories(tr, te):
    tr, te = tr.copy(deep=True), te.copy(deep=True)
    return tr, te

def bin_for_stratify(y, n_bins=10):
    y_cpu = y.to_pandas()
    return pd.qcut(y_cpu, q=min(n_bins, len(y_cpu)), duplicates="drop", labels=False).values



def tune_with_skopt(X, y, seed, n_iter):
    space = [
        Integer(4, 12, name="max_depth"),
        Integer(400, 1500, name="n_estimators"),
        Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
        Integer(1, 20, name="min_child_weight"),
        Real(0.6, 1.0, name="subsample"),
        Real(0.5, 1.0, name="colsample_bytree"),
        Real(1e-3, 10.0, prior="log-uniform", name="reg_lambda"),
        Real(1e-3, 10.0, prior="log-uniform", name="reg_alpha"),
        Integer(128, 512, name="max_bin"),
    ]

    def objective(params):
        (max_depth, n_estimators, learning_rate, min_child_weight,
         subsample, colsample_bytree, reg_lambda, reg_alpha, max_bin) = params

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        rmses = []

        for tr_idx, va_idx in kf.split(np.zeros(len(y)), y):
            X_tr, X_va = X.iloc[tr_idx].to_pandas(), X.iloc[va_idx].to_pandas()
            y_tr, y_va = y.iloc[tr_idx].to_pandas(), y.iloc[va_idx].to_pandas()

            model = XGBRegressor(
                                objective="reg:squarederror",
                                tree_method="hist",
                                device="cuda",
                                enable_categorical=True,
                                random_state=seed,
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                min_child_weight=min_child_weight,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                reg_lambda=reg_lambda,
                                reg_alpha=reg_alpha,
                                max_bin=max_bin,
                                eval_metric="rmse",
                                early_stopping_rounds=100,
                                )

            
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            X_va_np = X_va.to_numpy(dtype=np.float32)
            preds = model.get_booster().inplace_predict(cp.asarray(X_va_np))
            preds_np = cp.asnumpy(preds)
            rmse = float(np.sqrt(mean_squared_error(y_va, preds_np)))
            rmses.append(rmse)

        return np.mean(rmses)

    result = gp_minimize(objective, space, n_calls=n_iter, random_state=seed, n_initial_points=5, verbose=True)

    best_params = {
        "max_depth": result.x[0],
        "n_estimators": result.x[1],
        "learning_rate": result.x[2],
        "min_child_weight": result.x[3],
        "subsample": result.x[4],
        "colsample_bytree": result.x[5],
        "reg_lambda": result.x[6],
        "reg_alpha": result.x[7],
        "max_bin": result.x[8],
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "enable_categorical": True,
        "random_state": seed,
        "verbosity": 1,
    }
    return best_params

def fit_kfold_and_predict(X, y, X_test, params, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float32)
    test_accum = np.zeros(len(X_test), dtype=np.float32)

    for tr_idx, va_idx in kf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = XGBRegressor(
            **params,
            eval_metric="rmse",
            early_stopping_rounds=200,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        X_va_np   = X_va.to_pandas().to_numpy(dtype=np.float32)
        X_test_np = X_test.to_pandas().to_numpy(dtype=np.float32)

        oof_preds  = model.get_booster().inplace_predict(cp.asarray(X_va_np))
        test_preds = model.get_booster().inplace_predict(cp.asarray(X_test_np))

        oof[va_idx] = cp.asnumpy(oof_preds)
        test_accum += cp.asnumpy(test_preds)

    test_pred = test_accum / kf.n_splits
    metrics = dict(
        rmse=float(np.sqrt(mean_squared_error(y.to_pandas(), oof))),
        mae=float(mean_absolute_error(y.to_pandas(), oof)),
        r2=float(r2_score(y.to_pandas(), oof)),
    )
    return test_pred, metrics, oof



train_X_raw, test_X_raw = engineer(train.drop(columns=["id"])), engineer(test.drop(columns=["id"]))
train_X,test_X = align_categories(train_X_raw,test_X_raw)
y = train["accident_risk"].astype("float32")

best_params = tune_with_skopt(train_X, y, SEED, N_ITER)
best_params = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
              for k, v in best_params.items()}

print("Best params:", json.dumps(best_params, indent=2))


test_pred,metrics,oof = fit_kfold_and_predict(train_X, y, test_X, best_params, SEED)
print("CV metrics:", metrics)

submission = pd.DataFrame({"id": test["id"].to_pandas(), "accident_risk": np.clip(test_pred,0,1)})
submission.to_csv("submission.csv",index=False)
submission.head(7)


submission.to_csv("submission.csv",index=False)
print("Saved submission.csv")


train_X_pd = train_X.to_pandas()
y_pd = y.to_pandas()

N_SHAP = 10000
train_X_sample = train_X_pd.sample(n=min(N_SHAP, len(train_X_pd)), random_state=SEED)
y_sample = y_pd.loc[train_X_sample.index]

final_model = XGBRegressor(**best_params)
final_model.fit(train_X_pd, y_pd, verbose=False)

explainer = shap.TreeExplainer(final_model.get_booster())
shap_values = explainer.shap_values(train_X_sample)

plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, train_X_sample, max_display=20, show=False)
plt.title("SHAP Feature Importance", fontsize=14)
plt.show()

