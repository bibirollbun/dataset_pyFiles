# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install --upgrade xgboost


# -------------------- Imports --------------------
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# -------------------- Load Data --------------------
df = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
target = "Lap_Time_Seconds"

# -------------------- Drop High Cardinality & Redundant Identifiers --------------------
drop_cols = [
    "Unique ID", "Rider_ID", "rider_name", "team_name", "bike_name",
    "circuit_name", "shortname", "category_x", "year_x"
]
df.drop(columns=drop_cols, inplace=True)

# -------------------- Target Encoding for Riders/Teams/Bikes --------------------
for col in ["rider", "team", "bike"]:
    df[f"tar_enc_{col}"] = df.groupby(col)[target].transform("mean")

# -------------------- Feature Engineering --------------------
df["temp_diff"] = df["Track_Temperature_Celsius"] - df["Ambient_Temperature_Celsius"]
df["speed_per_km"] = df["Avg_Speed_kmh"] / df["Circuit_Length_km"]
df["degradation_per_corner"] = df["Tire_Degradation_Factor_per_Lap"] / df["Corners_per_Lap"]


# -------------------- Outlier Removal --------------------
Q1, Q3 = df[target].quantile([0.01, 0.99])
df = df[(df[target] >= Q1) & (df[target] <= Q3)]

# -------------------- Categorical Handling --------------------
cat_cols = [
    "Track_Condition", "Tire_Compound_Front", "Tire_Compound_Rear",
    "Penalty", "Session", "weather", "track"
]
df[cat_cols] = df[cat_cols].fillna("NaN").astype("category")

# -------------------- Final Dataset --------------------
X = df.drop([target], axis=1)
y = np.sqrt(df[target])  # Square root transform for stability

# -------------------- Model Training --------------------
def model_pipeline(X, y, cat_features):
    oof_preds = {}
    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def run_model(model_class, name, **kwargs):
        oof = np.zeros(len(y))
        scores = []
        for fold, (tr, va) in enumerate(cv.split(X), 1):
            model = model_class(**kwargs)
            model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
            preds = model.predict(X.iloc[va])
            oof[va] = preds
            score = mean_squared_error(y.iloc[va], preds, squared=False)
            scores.append(score)
            print(f"{name} Fold {fold} RMSE: {score:.5f}")
        mean_rmse = np.mean(scores)
        print(f"{name} Mean RMSE: {mean_rmse:.5f}")
        oof_preds[name] = oof
        results.append((name, mean_rmse))

    run_model(xgb.XGBRegressor, "xgb", n_estimators=1000, early_stopping_rounds=50, random_state=42)
    run_model(lgb.LGBMRegressor, "lgb", n_estimators=1000, early_stopping_rounds=50, random_state=42)
    run_model(cb.CatBoostRegressor, "catboost", iterations=1000, early_stopping_rounds=50, cat_features=cat_features, verbose=0, random_state=42)

    return oof_preds, results


# -------------------- Imports --------------------
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# -------------------- Load Data --------------------
df = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
target = "Lap_Time_Seconds"

# -------------------- Drop High Cardinality & Redundant Identifiers --------------------
drop_cols = [
    "Unique ID", "Rider_ID", "rider_name", "team_name", "bike_name",
    "circuit_name", "shortname", "category_x", "year_x"
]
df['Penalty'] = df['Penalty'].fillna('DNS')
df.drop(columns=drop_cols, inplace=True)

# -------------------- Target Encoding for Riders/Teams/Bikes --------------------
for col in ["rider", "team", "bike"]:
    df[f"tar_enc_{col}"] = df.groupby(col)[target].transform("mean")

# -------------------- Feature Engineering --------------------
df["temp_diff"] = df["Track_Temperature_Celsius"] - df["Ambient_Temperature_Celsius"]
df["speed_per_km"] = df["Avg_Speed_kmh"] / df["Circuit_Length_km"]
df["degradation_per_corner"] = df["Tire_Degradation_Factor_per_Lap"] / df["Corners_per_Lap"]

# -------------------- Outlier Removal --------------------
Q1, Q3 = df[target].quantile([0.01, 0.99])
df = df[(df[target] >= Q1) & (df[target] <= Q3)]

# -------------------- Categorical Handling --------------------
cat_cols = [
    "Track_Condition", "Tire_Compound_Front", "Tire_Compound_Rear",
    "Penalty", "Session", "weather", "track"
]
df[cat_cols] = df[cat_cols].fillna("NaN").astype("category")
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

# -------------------- Final Dataset --------------------
X = df.drop([target], axis=1)
y = np.sqrt(df[target])
cat_features_cb = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

# -------------------- Model Training --------------------
def model_pipeline(X, y, cat_features_cb):
    oof_preds = {}
    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def run_model(model_class, name, X_model, **kwargs):
        oof = np.zeros(len(y))
        scores = []
        for fold, (tr, va) in enumerate(cv.split(X_model), 1):
            if name == "lgb":
                model = model_class(**kwargs)
                model.fit(
                    X_model.iloc[tr], y.iloc[tr],
                    eval_set=[(X_model.iloc[va], y.iloc[va])],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            elif name == "catboost":
                model = model_class(**kwargs)
                model.fit(
                    X_model.iloc[tr], y.iloc[tr],
                    eval_set=[(X_model.iloc[va], y.iloc[va])],
                    cat_features=cat_features_cb
                )
            else:
                model = model_class(**kwargs)
                model.fit(
                    X_model.iloc[tr], y.iloc[tr],
                    eval_set=[(X_model.iloc[va], y.iloc[va])],
                    early_stopping_rounds=50,
                    verbose=False
                )
            preds = model.predict(X_model.iloc[va])
            oof[va] = preds
            score = mean_squared_error(y.iloc[va], preds, squared=False)
            scores.append(score)
            print(f"{name.upper()} Fold {fold} RMSE: {score:.4f}")
        mean_rmse = np.mean(scores)
        print(f"{name.upper()} OOF RMSE: {mean_rmse:.4f}\n")
        oof_preds[name] = oof
        results.append((name, mean_rmse))

    # -------- GPU-Enabled XGBoost --------
    run_model(
        xgb.XGBRegressor, "xgb", X,
        n_estimators=1000,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        enable_categorical=True,
        random_state=42
    )

    # -------- LightGBM (CPU Recommended) --------
    run_model(
        lgb.LGBMRegressor, "lgb", X,
        n_estimators=1000,
        random_state=42
    )

    # -------- GPU-Enabled CatBoost --------
    run_model(
        cb.CatBoostRegressor, "catboost", X,
        iterations=1000,
        task_type="GPU",
        devices='0',
        random_state=42
    )

    return oof_preds, results

# -------------------- Run Models --------------------
oof_preds_dict, base_results = model_pipeline(X, y, cat_features_cb)

# -------------------- Ridge Stacking --------------------
X_oof = pd.DataFrame(oof_preds_dict)

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
    ridge = Ridge(alpha=alpha)
    scores = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in cv.split(X_oof):
        ridge.fit(X_oof.iloc[tr], y.iloc[tr])
        preds = ridge.predict(X_oof.iloc[va])
        score = mean_squared_error(y.iloc[va], preds, squared=False)
        scores.append(score)
    return np.mean(scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best Ridge Params:", best_params)

ridge_model = Ridge(**best_params)
ridge_model.fit(X_oof, y)
stacked_preds = ridge_model.predict(X_oof)
stacked_rmse = mean_squared_error(y, stacked_preds, squared=False)
print(f"Stacked Ridge RMSE: {stacked_rmse:.4f}")



import joblib
joblib.dump(ridge_model, "stacked_ridge_model.pkl")


df1=pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
print(df1.columns.tolist())


# Load validation data
val_df = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")

# Store true values and drop target for prediction
val_true_y = np.sqrt(val_df["Lap_Time_Seconds"])
val_df.drop("Lap_Time_Seconds", axis=1, inplace=True)

# Drop high-cardinality and identifier columns
drop_cols = [
    "Unique ID", "Rider_ID", "rider_name", "team_name", "bike_name",
    "circuit_name", "shortname", "category_x", "year_x"
]
val_df['Penalty'] = val_df['Penalty'].fillna('DNS')
val_df.drop(columns=drop_cols, inplace=True)

# Target encoding (based on train df statistics)
for col in ["rider", "team", "bike"]:
    val_df[f"tar_enc_{col}"] = val_df[col].map(df.groupby(col)["Lap_Time_Seconds"].mean())

# Feature Engineering
val_df["temp_diff"] = val_df["Track_Temperature_Celsius"] - val_df["Ambient_Temperature_Celsius"]
val_df["speed_per_km"] = val_df["Avg_Speed_kmh"] / val_df["Circuit_Length_km"]
val_df["degradation_per_corner"] = val_df["Tire_Degradation_Factor_per_Lap"] / val_df["Corners_per_Lap"]

# Handle missing and categorical columns
cat_cols = [
    "Track_Condition", "Tire_Compound_Front", "Tire_Compound_Rear",
    "Penalty", "Session", "weather", "track"
]
val_df[cat_cols] = val_df[cat_cols].fillna("NaN").astype("category")
for col in val_df.select_dtypes(include='object').columns:
    val_df[col] = val_df[col].astype('category')

# Ensure columns match training features
val_df = val_df[X.columns]

val_df.head()


from sklearn.metrics import mean_squared_error
import joblib

# -------------------- Reload Ridge Stacking Model --------------------
ridge_model = joblib.load("stacked_ridge_model.pkl")

# -------------------- Retrain Base Models on Full Train Set --------------------
xgb_model = xgb.XGBRegressor(n_estimators=1000, enable_categorical=True, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=1000, random_state=42)
cat_model = cb.CatBoostRegressor(iterations=1000, random_state=42, verbose=0)

# Train each base model on full train data
xgb_model.fit(X, y)
lgb_model.fit(X, y)
cat_model.fit(X, y, cat_features=cat_features_cb)

# -------------------- Predict on val_df --------------------
val_preds_xgb = xgb_model.predict(val_df)
val_preds_lgb = lgb_model.predict(val_df)
val_preds_cat = cat_model.predict(val_df)

# -------------------- Stack Predictions --------------------
val_stack_input = pd.DataFrame({
    "xgb": val_preds_xgb,
    "lgb": val_preds_lgb,
    "catboost": val_preds_cat
})

# Ridge stack prediction
stacked_val_preds = ridge_model.predict(val_stack_input)

# -------------------- RMSE Evaluation --------------------
rmse_val = mean_squared_error(val_true_y, stacked_val_preds, squared=False)
print(f"✅ Stacked Ridge RMSE on val.csv: {rmse_val:.4f}")



import joblib
joblib.dump(xgb_model, "xgb_final_model.pkl")
joblib.dump(lgb_model, "lgb_final_model.pkl")
joblib.dump(cat_model, "catboost_final_model.pkl")


import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

# -------------------- Load Data --------------------
test_df = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
sample_submission = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv")

# -------------------- Drop High Cardinality & Redundant Identifiers --------------------
drop_cols = [
    "Unique ID", "Rider_ID", "rider_name", "team_name", "bike_name",
    "circuit_name", "shortname", "category_x", "year_x"
]
test_df['Penalty'] = test_df['Penalty'].fillna('DNS')
test_df.drop(columns=drop_cols, inplace=True)

# -------------------- Target Encoding --------------------
# Map using train dataframe statistics
for col in ["rider", "team", "bike"]:
    test_df[f"tar_enc_{col}"] = test_df[col].map(df.groupby(col)["Lap_Time_Seconds"].mean())

# -------------------- Feature Engineering --------------------
test_df["temp_diff"] = test_df["Track_Temperature_Celsius"] - test_df["Ambient_Temperature_Celsius"]
test_df["speed_per_km"] = test_df["Avg_Speed_kmh"] / test_df["Circuit_Length_km"]
test_df["degradation_per_corner"] = test_df["Tire_Degradation_Factor_per_Lap"] / test_df["Corners_per_Lap"]

# -------------------- Handle Categorical Columns --------------------
cat_cols = [
    "Track_Condition", "Tire_Compound_Front", "Tire_Compound_Rear",
    "Penalty", "Session", "weather", "track"
]
test_df[cat_cols] = test_df[cat_cols].fillna("NaN").astype("category")
for col in test_df.select_dtypes(include='object').columns:
    test_df[col] = test_df[col].astype('category')

# -------------------- Align with Training Features --------------------
test_df = test_df[X.columns]

# -------------------- Load Trained Models --------------------
xgb_model = joblib.load("xgb_final_model.pkl")
lgb_model = joblib.load("lgb_final_model.pkl")
cat_model = joblib.load("catboost_final_model.pkl")
ridge_model = joblib.load("stacked_ridge_model.pkl")

# -------------------- Predict with Base Models --------------------
test_preds_xgb = xgb_model.predict(test_df)
test_preds_lgb = lgb_model.predict(test_df)
test_preds_cat = cat_model.predict(test_df)

# -------------------- Stack Predictions --------------------
test_stack_input = pd.DataFrame({
    "xgb": test_preds_xgb,
    "lgb": test_preds_lgb,
    "catboost": test_preds_cat
})

final_preds = ridge_model.predict(test_stack_input)
final_preds = final_preds ** 2  # Since we trained on sqrt of Lap_Time_Seconds

# -------------------- Prepare Submission --------------------
sample_submission["Lap_Time_Seconds"] = final_preds
sample_submission.to_csv("solution.csv", index=False)
print("✅ solution.csv saved successfully!")



import os
print(os.getcwd())
print(os.listdir("/kaggle/working/"))










