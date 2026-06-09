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


# Regression pipeline: treat accident_risk as continuous [0,1], evaluate with RMSE
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import joblib
import shap
import seaborn as sns


# Parameters
CSV_PATH = "/kaggle/input/playground-series-s5e10/train.csv" 
RANDOM_STATE = 42
EDA_SAMPLE_FRAC = 0.1
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
LGB_ESTIMATORS = 2000
LGB_LEARNING_RATE = 0.05


id_col = "id"
target_col = "accident_risk"   # continuous in [0,1]

cat_cols = ["road_type", "weather"]
ord_cols = ["lighting", "time_of_day"]
bool_cols = ["road_signs_present", "public_road", "school_season", "holiday"]
num_cols = ["num_lanes", "curvature", "speed_limit"]
optional_cols = ["num_reported_accidents"]


# Load
df = pd.read_csv(CSV_PATH)
print("Loaded:", df.shape)


# Sanity
expected = [id_col, target_col] + cat_cols + ord_cols + bool_cols + num_cols + optional_cols
missing = [c for c in expected if c not in df.columns]
if missing:
    print("WARNING: missing columns:", missing)
available_cols = [c for c in expected if c in df.columns]
use_cols = [c for c in available_cols if c != id_col]
df = df[use_cols].copy()


# Convert boolean-ish columns to 0/1 (safe coercion)
for b in bool_cols:
    if b in df.columns:
        if df[b].dtype == object:
            df[b] = df[b].str.lower().map({"true":1,"false":0,"yes":1,"no":0,"1":1,"0":0}).astype(pd.Int64Dtype())
        else:
            df[b] = pd.to_numeric(df[b], errors="coerce").astype(pd.Int64Dtype())



# Quick EDA sample (optional)
sample = df.sample(frac=min(EDA_SAMPLE_FRAC,1.0), random_state=RANDOM_STATE)
if target_col in df.columns:
    print("Target stats (sample):", sample[target_col].describe())
    sns.histplot(sample[target_col].dropna(), kde=True)
    plt.title("accident_risk distribution (sample)")
    plt.show()


# Feature engineering (same as before)
def add_features(df_):
    df = df_.copy()
    if set(["speed_limit","curvature"]).issubset(df.columns):
        df["speed_x_curv"] = df["speed_limit"].astype(float) * df["curvature"].astype(float)
    if "curvature" in df.columns:
        df["curv_bin"] = pd.cut(df["curvature"].astype(float).fillna(0), bins=[-1, 0.05, 0.2, 0.5, 1.0], labels=["flat","slight","mod","sharp"])
    if "school_seaon" in df.columns and "time_of_day" in df.columns:
        df["school_peak"] = ((df["school_seaon"]==1) & (df["time_of_day"]=="morning")).astype(int)
    if "lighting" in df.columns and "speed_limit" in df.columns:
        df["night_highspeed"] = ((df["lighting"]=="night") & (df["speed_limit"]>=50)).astype(int)
    return df


df = add_features(df)
if "speed_x_curv" in df.columns and "speed_x_curv" not in num_cols:
    num_cols.append("speed_x_curv")
if "curv_bin" in df.columns and "curv_bin" not in cat_cols:
    cat_cols.append("curv_bin")
if "school_peak" in df.columns and "school_peak" not in bool_cols:
    bool_cols.append("school_peak")
if "night_highspeed" in df.columns and "night_highspeed" not in bool_cols:
    bool_cols.append("night_highspeed")


# Preprocessing (same structure)
lighting_order = ["daylight", "dim", "night"]
time_order = ["morning", "afternoon", "evening"]
ord_categories = []
for c in ord_cols:
    if c in df.columns:
        if c == "lighting":
            ord_categories.append(lighting_order)
        elif c == "time_of_day":
            ord_categories.append(time_order)
        else:
            ord_categories.append(sorted(df[c].dropna().unique().tolist()))



num_present = [c for c in num_cols if c in df.columns]
ord_present = [c for c in ord_cols if c in df.columns]
cat_present = [c for c in cat_cols if c in df.columns]
bool_present = [c for c in bool_cols if c in df.columns]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
ord_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder(categories=ord_categories, dtype=int))]) if ord_present else None
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))]) if cat_present else None
bool_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]) if bool_present else None



transformers = []
if num_present: transformers.append(("num", num_pipe, num_present))
if ord_present: transformers.append(("ord", ord_pipe, ord_present))
if cat_present: transformers.append(("cat", cat_pipe, cat_present))
if bool_present: transformers.append(("bool", bool_pipe, bool_present))
preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)



# Helper to extract feature names after fitting preprocessor
def get_feature_names_from_ct(ct):
    names = []
    for name, trans, cols in ct.transformers_:
        if name == 'remainder' or trans == 'drop':
            continue
        if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
            ohe = trans.named_steps["ohe"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols)
    return names



# Train / holdout split
if target_col not in df.columns:
    raise ValueError(f"Target '{target_col}' not found.")
X = df.drop(columns=[target_col])
y = df[target_col].astype(float)   # <-- important: continuous float between 0 and 1




# Small safety: clip into [0,1]
y = y.clip(0.0, 1.0)

X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
print("Train / Hold shapes:", X_train.shape, X_hold.shape)




# CV using stratified folds on binned y distribution
kb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
y_bins = kb.fit_transform(y_train.values.reshape(-1,1)).ravel().astype(int)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# OOF storage
oof_preds = np.zeros(len(X_train))
fold_metrics = []
# Regressor
base_reg = LGBMRegressor(n_estimators=LGB_ESTIMATORS, learning_rate=LGB_LEARNING_RATE, n_jobs=-1, random_state=RANDOM_STATE)
pipeline = Pipeline([("preproc", preprocessor), ("reg", base_reg)])





for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_bins), start=1):
    print(f"\n=== Fold {fold} ===")
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    # fit preprocessor on train
    pipeline.named_steps['preproc'].fit(X_tr)
    X_tr_t = pipeline.named_steps['preproc'].transform(X_tr)
    X_val_t = pipeline.named_steps['preproc'].transform(X_val)

    # fit regressor with early stopping (use valid set)
    pipeline.named_steps['reg'].fit(
        X_tr_t, y_tr,
        eval_set=[(X_val_t, y_val)],
        eval_metric='l2',
    )

    # predict on validation
    val_pred = pipeline.named_steps['reg'].predict(X_val_t)
    # clip predictions to [0,1]
    val_pred = np.clip(val_pred, 0.0, 1.0)
    oof_preds[val_idx] = val_pred

    # metrics
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    mae = mean_absolute_error(y_val, val_pred)
    fold_metrics.append({"fold": fold, "rmse": rmse, "mae": mae})
    print(f"Fold {fold} RMSE: {rmse:.6f}, MAE: {mae:.6f}")



print("\nCV fold metrics:\n", pd.DataFrame(fold_metrics))
oof_rmse = mean_squared_error(y_train, oof_preds, squared=False)
oof_mae = mean_absolute_error(y_train, oof_preds)
print(f"\nOOF RMSE: {oof_rmse:.6f}, OOF MAE: {oof_mae:.6f}")



# Final fit on full training data (use small val inside train for early stopping)
pipeline.named_steps['preproc'].fit(X_train)
X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)
pipeline.named_steps['reg'].fit(
    pipeline.named_steps['preproc'].transform(X_tr_full), y_tr_full,
    eval_set=[(pipeline.named_steps['preproc'].transform(X_val_full), y_val_full)],
    eval_metric='l2'
)

# Evaluate on holdout
X_hold_t = pipeline.named_steps['preproc'].transform(X_hold)
hold_pred = pipeline.named_steps['reg'].predict(X_hold_t)
hold_pred = np.clip(hold_pred, 0.0, 1.0)
hold_rmse = mean_squared_error(y_hold, hold_pred, squared=False)
hold_mae = mean_absolute_error(y_hold, hold_pred)
print(f"\nHoldout RMSE: {hold_rmse:.6f}, Holdout MAE: {hold_mae:.6f}")


# Plot actual vs predicted (holdout)
plt.figure(figsize=(6,6))
plt.scatter(y_hold.sample(n=min(2000,len(y_hold)), random_state=RANDOM_STATE),
            pd.Series(hold_pred, index=y_hold.index).loc[y_hold.sample(n=min(2000,len(y_hold)), random_state=RANDOM_STATE).index],
            alpha=0.3, s=8)
plt.plot([0,1],[0,1], linestyle='--', color='k')
plt.xlabel("True accident_risk")
plt.ylabel("Predicted accident_risk")
plt.title("Holdout: true vs predicted")
plt.grid(True)
plt.show()


# Feature importance
pipeline.named_steps['preproc'].fit(X_train)
feature_names = get_feature_names_from_ct(pipeline.named_steps['preproc'])
X_hold_df = pd.DataFrame(X_hold_t, columns=feature_names)
reg = pipeline.named_steps['reg']
fi = pd.DataFrame({"feature": feature_names, "importance": reg.feature_importances_}).sort_values("importance", ascending=False)
print(fi.head(20))
plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=fi.head(15))
plt.title("LGB Regressor feature importances")
plt.show()



# SHAP (sample to keep runtime reasonable)
explainer = shap.TreeExplainer(reg)
sample_for_shap = X_hold_df.sample(n=min(2000, len(X_hold_df)), random_state=RANDOM_STATE)
shap_vals = explainer.shap_values(sample_for_shap)
shap.summary_plot(shap_vals, sample_for_shap, feature_names=feature_names, show=True)





# Save pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/accident_pipeline_lgb_reg.pkl", compress=3)
print("Saved pipeline to models/accident_pipeline_lgb_reg.pkl")


# define feat_cols from the training features you used earlier
feat_cols = X.columns.tolist()   # X must be the DataFrame you used for training features
print("Using feature columns:", feat_cols)

# then run the test inference block
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
# make sure test has the expected id column
id_col = "id"   # change if different
missing_in_test = [c for c in feat_cols if c not in test_df.columns]
if missing_in_test:
    print("Warning: missing columns in test (will add as NaN):", missing_in_test)
    for c in missing_in_test:
        test_df[c] = np.nan

# drop extra columns we don't need
extra_cols = [c for c in test_df.columns if c not in feat_cols + [id_col]]
if extra_cols:
    print("Dropping extra columns from test:", extra_cols)
    test_df = test_df.drop(columns=extra_cols)

test_features = test_df[feat_cols]
test_t = pipeline.named_steps['preproc'].transform(test_features)
test_preds = pipeline.named_steps['reg'].predict(test_t)
test_preds = np.clip(test_preds, 0.0, 1.0)

submission = pd.DataFrame({id_col: test_df[id_col], "accident_risk": test_preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")



# Accident Risk - End-to-end pipeline with ensemble (script / notebook-ready via cell markers)
# Author: ChatGPT
# Usage: This file is organized with "# %%" cells so it can be opened as a Jupyter notebook (or run as a script).
# It builds preprocessing, several base regressors, a stacking ensemble, runs K-fold CV collecting OOF preds,
# evaluates with RMSE (primary), creates a holdout evaluation, produces SHAP plots, and saves the final pipeline.

# %%
# === Imports & Parameters ===
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.inspection import permutation_importance
import joblib

# try optional libs
try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# Parameters (edit for Kaggle paths)
CSV_PATH = '/kaggle/input/playground-series-s5e10/train.csv'  # << UPDATE THIS PATH
TEST_PATH = '/kaggle/input/playground-series-s5e10/test.csv'     # optional: update if you have test set
RANDOM_STATE = 42
EDA_SAMPLE_FRAC = 0.05  # smaller for big datasets
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
LGB_ESTIMATORS = 2000
LGB_LEARNING_RATE = 0.05
OUTPUT_DIR = 'models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column definitions (adapt if your column names differ)
id_col = 'id'
target_col = 'accident_risk'  # continuous in [0,1]

cat_cols = ['road_type', 'weather']
ord_cols = ['lighting', 'time_of_day']
bool_cols = ['road_sign', 'public_road', 'school_seaon', 'holiday']
num_cols = ['num_lane', 'curvature', 'speed_limit']
optional_cols = ['num_reported']

# %%
# === Utility functions & feature engineering ===

def add_features(df_):
    df = df_.copy()
    # interaction
    if set(['speed_limit','curvature']).issubset(df.columns):
        df['speed_x_curv'] = df['speed_limit'].astype(float) * df['curvature'].astype(float)
    # curvature bins
    if 'curvature' in df.columns:
        df['curv_bin'] = pd.cut(df['curvature'].astype(float).fillna(0),
                                 bins=[-1, 0.05, 0.2, 0.5, 1.0], labels=['flat','slight','mod','sharp'])
    # school peak
    if 'school_seaon' in df.columns and 'time_of_day' in df.columns:
        df['school_peak'] = ((df['school_seaon']==1) & (df['time_of_day']=='morning')).astype(int)
    # night_highspeed
    if 'lighting' in df.columns and 'speed_limit' in df.columns:
        df['night_highspeed'] = ((df['lighting']=='night') & (df['speed_limit']>=50)).astype(int)
    return df


def safe_bool_cast(df, bool_cols):
    for b in bool_cols:
        if b in df.columns:
            if df[b].dtype == object:
                df[b] = df[b].str.lower().map({'true':1,'false':0,'yes':1,'no':0,'1':1,'0':0}).astype(pd.Int64Dtype())
            else:
                df[b] = pd.to_numeric(df[b], errors='coerce').astype(pd.Int64Dtype())
    return df


# helper: build ColumnTransformer given present columns
from sklearn.pipeline import make_pipeline

def build_preprocessor(df):
    # determine present cols
    num_present = [c for c in num_cols if c in df.columns]
    ord_present = [c for c in ord_cols if c in df.columns]
    cat_present = [c for c in cat_cols if c in df.columns]
    bool_present = [c for c in bool_cols if c in df.columns]

    lighting_order = ['daylight','dim','night']
    time_order = ['morning','afternoon','evening']
    ord_categories = []
    for c in ord_cols:
        if c in df.columns:
            if c == 'lighting':
                ord_categories.append(lighting_order)
            elif c == 'time_of_day':
                ord_categories.append(time_order)
            else:
                ord_categories.append(sorted(df[c].dropna().unique().tolist()))

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    ord_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ord', OrdinalEncoder(categories=ord_categories, dtype=int))]) if ord_present else None
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))]) if cat_present else None
    bool_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))]) if bool_present else None

    transformers = []
    if num_present: transformers.append(('num', num_pipe, num_present))
    if ord_present: transformers.append(('ord', ord_pipe, ord_present))
    if cat_present: transformers.append(('cat', cat_pipe, cat_present))
    if bool_present: transformers.append(('bool', bool_pipe, bool_present))

    preprocessor = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    return preprocessor


# get feature names from fitted ColumnTransformer
def get_feature_names_from_ct(ct):
    names = []
    for name, trans, cols in ct.transformers_:
        if name == 'remainder' or trans == 'drop':
            continue
        if hasattr(trans, 'named_steps') and 'ohe' in trans.named_steps:
            ohe = trans.named_steps['ohe']
            names.extend(ohe.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols)
    return names

# %%
# === Load data & basic checks ===
print('Loading dataset from:', CSV_PATH)
df = pd.read_csv(CSV_PATH)
print('Loaded shape:', df.shape)

# sanity check
expected = [id_col, target_col] + cat_cols + ord_cols + bool_cols + num_cols + optional_cols
missing = [c for c in expected if c not in df.columns]
if missing:
    print('WARNING: Missing expected columns (update column lists if these are real):', missing)

# drop id if present, keep other columns
use_cols = [c for c in expected if c in df.columns and c != id_col]
df = df[use_cols].copy()

# cast target to float and ensure in [0,1]
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataframe")

df[target_col] = df[target_col].astype(float).clip(0.0, 1.0)

# bool casting
df = safe_bool_cast(df, bool_cols)

# feature engineering
df = add_features(df)

# update feature lists to include engineered cols if present
if 'speed_x_curv' in df.columns and 'speed_x_curv' not in num_cols:
    num_cols.append('speed_x_curv')
if 'curv_bin' in df.columns and 'curv_bin' not in cat_cols:
    cat_cols.append('curv_bin')
if 'school_peak' in df.columns and 'school_peak' not in bool_cols:
    bool_cols.append('school_peak')
if 'night_highspeed' in df.columns and 'night_highspeed' not in bool_cols:
    bool_cols.append('night_highspeed')

print('Final feature candidates sizes: num=%d, cat=%d, ord=%d, bool=%d' % (
    sum([1 for c in num_cols if c in df.columns]), sum([1 for c in cat_cols if c in df.columns]),
    sum([1 for c in ord_cols if c in df.columns]), sum([1 for c in bool_cols if c in df.columns])
))

# %%
# === Prepare X/y and split holdout ===
features = [c for c in df.columns if c != target_col]
X = df[features].copy()
y = df[target_col].copy()

# train/holdout split
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
print('Train / Hold shapes:', X_train.shape, X_hold.shape)

# %%
# === Build preprocessor and fit on training data ===
preprocessor = build_preprocessor(X_train)
preprocessor.fit(X_train)

feature_names = get_feature_names_from_ct(preprocessor)
print('Transformed feature count:', len(feature_names))

# transform a sample for sanity
X_train_t = preprocessor.transform(X_train)

# %%
# === Define base models ===
models = {}
# LightGBM (if available) - tuned defaults
if _HAS_LGB:
    models['lgb'] = LGBMRegressor(n_estimators=LGB_ESTIMATORS, learning_rate=LGB_LEARNING_RATE, n_jobs=-1, random_state=RANDOM_STATE)
else:
    print('LightGBM not available; skipping lgb model')

# XGBoost optional
if _HAS_XGB:
    models['xgb'] = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1)

# CatBoost optional
if _HAS_CAT:
    models['cat'] = CatBoostRegressor(iterations=1000, learning_rate=0.05, random_state=RANDOM_STATE)

# Strong sklearn models
models['rf'] = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
models['hgb'] = HistGradientBoostingRegressor(max_iter=1000, random_state=RANDOM_STATE)

print('Models to train:', list(models.keys()))

# %%
# === Cross-validated training: OOF predictions for each base model ===
# We'll stratify by binned y so folds have similar distribution
kb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
# fit on train only
y_bins = kb.fit_transform(y_train.values.reshape(-1,1)).ravel().astype(int)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# storage for OOF
oof_preds = {name: np.zeros(len(X_train)) for name in models.keys()}

fold_metrics = {name: [] for name in models.keys()}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_bins), start=1):
    print(f'--- Fold {fold} ---')
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    # fit preprocessor on fold train (to mimic realistic pipeline) and transform
    preprocessor.fit(X_tr)
    X_tr_t = preprocessor.transform(X_tr)
    X_val_t = preprocessor.transform(X_val)

    for name, model in models.items():
        print(' Training', name)
        # try to use early stopping if model supports it (LGB/XGB/Cat)
        if name == 'lgb' and _HAS_LGB:
            model.set_params(**{'n_estimators': LGB_ESTIMATORS})
            model.fit(X_tr_t, y_tr)
            preds = model.predict(X_val_t)
        elif name == 'xgb' and _HAS_XGB:
            model.set_params(**{'n_estimators':1000})
            model.fit(X_tr_t, y_tr)
            preds = model.predict(X_val_t)
        elif name == 'cat' and _HAS_CAT:
            model.fit(X_tr_t, y_tr)
            preds = model.predict(X_val_t)
        else:
            # sklearn model
            model.fit(X_tr_t, y_tr)
            preds = model.predict(X_val_t)

        preds = np.clip(preds, 0.0, 1.0)
        oof_preds[name][val_idx] = preds
        rmse = mean_squared_error(y_val, preds, squared=False)
        mae = mean_absolute_error(y_val, preds)
        fold_metrics[name].append({'fold': fold, 'rmse': rmse, 'mae': mae})
        print(f'  {name} fold RMSE: {rmse:.6f}, MAE: {mae:.6f}')

# summarize CV
for name in models.keys():
    dfm = pd.DataFrame(fold_metrics[name])
    print(f"\nModel {name} CV RMSE mean/std: {dfm['rmse'].mean():.6f} +/- {dfm['rmse'].std():.6f}")

# %%
# === Build stacking dataset (OOF features) ===
# create DataFrame of OOF preds (columns: model names)
oof_df = pd.DataFrame({name: oof_preds[name] for name in models.keys()})
oof_df['y'] = y_train.values
print('OOF training set head:')
print(oof_df.head())

# simple ensemble: average of models
oof_mean = oof_df[list(models.keys())].mean(axis=1)
print('OOF mean RMSE:', mean_squared_error(y_train, oof_mean, squared=False))

# stacking meta-model: use RidgeCV (fast) or LGBM if available
meta_X = oof_df[list(models.keys())].values
meta_y = oof_df['y'].values
meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta_model.fit(meta_X, meta_y)
print('Meta coeffs:', meta_model.coef_)

# compute OOF stacked preds
stacked_oof = meta_model.predict(meta_X)
print('Stacked OOF RMSE:', mean_squared_error(meta_y, np.clip(stacked_oof,0,1), squared=False))

# %%
# === Final training: fit base models on full training data, build final stacking pipeline ===
# fit preprocessor on full X_train
preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_hold_t = preprocessor.transform(X_hold)

fitted_models = {}
for name, model in models.items():
    print('Fitting', name, 'on full training data')
    if name == 'lgb' and _HAS_LGB:
        model.set_params(**{'n_estimators': LGB_ESTIMATORS})
        model.fit(X_train_t, y_train)
    elif name == 'xgb' and _HAS_XGB:
        model.set_params(**{'n_estimators':1000})
        model.fit(X_train_t, y_train)
    elif name == 'cat' and _HAS_CAT:
        model.fit(X_train_t, y_train)
    else:
        model.fit(X_train_t, y_train)
    fitted_models[name] = model

# create meta-features for holdout by predicting with base models
meta_hold = np.column_stack([np.clip(fitted_models[name].predict(X_hold_t),0,1) for name in fitted_models.keys()])
# meta model: use the RidgeCV trained on OOF above
hold_pred_stack = meta_model.predict(meta_hold)
hold_pred_stack = np.clip(hold_pred_stack, 0.0, 1.0)

# evaluate holdout for each base and ensemble
for i, name in enumerate(fitted_models.keys()):
    pred = meta_hold[:, i]
    print(f"Holdout {name} RMSE:", mean_squared_error(y_hold, pred, squared=False))
print('Holdout mean-ensemble RMSE:', mean_squared_error(y_hold, meta_hold.mean(axis=1), squared=False))
print('Holdout stacked RMSE:', mean_squared_error(y_hold, hold_pred_stack, squared=False))

# %%
# === Save final pipeline + models ===
# We'll save: preprocessor, fitted base models dict, meta model
joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'preprocessor.joblib'))
joblib.dump(fitted_models, os.path.join(OUTPUT_DIR, 'base_models.joblib'))
joblib.dump(meta_model, os.path.join(OUTPUT_DIR, 'meta_model.joblib'))
print('Saved preprocessor, base models, and meta model to', OUTPUT_DIR)

# Helpful wrapper for inference
def predict_ensemble(df_input):
    df_proc = add_features(df_input.copy())
    df_proc = safe_bool_cast(df_proc, bool_cols)
    Xf = df_proc[[c for c in features if c in df_proc.columns]]
    Xt = preprocessor.transform(Xf)
    base_preds = np.column_stack([np.clip(fitted_models[name].predict(Xt),0,1) for name in fitted_models.keys()])
    stacked = meta_model.predict(base_preds)
    return np.clip(stacked, 0.0, 1.0), base_preds

# %%
# === SHAP explanation (optional, if SHAP & LGB available) ===
if _HAS_SHAP and ('lgb' in fitted_models):
    print('Computing SHAP for LGB model (sample)...')
    lgb_model = fitted_models['lgb']
    # compute SHAP on a sample
    Xs = X_hold_t
    # convert Xt to DataFrame with feature names
    try:
        Xs_df = pd.DataFrame(Xs, columns=feature_names)
    except Exception:
        Xs_df = pd.DataFrame(Xs)
    sample_for_shap = Xs_df.sample(n=min(2000, len(Xs_df)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(lgb_model)
    shap_vals = explainer.shap_values(sample_for_shap)
    shap.summary_plot(shap_vals, sample_for_shap, feature_names=feature_names, show=True)
else:
    print('SHAP not available or no LGB model fitted; skipping SHAP')

# %%
# === Inference on test set and create submission (if test path exists) ===
if os.path.exists(TEST_PATH):
    print('Loading test set from', TEST_PATH)
    test_df = pd.read_csv(TEST_PATH)
    # make sure to apply same feature engineering
    test_df_proc = add_features(test_df)
    test_df_proc = safe_bool_cast(test_df_proc, bool_cols)
    # align columns
    missing_cols = [c for c in features if c not in test_df_proc.columns]
    for c in missing_cols:
        test_df_proc[c] = np.nan
    test_features = test_df_proc[features]
    preds_stack, base_preds_test = predict_ensemble(test_df_proc)
    submission = pd.DataFrame({id_col: test_df[id_col], target_col: preds_stack})
    submission.to_csv('submission.csv', index=False)
    print('Saved submission.csv (stacked predictions)')
else:
    print('No TEST_PATH found - skip test inference. Update TEST_PATH to generate submission.')

# %%
# === Quick reporting ===
print('\nFinal holdout evaluation summary:')
print('Stacked holdout RMSE: {:.6f}'.format(mean_squared_error(y_hold, hold_pred_stack, squared=False)))
print('Stacked holdout MAE : {:.6f}'.format(mean_absolute_error(y_hold, hold_pred_stack)))

print('\nTop feature importances (from RandomForest, if present):')
if 'rf' in fitted_models:
    try:
        rf = fitted_models['rf']
        perm = permutation_importance(rf, X_hold_t, y_hold, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        idx = np.argsort(perm.importances_mean)[::-1]
        for i in idx[:15]:
            print(f"{feature_names[i]:30s} : importance mean={perm.importances_mean[i]:.6f} std={perm.importances_std[i]:.6f}")
    except Exception as e:
        print('Permutation importance failed:', e)
else:
    print('RandomForest not in fitted models; cannot show RF importances')

print('\nPipeline complete. Files saved to', OUTPUT_DIR)




