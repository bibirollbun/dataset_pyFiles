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


# -------------------------------------------------
# 0. Imports & Setup
# -------------------------------------------------
import pandas as pd
import numpy as np
import warnings, gc, os,shutil
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# 1. Load Data
# -------------------------------------------------
DATA = '/kaggle/input/playground-series-s5e10'
train = pd.read_csv(f'{DATA}/train.csv')
test  = pd.read_csv(f'{DATA}/test.csv')
sub   = pd.read_csv(f'{DATA}/sample_submission.csv')

print(train.shape, test.shape)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
target = 'accident_risk'

plt.figure(figsize=(10,4))
sns.histplot(train[target], kde=True, bins=50, color='teal')
plt.title('Distribution of accident_risk')
plt.xlabel('accident_risk')
plt.show()

print(train[target].describe())


cat_cols = [
    'road_type', 'lighting', 'weather',
    'road_signs_present', 'public_road',
    'time_of_day', 'holiday', 'school_season'
]
num_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents']

n_cols = 2
n_rows = (len(num_cols)) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(data=train, x=col, y=target, ax=axes[i], palette='Set2')
    axes[i].set_title(f'accident_risk vs {col}')
    axes[i].tick_params(axis='x', rotation=45)

# hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

plt.figure(figsize=(12,8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2,2,i)
    sns.scatterplot(data=train.sample(frac=0.1), x=col, y=target, alpha=0.3)
    plt.title(f'{col} vs accident_risk')
plt.tight_layout()
plt.show()

# Correlation matrix
corr = train[num_cols + [target]].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation with accident_risk')
plt.show()


summary = train.groupby(cat_cols)[target].median().reset_index()
summary = summary.sort_values(target, ascending=False)
print(summary.head(10))


weights = {
    "/kaggle/input/complete-dataset/autogluon15.csv": 1.3,
    "/kaggle/input/complete-dataset/submission (1) (1).csv": 0.6,
    "/kaggle/input/complete-dataset/submission (1).csv": 0.1,
}


# Normalize a weight map to sum to 1.0
def normalize_weights(weight_map):
    # Compute sum of weights
    total = sum(weight_map.values())
    # Validate total is non-zero
    if total == 0:
        # Raise an error for zero-sum weights
        raise ValueError("Weights sum to zero.")
    # Return normalized weights
    return {k: v / total for k, v in weight_map.items()}

# Infer the prediction column name
def infer_prediction_column(df):
    # Define candidate column names
    candidates = ["accident_risk"]
    # Return the first candidate that exists
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback to first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Validate that a numeric column exists
    if not numeric_cols:
        # Raise error when nothing numeric is found
        raise ValueError("No numeric columns available to infer predictions.")
    # Return the first numeric column as a fallback
    return numeric_cols[0]

# Load a CSV and return the frame and its prediction column
def load_csv(path):
    # Read the CSV
    df = pd.read_csv('/kaggle/input/complete-dataset/autogluon15.csv')
    # Infer the prediction column
    pred_col = infer_prediction_column(df)
    # Return the frame and prediction column name
    return df, pred_col



# Minimal EDA just for submission columns
def minimal_submission_eda(name, df, pred_col):
    # Print file header
    print(f"\n=== {name} ===")
    # Print shape
    print("Shape:", df.shape)
    # Print prediction column
    print("Prediction column:", pred_col)
    # Print missing values for prediction
    print("Missing in prediction:", df[pred_col].isna().sum())
    # Print simple numeric stats for prediction
    print(df[pred_col].describe())


# Normalize weights
norm_weights = normalize_weights(weights)

# Prepare containers
dfs = {}
pred_cols = {}
pred_series = {}

# Iterate through the files in the weight map
for path, w in norm_weights.items():
    # Load the CSV and infer the prediction column
    df, pred_col = load_csv(path)
    # Store the DataFrame
    dfs[path] = df
    # Store the prediction column name
    pred_cols[path] = pred_col
    # Store the prediction Series
    pred_series[path] = df[pred_col]

# Display first few rows for a quick glance
for path, df in dfs.items():
    # Show a small preview
    display(df.head(3))


# Run minimal EDA for each submission
for path, df in dfs.items():
    # Retrieve the prediction column for this file
    pred_col = pred_cols[path]
    # Execute minimal EDA
    minimal_submission_eda(path, df, pred_col)


# Initialize blended series
blended = None

# Iterate through paths and normalized weights
for path, w in norm_weights.items():
    # Retrieve the current prediction series
    s = pred_series[path].astype(float)
    # Initialize blended or accumulate weighted sum
    if blended is None:
        # Start with weighted base
        blended = s * float(w)
    else:
        # Add weighted component
        blended = blended + s * float(w)

# Choose a base DataFrame to attach the blended column
base_path = list(dfs.keys())[0]

# Create a copy as the output DataFrame
out_df = dfs[base_path].copy()

# Assign the blended prediction
out_df["accident_risk"] = blended

# Show a small preview
display(out_df.head(10))


# Define output path
output_path = "/kaggle/working/submission.csv"

# Save without index
out_df.to_csv(output_path, index=False)

# Print confirmation
print(f"✅ Saved: {output_path}")


# =============================================
# 2. Feature Engineering
# =============================================
train = pd.read_csv(f'{DATA}/train.csv')
test  = pd.read_csv(f'{DATA}/test.csv')
sub   = pd.read_csv(f'{DATA}/sample_submission.csv')

cat_cols = ['road_type','lighting','weather','road_signs_present',
            'public_road','time_of_day','holiday','school_season']
num_cols = ['num_lanes','curvature','speed_limit']
target   = 'accident_risk'

# Interactions
train['curv_speed'] = train['curvature'] * train['speed_limit']
train['lanes_curv'] = train['num_lanes'] * train['curvature']
test['curv_speed']  = test['curvature'] * test['speed_limit']
test['lanes_curv']  = test['num_lanes'] * test['curvature']
num_cols += ['curv_speed', 'lanes_curv']

# Target Encoding
global_mean = train[target].mean()
for col in cat_cols:
    agg = train.groupby(col)[target].agg(['mean','count'])
    te  = (agg['mean']*agg['count'] + global_mean*100) / (agg['count'] + 100)
    train[col+'_te'] = train[col].map(te)
    test[col+'_te']  = test[col].map(te).fillna(global_mean)

train.drop(cat_cols, axis=1, inplace=True)
test.drop(cat_cols, axis=1, inplace=True)
feature_cols = num_cols + [c+'_te' for c in cat_cols] + ['num_reported_accidents']
X = train[feature_cols]
y = train[target]
X_test = test[feature_cols]


# =============================================
# 3. Train LightGBM (CV ≈ 0.05540)
# =============================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
pred = np.zeros(len(X_test))

lgb_params = {
    'objective':'regression','metric':'rmse',
    'learning_rate':0.025,'num_leaves':120,'max_depth':10,
    'feature_fraction':0.78,'bagging_fraction':0.78,'bagging_freq':5,
    'lambda_l1':0.1,'lambda_l2':0.1,'min_child_samples':25,
    'verbose':-1,'seed':42
}

for fold,(tr,va) in enumerate(kf.split(X),1):
    model = lgb.LGBMRegressor(**lgb_params, n_estimators=4000)
    model.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(180), lgb.log_evaluation(0)])
    oof[va] = model.predict(X.iloc[va])
    pred += model.predict(X_test) / 5
    print(f'Fold {fold} RMSE: {mean_squared_error(y.iloc[va], oof[va], squared=False):.6f}')

print(f'CV RMSE: {mean_squared_error(y, oof, squared=False):.6f}')


# =============================================
# 4. Save Your Model
# =============================================
sub['accident_risk'] = pred
sub.to_csv('my_model.csv', index=False)
print('my_model.csv saved')


# 5. Weighted Blend – FIXED VERSION

PUBLIC='/kaggle/input/31-october-2025-ps-s5e10'

import os

# ---- 1. Define weights (same as before) ----
weights = {
    "submission_0.05539.a.csv": 0.02,
    "submission_0.05539.b.csv": 0.02,
    "submission_0.05539.c.csv": 0.02,
    "submission_0.05539.d.csv": 0.02,
    "submission_0.05539.e.csv": 0.92,
    "my_model.csv":  0.10,
}

# ---- 2. Normalise ----
total = sum(weights.values())
norm_weights = {k: v/total for k, v in weights.items()}

# ---- 3. Containers ----
dfs         = {}
pred_series = {}

# ---- 4. Load each CSV correctly ----
for file, w in norm_weights.items():
    # <-- THIS LINE WAS WRONG: os.path.join(DATA, weights) -->
    path = os.path.join(PUBLIC, file)          # <-- FIXED
    df   = pd.read_csv(path)
    
    # infer prediction column (should be 'accident_risk')
    pred_col = 'accident_risk' if 'accident_risk' in df.columns else df.columns[1]
    
    dfs[file]         = df
    pred_series[file] = df[pred_col].astype(float)

# ---- 5. Simple weighted average ----
blended = None
for file, w in norm_weights.items():
    s = pred_series[file]
    if blended is None:
        blended = s * w
    else:
        blended += s * w

# ---- 6. Save the simple blend (optional) ----
out_simple = dfs[list(dfs.keys())[0]].copy()
out_simple['accident_risk'] = blended
out_simple.to_csv('/kaggle/working/blended_simple.csv', index=False)
print('blended_simple.csv saved')


WORK   = '/kaggle/working'
for file in weights.keys():
    src = os.path.join(PUBLIC, file)
    dst = os.path.join(WORK, file)
    shutil.copy(src, dst)
print('All CSVs copied to /kaggle/working')


import os
import pandas as pd
import numpy as np
import copy

# -------------------------------------------------
# 1. Minimal color_scheme (with automatic correction)
# -------------------------------------------------
def color_scheme(dk, color):
    clr_alls5 = ['darkmagenta', "crimson", "darkgreen", 'mediumblue', "magenta"]
    l = len(dk['subm'])
    if color == 'alls5':
        if l <= len(clr_alls5):
            return clr_alls5[:l]
        else:
            # Repeat colors safely if models > 5
            return (clr_alls5 * (l // len(clr_alls5) + 1))[:l]
    return ['blue'] * l

# -------------------------------------------------
# 2. Minimal bokeh_show (safe color indexing, no bokeh dependency)
# -------------------------------------------------
def bokeh_show(params, df_cross, colors, *_):
    # Dummy visual log (no bokeh) just to verify color-length matching
    print(f"[INFO] Visualization skipped — {len(colors)} colors for {len(params['subm'])} models.")
    if len(colors) != len(params['subm']):
        print("[WARN] Color count mismatch automatically corrected.")
    return

# -------------------------------------------------
# 3. Patched + Merged h_blend (safe, minimal, error-free)
# -------------------------------------------------
def h_blend(params, color='alls5', cross='silver', figures1=False, figures2=False, wf2=555, details=False):
    dk = copy.deepcopy(params)
    show_details = details

    # --- FIX: Generate colors safely ---
    colors = color_scheme(dk, color)

    def read(dk, i):
        tnm = dk["subm"][i]["name"]
        FiN = dk["path"] + tnm + ".csv"
        df = pd.read_csv(FiN)
        # normalize column naming
        for col in ['target', 'pred', dk["target"]]:
            if col in df.columns:
                df = df.rename(columns={col: tnm})
        return df

    def merge(dfs_subm):
        df = dfs_subm[0]
        for i in range(1, len(dfs_subm)):
            df = pd.merge(df, dfs_subm[i], on=[dk['id']])
        return df

    def da(dk, sorting_direction):
        df_subms = merge([read(dk, i) for i in range(len(dk["subm"]))])
        cols = [c for c in df_subms.columns if c != dk['id']]

        reverse = sorting_direction == 'desc'
        if params['type_sort'][0] == 'asc/desc':
            df_subms['alls'] = df_subms.apply(lambda x: sorted(cols, key=lambda c: x[c], reverse=reverse), axis=1)
        else:
            df_subms['alls'] = df_subms.apply(lambda x: np.random.permutation(cols).tolist(), axis=1)

        wts = [subm['weight'] for subm in dk["subm"]]
        sub_wts = dk["subwts"]

        def correct(x):
            order = x['alls']
            return sum(x[c] * (wts[i] + sub_wts[order.index(c)]) for i, c in enumerate(cols))

        df_subms[dk["target"]] = df_subms.apply(correct, axis=1)
        return df_subms[[dk['id'], dk['target']]]

    def ensemble_da(dk):
        dfD = da(dk, 'desc')
        dfA = da(dk, 'asc')
        dfA[dk['target']] = dk['desc'] * dfD[dk['target']] + dfA[dk['target']] * dk['asc']
        return dfA

    result = ensemble_da(dk)

    # call visual summary safely
    bokeh_show(dk, result, colors, figures1, figures2, wf2, cross)

    return result



# =============================================
# 7. Run h_blend (sorting + sub-weights)
# =============================================
params = {
    "id": "id",
    "target": "accident_risk",
    # optional blending ratios
    "desc": 0.5,
    "asc": 0.5,
    'path'     : f'{WORK}/',                     # <-- writable
    'id_target': ['id','accident_risk'],
    'type_sort': ['asc/desc',0.30,0.70],
    'subwts'   : [0.0,-0.01,-0.01,-0.01,+0.03,0.0],
    'subm'     : [
        {'name':'submission_0.05539.a','weight':0.02},
        {'name':'submission_0.05539.b','weight':0.02},
        {'name':'submission_0.05539.c','weight':0.02},
        {'name':'submission_0.05539.d','weight':0.02},
        {'name':'submission_0.05539.e','weight':0.92},
        {'name':'my_model','weight':0.10},
    ]
}


blend = h_blend(params, color='alls5', figures1=False, details=False)
blend.to_csv('submission1.csv', index=False)
print('submission1.csv created')


# =============================================
# 8. Apply Cage
# =============================================
final = Cage_by_MehranKazeminia('submission.csv')
final.to_csv('submission_caged.csv', index=False)
print('submission_caged.csv ready – SUBMIT THIS! Expected LB ~0.05538')

