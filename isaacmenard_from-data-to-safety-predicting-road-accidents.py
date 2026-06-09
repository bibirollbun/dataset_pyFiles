!pip install tqdm tqdm-joblib


import numpy as np # linear algebra
import seaborn as sns
import pandas as pd
import time, os, gc, random, warnings, math
import seaborn as sb
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import time
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor, Pool
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import numpy as np
import warnings
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.base import clone

warnings.filterwarnings(
    "ignore",
    message=".*Falling back to prediction using DMatrix due to mismatched devices.*"
)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


train['accident_risk'].hist(bins=30)
plt.title("Distribution of Target")
plt.xlabel("accident_risk")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(10,6))
sns.heatmap(train.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


cat_col = "speed_limit"  # change to a categorical column in your dataset
train.groupby(cat_col)["accident_risk"].mean().plot(kind="bar")
plt.ylabel("Mean accident_risk")
plt.title(f"Accident Risk by {cat_col}")
plt.show()


cat_col = "lighting"  # change to a categorical column in your dataset
train.groupby(cat_col)["accident_risk"].mean().plot(kind="bar")
plt.ylabel("Mean accident_risk")
plt.title(f"Accident Risk by {cat_col}")
plt.show()


def preprocessing(df, train):
    df['high_speed'] = df["speed_limit"] > 50
    
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2
    return df

train = preprocessing(train, True)
test = preprocessing(test, False)

train.head(20)


# ===============================
# Prepare data
# ===============================
X = train.drop(columns=['id', 'accident_risk'])
y = train['accident_risk'].astype(float)
test_data = test.drop(columns=['id'])

# Identify feature types
cat_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
num_features = [c for c in X.columns if c not in cat_features]

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

# ===============================
# Build pipeline (CUDA)
# ===============================
pipe = Pipeline(steps=[
    ("prep", make_preprocessor()),
    ("xgb", XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        device="cuda",
        verbosity=0
    ))
])

# ===============================
# Define grid (unchanged)
# ===============================
param_grid = {
    "xgb__n_estimators":     [300],
    "xgb__learning_rate":    [0.026],
    "xgb__max_depth":        [8],
    "xgb__subsample":        [0.9],
    "xgb__colsample_bytree": [1.0],
    "xgb__reg_lambda":       [5.0],
}
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Progress metadata
n_candidates = len(ParameterGrid(param_grid))
n_folds = cv.get_n_splits()
total_fits = n_candidates * n_folds

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=0,
    refit=True
)

print(f"ğŸš€ Starting Grid Searchâ€¦ ({n_folds} folds Ã— {n_candidates} candidates = {total_fits} fits)")
t0 = time.time()
with tqdm_joblib(tqdm(total=total_fits, desc="Grid Search progress")):
    gs.fit(X, y)

elapsed = time.time() - t0
print(f"âœ… Done in {elapsed:.1f}s")
print("Best params (by mean CV):", gs.best_params_)
print("Best mean CV RMSE:", -gs.best_score_)

# ===============================
# Select ONLY the best fold for the best params
# ===============================
cv_results = pd.DataFrame(gs.cv_results_)
best_row = cv_results.loc[gs.best_index_]

# Find which split (fold) had the highest score for the best params.
# Scores are "neg_root_mean_squared_error" (higher is better; less negative).
split_cols = [c for c in cv_results.columns if c.startswith("split") and c.endswith("_test_score")]
best_split_idx = int(np.argmax([best_row[c] for c in split_cols]))
best_split_score = best_row[split_cols[best_split_idx]]
best_split_rmse = -best_split_score

print(f"ğŸ�† Best fold index for best params: {best_split_idx}")
print(f"Best fold RMSE: {best_split_rmse:.5f}")

# Recreate the SAME CV to extract indices for that best fold
# (Same n_splits, shuffle, random_state)
folds = list(cv.split(X, y))
train_idx, valid_idx = folds[best_split_idx]
X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
X_valid_fold, y_valid_fold = X.iloc[valid_idx], y.iloc[valid_idx]

# Build a fresh pipeline with the best params and fit ONLY on the best fold's train split
best_fold_est = clone(pipe).set_params(**{k.replace("xgb__", "xgb__"): v for k, v in gs.best_params_.items()})
best_fold_est.fit(X_train_fold, y_train_fold)

# Evaluate on that fold's validation part (for transparency)
valid_preds = best_fold_est.predict(X_valid_fold)
valid_rmse = mean_squared_error(y_valid_fold, valid_preds, squared=False)
print(f"ğŸ”� Recomputed RMSE on held-out part of best fold: {valid_rmse:.5f}")

# ===============================
# Predict on test set using best-fold refit
# ===============================
print("\nğŸ“¡ Predicting on test data with best-fold modelâ€¦")
test_preds_best = best_fold_est.predict(test_data)
test_preds_best = np.clip(test_preds_best, 0, 1)

# (Optional) Show top 10 parameter sets by mean CV RMSE, unchanged
cv_results["rmse"] = -cv_results["mean_test_score"]
print("\nğŸ”� Top 10 parameter sets by mean CV RMSE:")
print(cv_results.sort_values("rmse").head(10)[["params", "rmse", "mean_fit_time"]])


submission = sub.copy()
submission["accident_risk"] = test_preds_best
submission.to_csv("submission.csv", index=False)
print("Submission saved: submission.csv")

