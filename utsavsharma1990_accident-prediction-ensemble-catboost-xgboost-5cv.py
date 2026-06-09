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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")



# ----------------------------
# 2. Load data
# ----------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


# 2. Calculate the correlation matrix
# The .corr() method in pandas calculates pairwise correlation coefficients
correlation_matrix = train.select_dtypes(include=['float64', 'int64']).corr()

# 3. Create the heatmap using seaborn
# Define the plot size for better readability
plt.figure(figsize=(10, 8))

# Use sns.heatmap() to create the visualization
sns.heatmap(
    correlation_matrix,
    annot=True,          # Display the correlation values on the heatmap
    cmap='coolwarm',     # Choose a divergent colormap for better visual distinction
    vmin=-1,             # Set the minimum color value to -1
    vmax=1,              # Set the maximum color value to 1
    fmt=".2f",           # Format the annotations to 2 decimal places
    linewidths=.5        # Add lines between cells for clarity
)

# 4. Add titles and show the plot
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.show()


# Target variable
y = train["accident_risk"]
X = train.drop(columns=["id", "accident_risk"]).copy()


# ----------------------------
# 3. Basic preprocessing
# ----------------------------

# Identify columns by type
cat_cols  = ['road_type', 'lighting', 'weather', 'time_of_day']
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
num_cols  = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

# Convert booleans to integers (0/1)
X[bool_cols] = X[bool_cols].astype(int)
test[bool_cols] = test[bool_cols].astype(int)

# Convert categoricals to pandas "category" dtype
for c in cat_cols:
    X[c] = X[c].astype("category")
    # Align test set categories with training
    test[c] = pd.Categorical(test[c], categories=X[c].cat.categories)


# ----------------------------
# 4. Simple feature engineering
# ----------------------------
X["speed_curvature"] = X["speed_limit"] * X["curvature"]
test["speed_curvature"] = test["speed_limit"] * test["curvature"]

X["lanes_speed_ratio"] = X["speed_limit"] / (X["num_lanes"] + 0.1)
test["lanes_speed_ratio"] = test["speed_limit"] / (test["num_lanes"] + 0.1)

# Curvature bins (categorical)
X["curvature_bin"] = pd.qcut(X["curvature"], q=10, labels=False)
test["curvature_bin"] = pd.qcut(test["curvature"], q=10, labels=False, duplicates='drop')

# Optional: ensure it's categorical for LightGBM
X["curvature_bin"] = X["curvature_bin"].astype("category")
test["curvature_bin"] = pd.Categorical(test["curvature_bin"], categories=X["curvature_bin"].cat.categories)

#Convert Speed limit into category
X['Speed_Category'] = np.where(X['speed_limit'] > 50, 'High speed', 'Low speed')
X['Speed_Category'] = X['Speed_Category'].astype("category")
test['Speed_Category'] = np.where(test['speed_limit'] > 50, 'High speed', 'Low speed')
test['Speed_Category'] = test['Speed_Category'].astype("category")



X.head()


X_test = test.drop(columns=["id"]).copy()


# Grid Search for XGBoost
xgb_param_grid = {
    'n_estimators': [200, 500,1000,5000,7000],
    'max_depth': [3, 5,7,9],
    'learning_rate': [0.05, 0.1, 0.2]
    
}
xgb_grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42, enable_categorical=True),
    param_grid=xgb_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
xgb_grid_search.fit(X, y) 
best_xgb_params = xgb_grid_search.best_params_
print(f"Best XGBoost Hyperparameters: {best_xgb_params}")


# Example Grid Search for CatBoost
# Define categorical features for CatBoost
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 'public_road','curvature_bin','Speed_Category']

cat_param_grid = {
    'iterations': [100,300, 500,1000,5000,7000],
    'learning_rate': [0.05, 0.1,0.2],
    'depth': [4, 6,9]
}
cat_grid_search = GridSearchCV(
    estimator=CatBoostRegressor(verbose=0, random_seed=42),
    param_grid=cat_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
cat_grid_search.fit(X, y, cat_features=categorical_features)
best_cat_params = cat_grid_search.best_params_
print(f"Best CatBoost Hyperparameters: {best_cat_params}")


# Ensemble to catboost & XGBoost with 5 cvfold
# --- Configuration ---
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Define categorical features for CatBoost
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 'public_road','curvature_bin','Speed_Category']

# Initialize out-of-fold and test prediction arrays
oof_preds_xgb = np.zeros(len(X))
oof_preds_cat = np.zeros(len(X))
#oof_preds_lgbm = np.zeros(len(X))
test_preds_xgb = []
test_preds_cat = []
#test_preds_lgbm = []

# --- Cross-validation loop for base models ---
for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n===== Fold {fold + 1} =====")

    # Split the data for the current fold
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # --- XGBoost Model ---
    #xgb_model = XGBRegressor(random_state=42, enable_categorical=True)
    xgb_model = XGBRegressor(**best_xgb_params, random_state=42, enable_categorical=True)
    xgb_model.fit(X_tr, y_tr)
    oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb.append(xgb_model.predict(X_test))
    xgb_rmse_val = np.sqrt(mean_squared_error(y_val, xgb_model.predict(X_val)))
    print(f"XGB RMSE (Validation): {xgb_rmse_val:.4f}")
    

    # --- CatBoost Model ---
    #cat_model = CatBoostRegressor(iterations=500,learning_rate=0.1,depth=6,verbose=0, random_seed=42,cat_features=categorical_features
    )
    cat_model = CatBoostRegressor(**best_cat_params, verbose=0, random_seed=42, cat_features=categorical_features)
    cat_model.fit(X_tr, y_tr)
    oof_preds_cat[val_idx] = cat_model.predict(X_val)
    test_preds_cat.append(cat_model.predict(X_test))

    # Calculate and print CatBoost RMSE for the current fold
    cat_rmse_tr = np.sqrt(mean_squared_error(y_tr, cat_model.predict(X_tr)))
    cat_rmse_val = np.sqrt(mean_squared_error(y_val, cat_model.predict(X_val)))
    print(f"CatBoost RMSE (Train): {cat_rmse_tr:.4f}, RMSE (Validation): {cat_rmse_val:.4f}")

    #LGBM Regressor model not
    #lgbm_model = LGBMRegressor(random_state=42)
    #lgbm_model.fit(X_tr, y_tr, categorical_feature = categorical_features)
    #oof_preds_lgbm[val_idx] = lgbm_model.predict(X_val)
    #test_preds_lgbm.append(lgbm_model.predict(X_test))
    #lgbm_rmse_val = np.sqrt(mean_squared_error(y_val, lgbm_model.predict(X_val)))
    #print(f"LGBM RMSE (Validation): {lgbm_rmse_val:.4f}")


# --- Meta-model training ---
# Create the meta-features using the out-of-fold predictions
meta_features = pd.DataFrame({
    'xgb_pred': oof_preds_xgb,
    'cat_pred': oof_preds_cat
    ,#'lgb_pred': oof_preds_lgbm
})

# Train a meta-model on the meta-features
meta_model = LinearRegression()
meta_model.fit(meta_features, y)

# --- Final predictions on the test set ---
# Average the test predictions from the base models
avg_test_preds_xgb = np.mean(test_preds_xgb, axis=0)
avg_test_preds_cat = np.mean(test_preds_cat, axis=0)
#avg_test_preds_lgbm = np.mean(test_preds_lgbm, axis=0)

# Create the meta-features for the test set
test_meta_features = pd.DataFrame({
    'xgb_pred': avg_test_preds_xgb,
    'cat_pred': avg_test_preds_cat
    #,'lgb_pred': avg_test_preds_lgbm
})

# Use the meta-model to generate final predictions
final_predictions = meta_model.predict(test_meta_features)

print("\n===== Ensemble Evaluation =====")

# Calculate and print overall RMSE for the stacking model on the out-of-fold predictions
ensemble_rmse_oof = np.sqrt(mean_squared_error(y, meta_model.predict(meta_features)))
print(f"Ensemble RMSE (Overall Validation/OOF): {ensemble_rmse_oof:.4f}")

# Calculate and print ensemble RMSE on the final test set
# (Note: This is an evaluation on unseen data)
#ensemble_rmse_test = np.sqrt(mean_squared_error(y_test, final_predictions))
#print(f"Ensemble RMSE (Test): {ensemble_rmse_test:.4f}")

print("Final Ensemble Predictions (Stacking):")
print(final_predictions[:10])


# ----------------------------
# 9. Submission
# ----------------------------
submission = pd.DataFrame({
    "id": test["id"],
    "target": final_predictions
})
submission.to_csv("submission.csv", index=False)
print("\n✅ submission.csv saved successfully.")

