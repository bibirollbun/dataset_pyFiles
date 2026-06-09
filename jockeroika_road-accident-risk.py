#system handling
import os
import time
import warnings
warnings.filterwarnings('ignore')

#data handling
import numpy as np # linear algebra
import pandas as pd # data processing, 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


#model handling
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler, LabelEncoder



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print('done')


#read data file
train =pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


#print(f'Shape of Train data = {train.shape}')
#print(f'Shape of Test data = {test.shape}')
#print(f'Shape of Submission data = {sub.shape}')


#print("STATISTICAL ANALYSIS: Training Dataset")
# Basic info
#print("Dataset Information:")
#print(f"  Total Records: {len(train):,}")
#print(f"  Total Features: {len(train.columns)}")
#print(f"  Duplicates: {train.duplicated().sum()}")



# Numerical columns
numerical_cols_train = train.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numerical_cols_train:
    numerical_cols_train.remove('id')
if 'accident_risk' in numerical_cols_train and 'accident_risk' in train.columns:
    print(f"\nTarget Variable Statistics (accident_risk):")
    print(f"  Mean: {train['accident_risk'].mean():.4f}")
    print(f"  Std: {train['accident_risk'].std():.4f}")
    print(f"  Min: {train['accident_risk'].min():.4f}")
    print(f"  Max: {train['accident_risk'].max():.4f}")
    print(f"  Median: {train['accident_risk'].median():.4f}")


# Categorical columns
categorical_cols_train = train.select_dtypes(include=['object']).columns.tolist()
if categorical_cols_train:
    print(f"\nCategorical Features: {len(categorical_cols_train)}")
    for col in categorical_cols_train:
        unique_count = train[col].nunique()
        print(f"  {col}: {unique_count} unique values")


# Numerical columns
numerical_cols_test = test.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numerical_cols_test:
    numerical_cols_test.remove('id')
if 'accident_risk' in numerical_cols_test and 'accident_risk' in test.columns:
    print(f"\nTarget Variable Statistics (accident_risk):")
    print(f"  Mean: {test['accident_risk'].mean():.4f}")
    print(f"  Std: {test['accident_risk'].std():.4f}")
    print(f"  Min: {test['accident_risk'].min():.4f}")
    print(f"  Max: {test['accident_risk'].max():.4f}")
    print(f"  Median: {test['accident_risk'].median():.4f}")


# Categorical columns
categorical_cols_test = test.select_dtypes(include=['object']).columns.tolist()
if categorical_cols_test:
    print(f"\nCategorical Features: {len(categorical_cols_test)}")
    for col in categorical_cols_test:
        unique_count = test[col].nunique()
        print(f"  {col}: {unique_count} unique values")


#target visualization
target_col = "accident_risk"

if target_col not in train.columns:
    print(f"Target column '{target_col}' not found in dataset.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram with KDE
    axes[0].hist(train[target_col], bins=50, density=True, alpha=0.7, 
                 color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Accident Risk', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Distribution of Accident Risk (Training Data)', 
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot for normality check
    stats.probplot(train[target_col], dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot: Normality Assessment', 
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Statistical tests
    shapiro_stat, shapiro_p = stats.shapiro(train[target_col].sample(min(5000, len(train))))
    print("\nShapiro-Wilk Test for Normality:")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  P-value: {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print("  Interpretation: Data is approximately Normal distribution")
    else:
        print("  Interpretation: Data is NOT Normal distribution")



CATEGORICAL_FEATURES = ['road_type', 'lighting', 'weather', 'time_of_day']
BOOLEAN_FEATURES = ['road_signs_present', 'public_road', 'holiday', 'school_season']
NUMERICAL_FEATURES = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
TARGET = 'accident_risk'
ID_COL = 'id'


def engineer_features(df):
    """
    Create domain-informed feature interactions.
    """
    df_eng = df.copy()
    # --- Base physical interactions ---
    df_eng['curv_speed'] = df_eng['curvature'] * df_eng['speed_limit']
    df_eng['curv_sq'] = df_eng['curvature']**2
    df_eng['speed_sq'] = df_eng['speed_limit']**2
    df_eng['curv_x_acc'] = df_eng['curvature'] * np.log1p(df_eng['num_reported_accidents'])
    df_eng['speed_x_acc'] = df_eng['speed_limit'] * np.log1p(df_eng['num_reported_accidents'])
    df_eng['curv_speed_acc'] = df_eng['curv_speed'] * np.log1p(df_eng['num_reported_accidents'])
    
    # --- Ratio features ---
    df_eng['acc_per_lane'] = df_eng['num_reported_accidents'] / (df_eng['num_lanes'] + 1)
    df_eng['curv_per_lane'] = df_eng['curvature'] / (df_eng['num_lanes'] + 1)
    df_eng['risk_density'] = df_eng['curv_speed'] / (df_eng['num_lanes'] + 1)

    # --- Nonlinear transforms ---
    df_eng['curv_log'] = np.log1p(df_eng['curvature'])
    df_eng['speed_log'] = np.log1p(df_eng['speed_limit'])
    df_eng['acc_log'] = np.log1p(df_eng['num_reported_accidents'])
    df_eng['inv_speed'] = 1 / (df_eng['speed_limit'] + 1)
    
     # --- Statistical combinations ---
    df_eng['risk_index'] = (df_eng['curv_speed'] * df_eng['acc_per_lane']) / (df_eng['speed_limit'] + 1)
    df_eng['stability_score'] = (df_eng['num_lanes'] / (1 + df_eng['curvature'])) * df_eng['speed_limit']
    
    # --- Binary conditions ---
    df_eng['tight_lane'] = (df_eng['num_lanes'] <= 2).astype(int)
    df_eng['sharp_curve'] = (df_eng['curvature'] > 0.6).astype(int)
    df_eng['high_speed_zone'] = (df_eng['speed_limit'] > 80).astype(int)
    df_eng['critical_zone'] = ((df_eng['sharp_curve']==1) & (df_eng['high_speed_zone']==1)).astype(int)
    

    
    # --- Polynomial mixes for smoother nonlinearity ---
    df_eng['poly_mix1'] = np.sqrt(df_eng['curvature'] * df_eng['speed_limit'])
    df_eng['poly_mix2'] = (df_eng['num_reported_accidents']**0.3) * df_eng['speed_limit']
    
    return df_eng

# Preprocessing
train_processed = train.copy()
test_processed = test.copy()

# Convert booleans
for col in BOOLEAN_FEATURES:
    train_processed[col] = train_processed[col].astype(int)
    test_processed[col] = test_processed[col].astype(int)

# Label encode categoricals
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    train_processed[f'{col}_enc'] = le.fit_transform(train_processed[col])
    test_processed[f'{col}_enc'] = le.transform(test_processed[col])
    label_encoders[col] = le

# Apply feature engineering
train_engineered = engineer_features(train_processed)
test_engineered = engineer_features(test_processed)

print(f"Feature engineering complete")
print(f"Original features: {len(CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERICAL_FEATURES)}")
print(f"Engineered features: {train_engineered.shape[1]}")
print(f"New features created: {train_engineered.shape[1] - train_processed.shape[1]}")


# Prepare feature matrix
exclude_cols = [ID_COL, TARGET] + CATEGORICAL_FEATURES
feature_cols = [col for col in train_engineered.columns if col not in exclude_cols]

X_train = train_engineered[feature_cols].values
y_train = train_engineered[TARGET].values
X_test = test_engineered[feature_cols].values

print(f"Training matrix: {X_train.shape}")
print(f"Test matrix: {X_test.shape}")


# Compare before vs after
before_cols = set(train_processed.columns)
after_cols = set(train_engineered.columns)
new_features = sorted(after_cols - before_cols)

# Bar Chart: Feature Count Comparison
plt.figure(figsize=(6, 4))
plt.bar(['Before', 'After'], [len(before_cols), len(after_cols)], color=['skyblue', 'lightgreen'])
plt.title('Feature Count Before vs After Engineering')
plt.ylabel('Number of Features')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Table Visualization: List of New Features
comparison_df = pd.DataFrame({
    'New Features': new_features
})

plt.figure(figsize=(6, len(new_features)*0.4))
plt.axis('off')
plt.title("Newly Created Features", fontsize=14, pad=10)
table = plt.table(cellText=comparison_df.values,
                  colLabels=comparison_df.columns,
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.show()



xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.003,
    "max_depth": 8,
    "min_child_weight": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bynode": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "gamma": 0.1,
    "n_estimators": 6000,
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "device": "cuda",
    "seed": 42,
    "random_state": 42
}


import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

#  Split validation data
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)


# âš™ï¸� OPTUNA HYPERPARAMETER TUNING FOR EACH MODEL

def objective_xgb(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "device": "cuda",
        "n_estimators": 4000,
        "random_state": 42
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)

def objective_lgb(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "n_estimators": 4000,
        "device_type": "gpu",
        "random_state": 42
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)

def objective_cat(trial):
    params = {
        "iterations": 4000,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": "GPU",
        "random_seed": 42,
        "verbose": False
    }
    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)


# ğŸš€ TRAIN FINAL MODELS WITH OPTIMAL SETTINGS

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=False)

lgb_model = lgb.LGBMRegressor(
    learning_rate=0.003,
    n_estimators=6000,
    max_depth=8,
    num_leaves=128,
    colsample_bytree=0.8,
    subsample=0.9,
    reg_alpha=0.5,
    reg_lambda=1.5,
    objective="regression",
    device_type="gpu",
    random_state=42
)
from lightgbm import early_stopping, log_evaluation

lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(200),
        log_evaluation(period=0)  # disables LightGBM logging
    ]
)

cat_model = CatBoostRegressor(
    iterations=6000,
    learning_rate=0.003,
    depth=8,
    l2_leaf_reg=4,
    subsample=0.9,                     # keep subsample
    bootstrap_type='Bernoulli',        # REQUIRED FIX
    task_type="GPU",
    eval_metric="RMSE",
    random_seed=42,
    verbose=False
)

cat_model.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    early_stopping_rounds=200,
    verbose=False
)


# âš–ï¸� ENSEMBLE PREDICTION (Weighted Average)
val_preds = (
    0.30 * xgb_model.predict(X_val) +
    0.35 * lgb_model.predict(X_val) +
    0.35 * cat_model.predict(X_val))

rmse = np.sqrt(mean_squared_error(y_val, val_preds))
r2 = r2_score(y_val, val_preds)

print(f"\nâœ… Ensemble RMSE: {rmse:.4f}")
print(f" Ensemble RÂ²: {r2:.4f}")

# ğŸ§¾ TEST PREDICTIONS & SUBMISSION
test_preds = (
    0.30 * xgb_model.predict(X_test) +
    0.35 * lgb_model.predict(X_test) +
    0.35 * cat_model.predict(X_test) )



#  Create submission file

test_preds_clipped = np.clip(test_preds, 0, 1)

submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': test_preds_clipped
})

# Validation checks
assert submission.shape[0] == test.shape[0], "Shape mismatch between test and submission!"
assert submission['accident_risk'].isna().sum() == 0, "There are missing predictions!"
assert (submission['accident_risk'] >= 0).all(), "Some predictions are below 0!"
assert (submission['accident_risk'] <= 1).all(), "Some predictions exceed 1!"

#  Save submission file
submission.to_csv('/kaggle/working/submission.csv', index=False)

#  Summary info
print(" Submission Created Successfully")
print("=" * 60)
print(f"Shape: {submission.shape}")
print(f"Prediction Mean: {submission['accident_risk'].mean():.4f}")
print(f"Prediction Std: {submission['accident_risk'].std():.4f}")
print(f"Prediction Min: {submission['accident_risk'].min():.4f}")
print(f"Prediction Max: {submission['accident_risk'].max():.4f}")
print("\nFirst 10 predictions:")
print(submission.head(10))



