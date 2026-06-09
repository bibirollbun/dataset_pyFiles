# ---------------------------------------------------------------------------
# Initial Setup and Data Loading
# ---------------------------------------------------------------------------
print("âœ… Importing essential libraries...")

# Core libraries for data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# ML libraries
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings('ignore')

# Setting style for plots
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

print("âœ… Libraries imported successfully.")

# ---------------------------------------------------------------------------
# Loading the Datasets
# ---------------------------------------------------------------------------
print("\nâœ… Loading the competition datasets...")

# Define the base path for the Kaggle input files
BASE_PATH = '/kaggle/input/playground-series-s5e10/'

# Load the training, testing, and sample submission files into pandas DataFrames
train_df = pd.read_csv(f'{BASE_PATH}train.csv')
test_df = pd.read_csv(f'{BASE_PATH}test.csv')
sample_submission_df = pd.read_csv(f'{BASE_PATH}sample_submission.csv')
print("âœ… Datasets loaded successfully.")


# ---------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
# DataFrame Information
# ---------------------------------------------------------------------------

# --- Training Data Info ---
print("\nâ˜‘ï¸� Training Data Info")
train_df.info()

# --- Testing Data Info ---
print("\nâ˜‘ï¸� Testing Data Info")
test_df.info()

# ---------------------------------------------------------------------------
# Statistical Summary
# ---------------------------------------------------------------------------

# --- Training Data Statistical Summary ---
print("\nâ˜‘ï¸� Training Data Summary")
display(train_df.describe())

# --- Testing Data Statistical Summary ---
print("\nâ˜‘ï¸� Testing Data Summary")
display(test_df.describe())

# ---------------------------------------------------------------------------
# Data frames
# ---------------------------------------------------------------------------

# --- Training Data ---
print("\nâ˜‘ï¸� Training Data")
display(train_df)

# --- Testing Data ---
print("\nâ˜‘ï¸� Testing Data")
display(test_df)


# ---------------------------------------------------------------------------
# Target Variable Distribution
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: Histogram and KDE Plot ---
sns.histplot(train_df['accident_risk'], kde=True, ax=axes[0], bins=50)
axes[0].set_title('Distribution of Accident Risk', fontsize=16)
axes[0].set_xlabel('Accident Risk', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)

# --- Plot 2: Box Plot ---
sns.boxplot(x=train_df['accident_risk'], ax=axes[1])
axes[1].set_title('Box Plot of Accident Risk', fontsize=16)
axes[1].set_xlabel('Accident Risk', fontsize=12)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# Categorical Feature Distribution
# ---------------------------------------------------------------------------

categorical_cols = [
    'road_type', 'lighting', 'weather', 'time_of_day',
    'road_signs_present', 'public_road', 'holiday', 'school_season'
]

fig, axes = plt.subplots(4, 2, figsize=(18, 24))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=train_df, ax=axes[i], palette='viridis', order = train_df[col].value_counts().index)
    axes[i].set_title(f'Distribution of {col}', fontsize=16)
    axes[i].set_xlabel(None) # Remove x-axis label for cleaner look
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability

plt.tight_layout(pad=3.0)
plt.show()


# ---------------------------------------------------------------------------
# Numerical Feature Distribution
# ---------------------------------------------------------------------------

numerical_cols = [
    'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents'
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten() # Flatten to a 1D array

for i, col in enumerate(numerical_cols):
    sns.histplot(train_df[col], ax=axes[i], kde=True, bins=30)
    axes[i].set_title(f'Distribution of {col}', fontsize=16)
    axes[i].set_xlabel(None)
    axes[i].set_ylabel('Frequency', fontsize=12)

plt.tight_layout(pad=3.0)
plt.show()


# ---------------------------------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------------------------------

# Boolean columns will be treated as 0s and 1s
correlation_cols = train_df.select_dtypes(include=['int64', 'float64', 'bool']).columns
correlation_matrix = train_df[correlation_cols].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical and Boolean Features', fontsize=18)
plt.show()


# ===========================================================================
# Feature Engineering Function and Label Encoding
# ===========================================================================

def create_all_features(train_df, test_df):
    
    # Combine train and test for consistent processing
    combined_df = pd.concat([train_df.drop('accident_risk', axis=1), test_df], ignore_index=True)
    
    # --- 1. Base Categorical Interactions ---
    categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
    for col1, col2 in combinations(categorical_cols, 2):
        combined_df[f'{col1}_x_{col2}'] = combined_df[col1].astype(str) + "_" + combined_df[col2].astype(str)
    
    # --- 2. Numerical & Binned Features ---
    combined_df['curvature_sq'] = combined_df['curvature'] ** 2
    combined_df['speed_limit_sq'] = combined_df['speed_limit'] ** 2
    combined_df['speed_x_curvature'] = combined_df['speed_limit'] * combined_df['curvature']
    combined_df['speed_bin'] = pd.cut(combined_df['speed_limit'], bins=[0, 35, 60, 100], labels=[0, 1, 2], include_lowest=True).astype(int)
    combined_df['curvature_bin'] = pd.cut(combined_df['curvature'], bins=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2], include_lowest=True).astype(int)

    # --- 3. Domain-Specific Features ---
    combined_df['is_night_time'] = (combined_df['lighting'].isin(['night', 'dim'])).astype(int)
    combined_df['is_bad_weather'] = (combined_df['weather'].isin(['rainy', 'foggy'])).astype(int)
    combined_df['high_risk_speed_curve'] = ((combined_df['speed_limit'] > 50) & (combined_df['curvature'] > 0.6)).astype(int)
    combined_df['danger_score'] = (
        combined_df['is_night_time'] +
        combined_df['is_bad_weather'] +
        combined_df['high_risk_speed_curve'] +
        (combined_df['num_reported_accidents'] >= 2).astype(int)
    )
    
    # --- 4. Ratio Feature ---
    combined_df['accidents_per_lane'] = combined_df['num_reported_accidents'] / (combined_df['num_lanes'] + 0.001)

    # --- 5. Label Encoding ---
    # Label encode all columns that are of 'object' type
    object_cols = combined_df.select_dtypes(include=['object']).columns
    for col in object_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
        
    # --- 6. Boolean to Int Encoding ---
    bool_cols = combined_df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        combined_df[col] = combined_df[col].astype(int)
        
    # Separate back into train and test
    X = combined_df.iloc[:len(train_df)]
    X_test = combined_df.iloc[len(train_df):]
    y = train_df['accident_risk']
    
    print("âœ… All features created and encoded successfully.")
    
    return X, y, X_test

# Apply the single feature engineering function
X_final, y_final, X_test_final = create_all_features(train_df, test_df)

# Define feature list
features = [col for col in X_final.columns if col not in ['id']]
X_final = X_final[features]
X_test_final = X_test_final[features]
print(f"âœ… Final data prepared. Using {len(features)} features.")


# ===========================================================================
# Define Final Tuned Hyperparameters
# ===========================================================================
print("\nâš™ï¸� Defining optimal hyperparameters...")

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

best_lgbm_params = {
    'learning_rate': 0.015778728502699454,
    'num_leaves': 236,
    'max_depth': 12,
    'reg_alpha': 1.6775928011547346e-07,
    'reg_lambda': 0.008028908359775445,
    'colsample_bytree': 0.6422975418687189,
    'subsample': 0.763950615406502,
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 5000,
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

best_xgb_params = {
    'learning_rate': 0.004390851865519877,
    'max_depth': 8,
    'min_child_weight': 8,
    'subsample': 0.903190076410162,
    'colsample_bytree': 0.7942266833752402,
    'reg_alpha': 1.451640813538819e-07,
    'reg_lambda': 7.501128449163389e-08,
    'gamma': 0.0003828162890838331,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 5000,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'seed': 42
}

best_cat_params = {
    'learning_rate': 0.03379519781916954,
    'depth': 7,
    'l2_leaf_reg': 0.003099308419141463,
    'bagging_temperature': 0.23770425457967304,
    'random_strength': 3.199276236190115e-05,
    'objective': 'RMSE',
    'eval_metric': 'RMSE',
    'iterations': 5000,
    'task_type': 'GPU',
    'allow_writing_files': False,
    'random_seed': 42
}

# ===========================================================================
# Train all 3 tuned models to get OOF predictions
# ===========================================================================
print("\nğŸ¤– Training all 3 tuned models to get OOF and Test predictions...")

# Arrays for Stacking
oof_preds_lgbm = np.zeros(len(X_final))
test_preds_lgbm = np.zeros(len(X_test_final))
oof_preds_xgb = np.zeros(len(X_final))
test_preds_xgb = np.zeros(len(X_test_final))
oof_preds_cat = np.zeros(len(X_final))
test_preds_cat = np.zeros(len(X_test_final))

# --- Loop for all models ---
for fold, (train_idx, val_idx) in enumerate(kf.split(X_final, y_final)):
    print(f"\n--- Fold {fold+1} of {N_SPLITS} ---")
    X_train, y_train = X_final.iloc[train_idx], y_final.iloc[train_idx]
    X_val, y_val = X_final.iloc[val_idx], y_final.iloc[val_idx]
    
    # --- LGBM ---
    print("âœ”ï¸� Training LGBM...")
    lgbm = lgb.LGBMRegressor(**best_lgbm_params)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=200)])
    oof_preds_lgbm[val_idx] = lgbm.predict(X_val)
    test_preds_lgbm += lgbm.predict(X_test_final) / N_SPLITS
    
    # --- XGB ---
    print("âœ”ï¸� Training XGB...")
    xgb_model = xgb.XGBRegressor(**best_xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=200)
    oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test_final) / N_SPLITS
    
    # --- CAT ---
    print("âœ”ï¸� Training CAT...")
    cat = cb.CatBoostRegressor(**best_cat_params)
    cat.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=200)
    oof_preds_cat[val_idx] = cat.predict(X_val)
    test_preds_cat += cat.predict(X_test_final) / N_SPLITS

print("\nğŸ�‰ All base models have been retrained.")

# ===========================================================================
# Build the Stacking Model
# ===========================================================================
print("\n--- ğŸ“š Building Stacking Ensemble ---")

# 1. Create the meta-features (OOF predictions from our 3 models)
X_meta = np.column_stack((oof_preds_lgbm, oof_preds_xgb, oof_preds_cat))
y_meta = y_final

# Create the meta-features for the test set
X_test_meta = np.column_stack((test_preds_lgbm, test_preds_xgb, test_preds_cat))

# 2. Get a CV score for the stacking model
print("ğŸ§® Calculating CV score for the Stacking Meta-Model (Ridge)...")
meta_model = Ridge(alpha=1.0, random_state=42)
stacking_fold_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_meta, y_meta)):
    X_train_meta, y_train_meta = X_meta[train_idx], y_meta[train_idx]
    X_val_meta, y_val_meta = X_meta[val_idx], y_meta[val_idx]
    
    meta_model.fit(X_train_meta, y_train_meta)
    stacking_preds = meta_model.predict(X_val_meta)
    stacking_fold_scores.append(np.sqrt(mean_squared_error(y_val_meta, stacking_preds)))

# Calculate final CV scores
lgbm_cv_score = np.sqrt(mean_squared_error(y_final, oof_preds_lgbm))
xgb_cv_score = np.sqrt(mean_squared_error(y_final, oof_preds_xgb))
cat_cv_score = np.sqrt(mean_squared_error(y_final, oof_preds_cat))
stacking_cv_score = np.mean(stacking_fold_scores)

print("\n--- ğŸ’¡ FINAL MODEL SCORES (from OOF predictions) ---")
print(f"ğŸ�† Tuned LGBM CV RMSE:   {lgbm_cv_score:.5f}")
print(f"ğŸ�† Tuned XGB CV RMSE:    {xgb_cv_score:.5f}")
print(f"ğŸ�† Tuned CAT CV RMSE:    {cat_cv_score:.5f}")
print(f"ğŸ�† STACKING CV RMSE:     {stacking_cv_score:.5f}")


# ===========================================================================
# FINAL STEP: Create All Submission Files
# ===========================================================================
print("\nğŸ“� Creating final submission files...")

# 1. Train the final meta-model on ALL OOF data
meta_model.fit(X_meta, y_meta)
final_stacking_preds = meta_model.predict(X_test_meta)

# 2. Create the submission DataFrames
submission_stacking = pd.DataFrame({'id': test_df['id'], 'accident_risk': final_stacking_preds})
submission_lgbm = pd.DataFrame({'id': test_df['id'], 'accident_risk': test_preds_lgbm})
submission_xgb = pd.DataFrame({'id': test_df['id'], 'accident_risk': test_preds_xgb})
submission_cat = pd.DataFrame({'id': test_df['id'], 'accident_risk': test_preds_cat})

# 3. Save the submission files
submission_stacking.to_csv('submission_stacking.csv', index=False)
submission_lgbm.to_csv('submission_lgbm.csv', index=False)
submission_xgb.to_csv('submission_xgb.csv', index=False)
submission_cat.to_csv('submission_cat.csv', index=False)

print("âœ… 'submission_stacking.csv' created.")
print("âœ… 'submission_lgbm.csv' created.")
print("âœ… 'submission_xgb.csv' created.")
print("âœ… 'submission_cat.csv' created.")
print("\nğŸ�‰ All final submissions are ready!")


# ---------------------------------------------------------------------------
# FINAL MODEL EVALUATION & DIAGNOSTICS
# ---------------------------------------------------------------------------
print("ğŸ“ˆ Generating final analysis plots for all 3 tuned models...")

# --- 1. Re-fit models on 100% of data to get a single, clean FI list ---
# We already have the OOF preds, but we need to fit one last time
# on all data to get the definitive Feature Importances.
# ---------------------------------------------------------------------------
print("Fitting final models on 100% of data for feature importance...")

# Fit LGBM
lgbm_final_model = lgb.LGBMRegressor(**best_lgbm_params)
lgbm_final_model.fit(X_final, y_final, callbacks=[lgb.log_evaluation(period=0)])

# Fit XGB
xgb_final_model = xgb.XGBRegressor(**best_xgb_params)
xgb_final_model.fit(X_final, y_final, verbose=False)

# Fit CAT
cat_final_model = cb.CatBoostRegressor(**best_cat_params)
cat_final_model.fit(X_final, y_final, verbose=False)

print("âœ… Final models fitted.")

# ---------------------------------------------------------------------------
# 2. Feature Importance Plots
# ---------------------------------------------------------------------------
print("\nGenerating Feature Importance plots...")

# Get FI data
fi_lgbm = pd.DataFrame({
    'feature': features,
    'importance': lgbm_final_model.feature_importances_
}).sort_values('importance', ascending=False)

fi_xgb = pd.DataFrame({
    'feature': features,
    'importance': xgb_final_model.feature_importances_
}).sort_values('importance', ascending=False)

fi_cat = pd.DataFrame({
    'feature': features,
    'importance': cat_final_model.get_feature_importance()
}).sort_values('importance', ascending=False)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 12))
fig.suptitle('Final Model Feature Importances (Top 20)', fontsize=20, y=1.02)

# LGBM Plot
sns.barplot(ax=axes[0], x='importance', y='feature', data=fi_lgbm.head(20))
axes[0].set_title('LightGBM')
axes[0].set_xlabel('Importance')

# XGB Plot
sns.barplot(ax=axes[1], x='importance', y='feature', data=fi_xgb.head(20))
axes[1].set_title('XGBoost')
axes[1].set_xlabel('Importance')
axes[1].set_ylabel('') # Clean up

# CAT Plot
sns.barplot(ax=axes[2], x='importance', y='feature', data=fi_cat.head(20))
axes[2].set_title('CatBoost')
axes[2].set_xlabel('Importance')
axes[2].set_ylabel('') # Clean up

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 3. Prediction vs. Actual Plots (using OOF predictions)
# ---------------------------------------------------------------------------
print("\nGenerating Prediction vs. Actual plots...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)
fig.suptitle('Out-of-Fold Predictions vs. Actual Values', fontsize=20, y=1.02)

# LGBM
sns.scatterplot(ax=axes[0], x=y_final, y=oof_preds_lgbm, alpha=0.15)
axes[0].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
axes[0].set_title(f'LGBM (RMSE: {lgbm_cv_score:.5f})')
axes[0].set_xlabel('Actual Risk')
axes[0].set_ylabel('Predicted Risk')
axes[0].legend()

# XGB
sns.scatterplot(ax=axes[1], x=y_final, y=oof_preds_xgb, alpha=0.15)
axes[1].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
axes[1].set_title(f'XGBoost (RMSE: {xgb_cv_score:.5f})')
axes[1].set_xlabel('Actual Risk')
axes[1].set_ylabel('')

# CAT
sns.scatterplot(ax=axes[2], x=y_final, y=oof_preds_cat, alpha=0.15)
axes[2].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
axes[2].set_title(f'CatBoost (RMSE: {cat_cv_score:.5f})')
axes[2].set_xlabel('Actual Risk')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 4. Residuals Plots (using OOF predictions)
# ---------------------------------------------------------------------------
print("\nGenerating Residuals plots...")
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
fig.suptitle('Residuals vs. Predicted Values', fontsize=20, y=1.02)

# Calculate residuals
res_lgbm = y_final - oof_preds_lgbm
res_xgb = y_final - oof_preds_xgb
res_cat = y_final - oof_preds_cat

# LGBM
sns.scatterplot(ax=axes[0], x=oof_preds_lgbm, y=res_lgbm, alpha=0.15)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('LGBM')
axes[0].set_xlabel('Predicted Risk')
axes[0].set_ylabel('Residual (Actual - Predicted)')

# XGB
sns.scatterplot(ax=axes[1], x=oof_preds_xgb, y=res_xgb, alpha=0.15)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('XGBoost')
axes[1].set_xlabel('Predicted Risk')
axes[1].set_ylabel('')

# CAT
sns.scatterplot(ax=axes[2], x=oof_preds_cat, y=res_cat, alpha=0.15)
axes[2].axhline(0, color='red', linestyle='--')
axes[2].set_title('CatBoost')
axes[2].set_xlabel('Predicted Risk')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 5. Textual Printout of Feature Importances
# ---------------------------------------------------------------------------
print("\n--- ğŸ“Š Textual Feature Importances ---")
print("\n--- LightGBM ---")
print(fi_lgbm)
print("\n--- XGBoost ---")
print(fi_xgb)
print("\n--- CatBoost ---")
print(fi_cat)

print("\n\nâœ… All analysis plots and outputs are complete.")

