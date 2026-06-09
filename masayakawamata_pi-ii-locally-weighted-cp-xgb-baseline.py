import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
import warnings
import gc

# --- 1. Configuration & Utility Functions ---

warnings.filterwarnings('ignore')

class CFG:
    """
    Configuration class for parameters and constants.
    """
    N_SPLITS = 5
    SEED = 42
    ALPHA = 0.1  # Corresponds to a 100*(1-ALPHA)% = 90% prediction interval
    DEVICE = 'cuda'

print(f"Using device: {CFG.DEVICE}")
print("Locally-Weighted Conformal Prediction Model")

def winkler_score(y_true, lower, upper, alpha=CFG.ALPHA):
    """
    Calculates the Winkler Score for a given prediction interval.
    A common metric for evaluating the quality of prediction intervals.
    """
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    width = upper - lower
    
    # Penalties for observations outside the interval
    lower_penalty = np.where(y_true < lower, 2 / alpha * (lower - y_true), 0)
    upper_penalty = np.where(y_true > upper, 2 / alpha * (y_true - upper), 0)
    
    score = width + lower_penalty + upper_penalty
    return np.mean(score)

def feature_engineer(df):
    """
    Performs minimal feature engineering on the dataframe.
    """
    data = df.copy()
    # Basic date features
    data["sale_date"] = pd.to_datetime(data.sale_date)
    data["sale_year"] = data["sale_date"].dt.year
    data["sale_month"] = data["sale_date"].dt.month
    data["sale_dayofweek"] = data["sale_date"].dt.dayofweek
    data.drop("sale_date", axis=1, inplace=True)
    return data

# --- 2. Data Loading and Preprocessing ---

print("\nLoading and preparing data...")
# Adjust this path if you are not using Kaggle notebooks
BASE_PATH = "/kaggle/input/prediction-interval-competition-ii-house-price" 
train_df = pd.read_csv(f"{BASE_PATH}/dataset.csv").set_index("id")
test_df = pd.read_csv(f"{BASE_PATH}/test.csv").set_index("id")

train = feature_engineer(train_df)
test = feature_engineer(test_df)

y_orig = train["sale_price"]
# Use log-transform for the target variable to stabilize variance
y_log = np.log1p(y_orig)
print("Target variable log-transformed.")

# Encode categorical features
cat_cols = [c for c in test.columns if test[c].dtype == 'object']
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train[cat_cols] = encoder.fit_transform(train[cat_cols]).astype(int)
test[cat_cols] = encoder.transform(test[cat_cols]).astype(int)

features = list(test.columns)
print(f"Data ready: Train {train.shape}, Test {test.shape}")

# --- 3. Two-Stage Model Training (Mean and Variance) ---

# Pre-tuned model parameters
m0_best_params = {
    'n_estimators': 10000, 'learning_rate': 0.015, 'max_depth': 8,
    'subsample': 0.8, 'colsample_bytree': 0.75,
    'min_child_weight': 26, 'random_state': CFG.SEED, 'device': CFG.DEVICE}

m1_best_params = {
    'objective': "reg:gamma", 'n_estimators': 10000, 'learning_rate': 0.018,
    'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.88,
    'min_child_weight': 22, 'random_state': CFG.SEED, 'device': CFG.DEVICE}

print("\nGenerating OOF predictions for mean and variance...")

oof_mu_hat_log = []
oof_sigma_sq_hat = []
oof_indices = []
oof_fold_ids = [] # To store fold ID for each OOF prediction

test_mu_hat_log_sum = np.zeros(len(test))
test_sigma_sq_hat_sum = np.zeros(len(test))

cv = KFold(shuffle=True, random_state=CFG.SEED, n_splits=CFG.N_SPLITS)
for i, (train_idx, val_idx) in enumerate(cv.split(train, y_log)):
    print(f"--- Processing Fold {i+1}/{CFG.N_SPLITS} ---")
    X_tr, X_vl = train.iloc[train_idx], train.iloc[val_idx]
    y_tr_log, y_vl_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

    # Model 0: Predicts the mean (μ)
    model0 = XGBRegressor(**m0_best_params)
    model0.fit(X_tr[features], y_tr_log, 
               eval_set=[(X_vl[features], y_vl_log)], 
               early_stopping_rounds=100, verbose=False)
    
    # Predict mean for validation set
    mu_hat_vl_log = model0.predict(X_vl[features])
    
    # Model 1: Predicts the squared error (variance σ^2)
    # Target for model1 is the squared residual from model0
    residuals_tr = y_tr_log - model0.predict(X_tr[features])
    target_m1 = residuals_tr**2
    
    model1 = XGBRegressor(**m1_best_params)
    model1.fit(X_tr[features], target_m1)
    
    # Predict variance for validation set
    sigma_sq_hat_vl = model1.predict(X_vl[features])

    # Store OOF predictions and their fold IDs
    oof_mu_hat_log.extend(mu_hat_vl_log)
    oof_sigma_sq_hat.extend(np.maximum(sigma_sq_hat_vl, 1e-6)) # Ensure non-negative variance
    oof_indices.extend(val_idx)
    oof_fold_ids.extend([i] * len(val_idx))
    
    # Accumulate test predictions
    test_mu_hat_log_sum += model0.predict(test[features])
    test_sigma_sq_hat_sum += model1.predict(test[features])
    
    gc.collect()

# --- 4. Conformalization Step ---
print("\nApplying Locally-Weighted Conformal Prediction logic...")

# Create a DataFrame with OOF predictions
oof_df = pd.DataFrame({
    'y_true_log': y_log.loc[oof_indices].values,
    'mu_hat_log': oof_mu_hat_log, 
    'sigma_sq_hat': oof_sigma_sq_hat,
    'fold': oof_fold_ids,
}, index=oof_indices).sort_index()

# Calculate conformity scores (locally-weighted residuals)
# R_i = |y_i - μ_hat(x_i)| / σ_hat(x_i)
conformity_scores = np.abs(oof_df['y_true_log'] - oof_df['mu_hat_log']) / np.sqrt(oof_df['sigma_sq_hat'])

# Determine the quantile 'd' for the prediction interval
# This is the core of the conformal prediction method
n_cal = len(conformity_scores)
q_level = np.ceil((n_cal + 1) * (1 - CFG.ALPHA)) / n_cal
d = np.quantile(conformity_scores, q_level, method="higher")

print(f"Calculated Quantile (d) for {100*(1-CFG.ALPHA):.0f}% PI: {d:.4f}")

# --- 5. Final Evaluation and Submission Generation ---
print("\nCalculating Final CV Score and Coverage...")

# Construct prediction intervals for OOF data
oof_df['pi_lower_log'] = oof_df['mu_hat_log'] - d * np.sqrt(oof_df['sigma_sq_hat'])
oof_df['pi_upper_log'] = oof_df['mu_hat_log'] + d * np.sqrt(oof_df['sigma_sq_hat'])

# Transform intervals back to the original scale
oof_df['pi_lower'] = np.expm1(oof_df['pi_lower_log'])
oof_df['pi_upper'] = np.expm1(oof_df['pi_upper_log'])
oof_df['y_true_orig'] = y_orig.loc[oof_indices].sort_index()

# Calculate and print score for each fold
for fold_num in range(CFG.N_SPLITS):
    fold_df = oof_df[oof_df['fold'] == fold_num]
    fold_score = winkler_score(fold_df['y_true_orig'], fold_df['pi_lower'], fold_df['pi_upper'])
    print(f"Fold {fold_num+1} Winkler Score: {fold_score:.4f}")

# Calculate final overall scores
final_cv_score = winkler_score(oof_df['y_true_orig'], oof_df['pi_lower'], oof_df['pi_upper'])
empirical_coverage = np.mean((oof_df['y_true_orig'] >= oof_df['pi_lower']) & (oof_df['y_true_orig'] <= oof_df['pi_upper']))

print(f"\nOverall CV Winkler Score: {final_cv_score:.4f}")
print(f"Overall Coverage: {empirical_coverage:.4f} (Target: {1-CFG.ALPHA:.2f})")


print("\nGenerating final submission predictions...")
# Average test predictions across folds
final_test_mu_hat_log = test_mu_hat_log_sum / CFG.N_SPLITS
final_test_sigma_sq_hat = test_sigma_sq_hat_sum / CFG.N_SPLITS
final_test_sigma_hat = np.sqrt(np.maximum(final_test_sigma_sq_hat, 1e-6))

# Construct prediction intervals for test data using the same quantile 'd'
pi_lower_log = final_test_mu_hat_log - d * final_test_sigma_hat
pi_upper_log = final_test_mu_hat_log + d * final_test_sigma_hat

# Transform to original scale
pi_lower = np.expm1(pi_lower_log)
pi_upper = np.expm1(pi_upper_log)

# Create submission DataFrame
submission_df = pd.DataFrame({'id': test.index, 'pi_lower': pi_lower, 'pi_upper': pi_upper})

print("Final prediction intervals for test set generated successfully!")
print("First 5 submission predictions:")
print(submission_df.head())




