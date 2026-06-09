!pip install mapie -qq


# ===================================================================
#  CQR Baseline with CV+: LightGBM with mapie (prefit=True)
# ===================================================================

# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from mapie.regression import ConformalizedQuantileRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
print("Starting: CQR_LGBM Baseline with prefit=True")

# --- 1. Configuration and Utility Functions ---
class CFG:
    # Global settings
    SEED = 42
    N_SPLITS = 5  # Number of folds for cross-validation
    CONFIDENCE_LEVEL = 0.9  # Target confidence level (90%)
    ALPHA = 1 - CONFIDENCE_LEVEL

    # File paths
    try:
        # Kaggle environment
        DATA_PATH = '/kaggle/input/prediction-interval-competition-ii-house-price/'
        if not os.path.exists(os.path.join(DATA_PATH, 'dataset.csv')):
            raise FileNotFoundError
    except FileNotFoundError:
        print("Kaggle path not found or files missing, switching to local path './'.")
        DATA_PATH = './'
    OUTPUT_PATH = './'

    # Base parameters for the quantile models
    LGBM_PARAMS = {
        'objective': 'quantile',
        'metric': 'quantile',
        'n_estimators': 2000, # Increased for early stopping
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'learning_rate': 0.05,
        'max_depth': -1,
        'min_child_samples': 150,
        'n_jobs': -1,
        'random_state': SEED,
        'verbose': -1,
    }

def winkler_score_func(y_true, lower, upper, alpha=CFG.ALPHA):
    """Utility function to calculate the Winkler score."""
    score = np.mean(upper - lower)
    score += np.mean(np.where(y_true < lower, (2 / alpha) * (lower - y_true), 0))
    score += np.mean(np.where(y_true > upper, (2 / alpha) * (y_true - upper), 0))
    return score

def plot_feature_importance(importances_df, model_name, output_path):
    """Plots and saves the average feature importance across folds."""
    mean_importances = importances_df.mean(axis=0).sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=mean_importances.head(25).values, y=mean_importances.head(25).index)
    plt.title(f'Top 25 Feature Importances for {model_name} Model (Avg over {CFG.N_SPLITS} Folds)')
    plt.xlabel('Average Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    print(f"Feature importance plot for {model_name} saved.")

# --- 2. Data Loading and Preprocessing ---
print("\n--- Phase 1: Loading and Preprocessing Data ---")
try:
    train_df_raw = pd.read_csv(os.path.join(CFG.DATA_PATH, 'dataset.csv'))
    test_df_raw = pd.read_csv(os.path.join(CFG.DATA_PATH, 'test.csv'))
except FileNotFoundError:
    print("Error: dataset.csv or test.csv not found.")
    print("Creating dummy data for demonstration purposes.")
    train_df_raw = pd.DataFrame({
        'id': range(1000), 'sale_price': np.random.rand(1000) * 500000 + 100000,
        'sale_date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=1000)),
        'category_feature': np.random.choice(['A', 'B', 'C'], 1000)
    })
    test_df_raw = pd.DataFrame({
        'id': range(1000, 1200),
        'sale_date': pd.to_datetime(pd.date_range(start='2024-09-01', periods=200)),
        'category_feature': np.random.choice(['A', 'B', 'C'], 200)
    })

def feature_engineer(df):
    """Simple feature engineering."""
    data = df.copy()
    if 'sale_date' in data.columns:
        data['sale_date'] = pd.to_datetime(data['sale_date'])
        data['sale_year'] = data['sale_date'].dt.year
        data['sale_month'] = data['sale_date'].dt.month
        data['sale_dayofweek'] = data['sale_date'].dt.dayofweek
        first_sale_month = data['sale_date'].dt.to_period('M').min()
        data['months_since_first_sale'] = (data['sale_date'].dt.to_period('M') - first_sale_month).apply(lambda x: x.n)
        data = data.drop('sale_date', axis=1)
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = pd.Categorical(data[col])
    return data

train_df = feature_engineer(train_df_raw)
test_df_processed = feature_engineer(test_df_raw)

features = [col for col in train_df.columns if col not in ['id', 'sale_price']]
y = train_df['sale_price']
X = train_df[features]
X_test = test_df_processed[features]

print(f"Training with {len(features)} features. Train shape: {X.shape}, Test shape: {X_test.shape}")

# --- 3. Cross-Validation Training with prefit=True ---
print(f"\n--- Phase 2: Training with {CFG.N_SPLITS}-Fold CV and prefit CQR ---")
kf = KFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)

oof_preds_lower = np.zeros(len(train_df))
oof_preds_upper = np.zeros(len(train_df))
test_preds_lower_sum = np.zeros(len(test_df_raw))
test_preds_upper_sum = np.zeros(len(test_df_raw))
fold_scores = []

# Initialize dataframes to store feature importances
lower_importances = pd.DataFrame(index=range(CFG.N_SPLITS), columns=features)
median_importances = pd.DataFrame(index=range(CFG.N_SPLITS), columns=features)
upper_importances = pd.DataFrame(index=range(CFG.N_SPLITS), columns=features)


# Define parameters for the three quantile models
params_lower = {**CFG.LGBM_PARAMS, 'alpha': CFG.ALPHA / 2}
params_median = {**CFG.LGBM_PARAMS, 'alpha': 0.5}
params_upper = {**CFG.LGBM_PARAMS, 'alpha': 1 - (CFG.ALPHA / 2)}

for fold, (fit_idx, calib_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{CFG.N_SPLITS} ---")
    X_fit, X_calib = X.iloc[fit_idx], X.iloc[calib_idx]
    y_fit, y_calib = y.iloc[fit_idx], y.iloc[calib_idx]

    # Step 1: Fit the three quantile models on the fitting dataset
    print("Fitting lower, median, and upper models...")
    model_lower = lgb.LGBMRegressor(**params_lower)
    model_median = lgb.LGBMRegressor(**params_median)
    model_upper = lgb.LGBMRegressor(**params_upper)

    callbacks = [lgb.early_stopping(100, verbose=False)]
    model_lower.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], callbacks=callbacks)
    model_median.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], callbacks=callbacks)
    model_upper.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], callbacks=callbacks)
    
    # Store feature importances for this fold
    lower_importances.loc[fold] = model_lower.feature_importances_
    median_importances.loc[fold] = model_median.feature_importances_
    upper_importances.loc[fold] = model_upper.feature_importances_

    # Step 2: Conformalize using the pre-fitted models and the calibration dataset
    print("Conformalizing models...")
    mapie_cqr = ConformalizedQuantileRegressor(
        estimator=[model_lower, model_upper, model_median], # [lower, upper, median] order
        confidence_level=CFG.CONFIDENCE_LEVEL,
        prefit=True
    ).conformalize(X_calib, y_calib)

    # Step 3: Generate OOF predictions for the calibration set
    _, oof_pis = mapie_cqr.predict_interval(X_calib)
    oof_preds_lower[calib_idx] = oof_pis[:, 0, 0]
    oof_preds_upper[calib_idx] = oof_pis[:, 1, 0]

    fold_score = winkler_score_func(y_calib, oof_pis[:, 0, 0], oof_pis[:, 1, 0])
    fold_scores.append(fold_score)
    print(f"Fold {fold+1} Winkler Score: {fold_score:,.2f}")

    # Step 4: Generate predictions for the test set and accumulate them
    print("Predicting on test data...")
    _, test_pis = mapie_cqr.predict_interval(X_test)
    test_preds_lower_sum += test_pis[:, 0, 0]
    test_preds_upper_sum += test_pis[:, 1, 0]

# --- 4. Final Evaluation and Submission ---
print("\n--- Phase 3: Final Evaluation and Submission ---")

overall_oof_score = winkler_score_func(y, oof_preds_lower, oof_preds_upper)
print(f"\nFold Scores: {[f'{s:,.2f}' for s in fold_scores]}")
print(f"Overall OOF Winkler Score: {overall_oof_score:,.2f}")

# Plot and save feature importances
plot_feature_importance(lower_importances, "Lower Quantile", CFG.OUTPUT_PATH)
plot_feature_importance(median_importances, "Median Quantile", CFG.OUTPUT_PATH)
plot_feature_importance(upper_importances, "Upper Quantile", CFG.OUTPUT_PATH)

test_preds_lower = test_preds_lower_sum / CFG.N_SPLITS
test_preds_upper = test_preds_upper_sum / CFG.N_SPLITS

submission_df = pd.DataFrame({
    'id': test_df_raw['id'],
    'pi_lower': test_preds_lower,
    'pi_upper': test_preds_upper
})
submission_df['pi_lower'] = np.minimum(submission_df['pi_lower'], submission_df['pi_upper'])
submission_df.to_csv(os.path.join(CFG.OUTPUT_PATH, 'submission_baseline_cqr_prefit.csv'), index=False)

print("\nSubmission file 'submission_baseline_cqr_prefit.csv' has been created.")
print(submission_df.head())




