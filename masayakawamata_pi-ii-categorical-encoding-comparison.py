!pip install mapie -qq


# ===================================================================
#  CQR LGBM: Categorical Encoding Comparison with Smoothing
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

def plot_feature_importance(importances_df, model_name, strategy_name, output_path):
    """Plots and saves the average feature importance across folds."""
    mean_importances = importances_df.mean(axis=0).sort_values(ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(x=mean_importances.head(25).values, y=mean_importances.head(25).index)
    plt.title(f'Top 25 Feature Importances for {model_name} Model\nStrategy: {strategy_name} (Avg over {CFG.N_SPLITS} Folds)')
    plt.xlabel('Average Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'feature_importance_{strategy_name}_{model_name.lower().replace(" ", "_")}.png'))
    plt.show() # Display plot in notebook
    plt.close()
    print(f"Feature importance plot for {model_name} ({strategy_name}) saved and displayed.")

def base_feature_engineer(df):
    """Base feature engineering for date and categorical columns."""
    data = df.copy()
    if 'sale_date' in data.columns:
        data['sale_date'] = pd.to_datetime(data['sale_date'])
        data['sale_year'] = data['sale_date'].dt.year
        data['sale_month'] = data['sale_date'].dt.month
        data['sale_dayofweek'] = data['sale_date'].dt.dayofweek
        first_sale_month = data['sale_date'].dt.to_period('M').min()
        data['months_since_first_sale'] = (data['sale_date'].dt.to_period('M') - first_sale_month).apply(lambda x: x.n)
        data = data.drop('sale_date', axis=1)
    
    # Convert object columns to category for consistent handling
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = pd.Categorical(data[col])
    return data

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
        'category_feature': np.random.choice(['A', 'B', 'C'], 1000),
        'another_cat_feature': np.random.choice(['X', 'Y', 'Z', 'W'], 1000)
    })
    test_df_raw = pd.DataFrame({
        'id': range(1000, 1200),
        'sale_date': pd.to_datetime(pd.date_range(start='2024-09-01', periods=200)),
        'category_feature': np.random.choice(['A', 'B', 'C', 'D'], 200), # Add 'D' to test unseen categories
        'another_cat_feature': np.random.choice(['X', 'Y', 'Z'], 200)
    })


train_df_base = base_feature_engineer(train_df_raw)
test_df_base = base_feature_engineer(test_df_raw)

# --- 3. Run Experiment for each Categorical Handling Strategy ---
strategies = ['lgbm_native', 'cat_codes', 'target_encoding']

for strategy in strategies:
    print(f"\n\n{'='*60}")
    print(f"  RUNNING EXPERIMENT FOR STRATEGY: {strategy.upper()}")
    print(f"{'='*60}\n")

    # --- Data Preparation ---
    train_df = train_df_base.copy()
    test_df_processed = test_df_base.copy()
    
    categorical_features = [col for col in train_df.columns if train_df[col].dtype.name == 'category']
    
    if strategy == 'cat_codes':
        print("Applying 'cat.codes' to categorical features.")
        for col in categorical_features:
            train_df[col] = train_df[col].cat.codes
            test_df_processed[col] = test_df_processed[col].cat.codes

    features = [col for col in train_df.columns if col not in ['id', 'sale_price']]
    y = train_df['sale_price']
    X = train_df[features]
    X_test = test_df_processed[features]
    
    if strategy == 'lgbm_native':
        for col in categorical_features:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    print(f"Training with {len(features)} features. Train shape: {X.shape}, Test shape: {X_test.shape}")

    # --- Cross-Validation Training ---
    kf = KFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)
    
    oof_preds_lower = np.zeros(len(train_df))
    oof_preds_upper = np.zeros(len(train_df))
    test_preds_lower_sum = np.zeros(len(test_df_processed))
    test_preds_upper_sum = np.zeros(len(test_df_processed))
    fold_scores = []

    # Initialize dataframes to store feature importances for this strategy
    lower_importances = pd.DataFrame(index=range(CFG.N_SPLITS), columns=features)
    median_importances = pd.DataFrame(index=range(CFG.N_SPLITS), columns=features)
    upper_importances = pd.DataFrame(index=range(CFG.N_SPLITS), columns=features)

    params_lower = {**CFG.LGBM_PARAMS, 'alpha': CFG.ALPHA / 2}
    params_median = {**CFG.LGBM_PARAMS, 'alpha': 0.5}
    params_upper = {**CFG.LGBM_PARAMS, 'alpha': 1 - (CFG.ALPHA / 2)}

    for fold, (fit_idx, calib_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{CFG.N_SPLITS} ---")
        X_fit, X_calib = X.iloc[fit_idx].copy(), X.iloc[calib_idx].copy()
        y_fit, y_calib = y.iloc[fit_idx], y.iloc[calib_idx]
        X_test_fold = X_test.copy() # Use a copy for this fold to avoid re-encoding

        # --- Target Encoding with Nested CV and Smoothing ---
        if strategy == 'target_encoding':
            print("Applying Target Encoding with Nested CV to prevent leakage, including smoothing.")
            
            for col in categorical_features:
                X_fit[col] = X_fit[col].astype(object)
                X_calib[col] = X_calib[col].astype(object)
                X_test_fold[col] = X_test_fold[col].astype(object)
            
            smoothing_factor = 10 
            
            inner_kf = KFold(n_splits=5, shuffle=True, random_state=CFG.SEED + fold)
            for col in categorical_features:
                target_mean_global = y_fit.mean()
                
                agg_df = y_fit.groupby(X_fit[col]).agg(['mean', 'count'])
                agg_df.columns = ['mean', 'count']
                
                # smoothed_mean = (count * category_mean + smoothing_factor * global_mean) / (count + smoothing_factor)
                smoothed_target_map = (agg_df['mean'] * agg_df['count'] + target_mean_global * smoothing_factor) / (agg_df['count'] + smoothing_factor)
                
                X_calib[col] = X_calib[col].map(smoothed_target_map).fillna(target_mean_global)
                X_test_fold[col] = X_test_fold[col].map(smoothed_target_map).fillna(target_mean_global)

                oof_encoding = pd.Series(index=X_fit.index, dtype=float)
                for inner_train_idx, inner_val_idx in inner_kf.split(X_fit, y_fit):
                    X_fit_inner, y_fit_inner = X_fit.iloc[inner_train_idx], y_fit.iloc[inner_train_idx]
                    X_val_inner = X_fit.iloc[inner_val_idx]
                    
                    inner_agg_df = y_fit_inner.groupby(X_fit_inner[col]).agg(['mean', 'count'])
                    inner_agg_df.columns = ['mean', 'count']
                    inner_global_mean = y_fit_inner.mean()

                    inner_smoothed_target_map = (inner_agg_df['mean'] * inner_agg_df['count'] + inner_global_mean * smoothing_factor) / (inner_agg_df['count'] + smoothing_factor)
                    
                    oof_encoding.iloc[inner_val_idx] = X_val_inner[col].map(inner_smoothed_target_map)
                
                X_fit[col] = oof_encoding.fillna(target_mean_global)

        # --- Model Fitting ---
        print("Fitting lower, median, and upper models...")
        model_lower = lgb.LGBMRegressor(**params_lower)
        model_median = lgb.LGBMRegressor(**params_median)
        model_upper = lgb.LGBMRegressor(**params_upper)

        callbacks = [lgb.early_stopping(100, verbose=False)]
        # LightGBMネイティブ戦略の場合のみcategorical_feature='auto'を使用
        model_lower.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], callbacks=callbacks, categorical_feature='auto' if strategy == 'lgbm_native' else [])
        model_median.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], callbacks=callbacks, categorical_feature='auto' if strategy == 'lgbm_native' else [])
        model_upper.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], callbacks=callbacks, categorical_feature='auto' if strategy == 'lgbm_native' else [])
        
        lower_importances.loc[fold] = model_lower.feature_importances_
        median_importances.loc[fold] = model_median.feature_importances_
        upper_importances.loc[fold] = model_upper.feature_importances_

        # --- Conformalization and Prediction ---
        print("Conformalizing models...")
        mapie_cqr = ConformalizedQuantileRegressor(
            estimator=[model_lower, model_upper, model_median],
            confidence_level=CFG.CONFIDENCE_LEVEL,
            prefit=True
        ).conformalize(X_calib, y_calib)

        _, oof_pis = mapie_cqr.predict_interval(X_calib)
        oof_preds_lower[calib_idx] = oof_pis[:, 0, 0]
        oof_preds_upper[calib_idx] = oof_pis[:, 1, 0]

        fold_score = winkler_score_func(y_calib, oof_pis[:, 0, 0], oof_pis[:, 1, 0])
        fold_scores.append(fold_score)
        print(f"Fold {fold+1} Winkler Score: {fold_score:,.2f}")

        print("Predicting on test data...")
        _, test_pis = mapie_cqr.predict_interval(X_test_fold) # Use the fold-specific encoded test set
        test_preds_lower_sum += test_pis[:, 0, 0]
        test_preds_upper_sum += test_pis[:, 1, 0]

    # --- 4. Final Evaluation and Submission for the Strategy ---
    print(f"\n--- Final Evaluation for Strategy: {strategy.upper()} ---")
    overall_oof_score = winkler_score_func(y, oof_preds_lower, oof_preds_upper)
    print(f"Fold Scores: {[f'{s:,.2f}' for s in fold_scores]}")
    print(f"Overall OOF Winkler Score: {overall_oof_score:,.2f}")

    plot_feature_importance(lower_importances, "Lower Quantile", strategy, CFG.OUTPUT_PATH)
    plot_feature_importance(median_importances, "Median Quantile", strategy, CFG.OUTPUT_PATH)
    plot_feature_importance(upper_importances, "Upper Quantile", strategy, CFG.OUTPUT_PATH)

    test_preds_lower = test_preds_lower_sum / CFG.N_SPLITS
    test_preds_upper = test_preds_upper_sum / CFG.N_SPLITS

    submission_df = pd.DataFrame({
        'id': test_df_raw['id'],
        'pi_lower': test_preds_lower,
        'pi_upper': test_preds_upper
    })
    submission_df['pi_lower'] = np.minimum(submission_df['pi_lower'], submission_df['pi_upper'])
    
    submission_filename = f'submission_cqr_{strategy}.csv'
    submission_df.to_csv(os.path.join(CFG.OUTPUT_PATH, submission_filename), index=False)

    print(f"\nSubmission file '{submission_filename}' has been created.")
    print(submission_df.head())





