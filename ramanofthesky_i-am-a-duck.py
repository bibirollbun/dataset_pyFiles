import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ===== Feature Engineering =====
def feature_engineering(df):
    """Original features plus new robust features"""
    # Original features
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    # New robust features
    df['log_volume'] = np.log1p(df['volume'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-8)
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    return df

# ===== Configuration =====
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333",
        "bid_qty", "ask_qty"
    ]
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42

XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",  # Will fallback to CPU if GPU not available
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1
}

# ===== Data Loading & EDA =====
def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    """Create time decay weights for more recent data importance"""
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def load_data():
    """Load and preprocess data"""
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    # Apply feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    engineered_features = [
        "volume_weighted_sell", "buy_sell_ratio", "selling_pressure", 
        "effective_spread_proxy", "log_volume", "bid_ask_imbalance",
        "order_flow_imbalance", "liquidity_ratio"
    ]
    Config.FEATURES = list(set(Config.FEATURES + engineered_features))
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    print(f"Total features: {len(Config.FEATURES)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

def perform_eda(df, title="Train Data"):
    """Perform comprehensive exploratory data analysis"""
    print(f"\n== {title} EDA ==")
    # 1. Basic Info
    print("\n--- Basic Info ---")
    print(df.info())
    # 2. Missing Values
    print("\n--- Missing Values ---")
    print(df.isna().sum())
    sns.heatmap(df.isna(), cbar=False, cmap='viridis')
    plt.title(f"{title} - Missing Values")
    plt.show()
    # 3. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(df.describe(include='all'))
    # 4. Label Distribution
    if Config.LABEL_COLUMN in df:
        print("\n--- Label Distribution ---")
        sns.histplot(df[Config.LABEL_COLUMN], kde=True)
        plt.title(f"{title} - Label Distribution")
        plt.show()
    # 5. Numeric Feature Distributions (sample)
    print("\n--- Numeric Feature Distributions (Sample) ---")
    sample_features = ["buy_qty", "sell_qty", "volume", "bid_qty", "ask_qty"]
    for feat in sample_features:
        if feat in df:
            sns.histplot(df[feat], kde=True)
            plt.title(f"{title} - {feat} Distribution")
            plt.show()
    # 6. Correlation Matrix (for numeric features)
    print("\n--- Correlation Matrix (Sample Numeric Features) ---")
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title(f"{title} - Correlation Matrix")
    plt.show()
    # 7. Time Series Plot (if index is time)
    if not df.index.name and not isinstance(df.index, pd.MultiIndex):
        if len(df) > 1000:
            sample_size = 1000
            sample_idx = np.linspace(0, len(df)-1, sample_size, dtype=int)
            df_sample = df.iloc[sample_idx]
        else:
            df_sample = df.copy()
        for feat in sample_features:
            if feat in df_sample:
                plt.plot(df_sample[feat], label=feat)
        plt.legend()
        plt.title(f"{title} - Time Series (Sampled)")
        plt.show()

# ===== Model Training =====
def get_model_slices(n_samples: int):
    """Define different data slices for training"""
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_90pct", "cutoff": int(0.10 * n_samples)},
        {"name": "last_80pct", "cutoff": int(0.20 * n_samples)},
        {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        {"name": "last_70pct", "cutoff": int(0.30 * n_samples)},
        {"name": "last_60pct", "cutoff": int(0.40 * n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples)},
    ]

def train_xgboost_model(X_train, y_train, X_valid, y_valid, X_test, sample_weights=None):
    """Train XGBoost model"""
    try:
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_valid, y_valid)], verbose=False)
        valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)
        return valid_pred, test_pred, model
    except Exception as e:
        print(f"    Error training XGBoost: {str(e)}")
        return None, None, None

def train_and_evaluate(train_df, test_df):
    """Train XGBoost models with cross-validation on different data slices"""
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)
    oof_preds = {s["name"]: np.zeros(n_samples) for s in model_slices}
    test_preds = {s["name"]: np.zeros(len(test_df)) for s in model_slices}
    trained_models = {s["name"]: [] for s in model_slices}
    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]
        X_test = test_df[Config.FEATURES]
        
        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff
            if len(rel_idx) == 0:
                continue
            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]
            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")
            valid_pred, test_pred, model = train_xgboost_model(X_train, y_train, X_valid, y_valid, X_test, sw)
            if valid_pred is None:
                continue
            mask = valid_idx >= cutoff
            if mask.any():
                oof_preds[slice_name][valid_idx[mask] if valid_idx[mask].size > 0 else valid_idx] = valid_pred[mask] if valid_pred[mask].size > 0 else valid_pred
            if cutoff > 0 and (~mask).any():
                oof_preds[slice_name][valid_idx[~mask]] = oof_preds["full_data"][valid_idx[~mask]]
            test_preds[slice_name] += test_pred
            trained_models[slice_name].append(model)
    
    for slice_name in test_preds:
        if len(trained_models[slice_name]) > 0:
            test_preds[slice_name] /= len(trained_models[slice_name])
    return oof_preds, test_preds, model_slices, trained_models

# ===== Evaluation and Submission =====
def evaluate_and_create_submissions(train_df, oof_preds, test_preds, model_slices, submission_df):
    """Evaluate different strategies and create submissions"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS:")
    print("="*60)
    slice_scores = {}
    for s in model_slices:
        slice_name = s["name"]
        if slice_name in oof_preds:
            score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds[slice_name])[0]
            slice_scores[slice_name] = score
            print(f"{slice_name:15s}: {score:.4f}")
    best_slice = max(slice_scores.items(), key=lambda x: x[1])
    print(f"\nBest Strategy: {best_slice[0]} (Score: {best_slice[1]:.4f})")
    submissions_created = []
    for s in model_slices:
        slice_name = s["name"]
        if slice_name in test_preds:
            submission = submission_df.copy()
            submission["prediction"] = test_preds[slice_name]
            filename = f"submission_xgb_{slice_name}.csv"
            submission.to_csv(filename, index=False)
            submissions_created.append(filename)
            print(f"Created: {filename}")
    if len(slice_scores) > 1:
        total_score = sum(max(0, score) for score in slice_scores.values())
        if total_score > 0:
            weighted_oof = np.zeros(len(train_df))
            weighted_test = np.zeros(len(test_df))
            for slice_name, score in slice_scores.items():
                if score > 0:
                    weight = score / total_score
                    weighted_oof += weight * oof_preds[slice_name]
                    weighted_test += weight * test_preds[slice_name]
            ensemble_score = pearsonr(train_df[Config.LABEL_COLUMN], weighted_oof)[0]
            print(f"\nWeighted Ensemble Score: {ensemble_score:.4f}")
            print("Weights:")
            for slice_name, score in slice_scores.items():
                if score > 0:
                    weight = score / total_score
                    print(f"  {slice_name}: {weight:.3f}")
            submission_ensemble = submission_df.copy()
            submission_ensemble["prediction"] = weighted_test
            submission_ensemble.to_csv("submission_xgb_weighted_ensemble.csv", index=False)
            submissions_created.append("submission_xgb_weighted_ensemble.csv")
            print("Created: submission_xgb_weighted_ensemble.csv")
    submission_best = submission_df.copy()
    submission_best["prediction"] = test_preds[best_slice[0]]
    submission_best.to_csv("submission_xgb_best.csv", index=False)
    submissions_created.append("submission_xgb_best.csv")
    print("Created: submission_xgb_best.csv (recommended)")
    print("\n" + "="*60)
    print("SUBMISSION SUMMARY:")
    print("="*60)
    for slice_name, score in sorted(slice_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{slice_name:20s}: {score:.4f}")
    return submissions_created, slice_scores

# ===== Main Execution =====
if __name__ == "__main__":
    print("ğŸš€ XGBoost Crypto Prediction Pipeline")
    print("="*50)
    print("Loading data...")
    train_df, test_df, submission_df = load_data()
    print("\nPerforming Exploratory Data Analysis...")
    perform_eda(train_df, "Train Data")
    print("\nTraining XGBoost models...")
    oof_preds, test_preds, model_slices, trained_models = train_and_evaluate(train_df, test_df)
    print("\nEvaluating and creating submissions...")
    submissions_created, slice_scores = evaluate_and_create_submissions(
        train_df, oof_preds, test_preds, model_slices, submission_df
    )
    print(f"\nâœ… Pipeline completed successfully!")
    print(f"ğŸ“� Created {len(submissions_created)} submission files:")
    for filename in submissions_created:
        print(f"   - {filename}")
    print(f"\nğŸ�† Recommended submission: submission_xgb_best.csv")
    print(f"ğŸ“Š Best score: {max(slice_scores.values()):.4f}")


