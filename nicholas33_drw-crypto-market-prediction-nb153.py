import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import warnings
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')


# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH       = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH        = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH  = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Use the actual feature names from the dataset (X1, X2, etc. instead of X863, X856, etc.)
    FEATURES = [
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume",
        "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10",
        "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20",
        "X21", "X22", "X23", "X24", "X25", "X26", "X27"
    ]
    
    LABEL_COLUMN     = "label"
    N_FOLDS          = 5  # Increased from 3 for better cross-validation
    RANDOM_STATE     = 42

# Enhanced hyperparameters
XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.015,  # Slightly reduced for better generalization
    "max_depth": 18,  # Reduced to prevent overfitting
    "max_leaves": 10,  # Reduced
    "min_child_weight": 20,  # Increased for regularization
    "n_estimators": 2000,  # Increased
    "subsample": 0.08,  # Slightly increased
    "reg_alpha": 45.0,  # Increased regularization
    "reg_lambda": 85.0,  # Increased regularization
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1,
    "verbose": False,
}

LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "device": "cpu",
    "n_jobs": -1,
    "verbose": -1,
    "random_state": Config.RANDOM_STATE,
    "colsample_bytree": 0.55,  # Slightly increased
    "learning_rate": 0.008,  # Reduced for more iterations
    "min_child_samples": 25,  # Increased
    "min_child_weight": 0.15,  # Increased
    "n_estimators": 1500,  # Increased
    "num_leaves": 120,  # Reduced
    "reg_alpha": 25.0,  # Increased
    "reg_lambda": 65.0,  # Increased
    "subsample": 0.95,  # Slightly reduced
    "max_depth": 8,  # Reduced
    "feature_fraction": 0.8,  # Added for regularization
    "bagging_fraction": 0.9,  # Added
    "bagging_freq": 5  # Added
}

# Add CatBoost for diversity
CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 30,
    "random_strength": 0.5,
    "bagging_temperature": 0.2,
    "od_type": "Iter",
    "od_wait": 50,
    "random_seed": Config.RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False
}

# Enhanced learners with CatBoost
LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS, "need_scale": False},
    {"name": "lgbm", "Estimator": LGBMRegressor, "params": LGBM_PARAMS, "need_scale": False},
    {"name": "catboost", "Estimator": CatBoostRegressor, "params": CATBOOST_PARAMS, "need_scale": False}
]


# =========================
# Utility Functions
# =========================
def create_time_decay_weights(n: int, decay: float = 0.98) -> np.ndarray:
    """Enhanced time decay with stronger emphasis on recent data"""
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with more sophisticated features"""
    df = df.copy()

    # Original features
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    df['log_volume'] = np.log1p(df['volume'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-8)

    # NEW ENHANCED FEATURES
    # Market microstructure features
    df['total_order_qty'] = df['bid_qty'] + df['ask_qty'] + df['buy_qty'] + df['sell_qty']
    df['market_impact'] = df['volume'] / (df['total_order_qty'] + 1e-8)
    df['price_pressure_indicator'] = df['buy_qty'] / (df['ask_qty'] + 1e-8)
    df['liquidity_absorption'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    
    # Volume-based features
    df['volume_intensity'] = df['volume'] / (df['total_order_qty'] + 1e-8)
    df['aggressive_buy_ratio'] = df['buy_qty'] / (df['volume'] + 1e-8)
    df['passive_order_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['total_order_qty'] + 1e-8)
    
    # Cross-feature interactions
    df['volume_bid_interaction'] = df['volume'] * df['bid_qty']
    df['volume_ask_interaction'] = df['volume'] * df['ask_qty']
    df['buy_sell_volume_ratio'] = (df['buy_qty'] * df['volume']) / (df['sell_qty'] * df['volume'] + 1e-8)
    
    # Volatility proxies using X features (works with X1, X2, etc.)
    x_features = [col for col in df.columns if col.startswith('X')]
    if len(x_features) >= 5:
        # Use first 10 X features for statistical measures
        selected_x = x_features[:min(10, len(x_features))]
        df['x_feature_mean'] = df[selected_x].mean(axis=1)
        df['x_feature_std'] = df[selected_x].std(axis=1)
        df['x_feature_skew'] = df[selected_x].skew(axis=1)
        df['x_feature_range'] = df[selected_x].max(axis=1) - df[selected_x].min(axis=1)
        
        # Additional interactions with more X features if available
        if len(x_features) >= 10:
            df['x_feature_sum'] = df[selected_x].sum(axis=1)
    
    # Replace infinite values and extreme outliers
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col != Config.LABEL_COLUMN:
            if df[col].notna().sum() > 0:  # Only process if column has non-null values
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                if pd.notna(q99) and pd.notna(q01):
                    df[col] = df[col].clip(lower=q01, upper=q99)
    
    return df


def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")

    # Feature Engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # Handle missing values more carefully
    train_df = train_df.fillna(train_df.median()).reset_index(drop=True)
    test_df = test_df.fillna(train_df.median())

    # Update features list after engineering
    engineered_features = [col for col in train_df.columns if col != Config.LABEL_COLUMN]
    setattr(Config, "FEATURES", engineered_features)

    print(f"Processed data - Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Total features after engineering: {len(engineered_features)}")

    return train_df, test_df, submission_df

def get_model_slices(n_samples: int):
    """Enhanced model slices with more granular time-based splits"""
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_80pct", "cutoff": int(0.20 * n_samples)},
        {"name": "last_60pct", "cutoff": int(0.40 * n_samples)},
        {"name": "last_40pct", "cutoff": int(0.60 * n_samples)},
        {"name": "last_20pct", "cutoff": int(0.80 * n_samples)}
    ]



# =========================
# Training and Evaluation
# =========================
def train_single_model(X_train, y_train, X_valid, y_valid, X_test, learner, sample_weights=None):
    """Enhanced model training with early stopping and validation"""
    if learner["need_scale"]:
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_valid_scaled = X_valid
        X_test_scaled = X_test
    
    model = learner["Estimator"](**learner["params"])
    
    # Enhanced training with early stopping
    if learner["name"] == "xgb":
        model.fit(
            X_train_scaled, y_train, 
            sample_weight=sample_weights,
            eval_set=[(X_valid_scaled, y_valid)], 
            early_stopping_rounds=50,
            verbose=False
        )
    elif learner["name"] == "lgbm":
        model.fit(
            X_train_scaled, y_train, 
            sample_weight=sample_weights,
            eval_set=[(X_valid_scaled, y_valid)],
            callbacks=[],
            eval_metric='rmse'
        )
    elif learner["name"] == "catboost":
        model.fit(
            X_train_scaled, y_train,
            sample_weight=sample_weights,
            eval_set=(X_valid_scaled, y_valid),
            verbose=False
        )
    else:
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    valid_pred = model.predict(X_valid_scaled)
    test_pred = model.predict(X_test_scaled)
    
    return valid_pred, test_pred

def train_and_evaluate(train_df, test_df):
    """Enhanced training with better OOF handling"""
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)
    
    # Initialize prediction dictionaries
    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        for learner in LEARNERS
    }
    
    # Use stronger time decay
    full_weights = create_time_decay_weights(n_samples, decay=0.98)
    
    # Use TimeSeriesSplit-like approach for financial data
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
            
            # Enhanced sample weights
            if cutoff > 0:
                sw = create_time_decay_weights(len(subset), decay=0.98)[rel_idx]
            else:
                sw = full_weights[train_idx]

            # --- ADD THIS CHECK ---
            MIN_SAMPLES_FOR_TRAINING = 20 # A sensible minimum
            if len(X_train) < MIN_SAMPLES_FOR_TRAINING:
                print(f"  Skipping slice: {slice_name}, not enough samples ({len(X_train)})")
                continue # Skips this slice and moves to the next one
            # --- END OF CHECK ---
            
            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")
            
            for learner in LEARNERS:
                try:
                    valid_pred, test_pred = train_single_model(
                        X_train, y_train, X_valid, y_valid, X_test, learner, sw
                    )
                    
                    # Better OOF prediction handling
                    valid_mask = valid_idx >= cutoff
                    if valid_mask.any():
                        oof_preds[learner["name"]][slice_name][valid_idx[valid_mask]] = valid_pred[valid_mask]
                    
                    # For samples before cutoff, use full_data predictions
                    if cutoff > 0 and (~valid_mask).any():
                        oof_preds[learner["name"]][slice_name][valid_idx[~valid_mask]] = \
                            oof_preds[learner["name"]]["full_data"][valid_idx[~valid_mask]]
                    
                    test_preds[learner["name"]][slice_name] += test_pred / Config.N_FOLDS
                    
                except Exception as e:
                    print(f"    Error training {learner['name']}: {str(e)}")
                    continue
    
    return oof_preds, test_preds, model_slices



# =========================
# Enhanced Ensemble & Submission
# =========================
def ensemble_and_submit(train_df, oof_preds, test_preds, submission_df):
    """Enhanced ensemble with better weighting strategy"""
    learner_ensembles = {}
    learner_weights = {}
    
    for learner_name in oof_preds:
        # Calculate performance scores for each slice
        scores = {}
        for s in oof_preds[learner_name]:
            mask = oof_preds[learner_name][s] != 0  # Only consider non-zero predictions
            if mask.sum() > 0:
                corr = pearsonr(
                    train_df[Config.LABEL_COLUMN][mask], 
                    oof_preds[learner_name][s][mask]
                )[0]
                scores[s] = max(0, corr)  # Ensure non-negative weights
            else:
                scores[s] = 0
        
        total_score = sum(scores.values())
        if total_score == 0:
            # Fallback to equal weights
            weights = {s: 1.0/len(scores) for s in scores}
        else:
            weights = {s: scores[s] / total_score for s in scores}
        
        # Create ensembles
        oof_simple = np.mean([oof_preds[learner_name][s] for s in oof_preds[learner_name]], axis=0)
        test_simple = np.mean([test_preds[learner_name][s] for s in test_preds[learner_name]], axis=0)
        
        oof_weighted = sum(weights[s] * oof_preds[learner_name][s] for s in weights)
        test_weighted = sum(weights[s] * test_preds[learner_name][s] for s in weights)
        
        # Calculate final scores
        mask_simple = oof_simple != 0
        mask_weighted = oof_weighted != 0
        
        score_simple = pearsonr(train_df[Config.LABEL_COLUMN][mask_simple], oof_simple[mask_simple])[0] if mask_simple.sum() > 0 else 0
        score_weighted = pearsonr(train_df[Config.LABEL_COLUMN][mask_weighted], oof_weighted[mask_weighted])[0] if mask_weighted.sum() > 0 else 0
        
        print(f"\n{learner_name.upper()} Simple Ensemble Pearson:   {score_simple:.4f}")
        print(f"{learner_name.upper()} Weighted Ensemble Pearson: {score_weighted:.4f}")
        
        # Choose better performing ensemble
        if score_weighted > score_simple:
            learner_ensembles[learner_name] = {"oof": oof_weighted, "test": test_weighted}
            learner_weights[learner_name] = score_weighted
        else:
            learner_ensembles[learner_name] = {"oof": oof_simple, "test": test_simple}
            learner_weights[learner_name] = score_simple
    
    # Final ensemble with learner-level weighting
    total_weight = sum(learner_weights.values())
    if total_weight == 0:
        # Equal weights fallback
        final_oof = np.mean([le["oof"] for le in learner_ensembles.values()], axis=0)
        final_test = np.mean([le["test"] for le in learner_ensembles.values()], axis=0)
    else:
        normalized_weights = {k: v/total_weight for k, v in learner_weights.items()}
        final_oof = sum(normalized_weights[name] * le["oof"] for name, le in learner_ensembles.items())
        final_test = sum(normalized_weights[name] * le["test"] for name, le in learner_ensembles.items())
    
    final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]
    
    print(f"\nFINAL ensemble across learners Pearson: {final_score:.4f}")
    print(f"Learner weights: {learner_weights}")

    submission_df["prediction"] = final_test
    # submission_df.to_csv("submission.csv", index=False)
    # print("Saved: submission.csv")
    submission_df.to_csv("pipeline_ensemble_submission.csv", index=False)
    print("Saved: pipeline_ensemble_submission.csv")



# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    train_df, test_df, submission_df = load_data()
    oof_preds, test_preds, model_slices = train_and_evaluate(train_df, test_df)
    ensemble_and_submit(train_df, oof_preds, test_preds, submission_df)


import pandas as pd
from IPython.display import display

# =======================================================
# FINAL BLENDING FUNCTION (Based on your high-scoring code)
# =======================================================
def iBlend_final(sls):
    # This is your tida function with one minor change to read from multiple paths
    def tida(sls):
        def read_subm(sls, i):
            subm_dict = sls["subm"][i]
            tnm = subm_dict["name"]
            # The ONLY change is here: building the path from the 'subm' dictionary
            # This allows us to use '/kaggle/working/', '/kaggle/input/13-juli...', etc.
            FiN = f"{subm_dict['path']}/{tnm}.csv"
            # The 'ID' column in your external files is uppercase. We'll standardize to lowercase 'id'.
            df = pd.read_csv(FiN).rename(columns={'ID': 'id'})
            return df.rename(columns={'prediction': tnm, sls["target"]: tnm})

        dfs_subm = [read_subm(sls, i) for i in range(len(sls["subm"]))]
        df_subms = dfs_subm[0]
        for i in range(1, len(dfs_subm)):
            df_subms = pd.merge(df_subms, dfs_subm[i], on='id')

        cols = [subm['name'] for subm in sls["subm"]]
        corrects = sls["subwts"]
        weights = [subm['weight'] for subm in sls["subm"]]
        corrects2 = sls["subwts2"]
        weights2 = [subm['weight'] for subm in sls["subm2"]]

        def alls(x, cs=cols):
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes, key=lambda k: k[1], reverse=True if sls["sort"] == 'desc' else False)]
            return subms_sorted

        def correct(x, cs=cols, w=weights, cw=corrects, w2=weights2, cw2=corrects2):
            ic = [x['alls'].index(c) for c in cs]
            # Your proven conditional logic is preserved
            if x['abs(mx-m)'] > 0.74:
                cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
            else:
                cS = [x[cols[j]] * (w2[j] + cw2[ic[j]]) for j in range(len(cols))]
            return sum(cS)

        def amxm(x, cs=cols):
            return abs(max(x[cs].tolist()) - min(x[cs].tolist()))

        df_subms['abs(mx-m)'] = df_subms.apply(amxm, axis=1)
        df_subms['alls'] = df_subms.apply(alls, axis=1)
        df_subms[sls["target"]] = df_subms.apply(correct, axis=1)
        return df_subms

    # Your proven ensemble_tida function is preserved
    def ensemble_tida(sls):
        sample_subm = pd.read_csv(f"{sls['subm'][0]['path']}/{sls['subm'][0]['name']}.csv").rename(columns={'ID': 'id'})
        sls['sort'] = 'desc'
        dfD = tida(sls)[['id', sls['target']]]
        sls['sort'] = 'asc'
        dfA = tida(sls)[['id', sls['target']]]
        target, d, a = sls['target'], sls['desc'], sls['asc']
        submission = sample_subm[['id']].copy()
        submission[target] = dfD[target] * d + a * dfA[target]
        return submission

    return ensemble_tida(sls)


# =======================================================
# CONFIGURATION (Based on your high-scoring template)
# =======================================================
# 1. Define file paths and names
path_our_model = '/kaggle/working'
path_ext1 = '/kaggle/input/13-juli-2025-drw'
path_ext2 = '/kaggle/input/15-juli-2025-drw'

our_model_name = 'pipeline_ensemble_submission'

# All 11 external model names
external_names = [
    'submission 0.70871', 'submission 0.72837', 'submission 0.73799', 'submission 0.81760', 'submission 0.82968',
    'submission 0.83975', 'submission 0.86767', 'submission 0.88377', 'submission 0.89178', 'submission 0.90038', 'submission 0.95002'
]

# 2. Set up the parameters dictionary
# ⚠️ IMPORTANT: These weights are an educated guess based on your template. Fine-tuning these is the key to a top score.
params = {
    'target': 'prediction',
    'q_rows': 538150,
    'desc'  : 0.30,
    'asc'   : 0.70,

    # Following your pattern: bonus for top models, penalty for bottom. Length must be 12.
    'subwts':  [+0.20, +0.15, +0.10, +0.05, 0.0, -0.05, -0.10, -0.10, -0.15, -0.15, -0.20, -0.20],
    'subwts2': [+0.18, +0.13, +0.09, +0.04, 0.0, -0.04, -0.09, -0.09, -0.13, -0.13, -0.18, -0.18],

    # Base weights for all 12 models.
    'subm': [
        {'path': path_our_model, 'name': our_model_name,      'weight': 0.25}, # Our new model
        {'path': path_ext2,    'name': external_names[10],    'weight': 0.20}, # 0.95002
        {'path': path_ext2,    'name': external_names[9],     'weight': 0.10}, # 0.90038
        {'path': path_ext2,    'name': external_names[8],     'weight': 0.10}, # 0.89178
        {'path': path_ext2,    'name': external_names[7],     'weight': 0.08}, # 0.88377
        {'path': path_ext2,    'name': external_names[6],     'weight': 0.05}, # 0.86767
        {'path': path_ext2,    'name': external_names[5],     'weight': 0.05}, # 0.83975
        {'path': path_ext1,    'name': external_names[4],     'weight': 0.05}, # 0.82968
        {'path': path_ext1,    'name': external_names[3],     'weight': 0.04}, # 0.81760
        {'path': path_ext1,    'name': external_names[2],     'weight': 0.03}, # 0.73799
        {'path': path_ext1,    'name': external_names[1],     'weight': 0.03}, # 0.72837
        {'path': path_ext1,    'name': external_names[0],     'weight': 0.02}, # 0.70871
    ],
}
# The 'subm2' list is a slight variation for the second condition.
params['subm2'] = [
        {'path': path_our_model, 'name': our_model_name,      'weight': 0.24}, # Our new model
        {'path': path_ext2,    'name': external_names[10],    'weight': 0.21}, # 0.95002
        {'path': path_ext2,    'name': external_names[9],     'weight': 0.11}, # 0.90038
        {'path': path_ext2,    'name': external_names[8],     'weight': 0.11}, # 0.89178
        {'path': path_ext2,    'name': external_names[7],     'weight': 0.08}, # 0.88377
        {'path': path_ext2,    'name': external_names[6],     'weight': 0.05}, # 0.86767
        {'path': path_ext2,    'name': external_names[5],     'weight': 0.05}, # 0.83975
        {'path': path_ext1,    'name': external_names[4],     'weight': 0.04}, # 0.82968
        {'path': path_ext1,    'name': external_names[3],     'weight': 0.03}, # 0.81760
        {'path': path_ext1,    'name': external_names[2],     'weight': 0.02}, # 0.73799
        {'path': path_ext1,    'name': external_names[1],     'weight': 0.02}, # 0.72837
        {'path': path_ext1,    'name': external_names[0],     'weight': 0.02}, # 0.70871
]


# =======================================================
# EXECUTION
# =======================================================
print("Starting final high-performance blend...")
final_submission_df = iBlend_final(params)

# The 'id' column might be named 'ID' in the final output, let's standardize it
if 'ID' in final_submission_df.columns:
    final_submission_df = final_submission_df.rename(columns={'ID': 'id'})
    
final_submission_df.to_csv('submission.csv', index=False)
print("Final submission.csv created successfully using your high-performance logic!")
display(final_submission_df.head())




