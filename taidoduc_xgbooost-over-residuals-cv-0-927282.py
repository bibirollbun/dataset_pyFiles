import random
import warnings
from itertools import combinations
from pathlib import Path
import torch, gc
import numpy as np
import cudf
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.simplefilter("ignore")
random.seed(42)
np.random.seed(42)

# Configuration
DATA_PATHS = {
    "train": "/kaggle/input/playground-series-s5e11/train.csv",
    "test": "/kaggle/input/playground-series-s5e11/test.csv",
    "orig": "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv",
}

TARGET = "loan_paid_back"
EXCLUDED_BASE_COLS = {"id", TARGET}
CATEGORICAL_COLS = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]
INTERACTION_EXCLUSIONS = {"annual_income", "loan_amount"}
N_SPLITS = 10
NUM_BOOST_ROUND = 20000
EARLY_STOPPING_ROUNDS = 500
VERBOSE_EVAL = 1000

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "device": "cuda",
    "random_state": 42,
    "nthread": -1,
    'max_depth': 6,
    'min_child_weight': 20,
    'subsample': 0.66,
    'colsample_bytree': 0.22,
    'learning_rate': 0.006,
}

LGBM_PARAMS={
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'device': 'gpu',  
    'num_leaves': 187, 
    'max_depth': 5, 
    'min_child_samples': 19, 
    'min_child_weight': 8, 
    'learning_rate': 0.005, 
    'feature_fraction': 0.4, 
    'bagging_fraction': 0.6, 
    'bagging_freq': 1, 
    'min_split_gain': 0.5, 
    'max_bin': 166,
    'verbosity': -1  # Disable all LightGBM logs
}



class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder with optional smoothing and multiple aggregations - cuDF compatible."""

    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', fallback_smooth=10.0, drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.fallback_smooth = fallback_smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def _compute_smooth_factor(self, temp_df, col):
        """Compute empirical Bayes smooth factor m = variance_within / variance_between."""
        stats = temp_df.groupby(col)['target'].agg(['mean', 'var'])
        
        variance_between = float(stats['mean'].var())
        avg_variance_within = float(stats['var'].mean())
        
        if pd.isna(avg_variance_within):
            avg_variance_within = 0.0
        if pd.isna(variance_between) or variance_between <= 0:
            return self.fallback_smooth
        
        m_value = avg_variance_within / variance_between if variance_between > 0 else self.fallback_smooth
        
        if np.isnan(m_value) or m_value < 0:
            return self.fallback_smooth
        
        return m_value

    def _apply_smoothing(self, temp_df, col, agg_func, fold_global):
        """Apply smoothing to mean aggregation using empirical Bayes or fixed value."""
        if agg_func != 'mean':
            mapping = temp_df.groupby(col)['target'].agg(agg_func)
            return mapping, fold_global
        
        stats = temp_df.groupby(col)['target'].agg(['mean', 'count', 'var'])
        
        if self.smooth == 'auto':
            m = self._compute_smooth_factor(temp_df, col)
        else:
            m = self.smooth
        
        if m == 0:
            smoothed_mapping = stats['mean']
        else:
            smoothed_mapping = (stats['count'] * stats['mean'] + m * fold_global) / (stats['count'] + m)
        
        return smoothed_mapping, fold_global

    def fit(self, X, y):
        """Learn mappings from the entire dataset."""
        temp_df = X.copy()
        temp_df['target'] = y

        for agg_func in self.aggs:
            if agg_func == 'mean':
                self.global_stats_[agg_func] = float(y.mean())
            elif agg_func == 'std':
                self.global_stats_[agg_func] = float(y.std() if len(y) > 1 else 0.0)
            elif agg_func == 'count':
                temp_counts = temp_df.groupby(self.cols_to_encode[0]).size() if len(self.cols_to_encode) > 0 else pd.Series([1])
                self.global_stats_[agg_func] = float(temp_counts.mean())
            elif agg_func == 'nunique':
                self.global_stats_[agg_func] = 1.0
            elif agg_func == 'median':
                self.global_stats_[agg_func] = float(y.median())
            elif agg_func == 'min':
                self.global_stats_[agg_func] = float(y.min())
            elif agg_func == 'max':
                self.global_stats_[agg_func] = float(y.max())
            else:
                self.global_stats_[agg_func] = float(y.agg(agg_func))

        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                if agg_func == 'mean':
                    smoothed_mapping, _ = self._apply_smoothing(
                        temp_df, col, agg_func, self.global_stats_[agg_func]
                    )
                    self.mappings_[col][agg_func] = smoothed_mapping
                elif agg_func == 'std':
                    agg_result = temp_df.groupby(col)['target'].agg('std').fillna(0.0)
                    self.mappings_[col][agg_func] = agg_result
                elif agg_func == 'count':
                    agg_result = temp_df.groupby(col).size().astype(float)
                    self.mappings_[col][agg_func] = agg_result
                elif agg_func == 'nunique':
                    agg_result = temp_df.groupby(col)['target'].nunique().astype(float)
                    self.mappings_[col][agg_func] = agg_result
                elif agg_func in ['median', 'min', 'max']:
                    agg_result = temp_df.groupby(col)['target'].agg(agg_func)
                    self.mappings_[col][agg_func] = agg_result
                else:
                    agg_result = temp_df.groupby(col)['target'].agg(agg_func)
                    self.mappings_[col][agg_func] = agg_result
        return self

    def transform(self, X):
        """Transform with learned mappings."""
        X_transformed = X.copy()
        cols_to_encode = [col for col in self.cols_to_encode if col in X.columns]

        for col in cols_to_encode:
            for agg_func in self.aggs:
                new_col = f'TE_{col}_{agg_func}'
                X_transformed[new_col] = (X[col].map(self.mappings_[col][agg_func])
                                         .fillna(self.global_stats_[agg_func]))

        if self.drop_original:
            existing_cols = [col for col in self.cols_to_encode if col in X_transformed.columns]
            X_transformed = X_transformed.drop(columns=existing_cols)
        return X_transformed

    def fit_transform(self, X, y):
        """Fit and transform using CV to prevent leakage."""
        self.fit(X, y)
        
        if self.cv == 1:
            return self.transform(X)
        
        is_cudf = isinstance(X, cudf.DataFrame)
        
        if is_cudf:
            encoded_features = cudf.DataFrame(index=X.index)
        else:
            encoded_features = pd.DataFrame(index=X.index)
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        X_pandas_for_split = X.to_pandas() if is_cudf else X
        y_pandas_for_split = y.to_pandas() if isinstance(y, cudf.Series) else y

        for train_idx, val_idx in kf.split(X_pandas_for_split, y_pandas_for_split):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            temp_df = X_train.copy()
            temp_df['target'] = y_train

            cols_to_encode = [col for col in self.cols_to_encode if col in X_val.columns]

            for col in cols_to_encode:
                for agg_func in self.aggs:
                    new_col = f'TE_{col}_{agg_func}'
                    
                    if agg_func == 'mean':
                        fold_global = float(y_train.mean())
                        mapping, fold_global = self._apply_smoothing(
                            temp_df, col, agg_func, fold_global
                        )
                        encoded_values = X_val[col].map(mapping).fillna(fold_global)
                    elif agg_func == 'std':
                        fold_global = float(y_train.std() if len(y_train) > 1 else 0.0)
                        mapping = temp_df.groupby(col)['target'].agg('std').fillna(0.0)
                        encoded_values = X_val[col].map(mapping).fillna(fold_global)
                    elif agg_func == 'count':
                        temp_counts = temp_df.groupby(col).size()
                        fold_global = float(temp_counts.mean())
                        mapping = temp_counts.astype(float)
                        encoded_values = X_val[col].map(mapping).fillna(fold_global)
                    elif agg_func == 'nunique':
                        fold_global = 1.0
                        mapping = temp_df.groupby(col)['target'].nunique().astype(float)
                        encoded_values = X_val[col].map(mapping).fillna(fold_global)
                    elif agg_func == 'median':
                        fold_global = float(y_train.median())
                        mapping = temp_df.groupby(col)['target'].agg('median')
                        encoded_values = X_val[col].map(mapping).fillna(fold_global)
                    elif agg_func == 'min':
                        fold_global = float(y_train.min())
                        mapping = temp_df.groupby(col)['target'].agg('min')
                        encoded_values = X_val[col].map(mapping).fillna(fold_global)
                    elif agg_func == 'max':
                        fold_global = float(y_train.max())
                        mapping = temp_df.groupby(col)['target'].agg('max')
                        encoded_values = X_val[col].map(mapping).fillna(fold_global)
                    else:
                        fold_global = float(y_train.agg(agg_func))
                        mapping, fold_global = self._apply_smoothing(
                            temp_df, col, agg_func, fold_global
                        )
                        encoded_values = X_val[col].map(mapping).fillna(fold_global)
                    
                    if is_cudf:
                        encoded_features.loc[X.index[val_idx], new_col] = encoded_values.values
                    else:
                        encoded_features.loc[X.index[val_idx], new_col] = encoded_values.values

        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]

        if self.drop_original:
            existing_cols = [col for col in self.cols_to_encode if col in X_transformed.columns]
            X_transformed = X_transformed.drop(columns=existing_cols)
        
        return X_transformed

def apply_target_encoders(encoders, X_train, y_train, *datasets):
    """Apply target encoding to training and other datasets."""
    for encoder in encoders:
        X_train = encoder.fit_transform(X_train, y_train)
        datasets = tuple(encoder.transform(dataset) for dataset in datasets)
        datasets = tuple(dataset.reset_index(drop=True) for dataset in datasets)
    return (X_train, *datasets)

def cast_categoricals(datasets, categorical_cols):
    """Cast specified columns to category type."""
    for df in datasets:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

def create_frequency_features(datasets, feature_columns, numeric_columns):
    """Create frequency encodings and quantile binning for numeric columns."""
    frequency_cols = []
    numeric_set = set(numeric_columns)

    for col in feature_columns:
        if col not in datasets[0].columns:
            continue
        
        # Add frequency feature name once
        freq_name = f"{col}_freq"
        frequency_cols.append(freq_name)
        
        # Create feature for all datasets
        for df in datasets:
            freq = df[col].value_counts()
            df[freq_name] = df[col].map(freq) 
            df[freq_name] = df[freq_name].astype('float64').fillna(freq.mean() if len(freq) else 0.0)

        # Quantile binning for numeric columns
        if col in numeric_set:
            for q in (5, 10, 15, 20, 25, 30, 40, 50):
                bin_name = f"{col}_bin{q}"
                frequency_cols.append(bin_name)  # Add bin name once
                
                for df in datasets:
                    try:
                        quantiles = df[col].quantile(np.linspace(0, 1, q + 1)).to_pandas().values
                        quantiles = np.unique(quantiles)
                        
                        if len(quantiles) > 1:
                            df[bin_name] = cudf.cut(df[col], bins=quantiles, labels=False, include_lowest=True)
                        else:
                            df[bin_name] = 0
                    except (ValueError, Exception):
                        df[bin_name] = 0

    return frequency_cols


def add_enhanced_features(datasets):
    enhanced_features = []
    
    for df in datasets:
        df['subgrade'] = df['grade_subgrade'].str[1:].astype(int)
        enhanced_features.append('subgrade')

        df['total_debt'] = df['loan_amount'] + (df['debt_to_income_ratio'] * df['annual_income'])
        enhanced_features.append('total_debt')

        df['annual_payment_estimate'] = df['loan_amount'] * df['interest_rate'] / 100
        enhanced_features.append('annual_payment_estimate')

        df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount']+1)
        enhanced_features.append('income_to_loan_ratio')

        df['debt_burden'] = df['debt_to_income_ratio'] * df['annual_income']
        enhanced_features.append('debt_burden')
    
    return list(set(enhanced_features))

def create_interaction_features(datasets, columns, exclusions=None):
    """Create pairwise interaction features."""
    interaction_cols = []
    interaction_base = [col for col in columns if col not in (exclusions or set())]

    for col1, col2 in combinations(interaction_base, 2):
        new_col = f"{col1}_{col2}"
        interaction_cols.append(new_col)
        for df in datasets:
            df[new_col] = df[col1].astype(str) + "_" + df[col2].astype(str)

    return interaction_cols

def create_round_features(datasets, columns, rounding_levels):
    """Create rounded versions of numeric columns."""
    rounded_cols = []
    for col in columns:
        for suffix, level in rounding_levels.items():
            new_col = f"{col}_ROUND_{suffix}"
            rounded_cols.append(new_col)
            for df in datasets:
                df[new_col] = df[col].round(level).astype(int)
    return rounded_cols

def prepare_features(train_df, test_df, orig_df):
    """Prepare all feature sets."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    orig_df = orig_df.copy()
    
    train_cols = [col for col in train_df.columns if col not in EXCLUDED_BASE_COLS]
    ORIGINAL_BASE_FEATURES = train_cols
    orig_df = orig_df[[col for col in ORIGINAL_BASE_FEATURES if col in orig_df.columns] + [TARGET]]
    
    datasets = (train_df, test_df, orig_df)

    enhanced_features = add_enhanced_features(datasets)
    print(f"{len(enhanced_features)} enhanced features created: {enhanced_features}")
    
    base_features = [col for col in train_df.columns if col not in EXCLUDED_BASE_COLS]
    numeric_cols = [col for col in base_features if train_df[col].dtype not in ["object", "category"]]
    
    frequency_features = create_frequency_features(datasets, base_features, numeric_cols)
    print(f"{len(frequency_features)} frequency features created.")

    interaction_features = create_interaction_features(datasets, ORIGINAL_BASE_FEATURES, INTERACTION_EXCLUSIONS)
    print(f"{len(interaction_features)} interaction features created.")

    round1 = create_round_features(datasets, ["annual_income", "loan_amount"], {"1s": 0, "10s": -1, '100s': -2, '1000s': -3})
    round2 = create_round_features(datasets, ["credit_score"], {"10s": -1, '100s': -2})
    round3 = create_round_features(datasets, ["interest_rate"], {'1s': 0, '1': 1})
    round4 = create_round_features(datasets, ["debt_to_income_ratio"], {'1': 1, '2': 2})

    all_round_features = round1 + round2 + round3 + round4
    print(f"{len(all_round_features)} round features created.")

    all_features = base_features + frequency_features + interaction_features + all_round_features
    print(f"{len(all_features)} total features.")

    return all_features, base_features, interaction_features, all_round_features, train_df, test_df, orig_df

# ===========================
# STAGE 1: Train on Original Data
# ===========================

def train_stage1(
    train_df, test_df, orig_df, 
    all_features, base_features,interaction_features, all_round_features
):
    """
    Stage 1: Train on original dataset and predict train/test logits
    Returns: train_logits (z0), test_logits (z0)
    """
    print("\n" + "=" * 60)
    print("STAGE 1: TRAINING ON ORIGINAL DATA")
    print("=" * 60)
    
    X_orig = orig_df[all_features].copy()
    y_orig = orig_df[TARGET]
    
    oof_orig_preds=np.zeros(len(orig_df))
    train_logits = np.zeros(len(train_df))
    test_logits = np.zeros(len(test_df))
    
    # Apply target encoding once for train/test (for final predictions)
    print("\nApplying target encoding to train/test for Stage 1 predictions...")
    encoders = [
        TargetEncoder(cols_to_encode=base_features, smooth="auto", aggs=["mean", "count"], drop_original=False),
        TargetEncoder(cols_to_encode=interaction_features, smooth="auto", aggs=["mean"], drop_original=True),
        TargetEncoder(cols_to_encode=all_round_features, smooth="auto", aggs=["mean"], drop_original=False),
    ]
    
    train_features_encoded, test_features_encoded = apply_target_encoders(
        encoders, train_df[all_features], train_df[TARGET], test_df[all_features]
    )
    cast_categoricals([train_features_encoded, test_features_encoded], CATEGORICAL_COLS)
    print(f"Train features encoded: {train_features_encoded.shape} | Test features encoded: {test_features_encoded.shape}")
    
    # K-Fold training on original data
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    X_orig_pd = X_orig.to_pandas() if isinstance(X_orig, cudf.DataFrame) else X_orig
    y_orig_pd = y_orig.to_pandas() if isinstance(y_orig, cudf.Series) else y_orig
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_orig_pd, y_orig_pd), 1):
        print(f"\nFold {fold}/{N_SPLITS}")
        
        X_train = X_orig.iloc[train_idx].copy()
        y_train = y_orig.iloc[train_idx]
        X_val = X_orig.iloc[val_idx]
        y_val = y_orig.iloc[val_idx]
        
        # Target encoding for this fold
        fold_encoders = [
            TargetEncoder(cols_to_encode=base_features, smooth="auto", aggs=["mean", "count"], drop_original=False),
            TargetEncoder(cols_to_encode=interaction_features, smooth="auto", aggs=["mean"], drop_original=True),
            TargetEncoder(cols_to_encode=all_round_features, smooth="auto", aggs=["mean"], drop_original=False),
        ]
        print("Applying target encoding for original data...")
        X_train_enc, X_val_enc = apply_target_encoders(fold_encoders, X_train, y_train, X_val)
        cast_categoricals([X_train_enc, X_val_enc], CATEGORICAL_COLS)
        print(f"Train features encoded: {X_train_enc.shape} | Test features encoded: {X_val_enc.shape}")
        
        # Convert to pandas for LightGBM
        X_train_pd = X_train_enc.to_pandas() if isinstance(X_train_enc, cudf.DataFrame) else X_train_enc
        X_val_pd = X_val_enc.to_pandas() if isinstance(X_val_enc, cudf.DataFrame) else X_val_enc
        y_train_pd = y_train.to_pandas() if isinstance(y_train, cudf.Series) else y_train
        y_val_pd = y_val.to_pandas() if isinstance(y_val, cudf.Series) else y_val

        # create lgbm datasets
        train_dataset = lgb.Dataset(X_train_pd, label=y_train_pd)
        valid_dataset = lgb.Dataset(X_val_pd, label=y_val_pd, reference=train_dataset)
        
        # Train model
        model = lgb.train(
            LGBM_PARAMS,
            train_dataset,
            num_boost_round=5000,
            valid_sets=[train_dataset, valid_dataset],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(period=500)  # 0 to suppress output
            ]
        )
        
        # Predict and evaluate
        val_preds = model.predict(X_val_pd, num_iteration=model.best_iteration)
        y_val_np = y_val_pd.to_numpy() if hasattr(y_val_pd, 'to_numpy') else np.array(y_val_pd)
        fold_auc = roc_auc_score(y_val_np, val_preds)
        print(f"  Fold {fold} AUC: {fold_auc:.6f} | Best iteration: {model.best_iteration}")
        oof_orig_preds[val_idx] = val_preds

        # predict on train and test (convert to pandas if needed)
        train_features_encoded_pd = train_features_encoded.to_pandas() if isinstance(train_features_encoded, cudf.DataFrame) else train_features_encoded
        test_features_encoded_pd = test_features_encoded.to_pandas() if isinstance(test_features_encoded, cudf.DataFrame) else test_features_encoded
        
        fold_train_preds = model.predict(train_features_encoded_pd, num_iteration=model.best_iteration)
        fold_test_preds = model.predict(test_features_encoded_pd, num_iteration=model.best_iteration)

        # accumulate predictions
        train_logits += fold_train_preds / N_SPLITS
        test_logits += fold_test_preds / N_SPLITS

        # Cleanup
        del model, train_dataset, valid_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # overall auc for original data
    y_orig_np = y_orig.to_numpy() if hasattr(y_orig, 'to_numpy') else np.array(y_orig)
    overall_auc = roc_auc_score(y_orig_np, oof_orig_preds)
    print(f"Stage 1 Overall OOF AUC: {overall_auc:.6f}")
    
    # Convert probabilities to logits: z0 = log(p0 / (1 - p0))
    train_logits = np.clip(train_logits, 1e-8, 1 - 1e-8)
    test_logits = np.clip(test_logits, 1e-8, 1 - 1e-8)
    
    train_logits = np.log(train_logits / (1 - train_logits))
    test_logits = np.log(test_logits / (1 - test_logits))
    
    print(f"\n{'='*60}")
    print(f"Stage 1 Complete:")
    # print(f"  Train logits - mean: {train_logits.mean():.4f}, std: {train_logits.std():.4f}")
    # print(f"  Test logits - mean: {test_logits.mean():.4f}, std: {test_logits.std():.4f}")
    print(f"{'='*60}")
    
    return train_logits, test_logits

# ===========================
# STAGE 2: Train on Train Data with Base Margins
# ===========================

def train_stage2(
    train_df, test_df, 
    all_features, base_features,interaction_features, all_round_features, 
    stage1_train_logits, stage1_test_logits
):
    """
    Stage 2: Train on train data to predict residual w0
    Using base_margin=z0, the model learns w0 such that:
    y_pred = sigma(z0 + w0) minimizes BCE with y_true
    
    Returns: OOF predictions and test predictions (as probabilities)
    """
    print("\n" + "=" * 60)
    print("STAGE 2: TRAINING ON TRAIN DATA WITH BASE MARGINS")
    print("=" * 60)
    
    X = train_df[all_features].copy()
    y = train_df[TARGET]
    test_features = test_df[all_features].copy()
    print(f"Train features: {X.shape} | Test features: {test_features.shape}")
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    X_pandas = X.to_pandas() if isinstance(X, cudf.DataFrame) else X
    y_pandas = y.to_pandas() if isinstance(y, cudf.Series) else y
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_pandas, y_pandas), 1):
        print(f"\nStage 2 - Fold {fold}/{N_SPLITS}")
        
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        X_test = test_features.copy()
        
        # Get base margins (z0) for this fold
        train_base_margin = stage1_train_logits[train_idx]
        val_base_margin = stage1_train_logits[val_idx]
        test_base_margin = stage1_test_logits
        
        # Target encoding
        encoders = [
            TargetEncoder(cols_to_encode=base_features, smooth="auto", aggs=["mean", "count"], drop_original=False),
            TargetEncoder(cols_to_encode=interaction_features, smooth="auto", aggs=["mean"], drop_original=True),
            TargetEncoder(cols_to_encode=all_round_features, smooth="auto", aggs=["mean"], drop_original=False),
        ]
        print("Applying target encoding for train data...")
        
        X_train_enc, X_val_enc, X_test_enc = apply_target_encoders(
            encoders, X_train, y_train, X_val, X_test
        )
        cast_categoricals([X_train_enc, X_val_enc, X_test_enc], CATEGORICAL_COLS)
        print(f"Train features encoded: {X_train_enc.shape} | Test features encoded: {X_test_enc.shape}")
        # Create DMatrix with base_margin
        # KEY: base_margin tells XGBoost to start from z0 and learn w0
        dtrain = xgb.DMatrix(X_train_enc, label=y_train, 
                            base_margin=train_base_margin, 
                            enable_categorical=True)
        dvalid = xgb.DMatrix(X_val_enc, label=y_val, 
                            base_margin=val_base_margin,
                            enable_categorical=True)
        dtest = xgb.DMatrix(X_test_enc, 
                           base_margin=test_base_margin,
                           enable_categorical=True)
        
        # Train model (it will learn residual w0)
        model = xgb.train(
            params=XGB_PARAMS,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=VERBOSE_EVAL,
        )

        best_iter = model.best_iteration
        
        # Predict (returns sigma(z0 + w0) directly as probabilities)
        val_preds = model.predict(dvalid, iteration_range=(0, best_iter + 1))
        test_fold_preds = model.predict(dtest, iteration_range=(0, best_iter + 1))
        
        # Store predictions
        oof_preds[val_idx] = val_preds
        test_preds += test_fold_preds / N_SPLITS
        
        # Calculate AUC
        y_val_np = y_val.to_numpy() if hasattr(y_val, 'to_numpy') else np.array(y_val)
        fold_auc = roc_auc_score(y_val_np, val_preds)
        print(f"  Fold {fold} AUC: {fold_auc:.6f} | Best iteration: {best_iter}")
        
        # Cleanup
        del model, dtrain, dvalid, dtest
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate overall OOF AUC
    y_np = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)
    overall_auc = roc_auc_score(y_np, oof_preds)
    print(f"\n{'='*60}")
    print(f"Stage 2 Overall OOF AUC: {overall_auc:.6f}")
    print(f"{'='*60}")
    
    return oof_preds, test_preds, overall_auc


# ===========================
# MAIN EXECUTION FUNCTION
# ===========================

def main():
    """
    Main function to execute two-stage XGBoost training strategy
    """
    print("\n" + "=" * 80)
    print("TWO-STAGE XGBOOST TRAINING")
    print("=" * 80)
    
    # Load data
    print("\nLoading data with cuDF...")
    train_df = cudf.read_csv(DATA_PATHS["train"])
    test_df = cudf.read_csv(DATA_PATHS["test"])
    orig_df = cudf.read_csv(DATA_PATHS["orig"])
    print(f"Train: {train_df.shape} | Test: {test_df.shape} | Orig: {orig_df.shape}")
    
    # Prepare features
    print("\nPreparing features...")
    all_features, base_features, interaction_features, all_round_features, train_df, test_df, orig_df = prepare_features(
        train_df, test_df, orig_df
    )
    
    # Stage 1: Train on original data
    stage1_train_logits, stage1_test_logits = train_stage1(
        train_df, test_df, orig_df, 
        all_features, base_features, 
        interaction_features, all_round_features
    )
    
    # Stage 2: Train on train data with base margins
    oof_preds, test_preds, overall_auc = train_stage2(
        train_df, test_df, 
        all_features, base_features,
        interaction_features, all_round_features,
        stage1_train_logits, stage1_test_logits
    )
    
    # Save final predictions
    oof_preds_df = pd.DataFrame({'id': train_df['id'].to_pandas(), TARGET: oof_preds})
    test_preds_df = pd.DataFrame({'id': test_df['id'].to_pandas(), TARGET: test_preds})
    oof_preds_df.to_csv(f"oof_predictions_{overall_auc:.6f}.csv", index=False)
    test_preds_df.to_csv(f"test_predictions_{overall_auc:.6f}.csv", index=False)
    print(f"\nOOF predictions saved to: {f'oof_predictions_{overall_auc:.6f}.csv'}")
    print(f"Test predictions saved to: {f'test_predictions_{overall_auc:.6f}.csv'}")
    
    print("\n" + "=" * 80)
    print("TWO-STAGE TRAINING COMPLETE!")
    print("=" * 80)
    

if __name__ == "__main__":
    main()




