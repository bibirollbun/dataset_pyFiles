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


# ============================================
# INSTALLATIONS
# ============================================
!pip install -q prophet
!pip install -q koolbox
!pip install -q scikit-learn==1.5.2
!pip install -q autogluon
!pip install -q flaml[automl]
!pip install -q mljar-supervised
!pip install -q h2o
!pip install -q optuna
!pip install -q lightgbm
!pip install -q xgboost
!pip install -q catboost

# ============================================
# IMPORTS
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.stats import pearsonr
from sklearn.base import clone, BaseEstimator, RegressorMixin
from koolbox import Trainer
import joblib
import gc
import os

# ============================================
# CONFIGURATION
# ============================================
class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    target = "label"
    n_folds = 5
    seed = 42
    
    # Feature list
    X_FEATURES = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779',
                  'X25', 'X532', 'X520', 'X329', 'X383', 'X751', 'X535', 'X639', 'X596', 'X761',
                  "X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51",
                  "X598", "X385", "X603", "X674", "X415", "X345", "X174", "X178", "X168", "X612",
                  "bid_qty", "ask_qty", "buy_qty", "sell_qty"]

# ============================================
# UTILITY FUNCTIONS
# ============================================
def _pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def reduce_mem_usage(dataframe, dataset):    
    print(f'Reducing memory usage for: {dataset}')
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype
        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
    print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
    print(f'--- Decreased memory usage by {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%\n')

    return dataframe

# ============================================
# FEATURE ENGINEERING
# ============================================
def feature_engineering(df):
    # Original features
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])
    
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    # New microstructure features
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    return df

# ============================================
# OUTLIER DETECTION WITH PROPHET
# ============================================
def prophet_outlier_detection(df, feature_col, timestamp_col='__index_level_0__'):
    """Simple Prophet-based outlier detection"""
    print(f"Detecting outliers in {feature_col}...")
    
    # Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': df[timestamp_col],
        'y': df[feature_col]
    })
    
    # Remove obvious bad values
    prophet_df = prophet_df[np.isfinite(prophet_df['y'])]
    
    # Resample to hourly for efficiency
    prophet_hourly = prophet_df.set_index('ds').resample('1H').mean().reset_index()
    prophet_hourly = prophet_hourly.dropna()
    
    # Fit Prophet model
    model = Prophet(
        changepoint_prior_scale=0.05,
        interval_width=0.95,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    
    model.fit(prophet_hourly)
    
    # Generate predictions
    forecast = model.predict(prophet_hourly)
    
    # Identify outliers
    residuals = prophet_hourly['y'] - forecast['yhat']
    residual_std = residuals.std()
    outliers_mask = np.abs(residuals) > 3 * residual_std
    
    outlier_pct = outliers_mask.sum() / len(prophet_hourly) * 100
    print(f"Found {outliers_mask.sum()} outliers ({outlier_pct:.2f}%)")
    
    return outliers_mask, residual_std

# ============================================
# ADVANCED MODELS
# ============================================
class NoiseAwareFeatureCompressor:
    """Advanced feature compression with noise awareness"""
    def __init__(self, n_components=30, noise_threshold=0.1):
        self.n_components = n_components
        self.noise_threshold = noise_threshold
        self.pca = None
        self.scaler = StandardScaler()
        self.noise_mask = None
        
    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        
        # Estimate noise level per feature
        noise_levels = []
        for col in range(X_scaled.shape[1]):
            diff = np.diff(X_scaled[:, col])
            noise_estimate = np.std(diff) / np.sqrt(2)
            noise_levels.append(noise_estimate)
        
        noise_levels = np.array(noise_levels)
        self.noise_mask = noise_levels < np.percentile(noise_levels, 100 * (1 - self.noise_threshold))
        
        # Apply PCA on low-noise features
        X_clean = X_scaled[:, self.noise_mask]
        self.pca = PCA(n_components=min(self.n_components, X_clean.shape[1]))
        self.pca.fit(X_clean)
        
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_clean = X_scaled[:, self.noise_mask]
        pca_features = self.pca.transform(X_clean)
        high_noise_features = X_scaled[:, ~self.noise_mask]
        
        return np.hstack([pca_features, high_noise_features])
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# ============================================
# AUTOML FUNCTIONS
# ============================================
def train_autogluon(X_train, y_train, X_test):
    """AutoGluon with hierarchical ensemble"""
    try:
        from autogluon.tabular import TabularPredictor
        
        train_data = X_train.copy()
        train_data['label'] = y_train
        
        predictor = TabularPredictor(
            label='label',
            problem_type='regression',
            eval_metric=lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
            path='autogluon_models'
        )
        
        predictor.fit(
            train_data=train_data,
            presets='best_quality',
            ag_args_fit={
                'num_bag_folds': 10,
                'num_bag_sets': 3,
                'num_stack_levels': 2,
                'refit_full': True
            },
            time_limit=1800,
            verbosity=1
        )
        
        predictions = predictor.predict(X_test)
        oof_predictions = predictor.get_oof_pred()
        
        return predictions, oof_predictions, predictor
    except Exception as e:
        print(f"AutoGluon failed: {e}")
        return None, None, None

def train_flaml(X_train, y_train, X_test):
    """FLAML AutoML"""
    try:
        from flaml import AutoML
        
        automl = AutoML()
        settings = {
            "time_budget": 1800,
            "metric": lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
            "task": "regression",
            "n_splits": 5,
            "eval_method": "cv",
            "seed": 42,
            "learner_selector": "tournament",
            "estimator_list": ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree', 'lrl1', 'lrl2']
        }
        
        automl.fit(X_train, y_train, **settings)
        predictions = automl.predict(X_test)
        
        # Get OOF predictions
        kf = KFold(n_splits=5, shuffle=False)
        oof_predictions = np.zeros(len(X_train))
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            
            model = automl.model.estimator.__class__(**automl.model.estimator.get_params())
            model.fit(X_fold_train, y_fold_train)
            oof_predictions[val_idx] = model.predict(X_fold_val)
        
        return predictions, oof_predictions, automl
    except Exception as e:
        print(f"FLAML failed: {e}")
        return None, None, None

# ============================================
# MAIN PIPELINE
# ============================================
def main():
    print("Starting DRW Crypto Market Prediction Pipeline...")
    
    # Load data
    print("\n1. Loading data...")
    train = pd.read_parquet(CFG.train_path).reset_index(drop=True)
    test = pd.read_parquet(CFG.test_path).reset_index(drop=True)
    
    # Select features
    selected_columns = CFG.X_FEATURES + ["volume"]
    train = train[selected_columns + [CFG.target]]
    test = test[selected_columns]
    
    # Add timestamp if missing
    if '__index_level_0__' not in train.columns:
        train['__index_level_0__'] = pd.date_range('2023-03-01', periods=len(train), freq='T')
    if '__index_level_0__' not in test.columns:
        test['__index_level_0__'] = pd.date_range('2024-03-01', periods=len(test), freq='T')
    
    # Apply feature engineering
    print("\n2. Feature Engineering...")
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    # Remove base features
    to_remove = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "__index_level_0__"]
    train = train.drop(columns=[col for col in to_remove if col in train.columns])
    test = test.drop(columns=[col for col in to_remove if col in test.columns])
    
    # Reduce memory
    train = reduce_mem_usage(train, "train")
    test = reduce_mem_usage(test, "test")
    
    # Prepare data
    X = train.drop(CFG.target, axis=1)
    y = train[CFG.target]
    X_test = test
    
    # Store results
    all_oof_preds = {}
    all_test_preds = {}
    all_scores = {}
    
    # ============================================
    # FIRST LEVEL MODELS
    # ============================================
    print("\n3. Training First Level Models...")
    
    # Model parameters
    lgbm_params = {
        "boosting_type": "gbdt",
        "colsample_bytree": 0.5625888953382505,
        "learning_rate": 0.029312951475451557,
        "min_child_samples": 63,
        "min_child_weight": 0.11456572852335424,
        "n_estimators": 126,
        "n_jobs": -1,
        "num_leaves": 37,
        "random_state": 42,
        "reg_alpha": 85.2476527854083,
        "reg_lambda": 99.38305361388907,
        "subsample": 0.450669817684892,
        "verbose": -1
    }
    
    xgb_params = {
        "colsample_bylevel": 0.4778015829774066,
        "colsample_bynode": 0.362764358742407,
        "colsample_bytree": 0.7107423488010493,
        "gamma": 1.7094857725240398,
        "learning_rate": 0.02213323588455387,
        "max_depth": 20,
        "max_leaves": 12,
        "min_child_weight": 16,
        "n_estimators": 1667,
        "n_jobs": -1,
        "random_state": 42,
        "reg_alpha": 39.352415706891264,
        "reg_lambda": 75.44843704068275,
        "subsample": 0.06566669853471274,
        "verbosity": 0
    }
    
    # Train models
    models = {
        'Lasso': Lasso(alpha=0.0005, max_iter=10000),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
        'LightGBM': LGBMRegressor(**lgbm_params),
        'XGBoost': XGBRegressor(**xgb_params),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=8, min_samples_leaf=30, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            trainer = Trainer(
                model,
                cv=KFold(n_splits=5, shuffle=False),
                metric=_pearsonr,
                task="regression",
                metric_precision=6
            )
            trainer.fit(X, y)
            
            all_scores[name] = trainer.fold_scores
            all_oof_preds[name] = trainer.oof_preds
            all_test_preds[name] = trainer.predict(X_test)
            
            print(f"{name} - Mean Score: {np.mean(trainer.fold_scores):.6f}")
        except Exception as e:
            print(f"{name} failed: {e}")
    
    # ============================================
    # AUTOML MODELS
    # ============================================
    print("\n4. Training AutoML Models...")
    
    # AutoGluon
    print("\nTraining AutoGluon...")
    ag_pred, ag_oof, ag_model = train_autogluon(X, y, X_test)
    if ag_pred is not None:
        all_test_preds['AutoGluon'] = ag_pred
        all_oof_preds['AutoGluon'] = ag_oof
        train_score = _pearsonr(y, ag_oof)
        all_scores['AutoGluon'] = [train_score] * 5
        print(f"AutoGluon - Train Score: {train_score:.6f}")
    
    # FLAML
    print("\nTraining FLAML...")
    flaml_pred, flaml_oof, flaml_model = train_flaml(X, y, X_test)
    if flaml_pred is not None:
        all_test_preds['FLAML'] = flaml_pred
        all_oof_preds['FLAML'] = flaml_oof
        train_score = _pearsonr(y, flaml_oof)
        all_scores['FLAML'] = [train_score] * 5
        print(f"FLAML - Train Score: {train_score:.6f}")
    
    # ============================================
    # ADVANCED TECHNIQUES
    # ============================================
    print("\n5. Applying Advanced Techniques...")
    
    # Noise-Aware Feature Compression
    print("\nApplying Noise-Aware Compression...")
    compressor = NoiseAwareFeatureCompressor(n_components=30)
    X_compressed = pd.DataFrame(compressor.fit_transform(X))
    X_test_compressed = pd.DataFrame(compressor.transform(X_test))
    
    # Train model on compressed features
    lgbm_compressed = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=20,
        reg_lambda=20,
        min_child_samples=50,
        random_state=42,
        verbose=-1
    )
    
    trainer_compressed = Trainer(
        lgbm_compressed,
        cv=KFold(n_splits=5, shuffle=False),
        metric=_pearsonr,
        task="regression",
        metric_precision=6
    )
    trainer_compressed.fit(X_compressed, y)
    
    all_scores['LGBM_Compressed'] = trainer_compressed.fold_scores
    all_oof_preds['LGBM_Compressed'] = trainer_compressed.oof_preds
    all_test_preds['LGBM_Compressed'] = trainer_compressed.predict(X_test_compressed)
    print(f"LGBM_Compressed - Mean Score: {np.mean(trainer_compressed.fold_scores):.6f}")
    
    # ============================================
    # FINAL ENSEMBLE
    # ============================================
    print("\n6. Creating Final Ensemble...")
    
    # Create ensemble features
    X_ensemble = pd.DataFrame(all_oof_preds)
    X_test_ensemble = pd.DataFrame(all_test_preds)
    
    # Train meta-model
    meta_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    meta_model.fit(X_ensemble, y)
    
    # Get final predictions
    final_predictions = meta_model.predict(X_test_ensemble)
    
    # Print ensemble weights
    print("\nEnsemble Weights:")
    for name, weight in zip(X_ensemble.columns, meta_model.coef_):
        print(f"{name}: {weight:.4f}")
    
    # Calculate ensemble OOF score
    ensemble_oof = meta_model.predict(X_ensemble)
    ensemble_score = _pearsonr(y, ensemble_oof)
    print(f"\nEnsemble Train Score: {ensemble_score:.6f}")
    
    # ============================================
    # SAVE RESULTS
    # ============================================
    print("\n7. Saving Results...")
    
    # Save predictions
    sub = pd.read_csv(CFG.sample_sub_path)
    sub["prediction"] = final_predictions
    sub.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")
    
    # Save OOF predictions
    joblib.dump(all_oof_preds, "all_oof_preds.pkl")
    joblib.dump(all_test_preds, "all_test_preds.pkl")
    
    # Plot results
    scores_df = pd.DataFrame(all_scores)
    mean_scores = scores_df.mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    scores_df.boxplot(vert=False)
    plt.title("Model Scores Distribution")
    plt.xlabel("Pearson Correlation")
    
    plt.subplot(2, 1, 2)
    mean_scores.plot(kind='barh')
    plt.title("Average Model Scores")
    plt.xlabel("Pearson Correlation")
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.show()
    
    print("\nPipeline completed successfully!")
    
    return final_predictions, ensemble_score

# ============================================
# RUN MAIN PIPELINE
# ============================================
if __name__ == "__main__":
    final_predictions, ensemble_score = main()

