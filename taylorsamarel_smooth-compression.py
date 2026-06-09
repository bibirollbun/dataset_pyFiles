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


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# List input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Install required packages
print("Installing required packages...")
os.system('pip install koolbox scikit-learn==1.5.2 prophet catboost --quiet')

# Import all required libraries
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.stats import pearsonr, rankdata
from statsmodels.robust.scale import mad
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from koolbox import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
import joblib
import gc
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)

print("Starting Comprehensive DRW Crypto Prediction Pipeline...")

# Configuration
class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    target = "label"
    n_folds = 5
    seed = 42
    random_state = 42

# Define features
X_FEATURES = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779',
              'X25', 'X532', 'X520', 'X329', 'X383', 'X751', 'X535', 'X639', 'X596', 'X761',
              'X752', 'X287', 'X298', 'X759', 'X302', 'X55', 'X56', 'X52', 'X303', 'X51',
              'X598', 'X385', 'X603', 'X674', 'X415', 'X345', 'X174', 'X178', 'X168', 'X612',
              'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']

SELECTED_COLUMNS = X_FEATURES + ['volume']

# Memory reduction function
def reduce_mem_usage(dataframe, dataset):
    print(f'Reducing memory usage for: {dataset}')
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype
        
        if col_type != 'object':
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

# Prophet-based outlier detection
def simple_prophet_outlier_detection(df, feature_col, timestamp_col='__index_level_0__'):
    """Use Prophet to detect outliers in a single feature"""
    print(f"Detecting outliers in {feature_col} using Prophet...")
    
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
    
    if len(prophet_hourly) < 100:
        print(f"    Insufficient data for {feature_col}")
        return None
    
    try:
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
        outliers_ci = (
            (prophet_hourly['y'] < forecast['yhat_lower']) | 
            (prophet_hourly['y'] > forecast['yhat_upper'])
        )
        
        residuals = prophet_hourly['y'] - forecast['yhat']
        residual_std = residuals.std()
        outliers_residual = np.abs(residuals) > 3 * residual_std
        
        # Combine both methods
        outliers = outliers_ci & outliers_residual
        
        outlier_info = {
            'n_outliers': outliers.sum(),
            'outlier_pct': outliers.sum() / len(prophet_hourly) * 100,
            'outlier_timestamps': prophet_hourly[outliers]['ds'].tolist(),
            'outlier_values': prophet_hourly[outliers]['y'].tolist(),
            'residual_std': residual_std
        }
        
        print(f"Found {outlier_info['n_outliers']} outliers ({outlier_info['outlier_pct']:.2f}%)")
        
        return outlier_info
        
    except Exception as e:
        print(f"    Prophet failed: {str(e)}")
        return None

def detect_outliers_multiple_features(df, features_to_check=None):
    """Run outlier detection on multiple features"""
    if features_to_check is None:
        features_to_check = ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']
    
    # Ensure timestamp
    if '__index_level_0__' not in df.columns:
        df['__index_level_0__'] = pd.date_range('2023-03-01', periods=len(df), freq='T')
    
    outlier_summary = {}
    
    for feature in features_to_check:
        if feature in df.columns:
            try:
                print(f"\n{'='*60}")
                outlier_info = simple_prophet_outlier_detection(df, feature)
                if outlier_info:
                    outlier_summary[feature] = outlier_info
                else:
                    # Create dummy outlier info if Prophet fails
                    outlier_summary[feature] = {
                        'n_outliers': 0,
                        'outlier_pct': 0,
                        'outlier_timestamps': [],
                        'outlier_values': [],
                        'residual_std': 1.0
                    }
            except Exception as e:
                print(f"Error processing {feature}: {str(e)}")
                outlier_summary[feature] = {
                    'n_outliers': 0,
                    'outlier_pct': 0,
                    'outlier_timestamps': [],
                    'outlier_values': [],
                    'residual_std': 1.0
                }
    
    return outlier_summary

def clean_feature_outliers(df, feature_col, outlier_info, method='cap', lower_percentile=1, upper_percentile=99):
    """Clean outliers from a feature using different methods"""
    outlier_timestamps = outlier_info['outlier_timestamps']
    df_clean = df.copy()
    
    if method == 'cap':
        # Cap outliers at specified percentiles
        lower_cap = df_clean[feature_col].quantile(lower_percentile / 100)
        upper_cap = df_clean[feature_col].quantile(upper_percentile / 100)
        
        df_clean[feature_col] = df_clean[feature_col].clip(lower=lower_cap, upper=upper_cap)
        print(f"Capped {feature_col} to range [{lower_cap:.2f}, {upper_cap:.2f}] (percentiles: {lower_percentile}th-{upper_percentile}th)")
    
    return df_clean

# Advanced Label Compressor (optimized version based on results)
class OptimizedLabelCompressor(BaseEstimator, TransformerMixin):
    """IQR-based label compression with optimal parameters"""
    
    def __init__(self, strategy='iqr', params=None):
        self.strategy = strategy
        self.params = params or {}
        self.fitted = False
        
    def fit(self, y):
        """Fit the compression parameters"""
        if self.strategy == 'iqr':
            self.q1 = np.percentile(y, 25)
            self.q3 = np.percentile(y, 75)
            self.iqr = self.q3 - self.q1
            self.whisker_width = self.params.get('whisker_width', 2.98)
        elif self.strategy == 'percentile':
            self.lower_percentile = self.params.get('lower_percentile', 5)
            self.upper_percentile = self.params.get('upper_percentile', 95)
            self.lower_bound = np.percentile(y, self.lower_percentile)
            self.upper_bound = np.percentile(y, self.upper_percentile)
        elif self.strategy == 'mad':
            self.median = np.median(y)
            self.mad = mad(y)
            self.n_mads = self.params.get('n_mads', 3)
        
        self.fitted = True
        return self
    
    def transform(self, y):
        """Apply compression to labels"""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        if self.strategy == 'iqr':
            lower_bound = self.q1 - self.whisker_width * self.iqr
            upper_bound = self.q3 + self.whisker_width * self.iqr
            
            if self.params.get('soft_clip', False):
                # Soft clipping
                y_compressed = y.copy()
                
                # Lower outliers
                mask_lower = y < lower_bound
                if np.any(mask_lower):
                    dist = lower_bound - y[mask_lower]
                    weight = np.exp(-dist / self.iqr)
                    y_compressed[mask_lower] = weight * y[mask_lower] + (1 - weight) * lower_bound
                
                # Upper outliers
                mask_upper = y > upper_bound
                if np.any(mask_upper):
                    dist = y[mask_upper] - upper_bound
                    weight = np.exp(-dist / self.iqr)
                    y_compressed[mask_upper] = weight * y[mask_upper] + (1 - weight) * upper_bound
                
                return y_compressed
            else:
                # Hard clipping
                return np.clip(y, lower_bound, upper_bound)
        
        elif self.strategy == 'percentile':
            return np.clip(y, self.lower_bound, self.upper_bound)
        
        elif self.strategy == 'mad':
            z_scores = (y - self.median) / (self.mad + 1e-8)
            compressed_z = self.n_mads * np.tanh(z_scores / self.n_mads)
            return self.median + compressed_z * self.mad
    
    def fit_transform(self, y):
        """Fit and transform in one step"""
        return self.fit(y).transform(y)

# Pearson correlation metric
def _pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# Model parameters (from best results)
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

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.34695458228489784,
    "learning_rate": 0.031023014900595287,
    "min_child_samples": 30,
    "min_child_weight": 0.4727729225033618,
    "n_estimators": 220,
    "n_jobs": -1,
    "num_leaves": 58,
    "random_state": 42,
    "reg_alpha": 38.665994901468224,
    "reg_lambda": 92.76991677464294,
    "subsample": 0.4810891284493255,
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

# GANDALF Model Implementation
class GANDALF(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, learning_rate=0.01, max_depth=5, 
                 feature_fraction=0.8, bagging_fraction=0.8, lambda_reg=1.0,
                 min_data_in_leaf=20, num_iterations=100, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.lambda_reg = lambda_reg
        self.min_data_in_leaf = min_data_in_leaf
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _build_network(self, input_dim):
        """Build the neural network architecture for GANDALF"""
        class GANDALFNet(nn.Module):
            def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
                super(GANDALFNet, self).__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers)
                
            def forward(self, x):
                return self.network(x)
        
        return GANDALFNet(input_dim).to(self.device)
    
    def fit(self, X, y):
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).reshape(-1, 1).to(self.device)
        
        # Build network
        self.model = self._build_network(X_tensor.shape[1])
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lambda_reg)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_iterations):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{self.num_iterations}], Loss: {total_loss/len(dataloader):.4f}')
        
        return self
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions

# AutoEncoder MLP
class AutoEncoderMLP(BaseEstimator, RegressorMixin):
    def __init__(self, num_columns, hidden_units, dropout_rates, lr=1e-3):
        self.num_columns = num_columns
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.lr = lr
        self.model = self._build_model()
    
    def _build_model(self):
        inp = tf.keras.layers.Input(shape=(self.num_columns,))
        x0 = tf.keras.layers.BatchNormalization()(inp)

        encoder = tf.keras.layers.GaussianNoise(self.dropout_rates[0])(x0)
        encoder = tf.keras.layers.Dense(self.hidden_units[0])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation('swish')(encoder)

        decoder = tf.keras.layers.Dropout(self.dropout_rates[1])(encoder)
        decoder = tf.keras.layers.Dense(self.num_columns, name='decoder')(decoder)

        x_reg = tf.keras.layers.Dense(self.hidden_units[1])(encoder)
        x_reg = tf.keras.layers.BatchNormalization()(x_reg)
        x_reg = tf.keras.layers.Activation('swish')(x_reg)
        x_reg = tf.keras.layers.Dropout(self.dropout_rates[2])(x_reg)

        out_reg = tf.keras.layers.Dense(1, activation='linear', name='target')(x_reg)

        model = tf.keras.models.Model(inputs=inp, outputs=[decoder, out_reg])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss={"decoder": tf.keras.losses.MeanSquaredError(),
                  "target": tf.keras.losses.MeanSquaredError()},
            loss_weights={"decoder": 0.3, "target": 1.0}
        )
        return model

    def fit(self, X, y):
        self.model.fit(
            X, {"decoder": X, "target": y},
            epochs=50,
            batch_size=8192,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5)
            ],
            verbose=0
        )
        return self

    def predict(self, X):
        _, y_pred = self.model.predict(X, verbose=0)
        return y_pred.flatten()

# Feature engineering
def create_features(df):
    """Create additional features"""
    # Spread features
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['buy_sell_imbalance'] = df['buy_qty'] - df['sell_qty']
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    
    # Volume features
    df['volume_per_trade'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['buy_volume_ratio'] = df['buy_qty'] / (df['volume'] + 1e-8)
    df['sell_volume_ratio'] = df['sell_qty'] / (df['volume'] + 1e-8)
    
    # Log transforms for skewed features
    for col in ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    return df

def run_pipeline_with_outlier_strategy(lower_percentile, upper_percentile, strategy_name):
    """Run the entire pipeline with a specific outlier capping strategy"""
    print(f"\n{'='*80}")
    print(f"Running pipeline with outlier capping strategy: {strategy_name}")
    print(f"Percentiles: {lower_percentile}th - {upper_percentile}th")
    print(f"{'='*80}\n")
    
    # Load data
    train = pd.read_parquet(CFG.train_path).reset_index(drop=True)
    test = pd.read_parquet(CFG.test_path).reset_index(drop=True)
    
    # Select columns
    train = train[SELECTED_COLUMNS + [CFG.target]]
    test = test[SELECTED_COLUMNS]
    
    # Add timestamp if needed
    if '__index_level_0__' not in train.columns:
        train['__index_level_0__'] = pd.date_range('2023-03-01', periods=len(train), freq='T')
    
    # Detect outliers
    print("\nDetecting outliers...")
    outlier_summary = detect_outliers_multiple_features(train, X_FEATURES)
    
    # Clean data with specified percentiles
    df_clean = train.copy()
    print(f"\nApplying outlier capping with {lower_percentile}th-{upper_percentile}th percentiles...")
    for feat in X_FEATURES:
        if feat in outlier_summary:
            df_clean = clean_feature_outliers(
                df_clean, feat, outlier_summary[feat], 
                method='cap', 
                lower_percentile=lower_percentile, 
                upper_percentile=upper_percentile
            )
    
    # Remove timestamp column if it was added
    if '__index_level_0__' in df_clean.columns:
        df_clean = df_clean.drop('__index_level_0__', axis=1)
    
    # Add feature engineering
    df_clean = create_features(df_clean)
    test = create_features(test)
    
    # Reduce memory
    df_clean = reduce_mem_usage(df_clean, "train")
    test = reduce_mem_usage(test, "test")
    
    # Prepare data
    X = df_clean.drop(CFG.target, axis=1)
    y = df_clean[CFG.target]
    X_test = test
    
    # Apply label compression (optimal IQR-based)
    label_compressor = OptimizedLabelCompressor(
        strategy='iqr', 
        params={'whisker_width': 2.98, 'soft_clip': False}
    )
    y_compressed = label_compressor.fit_transform(y)
    
    # Initialize storage
    scores = {}
    oof_preds = {}
    test_preds = {}
    
    # Train LightGBM (gbdt)
    print(f"\nTraining LightGBM (gbdt) for {strategy_name}...")
    lgbm_trainer = Trainer(
        LGBMRegressor(**lgbm_params),
        cv=KFold(n_splits=5, shuffle=False),
        metric=_pearsonr,
        task="regression",
        metric_precision=6
    )
    lgbm_trainer.fit(X, y_compressed)
    scores["LightGBM (gbdt)"] = lgbm_trainer.fold_scores
    oof_preds["LightGBM (gbdt)"] = lgbm_trainer.oof_preds
    test_preds["LightGBM (gbdt)"] = lgbm_trainer.predict(X_test)
    
    # Train LightGBM (goss)
    print(f"\nTraining LightGBM (goss) for {strategy_name}...")
    lgbm_goss_trainer = Trainer(
        LGBMRegressor(**lgbm_goss_params),
        cv=KFold(n_splits=5, shuffle=False),
        metric=_pearsonr,
        task="regression",
        metric_precision=6
    )
    lgbm_goss_trainer.fit(X, y_compressed)
    scores["LightGBM (goss)"] = lgbm_goss_trainer.fold_scores
    oof_preds["LightGBM (goss)"] = lgbm_goss_trainer.oof_preds
    test_preds["LightGBM (goss)"] = lgbm_goss_trainer.predict(X_test)
    
    # Train XGBoost
    print(f"\nTraining XGBoost for {strategy_name}...")
    xgb_trainer = Trainer(
        XGBRegressor(**xgb_params),
        cv=KFold(n_splits=5, shuffle=False),
        metric=_pearsonr,
        task="regression",
        metric_precision=6
    )
    xgb_trainer.fit(X, y_compressed)
    scores["XGBoost"] = xgb_trainer.fold_scores
    oof_preds["XGBoost"] = xgb_trainer.oof_preds
    test_preds["XGBoost"] = xgb_trainer.predict(X_test)
    
    # Train CatBoost
    print(f"\nTraining CatBoost for {strategy_name}...")
    cat_model = CatBoostRegressor(
        iterations=800,
        depth=10,
        learning_rate=0.02,
        l2_leaf_reg=3,
        subsample=0.85,
        random_state=42,
        verbose=False
    )
    cat_trainer = Trainer(
        cat_model,
        cv=KFold(n_splits=5, shuffle=False),
        metric=_pearsonr,
        task="regression",
        metric_precision=6
    )
    cat_trainer.fit(X, y_compressed)
    scores["CatBoost"] = cat_trainer.fold_scores
    oof_preds["CatBoost"] = cat_trainer.oof_preds
    test_preds["CatBoost"] = cat_trainer.predict(X_test)
    
    # Train GANDALF
    print(f"\nTraining GANDALF for {strategy_name}...")
    gandalf_model = GANDALF(
        n_estimators=100,
        learning_rate=0.001,
        max_depth=5,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        lambda_reg=0.1,
        min_data_in_leaf=20,
        num_iterations=50,
        random_state=42
    )
    gandalf_trainer = Trainer(
        gandalf_model,
        cv=KFold(n_splits=5, shuffle=False),
        metric=_pearsonr,
        task="regression",
        metric_precision=6
    )
    gandalf_trainer.fit(X, y_compressed)
    scores["GANDALF"] = gandalf_trainer.fold_scores
    oof_preds["GANDALF"] = gandalf_trainer.oof_preds
    test_preds["GANDALF"] = gandalf_trainer.predict(X_test)
    
    # Create ensemble predictions
    X_ensemble = pd.DataFrame(oof_preds)
    X_test_ensemble = pd.DataFrame(test_preds)
    
    # Train AutoEncoder
    print(f"\nTraining AutoEncoder for {strategy_name}...")
    ae_model = AutoEncoderMLP(
        num_columns=X_ensemble.shape[1],
        hidden_units=[128, 128],
        dropout_rates=[0.05, 0.1, 0.2],
        lr=1e-3
    )
    ae_trainer = Trainer(
        ae_model,
        cv=KFold(n_splits=5, shuffle=False),
        metric=_pearsonr,
        task="regression",
        metric_precision=6
    )
    ae_trainer.fit(X_ensemble, y)  # Note: use original y, not compressed
    scores["AutoEncoder"] = ae_trainer.fold_scores
    oof_preds["AutoEncoder"] = ae_trainer.oof_preds
    ae_test_preds = ae_trainer.predict(X_test_ensemble)
    
    # Create weighted ensemble
    ensemble_weights = {
        "LightGBM (gbdt)": 0.25,
        "LightGBM (goss)": 0.25,
        "XGBoost": 0.25,
        "CatBoost": 0.15,
        "GANDALF": 0.05,
        "AutoEncoder": 0.05
    }
    
    # Calculate weighted predictions
    weighted_test_pred = np.zeros(len(X_test))
    for model_name, weight in ensemble_weights.items():
        if model_name == "AutoEncoder":
            weighted_test_pred += weight * ae_test_preds
        else:
            weighted_test_pred += weight * test_preds[model_name]
    
    # Print results
    print(f"\nResults for {strategy_name}:")
    scores_df = pd.DataFrame(scores)
    mean_scores = scores_df.mean()
    print(mean_scores.sort_values(ascending=False))
    
    # Clean up memory
    del X, y, X_test, train, test, df_clean
    gc.collect()
    
    return weighted_test_pred, scores_df, label_compressor

# Main execution
def main():
    print("\n" + "="*80)
    print("DRW CRYPTO PREDICTION - COMPREHENSIVE SOLUTION")
    print("="*80)
    
    # Define outlier capping strategies
    strategies = [
        (1, 99, "1st-99th_percentile"),
        (5, 95, "5th-95th_percentile"),
        (10, 90, "10th-90th_percentile"),
        (25, 75, "25th-75th_percentile")
    ]
    
    # Store predictions for each strategy
    all_predictions = {}
    all_scores = {}
    
    # Run each strategy
    for lower, upper, name in strategies:
        try:
            predictions, scores, label_compressor = run_pipeline_with_outlier_strategy(lower, upper, name)
            all_predictions[name] = predictions
            all_scores[name] = scores
            
            # Save individual submission
            sub = pd.read_csv(CFG.sample_sub_path)
            sub["prediction"] = predictions
            sub.to_csv(f"submission_{name}.csv", index=False)
            print(f"\nSaved submission_{name}.csv")
            
        except Exception as e:
            print(f"\nError in strategy {name}: {str(e)}")
            continue
    
    # Create ensemble of all strategies
    print("\n" + "="*80)
    print("Creating ensemble of all strategies...")
    print("="*80)
    
    if len(all_predictions) > 0:
        # Simple average ensemble
        ensemble_predictions = np.zeros(len(next(iter(all_predictions.values()))))
        for strategy_name in all_predictions:
            ensemble_predictions += all_predictions[strategy_name]
        ensemble_predictions /= len(all_predictions)
        
        # Save ensemble submission
        sub = pd.read_csv(CFG.sample_sub_path)
        sub["prediction"] = ensemble_predictions
        sub.to_csv("submission_ensemble_all_strategies.csv", index=False)
        print("\nSaved submission_ensemble_all_strategies.csv")
        
        # Create weighted ensemble (favor middle strategies)
        if len(all_predictions) == 4:
            weights = [0.2, 0.3, 0.3, 0.2]  # Bell curve weights
            weighted_ensemble = np.zeros(len(ensemble_predictions))
            for i, (strategy_name, weight) in enumerate(zip(sorted(all_predictions.keys()), weights)):
                weighted_ensemble += weight * all_predictions[strategy_name]
            
            sub["prediction"] = weighted_ensemble
            sub.to_csv("submission_weighted_ensemble.csv", index=False)
            print("Saved submission_weighted_ensemble.csv")
        
        # Create median ensemble (most robust)
        predictions_matrix = np.array(list(all_predictions.values()))
        median_ensemble = np.median(predictions_matrix, axis=0)
        
        sub["prediction"] = median_ensemble
        sub.to_csv("submission_median_ensemble.csv", index=False)
        print("Saved submission_median_ensemble.csv")
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (strategy_name, scores_df) in enumerate(all_scores.items()):
            if idx < 4:
                mean_scores = scores_df.mean().sort_values(ascending=False)
                
                ax = axes[idx]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                bars = ax.barh(range(len(mean_scores)), mean_scores.values, color=colors[:len(mean_scores)])
                ax.set_yticks(range(len(mean_scores)))
                ax.set_yticklabels(mean_scores.index)
                ax.set_title(f"Mean Scores - {strategy_name}", fontsize=12)
                ax.set_xlabel("Pearson Correlation")
                
                # Add value labels
                for i, (value, bar) in enumerate(zip(mean_scores.values, bars)):
                    ax.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{value:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig("outlier_strategies_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary table
        print("\n" + "="*80)
        print("SUMMARY OF ALL STRATEGIES")
        print("="*80)
        
        summary_data = []
        for strategy_name in all_scores:
            scores_df = all_scores[strategy_name]
            mean_scores = scores_df.mean()
            summary_data.append({
                'Strategy': strategy_name,
                'LightGBM (gbdt)': mean_scores.get('LightGBM (gbdt)', 0),
                'LightGBM (goss)': mean_scores.get('LightGBM (goss)', 0),
                'XGBoost': mean_scores.get('XGBoost', 0),
                'CatBoost': mean_scores.get('CatBoost', 0),
                'GANDALF': mean_scores.get('GANDALF', 0),
                'AutoEncoder': mean_scores.get('AutoEncoder', 0),
                'Average': mean_scores.mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Correlation analysis
        print("\n" + "="*80)
        print("PREDICTION CORRELATION ANALYSIS")
        print("="*80)
        
        pred_df = pd.DataFrame(all_predictions)
        corr_matrix = pred_df.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0.9,
                    square=True, linewidths=1, cbar_kws={"shrink": .8})
        plt.title('Correlation Between Different Outlier Strategies')
        plt.tight_layout()
        plt.savefig('strategy_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED!")
        print("="*80)
        print("Generated files:")
        for i, (_, _, name) in enumerate(strategies, 1):
            if name in all_predictions:
                print(f"{i}. submission_{name}.csv")
        print(f"{len(strategies)+1}. submission_ensemble_all_strategies.csv")
        print(f"{len(strategies)+2}. submission_weighted_ensemble.csv")
        print(f"{len(strategies)+3}. submission_median_ensemble.csv")
        print(f"{len(strategies)+4}. outlier_strategies_comparison.png")
        print(f"{len(strategies)+5}. strategy_correlation_matrix.png")
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("1. Try 'submission_weighted_ensemble.csv' first (bell-curve weighted)")
        print("2. If unstable, use 'submission_median_ensemble.csv' (most robust)")
        print("3. Individual strategy files available for experimentation")
        print("="*80)
        
    else:
        print("No successful strategies completed. Please check the errors above.")

if __name__ == "__main__":
    main()

