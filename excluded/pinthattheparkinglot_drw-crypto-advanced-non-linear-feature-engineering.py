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


#!/usr/bin/env python3
"""
DRW Crypto Advanced Non-Linear Feature Engineering
=================================================

This implementation uses Random Forest and Gradient Boosting based transformations
to capture complex non-linear relationships without the memory constraints of PCA.
Maintains important feature preservation while applying advanced transformations
to the remaining features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction import FeatureHasher
from scipy.stats import pearsonr
import joblib

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, regularizers
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - deep learning methods will be skipped")

# Set random seed
np.random.seed(42)

# Define important features - original plus high-importance augmented features
IMPORTANT_FEATURES = [
    # Original important features
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333",
    # High-importance augmented features
    "X862_minus_X852", "X598_minus_X862", "regime_indicator_137_302", "X852_minus_X345",
    "rbf_862_852", "multidim_distance_kernel", "X168_minus_X612", "X532_plus_X888",
    "complex_interaction_862_852_345", "volume_weighted_technical"
]


class FeatureAugmenter:
    """Creates augmented features that have proven to be highly predictive"""
    
    def __init__(self):
        self.feature_stats = {}
        
    def fit(self, df):
        """Store statistics needed for transformation"""
        for col in ['X862', 'X852', 'X345', 'X532', 'X888', 'X137', 'X302', 
                   'X178', 'X168', 'X612', 'X598', 'volume']:
            if col in df.columns:
                col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(col_data) > 0:
                    self.feature_stats[f'{col}_mean'] = col_data.mean()
                    self.feature_stats[f'{col}_std'] = col_data.std()
                    self.feature_stats[f'{col}_p25'] = col_data.quantile(0.25)
                    self.feature_stats[f'{col}_p75'] = col_data.quantile(0.75)
                else:
                    self.feature_stats[f'{col}_mean'] = 0
                    self.feature_stats[f'{col}_std'] = 1
                    self.feature_stats[f'{col}_p25'] = -1
                    self.feature_stats[f'{col}_p75'] = 1
        return self
    
    def transform(self, df):
        """Create augmented features with robust handling"""
        augmented_features = []
        feature_names = []
        
        # Complex interactions between top features
        if all(f in df.columns for f in ['X862', 'X852', 'X345']):
            feat1 = np.nan_to_num(df['X862'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            feat2 = np.nan_to_num(df['X852'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            feat3 = np.nan_to_num(df['X345'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            
            interaction = np.tanh(np.clip(feat1, -10, 10)) * \
                         np.exp(-np.clip(np.abs(feat2) / 2, 0, 10)) * \
                         np.sign(feat3)
            augmented_features.append(interaction)
            feature_names.append('complex_interaction_862_852_345')
            
            ratio_poly = (feat1 ** 2) / (feat2 ** 2 + 1)
            ratio_poly = np.clip(ratio_poly, -1e6, 1e6)
            augmented_features.append(ratio_poly)
            feature_names.append('poly_ratio_862_852')
            
            augmented_features.append(feat1 - feat2)
            feature_names.append('X862_minus_X852')
            
            augmented_features.append(feat2 - feat3)
            feature_names.append('X852_minus_X345')
            
            rbf_sigma = max(self.feature_stats.get('X862_std', 1), 0.1)
            diff = np.clip(feat1 - feat2, -100, 100)
            rbf_862_852 = np.exp(-((diff) ** 2) / (2 * rbf_sigma ** 2))
            augmented_features.append(rbf_862_852)
            feature_names.append('rbf_862_852')
        
        # Market microstructure features
        if all(f in df.columns for f in ['bid_qty', 'ask_qty', 'volume', 'buy_qty', 'sell_qty']):
            bid = np.nan_to_num(df['bid_qty'].values, nan=0.0, posinf=1e10, neginf=0)
            ask = np.nan_to_num(df['ask_qty'].values, nan=0.0, posinf=1e10, neginf=0)
            vol = np.nan_to_num(df['volume'].values, nan=0.0, posinf=1e10, neginf=0)
            buy = np.nan_to_num(df['buy_qty'].values, nan=0.0, posinf=1e10, neginf=0)
            sell = np.nan_to_num(df['sell_qty'].values, nan=0.0, posinf=1e10, neginf=0)
            
            order_imbalance = np.clip((bid - ask) / (bid + ask + 1), -1, 1)
            flow_imbalance = np.clip((buy - sell) / (buy + sell + 1), -1, 1)
            kyle_lambda = flow_imbalance * np.sqrt(np.abs(order_imbalance)) / (np.log1p(vol) + 1)
            kyle_lambda = np.clip(kyle_lambda, -10, 10)
            augmented_features.append(kyle_lambda)
            feature_names.append('kyle_lambda_complex')
            
            total_pressure = bid + ask
            vol_mean = max(self.feature_stats.get('volume_mean', 1), 1)
            vol_adj_pressure = np.log1p(total_pressure) * np.exp(-np.clip(vol / vol_mean, 0, 10))
            augmented_features.append(vol_adj_pressure)
            feature_names.append('vol_adjusted_pressure')
            
            buy_intensity = np.clip(buy / (vol + 1), 0, 1)
            sell_intensity = np.clip(sell / (vol + 1), 0, 1)
            intensity_diff = buy_intensity - sell_intensity
            intensity_asymmetry = np.sign(intensity_diff) * np.log1p(np.abs(intensity_diff))
            augmented_features.append(intensity_asymmetry)
            feature_names.append('trade_intensity_asymmetry')
        
        # Cross-domain interactions
        if all(f in df.columns for f in ['X532', 'X888', 'volume']):
            x532 = np.nan_to_num(df['X532'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            x888 = np.nan_to_num(df['X888'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            vol = np.nan_to_num(df['volume'].values, nan=0.0, posinf=1e10, neginf=0)
            
            technical = np.clip(x532 * x888, -1e10, 1e10)
            vol_std = max(self.feature_stats.get('volume_std', 1), 0.1)
            vol_weighted_tech = technical * np.log1p(vol) / vol_std
            vol_weighted_tech = np.clip(vol_weighted_tech, -1e6, 1e6)
            augmented_features.append(vol_weighted_tech)
            feature_names.append('volume_weighted_technical')
            
            augmented_features.append(x532 + x888)
            feature_names.append('X532_plus_X888')
        
        # Regime-based features
        if 'X137' in df.columns and 'X302' in df.columns:
            feat1 = np.nan_to_num(df['X137'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            feat2 = np.nan_to_num(df['X302'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            
            p75_1 = self.feature_stats.get('X137_p75', np.percentile(feat1[np.isfinite(feat1)], 75) if np.any(np.isfinite(feat1)) else 1)
            p75_2 = self.feature_stats.get('X302_p75', np.percentile(feat2[np.isfinite(feat2)], 75) if np.any(np.isfinite(feat2)) else 1)
            p25_1 = self.feature_stats.get('X137_p25', np.percentile(feat1[np.isfinite(feat1)], 25) if np.any(np.isfinite(feat1)) else -1)
            p25_2 = self.feature_stats.get('X302_p25', np.percentile(feat2[np.isfinite(feat2)], 25) if np.any(np.isfinite(feat2)) else -1)
            
            regime_indicator = np.where(
                (feat1 > p75_1) & (feat2 > p75_2), 1,
                np.where((feat1 < p25_1) & (feat2 < p25_2), -1, 0)
            ).astype(np.float32)
            augmented_features.append(regime_indicator)
            feature_names.append('regime_indicator_137_302')
        
        # Distance-based features
        if all(f in df.columns for f in ['X178', 'X168', 'X612']):
            x178 = np.nan_to_num(df['X178'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            x168 = np.nan_to_num(df['X168'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            x612 = np.nan_to_num(df['X612'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            
            feat1 = (x178 - self.feature_stats.get('X178_mean', 0)) / max(self.feature_stats.get('X178_std', 1), 0.1)
            feat2 = (x168 - self.feature_stats.get('X168_mean', 0)) / max(self.feature_stats.get('X168_std', 1), 0.1)
            feat3 = (x612 - self.feature_stats.get('X612_mean', 0)) / max(self.feature_stats.get('X612_std', 1), 0.1)
            
            feat1 = np.clip(feat1, -10, 10)
            feat2 = np.clip(feat2, -10, 10)
            feat3 = np.clip(feat3, -10, 10)
            
            distance = np.sqrt(feat1**2 + feat2**2 + feat3**2)
            distance_kernel = np.exp(-np.clip(distance**2 / 2, 0, 50))
            augmented_features.append(distance_kernel)
            feature_names.append('multidim_distance_kernel')
            
            augmented_features.append(x168 - x612)
            feature_names.append('X168_minus_X612')
        
        # Additional top feature interactions
        if 'X598' in df.columns and 'X862' in df.columns:
            f598 = np.nan_to_num(df['X598'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            f862 = np.nan_to_num(df['X862'].values, nan=0.0, posinf=1e10, neginf=-1e10)
            
            augmented_features.append(f598 - f862)
            feature_names.append('X598_minus_X862')
        
        if augmented_features:
            for i, feat in enumerate(augmented_features):
                augmented_features[i] = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
            
            augmented_df = pd.DataFrame(
                np.column_stack(augmented_features),
                columns=feature_names,
                index=df.index
            )
            return pd.concat([df, augmented_df], axis=1)
        else:
            return df
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


class RandomForestFeatureTransformer:
    """Uses Random Forest to create non-linear feature transformations"""
    
    def __init__(self, n_estimators=100, max_leaf_nodes=32, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.n_jobs = n_jobs
        self.forest = None
        self.leaf_encoders = {}
        self.n_features_out = None
        
    def fit(self, X, y):
        """Fit the random forest and prepare for feature extraction"""
        print(f"Training Random Forest with {self.n_estimators} trees...")
        
        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_leaf_nodes=self.max_leaf_nodes,
            max_features='sqrt',
            min_samples_leaf=50,
            random_state=42,
            n_jobs=self.n_jobs
        )
        
        self.forest.fit(X, y)
        
        # Calculate output feature dimensions
        self.n_features_out = 0
        for i in range(self.n_estimators):
            tree = self.forest.estimators_[i]
            n_leaves = tree.tree_.node_count
            self.n_features_out += n_leaves
        
        print(f"Random Forest trained. Will generate {self.n_features_out} features")
        
        return self
    
    def transform(self, X):
        """Transform features using tree leaf indices"""
        # Get leaf indices for each tree
        leaf_indices = self.forest.apply(X)
        
        # Create feature matrix using hashing trick to reduce dimensionality
        hasher = FeatureHasher(n_features=min(200, self.n_features_out), 
                              input_type='pair')
        
        # Create feature pairs for hashing
        feature_dicts = []
        for sample_idx in range(X.shape[0]):
            pairs = [(f'tree_{tree_idx}', int(leaf_indices[sample_idx, tree_idx]))
                    for tree_idx in range(self.n_estimators)]
            feature_dicts.append(pairs)
        
        # Apply hashing
        hashed_features = hasher.transform(feature_dicts).toarray()
        
        return hashed_features
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class GradientBoostingFeatureEngineering:
    """Creates features by capturing patterns at different scales using gradient boosting"""
    
    def __init__(self, n_rounds=3, trees_per_round=50):
        self.n_rounds = n_rounds
        self.trees_per_round = trees_per_round
        self.models = []
        self.feature_importances = []
        
    def fit(self, X, y):
        """Fit gradient boosting models to capture residual patterns"""
        print(f"Training {self.n_rounds} rounds of gradient boosting...")
        
        residuals = y.copy()
        
        for round_idx in range(self.n_rounds):
            print(f"  Round {round_idx + 1}/{self.n_rounds}...")
            
            gb = GradientBoostingRegressor(
                n_estimators=self.trees_per_round,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42 + round_idx
            )
            
            gb.fit(X, residuals)
            self.models.append(gb)
            
            # Update residuals
            predictions = gb.predict(X)
            residuals = residuals - predictions
            
            # Store feature importances
            self.feature_importances.append(gb.feature_importances_)
        
        return self
    
    def transform(self, X):
        """Transform features using the trained models"""
        all_features = []
        
        # Get predictions from each round
        for round_idx, model in enumerate(self.models):
            # Predictions as features
            predictions = model.predict(X).reshape(-1, 1)
            all_features.append(predictions)
            
            # Extract leaf indices and create hashed features
            leaf_features = []
            for tree_idx in range(len(model.estimators_)):
                tree = model.estimators_[tree_idx, 0]
                leaf_indices = tree.apply(X)
                leaf_features.append(leaf_indices)
            
            # Hash tree features
            hasher = FeatureHasher(n_features=30, input_type='pair')
            hashed = hasher.transform(
                [[(f'r{round_idx}_t{tree_idx}', int(leaf_features[tree_idx][sample_idx]))
                  for tree_idx in range(len(leaf_features))]
                 for sample_idx in range(X.shape[0])]
            ).toarray()
            
            all_features.append(hashed)
        
        return np.hstack(all_features)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class PolynomialInteractionExtractor:
    """Extracts polynomial interactions from top features"""
    
    def __init__(self, n_top_features=20, degree=2):
        self.n_top_features = n_top_features
        self.degree = degree
        self.selector = None
        self.poly = None
        self.selected_indices = None
        
    def fit(self, X, y):
        """Select top features and fit polynomial transformer"""
        # Select top features using mutual information
        self.selector = SelectKBest(mutual_info_regression, k=min(self.n_top_features, X.shape[1]))
        X_selected = self.selector.fit_transform(X, y)
        self.selected_indices = self.selector.get_support(indices=True)
        
        # Create polynomial features
        self.poly = PolynomialFeatures(degree=self.degree, interaction_only=True, include_bias=False)
        self.poly.fit(X_selected)
        
        return self
    
    def transform(self, X):
        """Transform using fitted components"""
        X_selected = self.selector.transform(X)
        X_poly = self.poly.transform(X_selected)
        
        # Remove the original features (keep only interactions)
        X_interactions = X_poly[:, self.n_top_features:]
        
        return X_interactions
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class DeepAutoencoderReducer:
    """Neural network based dimensionality reduction (if TensorFlow available)"""
    
    def __init__(self, encoding_dim=50, hidden_layers=[256, 128]):
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.encoder = None
        self.autoencoder = None
        self.input_dim = None
        
    def build_model(self, input_dim):
        """Build the autoencoder architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        self.input_dim = input_dim
        
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        x = encoder_input
        
        # Add noise for denoising autoencoder
        x = layers.GaussianNoise(0.1)(x)
        
        for units in self.hidden_layers:
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.1)(x)
            x = layers.Dropout(0.2)(x)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='linear',
                              kernel_regularizer=regularizers.l1(1e-5),
                              name='encoded')(x)
        
        # Decoder
        x = encoded
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.1)(x)
        
        decoded = layers.Dense(input_dim, activation='linear')(x)
        
        # Create models
        self.autoencoder = Model(encoder_input, decoded)
        self.encoder = Model(encoder_input, encoded)
        
        # Compile
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return self
    
    def fit(self, X, epochs=50, batch_size=256, validation_split=0.1):
        """Train the autoencoder"""
        if not TENSORFLOW_AVAILABLE:
            print("Skipping deep autoencoder - TensorFlow not available")
            return self
        
        if self.encoder is None:
            self.build_model(X.shape[1])
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.autoencoder.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        print(f"Autoencoder trained. Final val_loss: {history.history['val_loss'][-1]:.4f}")
        
        return self
    
    def transform(self, X):
        """Transform using the encoder"""
        if not TENSORFLOW_AVAILABLE or self.encoder is None:
            # Return truncated version if TensorFlow not available
            return X[:, :self.encoding_dim]
        
        return self.encoder.predict(X, verbose=0)
    
    def fit_transform(self, X, **kwargs):
        self.fit(X, **kwargs)
        return self.transform(X)


class AdvancedNonLinearReducer:
    """Main class that combines all non-linear transformation methods"""
    
    def __init__(self, important_features, target_dims=100, use_deep_learning=False):
        self.important_features = important_features
        self.target_dims = target_dims
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        
        # Initialize components
        self.rf_transformer = RandomForestFeatureTransformer(n_estimators=50, max_leaf_nodes=32)
        self.gb_engineer = GradientBoostingFeatureEngineering(n_rounds=3, trees_per_round=30)
        self.poly_extractor = PolynomialInteractionExtractor(n_top_features=15, degree=2)
        
        if self.use_deep_learning:
            self.deep_reducer = DeepAutoencoderReducer(encoding_dim=30)
        
        self.feature_selector = None
        self.important_scaler = None
        self.other_scaler = None
        self.important_indices = None
        self.other_indices = None
        
    def fit(self, X, feature_names, y):
        """Fit all transformation components"""
        print("\nAdvanced Non-Linear Feature Engineering")
        print("=" * 50)
        
        # Separate important and other features
        self.important_indices = [i for i, name in enumerate(feature_names) 
                                 if name in self.important_features]
        self.other_indices = [i for i in range(len(feature_names)) 
                             if i not in self.important_indices]
        
        print(f"Preserving {len(self.important_indices)} important features")
        print(f"Transforming {len(self.other_indices)} other features")
        
        if len(self.other_indices) == 0:
            print("Warning: All features marked as important. No transformation needed.")
            return self
        
        X_important = X[:, self.important_indices]
        X_other = X[:, self.other_indices]
        
        # Scale features
        self.important_scaler = RobustScaler()
        X_important_scaled = self.important_scaler.fit_transform(X_important)
        
        self.other_scaler = RobustScaler()
        X_other_scaled = self.other_scaler.fit_transform(X_other)
        
        # Apply transformations
        print("\n1. Random Forest transformation...")
        rf_features = self.rf_transformer.fit_transform(X_other_scaled, y)
        
        print("\n2. Gradient Boosting feature engineering...")
        gb_features = self.gb_engineer.fit_transform(X_other_scaled, y)
        
        print("\n3. Polynomial interaction extraction...")
        poly_features = self.poly_extractor.fit_transform(X_other_scaled, y)
        
        # Combine all engineered features
        all_engineered = [rf_features, gb_features, poly_features]
        
        if self.use_deep_learning:
            print("\n4. Deep autoencoder transformation...")
            deep_features = self.deep_reducer.fit_transform(X_other_scaled, epochs=30)
            all_engineered.append(deep_features)
        
        X_all_engineered = np.hstack(all_engineered)
        
        print(f"\nTotal engineered features: {X_all_engineered.shape[1]}")
        
        # Select best engineered features to reach target dimensions
        n_select = min(self.target_dims, X_all_engineered.shape[1])
        print(f"Selecting top {n_select} engineered features...")
        
        self.feature_selector = SelectKBest(f_regression, k=n_select)
        self.feature_selector.fit(X_all_engineered, y)
        
        return self
    
    def transform(self, X):
        """Transform new data using fitted components"""
        if len(self.other_indices) == 0:
            # All features are important, just scale
            return self.important_scaler.transform(X)
        
        # Separate features
        X_important = X[:, self.important_indices]
        X_other = X[:, self.other_indices]
        
        # Scale
        X_important_scaled = self.important_scaler.transform(X_important)
        X_other_scaled = self.other_scaler.transform(X_other)
        
        # Apply transformations
        rf_features = self.rf_transformer.transform(X_other_scaled)
        gb_features = self.gb_engineer.transform(X_other_scaled)
        poly_features = self.poly_extractor.transform(X_other_scaled)
        
        all_engineered = [rf_features, gb_features, poly_features]
        
        if self.use_deep_learning and hasattr(self, 'deep_reducer'):
            deep_features = self.deep_reducer.transform(X_other_scaled)
            all_engineered.append(deep_features)
        
        X_all_engineered = np.hstack(all_engineered)
        
        # Select features
        X_selected = self.feature_selector.transform(X_all_engineered)
        
        # Combine with important features
        X_final = np.hstack([X_important_scaled, X_selected])
        
        return X_final
    
    def fit_transform(self, X, feature_names, y):
        self.fit(X, feature_names, y)
        return self.transform(X)
    
    def get_feature_info(self):
        """Get information about the transformation"""
        n_engineered = self.feature_selector.k if self.feature_selector else 0
        
        info = {
            'n_important': len(self.important_indices),
            'n_other': len(self.other_indices),
            'n_engineered': n_engineered,
            'total_output': len(self.important_indices) + n_engineered,
            'rf_features': 200,  # Hashed features
            'gb_features': self.gb_engineer.n_rounds * (1 + 30),  # predictions + hashed
            'poly_features': self.poly_extractor.poly.n_output_features_ - self.poly_extractor.n_top_features if hasattr(self.poly_extractor, 'poly') and self.poly_extractor.poly else 0
        }
        
        if self.use_deep_learning:
            info['deep_features'] = self.deep_reducer.encoding_dim
        
        return info


class AdvancedCryptoPreprocessor:
    """Enhanced preprocessor for crypto regression task"""
    
    def __init__(self):
        self.scaler = None
        self.variance_selector = None
        self.outlier_percentiles = {}
        self.feature_augmenter = FeatureAugmenter()
        self.nan_fill_values = {}
    
    def fit_transform(self, X, feature_names, y=None):
        """Fit and transform for regression task"""
        
        # Convert to DataFrame for feature augmentation
        df = pd.DataFrame(X, columns=feature_names)
        
        # Apply feature augmentation
        df_augmented = self.feature_augmenter.fit_transform(df)
        
        # Extract augmented arrays and names
        feature_names_augmented = list(df_augmented.columns)
        X_augmented = df_augmented.values
        
        print(f"Created {len(feature_names_augmented) - len(feature_names)} augmented features")
        
        # Store median values for each feature
        for i, feat_name in enumerate(feature_names_augmented):
            finite_vals = X_augmented[:, i][np.isfinite(X_augmented[:, i])]
            if len(finite_vals) > 0:
                self.nan_fill_values[i] = np.median(finite_vals)
            else:
                self.nan_fill_values[i] = 0.0
        
        # Handle infinite and NaN values
        for i in range(X_augmented.shape[1]):
            col = X_augmented[:, i]
            
            finite_vals = col[np.isfinite(col)]
            if len(finite_vals) > 0:
                max_val = np.percentile(finite_vals, 99.9)
                min_val = np.percentile(finite_vals, 0.1)
                col = np.where(np.isposinf(col), max_val * 10, col)
                col = np.where(np.isneginf(col), min_val * 10, col)
            else:
                col = np.where(np.isposinf(col), 1e6, col)
                col = np.where(np.isneginf(col), -1e6, col)
            
            col = np.where(np.isnan(col), self.nan_fill_values[i], col)
            X_augmented[:, i] = col
        
        # Remove zero variance features
        self.variance_selector = VarianceThreshold(threshold=1e-8)
        X_var_filtered = self.variance_selector.fit_transform(X_augmented)
        feature_mask = self.variance_selector.get_support()
        feature_names_filtered = [f for f, m in zip(feature_names_augmented, feature_mask) if m]
        
        print(f"Removed {len(feature_names_augmented) - len(feature_names_filtered)} zero-variance features")
        
        # Robust scaling with outlier clipping
        self.outlier_percentiles = {}
        X_clipped = X_var_filtered.copy()
        
        for i in range(X_clipped.shape[1]):
            col = X_clipped[:, i]
            p01 = np.percentile(col, 1)
            p99 = np.percentile(col, 99)
            self.outlier_percentiles[i] = (p01, p99)
            X_clipped[:, i] = np.clip(col, p01, p99)
        
        # Scale
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_clipped)
        
        # Final check
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return X_scaled, feature_names_filtered
    
    def transform(self, X, feature_names):
        """Transform new data"""
        df = pd.DataFrame(X, columns=feature_names)
        df_augmented = self.feature_augmenter.transform(df)
        X_augmented = df_augmented.values
        
        # Handle infinite and NaN values
        for i in range(X_augmented.shape[1]):
            col = X_augmented[:, i]
            
            if i in self.outlier_percentiles:
                p01, p99 = self.outlier_percentiles[i]
                col = np.where(np.isposinf(col), p99 * 10, col)
                col = np.where(np.isneginf(col), p01 * 10, col)
            else:
                col = np.where(np.isposinf(col), 1e6, col)
                col = np.where(np.isneginf(col), -1e6, col)
            
            fill_value = self.nan_fill_values.get(i, 0.0)
            col = np.where(np.isnan(col), fill_value, col)
            X_augmented[:, i] = col
        
        X_var_filtered = self.variance_selector.transform(X_augmented)
        
        X_clipped = X_var_filtered.copy()
        for i, (p01, p99) in self.outlier_percentiles.items():
            if i < X_clipped.shape[1]:
                X_clipped[:, i] = np.clip(X_clipped[:, i], p01, p99)
        
        X_scaled = self.scaler.transform(X_clipped)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return X_scaled


def evaluate_model(X_train, X_test, y_train, y_test, model_name, model):
    """Evaluate a single model and return metrics"""
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if np.any(~np.isfinite(y_pred)):
            y_pred = np.nan_to_num(y_pred, nan=np.mean(y_train))
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        pearson_corr, _ = pearsonr(y_test, y_pred)
        
        return {
            'model': model_name,
            'mse': mse,
            'r2': r2,
            'pearson': pearson_corr
        }
    except Exception as e:
        print(f"Model {model_name} failed: {e}")
        return {
            'model': model_name,
            'mse': np.inf,
            'r2': -np.inf,
            'pearson': -1
        }


def main():
    """Main execution function"""
    
    print("DRW Crypto Competition - Advanced Non-Linear Feature Engineering")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    # Determine file path
    if os.path.exists('/kaggle/input/drw-crypto-market-prediction/train.parquet'):
        file_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
    elif os.path.exists('train.parquet'):
        file_path = 'train.parquet'
    else:
        print("\nERROR: train.parquet not found!")
        print("Please ensure the data file is in the current directory or Kaggle input.")
        return None
    
    # Load data
    print(f"\nLoading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} total samples")
    
    # Sample for faster experimentation (use all data for final submission)
    n_samples = min(35000, len(df))
    if len(df) > n_samples:
        df_subset = df.sample(n=n_samples, random_state=42)
    else:
        df_subset = df.copy()
    
    print(f"Using {len(df_subset)} samples for analysis")
    
    # Get feature columns
    feature_cols = [col for col in df_subset.columns if col not in ['timestamp', 'label']]
    X = df_subset[feature_cols].values
    y = df_subset['label'].values
    
    print(f"\nOriginal dataset shape: {X.shape}")
    
    # Preprocess
    print("\nPreprocessing data and creating augmented features...")
    preprocessor = AdvancedCryptoPreprocessor()
    X_processed, feature_cols_filtered = preprocessor.fit_transform(X, feature_cols, y)
    
    print(f"After augmentation and preprocessing: {X_processed.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, shuffle=True
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Test different approaches
    print("\n" + "=" * 60)
    print("TESTING NON-LINEAR TRANSFORMATION APPROACHES")
    print("=" * 60)
    
    results = []
    
    # 1. Baseline: No reduction
    print("\n1. Baseline (no reduction)...")
    start_time = time.time()
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.01, random_state=42, max_iter=2000),
        'Ridge': Ridge(alpha=1.0, random_state=42)
    }
    
    baseline_results = []
    for model_name, model in models.items():
        result = evaluate_model(X_train, X_test, y_train, y_test, model_name, model)
        baseline_results.append(result)
    
    best_baseline = max(baseline_results, key=lambda x: x['pearson'])
    print(f"Best baseline: {best_baseline['model']} - Pearson: {best_baseline['pearson']:.4f}")
    print(f"Time: {time.time() - start_time:.2f}s")
    
    results.append({
        'approach': 'Baseline',
        'dimensions': X_train.shape[1],
        'best_pearson': best_baseline['pearson'],
        'best_model': best_baseline['model'],
        'time': time.time() - start_time
    })
    
    # 2. Advanced Non-Linear Reduction
    for target_dims in [50, 100, 150]:
        print(f"\n2. Advanced Non-Linear Reduction (target_dims={target_dims})...")
        start_time = time.time()
        
        reducer = AdvancedNonLinearReducer(
            important_features=IMPORTANT_FEATURES,
            target_dims=target_dims,
            use_deep_learning=TENSORFLOW_AVAILABLE
        )
        
        X_train_reduced = reducer.fit_transform(X_train, feature_cols_filtered, y_train)
        X_test_reduced = reducer.transform(X_test)
        
        # Get info
        info = reducer.get_feature_info()
        print(f"Output dimensions: {X_train_reduced.shape[1]}")
        print(f"  Important features: {info['n_important']}")
        print(f"  Engineered features: {info['n_engineered']}")
        
        # Evaluate
        approach_results = []
        for model_name, model in models.items():
            result = evaluate_model(X_train_reduced, X_test_reduced, y_train, y_test, model_name, model)
            approach_results.append(result)
        
        best_result = max(approach_results, key=lambda x: x['pearson'])
        print(f"Best model: {best_result['model']} - Pearson: {best_result['pearson']:.4f}")
        print(f"Time: {time.time() - start_time:.2f}s")
        
        results.append({
            'approach': f'NonLinear-{target_dims}',
            'dimensions': X_train_reduced.shape[1],
            'best_pearson': best_result['pearson'],
            'best_model': best_result['model'],
            'time': time.time() - start_time
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_pearson', ascending=False)
    
    print("\nPerformance Summary:")
    print(results_df.to_string(index=False))
    
    # Best approach
    best_approach = results_df.iloc[0]
    
    print(f"\n✅ BEST APPROACH: {best_approach['approach']}")
    print(f"   Pearson Correlation: {best_approach['best_pearson']:.4f}")
    print(f"   Dimensions: {best_approach['dimensions']}")
    print(f"   Best Model: {best_approach['best_model']}")
    
    # Save results
    results_df.to_csv('nonlinear_reduction_results.csv', index=False)
    print("\nResults saved to 'nonlinear_reduction_results.csv'")
    
    # Save the best transformer for production use
    if 'NonLinear' in best_approach['approach']:
        print("\nSaving best transformer for production use...")
        
        # Retrain on full training data
        best_dims = int(best_approach['approach'].split('-')[1])
        final_reducer = AdvancedNonLinearReducer(
            important_features=IMPORTANT_FEATURES,
            target_dims=best_dims,
            use_deep_learning=TENSORFLOW_AVAILABLE
        )
        
        # Use all available data for final training
        X_full = np.vstack([X_train, X_test])
        y_full = np.hstack([y_train, y_test])
        
        final_reducer.fit(X_full, feature_cols_filtered, y_full)
        
        # Save components
        joblib.dump(preprocessor, 'preprocessor.pkl')
        joblib.dump(final_reducer, 'nonlinear_reducer.pkl')
        joblib.dump(feature_cols, 'original_feature_names.pkl')
        
        print("Saved: preprocessor.pkl, nonlinear_reducer.pkl, original_feature_names.pkl")
    
    print(f"\nCompleted at: {datetime.now()}")
    
    return results_df


if __name__ == "__main__":
    results = main()

