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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# CPU-only libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import xgboost as xgb
import lightgbm as lgb

print("Using CPU-only implementation (pandas + sklearn)")

class PersonalityPredictor:
    def __init__(self):
        self.target_encoders = {}
        self.feature_combinations = {}
        self.original_df = None
        
    def load_data(self):
        """Load competition data and explore structure"""
        print("Loading Personality Prediction data...")
        
        # Load competition data
        self.train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
        self.test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
        self.sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        print(f"Sample submission shape: {self.sample_submission.shape}")
        
        # Explore the data
        print("\nTrain columns:")
        print(self.train_df.columns.tolist())
        
        print("\nTarget distribution:")
        print(self.train_df['Personality'].value_counts())
        print("Target proportions:")
        print(self.train_df['Personality'].value_counts(normalize=True))
        
        print("\nFirst few rows of training data:")
        print(self.train_df.head())
        
        print("\nData types:")
        print(self.train_df.dtypes)
        
        print("\nMissing values in train:")
        print(self.train_df.isnull().sum().sum())
        print("Missing values in test:")
        print(self.test_df.isnull().sum().sum())
        
        # Identify feature columns (excluding id and target)
        self.feature_cols = [col for col in self.train_df.columns 
                           if col not in ['id', 'Personality']]
        print(f"\nFeature columns ({len(self.feature_cols)}):")
        print(self.feature_cols)
        
        # Basic statistics
        print("\nBasic statistics:")
        print(self.train_df[self.feature_cols].describe())
        
        return self
    
    def explore_features(self):
        """Explore feature distributions and relationships"""
        print("\nExploring feature characteristics...")
        
        # Check if features are numerical or categorical
        numerical_features = []
        categorical_features = []
        
        for col in self.feature_cols:
            unique_vals = self.train_df[col].nunique()
            dtype = self.train_df[col].dtype
            
            print(f"{col}: {unique_vals} unique values, dtype: {dtype}")
            
            # Following the winning approach - treat features with reasonable number of unique values as categorical
            if unique_vals <= 50 or dtype == 'object':  
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # For target encoding, we can discretize numerical features to make them categorical
        # This follows the winning approach of treating all features as categorical
        self.all_features_for_combinations = self.feature_cols.copy()
        
        return self
    
    def create_feature_combinations(self):
        """Create pairs, triples, and quadruples of features like the winning solution"""
        print("\nCreating feature combinations for target encoding...")
        
        n_features = len(self.all_features_for_combinations)
        print(f"Starting with {n_features} features")
        
        # Create pairs
        pairs = list(combinations(self.all_features_for_combinations, 2))
        print(f"Created {len(pairs)} pairs")
        
        # Create triples  
        triples = list(combinations(self.all_features_for_combinations, 3))
        print(f"Created {len(triples)} triples")
        
        # Create quadruples
        quadruples = list(combinations(self.all_features_for_combinations, 4))
        print(f"Created {len(quadruples)} quadruples")
        
        self.feature_combinations = {
            'singles': [(col,) for col in self.all_features_for_combinations],
            'pairs': pairs,
            'triples': triples,
            'quadruples': quadruples
        }
        
        total_combinations = (len(self.all_features_for_combinations) + len(pairs) + 
                            len(triples) + len(quadruples))
        print(f"Total feature combinations: {total_combinations}")
        
        return self
    
    def discretize_numerical_features(self, df):
        """Convert numerical features to categorical for combination creation"""
        result_df = df.copy()
        
        for col in self.numerical_features:
            if col in result_df.columns:
                # Use quantile-based binning
                try:
                    result_df[col] = pd.qcut(result_df[col], q=10, duplicates='drop').astype(str)
                except:
                    # If qcut fails, use regular cut
                    result_df[col] = pd.cut(result_df[col], bins=10, duplicates='drop').astype(str)
        
        return result_df
    
    def create_combination_columns(self, df):
        """Create actual combination columns from features"""
        # First discretize numerical features
        result_df = self.discretize_numerical_features(df)
        
        print("Creating combination columns...")
        
        # Create pair columns
        for pair in self.feature_combinations['pairs']:
            col_name = f"pair_{pair[0]}_{pair[1]}"
            result_df[col_name] = (result_df[pair[0]].astype(str) + "_" + 
                                 result_df[pair[1]].astype(str))
        
        # Create triple columns
        for triple in self.feature_combinations['triples']:
            col_name = f"triple_{triple[0]}_{triple[1]}_{triple[2]}"
            result_df[col_name] = (result_df[triple[0]].astype(str) + "_" + 
                                 result_df[triple[1]].astype(str) + "_" + 
                                 result_df[triple[2]].astype(str))
        
        # Create quadruple columns  
        for quad in self.feature_combinations['quadruples']:
            col_name = f"quad_{quad[0]}_{quad[1]}_{quad[2]}_{quad[3]}"
            result_df[col_name] = (result_df[quad[0]].astype(str) + "_" + 
                                 result_df[quad[1]].astype(str) + "_" + 
                                 result_df[quad[2]].astype(str) + "_" + 
                                 result_df[quad[3]].astype(str))
        
        print(f"Created dataframe with {result_df.shape[1]} total columns")
        return result_df

# Initialize and run first step
if __name__ == "__main__":
    predictor = PersonalityPredictor()
    
    # Load and explore data
    predictor.load_data()
    predictor.explore_features()
    predictor.create_feature_combinations()
    
    # Test combination creation on a small sample
    print("\nTesting combination creation on first 100 rows...")
    sample_train = predictor.train_df.head(100)
    train_with_combos = predictor.create_combination_columns(sample_train)
    
    print(f"Original columns: {len(predictor.train_df.columns)}")
    print(f"Columns after combinations: {len(train_with_combos.columns)}")
    
    # Show some example combination columns
    combo_cols = [col for col in train_with_combos.columns 
                  if col.startswith(('pair_', 'triple_', 'quad_'))]
    print(f"\nExample combination columns (first 5):")
    for col in combo_cols[:5]:
        print(f"  {col}")
    
    print(f"\nSample values from first combination column:")
    if len(combo_cols) > 0:
        print(train_with_combos[combo_cols[0]].head())
    
    print("\n" + "="*50)
    print("Step 1 complete! Data loaded and feature combinations prepared.")
    print("Ready for target encoding step.")
    print("="*50)



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class TargetEncoder:
    def __init__(self, smoothing=1.0, min_samples_leaf=1, noise_level=0.01):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.target_encodings = {}
        self.global_mean = None
        
    def fit_transform(self, X, y, categorical_cols):
        """Fit target encoder and transform training data with CV to avoid overfitting"""
        X_encoded = X.copy()
        
        # Convert target to binary (1 for Extrovert, 0 for Introvert)
        y_binary = (y == 'Extrovert').astype(int)
        self.global_mean = y_binary.mean()
        
        # Use StratifiedKFold to create out-of-fold target encodings
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col not in X.columns:
                continue
                
            print(f"Target encoding {col}...")
            
            # Initialize column with global mean
            X_encoded[f'te_{col}'] = self.global_mean
            
            # Create out-of-fold encodings
            for train_idx, val_idx in skf.split(X, y_binary):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y_binary.iloc[train_idx]
                
                # Calculate target encoding for this fold
                te_map = self._calculate_target_encoding(X_train_fold[col], y_train_fold)
                
                # Apply to validation fold
                X_encoded.loc[val_idx, f'te_{col}'] = X.iloc[val_idx][col].map(te_map).fillna(self.global_mean)
            
            # Fit on full data for test set transformation
            self.target_encodings[col] = self._calculate_target_encoding(X[col], y_binary)
        
        return X_encoded
    
    def transform(self, X, categorical_cols):
        """Transform test data using fitted encodings"""
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col not in X.columns or col not in self.target_encodings:
                continue
                
            X_encoded[f'te_{col}'] = X[col].map(self.target_encodings[col]).fillna(self.global_mean)
        
        return X_encoded
    
    def _calculate_target_encoding(self, categorical_series, target_series):
        """Calculate smoothed target encoding"""
        # Group by categorical values and calculate statistics
        stats = pd.DataFrame({
            'sum': target_series.groupby(categorical_series).sum(),
            'count': target_series.groupby(categorical_series).count()
        })
        
        # Apply smoothing
        stats['te'] = (stats['sum'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
        
        return stats['te'].to_dict()

class PersonalityPredictorStep2:
    def __init__(self):
        self.target_encoders = {}
        self.feature_combinations = {}
        self.le = LabelEncoder()
        
    def load_and_prepare_data(self):
        """Load data and create feature combinations"""
        print("Loading data...")
        self.train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
        self.test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
        
        # Handle missing values
        print("Handling missing values...")
        self.train_df = self.handle_missing_values(self.train_df)
        self.test_df = self.handle_missing_values(self.test_df)
        
        self.feature_cols = [col for col in self.train_df.columns 
                           if col not in ['id', 'Personality']]
        
        print(f"Features: {self.feature_cols}")
        
        # Create feature combinations
        self.create_feature_combinations()
        
        return self
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df_filled = df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().sum() > 0:
                if df_filled[col].dtype == 'object':
                    # For categorical, use mode or 'Unknown'
                    mode_val = df_filled[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    df_filled[col] = df_filled[col].fillna(fill_val)
                else:
                    # For numerical, use median
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        return df_filled
    
    def create_feature_combinations(self):
        """Create feature combinations like the winning solution"""
        from itertools import combinations
        
        print("Creating feature combinations...")
        
        # Create pairs, triples, quadruples
        pairs = list(combinations(self.feature_cols, 2))
        triples = list(combinations(self.feature_cols, 3))
        quadruples = list(combinations(self.feature_cols, 4))
        
        self.feature_combinations = {
            'singles': [(col,) for col in self.feature_cols],
            'pairs': pairs,
            'triples': triples,
            'quadruples': quadruples
        }
        
        print(f"Singles: {len(self.feature_combinations['singles'])}")
        print(f"Pairs: {len(pairs)}")
        print(f"Triples: {len(triples)}")
        print(f"Quadruples: {len(quadruples)}")
        
    def create_combination_features(self, df):
        """Create combination columns"""
        result_df = df.copy()
        
        # Convert numerical to categorical for combinations
        for col in self.feature_cols:
            if result_df[col].dtype in ['float64', 'int64']:
                # Use quantile-based binning
                try:
                    result_df[col] = pd.qcut(result_df[col], q=10, duplicates='drop').astype(str)
                except:
                    result_df[col] = pd.cut(result_df[col], bins=10, duplicates='drop').astype(str)
        
        # Create pair features
        for pair in self.feature_combinations['pairs']:
            col_name = f"pair_{pair[0]}_{pair[1]}"
            result_df[col_name] = (result_df[pair[0]].astype(str) + "_" + 
                                 result_df[pair[1]].astype(str))
        
        # Create triple features
        for triple in self.feature_combinations['triples']:
            col_name = f"triple_{triple[0]}_{triple[1]}_{triple[2]}"
            result_df[col_name] = (result_df[triple[0]].astype(str) + "_" + 
                                 result_df[triple[1]].astype(str) + "_" + 
                                 result_df[triple[2]].astype(str))
        
        # Create quadruple features
        for quad in self.feature_combinations['quadruples']:
            col_name = f"quad_{quad[0]}_{quad[1]}_{quad[2]}_{quad[3]}"
            result_df[col_name] = (result_df[quad[0]].astype(str) + "_" + 
                                 result_df[quad[1]].astype(str) + "_" + 
                                 result_df[quad[2]].astype(str) + "_" + 
                                 result_df[quad[3]].astype(str))
        
        return result_df
    
    def apply_target_encoding(self):
        """Apply target encoding to all features and combinations"""
        print("Applying target encoding...")
        
        # Create combination features
        print("Creating combination features for train...")
        train_with_combos = self.create_combination_features(self.train_df)
        
        print("Creating combination features for test...")
        test_with_combos = self.create_combination_features(self.test_df)
        
        # Get all categorical columns (original + combinations)
        categorical_cols = [col for col in train_with_combos.columns 
                          if col not in ['id', 'Personality']]
        
        print(f"Total categorical columns for target encoding: {len(categorical_cols)}")
        
        # Apply target encoding
        target_encoder = TargetEncoder(smoothing=1.0, noise_level=0.01)
        
        # Fit and transform training data
        X_train = train_with_combos[categorical_cols]
        y_train = train_with_combos['Personality']
        
        print("Fitting target encoder on training data...")
        train_encoded = target_encoder.fit_transform(X_train, y_train, categorical_cols)
        
        # Transform test data
        print("Transforming test data...")
        X_test = test_with_combos[categorical_cols]
        test_encoded = target_encoder.transform(X_test, categorical_cols)
        
        # Keep only target encoded features
        te_cols = [col for col in train_encoded.columns if col.startswith('te_')]
        
        print(f"Created {len(te_cols)} target encoded features")
        
        # Prepare final datasets
        self.X_train_encoded = train_encoded[te_cols]
        self.X_test_encoded = test_encoded[te_cols]
        self.y_train = (y_train == 'Extrovert').astype(int)  # Binary encoding
        self.test_ids = self.test_df['id']
        
        print(f"Final training shape: {self.X_train_encoded.shape}")
        print(f"Final test shape: {self.X_test_encoded.shape}")
        
        return self

# Run Step 2
if __name__ == "__main__":
    print("="*60)
    print("STEP 2: TARGET ENCODING")
    print("="*60)
    
    predictor = PersonalityPredictorStep2()
    predictor.load_and_prepare_data()
    predictor.apply_target_encoding()
    
    print("\nTarget encoding summary:")
    print(f"Training features shape: {predictor.X_train_encoded.shape}")
    print(f"Test features shape: {predictor.X_test_encoded.shape}")
    print(f"Target distribution: {predictor.y_train.value_counts()}")
    
    print("\nSample of target encoded features:")
    print(predictor.X_train_encoded.head())
    
    print("\n" + "="*60)
    print("Step 2 complete! Ready for model training.")
    print("="*60)



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class TargetEncoder:
    def __init__(self, smoothing=1.0, min_samples_leaf=1, noise_level=0.01):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.target_encodings = {}
        self.global_mean = None
        
    def fit_transform(self, X, y, categorical_cols):
        """Fit target encoder and transform training data with CV to avoid overfitting"""
        X_encoded = X.copy()
        
        # Convert target to binary (1 for Extrovert, 0 for Introvert)
        y_binary = (y == 'Extrovert').astype(int)
        self.global_mean = y_binary.mean()
        
        # Use StratifiedKFold to create out-of-fold target encodings
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col not in X.columns:
                continue
                
            # Initialize column with global mean
            X_encoded[f'te_{col}'] = self.global_mean
            
            # Create out-of-fold encodings
            for train_idx, val_idx in skf.split(X, y_binary):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y_binary.iloc[train_idx]
                
                # Calculate target encoding for this fold
                te_map = self._calculate_target_encoding(X_train_fold[col], y_train_fold)
                
                # Apply to validation fold
                X_encoded.loc[val_idx, f'te_{col}'] = X.iloc[val_idx][col].map(te_map).fillna(self.global_mean)
            
            # Fit on full data for test set transformation
            self.target_encodings[col] = self._calculate_target_encoding(X[col], y_binary)
        
        return X_encoded
    
    def transform(self, X, categorical_cols):
        """Transform test data using fitted encodings"""
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col not in X.columns or col not in self.target_encodings:
                continue
                
            X_encoded[f'te_{col}'] = X[col].map(self.target_encodings[col]).fillna(self.global_mean)
        
        return X_encoded
    
    def _calculate_target_encoding(self, categorical_series, target_series):
        """Calculate smoothed target encoding"""
        # Group by categorical values and calculate statistics
        stats = pd.DataFrame({
            'sum': target_series.groupby(categorical_series).sum(),
            'count': target_series.groupby(categorical_series).count()
        })
        
        # Apply smoothing
        stats['te'] = (stats['sum'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
        
        return stats['te'].to_dict()

class PersonalityEnsemble:
    def __init__(self):
        self.models = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        self.feature_importance = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data with target encoding"""
        print("Loading and preparing data...")
        
        # Load raw data
        train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
        test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
        
        # Handle missing values
        train_df = self.handle_missing_values(train_df)
        test_df = self.handle_missing_values(test_df)
        
        feature_cols = [col for col in train_df.columns if col not in ['id', 'Personality']]
        
        # Create feature combinations
        train_with_combos = self.create_combination_features(train_df, feature_cols)
        test_with_combos = self.create_combination_features(test_df, feature_cols)
        
        # Get all categorical columns for target encoding
        categorical_cols = [col for col in train_with_combos.columns 
                          if col not in ['id', 'Personality']]
        
        print(f"Total features for target encoding: {len(categorical_cols)}")
        
        # Apply target encoding
        target_encoder = TargetEncoder(smoothing=1.0, noise_level=0.01)
        
        X_train = train_with_combos[categorical_cols]
        y_train = train_with_combos['Personality']
        
        print("Applying target encoding...")
        train_encoded = target_encoder.fit_transform(X_train, y_train, categorical_cols)
        
        X_test = test_with_combos[categorical_cols]
        test_encoded = target_encoder.transform(X_test, categorical_cols)
        
        # Keep only target encoded features
        te_cols = [col for col in train_encoded.columns if col.startswith('te_')]
        
        self.X_train = train_encoded[te_cols]
        self.X_test = test_encoded[te_cols]
        self.y_train = (y_train == 'Extrovert').astype(int)
        self.test_ids = test_df['id']
        
        print(f"Final training shape: {self.X_train.shape}")
        print(f"Final test shape: {self.X_test.shape}")
        print(f"Target distribution: {self.y_train.value_counts()}")
        
        return self
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        df_filled = df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().sum() > 0:
                if df_filled[col].dtype == 'object':
                    mode_val = df_filled[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    df_filled[col] = df_filled[col].fillna(fill_val)
                else:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        return df_filled
    
    def create_combination_features(self, df, feature_cols):
        """Create feature combinations"""
        result_df = df.copy()
        
        # Convert numerical to categorical for combinations
        for col in feature_cols:
            if result_df[col].dtype in ['float64', 'int64']:
                try:
                    result_df[col] = pd.qcut(result_df[col], q=10, duplicates='drop').astype(str)
                except:
                    result_df[col] = pd.cut(result_df[col], bins=10, duplicates='drop').astype(str)
        
        # Create combinations
        pairs = list(combinations(feature_cols, 2))
        triples = list(combinations(feature_cols, 3))
        quadruples = list(combinations(feature_cols, 4))
        
        print(f"Creating {len(pairs)} pairs, {len(triples)} triples, {len(quadruples)} quadruples...")
        
        # Create pair features
        for pair in pairs:
            col_name = f"pair_{pair[0]}_{pair[1]}"
            result_df[col_name] = (result_df[pair[0]].astype(str) + "_" + 
                                 result_df[pair[1]].astype(str))
        
        # Create triple features
        for triple in triples:
            col_name = f"triple_{triple[0]}_{triple[1]}_{triple[2]}"
            result_df[col_name] = (result_df[triple[0]].astype(str) + "_" + 
                                 result_df[triple[1]].astype(str) + "_" + 
                                 result_df[triple[2]].astype(str))
        
        # Create quadruple features
        for quad in quadruples:
            col_name = f"quad_{quad[0]}_{quad[1]}_{quad[2]}_{quad[3]}"
            result_df[col_name] = (result_df[quad[0]].astype(str) + "_" + 
                                 result_df[quad[1]].astype(str) + "_" + 
                                 result_df[quad[2]].astype(str) + "_" + 
                                 result_df[quad[3]].astype(str))
        
        return result_df
    
    def train_model_cv(self, model_name, model_config, model_type='xgb'):
        """Train a model using cross-validation"""
        print(f"Training {model_name}...")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        oof_preds = np.zeros(len(self.X_train))
        test_preds = np.zeros(len(self.X_test))
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            y_fold_val = self.y_train.iloc[val_idx]
            
            if model_type == 'xgb':
                model = xgb.XGBClassifier(**model_config)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            elif model_type == 'lgb':
                model = lgb.LGBMClassifier(**model_config)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            else:  # sklearn models
                model = model_config
                model.fit(X_fold_train, y_fold_train)
            
            # Predict validation set
            val_preds = model.predict_proba(X_fold_val)[:, 1]
            oof_preds[val_idx] = val_preds
            
            # Predict test set
            test_preds += model.predict_proba(self.X_test)[:, 1] / 5
            
            # Calculate fold score
            fold_accuracy = accuracy_score(y_fold_val, (val_preds > 0.5).astype(int))
            fold_scores.append(fold_accuracy)
            print(f"    Fold {fold + 1} Accuracy: {fold_accuracy:.6f}")
        
        # Overall CV score
        cv_accuracy = accuracy_score(self.y_train, (oof_preds > 0.5).astype(int))
        cv_auc = roc_auc_score(self.y_train, oof_preds)
        
        print(f"  {model_name} CV Accuracy: {cv_accuracy:.6f}")
        print(f"  {model_name} CV AUC: {cv_auc:.6f}")
        
        # Store results
        self.oof_predictions[model_name] = oof_preds
        self.test_predictions[model_name] = test_preds
        
        return cv_accuracy, cv_auc
    
    def train_all_models(self):
        """Train all models"""
        print("="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        results = {}
        
        # XGBoost models
        xgb_configs = {
            'xgb_1': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'random_state': 42,
                'n_estimators': 1000
            },
            'xgb_2': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'min_child_weight': 3,
                'reg_alpha': 0.5,
                'reg_lambda': 2,
                'random_state': 123,
                'n_estimators': 1500
            }
        }
        
        for name, config in xgb_configs.items():
            acc, auc = self.train_model_cv(name, config, 'xgb')
            results[name] = {'accuracy': acc, 'auc': auc}
        
        # LightGBM models
        lgb_configs = {
            'lgb_1': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'n_estimators': 1000,
                'verbose': -1
            }
        }
        
        for name, config in lgb_configs.items():
            acc, auc = self.train_model_cv(name, config, 'lgb')
            results[name] = {'accuracy': acc, 'auc': auc}
        
        # Random Forest
        rf_config = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        acc, auc = self.train_model_cv('rf_1', rf_config, 'sklearn')
        results['rf_1'] = {'accuracy': acc, 'auc': auc}
        
        # Logistic Regression
        lr_config = LogisticRegression(
            C=1.0,
            random_state=42,
            max_iter=1000,
            solver='liblinear'
        )
        
        acc, auc = self.train_model_cv('lr_1', lr_config, 'sklearn')
        results['lr_1'] = {'accuracy': acc, 'auc': auc}
        
        self.model_results = results
        return self
    
    def create_ensemble(self):
        """Create ensemble predictions"""
        print("\n" + "="*60)
        print("CREATING ENSEMBLE")
        print("="*60)
        
        # Simple average ensemble
        oof_ensemble = np.mean(list(self.oof_predictions.values()), axis=0)
        test_ensemble = np.mean(list(self.test_predictions.values()), axis=0)
        
        ensemble_accuracy = accuracy_score(self.y_train, (oof_ensemble > 0.5).astype(int))
        ensemble_auc = roc_auc_score(self.y_train, oof_ensemble)
        
        print(f"Simple Average Ensemble CV Accuracy: {ensemble_accuracy:.6f}")
        print(f"Simple Average Ensemble CV AUC: {ensemble_auc:.6f}")
        
        # Weighted ensemble (weight by CV performance)
        weights = []
        model_names = []
        for name, results in self.model_results.items():
            weights.append(results['accuracy'])
            model_names.append(name)
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        print(f"\nModel weights based on CV accuracy:")
        for name, weight in zip(model_names, weights):
            print(f"  {name}: {weight:.4f}")
        
        # Weighted ensemble
        oof_weighted = np.zeros(len(self.y_train))
        test_weighted = np.zeros(len(self.X_test))
        
        for name, weight in zip(model_names, weights):
            oof_weighted += self.oof_predictions[name] * weight
            test_weighted += self.test_predictions[name] * weight
        
        weighted_accuracy = accuracy_score(self.y_train, (oof_weighted > 0.5).astype(int))
        weighted_auc = roc_auc_score(self.y_train, oof_weighted)
        
        print(f"\nWeighted Ensemble CV Accuracy: {weighted_accuracy:.6f}")
        print(f"Weighted Ensemble CV AUC: {weighted_auc:.6f}")
        
        # Use the better performing ensemble
        if weighted_accuracy > ensemble_accuracy:
            print(f"\nUsing weighted ensemble (better performance)")
            self.final_oof = oof_weighted
            self.final_test = test_weighted
            self.final_accuracy = weighted_accuracy
            self.final_auc = weighted_auc
        else:
            print(f"\nUsing simple average ensemble (better performance)")
            self.final_oof = oof_ensemble
            self.final_test = test_ensemble
            self.final_accuracy = ensemble_accuracy
            self.final_auc = ensemble_auc
        
        return self
    
    def create_submission(self):
        """Create submission file"""
        print("\n" + "="*60)
        print("CREATING SUBMISSION")
        print("="*60)
        
        # Convert probabilities to class predictions
        final_predictions = (self.final_test > 0.5).astype(int)
        
        # Convert back to original labels
        final_labels = ['Extrovert' if pred == 1 else 'Introvert' for pred in final_predictions]
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': self.test_ids,
            'Personality': final_labels
        })
        
        print(f"Submission shape: {submission.shape}")
        print(f"Prediction distribution:")
        print(submission['Personality'].value_counts())
        print(f"Prediction proportions:")
        print(submission['Personality'].value_counts(normalize=True))
        
        # Save submission
        submission.to_csv('submission.csv', index=False)
        print(f"\nSubmission saved to 'submission.csv'")
        
        # Display first few rows
        print(f"\nFirst 10 predictions:")
        print(submission.head(10))
        
        return submission
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        print("Individual Model Performance:")
        for name, results in self.model_results.items():
            print(f"  {name:10s}: Accuracy={results['accuracy']:.6f}, AUC={results['auc']:.6f}")
        
        print(f"\nFinal Ensemble Performance:")
        print(f"  CV Accuracy: {self.final_accuracy:.6f}")
        print(f"  CV AUC: {self.final_auc:.6f}")
        
        print(f"\nData Summary:")
        print(f"  Training samples: {len(self.X_train):,}")
        print(f"  Test samples: {len(self.X_test):,}")
        print(f"  Features used: {self.X_train.shape[1]}")
        
        print(f"\nTarget Distribution (Training):")
        target_dist = pd.Series(self.y_train).value_counts()
        print(f"  Extrovert (1): {target_dist[1]:,} ({target_dist[1]/len(self.y_train):.1%})")
        print(f"  Introvert (0): {target_dist[0]:,} ({target_dist[0]/len(self.y_train):.1%})")

# Run the complete pipeline
if __name__ == "__main__":
    print("="*80)
    print("PERSONALITY PREDICTION - COMPLETE PIPELINE")
    print("="*80)
    
    # Initialize and run
    ensemble = PersonalityEnsemble()
    
    # Step 1: Load and prepare data
    ensemble.load_and_prepare_data()
    
    # Step 2: Train all models
    ensemble.train_all_models()
    
    # Step 3: Create ensemble
    ensemble.create_ensemble()
    
    # Step 4: Create submission
    submission = ensemble.create_submission()
    
    # Step 5: Print summary
    ensemble.print_summary()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from itertools import combinations
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class AdvancedTargetEncoder:
    def __init__(self, smoothing=1.0, noise_level=0.01):
        self.smoothing = smoothing
        self.noise_level = noise_level
        self.target_encodings = {}
        self.global_mean = None
        
    def fit_transform(self, X, y, categorical_cols):
        """Enhanced target encoding with multiple smoothing strategies"""
        X_encoded = X.copy()
        
        y_binary = (y == 'Extrovert').astype(int)
        self.global_mean = y_binary.mean()
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # More folds
        
        for col in categorical_cols:
            if col not in X.columns:
                continue
                
            # Multiple encoding strategies
            X_encoded[f'te_{col}'] = self.global_mean
            X_encoded[f'te_smooth_{col}'] = self.global_mean
            X_encoded[f'te_count_{col}'] = 0
            
            for train_idx, val_idx in skf.split(X, y_binary):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y_binary.iloc[train_idx]
                
                # Standard target encoding
                te_map = self._calculate_target_encoding(X_train_fold[col], y_train_fold, smoothing=1.0)
                X_encoded.loc[val_idx, f'te_{col}'] = X.iloc[val_idx][col].map(te_map).fillna(self.global_mean)
                
                # Heavy smoothing
                te_smooth_map = self._calculate_target_encoding(X_train_fold[col], y_train_fold, smoothing=10.0)
                X_encoded.loc[val_idx, f'te_smooth_{col}'] = X.iloc[val_idx][col].map(te_smooth_map).fillna(self.global_mean)
                
                # Count encoding
                count_map = X_train_fold[col].value_counts().to_dict()
                X_encoded.loc[val_idx, f'te_count_{col}'] = X.iloc[val_idx][col].map(count_map).fillna(0)
            
            # Fit on full data for test transformation
            self.target_encodings[f'te_{col}'] = self._calculate_target_encoding(X[col], y_binary, smoothing=1.0)
            self.target_encodings[f'te_smooth_{col}'] = self._calculate_target_encoding(X[col], y_binary, smoothing=10.0)
            self.target_encodings[f'te_count_{col}'] = X[col].value_counts().to_dict()
        
        return X_encoded
    
    def transform(self, X, categorical_cols):
        """Transform test data"""
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col not in X.columns:
                continue
                
            # Apply all encoding strategies
            X_encoded[f'te_{col}'] = X[col].map(self.target_encodings.get(f'te_{col}', {})).fillna(self.global_mean)
            X_encoded[f'te_smooth_{col}'] = X[col].map(self.target_encodings.get(f'te_smooth_{col}', {})).fillna(self.global_mean)
            X_encoded[f'te_count_{col}'] = X[col].map(self.target_encodings.get(f'te_count_{col}', {})).fillna(0)
        
        return X_encoded
    
    def _calculate_target_encoding(self, categorical_series, target_series, smoothing=1.0):
        """Calculate target encoding with specified smoothing"""
        stats = pd.DataFrame({
            'sum': target_series.groupby(categorical_series).sum(),
            'count': target_series.groupby(categorical_series).count()
        })
        
        stats['te'] = (stats['sum'] + smoothing * self.global_mean) / (stats['count'] + smoothing)
        return stats['te'].to_dict()

class EnhancedPersonalityEnsemble:
    def __init__(self):
        self.models = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        
    def load_and_prepare_data(self):
        """Enhanced data preparation"""
        print("Loading and preparing data...")
        
        train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
        test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
        
        # Enhanced missing value handling
        train_df = self.advanced_missing_value_handling(train_df)
        test_df = self.advanced_missing_value_handling(test_df)
        
        feature_cols = [col for col in train_df.columns if col not in ['id', 'Personality']]
        
        # Enhanced feature engineering
        train_with_features = self.create_advanced_features(train_df, feature_cols)
        test_with_features = self.create_advanced_features(test_df, feature_cols)
        
        # Get categorical columns
        categorical_cols = [col for col in train_with_features.columns 
                          if col not in ['id', 'Personality']]
        
        print(f"Total features for encoding: {len(categorical_cols)}")
        
        # Enhanced target encoding
        target_encoder = AdvancedTargetEncoder(smoothing=1.0, noise_level=0.01)
        
        X_train = train_with_features[categorical_cols]
        y_train = train_with_features['Personality']
        
        print("Applying enhanced target encoding...")
        train_encoded = target_encoder.fit_transform(X_train, y_train, categorical_cols)
        
        X_test = test_with_features[categorical_cols]
        test_encoded = target_encoder.transform(X_test, categorical_cols)
        
        # Get all encoded features
        encoded_cols = [col for col in train_encoded.columns if col.startswith('te_')]
        
        self.X_train = train_encoded[encoded_cols]
        self.X_test = test_encoded[encoded_cols]
        self.y_train = (y_train == 'Extrovert').astype(int)
        self.test_ids = test_df['id']
        
        print(f"Final training shape: {self.X_train.shape}")
        print(f"Final test shape: {self.X_test.shape}")
        
        return self
    
    def advanced_missing_value_handling(self, df):
        """Advanced missing value handling"""
        df_filled = df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().sum() > 0:
                if df_filled[col].dtype == 'object':
                    # Use mode for categorical
                    mode_val = df_filled[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Missing'
                    df_filled[col] = df_filled[col].fillna(fill_val)
                else:
                    # Use median for numerical
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        return df_filled
    
    def create_advanced_features(self, df, feature_cols):
        """Enhanced feature engineering"""
        result_df = df.copy()
        
        # Convert numerical to categorical with different strategies
        for col in feature_cols:
            if result_df[col].dtype in ['float64', 'int64']:
                # Multiple binning strategies
                try:
                    result_df[f'{col}_qcut'] = pd.qcut(result_df[col], q=5, duplicates='drop').astype(str)
                    result_df[f'{col}_cut'] = pd.cut(result_df[col], bins=5, duplicates='drop').astype(str)
                except:
                    result_df[f'{col}_qcut'] = result_df[col].astype(str)
                    result_df[f'{col}_cut'] = result_df[col].astype(str)
        
        # Get all categorical columns
        cat_cols = [col for col in result_df.columns if col not in ['id', 'Personality']]
        
        # Create combinations more strategically
        pairs = list(combinations(feature_cols, 2))
        triples = list(combinations(feature_cols, 3))
        
        print(f"Creating {len(pairs)} pairs and {len(triples)} triples...")
        
        # Pair combinations
        for pair in pairs:
            col_name = f"pair_{pair[0]}_{pair[1]}"
            result_df[col_name] = (result_df[pair[0]].astype(str) + "_" + 
                                 result_df[pair[1]].astype(str))
        
        # Triple combinations
        for triple in triples:
            col_name = f"triple_{triple[0]}_{triple[1]}_{triple[2]}"
            result_df[col_name] = (result_df[triple[0]].astype(str) + "_" + 
                                 result_df[triple[1]].astype(str) + "_" + 
                                 result_df[triple[2]].astype(str))
        
        return result_df
    
    def train_enhanced_models(self):
        """Train enhanced model ensemble"""
        print("="*60)
        print("TRAINING ENHANCED MODELS")
        print("="*60)
        
        results = {}
        
        # Enhanced XGBoost configurations
        xgb_configs = {
            'xgb_1': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'random_state': 42,
                'n_estimators': 2000
            },
            'xgb_2': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 4,
                'learning_rate': 0.03,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'min_child_weight': 3,
                'reg_alpha': 0.5,
                'reg_lambda': 2,
                'random_state': 123,
                'n_estimators': 3000
            },
            'xgb_3': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 8,
                'learning_rate': 0.02,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 5,
                'reg_alpha': 1.0,
                'reg_lambda': 3,
                'random_state': 456,
                'n_estimators': 4000
            }
        }
        
        for name, config in xgb_configs.items():
            acc, auc = self.train_model_cv(name, config, 'xgb')
            results[name] = {'accuracy': acc, 'auc': auc}
        
        # Enhanced LightGBM configurations
        lgb_configs = {
            'lgb_1': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'n_estimators': 2000,
                'verbose': -1
            },
            'lgb_2': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 15,
                'learning_rate': 0.03,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 3,
                'min_child_samples': 30,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'random_state': 789,
                'n_estimators': 3000,
                'verbose': -1
            }
        }
        
        for name, config in lgb_configs.items():
            acc, auc = self.train_model_cv(name, config, 'lgb')
            results[name] = {'accuracy': acc, 'auc': auc}
        
        # Enhanced Random Forest
        rf_configs = {
            'rf_1': RandomForestClassifier(
                n_estimators=1000,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'rf_2': RandomForestClassifier(
                n_estimators=800,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=123,
                n_jobs=-1
            )
        }
        
        for name, config in rf_configs.items():
            acc, auc = self.train_model_cv(name, config, 'sklearn')
            results[name] = {'accuracy': acc, 'auc': auc}
        
        # Extra Trees
        et_config = ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        acc, auc = self.train_model_cv('et_1', et_config, 'sklearn')
        results['et_1'] = {'accuracy': acc, 'auc': auc}
        
        self.model_results = results
        return self
    
    def train_model_cv(self, model_name, model_config, model_type='xgb'):
        """Enhanced cross-validation with more folds"""
        print(f"Training {model_name}...")
        
        # Use 10-fold CV for better stability
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        oof_preds = np.zeros(len(self.X_train))
        test_preds = np.zeros(len(self.X_test))
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            y_fold_val = self.y_train.iloc[val_idx]
            
            if model_type == 'xgb':
                model = xgb.XGBClassifier(**model_config)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    early_stopping_rounds=100,
                    verbose=False
                )
            elif model_type == 'lgb':
                model = lgb.LGBMClassifier(**model_config)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
            else:  # sklearn models
                model = model_config
                model.fit(X_fold_train, y_fold_train)
            
            # Predict validation set
            val_preds = model.predict_proba(X_fold_val)[:, 1]
            oof_preds[val_idx] = val_preds
            
            # Predict test set
            test_preds += model.predict_proba(self.X_test)[:, 1] / 10
            
            # Calculate fold score
            fold_accuracy = accuracy_score(y_fold_val, (val_preds > 0.5).astype(int))
            fold_scores.append(fold_accuracy)
            
            if fold % 2 == 0:  # Print every 2nd fold to reduce output
                print(f"    Fold {fold + 1} Accuracy: {fold_accuracy:.6f}")
        
        # Overall CV score
        cv_accuracy = accuracy_score(self.y_train, (oof_preds > 0.5).astype(int))
        cv_auc = roc_auc_score(self.y_train, oof_preds)
        
        print(f"  {model_name} CV Accuracy: {cv_accuracy:.6f}")
        print(f"  {model_name} CV AUC: {cv_auc:.6f}")
        
        # Store results
        self.oof_predictions[model_name] = oof_preds
        self.test_predictions[model_name] = test_preds
        
        return cv_accuracy, cv_auc
    
    def create_advanced_ensemble(self):
        """Create advanced ensemble with multiple strategies"""
        print("\n" + "="*60)
        print("CREATING ADVANCED ENSEMBLE")
        print("="*60)
        
        # Strategy 1: Simple average
        oof_simple = np.mean(list(self.oof_predictions.values()), axis=0)
        test_simple = np.mean(list(self.test_predictions.values()), axis=0)
        
        simple_accuracy = accuracy_score(self.y_train, (oof_simple > 0.5).astype(int))
        simple_auc = roc_auc_score(self.y_train, oof_simple)
        
        print(f"Simple Average Ensemble:")
        print(f"  CV Accuracy: {simple_accuracy:.6f}")
        print(f"  CV AUC: {simple_auc:.6f}")
        
        # Strategy 2: Weighted by accuracy
        acc_weights = []
        model_names = []
        for name, results in self.model_results.items():
            acc_weights.append(results['accuracy'])
            model_names.append(name)
        
        acc_weights = np.array(acc_weights)
        acc_weights = acc_weights / acc_weights.sum()
        
        oof_weighted_acc = np.zeros(len(self.y_train))
        test_weighted_acc = np.zeros(len(self.X_test))
        
        for name, weight in zip(model_names, acc_weights):
            oof_weighted_acc += self.oof_predictions[name] * weight
            test_weighted_acc += self.test_predictions[name] * weight
        
        weighted_acc_accuracy = accuracy_score(self.y_train, (oof_weighted_acc > 0.5).astype(int))
        weighted_acc_auc = roc_auc_score(self.y_train, oof_weighted_acc)
        
        print(f"\nWeighted by Accuracy Ensemble:")
        print(f"  CV Accuracy: {weighted_acc_accuracy:.6f}")
        print(f"  CV AUC: {weighted_acc_auc:.6f}")
        
        # Strategy 3: Weighted by AUC
        auc_weights = []
        for name, results in self.model_results.items():
            auc_weights.append(results['auc'])
        
        auc_weights = np.array(auc_weights)
        auc_weights = auc_weights / auc_weights.sum()
        
        oof_weighted_auc = np.zeros(len(self.y_train))
        test_weighted_auc = np.zeros(len(self.X_test))
        
        for name, weight in zip(model_names, auc_weights):
            oof_weighted_auc += self.oof_predictions[name] * weight
            test_weighted_auc += self.test_predictions[name] * weight
        
        weighted_auc_accuracy = accuracy_score(self.y_train, (oof_weighted_auc > 0.5).astype(int))
        weighted_auc_auc = roc_auc_score(self.y_train, oof_weighted_auc)
        
        print(f"\nWeighted by AUC Ensemble:")
        print(f"  CV Accuracy: {weighted_auc_accuracy:.6f}")
        print(f"  CV AUC: {weighted_auc_auc:.6f}")
        
        # Strategy 4: Rank averaging
        oof_ranks = np.zeros((len(self.y_train), len(self.oof_predictions)))
        test_ranks = np.zeros((len(self.X_test), len(self.test_predictions)))
        
        for i, (name, preds) in enumerate(self.oof_predictions.items()):
            oof_ranks[:, i] = pd.Series(preds).rank(pct=True)
            test_ranks[:, i] = pd.Series(self.test_predictions[name]).rank(pct=True)
        
        oof_rank_avg = np.mean(oof_ranks, axis=1)
        test_rank_avg = np.mean(test_ranks, axis=1)
        
        rank_accuracy = accuracy_score(self.y_train, (oof_rank_avg > 0.5).astype(int))
        rank_auc = roc_auc_score(self.y_train, oof_rank_avg)
        
        print(f"\nRank Averaging Ensemble:")
        print(f"  CV Accuracy: {rank_accuracy:.6f}")
        print(f"  CV AUC: {rank_auc:.6f}")
        
        # Select best ensemble
        ensemble_results = {
            'simple': (simple_accuracy, oof_simple, test_simple),
            'weighted_acc': (weighted_acc_accuracy, oof_weighted_acc, test_weighted_acc),
            'weighted_auc': (weighted_auc_accuracy, oof_weighted_auc, test_weighted_auc),
            'rank_avg': (rank_accuracy, oof_rank_avg, test_rank_avg)
        }
        
        best_ensemble = max(ensemble_results.items(), key=lambda x: x[1][0])
        best_name, (best_acc, best_oof, best_test) = best_ensemble
        
        print(f"\nBest ensemble: {best_name} with CV Accuracy: {best_acc:.6f}")
        
        self.final_oof = best_oof
        self.final_test = best_test
        self.final_accuracy = best_acc
        self.best_ensemble_name = best_name
        
        return self
    
    def create_submission(self):
        """Create enhanced submission"""
        print("\n" + "="*60)
        print("CREATING SUBMISSION")
        print("="*60)
        
        # Use probability threshold optimization
        thresholds = np.arange(0.45, 0.56, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            score = accuracy_score(self.y_train, (self.final_oof > threshold).astype(int))
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold:.3f} (CV Accuracy: {best_score:.6f})")
        
        # Apply optimal threshold
        final_predictions = (self.final_test > best_threshold).astype(int)
        final_labels = ['Extrovert' if pred == 1 else 'Introvert' for pred in final_predictions]
        
        submission = pd.DataFrame({
            'id': self.test_ids,
            'Personality': final_labels
        })
        
        print(f"Submission shape: {submission.shape}")
        print(f"Prediction distribution:")
        print(submission['Personality'].value_counts())
        print(f"Prediction proportions:")
        print(submission['Personality'].value_counts(normalize=True))
        
        # Save submission
        submission.to_csv('enhanced_submission.csv', index=False)
        print(f"\nSubmission saved to 'enhanced_submission.csv'")
        
        return submission
    
    def print_enhanced_summary(self):
        """Print enhanced summary"""
        print("\n" + "="*80)
        print("ENHANCED FINAL SUMMARY")
        print("="*80)
        
        print("Individual Model Performance:")
        sorted_models = sorted(self.model_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for name, results in sorted_models:
            print(f"  {name:12s}: Accuracy={results['accuracy']:.6f}, AUC={results['auc']:.6f}")
        
        print(f"\nBest Ensemble: {self.best_ensemble_name}")
        print(f"  CV Accuracy: {self.final_accuracy:.6f}")
        
        print(f"\nData Summary:")
        print(f"  Training samples: {len(self.X_train):,}")
        print(f"  Test samples: {len(self.X_test):,}")
        print(f"  Features used: {self.X_train.shape[1]}")
        
        target_dist = pd.Series(self.y_train).value_counts()
        print(f"\nTarget Distribution:")
        print(f"  Extrovert: {target_dist[1]:,} ({target_dist[1]/len(self.y_train):.1%})")
        print(f"  Introvert: {target_dist[0]:,} ({target_dist[0]/len(self.y_train):.1%})")

# Run the enhanced pipeline
if __name__ == "__main__":
    print("="*80)
    print("ENHANCED PERSONALITY PREDICTION PIPELINE")
    print("="*80)
    
    ensemble = EnhancedPersonalityEnsemble()
    
    # Enhanced pipeline
    ensemble.load_and_prepare_data()
    ensemble.train_enhanced_models()
    ensemble.create_advanced_ensemble()
    submission = ensemble.create_submission()
    ensemble.print_enhanced_summary()
    
    print("\n" + "="*80)
    print("ENHANCED PIPELINE COMPLETE!")
    print("="*80)


