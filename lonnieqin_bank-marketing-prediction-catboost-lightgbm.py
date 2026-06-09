import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')
class BankMarketingPredictor:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.categorical_features = []
        self.numerical_features = []
        self.label_encoders = {}
        self.imputers = {}
        self.scaler = StandardScaler()
        
    def identify_feature_types(self, df):
        """Identify categorical and numerical features"""
        # Define categorical features based on the dataset description
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                          'loan', 'contact', 'month', 'poutcome']
        
        # Numerical features
        numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 
                         'pdays', 'previous']
        
        # Filter existing columns
        self.categorical_features = [col for col in categorical_cols if col in df.columns]
        self.numerical_features = [col for col in numerical_cols if col in df.columns]
        
        print(f"Categorical features: {self.categorical_features}")
        print(f"Numerical features: {self.numerical_features}")
        
    def preprocess_data(self, df, is_train=True):
        """Preprocess the data with imputation and encoding"""
        df_processed = df.copy()
        
        # Handle missing values for numerical features
        for col in self.numerical_features:
            if col in df_processed.columns:
                if is_train:
                    # Fit imputer on training data
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df_processed[col] = self.imputers[col].fit_transform(
                        df_processed[[col]]).flatten()
                else:
                    # Transform test data using fitted imputer
                    if col in self.imputers:
                        df_processed[col] = self.imputers[col].transform(
                            df_processed[[col]]).flatten()
        
        # Handle missing values for categorical features
        for col in self.categorical_features:
            if col in df_processed.columns:
                # Fill missing values with 'unknown'
                df_processed[col] = df_processed[col].fillna('unknown')
                
                if is_train:
                    # Fit label encoder on training data
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(
                        df_processed[col].astype(str))
                else:
                    # Transform test data using fitted encoder
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_categories = set(self.label_encoders[col].classes_)
                        df_processed[col] = df_processed[col].astype(str)
                        
                        # Replace unseen categories with most frequent category
                        most_frequent = self.label_encoders[col].classes_[0]
                        mask = ~df_processed[col].isin(unique_categories)
                        df_processed.loc[mask, col] = most_frequent
                        
                        df_processed[col] = self.label_encoders[col].transform(
                            df_processed[col])
        
        # Feature engineering
        df_processed = self.create_features(df_processed)
        
        return df_processed
    
    def create_features(self, df):
        """Create additional features"""
        df_new = df.copy()
        
        # Age groups
        if 'age' in df_new.columns:
            df_new['age_group'] = pd.cut(df_new['age'], 
                                       bins=[0, 30, 45, 60, 100], 
                                       labels=['young', 'middle', 'senior', 'elderly'])
            df_new['age_group'] = df_new['age_group'].cat.codes
        
        # Balance features
        if 'balance' in df_new.columns:
            df_new['balance_positive'] = (df_new['balance'] > 0).astype(int)
            df_new['balance_log'] = np.log1p(np.abs(df_new['balance']))
        
        # Campaign features
        if 'campaign' in df_new.columns:
            df_new['high_campaign'] = (df_new['campaign'] > 3).astype(int)
        
        # Previous contact features
        if 'pdays' in df_new.columns:
            df_new['was_contacted_before'] = (df_new['pdays'] != -1).astype(int)
            df_new['pdays_log'] = np.log1p(np.maximum(df_new['pdays'], 0))
        
        # Duration features
        if 'duration' in df_new.columns:
            df_new['duration_log'] = np.log1p(df_new['duration'])
            df_new['long_duration'] = (df_new['duration'] > 300).astype(int)
        
        return df_new
    
    def train_catboost(self, X_train, y_train, X_val, y_val, categorical_indices):
        """Train CatBoost model"""
        cb_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'random_seed': self.random_state,
            'od_type': 'Iter',
            'od_wait': 50,
            'verbose': False,
            'eval_metric': 'AUC',
            'task_type': 'GPU'
        }
        
        model = cb.CatBoostClassifier(**cb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=categorical_indices,
            early_stopping_rounds=50,
            verbose=False
        )
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val, categorical_indices):
        """Train LightGBM model"""
        # Convert categorical indices to column names for LightGBM
        categorical_features_lgb = [X_train.columns[i] for i in categorical_indices] if len(categorical_indices) > 0 else 'auto'
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features_lgb)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features_lgb, reference=train_data)
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'random_state': self.random_state,
            'verbose': -1,
            'device': 'gpu',                 # GPU device
            'gpu_platform_id': 0,            # CUDA platform
            'gpu_device_id': 0,              # First GPU
            'num_leaves': 63,                
            'learning_rate': 0.1,           
            'max_bin': 63,                   # GPU memory optimized
            'gpu_use_dp': True      
        }
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        return model
    
    def train_models(self, train_df, target_col='y'):
        """Train models using KFold cross-validation"""
        # Identify feature types
        self.identify_feature_types(train_df)
        
        # Preprocess data
        X = self.preprocess_data(train_df, is_train=True)
        y = train_df[target_col].values
        
        # Remove id and target columns
        feature_cols = [col for col in X.columns if col not in ['id', target_col]]
        X = X[feature_cols]
        
        # Get categorical feature indices for the processed data
        categorical_indices = []
        for i, col in enumerate(X.columns):
            if any(cat_feat in col for cat_feat in self.categorical_features) or col in ['age_group']:
                categorical_indices.append(i)
        
        print(f"Categorical feature indices: {categorical_indices}")
        
        # Initialize cross-validation
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Store models and predictions
        catboost_models = []
        lightgbm_models = []
        oof_catboost = np.zeros(len(X))
        oof_lightgbm = np.zeros(len(X))
        
        cv_scores_cb = []
        cv_scores_lgb = []
        
        print("Starting cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\nFold {fold + 1}/{self.n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train CatBoost
            print("Training CatBoost...")
            cb_model = self.train_catboost(X_train, y_train, X_val, y_val, categorical_indices)
            catboost_models.append(cb_model)
            
            # Predict with CatBoost
            val_pred_cb = cb_model.predict_proba(X_val)[:, 1]
            oof_catboost[val_idx] = val_pred_cb
            auc_cb = roc_auc_score(y_val, val_pred_cb)
            cv_scores_cb.append(auc_cb)
            print(f"CatBoost AUC: {auc_cb:.5f}")
            
            # Train LightGBM
            print("Training LightGBM...")
            lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val, categorical_indices)
            lightgbm_models.append(lgb_model)
            
            # Predict with LightGBM
            val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
            oof_lightgbm[val_idx] = val_pred_lgb
            auc_lgb = roc_auc_score(y_val, val_pred_lgb)
            cv_scores_lgb.append(auc_lgb)
            print(f"LightGBM AUC: {auc_lgb:.5f}")
        
        # Calculate overall CV scores
        cv_catboost = roc_auc_score(y, oof_catboost)
        cv_lightgbm = roc_auc_score(y, oof_lightgbm)
        
        print(f"\n{'='*50}")
        print(f"CatBoost CV AUC: {cv_catboost:.5f} (+/- {np.std(cv_scores_cb):.5f})")
        print(f"LightGBM CV AUC: {cv_lightgbm:.5f} (+/- {np.std(cv_scores_lgb):.5f})")
        
        # Ensemble prediction
        oof_ensemble = (oof_catboost + oof_lightgbm) / 2
        cv_ensemble = roc_auc_score(y, oof_ensemble)
        print(f"Ensemble CV AUC: {cv_ensemble:.5f}")
        print(f"{'='*50}")
        
        # Store models and feature columns
        self.catboost_models = catboost_models
        self.lightgbm_models = lightgbm_models
        self.feature_columns = X.columns.tolist()
        self.categorical_indices = categorical_indices
        
        return {
            'cv_catboost': cv_catboost,
            'cv_lightgbm': cv_lightgbm,
            'cv_ensemble': cv_ensemble,
            'oof_catboost': oof_catboost,
            'oof_lightgbm': oof_lightgbm,
            'oof_ensemble': oof_ensemble
        }
    
    def predict(self, test_df):
        """Make predictions on test data"""
        # Preprocess test data
        X_test = self.preprocess_data(test_df, is_train=False)
        
        # Select feature columns
        X_test = X_test[self.feature_columns]
        
        # Initialize prediction arrays
        pred_catboost = np.zeros(len(X_test))
        pred_lightgbm = np.zeros(len(X_test))
        
        # Average predictions from all folds
        for cb_model in self.catboost_models:
            pred_catboost += cb_model.predict_proba(X_test)[:, 1] / len(self.catboost_models)
        
        for lgb_model in self.lightgbm_models:
            pred_lightgbm += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / len(self.lightgbm_models)
        
        # Ensemble prediction
        pred_ensemble = (pred_catboost + pred_lightgbm) / 2
        
        return {
            'catboost': pred_catboost,
            'lightgbm': pred_lightgbm,
            'ensemble': pred_ensemble
        }

# Example usage
def main():
    # Load data
    train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Initialize predictor
    predictor = BankMarketingPredictor(n_splits=5, random_state=42)
    
    # Train models
    results = predictor.train_models(train, target_col='y')
    
    # Make predictions
    predictions = predictor.predict(test)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test['id'],
        'y': predictions['ensemble']
    })
    
    # Save submissions for all models
    submission.to_csv('submission_ensemble.csv', index=False)
    
    # Individual model submissions
    submission_cb = pd.DataFrame({
        'id': test['id'],
        'y': predictions['catboost']
    })
    submission_cb.to_csv('submission_catboost.csv', index=False)
    
    submission_lgb = pd.DataFrame({
        'id': test['id'],
        'y': predictions['lightgbm']
    })
    submission_lgb.to_csv('submission_lightgbm.csv', index=False)
    
    print("\nSubmission files created:")
    print("- submission_ensemble.csv (recommended)")
    print("- submission_catboost.csv")
    print("- submission_lightgbm.csv")
    
    return predictor, results, predictions


main()

