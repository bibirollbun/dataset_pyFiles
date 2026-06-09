#!/usr/bin/env python
# coding: utf-8

# ## Fertilizer Prediction - Error-Free Implementation
# **Playground Series S5E6 - Final Corrected Version**

# ### 1. Configuration & Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

# Optimal configuration
class Config:
    n_folds = 3  # Faster training with 3 folds
    random_state = 42
    early_stopping = 50  # Earlier stopping for speed
    n_estimators = 800  # Reduced number of trees


# ### 2. Robust Data Processing
class DataProcessor:
    @staticmethod
    def load_and_fix_data():
        """Load data with automatic column fix"""
        train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
        test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
        return (
            train.rename(columns={'Temparature': 'Temperature'}),
            test.rename(columns={'Temparature': 'Temperature'})
        )

    @staticmethod
    def create_essential_features(df):
        """Only most impactful features"""
        eps = 1e-5
        df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + eps)
        df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
        return df

    @staticmethod
    def preprocess(train, test):
        """Proper preprocessing with categorical handling"""
        train = DataProcessor.create_essential_features(train)
        test = DataProcessor.create_essential_features(test)
        
        le = LabelEncoder()
        train['Fertilizer_encoded'] = le.fit_transform(train['Fertilizer Name'])
        
        # Convert categorical columns to numeric codes
        for col in ['Soil Type', 'Crop Type']:
            train[col] = train[col].astype('category').cat.codes
            test[col] = test[col].astype('category').cat.codes
        
        X = train.drop(['id', 'Fertilizer Name', 'Fertilizer_encoded'], axis=1)
        y = train['Fertilizer_encoded']
        X_test = test.drop('id', axis=1)
        
        return X, y, X_test, le


# ### 3. Error-Free Model Training
class FertilizerModel:
    def __init__(self):
        """Pre-configured optimal models with proper settings"""
        self.models = {
            'xgb': XGBClassifier(
                n_estimators=Config.n_estimators,
                learning_rate=0.1,
                max_depth=6,
                tree_method='hist',
                random_state=Config.random_state,
                eval_metric='mlogloss',
                early_stopping_rounds=Config.early_stopping
            ),
            'lgb': LGBMClassifier(
                n_estimators=Config.n_estimators,
                learning_rate=0.1,
                num_leaves=31,
                random_state=Config.random_state
            ),
            'cat': CatBoostClassifier(
                iterations=Config.n_estimators,
                learning_rate=0.1,
                depth=6,
                random_state=Config.random_state,
                verbose=0
            )
        }

    def train_predict(self, X, y, X_test, le):
        """Robust training with validation"""
        test_preds = np.zeros((len(X_test), len(le.classes_)))
        
        skf = StratifiedKFold(n_splits=Config.n_folds, 
                            shuffle=True, 
                            random_state=Config.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            for name, model in self.models.items():
                print(f"Fold {fold} - Training {name.upper():<4}", end=' ')
                
                if name == 'xgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=0
                    )
                elif name == 'lgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[early_stopping(stopping_rounds=Config.early_stopping)],
                        eval_metric='multi_logloss'
                    )
                else:  # CatBoost
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=Config.early_stopping
                    )
                
                test_preds += model.predict_proba(X_test) / Config.n_folds
                val_score = top_k_accuracy_score(y_val, model.predict_proba(X_val), k=3)
                print(f"âœ“ (Val MAP@3: {val_score:.4f})")
        
        # Generate competition-format predictions
        top3 = np.argsort(-test_preds, axis=1)[:, :3]
        return [' '.join(le.inverse_transform(row)) for row in top3]


# ### 4. Main Execution
if __name__ == "__main__":
    print("ğŸ”� Loading data...")
    train, test = DataProcessor.load_and_fix_data()
    
    print("âš¡ Preprocessing...")
    X, y, X_test, le = DataProcessor.preprocess(train, test)
    
    print("\nğŸš€ Training models...")
    model = FertilizerModel()
    predictions = model.train_predict(X, y, X_test, le)
    
    print("\nğŸ�¯ Creating submission...")
    pd.DataFrame({
        'id': test['id'],
        'Fertilizer Name': predictions
    }).to_csv('submission.csv', index=False)
    
    print("âœ… Done! Submission file created.")

