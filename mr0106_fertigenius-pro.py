#!/usr/bin/env python
# coding: utf-8

# # FertiGenius Pro: Final Optimized Solution
# **Playground Series S5E6 - Error-Free Implementation**

# ## 1. Configuration & Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

class Config:
    n_folds = 3  # Faster validation
    random_state = 42
    early_stopping_rounds = 50  # Early stopping
    n_estimators = 500  # Balanced speed/performance


# ## 2. Robust Data Processing
class DataProcessor:
    @staticmethod
    def load_and_prepare():
        """Optimized data loading and preparation"""
        train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
        test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
        
        # Fix column name and convert categoricals
        for df in [train, test]:
            df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
            df['Soil Type'] = df['Soil Type'].astype('category').cat.codes
            df['Crop Type'] = df['Crop Type'].astype('category').cat.codes
            
        return train, test

    @staticmethod
    def create_features(df):
        """Essential feature engineering"""
        eps = 1e-5
        df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + eps)
        df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + eps)
        df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
        return df



# ## 3. Error-Free Model Training
class FertilizerPredictor:
    def __init__(self):
        """Initialize models with correct parameters"""
        self.models = {
            'xgb': XGBClassifier(
                n_estimators=Config.n_estimators,
                learning_rate=0.1,
                max_depth=6,
                tree_method='hist',
                enable_categorical=True,
                random_state=Config.random_state,
                eval_metric='mlogloss',
                early_stopping_rounds=Config.early_stopping_rounds
            ),
            'lgb': LGBMClassifier(
                n_estimators=Config.n_estimators,
                learning_rate=0.1,
                num_leaves=31,
                random_state=Config.random_state
            )
        }

    def train_predict(self, X, y, X_test, le):
        """Robust training and prediction"""
        test_preds = np.zeros((len(X_test), len(le.classes_)))
        
        skf = StratifiedKFold(n_splits=Config.n_folds, 
                            shuffle=True, 
                            random_state=Config.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            for name, model in self.models.items():
                print(f"Fold {fold} - Training {name.upper()}", end=' ')
                
                if name == 'xgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=0
                    )
                else:  # LightGBM
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[early_stopping(stopping_rounds=Config.early_stopping_rounds)],
                        eval_metric='multi_logloss'
                    )
                
                test_preds += model.predict_proba(X_test) / Config.n_folds
                val_score = top_k_accuracy_score(y_val, model.predict_proba(X_val), k=3)
                print(f"âœ“ (MAP@3: {val_score:.4f})")
        
        # Generate top-3 predictions
        top3 = np.argsort(-test_preds, axis=1)[:, :3]
        return [' '.join(le.inverse_transform(row)) for row in top3]


# ## 4. Execution Pipeline
if __name__ == "__main__":
    print("âš¡ Optimized Execution Mode âš¡")
    
    print("\nğŸ”� Loading and preparing data...")
    train, test = DataProcessor.load_and_prepare()
    
    print("ğŸ§¹ Engineering features...")
    train = DataProcessor.create_features(train)
    test = DataProcessor.create_features(test)
    
    le = LabelEncoder()
    train['target'] = le.fit_transform(train['Fertilizer Name'])
    
    X = train.drop(['id', 'Fertilizer Name', 'target'], axis=1)
    y = train['target']
    X_test = test.drop('id', axis=1)
    
    print("\nğŸš€ Training optimized models...")
    predictor = FertilizerPredictor()
    predictions = predictor.train_predict(X, y, X_test, le)
    
    print("\nğŸ�¯ Creating submission file...")
    pd.DataFrame({
        'id': test['id'],
        'Fertilizer Name': predictions
    }).to_csv('submission.csv', index=False)
    
    print("âœ… Submission created successfully!")

