# # FertiGenius Pro: Competition-Ready Solution
# **Playground Series S5E6 | Optimized Pipeline**

# %% [markdown]
# ## 1. Configuration & Imports
# %%
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping  # Import added here

class Config:
    n_folds = 5
    random_state = 42
    early_stop = 50
    n_estimators = 500
    learning_rate = 0.1


# ## 2. Data Processing Pipeline
# %%
def load_and_preprocess():
    """Robust data loading with type conversion"""
    num_dtypes = {
        'Nitrogen': 'float32',
        'Phosphorous': 'float32', 
        'Potassium': 'float32',
        'Temperature': 'float32',
        'Humidity': 'float32'
    }
    
    train = pd.read_csv(
        '/kaggle/input/playground-series-s5e6/train.csv',
        dtype=num_dtypes
    ).rename(columns={'Temparature': 'Temperature'})
    
    test = pd.read_csv(
        '/kaggle/input/playground-series-s5e6/test.csv', 
        dtype=num_dtypes
    ).rename(columns={'Temparature': 'Temperature'})
    
    for df in [train, test]:
        df['Soil Type'] = pd.factorize(df['Soil Type'])[0].astype('int8')
        df['Crop Type'] = pd.factorize(df['Crop Type'])[0].astype('int8')
    
    return train, test

def create_features(df):
    """Optimized feature engineering"""
    eps = 1e-6
    df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + eps)
    df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + eps)
    df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
    df['Soil_Crop_Combo'] = (df['Soil Type'] * 100 + df['Crop Type']).astype('int16')
    return df


# ## 3. Model Training & Validation
# %%
def build_models():
    """Create optimized ensemble models"""
    return [
        XGBClassifier(
            n_estimators=Config.n_estimators,
            learning_rate=Config.learning_rate,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            early_stopping_rounds=Config.early_stop,
            random_state=Config.random_state,
            n_jobs=-1,
            verbosity=0
        ),
        LGBMClassifier(
            n_estimators=Config.n_estimators,
            learning_rate=Config.learning_rate,
            num_leaves=63,
            feature_fraction=0.8,
            random_state=Config.random_state,
            n_jobs=-1,
            verbose=-1
        )
    ]

def cross_val_predict(models, X, y, X_test, le):
    """Enhanced cross-validation"""
    test_preds = np.zeros((len(X_test), len(le.classes_)))
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.random_state)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        for model in models:
            if isinstance(model, XGBClassifier):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=0
                )
            else:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='multi_logloss',
                    callbacks=[early_stopping(stopping_rounds=Config.early_stop)]
                )
            test_preds += model.predict_proba(X_test) / (Config.n_folds * len(models))
    
    return test_preds


# ## 4. Execution Pipeline
# %%
# Data Preparation
print("ðŸš€ Loading and preprocessing data...")
train, test = load_and_preprocess()

print("ðŸ§  Engineering features...")
train = create_features(train)
test = create_features(test)

# Target Processing
print("ðŸ”§ Encoding target variable...")
le = LabelEncoder()
train['target'] = le.fit_transform(train['Fertilizer Name'])

# Feature Selection
X = train.drop(['id', 'Fertilizer Name', 'target'], axis=1)
y = train['target']
X_test = test.drop('id', axis=1)

# Model Training
print("ðŸŽ¯ Training optimized ensemble...")
models = build_models()
test_preds = cross_val_predict(models, X, y, X_test, le)

# Submission Generation
print("ðŸ’¾ Creating competition submission...")
top3_preds = np.argsort(-test_preds, axis=1)[:, :3]
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(le.inverse_transform(pred)) for pred in top3_preds]
})
submission.to_csv('submission.csv', index=False)

print("âœ… Submission successfully created!")
print("\nSubmission preview:")
print(submission.head())

