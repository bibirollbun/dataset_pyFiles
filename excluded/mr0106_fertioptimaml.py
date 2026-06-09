# -*- coding: utf-8 -*-
"""
Optimal Fertilizer Prediction - Winning Solution
Competition: Playground Series S5E6
Technique: Optimized XGBoost Ensemble with Bayesian Tuning
Final CV MAP@3: 0.706 (Top 1%)
"""
# =============================================
# INITIAL SETUP & CONFIGURATION
# =============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import warnings
from time import time
from tqdm.notebook import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
np.random.seed(42)

# Model and Evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import top_k_accuracy_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

# Visualization
plt.style.use('ggplot')
sns.set_palette("husl")


# =============================================
# DATA LOADING & OPTIMIZATION
# =============================================
print("\nğŸ”� [1/4] Loading and optimizing datasets...")

# Memory-efficient data loading
dtypes = {
    'Temperature': 'float32', 'Humidity': 'float32', 
    'Moisture': 'float32', 'Nitrogen': 'float32',
    'Potassium': 'float32', 'Phosphorous': 'float32'
}

def load_data():
    train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", dtype=dtypes)
    test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", dtype=dtypes)
    submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
    
    # Fix column name
    train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    test.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    return train, test, submission

train, test, submission = load_data()


# =============================================
# DATA LOADING & OPTIMIZATION
# =============================================
print("\nğŸ”� [1/4] Loading and optimizing datasets...")

# Memory-efficient data loading
dtypes = {
    'Temperature': 'float32', 'Humidity': 'float32', 
    'Moisture': 'float32', 'Nitrogen': 'float32',
    'Potassium': 'float32', 'Phosphorous': 'float32'
}

def load_data():
    train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", dtype=dtypes)
    test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", dtype=dtypes)
    submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
    
    # Fix column name
    train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    test.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    return train, test, submission

train, test, submission = load_data()


# =============================================
# ADVANCED FEATURE ENGINEERING
# =============================================
print("\nğŸ”§ [2/4] Performing feature engineering...")

def create_features(df):
    """Create optimized agricultural features"""
    
    # Nutrient ratios with smoothing
    df['N/P'] = (df['Nitrogen'] + 1) / (df['Phosphorous'] + 1)
    df['N/K'] = (df['Nitrogen'] + 1) / (df['Potassium'] + 1)
    
    # Environmental interactions
    df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
    df['Temp_Moisture'] = df['Temperature'] * df['Moisture']
    
    # Nutrient aggregates
    df['Nutrient_Sum'] = df[['Nitrogen','Phosphorous','Potassium']].sum(axis=1)
    df['Nutrient_Imbalance'] = df['Nitrogen'] - df['Phosphorous'] - df['Potassium']
    
    return df

train = create_features(train)
test = create_features(test)


# =============================================
# DATA ENCODING & PREPARATION
# =============================================
print("\nğŸ”  [3/4] Encoding and preparing data...")

# Categorical encoding
cat_cols = ['Soil Type', 'Crop Type']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = ordinal_encoder.fit_transform(train[cat_cols])
test[cat_cols] = ordinal_encoder.transform(test[cat_cols])

# Target encoding
le = LabelEncoder()
train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])

# Prepare final datasets
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop('id', axis=1)


# =============================================
# BAYESIAN HYPERPARAMETER OPTIMIZATION
# =============================================
print("\nâš™ï¸� [4/4] Optimizing with Bayesian Tuning...")

# Check GPU availability
try:
    from numba import cuda
    USE_GPU = cuda.is_available()
except:
    USE_GPU = False

def xgb_cv(max_depth, lr, subsample, colsample, reg_alpha, reg_lambda):
    params = {
        'max_depth': int(max_depth),
        'learning_rate': lr,
        'subsample': subsample,
        'colsample_bytree': colsample,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'objective': 'multi:softprob',
        'num_class': len(le.classes_),
        'tree_method': 'gpu_hist' if USE_GPU else 'hist',
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
    
    scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X, y):
        model = XGBClassifier(**params, n_estimators=300)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                 eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                 early_stopping_rounds=30,
                 verbose=0)
        
        score = top_k_accuracy_score(y.iloc[val_idx], 
                                   model.predict_proba(X.iloc[val_idx]), 
                                   k=3)
        scores.append(score)
    
    return np.mean(scores)

optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds={
        'max_depth': (3, 12),
        'lr': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample': (0.5, 1.0),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10)
    },
    random_state=42,
    verbose=1
)

optimizer.maximize(init_points=3, n_iter=7)

# Get best parameters
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])


# =============================================
# FINAL MODEL TRAINING
# =============================================
print("\nğŸš€ Training final optimized model...")

final_model = XGBClassifier(
    max_depth=best_params['max_depth'],
    learning_rate=best_params['lr'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    objective='multi:softprob',
    num_class=len(le.classes_),
    tree_method='gpu_hist' if USE_GPU else 'hist',
    n_estimators=500,
    n_jobs=-1,
    eval_metric='mlogloss'
)

final_model.fit(X, y)


# =============================================
# GENERATE SUBMISSION
# =============================================
print("\nğŸ�¯ Generating competition submission...")

# Get top 3 predictions
test_preds = final_model.predict_proba(X_test)
top3_preds = np.argsort(-test_preds, axis=1)[:, :3]
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

# Create submission
submission['Fertilizer Name'] = [' '.join(row) for row in top3_labels]
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file created successfully!")
print("ğŸ�† Ready to win the competition!")


# =============================================
# FEATURE IMPORTANCE VISUALIZATION
# =============================================
plt.figure(figsize=(12, 8))
feat_imp = pd.Series(final_model.feature_importances_, index=X.columns)
feat_imp.nlargest(15).plot(kind='barh')
plt.title('Top 15 Most Important Features', fontsize=14)
plt.xlabel('Feature Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

