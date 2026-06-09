# %% [markdown]
# # FertiGenius Pro: Kaggle Competition Submission
# **Playground Series S5E6 - Final Optimized Pipeline**

# %% [markdown]
# ## 1. Configuration & Setup

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Configuration
class Config:
    n_folds = 5
    random_state = 42
    early_stopping_rounds = 50
    n_estimators = 1000


# ## 2. Data Loading & Preprocessing

# %%
def load_and_preprocess():
    """Load and prepare the competition data"""
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    
    # Handle column name variations
    for df in [train, test]:
        df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
        
        # Convert categoricals to numeric codes
        df['Soil Type'] = df['Soil Type'].astype('category').cat.codes
        df['Crop Type'] = df['Crop Type'].astype('category').cat.codes
    
    return train, test

# %%
def create_features(df):
    """Feature engineering"""
    eps = 1e-5
    df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + eps)
    df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + eps)
    df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
    return df


# ## 3. Model Training & Prediction

# %%
def train_and_predict(X, y, X_test, le):
    """Train models and generate predictions"""
    models = [
        XGBClassifier(
            n_estimators=Config.n_estimators,
            learning_rate=0.1,
            max_depth=6,
            random_state=Config.random_state,
            eval_metric='mlogloss'
        ),
        LGBMClassifier(
            n_estimators=Config.n_estimators,
            random_state=Config.random_state,
            verbose=-1
        )
    ]
    
    test_preds = np.zeros((len(X_test), len(le.classes_)))
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.random_state)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        for model in models:
            model.fit(X_train, y_train)
            test_preds += model.predict_proba(X_test) / (Config.n_folds * len(models))
    
    top3 = np.argsort(-test_preds, axis=1)[:, :3]
    return [' '.join(le.inverse_transform(row)) for row in top3]


# ## 4. Execution Pipeline

# %%
try:
    print("ğŸš€ Starting pipeline...")
    
    # Load and prepare data
    print("ğŸ”� Loading data...")
    train, test = load_and_preprocess()
    
    # Feature engineering
    print("ğŸ§  Creating features...")
    train = create_features(train)
    test = create_features(test)
    
    # Prepare target
    le = LabelEncoder()
    train['target'] = le.fit_transform(train['Fertilizer Name'])
    
    # Split data
    X = train.drop(['id', 'Fertilizer Name', 'target'], axis=1)
    y = train['target']
    X_test = test.drop('id', axis=1)
    
    # Train and predict
    print("ğŸ�¯ Training models...")
    predictions = train_and_predict(X, y, X_test, le)
    
    # Create submission
    print("ğŸ’¾ Generating submission file...")
    submission = pd.DataFrame({
        'id': test['id'],
        'Fertilizer Name': predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    print("âœ… Submission created successfully!")
    print("\nSubmission preview:")
    print(submission.head())

except Exception as e:
    print(f"â�Œ Error: {str(e)}")
    # Ensure we always create a submission
    if 'test' in locals():
        pd.DataFrame({
            'id': test['id'],
            'Fertilizer Name': ['Urea 14-35-14 10-26-26'] * len(test)
        }).to_csv('submission.csv', index=False)
        print("âš ï¸� Created fallback submission")

