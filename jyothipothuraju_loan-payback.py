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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os
import warnings

# Ignore minor warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = './' # Change this to the directory where your train.csv and test.csv are located
TARGET_COLUMN = 'loan_paid_back'
ID_COLUMN = 'id'
N_SPLITS = 5 # Number of folds for Cross-Validation

def load_data():
    """Load the training and test datasets."""
    print("Loading data...")
    try:
        train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
        test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'train.csv' and 'test.csv' are in the DATA_PATH directory.")
        # --- SIMULATE DATA FOR DEMO PURPOSES IF FILES ARE MISSING ---
        print("\n*** Generating Simulated Data for Demo ***")
        N_TRAIN = 50000
        N_TEST = 10000
        
        data_dict = {
            ID_COLUMN: range(1, N_TRAIN + 1),
            'annual_income': np.random.lognormal(mean=10.5, sigma=0.5, size=N_TRAIN) * 1000,
            'loan_amount': np.random.randint(1000, 50000, N_TRAIN),
            'credit_score': np.random.randint(580, 800, N_TRAIN),
            'debt_to_income': np.random.beta(a=2, b=5, size=N_TRAIN) * 50,
            'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'medical', 'other'], N_TRAIN),
            'employment_years': np.random.choice(list(range(11)) + ['<1'], N_TRAIN),
            TARGET_COLUMN: np.random.choice([0, 1], N_TRAIN, p=[0.2, 0.8])
        }
        train_df = pd.DataFrame(data_dict)

        test_dict = {
            ID_COLUMN: range(N_TRAIN + 1, N_TRAIN + N_TEST + 1),
            'annual_income': np.random.lognormal(mean=10.5, sigma=0.5, size=N_TEST) * 1000,
            'loan_amount': np.random.randint(1000, 50000, N_TEST),
            'credit_score': np.random.randint(580, 800, N_TEST),
            'debt_to_income': np.random.beta(a=2, b=5, size=N_TEST) * 50,
            'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'medical', 'other'], N_TEST),
            'employment_years': np.random.choice(list(range(11)) + ['<1'], N_TEST),
        }
        test_df = pd.DataFrame(test_dict)
        return train_df, test_df


def preprocess(df):
    """Clean, engineer features, and encode data."""
    
    # 1. Feature Engineering: Create Ratios
    # Total Income feature is often helpful
    df['log_annual_income'] = np.log1p(df['annual_income'])
    
    # Debt Burden Ratio (Loan Amount relative to Income)
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1e-6) # Add small constant to prevent division by zero
    
    # 2. Handle 'employment_years'
    df['employment_years'] = df['employment_years'].replace({'<1': 0, '10+': 10}).astype(float).fillna(df['employment_years'].mode()[0])
    
    # 3. Handle Categorical Features (Label Encoding for simple use with LGBM)
    categorical_features = ['loan_purpose']
    for col in categorical_features:
        if col in df.columns:
            # Fill missing values with a placeholder category
            df[col] = df[col].fillna('Missing') 
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            df = df.drop(columns=[col])

    # 4. Handle Missing Values (Impute continuous features with median)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
        
    return df

def train_and_predict(train_df, test_df):
    """Train LightGBM using Stratified K-Fold and generate predictions."""
    
    X = train_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[ID_COLUMN])
    
    # Align columns between train and test after preprocessing
    common_cols = list(set(X.columns) & set(X_test.columns))
    X = X[common_cols]
    X_test = X_test[common_cols]

    # Initialize variables
    oof_preds = np.zeros(len(X)) # Out-Of-Fold Predictions (for validation)
    test_preds = np.zeros(len(X_test)) # Final test predictions
    feature_importances = pd.DataFrame(index=X.columns)
    
    # LightGBM Model Parameters (tuned for good performance on tabular data)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    print(f"\nStarting LightGBM Training with {N_SPLITS} Folds...")
    
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"--- Fold {fold+1}/{N_SPLITS} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMClassifier(**lgb_params)
        
        # Train the model
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        # Predict probabilities (Crucial: we need the probability of class 1)
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_index] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS # Average predictions

        # Save feature importance for analysis
        feature_importances[f'Fold_{fold+1}'] = model.feature_importances_

    # Calculate final OOF AUC Score
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n\n*** Cross-Validation ROC-AUC Score: {oof_auc:.5f} ***")

    return test_preds, feature_importances

def create_submission(test_df, test_preds):
    """Create the submission file."""
    
    submission_df = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        TARGET_COLUMN: test_preds
    })
    
    SUBMISSION_FILE = 'submission.csv'
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nSuccessfully created submission file: '{SUBMISSION_FILE}'")
    print("Head of Submission File:")
    print(submission_df.head())

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 1. Load Data
    train_df, test_df = load_data()
    
    # 2. Preprocess Data
    # Concatenate for consistent preprocessing across train and test sets
    combined_df = pd.concat([train_df.drop(columns=[TARGET_COLUMN], errors='ignore'), test_df], ignore_index=True)
    
    # Perform feature engineering and cleaning
    combined_processed = preprocess(combined_df.copy())
    
    # Split back into training and testing sets
    train_processed = combined_processed.iloc[:len(train_df)]
    test_processed = combined_processed.iloc[len(train_df):]
    
    # Add the target column back to the training set
    train_processed[TARGET_COLUMN] = train_df[TARGET_COLUMN]
    
    # 3. Train Model and Predict
    test_preds, feature_importances_df = train_and_predict(train_processed, test_processed)
    
    # 4. Create Submission File
    create_submission(test_df, test_preds)
    
    # 5. Analyze Feature Importance (Optional but Recommended)
    print("\nTop 10 Feature Importances (Avg across Folds):")
    avg_importance = feature_importances_df.mean(axis=1).sort_values(ascending=False)
    print(avg_importance.head(10))
    




