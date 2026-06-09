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


!pip install optuna-integration[lightgbm]


import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from optuna.integration.lightgbm import LightGBMPruningCallback
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Load all required datasets"""
    train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
    test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
    bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
    bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
    pos_cash = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
    credit_card = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
    installments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
    
    return train, test, bureau, bureau_balance, pos_cash, credit_card, installments

def preprocess_grouped_data(df, group_col):
    """Aggregate grouped numeric data"""
    if group_col not in df.columns:
        return pd.DataFrame()
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != group_col]  # Exclude ID column
    
    if not numeric_cols:
        return pd.DataFrame()
    
    # Perform aggregation and reset the index to keep the group column as a regular column
    df_agg = df.groupby(group_col)[numeric_cols].agg(['mean', 'sum', 'max', 'min']).reset_index()
    
    # Flatten column names
    df_agg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_agg.columns]
    
    return df_agg

def merge_data(train, test, bureau, bureau_balance, pos_cash, credit_card, installments):
    """Merge datasets with train and test"""
    
    # Print column names to check for SK_ID_CURR variations
    print("train columns:", train.columns)
    print("test columns:", test.columns)
    print("bureau columns:", bureau.columns)
    print("pos_cash columns:", pos_cash.columns)
    print("credit_card columns:", credit_card.columns)
    print("installments columns:", installments.columns)

    # Check if the columns are named differently and rename if necessary
    if 'SK_ID_CURR_bureau' in bureau.columns:
        bureau.rename(columns={'SK_ID_CURR_bureau': 'SK_ID_CURR'}, inplace=True)
    if 'SK_ID_CURR_bureau_balance' in bureau_balance.columns:
        bureau_balance.rename(columns={'SK_ID_CURR_bureau_balance': 'SK_ID_CURR'}, inplace=True)
    if 'SK_ID_CURR_POS_CASH_balance' in pos_cash.columns:
        pos_cash.rename(columns={'SK_ID_CURR_POS_CASH_balance': 'SK_ID_CURR'}, inplace=True)
    if 'SK_ID_CURR_credit_card_balance' in credit_card.columns:
        credit_card.rename(columns={'SK_ID_CURR_credit_card_balance': 'SK_ID_CURR'}, inplace=True)
    if 'SK_ID_CURR_installments' in installments.columns:
        installments.rename(columns={'SK_ID_CURR_installments': 'SK_ID_CURR'}, inplace=True)

    # Aggregate data
    bureau_agg = preprocess_grouped_data(bureau, 'SK_ID_CURR')
    pos_agg = preprocess_grouped_data(pos_cash, 'SK_ID_CURR')
    credit_agg = preprocess_grouped_data(credit_card, 'SK_ID_CURR')
    installments_agg = preprocess_grouped_data(installments, 'SK_ID_CURR')

    # Make sure 'SK_ID_CURR' is present in the aggregated datasets
    for df in [bureau_agg, pos_agg, credit_agg, installments_agg]:
        if 'SK_ID_CURR' not in df.columns:
            print("Missing SK_ID_CURR in aggregated dataset. Columns are:", df.columns)
            raise KeyError("SK_ID_CURR not found in processed dataset!")

    # Merge datasets with train and test
    for df in [train, test]:
        # Merge each dataframe with the aggregated ones on 'SK_ID_CURR'
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
        df = df.merge(credit_agg, on='SK_ID_CURR', how='left')
        df = df.merge(installments_agg, on='SK_ID_CURR', how='left')
        df.fillna(0, inplace=True)

    return train, test

def remove_correlated_features(train, threshold=0.9):
    """Remove highly correlated numerical features"""
    numeric_train = train.select_dtypes(include=['number'])
    corr_matrix = numeric_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    train.drop(columns=to_drop, inplace=True, errors='ignore')
    return train

def handle_categorical_data(train, test):
    """Handle categorical columns in the train and test data"""
    combined_data = pd.concat([train, test], axis=0, ignore_index=True)
    
    categorical_columns = combined_data.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        le = LabelEncoder()
        combined_data[column] = combined_data[column].astype(str)
        combined_data[column] = le.fit_transform(combined_data[column])

    train_data = combined_data.iloc[:len(train), :].copy()
    test_data = combined_data.iloc[len(train):, :].copy()

    return train_data, test_data

def optimize_lgbm(X, y):
    """Optimize LightGBM hyperparameters using Optuna"""
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150, step=10),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100, step=10)
        }
        
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMClassifier(**param)
        model.fit(
            X_train, y_train, eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        preds = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, preds)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    return study.best_params

def train_model(train, test):
    """Train the LightGBM model"""
    features = [col for col in train.columns if col not in ['SK_ID_CURR', 'TARGET']]
    X = train[features]
    y = train['TARGET']
    
    best_params = optimize_lgbm(X, y)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train, y_train, eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )
    
    test['TARGET'] = model.predict_proba(test[features])[:, 1]
    test[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)
    
    return model

def plot_feature_importances(model, features):
    """Plot feature importances"""
    fi = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(fi['feature'][:15], fi['importance'][:15], edgecolor='k')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()
    
    return fi

# Load Data
train, test, bureau, bureau_balance, pos_cash, credit_card, installments = load_data()

# Merge Data
train, test = merge_data(train, test, bureau, bureau_balance, pos_cash, credit_card, installments)

# Handle categorical columns
train, test = handle_categorical_data(train, test)

# Feature Selection (Remove correlated features)
train = remove_correlated_features(train)

# Train Model
model = train_model(train, test)

# Feature Importance
features = [col for col in train.columns if col not in ['SK_ID_CURR', 'TARGET']]
fi_sorted = plot_feature_importances(model, features)

# Save Additional Submissions
test[['SK_ID_CURR', 'TARGET']].to_csv('submission_raw.csv', index=False)
test[['SK_ID_CURR', 'TARGET']].to_csv('submission_corrs.csv', index=False)

