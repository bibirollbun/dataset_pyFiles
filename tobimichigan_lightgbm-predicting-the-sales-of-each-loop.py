# pipeline.py

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from datetime import datetime

# Modeling libraries
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# -------------------------------
# 1. DATA LOADING & PREPROCESSING
# -------------------------------

def load_data(data_dir='./data'):
    """
    Load all CSV files into DataFrames.
    """
    sales_train = pd.read_csv(os.path.join(data_dir, 'sales_train.csv'))
    sales_test  = pd.read_csv(os.path.join(data_dir, 'sales_test.csv'))
    inventory   = pd.read_csv(os.path.join(data_dir, 'inventory.csv'))
    calendar    = pd.read_csv(os.path.join(data_dir, 'calendar.csv'))
    test_weights= pd.read_csv(os.path.join(data_dir, 'test_weights.csv'))
    
    return sales_train, sales_test, inventory, calendar, test_weights


def preprocess_data(df, date_col='date'):
    """
    Convert date columns to datetime (if present) and fill missing numerical values with median.
    """
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    # Fill missing numerical values with the median value
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


def merge_data(sales_df, calendar_df, inventory_df):
    """
    Merge sales data with calendar and inventory information.
    """
    # Merge calendar on date and warehouse (calendar may contain extra dates)
    df = pd.merge(sales_df, calendar_df, on=['date', 'warehouse'], how='left')
    # Merge inventory on unique_id and warehouse
    df = pd.merge(df, inventory_df, on=['unique_id', 'warehouse'], how='left')
    
    gc.collect()
    return df


# -------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------

def create_date_features(df, date_col='date'):
    """
    Create time-based features.
    """
    df['year']       = df[date_col].dt.year
    df['month']      = df[date_col].dt.month
    df['day']        = df[date_col].dt.day
    df['weekday']    = df[date_col].dt.weekday
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
    return df


def compute_max_discount(df):
    """
    Create a feature for the maximum discount applied from available discount types.
    """
    discount_cols = [col for col in df.columns if 'type_' in col and 'discount' in col]
    df['max_discount'] = df[discount_cols].max(axis=1)
    return df


def create_lag_features(df, group_col='unique_id', target_col='sales', lags=[7, 14]):
    """
    Create lag features for the target column.
    Uses tqdm to monitor progress.
    """
    df = df.sort_values([group_col, 'date'])
    
    for lag in lags:
        lag_col = f'{target_col}_lag_{lag}'
        df[lag_col] = np.nan
        
        # Process each unique inventory id with progress monitoring
        for uid, group in tqdm(df.groupby(group_col), desc=f"Creating lag_{lag} features", leave=False):
            df.loc[group.index, lag_col] = group[target_col].shift(lag)
    
    # Fill missing lag values (using 0 here; adjust as needed)
    lag_cols = [f'{target_col}_lag_{lag}' for lag in lags]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    return df


def feature_engineering(df, is_train=True):
    """
    Run all feature engineering steps.
    """
    if 'date' in df.columns:
        df = create_date_features(df, date_col='date')
    df = compute_max_discount(df)
    
    # Only create lag features for training data when target is available
    if is_train and 'sales' in df.columns:
        df = create_lag_features(df, group_col='unique_id', target_col='sales', lags=[7, 14])
    
    # Encode categorical features (e.g., warehouse and category names)
    categorical_cols = ['warehouse', 'L1_category_name_en', 'L2_category_name_en', 
                        'L3_category_name_en', 'L4_category_name_en']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes  # Convert to numerical codes
    
    gc.collect()
    return df


# -------------------------------
# 3. MODEL TRAINING & EVALUATION
# -------------------------------

def train_model(X_train, y_train, X_val, y_val):
    """
    Train a LightGBM regression model and plot training history.
    """
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    
    evals_result = {}
    evaluation_callback = lgb.record_evaluation(evals_result)
    
    model = lgb.train(
        params, 
        lgb_train, 
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100),
            evaluation_callback
        ]
    )
    
    # Plot training & validation loss
    plt.figure(figsize=(10, 5))
    epochs = len(evals_result['train']['l1'])
    plt.plot(range(epochs), evals_result['train']['l1'], label='Train MAE')
    plt.plot(range(epochs), evals_result['valid']['l1'], label='Validation MAE')
    plt.xlabel("Boosting Rounds")
    plt.ylabel("MAE")
    plt.title("Training & Validation MAE")
    plt.legend()
    plt.show()
    
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on validation data and produce various plots.
    """
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae:.4f}")
    
    # Plot Actual vs Predicted Sales
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.show()
    
    # Plot Residuals Distribution
    residuals = y_val - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.title("Residuals Distribution")
    plt.show()
    
    # Placeholder: ROC Curve and AUC (using binarized sales for demonstration)
    y_val_binary = (y_val > np.median(y_val)).astype(int)
    y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
    try:
        auc_score = roc_auc_score(y_val_binary, y_pred_binary)
        print(f"ROC AUC (placeholder for regression): {auc_score:.4f}")
    except Exception as e:
        print("ROC AUC not applicable for regression:", e)
    
    # Placeholder: Confusion Matrix (using binarized predictions)
    cm = confusion_matrix(y_val_binary, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Binarized Sales)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# -------------------------------
# 4. PREDICTION & SUBMISSION
# -------------------------------

def generate_submission(model, test_df, feature_cols, output_file='submission.csv'):
    """
    Use the trained model to predict on the test set and write out the submission file.
    Ensures the test set contains all feature columns used in training by adding any missing columns with default value 0.
    """
    # Ensure that the test DataFrame has all the feature columns
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0  # Default value for missing features
    
    X_test = test_df[feature_cols]
    test_df['sales_hat'] = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Create the submission id column by combining unique_id and formatted date
    test_df['date_str'] = test_df['date'].dt.strftime('%Y-%m-%d')
    test_df['id'] = test_df['unique_id'].astype(str) + "_" + test_df['date_str']
    
    submission = test_df[['id', 'sales_hat']].copy()
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved as {output_file}")


# -------------------------------
# 5. MAIN PIPELINE
# -------------------------------

def main():
    # Set the data directory path (adjust as needed)
    data_dir = '/kaggle/input/rohlik-sales-forecasting-challenge-v2'
    
    # Load data
    sales_train, sales_test, inventory, calendar, test_weights = load_data(data_dir)
    
    # Preprocess data (convert dates where available and fill missing values)
    sales_train = preprocess_data(sales_train, date_col='date')
    sales_test  = preprocess_data(sales_test, date_col='date')
    calendar    = preprocess_data(calendar, date_col='date')
    inventory   = preprocess_data(inventory, date_col='date')
    
    # Merge calendar and inventory info into training data
    train_df = merge_data(sales_train, calendar, inventory)
    
    # Feature engineering for training data (lag features are created only if target 'sales' exists)
    train_df = feature_engineering(train_df, is_train=True)
    
    # Merge and feature engineer for test data (note: 'sales' and 'availability' are missing in test)
    test_df = merge_data(sales_test, calendar, inventory)
    test_df = feature_engineering(test_df, is_train=False)
    
    # Define features and target – exclude columns not used for training
    drop_cols = ['sales', 'holiday_name', 'name']
    feature_cols = [col for col in train_df.columns if col not in drop_cols + ['date']]
    
    # Prepare training data with a random train/validation split
    X = train_df[feature_cols]
    y = train_df['sales']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate the model
    evaluate_model(model, X_val, y_val)
    
    # (Optional) Compute Weighted MAE if desired by merging predictions with test_weights.
    
    # Generate the submission file
    generate_submission(model, test_df, feature_cols, output_file='submission.csv')
    
    gc.collect()


if __name__ == '__main__':
    main()


|

