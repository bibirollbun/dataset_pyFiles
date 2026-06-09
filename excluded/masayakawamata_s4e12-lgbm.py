import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gc

# --- 1. Data Loading and Initial Preparation ---

# Load the datasets
# Make sure to adjust the path if you are not running this in a Kaggle environment
try:
    train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
except FileNotFoundError:
    print("CSV files not found. Creating dummy data for demonstration.")
    # Create a dummy dataset if the original files are not available
    def create_dummy_data(n_samples, name):
        data = {}
        data['id'] = range(n_samples)
        CATS_dummy = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location', 'Policy Type', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']
        NUMS_dummy = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']
        for col in CATS_dummy:
            data[col] = np.random.choice([f'{col}_A', f'{col}_B', f'{col}_C'], n_samples)
        for col in NUMS_dummy:
            data[col] = np.random.randint(0, 100, n_samples)
        data['Policy Start Date'] = pd.to_datetime(pd.Timestamp('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365*2, n_samples), 'D'))
        if name == 'train':
            data['Premium Amount'] = np.random.rand(n_samples) * 1000 + 50
        return pd.DataFrame(data)
    train = create_dummy_data(1000, 'train')
    test = create_dummy_data(500, 'test')

print('Train Shape', train.shape)
print('Test Shape', test.shape)

# Define the target variable
TARGET = 'Premium Amount'

# Apply log1p transformation for RMSLE optimization
print(f"\nApplying log1p transformation to the target variable: {TARGET}")
train[TARGET] = np.log1p(train[TARGET])

# Function to decompose the date feature
def date_feat(df):
  """Decomposes the 'Policy Start Date' into time-based features."""
  df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
  df['Year'] = df['Policy Start Date'].dt.year
  df['Month'] = df['Policy Start Date'].dt.month
  df['Day'] = df['Policy Start Date'].dt.day
  df['Dayofweek'] = df['Policy Start Date'].dt.dayofweek
  return df

# Apply the date feature decomposition
train = date_feat(train)
test = date_feat(test)

# --- 2. Feature and Model Preparation ---

# Identify categorical and numerical features
CATS = train.select_dtypes(include='object').columns.tolist()
# All other numeric columns besides 'id' and the target are features
NUMS = [col for col in train.select_dtypes(include='number').columns.tolist() if col not in ['id', TARGET]]

# Convert categorical features to pandas 'category' dtype for LightGBM
for col in CATS:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# Define the complete feature set for this basic model
model_features = NUMS + CATS
print(f"\nUsing {len(model_features)} basic features for the model.")
print("Features:", model_features)


# --- 3. Cross-Validation, Modeling, and Prediction ---

# KFold setup
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Placeholders for predictions
oof_preds = np.zeros(train.shape[0])
test_preds = np.zeros(test.shape[0])
best_iterations = []

# LGBM parameters
lgbm_params = {
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
    'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
    'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    'boosting_type': 'gbdt',
}

# --- Start CV Loop ---
print(f"\nStarting {NFOLDS}-Fold CV for the basic model...")
for fold, (train_index, val_index) in enumerate(kf.split(train, train[TARGET])):
    print(f"===== FOLD {fold+1} =====")
    
    X_train, X_val = train.iloc[train_index], train.iloc[val_index]
    y_train, y_val = train[TARGET].iloc[train_index], train[TARGET].iloc[val_index]
    
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train[model_features], y_train,
              eval_set=[(X_val[model_features], y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)],
              categorical_feature=CATS) # Inform LGBM about categorical features
    
    val_preds = model.predict(X_val[model_features])
    oof_preds[val_index] = val_preds
    test_preds += model.predict(test[model_features]) / NFOLDS
    best_iterations.append(model.best_iteration_)
    
    # Calculate and display RMSLE for the fold
    fold_rmsle = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")
    print(f"Fold {fold+1} Best Iteration: {model.best_iteration_}")
    
    del X_train, X_val, y_train, y_val, model
    gc.collect()

# --- 4. Save Results ---

# Apply expm1 to convert log-predictions back to the original scale
oof_df = pd.DataFrame({'id': train['id'], TARGET: np.expm1(oof_preds)})
oof_filename = 'oof_basic_lgbm.csv'
oof_df.to_csv(oof_filename, index=False)
print(f"\nOOF predictions for the basic model saved to {oof_filename}")

test_df = pd.DataFrame({'id': test['id'], TARGET: np.expm1(test_preds)})
test_filename = 'test_basic_lgbm.csv'
test_df.to_csv(test_filename, index=False)
print(f"Test predictions for the basic model saved to {test_filename}")




