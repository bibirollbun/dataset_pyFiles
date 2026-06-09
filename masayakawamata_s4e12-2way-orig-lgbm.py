import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error # Import for RMSLE calculation
from itertools import combinations
import gc # Import the garbage collector

# --- 1. Data Loading and Initial Preparation ---

# Load the datasets
# Make sure to adjust the path if you are not running this in a Kaggle environment
try:
    train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
    # Load the external dataset
    orig = pd.read_csv('/kaggle/input/insurance-premium-prediction/Insurance Premium Prediction Dataset.csv')
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
        if name in ['train', 'orig']:
            data['Premium Amount'] = np.random.rand(n_samples) * 1000 + 50
        return pd.DataFrame(data)
    train = create_dummy_data(1000, 'train')
    test = create_dummy_data(500, 'test')
    orig = create_dummy_data(1200, 'orig')


print('Train Shape', train.shape)
print('Test Shape', test.shape)
print('Original Data Shape', orig.shape)


# Define the target variable
TARGET = 'Premium Amount'

# --- [MODIFIED] Log-transform the target variable for RMSLE optimization ---
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

# Apply the date feature decomposition to all datasets
train = date_feat(train)
test = date_feat(test)
orig = date_feat(orig)

# Identify categorical and numerical features
CATS = train.select_dtypes(include='object').columns.tolist()
NUMS = [col for col in train.select_dtypes(include='number').columns.tolist() if col not in ['id', TARGET]]
print(len(CATS), 'Categoricals:',CATS, '\n')
print(len(NUMS), 'Numericals:',NUMS, '\n')
print('-->', len(NUMS+CATS), 'Features')


# --- 2. Memory-Efficient Feature Engineering (Unchanged) ---

features_to_combine = NUMS + CATS
ext_te_features = []

print(f"\nGenerating External Target Encoding features on-the-fly...")
# Note: Feature engineering uses the original, non-transformed target from the 'orig' dataset
global_mean_orig = orig['Premium Amount'].mean()

# This loop now generates one interaction feature at a time, encodes it, and then deletes it.
for col1, col2 in combinations(features_to_combine, 2):
    feature_name = f'{col1}_{col2}'
    ext_te_name = f'EXT_TE_{feature_name}'
    
    # 1. Create temporary interaction feature on all three dataframes
    train[feature_name] = train[col1].astype(str) + '_' + train[col2].astype(str)
    test[feature_name] = test[col1].astype(str) + '_' + test[col2].astype(str)
    orig[feature_name] = orig[col1].astype(str) + '_' + orig[col2].astype(str)
    
    # 2. Calculate mean of target in original data
    agg = orig.groupby(feature_name)['Premium Amount'].agg('mean')
    
    # 3. Map the mean to train and test data to create the new feature
    train[ext_te_name] = train[feature_name].map(agg).fillna(global_mean_orig)
    test[ext_te_name] = test[feature_name].map(agg).fillna(global_mean_orig)
    ext_te_features.append(ext_te_name)
    
    # 4. Drop the temporary interaction column from all dataframes to save memory
    train = train.drop(feature_name, axis=1)
    test = test.drop(feature_name, axis=1)
    orig = orig.drop(feature_name, axis=1)

print(f"Finished generating {len(ext_te_features)} external TE features.")
# Explicitly call the garbage collector to free up memory
gc.collect()


# --- 3. Cross-Validation, Modeling, and Prediction ---

# KFold setup
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Placeholders for predictions (will store log-transformed values)
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

# Define the features to be used in the model
model_features = NUMS + ext_te_features

# --- Start CV Loop ---
print(f"\nStarting {NFOLDS}-Fold CV...")
for fold, (train_index, val_index) in enumerate(kf.split(train, train[TARGET])):
    print(f"===== FOLD {fold+1} =====")
    
    X_train, X_val = train.iloc[train_index], train.iloc[val_index]
    y_train, y_val = train[TARGET].iloc[train_index], train[TARGET].iloc[val_index]
    
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train[model_features], y_train,
              eval_set=[(X_val[model_features], y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    val_preds = model.predict(X_val[model_features])
    oof_preds[val_index] = val_preds
    test_preds += model.predict(test[model_features]) / NFOLDS
    best_iterations.append(model.best_iteration_)
    
    # --- [MODIFIED] Calculate and display RMSLE for the fold ---
    # RMSE on log-transformed values is equivalent to RMSLE on original values
    fold_rmsle = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")
    print(f"Fold {fold+1} Best Iteration: {model.best_iteration_}")

# --- 4. Final Model Training and Prediction ---

final_iterations = int(np.mean(best_iterations) * 1.2)
print(f"\nAverage best iteration: {np.mean(best_iterations):.0f}")
print(f"Training final model with {final_iterations} iterations...")

# Use the same feature set for the final model
final_model_features = NUMS + ext_te_features
lgbm_params['n_estimators'] = final_iterations

final_model = lgb.LGBMRegressor(**lgbm_params)
# Train on the full dataset with the log-transformed target
final_model.fit(train[final_model_features], train[TARGET])
# Final predictions will be on the log scale
final_test_predictions = final_model.predict(test[final_model_features])

# --- 5. Save Results ---

# --- [MODIFIED] Inverse transform predictions before saving ---
# Apply expm1 to convert log-predictions back to the original scale
oof_df = pd.DataFrame({'id': train['id'], TARGET: np.expm1(oof_preds)})
oof_filename = 'oof_2way_ext_te_lgbm_rmsle.csv'
oof_df.to_csv(oof_filename, index=False)
print(f"\nOOF predictions saved to {oof_filename}")

test_df = pd.DataFrame({'id': test['id'], TARGET: np.expm1(final_test_predictions)})
test_filename = 'test_2way_ext_te_lgbm_rmsle.csv'
test_df.to_csv(test_filename, index=False)
print(f"Test predictions saved to {test_filename}")




