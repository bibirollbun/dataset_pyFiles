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
    train = create_dummy_data(10000, 'train') # Increased data size for testing
    test = create_dummy_data(5000, 'test')


print('Train Shape', train.shape)
print('Test Shape', test.shape)

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

# Apply the date feature decomposition
train = date_feat(train)
test = date_feat(test)

# Identify categorical and numerical features
CATS = train.select_dtypes(include='object').columns.tolist()
NUMS = [col for col in train.select_dtypes(include='number').columns.tolist() if col not in ['id', TARGET]]
print(len(CATS), 'Categoricals:',CATS, '\n')
print(len(NUMS), 'Numericals:',NUMS, '\n')
print('-->', len(NUMS+CATS), 'Features')


# --- 2. Encoding Functions ---

def target_encode(train_df, valid_df, col, target=TARGET, kfold=5, smooth=20):
    col_name = '_'.join(col) if isinstance(col, list) else col
    
    # --- Create OOF predictions for the training part ---
    # Note: This implementation of target encoding is for the full dataset,
    # not within the CV fold's training set. A true OOF TE would be more complex.
    train_df['kfold'] = ((train_df.index) % kfold)
    oof_preds = pd.Series(index=train_df.index, dtype=float)
    
    for i in range(kfold):
        train_fold = train_df[train_df['kfold'] != i]
        val_fold = train_df[train_df['kfold'] == i]
        
        global_mean = train_fold[target].mean()
        agg = train_fold.groupby(col)[target].agg(['mean', 'count'])
        smoothed_mean = (agg['mean'] * agg['count'] + global_mean * smooth) / (agg['count'] + smooth)
        
        oof_preds.loc[val_fold.index] = val_fold[col].map(smoothed_mean).fillna(global_mean)
        
    train_df[f'TE_{col_name}'] = oof_preds
    train_df = train_df.drop('kfold', axis=1)

    # --- Create mapping for the validation/test set ---
    global_mean_full = train_df[target].mean()
    agg_full = train_df.groupby(col)[target].agg(['mean', 'count'])
    smoothed_mean_full = (agg_full['mean'] * agg_full['count'] + global_mean_full * smooth) / (agg_full['count'] + smooth)
    
    valid_df[f'TE_{col_name}'] = valid_df[col].map(smoothed_mean_full).fillna(global_mean_full)
    
    return train_df, valid_df

def count_encode(train_df, valid_df, col):
    col_name = '_'.join(col) if isinstance(col, list) else col
    counts = train_df[col].value_counts()
    train_df[f'CE_{col_name}'] = train_df[col].map(counts).fillna(0)
    valid_df[f'CE_{col_name}'] = valid_df[col].map(counts).fillna(0)
    return train_df, valid_df

# --- 3 & 4. Memory-Efficient Feature Engineering and CV ---

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

features_to_combine = NUMS + CATS

# --- Start CV Loop ---
print(f"\nStarting {NFOLDS}-Fold CV with on-the-fly feature engineering...")
for fold, (train_index, val_index) in enumerate(kf.split(train, train[TARGET])):
    print(f"===== FOLD {fold+1} =====")
    
    X_train, X_val = train.iloc[train_index].copy(), train.iloc[val_index].copy()
    y_train, y_val = train[TARGET].iloc[train_index], train[TARGET].iloc[val_index]
    test_fold = test.copy()

    encoded_cols = []
    
    # On-the-fly feature generation and encoding
    print("Generating and encoding features for this fold...")
    for col1, col2 in combinations(features_to_combine, 2):
        feature_name = f'{col1}_{col2}'
        
        # Create temporary interaction feature
        X_train[feature_name] = X_train[col1].astype(str) + '_' + X_train[col2].astype(str)
        X_val[feature_name] = X_val[col1].astype(str) + '_' + X_val[col2].astype(str)
        test_fold[feature_name] = test_fold[col1].astype(str) + '_' + test_fold[col2].astype(str)
        
        # Apply encoding
        X_train, X_val = count_encode(X_train, X_val, feature_name)
        _, test_fold = count_encode(X_train, test_fold, feature_name)
        
        X_train, X_val = target_encode(X_train, X_val, feature_name, target=TARGET)
        _, test_fold = target_encode(X_train, test_fold, feature_name, target=TARGET)

        # Drop the temporary interaction column to save memory
        X_train = X_train.drop(feature_name, axis=1)
        X_val = X_val.drop(feature_name, axis=1)
        test_fold = test_fold.drop(feature_name, axis=1)
        
        # Keep track of the newly created encoded columns
        encoded_cols.extend([f'CE_{feature_name}', f'TE_{feature_name}'])
        
    print(f"Finished encoding. Total encoded features: {len(encoded_cols)}")
    
    model_features = NUMS + encoded_cols
    
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train[model_features], y_train,
              eval_set=[(X_val[model_features], y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    val_preds = model.predict(X_val[model_features])
    oof_preds[val_index] = val_preds
    test_preds += model.predict(test_fold[model_features]) / NFOLDS
    best_iterations.append(model.best_iteration_)
    
    # --- [MODIFIED] Calculate and display RMSLE for the fold ---
    # RMSE on log-transformed values is equivalent to RMSLE on original values
    fold_rmsle = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")
    print(f"Fold {fold+1} Best Iteration: {model.best_iteration_}")
    
    # Clean up memory
    del X_train, X_val, test_fold, model
    gc.collect()


# --- 5. Final Model Training and Prediction ---

final_iterations = int(np.mean(best_iterations) * 1.2)
print(f"\nAverage best iteration: {np.mean(best_iterations):.0f}")
print(f"Training final model with {final_iterations} iterations...")

# We need to re-create the encoded features on the full dataset for the final model
train_full = train.copy()
test_full = test.copy()
encoded_cols_final = []

print("Generating and encoding features for the final model...")
for col1, col2 in combinations(features_to_combine, 2):
    feature_name = f'{col1}_{col2}'

    # Create temporary interaction feature
    train_full[feature_name] = train_full[col1].astype(str) + '_' + train_full[col2].astype(str)
    test_full[feature_name] = test_full[col1].astype(str) + '_' + test_full[col2].astype(str)

    # Apply encoding (using the whole training data)
    train_full, test_full = count_encode(train_full, test_full, feature_name)
    train_full, test_full = target_encode(train_full, test_full, feature_name, target=TARGET)

    # Drop temporary column
    train_full = train_full.drop(feature_name, axis=1)
    test_full = test_full.drop(feature_name, axis=1)

    encoded_cols_final.extend([f'CE_{feature_name}', f'TE_{feature_name}'])

print(f"Finished encoding. Total encoded features: {len(encoded_cols_final)}")

final_model_features = NUMS + encoded_cols_final
lgbm_params['n_estimators'] = final_iterations

final_model = lgb.LGBMRegressor(**lgbm_params)
final_model.fit(train_full[final_model_features], train_full[TARGET])
final_test_predictions = final_model.predict(test_full[final_model_features])


# --- 6. Save Results ---

# --- [MODIFIED] Inverse transform predictions before saving ---
# Apply expm1 to convert log-predictions back to the original scale
oof_df = pd.DataFrame({'id': train['id'], TARGET: np.expm1(oof_preds)})
oof_filename = 'oof_2way_te_lgbm_rmsle.csv'
oof_df.to_csv(oof_filename, index=False)
print(f"\nOOF predictions saved to {oof_filename}")

test_df = pd.DataFrame({'id': test['id'], TARGET: np.expm1(final_test_predictions)})
test_filename = 'test_2way_te_lgbm_rmsle.csv'
test_df.to_csv(test_filename, index=False)
print(f"Test predictions saved to {test_filename}")




