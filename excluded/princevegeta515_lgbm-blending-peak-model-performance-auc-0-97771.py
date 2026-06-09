# BASELINE MODEL


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import gc

# --- Phase 1: Data Loading and Preparation ---
print("--- Loading Data ---")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_ids = test_df['id']

# --- Phase 2: Feature Engineering (from the high-scoring notebook) ---
print("--- Applying Feature Engineering ---")

def new_feats(df):
    """
    This function replicates the exact feature engineering from the 0.975+ notebook.
    """
    df = df.copy()
    # Binary flags based on thresholds
    df['balance_posi'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['long_duration'] = (df['duration'] >= 360).astype(int)
    df['campaign_multi'] = (df['campaign'] >= 2).astype(int)
    df['is_first_contact'] = (df['campaign'] == 1).astype(int)
    df['high_campaign'] = (df['campaign'] >= 3).astype(int)
    
    # Age binning
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 97], 
                             labels=['young', 'adult', 'middle', 'senior', 'elderly'])
    
    # Mathematical transformations
    df['log_duration'] = np.log1p(df['duration'])
    df['sqrt_duration'] = np.sqrt(df['duration'])
    df['log_campaign'] = np.log1p(df['campaign'])
    df['sqrt_age'] = df['age'] ** 2  # Typo in original notebook, this is squared_age
    df['cubed_age'] = df['age'] ** 3
    df['log_age'] = np.log1p(df['age'])
    
    return df

# Apply the feature engineering
train_featured = new_feats(train_df)
test_featured = new_feats(test_df)

# Prepare data for LightGBM
y_train = train_featured['y']
X_train = train_featured.drop(['id', 'y'], axis=1)
X_test = test_featured.drop('id', axis=1)

# --- Phase 3: Label Encoding for Categorical Features ---
print("--- Encoding Categorical Features ---")
# Identify categorical columns (including the new 'age_group')
object_cols = X_train.select_dtypes(include=['object', 'category']).columns

for col_name in object_cols:
    le = LabelEncoder()
    # Fit on combined train and test data to handle all possible categories
    combined_data = pd.concat([X_train[col_name], X_test[col_name]]).astype(str)
    le.fit(combined_data)
    
    # Transform train and test data
    X_train[col_name] = le.transform(X_train[col_name].astype(str))
    X_test[col_name] = le.transform(X_test[col_name].astype(str))

print("Data Preparation Complete.")

# --- Phase 4: Model Training and Prediction ---
print("\n--- Starting Model Training (10-Fold CV) ---")

N_SPLITS = 10
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_predictions = np.zeros(len(X_test))
oof_predictions = np.zeros(len(X_train))

# Use the exact parameters from the high-scoring notebook
lgbm_params = {
    'random_state': 42,
    'verbosity': -1,
    'n_estimators': 25000,
    'learning_rate': 0.05,
    'min_child_samples': 13,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'num_leaves': 100,
    'max_depth': 10,
    'max_bin': 4840,  # The key parameter identified
    'reg_alpha': 0.8,
    'reg_lambda': 3,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"--- Processing Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

    # Initialize and train the model
    model = lgb.LGBMClassifier(**lgbm_params)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=500)
    ]
    
    model.fit(
        X_train_fold, 
        y_train_fold, 
        eval_set=[(X_val_fold, y_val_fold)], 
        callbacks=callbacks
    )
    
    # Store predictions
    test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
    oof_predictions[val_idx] = model.predict_proba(X_val_fold)[:, 1]
    
    del X_train_fold, y_train_fold, X_val_fold, y_val_fold, model
    gc.collect()

print("\n--- Model Training Complete ---")

# --- Phase 5: Create Submission File ---
print("--- Creating Submission File ---")
submission_df = pd.DataFrame({'id': test_ids, 'y': test_predictions})
submission_df.to_csv('submission_lgbm_replication.csv', index=False)

print("\nSubmission file 'submission_lgbm_replication.csv' created successfully!")
print("Top 5 rows:")
print(submission_df.head())


# BLENDED MULTIPLE MODELS
import pandas as pd
from scipy.stats import rankdata

# --- Step 1: Load the Submission Files ---
print("Loading submission files...")

# IMPORTANT: Make sure the 'id' column is the same in both files.
sub_A = pd.read_csv('C:\\Users\\tirth\\OneDrive\\Desktop\\Code Playground\\kaggle\\submission_rank_blend.csv')
sub_B = pd.read_csv('C:\\Users\\tirth\\OneDrive\\Desktop\\Code Playground\\kaggle\\submission.csv')

# Weighted Average Blend
print("Creating Weighted Average Blend...")
# --- TUNE THESE WEIGHTS ---
# Give more weight to the submission with the higher score.
weight_A = 0.30  # Your submission's weight
weight_B = 0.70  # Their submission's weight

weighted_blend = sub_A.copy()
weighted_blend['y'] = (sub_A['y'] * weight_A) + (sub_B['y'] * weight_B)
weighted_blend.to_csv('final_weighted_blend.csv', index=False)

# Rank Average Blend
print("Creating Rank Average Blend...")
rank_blend = sub_A.copy()

# Use the same weights as the weighted average
rank_blend['y'] = (rankdata(sub_A['y']) * weight_A) + (rankdata(sub_B['y']) * weight_B)

# Scale the result back to the 0-1 range
rank_blend['y'] = (rank_blend['y'] - rank_blend['y'].min()) / (rank_blend['y'].max() - rank_blend['y'].min())
rank_blend.to_csv('final_rank_blend.csv', index=False)

print("\nBlending complete! Two new submission files have been created:")
print("1. submission_weighted_blend.csv")
print("2. submission_rank_blend.csv")

# Display the head of one of the blended files to check
print("\nTop 5 rows of the Weighted Blend:")
print(weighted_blend.head())




