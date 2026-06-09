# Core libraries for data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Model libraries
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

# Machine learning libraries from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set some visual styles for our plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


TRAIN_FILE_PATH = "/kaggle/input/playground-series-s5e9/train.csv"
TEST_FILE_PATH = "/kaggle/input/playground-series-s5e9/test.csv"
SAMPLE_SUBMISSION_FILE_PATH = "/kaggle/input/playground-series-s5e9/sample_submission.csv"



# Load the datasets
try:
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)
    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE_PATH)
except FileNotFoundError:
    print("Ensure train.csv, test.csv, and sample_submission.csv are in the same directory.")
    # In a Kaggle environment, the paths would be /kaggle/input/<competition-folder>/...
    # For now, let's create dummy dataframes if files are not found.
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()


# Let's get a first look at our training data
print("Training Data Shape:", train_df.shape)
print("\nFirst 5 Rows of Training Data:")
print(train_df.head())

print("\nTraining Data Info:")
train_df.info()


# Plot the distribution of the target variable
sns.histplot(train_df['BeatsPerMinute'], kde=True, bins=50)
plt.title('Distribution of BeatsPerMinute (BPM)')
plt.xlabel('BPM')
plt.ylabel('Frequency')
plt.show()

# Get summary statistics
print(train_df['BeatsPerMinute'].describe())


# Calculate the correlation matrix
corr_matrix = train_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# Define features (X) and target (y)
features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 
            'TrackDurationMs', 'Energy']
target = 'BeatsPerMinute'

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Make predictions on the validation set
preds_val = model_lr.predict(X_val)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, preds_val))
print(f"Baseline Linear Regression Validation RMSE: {rmse:.4f}")


# Predict on the actual test data
test_predictions = model_lr.predict(X_test)

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': test_predictions})

# Save the submission file
submission_df.to_csv('submission_baseline_lr.csv', index=False)
# submission_df.to_csv('submission.csv', index=False)

print("Baseline submission file created successfully!")
print(submission_df.head())


import lightgbm as lgb

# Initialize and train the LightGBM model
# We use the same data split as before
model_lgb = lgb.LGBMRegressor(random_state=42)
model_lgb.fit(X_train, y_train)

# Make predictions on the validation set
preds_val_lgb = model_lgb.predict(X_val)

# Calculate RMSE
rmse_lgb = np.sqrt(mean_squared_error(y_val, preds_val_lgb))
print(f"LightGBM Validation RMSE (default params): {rmse_lgb:.4f}")


# Create new interaction features for both train and test sets
def create_features(df):
    df['Energy_Loudness_Interaction'] = df['Energy'] * df['AudioLoudness']
    df['Acoustic_Energy_Ratio'] = df['AcousticQuality'] / (df['Energy'] + 1e-6) # Add small epsilon to avoid division by zero
    return df

X_fe = create_features(X.copy())
X_test_fe = create_features(X_test.copy())

# New feature list
features_fe = list(X_fe.columns)
print("Features with new additions:", features_fe)

# Split the data again with the new features
X_train_fe, X_val_fe, y_train_fe, y_val_fe = train_test_split(X_fe, y, test_size=0.2, random_state=42)

# Train a new LGBM model on the engineered features
model_lgb_fe = lgb.LGBMRegressor(random_state=42)
model_lgb_fe.fit(X_train_fe, y_train_fe)

# Evaluate the new model
preds_val_lgb_fe = model_lgb_fe.predict(X_val_fe)
rmse_lgb_fe = np.sqrt(mean_squared_error(y_val_fe, preds_val_lgb_fe))
print(f"LightGBM with Feature Engineering Validation RMSE: {rmse_lgb_fe:.4f}")


# Predict on the test set with the feature-engineered model
test_predictions_lgb = model_lgb_fe.predict(X_test_fe)

# Create the submission file
submission_df_lgb = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': test_predictions_lgb})
# submission_df_lgb.to_csv('submission.csv', index=False)
submission_df_lgb.to_csv('submission_lgbm_fe.csv', index=False)
print("LGBM with Feature Engineering submission file created successfully!")


from sklearn.model_selection import KFold

# Prepare data and model
X_final = X_fe.copy()
y_final = y.copy()
X_test_final = X_test_fe.copy()

# K-Fold setup
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(X_final.shape[0])
sub_preds = np.zeros(X_test_final.shape[0])

# LightGBM parameters (we can tune these later)
lgb_params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}


# K-Fold training loop
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_final, y_final)):
    X_train_fold, y_train_fold = X_final.iloc[train_idx], y_final.iloc[train_idx]
    X_valid_fold, y_valid_fold = X_final.iloc[valid_idx], y_final.iloc[valid_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_valid_fold, y_valid_fold)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds[valid_idx] = model.predict(X_valid_fold)
    sub_preds += model.predict(X_test_final) / folds.n_splits

cv_rmse = np.sqrt(mean_squared_error(y_final, oof_preds))
print(f"K-Fold CV LightGBM RMSE: {cv_rmse:.4f}")


# This is a conceptual example. You would need to run the full training loops
# for XGBoost and CatBoost to get their respective predictions.

# Let's assume we have these prediction arrays after running CV for each model:
sub_preds_lgb = sub_preds # From our LGBM run
# sub_preds_xgb = ... # Placeholder for XGBoost predictions
# sub_preds_cat = ... # Placeholder for CatBoost predictions

# Simple average blend (a good starting point)
# final_predictions = (sub_preds_lgb + sub_preds_xgb + sub_preds_cat) / 3

# For now, let's just use our robust LGBM predictions for the final submission
final_predictions = sub_preds_lgb


# Create the final submission file
final_submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': final_predictions})
final_submission_df.to_csv('submission_final_lgbm_cv.csv', index=False)
# final_submission_df.to_csv('submission.csv', index=False)
print("Final CV-based submission file created successfully!")
print(final_submission_df.head())


# K-Fold setup
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)


print("--- Training LightGBM ---")

# Setup arrays for predictions
oof_preds_lgb = np.zeros(X.shape[0])
sub_preds_lgb = np.zeros(X_test.shape[0])

# LightGBM parameters
lgb_params = {
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
    'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
    'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    'boosting_type': 'gbdt',
}

# K-Fold training loop
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds_lgb[valid_idx] = model.predict(X_valid)
    sub_preds_lgb += model.predict(X_test) / folds.n_splits

cv_rmse_lgb = np.sqrt(mean_squared_error(y, oof_preds_lgb))
print(f"LGBM CV Score: {cv_rmse_lgb:.4f}")


print("\n--- Training XGBoost ---")

# Setup arrays for predictions
oof_preds_xgb = np.zeros(X.shape[0])
sub_preds_xgb = np.zeros(X_test.shape[0])

# XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'eta': 0.01,
    'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'seed': 42, 'n_jobs': -1
}

# K-Fold training loop
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = xgb.XGBRegressor(**xgb_params, n_estimators=2000, early_stopping_rounds=100)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              verbose=False)

    oof_preds_xgb[valid_idx] = model.predict(X_valid)
    sub_preds_xgb += model.predict(X_test) / folds.n_splits

cv_rmse_xgb = np.sqrt(mean_squared_error(y, oof_preds_xgb))
print(f"XGBoost CV Score: {cv_rmse_xgb:.4f}")


print("\n--- Training CatBoost ---")

# Setup arrays for predictions
oof_preds_cat = np.zeros(X.shape[0])
sub_preds_cat = np.zeros(X_test.shape[0])

# CatBoost parameters
cat_params = {
    'loss_function': 'RMSE', 'eval_metric': 'RMSE', 'iterations': 2000,
    'learning_rate': 0.05, 'depth': 6, 'random_seed': 42,
    'verbose': 0, 'early_stopping_rounds': 100
}

# K-Fold training loop
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = cat.CatBoostRegressor(**cat_params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)])

    oof_preds_cat[valid_idx] = model.predict(X_valid)
    sub_preds_cat += model.predict(X_test) / folds.n_splits

cv_rmse_cat = np.sqrt(mean_squared_error(y, oof_preds_cat))
print(f"CatBoost CV Score: {cv_rmse_cat:.4f}")


print("\n--- Blending Models ---")

# Simple average blend
# final_predictions = (sub_preds_lgb + sub_preds_xgb + sub_preds_cat) / 3.0

# Weighted average blend based on CV scores (invert RMSE so lower is better)
lgb_weight = 1 / cv_rmse_lgb
xgb_weight = 1 / cv_rmse_xgb
cat_weight = 1 / cv_rmse_cat

total_weight = lgb_weight + xgb_weight + cat_weight

final_predictions = (
    (lgb_weight / total_weight) * sub_preds_lgb +
    (xgb_weight / total_weight) * sub_preds_xgb +
    (cat_weight / total_weight) * sub_preds_cat
)

print("Blending complete.")


# Create the final ensemble submission DataFrame
submission_ensemble = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': final_predictions})

# Save the submission file
# submission_ensemble.to_csv('submission_ensemble.csv', index=False)
submission_ensemble.to_csv('submission.csv', index=False)

print("\nFinal ensemble submission file created successfully!")
print(submission_ensemble.head())

