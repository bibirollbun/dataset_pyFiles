import numpy as np
import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb # Ensure this is installed: !pip install xgboost

warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

# --- Data Exploration (Your existing excellent work) ---
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)

print("\nMissing values in train_df:")
print(train_df.isnull().sum())

sns.heatmap(train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].isnull(), cbar=False)
plt.title("Missing Targets in Train Set")
plt.show()

for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    plt.figure()
    train_df[col].dropna().hist(bins=30)
    plt.title(col)
    plt.show()

# --- Advanced Feature Engineering (TF-IDF Only) ---

# Define the target columns
TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Pre-processing SMILES: Fill potential NaNs or empty strings
train_df['SMILES'] = train_df['SMILES'].fillna('').astype(str)
test_df['SMILES'] = test_df['SMILES'].fillna('').astype(str)


# Initialize TF-IDF Vectorizer with more aggressive parameter tuning
tfidf = TfidfVectorizer(analyzer='char',
                        ngram_range=(1, 7), # Increased range
                        max_features=5000, # Increased features
                        min_df=3,          # Keep this to filter noise
                        token_pattern=r'\[[^\]]+\]|\(\S+\)|\w|\S') # More comprehensive token pattern

# Fit TF-IDF on the combined SMILES data (train + test) for comprehensive vocabulary
combined_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']], axis=0).unique()
tfidf.fit(combined_smiles)

# Transform SMILES for both train and test
X_train_smiles_features = tfidf.transform(train_df['SMILES'])
X_test_smiles_features = tfidf.transform(test_df['SMILES'])

print(f"\nTF-IDF training features shape: {X_train_smiles_features.shape}")
print(f"TF-IDF test features shape: {X_test_smiles_features.shape}")


# --- Highly Tuned XGBoost Modeling with K-Fold CV ---

def train_and_predict_property_xgboost_optimized(target_col, train_features, train_data_target, test_features, n_splits=10):
    """
    Trains an XGBoostRegressor for a given target property using K-Fold Cross-Validation
    with highly optimized parameters and makes predictions.
    """
    print(f"\n--- Training model for {target_col} ---")

    # Filter out rows where the target is missing
    # We apply the mask directly to the sparse matrix
    valid_indices_mask = train_data_target[target_col].notnull().values # Get boolean mask as numpy array
    X_target_train = train_features[valid_indices_mask] # Apply boolean indexing to sparse matrix
    y_target_train = train_data_target[target_col][valid_indices_mask].values

    if y_target_train.size == 0:
        print(f"No non-null data for {target_col}. Skipping prediction.")
        return np.zeros(test_features.shape[0])

    # K-Fold Cross-Validation setup: Increased folds for more robust evaluation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_preds_folds = []
    fold_rmses = []

    # Ensure X_target_train is a CSR matrix for efficient row slicing
    # It should already be, but this makes sure for safety.
    X_target_train_csr = X_target_train.tocsr()


    for fold, (train_index, val_index) in enumerate(kf.split(X_target_train_csr, y_target_train)):
        print(f"  Fold {fold+1}/{n_splits} for {target_col}")
        # Direct indexing of CSR matrix for rows
        X_train_fold, X_val_fold = X_target_train_csr[train_index], X_target_train_csr[val_index]
        y_train_fold, y_val_fold = y_target_train[train_index], y_target_train[val_index]

        # XGBoost Model Initialization and Training with aggressive tuning
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist', # Use 'gpu_hist' if GPU is enabled and installed
            eval_metric='rmse',
            reg_alpha=0.2,
            reg_lambda=0.2,
            gamma=0.05,
            min_child_weight=1,
        )

        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  early_stopping_rounds=150,
                  verbose=False)

        # Predict using the best iteration found by early stopping
        # Check if best_iteration attribute exists before using it
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            best_iteration_preds = model.predict(X_val_fold, iteration_range=(0, model.best_iteration))
        else: # Fallback if early stopping didn't trigger
            best_iteration_preds = model.predict(X_val_fold)

        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, best_iteration_preds))
        fold_rmses.append(fold_rmse)
        print(f"    Fold {fold+1} Validation RMSE: {fold_rmse:.4f}")

        # Predict on the actual test set for this fold using the best iteration
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            test_preds_folds.append(model.predict(test_features, iteration_range=(0, model.best_iteration)))
        else: # Fallback
            test_preds_folds.append(model.predict(test_features))


    avg_rmse = np.mean(fold_rmses)
    print(f"Average K-Fold Validation RMSE for {target_col}: {avg_rmse:.4f}")

    final_test_preds = np.mean(test_preds_folds, axis=0)
    return final_test_preds

# --- Generate predictions for all target columns ---
submission = pd.DataFrame({'id': test_df['id']})

for col in TARGET_COLS:
    submission[col] = train_and_predict_property_xgboost_optimized(col, X_train_smiles_features, train_df, X_test_smiles_features)

# Ensure no negative predictions for properties that should be positive
for col in ['FFV', 'Density', 'Tg', 'Tc', 'Rg']:
    submission[col] = submission[col].apply(lambda x: max(x, 0))

# Save to CSV
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSubmission file created successfully!")
print(submission.head())




