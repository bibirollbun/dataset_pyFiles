import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# --- 0. Data Preparation ---
print("--- 0. Starting Data Preparation ---")

# Load datasets
print("Loading datasets...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
augmentation_df = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')

# --- FIX: Map target variable in augmentation data ---
# Map 'no' to 0 and 'yes' to 1 to match the main dataset's format
print("Mapping target variable 'y' in the augmentation dataset ('no'->0, 'yes'->1)...")
augmentation_df['y'] = augmentation_df['y'].map({'no': 0, 'yes': 1})

# Rename augmentation data columns to match the main dataset
print("Renaming columns in the augmentation dataset...")
# The target column 'y' already matches.
augmentation_df.columns = train_df.columns.drop('id')

print(f"Main training data rows: {len(train_df)}")
print(f"Augmentation data rows: {len(augmentation_df)}")
print(f"Test data rows: {len(test_df)}")
print("--- Data Preparation Complete ---")
print("-" * 50)


# --- 1. Feature Engineering and Encoding ---
print("--- 1. Starting Feature Engineering & Encoding ---")

# Store IDs and target variable
train_ids = train_df['id']
test_ids = test_df['id']
y_main = train_df['y']
y_aug = augmentation_df['y']

# Drop unnecessary columns
train_df = train_df.drop(columns=['id', 'y'])
test_df = test_df.drop(columns=['id'])
augmentation_df = augmentation_df.drop(columns=['y'])

# Combine all dataframes for consistent encoding
all_data = pd.concat([train_df, test_df, augmentation_df], ignore_index=True)

# Identify categorical features
categorical_features = all_data.select_dtypes(include=['object']).columns
print(f"Applying one-hot encoding to: {list(categorical_features)}")

# Apply one-hot encoding
all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, dummy_na=False)

# Separate back into their original sets
X_main = all_data_encoded.iloc[:len(train_df)]
X_test = all_data_encoded.iloc[len(train_df):len(train_df) + len(test_df)]
X_aug = all_data_encoded.iloc[len(train_df) + len(test_df):]

print(f"Main training features shape: {X_main.shape}")
print(f"Augmentation features shape: {X_aug.shape}")
print(f"Test features shape: {X_test.shape}")
print("--- Feature Engineering Complete ---")
print("-" * 50)


# --- 2. Comparative Cross-Validation ---
print("--- 2. Starting Comparative Cross-Validation ---")

N_SPLITS = 10
MAX_AUG_REPEATS = 4
results = []

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'num_leaves': 20,
    'max_depth': 5,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}

# Loop to test different numbers of augmentation repeats
for n_repeats in range(MAX_AUG_REPEATS + 1):
    print(f"\n===== Running CV with augmentation data repeated {n_repeats} time(s) =====")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X_main))
    test_preds = np.zeros(len(X_test))

    # The CV split is ALWAYS based on the main training data
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_main, y_main)):
        
        # --- Data setup for the fold ---
        X_train_fold, y_train_fold = X_main.iloc[train_idx], y_main.iloc[train_idx]
        X_val_fold, y_val_fold = X_main.iloc[val_idx], y_main.iloc[val_idx]

        # Augment the training data for this fold
        if n_repeats > 0:
            X_train_augmented = pd.concat([X_train_fold] + [X_aug] * n_repeats, ignore_index=True)
            y_train_augmented = pd.concat([y_train_fold] + [y_aug] * n_repeats, ignore_index=True)
        else:
            X_train_augmented, y_train_augmented = X_train_fold, y_train_fold

        # --- Model training ---
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train_augmented, y_train_augmented,
                  eval_set=[(X_val_fold, y_val_fold)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        # --- Prediction ---
        val_preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    # --- Store results for this n_repeats run ---
    overall_oof_auc = roc_auc_score(y_main, oof_preds)
    print(f"Overall OOF AUC for {n_repeats} repeats: {overall_oof_auc:.6f}")
    results.append({
        'repeats': n_repeats,
        'oof_auc': overall_oof_auc,
        'test_predictions': test_preds
    })

print("\n--- Cross-Validation Comparison Complete ---")
print("-" * 50)


# --- 3. Final Results Summary and Submission ---
print("--- 3. Final Results and Submission ---")

# Find the best result based on OOF AUC
best_result = max(results, key=lambda x: x['oof_auc'])

print("Summary of OOF AUC by number of augmentation repeats:")
for res in results:
    highlight = "<-- BEST" if res['repeats'] == best_result['repeats'] else ""
    print(f"  Repeats: {res['repeats']}, OOF AUC: {res['oof_auc']:.6f} {highlight}")

print(f"\nSelecting predictions from the best run ({best_result['repeats']} repeats).")

# Create submission file using the best predictions
submission_df = pd.DataFrame({'id': test_ids, 'y': best_result['test_predictions']})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' has been saved successfully.")
print("--- All processes completed ---")

