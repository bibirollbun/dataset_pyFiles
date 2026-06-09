import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data - assuming 'full_train_data' and 'full_test_data' are available
# In a real scenario, you would have:
full_train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
full_test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# Prepare the data
X = full_train_data.drop(columns=['id', 'Fertilizer Name'])
y = full_train_data['Fertilizer Name']

# Prepare test data for final prediction
X_test = full_test_data.drop(columns=['id'])

# Preprocessing setup (fit on full training data, transform train/val/test)
standard_scaler = StandardScaler()
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
label_encoder = LabelEncoder()

# Fit encoders/scalers on the *entire* training data for consistency across folds and test set
# This prevents data leakage from validation sets into preprocessing steps.
num_columns = ['Phosphorous']
cat_columns = ['Soil Type', 'Crop Type']

X[num_columns] = standard_scaler.fit_transform(X[num_columns])
X_test[num_columns] = standard_scaler.transform(X_test[num_columns])

# Fit on all X for OrdinalEncoder
# It's crucial that the OrdinalEncoder sees all possible categories during fit
# to assign consistent integer mappings.
combined_cat_data = pd.concat([X[cat_columns], X_test[cat_columns]], axis=0)
ordinal_encoder.fit(combined_cat_data)

X[cat_columns] = ordinal_encoder.transform(X[cat_columns])
X_test[cat_columns] = ordinal_encoder.transform(X_test[cat_columns])

y_encoded = label_encoder.fit_transform(y)

# Convert all data to the correct format if it's not already
for col in X.columns:
    if X[col].dtype == 'float64':
        X[col] = X[col].astype('float32')
    elif X[col].dtype == 'int64':
        X[col] = X[col].astype('int32')

for col in X_test.columns:
    if X_test[col].dtype == 'float64':
        X_test[col] = X_test[col].astype('float32')
    elif X_test[col].dtype == 'int64':
        X_test[col] = X_test[col].astype('int32')


# --- Cross-validation setup ---
N_SPLITS = 5 # Number of folds for cross-validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Lists to store OOF (Out-Of-Fold) predictions and test predictions for ensembling
oof_preds_xgb = np.zeros((len(X), len(label_encoder.classes_)))
oof_preds_lgb = np.zeros((len(X), len(label_encoder.classes_)))
test_preds_xgb = np.zeros((len(X_test), len(label_encoder.classes_)))
test_preds_lgb = np.zeros((len(X_test), len(label_encoder.classes_)))

# Store models for potential inspection
xgb_models = []
lgb_models = []

# Model parameters
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'eval_metric': 'mlogloss',
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.9,
    'seed': 42,
    'n_estimators': 1000,
    'tree_method': 'hist',
    'early_stopping_rounds': 50,
    'n_jobs': -1 # Use all available cores
}

lgb_params = {
    'objective': 'multiclass',
    'num_class': len(label_encoder.classes_),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31, # Roughly equivalent to max_depth: 2^num_leaves > max_depth
    'colsample_bytree': 0.9, # Feature sampling
    'subsample': 0.9, # Data sampling
    'random_state': 42,
    'n_estimators': 1000,
    'early_stopping_round': 50,
    'n_jobs': -1 # Use all available cores
}

print("Starting cross-validation training...")

for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y_encoded[train_index], y_encoded[val_index]

    # --- XGBoost Training ---
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False) # Suppress verbose output
    xgb_models.append(xgb_model)

    oof_preds_xgb[val_index] = xgb_model.predict_proba(X_val_fold)
    test_preds_xgb += xgb_model.predict_proba(X_test) / N_SPLITS # Average test predictions

    # --- LightGBM Training ---
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  callbacks=[lgb.early_stopping(lgb_params['early_stopping_round'], verbose=False)]) # LightGBM callback
    lgb_models.append(lgb_model)

    oof_preds_lgb[val_index] = lgb_model.predict_proba(X_val_fold)
    test_preds_lgb += lgb_model.predict_proba(X_test) / N_SPLITS # Average test predictions

# --- Ensemble Predictions ---
# Simple averaging ensemble
# You can also explore weighted averaging (e.g., based on OOF performance)
# or stacking (training another model on oof_preds_xgb and oof_preds_lgb)
ensemble_oof_preds = (oof_preds_xgb + oof_preds_lgb) / 2
ensemble_test_preds = (test_preds_xgb + test_preds_lgb) / 2

# Convert probabilities to class labels for accuracy
ensemble_oof_pred_labels = np.argmax(ensemble_oof_preds, axis=1)
ensemble_test_pred_labels = np.argmax(ensemble_test_preds, axis=1)

# --- Evaluate Ensemble Performance ---
ensemble_accuracy = accuracy_score(y_encoded, ensemble_oof_pred_labels)
ensemble_logloss = log_loss(y_encoded, ensemble_oof_preds) # Use log_loss for probability output

print(f"\n--- Ensemble Results (Out-Of-Fold) ---")
print(f"Ensemble OOF Accuracy: {ensemble_accuracy:.4f}")
print(f"Ensemble OOF LogLoss: {ensemble_logloss:.4f}")


# --- MAP@K (K=3) Calculation for Ensemble ---
def apk(actual, predicted, k=10):
    if not actual:
        return 0.0
    score = 0.0
    num_hits = 0.0
    actual_item = actual[0]
    for i, p in enumerate(predicted):
        if p == actual_item:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            break
    if not num_hits:
        return 0.0
    return score / 1.0

def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

K = 3
# For each instance, get the top K predicted class indices based on probabilities
top_k_ensemble_oof_indices = np.argsort(ensemble_oof_preds, axis=1)[:, ::-1][:, :K]
predicted_labels_oof_topk = []
for row_indices in top_k_ensemble_oof_indices:
    predicted_labels_oof_topk.append(label_encoder.inverse_transform(row_indices).tolist())

actual_labels_oof_mapk = [[label] for label in label_encoder.inverse_transform(y_encoded).tolist()]

map_at_3_ensemble_oof = mapk(actual_labels_oof_mapk, predicted_labels_oof_topk, k=K)
print(f"Ensemble OOF Mean Average Precision at K={K} (MAP@{K}): {map_at_3_ensemble_oof:.4f}")

# --- Generate Submission File (using ensemble_test_preds) ---
# For submission, you'll want the predicted class names, not encoded integers
submission_predictions = label_encoder.inverse_transform(ensemble_test_pred_labels)

submission_df = pd.DataFrame({'id': full_test_data['id'], 'Fertilizer Name': submission_predictions})
submission_df.to_csv('ensemble_submission.csv', index=False)
print("\nSubmission file 'ensemble_submission.csv' created.")

# Optional: Plotting individual model's OOF loss (if desired, though harder with CV)
# You would need to store the eval_results for each fold and average them to plot.
# For simplicity, we are focusing on the final ensemble performance.


# --- Generate Submission File (using ensemble_test_preds for MAP@K style) ---

# We already have ensemble_test_preds from the cross-validation loop,
# which are the averaged probabilities for the test set across all folds.

# Get the top K predicted class indices for the test set
K_SUBMISSION = 3 # As per the example, you want top 3
top_k_ensemble_test_indices = np.argsort(ensemble_test_preds, axis=1)[:, ::-1][:, :K_SUBMISSION]

# Convert these indices back to original fertilizer names
submission_topk_labels = []
for row_indices in top_k_ensemble_test_indices:
    # label_encoder.inverse_transform expects a 1D array-like input
    submission_topk_labels.append(label_encoder.inverse_transform(row_indices).tolist())

# Format the predictions into a space-separated string for each row
formatted_predictions = [' '.join(row) for row in submission_topk_labels]

# Create the submission DataFrame
submission_df_topk = pd.DataFrame({
    'id': full_test_data['id'],
    'Fertilizer Name': formatted_predictions
})

# Save the submission file
submission_df_topk.to_csv('submission_topk_ensemble.csv', index=False)
print(f"\nSubmission file 'submission_topk_ensemble.csv' with top {K_SUBMISSION} predictions created.")

