# ========================
# LIBRARY IMPORTS
# ========================
!pip install xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)


# ========================
# LOAD DATA
# ========================
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


# ========================
# DATA AUGMENTATION (optional)
# Replicate the original dataset 7 times to increase training size
# ========================
original_copy = original.copy()
for _ in range(7):
    original = pd.concat([original, original_copy], axis=0)


# ========================
# FEATURE/TARGET SPLITTING
# ========================
X_train = train.drop(['id', 'Personality'], axis=1, errors='ignore')
y_train = train['Personality']
X_test = test.drop(['id'], axis=1, errors='ignore')

X_original = original.drop(['id', 'Personality'], axis=1, errors='ignore')
y_original = original['Personality']

print(f"Features shape: {X_train.shape}")
print(f"Target shape: {y_train.shape if y_train is not None else 'None'}")


# ========================
# LABEL ENCODING (Categorical Features + Target)
# ========================
label_encoders = {}
categorical_columns = X_train.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    X_original[col] = le.transform(X_original[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {le.classes_}")

# Encode the target label
target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)
y_original_encoded = target_encoder.fit_transform(y_original)
print(f"Target classes: {target_encoder.classes_}")


# ========================
# FEATURE SCALING (optional)
# Not necessary for XGBoost but can sometimes help
# ========================
scaler = StandardScaler()
feature_names = X_train.columns.tolist()
print(f"Feature columns: {feature_names}")


# ========================
# XGBOOST PARAMETERS
# ========================
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_leaves': 25,
    'min_child_weight': np.float64(0.0034),
    'learning_rate': np.float64(0.0947),
    'n_estimators': 10000,
    'subsample': np.float64(0.8025),
    'colsample_bylevel': np.float64(0.8360),
    'colsample_bytree': np.float64(0.8732),
    'reg_alpha': np.float64(0.0029),
    'reg_lambda': np.float64(27.126),
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda'
}


# ========================
# STRATIFIED K-FOLD CROSS VALIDATION USING xgb.train() WITH EARLY STOPPING
# ========================
cv_scores = []
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\nPerforming {n_splits}-fold Stratified Cross Validation using xgb.train...")

fold_num = 1
for train_idx, val_idx in skf.split(X_train, y_train_encoded):
    print(f"\nTraining Fold {fold_num}/{n_splits}...")
    
    # Split into train and validation
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train_encoded[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train_encoded[val_idx]

    # Augment with original dataset
    X_fold_train = pd.concat([X_fold_train, X_original], axis=0, ignore_index=True)
    y_fold_train = np.concatenate((y_fold_train, y_original_encoded))

    # Convert to DMatrix (required for xgb.train)
    dtrain = xgb.DMatrix(data=X_fold_train, label=y_fold_train)
    dval = xgb.DMatrix(data=X_fold_val, label=y_fold_val)

    # Setup evaluation list
    evals = [(dtrain, 'train'), (dval, 'eval')]

    # Train using xgb.train with early stopping
    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=10000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Predict on validation set
    val_preds = booster.predict(dval)
    val_preds_binary = (val_preds > 0.5).astype(int)

    # Accuracy
    fold_accuracy = accuracy_score(y_fold_val, val_preds_binary)
    cv_scores.append(fold_accuracy)

    print(f"Fold {fold_num} Accuracy: {fold_accuracy:.4f}")
    print(f"Best iteration: {booster.best_iteration}")
    fold_num += 1

# Print summary of cross-validation
cv_scores = np.array(cv_scores)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


# ========================
# DETERMINE OPTIMAL NUMBER OF ESTIMATORS USING xgb.train()
# ========================
best_iterations = []
fold_num = 1

print("\nExtracting best iterations from each CV fold using xgb.train...")
for train_idx, val_idx in skf.split(X_train, y_train_encoded):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train_encoded[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train_encoded[val_idx]

    # Augment with external dataset
    X_fold_train = pd.concat([X_fold_train, X_original], axis=0, ignore_index=True)
    y_fold_train = np.concatenate((y_fold_train, y_original_encoded))

    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval = xgb.DMatrix(X_fold_val, label=y_fold_val)

    evals = [(dtrain, 'train'), (dval, 'eval')]

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=10000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    best_iterations.append(booster.best_iteration)
    print(f"Fold {fold_num} best iteration: {booster.best_iteration}")
    fold_num += 1

# Average optimal estimators across folds
optimal_n_estimators = int(np.mean(best_iterations))
print(f"\nOptimal n_estimators (average): {optimal_n_estimators}")
print(f"Range: {min(best_iterations)} - {max(best_iterations)}")


# ========================
# FINAL MODEL TRAINING
# ========================
print(f"\nTraining final model on full dataset with {optimal_n_estimators} estimators...")
xgb_params_final = xgb_params.copy()
xgb_params_final['n_estimators'] = optimal_n_estimators

xgb_model_final = xgb.XGBClassifier(**xgb_params_final)
xgb_model_final.fit(X_train, y_train_encoded)
print("Final model trained on 100% of training data!")


# ========================
# FEATURE IMPORTANCE
# ========================
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))


# ========================
# PREDICTION AND SUBMISSION
# ========================
test_predictions = xgb_model_final.predict(X_test)

# Convert predictions back to original labels
if y_train is not None:
    test_pred_labels = target_encoder.inverse_transform(test_predictions)
else:
    test_pred_labels = ['Introvert' if pred == 0 else 'Extrovert' for pred in test_predictions]

# Create submission file
print("\n=== CREATING SUBMISSION FILE ===")
submission_df = pd.DataFrame({
    'id': test['id'],
    'Personality': test_pred_labels
})
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")


