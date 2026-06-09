import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


# Load the data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


# Explore basic info
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)
print("\nTrain Data Info:")
train_data.info()


print(train_data['loan_paid_back'].value_counts(normalize=True))


print(train_data.describe())


# Separate features and target
X = train_data.drop(['id', 'loan_paid_back'], axis=1)
y = train_data['loan_paid_back']
X_test = test_data.drop('id', axis=1)


# Label encode categorical variables
categorical_cols = ['gender', 'marital_status', 'education_level', 
                   'employment_status', 'loan_purpose', 'grade_subgrade']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le


# Convert to LightGBM dataset format for better performance
categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_cols]


# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Create LightGBM datasets
train_data_lgb = lgb.Dataset(
    X_train, 
    label=y_train,
    categorical_feature=categorical_feature_indices,
    free_raw_data=False
)

val_data_lgb = lgb.Dataset(
    X_val,
    label=y_val,
    categorical_feature=categorical_feature_indices,
    free_raw_data=False
)


# LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'is_unbalance': True,  # Handles the 80/20 imbalance
    'random_state': 42,
    'verbosity': -1
}


# Train model
print("Training LightGBM...")
model_lgb = lgb.train(
    params,
    train_data_lgb,
    valid_sets=[val_data_lgb],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(50)
    ]
)


# Validate model
val_preds_lgb = model_lgb.predict(X_val)
val_auc_lgb = roc_auc_score(y_val, val_preds_lgb)
print(f"LightGBM Validation AUC: {val_auc_lgb:.5f}")


# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_lgb.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))


def get_tuned_lightgbm_params(strategy="balanced"):
    """Return tuned parameters based on strategy"""
    
    base_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbosity': -1,
        'is_unbalance': True
    }
    
    if strategy == "conservative":
        # More regularization, less overfitting
        return {**base_params, **{
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 7,
            'min_child_samples': 100,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }}
    elif strategy == "aggressive":
        # More complexity, higher risk of overfitting
        return {**base_params, **{
            'learning_rate': 0.1,
            'num_leaves': 127,
            'max_depth': -1,
            'min_child_samples': 20,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 3
        }}
    else:  # balanced (recommended)
        return {**base_params, **{
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': -1,
            'min_child_samples': 40,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }}


# Quick Experiment
strategies = ['conservative', 'balanced', 'aggressive']
best_auc = 0.921444
best_strategy = 'original'

for strategy in strategies:
    print(f"\nTesting {strategy} strategy...")
    
    params_tuned = get_tuned_lightgbm_params(strategy)
    
    model_tuned = lgb.train(
        params_tuned,
        train_data_lgb,  # Using original features (no engineering)
        valid_sets=[val_data_lgb],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ]
    )
    
    val_preds_tuned = model_tuned.predict(X_val)
    val_auc_tuned = roc_auc_score(y_val, val_preds_tuned)
    
    print(f"{strategy} strategy AUC: {val_auc_tuned:.5f}")


# Make predictions on test set
# Since the variations in hyperparameters did not make any progress submitting the original model to test the scores.
test_preds_lgb = model_lgb.predict(X_test)


# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'loan_paid_back': test_preds_lgb
})


# Save submission
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print("\nSubmission head:")
print(submission.head())





