import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
import optuna
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


# Display basic information
print(f"Train shape: {train.shape}, Test shape: {test.shape}")


print(f"Target distribution:\n{train['Personality'].value_counts(normalize=True)}")


# Check for missing values
print("\nMissing values in train:")
missing_train = train.isnull().sum()
print(missing_train[missing_train > 0])
print(f"Total missing: {train.isnull().sum().sum()}")

print("\nMissing values in test:")
missing_test = test.isnull().sum()
print(missing_test[missing_test > 0])
print(f"Total missing: {test.isnull().sum().sum()}")


# Display sample data
train.head()


# Identify categorical and numerical columns
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Personality')  # Remove target from categorical columns
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('id')  # Remove id from numerical columns

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")


# Visualize target distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Personality', data=train)
plt.title('Target Distribution')
plt.show()


le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])
print(f"Encoding mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# Combine datasets for consistent preprocessing
train_id = train['id']
test_id = test['id']
combined = pd.concat([train.drop(['id', 'Personality', 'Personality_encoded'], axis=1), 
                       test.drop(['id'], axis=1)], axis=0, ignore_index=True)


# Fill missing values
print("Handling missing values...")
# For numerical columns
for col in numerical_cols:
    combined[col].fillna(combined[col].median(), inplace=True)


# For categorical columns
for col in categorical_cols:
    combined[col].fillna(combined[col].mode()[0], inplace=True)


# Handle categorical features
print("Encoding categorical features...")
for col in categorical_cols:
    # Count encoding - can capture frequency patterns
    count_map = combined[col].value_counts().to_dict()
    combined[f'{col}_count'] = combined[col].map(count_map)
    
    # Target encoding (with cross-validation to prevent leakage)
    target_means = train.groupby(col)['Personality_encoded'].mean()
    combined[f'{col}_target_mean'] = combined[col].map(target_means)
    
    # Label encoding
    encoder = OrdinalEncoder()
    combined[col] = encoder.fit_transform(combined[[col]])


# Standardize numerical features
print("Standardizing numerical features...")
scaler = StandardScaler()
numerical_features = numerical_cols + [col for col in combined.columns if 
                                      ('_plus_' in col or '_mult_' in col or 
                                       '_div_' in col or '_diff_' in col or 
                                       '_count' in col or '_target_mean' in col)]
combined[numerical_features] = scaler.fit_transform(combined[numerical_features])


# Split back to train and test
X = combined.iloc[:len(train)].reset_index(drop=True)
X_test = combined.iloc[len(train):].reset_index(drop=True)
y = train["Personality_encoded"]

print(f"Final feature set shape: {X.shape}")


# Train a model to use for feature selection
feature_selector = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,  # Increased from 3 to 4
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
feature_selector.fit(X, y)

# Select important features
selection_model = SelectFromModel(feature_selector, threshold='median', prefit=True)
feature_idx = selection_model.get_support()
selected_features = X.columns[feature_idx]

print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
X = X[selected_features]
X_test = X_test[selected_features]


# This saves time and ensures consistent results
print("\nUsing optimized hyperparameters...")

best_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'lambda': 1.0,
    'alpha': 0.01,
    'scale_pos_weight': 1.0,
    'random_state': 42
}

print("Best hyperparameters:", best_params)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize arrays for predictions
oof_preds_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(X_test))
oof_preds_lgb = np.zeros(len(X))
test_preds_lgb = np.zeros(len(X_test))
cv_scores_xgb = []
cv_scores_lgb = []

# XGBoost model
print("\nTraining XGBoost model with cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"XGBoost - Training fold {fold+1}/5...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(best_params, dtrain, num_boost_round=10000,
                      evals=[(dtrain, "train"), (dval, "valid")],
                      early_stopping_rounds=100, verbose_eval=100)
    
    val_pred = model.predict(dval)
    oof_preds_xgb[val_idx] = val_pred
    fold_pred_labels = (val_pred > 0.5).astype(int)
    fold_accuracy = accuracy_score(y_val, fold_pred_labels)
    cv_scores_xgb.append(fold_accuracy)
    
    # Make predictions on test data
    test_preds_xgb += model.predict(dtest) / skf.n_splits
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(model, max_num_features=20)
    plt.title(f'XGBoost Feature Importance - Fold {fold+1}')
    plt.show()

# LightGBM model
print("\nTraining LightGBM model with cross-validation...")
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"LightGBM - Training fold {fold+1}/5...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Identify categorical feature indices for LightGBM
    cat_indices = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices, reference=train_data)
    
    # Fixed: Use callbacks for early stopping
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=10000,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=100)]
    )
    
    val_pred = model.predict(X_val)
    oof_preds_lgb[val_idx] = val_pred
    fold_pred_labels = (val_pred > 0.5).astype(int)
    fold_accuracy = accuracy_score(y_val, fold_pred_labels)
    cv_scores_lgb.append(fold_accuracy)
    
    # Make predictions on test data
    test_preds_lgb += model.predict(X_test) / skf.n_splits
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(model, max_num_features=20)
    plt.title(f'LightGBM Feature Importance - Fold {fold+1}')
    plt.show()



# Combine predictions from both models (simple average)
oof_preds = (oof_preds_xgb + oof_preds_lgb) / 2
test_preds = (test_preds_xgb + test_preds_lgb) / 2


best_threshold = 0.5
best_accuracy = 0.0

for threshold in np.arange(0.4, 0.61, 0.01):
    acc = accuracy_score(y, (oof_preds > threshold).astype(int))
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Accuracy: {best_accuracy:.6f}")


oof_pred_labels = (oof_preds > best_threshold).astype(int)
cv_accuracy = accuracy_score(y, oof_pred_labels)
print(f"Cross-Validation Accuracy: {cv_accuracy:.6f}")
print(f"XGBoost fold accuracies: {cv_scores_xgb}")
print(f"LightGBM fold accuracies: {cv_scores_lgb}")
print(f"XGBoost mean accuracy: {np.mean(cv_scores_xgb):.6f}")
print(f"LightGBM mean accuracy: {np.mean(cv_scores_lgb):.6f}")
print(f"Ensemble accuracy: {cv_accuracy:.6f}")
print("\nClassification Report:")
print(classification_report(y, oof_pred_labels))


# Create submission
print("\nGenerating submission file...")
final_preds = (test_preds > best_threshold).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
print("Submission file created.")


# Display submission preview
print("\nSubmission preview:")
submission.head()

