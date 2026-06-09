import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load competition data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

# Load original dataset (add it to your notebook first; adjust path if slug differs)
try:
    original = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv') 
    # Align columns if needed (assume same structure)
    original = original[train.columns]  # Keep only matching columns
    train = pd.concat([train, original], ignore_index=True)
    print(f"Added {len(original)} rows from original dataset.")
except:
    print("Original dataset not found; proceeding without it.")

# Basic cleaning (drop duplicates, handle missing if any)
train = train.drop_duplicates()
# Fill missing with median (if any; adjust based on your EDA)
for col in train.columns:
    if train[col].isnull().sum() > 0 and col not in ['id', 'BeatsPerMinute']:
        train[col] = train[col].fillna(train[col].median())
        test[col] = test[col].fillna(train[col].median())  # Use train median for test

# Prepare features and target
target = 'BeatsPerMinute'
ID = 'id'
X = train.drop(columns=[ID, target])
y = train[target]
X_test = test.drop(columns=[ID])

# Identify categorical columns (if any, like 'key' or 'mode'; XGBoost can handle them)
cat_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 20]  # Assume low-cardinality as cat
if cat_cols:
    print(f"Categorical columns: {cat_cols}")
    # XGBoost handles categories if we set enable_categorical=True, but for simplicity, we'll one-hot if many
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)
        X_test = pd.get_dummies(X_test, columns=cat_cols)
        X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)  # Align columns

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
test_preds = np.zeros(len(test))

# Train with XGBoost
for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        eval_metric='rmse',
        early_stopping_rounds=100,
        tree_method='hist',  # Faster
        enable_categorical=True if cat_cols else False  # If cats present
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=200
    )
    
    valid_pred = model.predict(X_valid)
    fold_rmse = mean_squared_error(y_valid, valid_pred, squared=False)
    rmse_scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse}")
    
    # Predict on test
    test_preds += model.predict(X_test) / kf.n_splits

print(f"Mean CV RMSE: {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")

# Submission
submission['BeatsPerMinute'] = test_preds
submission.to_csv('submission.csv', index=False)
print("Submission file created. Submit it on Kaggle!")

# Optional: Feature importance
import matplotlib.pyplot as plt
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_imp.plot(kind='barh')
plt.title('Feature Importance')
plt.show()

