# ============================================================
# Road Accident Risk Prediction â€” EDA & LightGBM Baseline
# Competition: Kaggle Playground Series (S5E10)
# Author: Naimul Hasan Shadesh
# ============================================================

# =====================
# 1. Import Libraries
# =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set global style for plots
plt.style.use('seaborn-v0_8-whitegrid')

# =====================
# 2. Load Dataset
# =====================
train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'
sample_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

print("âœ… Data Loaded Successfully!")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# =====================
# 3. Quick Overview
# =====================
print("\nðŸ”¹ Training Data Overview:")
display(train.head())

print("\nðŸ”¹ Test Data Overview:")
display(test.head())

print("\nðŸ”¹ Columns in the dataset:")
print(train.columns.tolist())

# =====================
# 4. Basic Info and Missing Values
# =====================
print("\nðŸ”¹ Dataset Info:")
print(train.info())

print("\nðŸ”¹ Checking Missing Values in Train Set:")
missing_values = train.isnull().sum().sort_values(ascending=False)
print(missing_values[missing_values > 0])

# Visualize missing values
plt.figure(figsize=(10, 5))
sns.heatmap(train.isnull(), cbar=False)
plt.title("Missing Values Heatmap (Train Data)")
plt.show()

# =====================
# 5. Target Variable Analysis
# =====================
target = 'accident_risk'

plt.figure(figsize=(8, 5))
sns.histplot(train[target], kde=True, bins=30, color='blue')
plt.title('Distribution of Target Variable (accident_risk)')
plt.xlabel('accident_risk')
plt.ylabel('Frequency')
plt.show()

print(f"Mean Accident Risk: {train[target].mean():.4f}")
print(f"Std Accident Risk: {train[target].std():.4f}")

# =====================
# 6. Numerical & Categorical Feature Split
# =====================
numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove target from numerical
if target in numerical_features:
    numerical_features.remove(target)

print(f"\nðŸ”¹ Numerical Features ({len(numerical_features)}):")
print(numerical_features)
print(f"\nðŸ”¹ Categorical Features ({len(categorical_features)}):")
print(categorical_features)

# =====================
# 7. Correlation Analysis
# =====================
corr = train[numerical_features + [target]].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

# Top 10 correlated features
corr_target = corr[target].drop(target).sort_values(ascending=False)
print("\nðŸ”¹ Top 10 Features Positively Correlated with Target:")
print(corr_target.head(10))
print("\nðŸ”¹ Top 10 Features Negatively Correlated with Target:")
print(corr_target.tail(10))

# =====================
# 8. Basic Feature Analysis
# =====================
# Plot distributions of a few numerical columns
sample_num_cols = numerical_features[:4]
train[sample_num_cols].hist(bins=25, figsize=(12, 8))
plt.suptitle("Distribution of Selected Numerical Features")
plt.show()

# =====================
# 9. Data Preprocessing
# =====================
# Fill missing numeric with median
for col in numerical_features:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(train[col].median(), inplace=True)

# Encode categorical using Label Encoding (simple approach)
from sklearn.preprocessing import LabelEncoder
for col in categorical_features:
    le = LabelEncoder()
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

print("\nâœ… Data Preprocessing Completed!")

# =====================
# 10. Feature and Target Split
# =====================
X = train.drop(columns=[target])
y = train[target]
X_test = test.copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# =====================
# 11. LightGBM Model Training (5-Fold CV)
# =====================
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'random_state': 42
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nðŸ”¸ Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    
    # âœ… Corrected: Use callback for early stopping and logging
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

# =====================
# 12. Model Evaluation
# =====================
cv_rmse = mean_squared_error(y, oof, squared=False)
print(f"\nâœ… CV RMSE Score: {cv_rmse:.6f}")

# Feature Importance
lgb.plot_importance(model, max_num_features=20, figsize=(10, 6))
plt.title("Top 20 Feature Importances (LightGBM)")
plt.show()

# =====================
# 13. Submission File
# =====================
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': np.clip(preds, 0, 1)  # clip between [0, 1]
})
submission.to_csv('submission.csv', index=False)

print("\nâœ… Submission file created successfully!")
print(submission.head())

