# Relationship Status Prediction - Baseline Model
# GDGC NIT Jalandhar - AI/ML Inductions

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train_df = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/train.csv')
test_df = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/test.csv')
lookup_df = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/feature_lookup.csv')
sample_submission = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Features: {len(lookup_df)} features")


# Check feature lookup
print("\nFeature Lookup Preview:")
print(lookup_df.head(10))

# Target distribution
print("\nTarget Variable Distribution:")
print(train_df['relationship_probability'].describe())

# Visualize target distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(train_df['relationship_probability'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Relationship Probability')
plt.ylabel('Frequency')
plt.title('Training Set - Target Distribution')
plt.axvline(train_df['relationship_probability'].mean(), color='red', 
            linestyle='--', label=f'Mean: {train_df["relationship_probability"].mean():.2f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(train_df['relationship_probability'], vert=True)
plt.ylabel('Relationship Probability')
plt.title('Training Set - Target Boxplot')

plt.tight_layout()
plt.show()


# Check for missing values
print("\nMissing Values:")
print(train_df.isnull().sum().sum())

# Identify numeric and categorical features
numeric_features = []
categorical_features = []

for col in train_df.columns:
    if col not in ['ID', 'relationship_probability']:
        if train_df[col].dtype in ['float64', 'int64']:
            numeric_features.append(col)
        else:
            categorical_features.append(col)

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")




# Separate features and target
X_train = train_df.drop(['ID', 'relationship_probability'], axis=1)
y_train = train_df['relationship_probability']
X_test = test_df.drop(['ID'], axis=1)

# Encode categorical variables
le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    le_dict[col] = le

print("Categorical variables encoded using Label Encoding")

# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("Numeric features standardized")



# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_tr, y_tr)

# Predictions
y_train_pred = lr_model.predict(X_tr)
y_val_pred = lr_model.predict(X_val)

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_tr, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
train_mae = mean_absolute_error(y_tr, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
train_r2 = r2_score(y_tr, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Validation MAE: {val_mae:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Validation R²: {val_r2:.4f}")




# Train XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_tr, y_tr)

# Predictions
y_train_pred_xgb = xgb_model.predict(X_tr)
y_val_pred_xgb = xgb_model.predict(X_val)

# Evaluation
train_rmse_xgb = np.sqrt(mean_squared_error(y_tr, y_train_pred_xgb))
val_rmse_xgb = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb))
train_mae_xgb = mean_absolute_error(y_tr, y_train_pred_xgb)
val_mae_xgb = mean_absolute_error(y_val, y_val_pred_xgb)
train_r2_xgb = r2_score(y_tr, y_train_pred_xgb)
val_r2_xgb = r2_score(y_val, y_val_pred_xgb)

print(f"Training RMSE: {train_rmse_xgb:.4f}")
print(f"Validation RMSE: {val_rmse_xgb:.4f}")
print(f"Training MAE: {train_mae_xgb:.4f}")
print(f"Validation MAE: {val_mae_xgb:.4f}")
print(f"Training R²: {train_r2_xgb:.4f}")
print(f"Validation R²: {val_r2_xgb:.4f}")



# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_tr, y_tr)

# Predictions
y_train_pred_rf = rf_model.predict(X_tr)
y_val_pred_rf = rf_model.predict(X_val)

# Evaluation
train_rmse_rf = np.sqrt(mean_squared_error(y_tr, y_train_pred_rf))
val_rmse_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))
train_mae_rf = mean_absolute_error(y_tr, y_train_pred_rf)
val_mae_rf = mean_absolute_error(y_val, y_val_pred_rf)
train_r2_rf = r2_score(y_tr, y_train_pred_rf)
val_r2_rf = r2_score(y_val, y_val_pred_rf)

print(f"Training RMSE: {train_rmse_rf:.4f}")
print(f"Validation RMSE: {val_rmse_rf:.4f}")
print(f"Training MAE: {train_mae_rf:.4f}")
print(f"Validation MAE: {val_mae_rf:.4f}")
print(f"Training R²: {train_r2_rf:.4f}")
print(f"Validation R²: {val_r2_rf:.4f}")


# Predict on test set using each model
lr_preds = lr_model.predict(X_test_scaled)
xgb_preds = xgb_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test_scaled)

# Simple average ensemble
ensemble_preds = (lr_preds + xgb_preds + rf_preds) / 3

# Clip predictions to valid range [0, 100]
ensemble_preds = np.clip(ensemble_preds, 0, 100)



submission = pd.DataFrame({
    'ID': test_df['ID'],
    'relationship_probability': ensemble_preds
})

submission.to_csv('baseline_submission.csv', index=False)

print("Submission file created: baseline_submission.csv")
print(f"Submission shape: {submission.shape}")

