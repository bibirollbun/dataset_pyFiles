import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


print(f" Train data shape: {train_df.shape}")
print(f" Test data shape: {test_df.shape}")
print(f"\n Available columns:\n{train_df.columns.tolist()}")
print(f"\n Data info:\n{train_df.info()}")
print(f"\n Target distribution:\n{train_df.iloc[:, -1].describe()}")


# Check for missing values
print("\n Missing values in train:")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0] if missing_train.sum() > 0 else "No missing values")

print("\n Missing values in test:")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0] if missing_test.sum() > 0 else "No missing values")

# Save target and id column names
target_col = train_df.columns[-1]  
id_col = 'id' if 'id' in train_df.columns else train_df.columns[0]

print(f"\n Target column: {target_col}")
print(f" ID column: {id_col}")


# Separate features and target
X = train_df.drop([target_col, id_col], axis=1)
y = train_df[target_col]
X_test = test_df.drop([id_col], axis=1)
test_ids = test_df[id_col]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
boolean_cols = X.select_dtypes(include=['bool']).columns.tolist()

print(f"\n Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f" Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Boolean columns ({len(boolean_cols)}): {boolean_cols}")

# Convert boolean columns to int
for col in boolean_cols:
    X[col] = X[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# Handle missing values

# For numerical columns: fill with median
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# For categorical columns: fill with mode
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le


# High curvature + high speed = more dangerous
X['curvature_speed_interaction'] = X['curvature'] * X['speed_limit']
X_test['curvature_speed_interaction'] = X_test['curvature'] * X_test['speed_limit']

# Low lighting + high speed = more dangerous
X['speed_per_lane'] = X['speed_limit'] / (X['num_lanes'] + 1)
X_test['speed_per_lane'] = X_test['speed_limit'] / (X_test['num_lanes'] + 1)

# Accidents density per lane
X['accidents_per_lane'] = X['num_reported_accidents'] / (X['num_lanes'] + 1)
X_test['accidents_per_lane'] = X_test['num_reported_accidents'] / (X_test['num_lanes'] + 1)

# Create polynomial features for important numerical columns
if 'curvature' in X.columns:
    X['curvature_squared'] = X['curvature'] ** 2
    X_test['curvature_squared'] = X_test['curvature'] ** 2

if 'speed_limit' in X.columns:
    X['speed_squared'] = X['speed_limit'] ** 2
    X_test['speed_squared'] = X_test['speed_limit'] ** 2

print(f" Features after engineering: {X.shape[1]}")


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n Training set: {X_train.shape}")
print(f" Validation set: {X_val.shape}")


# Initialize models
models = {
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        random_state=42,
        verbose=-1
    ),
    'XGBoost': XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        verbosity=0
    ),
    'CatBoost': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        random_state=42,
        verbose=0
    )
}

# Train and evaluate models
predictions = {}
val_predictions = {}
model_scores = {}

for name, model in models.items():
    print(f"\n Training {name}...")
    
    # Train model 
    if name == 'LightGBM':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
    elif name == 'XGBoost':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:  # CatBoost
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
    
    # Predictions
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Store predictions
    predictions[name] = test_pred
    val_predictions[name] = val_pred
    
    # Evaluate with metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    model_scores[name] = val_rmse
    
    print(f" {name} - Validation RMSE: {val_rmse:.6f}")
    print(f" {name} - Validation MAE: {val_mae:.6f}")
    print(f" {name} - Validation R²: {val_r2:.6f}")


# Calculate weights based on inverse of RMSE (lower RMSE = higher weight)
weights = []
for name in predictions.keys():
    rmse = model_scores[name]
    weight = 1 / rmse  
    weights.append(weight)

# Normalize weights
weights = np.array(weights) / sum(weights)
print(f"Model weights: {dict(zip(predictions.keys(), weights))}")

# Create ensemble prediction
ensemble_pred = np.zeros(len(X_test))
for (name, pred), weight in zip(predictions.items(), weights):
    ensemble_pred += pred * weight

# Evaluate ensemble on validation set
ensemble_val_pred = np.zeros(len(X_val))
for (name, pred), weight in zip(val_predictions.items(), weights):
    ensemble_val_pred += pred * weight

ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
ensemble_mae = mean_absolute_error(y_val, ensemble_val_pred)
ensemble_r2 = r2_score(y_val, ensemble_val_pred)

print(f"\n Ensemble - Validation RMSE: {ensemble_rmse:.6f}")
print(f" Ensemble - Validation MAE: {ensemble_mae:.6f}")
print(f" Ensemble - Validation R²: {ensemble_r2:.6f}")


# Prepare submission dataframe
submission = pd.DataFrame({
    id_col: test_ids,
    target_col: ensemble_pred
})

# Clip predictions to valid range (0 to 1 for accident risk)
submission[target_col] = submission[target_col].clip(0, 1)

# Save submission file
submission.to_csv('submission.csv', index=False)
print("\n Submission file saved as 'submission.csv'")
print(f"\n Submission preview:\n{submission.head(10)}")
print(f"\n Prediction statistics:")
print(submission[target_col].describe())


# LightGBM Feature Importance
print("\n LightGBM Feature Importance:")
lgb_model = models['LightGBM']
lgb_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(lgb_feature_importance.head(10))

# CatBoost Feature Importance
print("\n CatBoost Feature Importance:")
catboost_model = models['CatBoost']
catboost_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': catboost_model.feature_importances_
}).sort_values('importance', ascending=False)

print(catboost_feature_importance.head(10))

print("\n * Top 10 Important Features")

# Plot feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# LightGBM plot
sns.barplot(data=lgb_feature_importance.head(10), x='importance', y='feature', ax=axes[0], palette='viridis')
axes[0].set_title('Top 10 Feature Importance (LightGBM)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_ylabel('Feature', fontsize=12)

# CatBoost plot
sns.barplot(data=catboost_feature_importance.head(10), x='importance', y='feature', ax=axes[1], palette='magma')
axes[1].set_title('Top 10 Feature Importance (CatBoost)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_ylabel('Feature', fontsize=12)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_train, bins=50, alpha=0.7, label='Train', edgecolor='black')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.title('Training Data Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(ensemble_pred, bins=50, alpha=0.7, label='Test Predictions', color='orange', edgecolor='black')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.title('Test Predictions Distribution')
plt.legend()

plt.tight_layout()
plt.show()

