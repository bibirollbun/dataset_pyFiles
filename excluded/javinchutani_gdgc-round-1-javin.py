# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing required libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV


# Loading data from the files
train_df = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/train.csv')
test_df = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/test.csv')
lookup_df = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/feature_lookup.csv')
sample_submission = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Features: {len(lookup_df)} features")


lookup_df


train_df.head()


# Removing ID as it is not of any use
train_df.drop('ID', axis = 1, inplace = True)


# Check for missing values 
print("\nMissing Values:")
print(train_df.isnull().sum())

# Identify numeric and categorical features
numeric_features = []
categorical_features = []

for col in train_df.columns:
    if col not in ['relationship_probability']:
        if train_df[col].dtype in ['float64', 'int64']:
            numeric_features.append(col)
        else:
            categorical_features.append(col)

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# Finding out the statistical properties of the numeric features
train_df[numeric_features].describe()


X = train_df.drop('relationship_probability', axis = 1)
y = train_df['relationship_probability']


# Value counts for each categorical feature
for col in categorical_features:
    print(f"\n--{col}--:")
    print(train_df[col].value_counts())
    print(f"Unique values: {train_df[col].nunique()}")


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


# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation = train_df[numeric_features].corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()



correlation.head()

# we can see that the features are not very correlated with each other


# For numeric features
train_df[numeric_features].hist(figsize=(20, 15), bins=30)
plt.tight_layout()
plt.show()

# We can see that almost all the features are almost normally distributed


outlier_counts = {}   # store results

for col in numeric_features:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    col_mask = (X[col] < lower) | (X[col] > upper)
    outliers_in_col = col_mask.sum()

    outlier_counts[col] = outliers_in_col

    keep_mask = ~col_mask
    X = X[keep_mask]
    y = y[keep_mask]

print("Outlier removal complete.\n")

for col, count in outlier_counts.items():
    print(f"{col}: {count} outliers removed")



if categorical_features:
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        print("Categorical encoding done")


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print("Features scaled")


    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )


# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_train_pred = lr_model.predict(X_train)
y_val_pred = lr_model.predict(X_val)

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Validation MAE: {val_mae:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Validation R²: {val_r2:.4f}")


xgb_model = XGBRegressor(
    n_estimators=500,      
    learning_rate=0.05,   
    max_depth=5,         
    subsample=0.8,       
    colsample_bytree=0.8, 
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror'
)

# Train
xgb_model.fit(X_train, y_train)

# Predictions
y_train_pred_xgb = xgb_model.predict(X_train)
y_val_pred_xgb = xgb_model.predict(X_val)

# Evaluation
train_rmse_xgb = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
val_rmse_xgb = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb))
train_mae_xgb = mean_absolute_error(y_train, y_train_pred_xgb)
val_mae_xgb = mean_absolute_error(y_val, y_val_pred_xgb)
train_r2_xgb = r2_score(y_train, y_train_pred_xgb)
val_r2_xgb = r2_score(y_val, y_val_pred_xgb)

print(f"Training RMSE: {train_rmse_xgb:.4f}")
print(f"Validation RMSE: {val_rmse_xgb:.4f}")
print(f"Training MAE: {train_mae_xgb:.4f}")
print(f"Validation MAE: {val_mae_xgb:.4f}")
print(f"Training R²: {train_r2_xgb:.4f}")
print(f"Validation R²: {val_r2_xgb:.4f}")


rf_model = RandomForestRegressor(
    n_estimators=300,   
    max_depth=None,       
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_val_pred_rf = rf_model.predict(X_val)

# Evaluation
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
val_rmse_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))
train_mae_rf = mean_absolute_error(y_train, y_train_pred_rf)
val_mae_rf = mean_absolute_error(y_val, y_val_pred_rf)
train_r2_rf = r2_score(y_train, y_train_pred_rf)
val_r2_rf = r2_score(y_val, y_val_pred_rf)

print(f"Training RMSE: {train_rmse_rf:.4f}")
print(f"Validation RMSE: {val_rmse_rf:.4f}")
print(f"Training MAE: {train_mae_rf:.4f}")
print(f"Validation MAE: {val_mae_rf:.4f}")
print(f"Training R²: {train_r2_rf:.4f}")
print(f"Validation R²: {val_r2_rf:.4f}")


lgbm_model = LGBMRegressor(
    n_estimators=500,        
    learning_rate=0.05,     
    num_leaves=31,         
    max_depth=-1,           
    subsample=0.8,          
    colsample_bytree=0.8,
    random_state=42,
    objective='regression',
    n_jobs=-1
)

# Train
lgbm_model.fit(X_train, y_train)

# Predictions
y_train_pred_lgbm = lgbm_model.predict(X_train)
y_val_pred_lgbm = lgbm_model.predict(X_val)

# Evaluation
train_rmse_lgbm = np.sqrt(mean_squared_error(y_train, y_train_pred_lgbm))
val_rmse_lgbm = np.sqrt(mean_squared_error(y_val, y_val_pred_lgbm))
train_mae_lgbm = mean_absolute_error(y_train, y_train_pred_lgbm)
val_mae_lgbm = mean_absolute_error(y_val, y_val_pred_lgbm)
train_r2_lgbm = r2_score(y_train, y_train_pred_lgbm)
val_r2_lgbm = r2_score(y_val, y_val_pred_lgbm)

print(f"Training RMSE: {train_rmse_lgbm:.4f}")
print(f"Validation RMSE: {val_rmse_lgbm:.4f}")
print(f"Training MAE: {train_mae_lgbm:.4f}")
print(f"Validation MAE: {val_mae_lgbm:.4f}")
print(f"Training R²: {train_r2_lgbm:.4f}")
print(f"Validation R²: {val_r2_lgbm:.4f}")


X_1 = test_df.drop('ID', axis = 1)
if categorical_features:
        for col in categorical_features:
            le = LabelEncoder()
            X_1[col] = le.fit_transform(X_1[col].astype(str))


lr_preds = lr_model.predict(X_1)
xgb_preds = xgb_model.predict(X_1)
rf_preds =rf_model.predict(X_1)
lgbm_preds = lgbm_model.predict(X_1)

ensemble_preds = (lr_preds + xgb_preds + rf_preds + lgbm_preds) / 4


ensemble_preds = np.clip(ensemble_preds, 0, 100)


ID = np.arange(1, len(ensemble_preds) + 1) 

submission = pd.DataFrame({
    'ID': ID,
    'relationship_probability': ensemble_preds
})

submission.to_csv('baseline_submission.csv', index=False)

print("Submission file created: baseline_submission.csv")
print(f"Submission shape: {submission.shape}")

