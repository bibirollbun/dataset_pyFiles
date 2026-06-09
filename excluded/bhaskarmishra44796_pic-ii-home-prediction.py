import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as w
w.filterwarnings('ignore')

# Load data
dt = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')

# Fill missing values
dt['sale_nbr'].fillna(dt['sale_nbr'].median(), inplace=True)
dt['subdivision'].fillna(dt['subdivision'].mode()[0], inplace=True)
dt['submarket'].fillna(dt['submarket'].mode()[0], inplace=True)

test['sale_nbr'].fillna(test['sale_nbr'].median(), inplace=True)
test['subdivision'].fillna(test['subdivision'].mode()[0], inplace=True)
test['submarket'].fillna(test['submarket'].mode()[0], inplace=True)

# Identify numeric and categorical columns
num_cols_data = dt.select_dtypes(['number']).columns.tolist()
cat_cols_data = dt.select_dtypes(['object']).columns.tolist()

num_cols_test = test.select_dtypes(['number']).columns.tolist()
cat_cols_test = test.select_dtypes(['object']).columns.tolist()

# Remove target column from scaling
target = 'sale_price'
num_cols_data.remove(target)

# Scale numeric features
from sklearn.preprocessing import StandardScaler

def scaler(df, columns):
    scaled = {}
    for col in columns:
        std = StandardScaler()
        df[col] = std.fit_transform(df[[col]])
        scaled[col] = std
    return df, scaled

dt, scaled = scaler(dt, num_cols_data)

def apply_scaler(df, columns, scaler_dict):
    for col in columns:
        if col in scaler_dict:
            df[col] = scaler_dict[col].transform(df[[col]])
    return df

test = apply_scaler(test, num_cols_test, scaled)

# Encode categorical features
from sklearn.preprocessing import LabelEncoder

def encode(df, columns):
    encoder = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoder[col] = le
    return df, encoder

dt, encoder = encode(dt, cat_cols_data)
test, _ = encode(test, cat_cols_test)

# Prepare data
X = dt.drop(target, axis=1)
y = dt[target]

# Train-validation split
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Quantile Regressors using LightGBM
import lightgbm as lgb

def train_quantile_model(alpha):
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=alpha,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# 90% interval: lower = 5th percentile, upper = 95th percentile
model_lower = train_quantile_model(0.05)
model_upper = train_quantile_model(0.95)

# Predict on validation set (optional local evaluation)
y_pred_lower = model_lower.predict(X_valid)
y_pred_upper = model_upper.predict(X_valid)

# Optional: Winkler Score Evaluation
def winkler_score(y_true, lower, upper, alpha=0.1):
    width = upper - lower
    score = width.copy()
    mask_lower = y_true < lower
    mask_upper = y_true > upper
    score[mask_lower] += (2 / alpha) * (lower[mask_lower] - y_true[mask_lower])
    score[mask_upper] += (2 / alpha) * (y_true[mask_upper] - upper[mask_upper])
    return score.mean()

print("Local Winkler Score on Validation Set:", winkler_score(y_valid.values, y_pred_lower, y_pred_upper))

# Predict on test set
pi_lower = model_lower.predict(test)
pi_upper = model_upper.predict(test)




# Load sample submission to get 'id' column
submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")

# Ensure test has the same number of rows
assert len(submission) == len(test), "Mismatch in test and submission lengths."

# Assign predicted lower and upper bounds
submission['pi_lower'] = pi_lower
submission['pi_upper'] = pi_upper

# Drop any unexpected columns like 'sale_price' if present
submission = submission[['id', 'pi_lower', 'pi_upper']]

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


