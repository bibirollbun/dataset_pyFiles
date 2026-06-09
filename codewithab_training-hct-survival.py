import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Training Notebook for XGBoost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib

# Define selected columns
selected_columns = [
    "prim_disease_hct", "hla_match_b_low", "prod_type", "year_hct", "obesity", 
    "donor_age", "prior_tumor", "gvhd_proph", "sex_match", "comorbidity_score", 
    "karnofsky_score", "donor_related", "age_at_hct", "efs"  # Target column
]

# Load dataset
train_file_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
df = pd.read_csv(train_file_path)

# Keep only selected columns
df = df[selected_columns]

# Separate features and target
X = df.drop(columns=["efs"])
y = df["efs"]

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Handle missing values
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_cats = encoder.fit_transform(X[cat_cols])
cat_feature_names = encoder.get_feature_names_out(cat_cols)
X_encoded = pd.DataFrame(encoded_cats, columns=cat_feature_names)

# Drop original categorical columns and merge encoded ones
X = X.drop(columns=cat_cols)
X = pd.concat([X, X_encoded], axis=1)

# Standardize numerical features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Save preprocessors for testing
joblib.dump(num_imputer, "num_imputer.pkl")
joblib.dump(cat_imputer, "cat_imputer.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save numerical and categorical columns for testing
import joblib
joblib.dump(num_cols, "num_cols.pkl")
joblib.dump(cat_cols, "cat_cols.pkl")
print("Numerical and categorical columns saved!")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Save the list of columns used in the training dataset
import joblib
joblib.dump(X.columns.tolist(), "training_columns.pkl")
print("Training columns saved!")

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train the model
num_round = 100
bst = xgb.train(params, dtrain, num_round, evals=[(dval, 'validation')])

# Save the model
bst.save_model("xgboost_model.model")
print("XGBoost model saved!")




