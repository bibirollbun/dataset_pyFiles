import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Testing Notebook for XGBoost

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib

# Define selected columns (same as training)
selected_columns = [
    "prim_disease_hct", "hla_match_b_low", "prod_type", "year_hct", "obesity", 
    "donor_age", "prior_tumor", "gvhd_proph", "sex_match", "comorbidity_score", 
    "karnofsky_score", "donor_related", "age_at_hct"
]

# Load test dataset
test_file_path = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
df_test = pd.read_csv(test_file_path)

# Keep only selected columns
df_test = df_test[selected_columns]

# Load preprocessors
num_imputer = joblib.load("/kaggle/input/training-hct-survival/num_imputer.pkl")
cat_imputer = joblib.load("/kaggle/input/training-hct-survival/cat_imputer.pkl")
encoder = joblib.load("/kaggle/input/training-hct-survival/encoder.pkl")
scaler = joblib.load("/kaggle/input/training-hct-survival/scaler.pkl")

# Load numerical and categorical columns
num_cols = joblib.load("/kaggle/input/training-hct-survival/num_cols.pkl")
cat_cols = joblib.load("/kaggle/input/training-hct-survival/cat_cols.pkl")
print("Numerical and categorical columns loaded!")

# Load the list of columns used in the training dataset
training_columns = joblib.load("/kaggle/input/training-hct-survival/training_columns.pkl")

# Ensure all categorical columns exist before transformation
for col in cat_cols:
    if col not in df_test.columns:
        df_test[col] = np.nan  # Fill missing categorical columns

# Handle missing values for categorical columns
df_test[cat_cols] = cat_imputer.transform(df_test[cat_cols])

# Encode categorical features
encoded_cats_test = encoder.transform(df_test[cat_cols])
df_test_encoded = pd.DataFrame(encoded_cats_test, columns=encoder.get_feature_names_out(cat_cols))

# Drop original categorical columns and merge encoded ones
df_test = df_test.drop(columns=cat_cols, errors='ignore')  # Avoid KeyError
df_test = pd.concat([df_test, df_test_encoded], axis=1)

# Standardize numerical features
df_test[num_cols] = scaler.transform(df_test[num_cols])

# Convert to DMatrix
dtest = xgb.DMatrix(df_test)

# Load the trained model
bst = xgb.Booster()
bst.load_model("/kaggle/input/training-hct-survival/xgboost_model.model")

# Make predictions
predictions = bst.predict(dtest)

# Save predictions to a CSV file
submission = pd.DataFrame({
    "ID": df_test.index,
    "prediction": predictions
})
submission.to_csv("submission.csv", index=False)
print("XGBoost predictions saved!")




