import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load training data
data = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")

# Handling missing values
data["fuel_type"] = data["fuel_type"].fillna("Unknown")
data["accident"] = data["accident"].fillna("Unknown")
data["clean_title"] = data["clean_title"].fillna("Unknown")

# Feature Engineering
data["car_age"] = 2025 - data["model_year"]
data.drop("model_year", axis=1, inplace=True)

# Extract Horsepower and Engine Size from 'engine' column
data["horsepower"] = data["engine"].apply(lambda x: float(re.search(r"(\d+(\.\d+)?)HP", str(x)).group(1)) if pd.notna(x) and re.search(r"(\d+(\.\d+)?)HP", str(x)) else None)
data["engine_size"] = data["engine"].apply(lambda x: float(re.search(r"(\d+(\.\d+)?)L", str(x)).group(1)) if pd.notna(x) and re.search(r"(\d+(\.\d+)?)L", str(x)) else None)
data.drop("engine", axis=1, inplace=True)

# Convert categorical variables to binary
data["accident"] = data["accident"].apply(lambda x: 0 if x == "None reported" else 1)
data["clean_title"] = data["clean_title"].apply(lambda x: 1 if x == "Yes" else 0)

# Encoding Categorical Variables
categorical_cols = ["brand", "model", "fuel_type", "transmission", "ext_col", "int_col"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoder for test set

# One-Hot Encoding for low-cardinality categorical features
data = pd.get_dummies(data, columns=["fuel_type", "transmission", "ext_col", "int_col"], drop_first=True)

# Save cleaned train data
train_df = data.copy()

# Load competition test dataset
test_df = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")

# Apply same feature engineering as train_df
test_df["car_age"] = 2025 - test_df["model_year"]
test_df.drop("model_year", axis=1, inplace=True)

# Extract numerical features from 'engine'
test_df["horsepower"] = test_df["engine"].apply(lambda x: float(re.search(r"(\d+(\.\d+)?)HP", str(x)).group(1)) if pd.notna(x) and re.search(r"(\d+(\.\d+)?)HP", str(x)) else None)
test_df["engine_size"] = test_df["engine"].apply(lambda x: float(re.search(r"(\d+(\.\d+)?)L", str(x)).group(1)) if pd.notna(x) and re.search(r"(\d+(\.\d+)?)L", str(x)) else None)
test_df.drop("engine", axis=1, inplace=True)

# Convert categorical variables to binary in test_df
test_df["accident"] = test_df["accident"].apply(lambda x: 0 if x == "None reported" else 1)
test_df["clean_title"] = test_df["clean_title"].apply(lambda x: 1 if x == "Yes" else 0)

# Apply Label Encoding to test_df safely
for col in categorical_cols:
    if col in test_df.columns:  # Ensure column exists before transformation
        test_df[col] = test_df[col].apply(lambda x: label_encoders[col].classes_[0] if x not in label_encoders[col].classes_ else x)
        test_df[col] = label_encoders[col].transform(test_df[col])

# One-Hot Encoding for test data (ensuring same structure as train_df)
test_df = pd.get_dummies(test_df, columns=["fuel_type", "transmission", "ext_col", "int_col"], drop_first=True)

# Ensure test set has the same feature columns as train set
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0  # Add missing columns with default 0

# Ensure column order matches between train and test
X_train = train_df.drop(columns=["id", "price"])
y_train = train_df["price"]
X_test = test_df.drop(columns=["id"], errors="ignore")[X_train.columns]  # Align columns

# Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)
test_df["price"] = xgb_model.predict(X_test)

# Prepare submission file
submission = test_df[["id", "price"]]
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("XGBoost Model Training Completed. Submission file saved as 'submission.csv'.")


