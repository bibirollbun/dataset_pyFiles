import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error




train_df = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/test.csv")
test_ids = test_df["id"] 


train_df.head()


test_df.head()


# Log-transform revenue and budget
train_df["log_revenue"] = np.log1p(train_df["revenue"])
train_df["log_budget"] = np.log1p(train_df["budget"])




# Convert release_date to datetime with explicit format handling
train_df["release_date"] = pd.to_datetime(train_df["release_date"], format='%m/%d/%y', errors='coerce')

# Adjust years if incorrectly parsed as future dates (e.g., 2099 → 1999)
train_df["release_year"] = train_df["release_date"].dt.year
train_df.loc[train_df["release_year"] > 2025, "release_year"] -= 100

# Extract release month
train_df["release_month"] = train_df["release_date"].dt.month



# Fill missing runtime with median
train_df["runtime"] = train_df["runtime"].fillna(train_df["runtime"].median())


# Function to count elements in JSON-like fields
def count_json_elements(json_str):
    if pd.isna(json_str) or json_str == "[]":
        return 0
    try:
        elements = literal_eval(json_str)
        return len(elements) if isinstance(elements, list) else 0
    except:
        return 0


# Extract numeric features
train_df["num_genres"] = train_df["genres"].apply(count_json_elements)
train_df["num_production_companies"] = train_df["production_companies"].apply(count_json_elements)
train_df["num_keywords"] = train_df["Keywords"].apply(count_json_elements)

# One-hot encode categorical variables
train_df = pd.get_dummies(train_df, columns=["original_language"], drop_first=True)

# Drop unused columns
drop_columns = [
    "id", "belongs_to_collection", "homepage", "imdb_id", "original_title",
    "overview", "poster_path", "status", "tagline", "title", "genres",
    "production_companies", "production_countries", "spoken_languages",
    "Keywords", "cast", "crew", "release_date", "revenue"
]
train_df.drop(columns=drop_columns, inplace=True)


# Define features and target
X = train_df.drop(columns=["log_revenue"])
y = train_df["log_revenue"]

# Fix missing values in X
X = X.apply(lambda col: col.fillna(col.median()) if col.dtype != "O" else col)
X.fillna(0, inplace=True)

# Verify missing values are handled
print("Missing values in X before training:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# Split into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict on validation set and calculate RMSLE
y_pred = model.predict(X_valid)
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))
print(f"Validation RMSLE: {rmsle:.5f}")


# Apply same preprocessing to test data
test_df["log_budget"] = np.log1p(test_df["budget"])
test_df["release_date"] = pd.to_datetime(test_df["release_date"], format='%m/%d/%y', errors='coerce')
test_df["release_year"] = test_df["release_date"].dt.year
test_df.loc[test_df["release_year"] > 2025, "release_year"] -= 100
test_df["release_month"] = test_df["release_date"].dt.month
test_df["runtime"] = test_df["runtime"].fillna(train_df["runtime"].median())
test_df["num_genres"] = test_df["genres"].apply(count_json_elements)
test_df["num_production_companies"] = test_df["production_companies"].apply(count_json_elements)
test_df["num_keywords"] = test_df["Keywords"].apply(count_json_elements)
test_df = pd.get_dummies(test_df, columns=["original_language"], drop_first=True)


# Align test set with training features
missing_cols = set(X.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0
test_df = test_df[X.columns]


# Check for NaNs in test data
print("Missing values in test_df before prediction:")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# Fix missing values in test data
test_df = test_df.apply(lambda col: col.fillna(col.median()) if col.dtype != "O" else col)
test_df.fillna(0, inplace=True)

# Ensure test_df has the same features as X (drop extra columns)
test_df = test_df[X.columns]

# Verify NaNs are removed
print("Missing values after fixing test_df:")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# Predict revenue and convert back from log scale
test_df["predicted_log_revenue"] = model.predict(test_df)
test_df["revenue"] = np.expm1(test_df["predicted_log_revenue"])

# Prepare submission file with correct test IDs
submission = test_df[["revenue"]].copy()
submission.insert(0, "id", test_ids)  # Use correct test IDs from test.csv
submission.to_csv("submission.csv", index=False)

print("Prediction complete. Submission file saved as 'submission.csv'.")




