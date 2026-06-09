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


print(train.columns)



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

print("âœ… Loaded data")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Fix missing values
print("\nMissing values before fixing:\n", train.isnull().sum())
train.fillna(train.median(numeric_only=True), inplace=True)
test.fillna(test.median(numeric_only=True), inplace=True)
print("\nâœ… Missing values fixed.")
print("Missing values after fixing:\n", train.isnull().sum())

# Heatmap
plt.figure(figsize=(12, 8))
corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.close()
print("\nğŸ“Š Heatmap saved as heatmap.png")

# Prepare data
X = train.drop(columns=["id", "Listening_Time_minutes"])
y = train["Listening_Time_minutes"]
X_test = test.drop(columns=["id"])

# Columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Pipelines
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer([
    ("num", num_transformer, numerical_cols),
    ("cat", cat_transformer, categorical_cols)
])

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in tqdm(models.items(), desc="Training Models"):
    print(f"\nğŸ”§ Training: {name}")
    start = time.time()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)

    preds_val = pipeline.predict(X_val)
    rmse_val = mean_squared_error(y_val, preds_val, squared=False)

    duration = time.time() - start
    print(f"âœ… {name} RMSE: {rmse_val:.4f} | Time: {round(duration, 2)}s")
    results[name] = (pipeline, rmse_val)

# Select best model
best_model_name = min(results, key=lambda k: results[k][1])
best_model = results[best_model_name][0]
print(f"\nğŸ�† Best Model: {best_model_name} with RMSE {results[best_model_name][1]:.4f}")

# Predict on test set
final_preds = best_model.predict(X_test)
submission["Listening_Time_minutes"] = final_preds
submission.to_csv("final_submission.csv", index=False)
print("\nğŸ“� Submission saved to final_submission.csv")



# Calculate mean Listening_Time_minutes for each Podcast_Name
podcast_target_mean = train.groupby("Podcast_Name")["Listening_Time_minutes"].mean()

# Map to train and test
train["Podcast_Name_TE"] = train["Podcast_Name"].map(podcast_target_mean)
test["Podcast_Name_TE"] = test["Podcast_Name"].map(podcast_target_mean)

# For unseen Podcast_Name in test set, fill with global mean
global_mean = train["Listening_Time_minutes"].mean()
test["Podcast_Name_TE"].fillna(global_mean, inplace=True)



# ğŸ§  Feature Engineering: Encode time and day manually
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

time_map = {
    "Early Morning": 5,
    "Morning": 8,
    "Afternoon": 13,
    "Evening": 18,
    "Night": 21,
    "Late Night": 23
}

def encode_features(df):
    df["Publication_Day_Encoded"] = df["Publication_Day"].map(day_map)
    df["Publication_Time_Encoded"] = df["Publication_Time"].map(time_map)
    return df

# Apply encoding to train and test
train = encode_features(train)
test = encode_features(test)



# Drop unused or text-heavy columns
drop_cols = ["id", "Episode_Title", "Publication_Day", "Publication_Time"]
X = train.drop(columns=drop_cols + ["Listening_Time_minutes"])
y = train["Listening_Time_minutes"]
X_test = test.drop(columns=drop_cols)

# Recalculate column types
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()



# Pipelines
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, numerical_cols),
    ("cat", cat_transformer, categorical_cols)
])

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Add new models to the dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

results = {}
for name, model in tqdm(models.items(), desc="Training Models"):
    print(f"\nğŸ”§ Training: {name}")
    start = time.time()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)

    preds_val = pipeline.predict(X_val)
    rmse_val = mean_squared_error(y_val, preds_val, squared=False)

    duration = time.time() - start
    print(f"âœ… {name} RMSE: {rmse_val:.4f} | Time: {round(duration, 2)}s")
    results[name] = (pipeline, rmse_val)

# Pick best model
best_model_name = min(results, key=lambda k: results[k][1])
best_model = results[best_model_name][0]
print(f"\nğŸ�† Best Model: {best_model_name} with RMSE {results[best_model_name][1]:.4f}")

# Final prediction
final_preds = best_model.predict(X_test)
submission["Listening_Time_minutes"] = final_preds
submission.to_csv("final_submission.csv", index=False)
print("\nğŸ“� Submission saved to final_submission.csv")



!pip install lightgbm xgboost -q



from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Add new models to the dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}





