import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

print("Train Data:")
print(train_df.head())

print("\nTest Data:")
print(test_df.head())



cat_cols = train_df.select_dtypes(include=["object"]).columns  # Categorical features
num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns  # Numerical features
num_cols = num_cols.drop("Price")  # Exclude target column



# Fill missing values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Encode categorical variables
encoder = OneHotEncoder(handle_unknown="ignore")

# Scale numerical data
scaler = StandardScaler()

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", num_imputer), ("scaler", scaler)]), num_cols),
        ("cat", Pipeline([("imputer", cat_imputer), ("encoder", encoder)]), cat_cols),
    ]
)



X_train = train_df.drop(columns=["Price"])  # Features
y_train = train_df["Price"]  # Target

X_test = test_df  # Features from test set (No target variable)



dt_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(max_depth=10, random_state=42))
])

dt_model.fit(X_train, y_train)



y_pred_test = dt_model.predict(X_test)



submission = pd.DataFrame({"id": test_df["id"], "Predicted Price": y_pred_test})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv!")





