# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load the data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
extra_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
df_combined = pd.concat([train_df, extra_df], ignore_index=True)
X_full = df_combined.drop(columns=["Price"])
y_full = df_combined["Price"]
X_test = test_df.drop(columns=["id"])
test_ids = test_df["id"]


def add_features(df):
    df = df.copy()

    weight_cleaned = df["Weight Capacity (kg)"].fillna(0).clip(lower=0)
    df["Log Weight"] = np.log1p(weight_cleaned)
    df["HasSpecialMaterial"] = df["Material"].isin(["Leather", "Carbon Fiber"]).astype(int)

    return df

feature_engineer = FunctionTransformer(add_features)


categorical_cols = ["Size", "Laptop Compartment", "Waterproof", "Style", "Color", "Brand", "Material"]
df_encoded = add_features(df_combined.copy())
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_encoded[categorical_cols] = encoder.fit_transform(df_encoded[categorical_cols])
# Compute correlations
correlation_matrix = df_encoded.corr(numeric_only=True)
correlation_with_price = correlation_matrix["Price"].sort_values(key=abs, ascending=False)
print(correlation_with_price)


numerical_features = ["Weight Capacity (kg)", "Log Weight"]
binary_features = ["HasSpecialMaterial"]
target_encode_features = ["Brand"]
onehot_features = []  # Dropped due to weak correlation

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

binary_transformer = SimpleImputer(strategy="most_frequent")

target_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("targetenc", TargetEncoder())])

# Combined Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("bin", binary_transformer, binary_features),
    ("target", target_transformer, target_encode_features)])


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Define the ensemble models
rf_model = RandomForestRegressor(
    n_estimators=200,  # Increased number of trees
    random_state=1231,
    max_depth=15,
    min_samples_leaf=3,
    min_samples_split=6,
    n_jobs=-1  # Use all available cores
)

gb_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    random_state=1231,
    loss='squared_error'
)

# Create pipelines for each model
pipeline_rf = Pipeline(steps=[
    ("feature_engineering", FunctionTransformer(add_features)),
    ("preprocessor", preprocessor),
    ("model", rf_model)])

pipeline_gb = Pipeline(steps=[
    ("feature_engineering", FunctionTransformer(add_features)),
    ("preprocessor", preprocessor),
    ("model", gb_model)])

# Fit the ensemble models on all training data
pipeline_rf.fit(X_full, y_full)
pipeline_gb.fit(X_full, y_full)


# Evaluate on training set (for RMSE insight)
X_full_fe = add_features(X_full)
y_pred_train_rf = pipeline_rf.predict(X_full_fe)
rmse_train_rf = np.sqrt(mean_squared_error(y_full, y_pred_train_rf))
print(f"RMSE on full training data (Random Forest): {rmse_train_rf:.4f}")

y_pred_train_gb = pipeline_gb.predict(X_full_fe)
rmse_train_gb = np.sqrt(mean_squared_error(y_full, y_pred_train_gb))
print(f"RMSE on full training data (Gradient Boosting): {rmse_train_gb:.4f}")

# Make predictions on the test set with both models
X_test_fe = add_features(X_test)
y_pred_test_rf = pipeline_rf.predict(X_test_fe)
y_pred_test_gb = pipeline_gb.predict(X_test_fe)

# Simple Averaging Ensemble
y_pred_ensemble = (y_pred_test_rf + y_pred_test_gb) / 2

# Create submission file with the ensemble predictions
submission = pd.DataFrame({
    "id": test_ids,
    "Price": y_pred_ensemble})
submission.to_csv("submission.csv", index=False)
print("Submission file saved (using ensemble predictions)!")

