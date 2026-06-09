# import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the data
file_path = "/kaggle/input/playground-series-s5e2/train.csv"
df = pd.read_csv(file_path)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Define features and target variable
X = df.drop(columns=["id", "Price"])
y = df["Price"]

# Identify categorical and numerical columns
categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_cols = ["Compartments", "Weight Capacity (kg)"]

# Preprocessing: One-hot encode categorical variables, impute and scale numerical variables
num_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ("num", num_preprocessor, numerical_cols),
    ("cat", cat_preprocessor, categorical_cols)
])

# Define the model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on validation set
y_pred = model.predict(X_valid)

# Evaluate model performance
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"RMSE on validation set: {rmse:.4f}")

# Generate predictions for test set (assuming a separate test.csv is provided)
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Handle missing values in test data
test_df.fillna(test_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.mode().iloc[0], inplace=True)

test_ids = test_df["id"]
X_test = test_df.drop(columns=["id"])
test_predictions = model.predict(X_test)

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("Data Distributions and Relationships", fontsize=16)

sns.histplot(df["Price"], bins=30, kde=True, ax=axes[0, 0]).set(title="Price Distribution")
sns.histplot(df["Weight Capacity (kg)"], bins=30, kde=True, ax=axes[0, 1]).set(title="Weight Capacity Distribution")
sns.boxplot(x=df["Size"], y=df["Price"], ax=axes[0, 2]).set(title="Size vs Price")
sns.scatterplot(x=df["Compartments"], y=df["Price"], ax=axes[1, 0]).set(title="Compartments vs Price")
sns.boxplot(x=df["Waterproof"], y=df["Price"], ax=axes[1, 1]).set(title="Waterproof vs Price")
sns.violinplot(x=df["Laptop Compartment"], y=df["Price"], ax=axes[1, 2]).set(title="Laptop Compartment vs Price")
sns.countplot(x=df["Brand"], order=df["Brand"].value_counts().index, ax=axes[2, 0]).set(title="Brand Distribution")
sns.countplot(x=df["Material"], order=df["Material"].value_counts().index, ax=axes[2, 1]).set(title="Material Distribution")
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=axes[2, 2]).set(title="Feature Correlation")

plt.tight_layout()
plt.show()

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Price": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


