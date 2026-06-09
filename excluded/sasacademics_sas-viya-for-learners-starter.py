import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sasviya.ml.linear_model import LinearRegression


# We also set basic display and visualisation options to improve the readability of outputs and plots throughout the notebook
%matplotlib inline

# Display all rows and columns in output if needed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


train = pd.read_csv("/your-working-directory/domain_properties.csv")
print(train.head())


print("Data Type:\n", train.dtypes)
print(f"There are {train.shape[1]} columns and {train.shape[0]} rows in the train dataset.")
print("Column names and data type of each column:")
print(train.dtypes)
print("Summary statistics for the numeric columns:")
print(train.describe())
print("Checking for missing values in each column:")
print(train.isnull().sum())


# Identify numeric and categorical columns
numeric_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Numeric Cols:\n", numeric_cols)

train["date_sold"] = pd.to_datetime(train["date_sold"])
train["year_sold"] = train["date_sold"].dt.year
train["month_sold"] = train["date_sold"].dt.month
train["quarter_sold"] = train["date_sold"].dt.to_period("Q").astype(str)

train.drop(columns="date_sold", inplace=True)
categoric_cols = train.select_dtypes(include=["object"]).columns.tolist()

print("\nCategoric Cols:\n", categoric_cols)

train.tail()


plt.figure(figsize=(8, 5))
sns.histplot(train['price'], bins=30)
plt.title('Original House Price Distribution')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.show()


train_encoded = pd.get_dummies(train, columns=categoric_cols, drop_first=True)

# Convert boolean columns to integers
bool_cols = train_encoded.select_dtypes(include="bool").columns
train_encoded[bool_cols] = train_encoded[bool_cols].astype(int)

train_encoded.shape


X = train_encoded.drop("price", axis=1) 
y = train_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sasviya.ml.linear_model import Ridge, Lasso
from sasviya.ml.tree import ForestRegressor

# 1. Define all four models in a dictionary
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=0.1),
    "Lasso": Lasso(),
    "Random Forest": ForestRegressor(n_estimators=100, max_depth=5, random_state=42)
}

# 2. Train each model on the training set
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} model trained.")

# 3. Generate and store predictions on the test set for each model
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_test)


from sklearn.metrics import r2_score, mean_squared_error

for name, y_pred in predictions.items():
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{name} → R²: {r2:.3f}, RMSE: {rmse:.2f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="purple")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color="gray")
    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title(f"Actual vs Predicted Price: {name}")
    plt.show()


# Select the best model. For example, assume Ridge performed best
best_model = models["Ridge"]

# Generate predictions on the test set
final_predictions = best_model.predict(X_test)

# Create a submission DataFrame — using test set index as ID
submission_df = pd.DataFrame({
    "ID": X_test.index,
    "Predicted_Selling_Price": final_predictions
})

# Save to CSV
submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")

