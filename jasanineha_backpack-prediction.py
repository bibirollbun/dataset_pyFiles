# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import optuna


# Load train and test datasets

# Load the additional dataset
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Combine the two datasets row-wise
train = pd.concat([train, train_extra], axis=0, ignore_index=True)


# Display the first 5 rows of train and test datasets
print("Train Dataset:")
train.head()


print("\nTest Dataset:")
test.head()


# Print dataset shapes
print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")


# Check data types and missing values
print("\nTrain Data Info:")
print(train.info())


print("\nTest Data Info:")
print(test.info())


# Describe numerical columns
print("\nTrain Summary Statistics:")
print(train.describe())


print("\nTest Summary Statistics:")
print(test.describe())


# Calculate missing values in datasets
missing_train = train.isnull().sum() / len(train) * 100
missing_test = test.isnull().sum() / len(test) * 100


# Display missing values greater than 0%
print("Missing values in train dataset (%):")
print(missing_train[missing_train > 0].sort_values(ascending=False))


print("\nMissing values in test dataset (%):")
print(missing_test[missing_test > 0].sort_values(ascending=False))


# Fill missing categorical values with mode
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

for col in categorical_features:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)


# Fill missing numerical values
train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean(), inplace=True)
test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean(), inplace=True)


print("Train Missing Values After All Imputations:")
print(train.isnull().mean().round(4))

print("\nTest Missing Values After All Imputations:")
print(test.isnull().mean().round(4))


train.head()


test.head()


# 1. Distribution of Price in Train Dataset
plt.figure(figsize=(8, 5))
sns.histplot(train['Price'], bins=30, kde=True, color='blue')
plt.title("Distribution of Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


# 2. Count of Different Brands
plt.figure(figsize=(10, 5))
sns.countplot(y=train['Brand'], order=train['Brand'].value_counts().index, palette="viridis")
plt.title("Count of Different Brands")
plt.xlabel("Count")
plt.ylabel("Brand")
plt.show()


# 3. Material Distribution
plt.figure(figsize=(10, 5))
sns.countplot(y=train['Material'], order=train['Material'].value_counts().index, palette="coolwarm")
plt.title("Distribution of Materials")
plt.xlabel("Count")
plt.ylabel("Material")
plt.show()


# 4. Price Distribution by Brand
plt.figure(figsize=(12, 6))
sns.boxplot(x='Brand', y='Price', data=train)
plt.xticks(rotation=45)
plt.title("Price Distribution by Brand")
plt.xlabel("Brand")
plt.ylabel("Price")
plt.show()


# 5. Size Distribution
plt.figure(figsize=(8, 5))
sns.countplot(y=train['Size'], order=train['Size'].value_counts().index, palette="pastel")
plt.title("Distribution of Bag Sizes")
plt.xlabel("Count")
plt.ylabel("Size")
plt.show()


# 6. Style Distribution
plt.figure(figsize=(10, 5))
sns.countplot(y=train['Style'], order=train['Style'].value_counts().index, palette="Set2")
plt.title("Distribution of Bag Styles")
plt.xlabel("Count")
plt.ylabel("Style")
plt.show()


# 7. Waterproof vs. Non-Waterproof Bags
plt.figure(figsize=(6, 4))
sns.countplot(x=train['Waterproof'], palette=["#3498db", "#e74c3c"])
plt.title("Count of Waterproof vs. Non-Waterproof Bags")
plt.xlabel("Waterproof")
plt.ylabel("Count")
plt.show()


# 8. Laptop Compartment Availability
plt.figure(figsize=(6, 4))
sns.countplot(x=train['Laptop Compartment'], palette=["#2ecc71", "#f1c40f"])
plt.title("Laptop Compartment Availability")
plt.xlabel("Laptop Compartment")
plt.ylabel("Count")
plt.show()


# 9. Color Distribution
plt.figure(figsize=(12, 5))
sns.countplot(y=train['Color'], order=train['Color'].value_counts().index, palette="coolwarm")
plt.title("Color Distribution of Bags")
plt.xlabel("Count")
plt.ylabel("Color")
plt.show()


# List of numerical columns
numerical_cols = ['Compartments', 'Weight Capacity (kg)', 'Price']

# Plot boxplots for numerical features
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=train[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()


# Detecting outliers using IQR
for col in numerical_cols:
    Q1 = train[col].quantile(0.25)  # 25th percentile
    Q3 = train[col].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile range

    # Define the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers
    outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
    print(f"Outliers in {col}: {outliers.shape[0]}")



# Check skewness of numerical features
numerical_cols = ['Compartments', 'Weight Capacity (kg)', 'Price']
skewness = train[numerical_cols].apply(skew)
print("Skewness of numerical features:\n", skewness)


categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]


# One-Hot Encoding (removes sparse format)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train[categorical_features]))
train_encoded.columns = encoder.get_feature_names_out(categorical_features)


test_encoded = pd.DataFrame(encoder.fit_transform(test[categorical_features]))
test_encoded.columns = encoder.get_feature_names_out(categorical_features)


# Drop original categorical columns and merge encoded ones
train = train.drop(columns=categorical_features)
train = pd.concat([train, train_encoded], axis=1)


# Drop original categorical columns and merge encoded ones
test = test.drop(columns=categorical_features)
test = pd.concat([test, test_encoded], axis=1)


train.head()


test.head()


numerical_features = ["Compartments", "Weight Capacity (kg)"]


scaler = StandardScaler()
train[numerical_features] = scaler.fit_transform(train[numerical_features])
test[numerical_features] = scaler.fit_transform(test[numerical_features])


# Features (X)
X = train.drop(["id", "Price"], axis=1)  # Drop non-feature columns
y = train["Price"]  # Use Price target


# Test data (X_test)
X_test = test.drop("id", axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    # Define hyperparameters to tune
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),  # Number of trees
        "max_depth": trial.suggest_int("max_depth", 3, 10),  # Depth of trees
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),  # Shrinkage
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),  # Fraction of samples
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),  # Fraction of features
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),  # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),  # L2 regularization
        "random_state": 42,
    }

    # Initialize and train the model
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

    # Predict and calculate RMSE
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse



# Create a study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Print the best parameters and RMSE
print("Best Parameters:", study.best_params)
print("Best RMSE:", study.best_value)


# Use the best parameters to train the final model
best_params = study.best_params
final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=True,
)


# Predict on validation set
val_preds = final_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {val_rmse:.4f}")


# Predict log prices for test data
test_preds = final_model.predict(X_test)
print(test_preds)


# Prepare submission
submission = pd.DataFrame({
    "id": test["id"],
    "Price": test_preds
})


# Ensure prices are rounded to three decimal places
submission["Price"] = submission["Price"].apply(lambda x: round(x, 3))

# Save to CSV
submission.to_csv("submission.csv", index=False, float_format="%.3f")


submission.head()

