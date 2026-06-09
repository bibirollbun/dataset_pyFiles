import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

# Display dataset info
print("Train Data Info:")
print(train_df.info())
print("\nTest Data Info:")
print(test_df.info())

# Preview Data
print("Train Data Preview:")
print(train_df.head())

print("Test Data Preview:")
print(test_df.head())


# Count missing values per column
missing_train = train_df.isnull().sum().sort_values(ascending=False)
missing_test = test_df.isnull().sum().sort_values(ascending=False)

# Display missing values
print("\nMissing Values in Train Data:\n", missing_train[missing_train > 0])
print("\nMissing Values in Test Data:\n", missing_test[missing_test > 0])

# Visualizing missing values
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Data Heatmap - Train Set")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(test_df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Data Heatmap - Test Set")
plt.show()


# Separate categorical and numerical columns
categorical_cols = train_df.select_dtypes(include=["object"]).columns
numerical_cols = train_df.select_dtypes(include=["int64", "float64"]).columns

print("\nCategorical Columns:", categorical_cols.tolist())
print("\nNumerical Columns:", numerical_cols.tolist())


# Select numerical columns **only present** in both datasets
numerical_cols = list(set(train_df.select_dtypes(include=["int64", "float64"]).columns) & 
                       set(test_df.select_dtypes(include=["int64", "float64"]).columns))

print("Numerical columns used:", numerical_cols)

# Fill missing values only for common numerical columns
train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].median())
test_df[numerical_cols] = test_df[numerical_cols].fillna(test_df[numerical_cols].median())



# Summary statistics for numerical features
print("\nNumerical Feature Summary:")
print(train_df[numerical_cols].describe())

# Summary of categorical variables
print("\nCategorical Feature Summary:")
print(train_df[categorical_cols].describe())


# Convert categorical columns to numerical using one-hot encoding
train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

print("Encoded Train Data Shape:", train_encoded.shape)
print("Encoded Test Data Shape:", test_encoded.shape)


train_df[numerical_cols].hist(figsize=(12, 10), bins=30)
plt.suptitle("Distribution of Numerical Features")
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(train_encoded.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=train_df, palette="Set2")
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")
    plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df[numerical_cols])
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Features (Outliers Detection)")
plt.show()


train_encoded.to_csv("cleaned_train.csv", index=False)
test_encoded.to_csv("cleaned_test.csv", index=False)
print("Cleaned datasets saved successfully!")

