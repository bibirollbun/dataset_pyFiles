import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


training_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv",index_col="id")
training_addon_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",index_col="id")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


training_data.head()


training_addon_data.head()


cumilative_training_data = pd.concat([training_data,training_addon_data],ignore_index=True)
cumilative_training_data.head()


print(f"Original Train Shape: {training_data.shape}, Addon Train Shape: {training_addon_data.shape}")
print(f"Merged Train Shape: {cumilative_training_data.shape}")


pd.set_option("display.float_format", "{:.2f}".format)
cumilative_training_data.describe()


cumilative_training_data.info()


duplicates = cumilative_training_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
missing_percentage = (cumilative_training_data.isnull().sum() / len(cumilative_training_data)) * 100
print("Missing Values Percentage:\n", missing_percentage)


plt.figure(figsize=(10, 5))
sns.histplot(cumilative_training_data["Price"], bins=50, kde=True, color="blue")
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Check skewness
from scipy.stats import skew
print("Skewness of Price:", skew(cumilative_training_data["Price"]))

# Boxplot for outlier detection
plt.figure(figsize=(8, 5))
sns.boxplot(x=cumilative_training_data["Price"], color="red")
plt.title("Boxplot of Price")
plt.show()


# Separate categorical and numerical columns
categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_cols = ["Compartments", "Weight Capacity (kg)", "Price"]

# Plot distributions for numerical features
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 3, i)
    sns.histplot(cumilative_training_data[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()



corr_matrix = cumilative_training_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# Count plots for categorical variables
plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 3, i)
    sns.countplot(y=cumilative_training_data[col], order=cumilative_training_data[col].value_counts().index)
    plt.title(f"Count of {col}")
plt.tight_layout()
plt.show()



# Price distribution across categories
plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=cumilative_training_data["Price"], y=cumilative_training_data[col])
    plt.title(f"Price Distribution by {col}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





