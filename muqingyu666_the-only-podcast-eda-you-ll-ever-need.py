import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Set a consistent style for visualizations
plt.style.use("seaborn-v0_8") # My personal taste
sns.set_palette("husl")

# Set global font family for plots
plt.rcParams["font.family"] = "times new roman"


# Load the dataset (I only EDA train dataset here since this notebook is for EDA)
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


# Check the number of rows and columns
print(f"Dataset Shape: {train_df.shape}")

# Display data types and check for missing values
print("\nData Info:")
train_df.info()

# Show basic statistics for numerical features
print("\nNumerical Features Summary:")
display(train_df.describe())

# Display the first few rows to inspect the data
print("\nFirst 10 Rows of the Dataset:")
display(train_df.head(10))


# List of numerical features
numerical_features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
    "Listening_Time_minutes",
]

# Plot histograms and box plots for each numerical feature
for feature in numerical_features:
    plt.figure(figsize=(12, 5))

    # Histogram with KDE (Kernel Density Estimate)
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    # Box plot to identify outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    # Print additional statistics
    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {train_df[feature].skew():.2f}")
    print(f"Number of Missing Values: {train_df[feature].isnull().sum()}")


# List of categorical features
categorical_features = [
    "Podcast_Name",
    "Episode_Title",
    "Genre",
    "Publication_Day",
    "Publication_Time",
]

# Plot bar charts for each categorical feature
for feature in categorical_features:
    plt.figure(figsize=(10, 6))

    if feature in ["Podcast_Name", "Episode_Title"]:
        # For features with many unique values, plot top 10 categories
        top_categories = train_df[feature].value_counts().nlargest(10)
        sns.barplot(x=top_categories.index, y=top_categories.values)
        plt.title(f"Top 10 {feature} Categories")
    else:
        # For features with fewer categories, plot all
        sns.countplot(
            x=train_df[feature], order=train_df[feature].value_counts().index
        )
        plt.title(f"Distribution of {feature}")

    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    # Print the number of unique values
    print(f"Number of Unique {feature}: {train_df[feature].nunique()}")
    print(f"Missing Values in {feature}: {train_df[feature].isnull().sum()}")


# Scatter plots for numerical features vs. Label
for feature in numerical_features[:-1]:  # Exclude Label itself
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=train_df[feature], y=train_df["Listening_Time_minutes"], alpha=0.5
    )
    plt.title(f"{feature} vs. Listening_Time_minutes")
    plt.xlabel(feature)
    plt.ylabel("Listening_Time_minutes")
    plt.show()

# Correlation matrix for numerical features
correlation_matrix = train_df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# Box plots for categorical features vs. Label
for feature in categorical_features:
    if feature not in [
        "Podcast_Name",
        "Episode_Title",
    ]:  # Skip high-cardinality features
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_df[feature], y=train_df["Listening_Time_minutes"])
        plt.title(f"{feature} vs. Listening_Time_minutes")
        plt.xlabel(feature)
        plt.ylabel("Listening_Time_minutes")
        plt.xticks(rotation=45)
        plt.show()


# Check for missing values in the dataset
print("Missing Values per Column:")
print(train_df.isnull().sum())

# Example imputation for numerical features (if missing values exist)
from sklearn.impute import SimpleImputer

# Impute numerical features with median
num_imputer = SimpleImputer(strategy="median")
train_df[numerical_features] = num_imputer.fit_transform(
    train_df[numerical_features]
)

# Impute categorical features with mode (most frequent value)
cat_imputer = SimpleImputer(strategy="most_frequent")
train_df[categorical_features] = cat_imputer.fit_transform(
    train_df[categorical_features]
)

# Verify no missing values remain
print("\nMissing Values After Imputation:")
print(train_df.isnull().sum())




