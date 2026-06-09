import pandas as pd
import numpy as np

# Load the dataset
file_path = '/kaggle/input/jane-street-real-time-market-data-forecasting/features.csv'
df = pd.read_csv(file_path)

# Inspect the dataset
print("Dataset Shape:", df.shape)
print("Column Types Overview:")
print(df.dtypes.value_counts())
print("\nFirst Few Rows of the Dataset:")
print(df.head())

# Check for numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Numeric Columns: {numeric_columns}")
print(f"Non-Numeric Columns: {non_numeric_columns}")

# Attempt to convert non-numeric columns to numeric
for col in non_numeric_columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"Could not convert column {col} to numeric. Error: {e}")

# Re-check numeric columns after conversion
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Updated Numeric Columns: {numeric_columns}")

# Check for columns with variance
valid_numeric_columns = [col for col in numeric_columns if df[col].nunique() > 1]
print(f"Valid Numeric Columns (with variance): {valid_numeric_columns}")

# If no valid numeric columns, inspect non-numeric data
if not valid_numeric_columns:
    print("No valid numeric columns found. Investigating non-numeric columns...")
    print(df[non_numeric_columns].head())
else:
    # Proceed with analysis (e.g., correlation matrix, heatmaps)
    correlation_matrix = df[valid_numeric_columns].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/kaggle/input/jane-street-real-time-market-data-forecasting/responders.csv'
df = pd.read_csv(file_path)

# Check basic info about the dataset
print("Dataset Shape:", df.shape)
print("\nColumn Data Types:")
print(df.dtypes)
print("\nFirst Few Rows:")
print(df.head())
print("\nMissing Values Per Column:")
print(df.isnull().sum())

# Debug: Ensure DataFrame is not empty
if df.empty:
    raise ValueError("The dataset is empty. Please check the file.")

# Identify numeric and categorical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Debug: Ensure there are numeric columns
if numeric_columns.empty:
    print("No numeric columns found in the dataset.")
else:
    print("\nSummary of Numeric Columns:")
    print(df[numeric_columns].describe())

# Debug: Ensure there are categorical columns
if categorical_columns.empty:
    print("No categorical columns found in the dataset.")
else:
    print("\nSummary of Categorical Columns:")
    for col in categorical_columns:
        print(f"{col}: {df[col].nunique()} unique values")

# Handle Missing Values
if not numeric_columns.empty:
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Visualizations (if applicable)
if not numeric_columns.empty:
    for col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

if not categorical_columns.empty:
    for col in categorical_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Counts of {col}")
        plt.xticks(rotation=45)
        plt.show()

# Correlation Matrix (if numeric columns exist)
if len(numeric_columns) > 1:
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
    plt.title("Correlation Matrix of Numeric Features")
    plt.show()
else:
    print("Not enough numeric columns for correlation analysis.")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/sample_submission.csv"
df = pd.read_csv(file_path)

# Display basic information
print("Dataset Information:")
print(df.info())

# Display first few rows
print("\nFirst few rows:")
print(df.head())

# Replace inf and -inf values with NaN
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

# 1. Summary of numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
print("\nSummary of Numeric Columns:")
if not numeric_columns.empty:
    print(df[numeric_columns].describe())
else:
    print("No numeric columns found in the dataset.")

# 2. Summary of categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print("\nSummary of Categorical Columns:")
if not categorical_columns.empty:
    print(df[categorical_columns].describe())
else:
    print("No categorical columns found in the dataset.")

# 3. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Visualize the distribution of numeric columns
if not numeric_columns.empty:
    for col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()
else:
    print("No numeric columns to visualize.")

# 5. Correlation analysis for numeric columns
if len(numeric_columns) > 1:
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()
else:
    print("Not enough numeric columns for correlation analysis.")

# 6. Countplot for categorical columns (if any)
if not categorical_columns.empty:
    for col in categorical_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, palette="viridis")
        plt.title(f"Countplot of {col}")
        plt.xticks(rotation=45)
        plt.show()
else:
    print("No categorical columns to visualize.")

# Additional Insights
print("\nDataset Shape:")
print(df.shape)
print("\nUnique Values in Each Column:")
print(df.nunique())




