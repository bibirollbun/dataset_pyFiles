import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Create empty list for data paths
data_paths = []

# Collect paths
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data_paths.append(os.path.join(dirname, filename))

# Specify paths
train_path = data_paths[1]
test_path = data_paths[2]

# Read data
df = pd.read_csv(train_path).drop("id", axis=1)
test = pd.read_csv(test_path).drop("id", axis=1)


# View first few rows of train
df.head()


df.info()


# Count of duplicate rows
df.duplicated().sum()


cat_cols = [col for col in df.columns if df[col].dtype == "object"]

for col in cat_cols:
    print("="*10 + col.upper() + "="*10)
    print(df[col].unique())
    print()


# Summary statistics
df.describe()


def remove_outliers(df):
    """
    Removes outliers under each numerical variable using the IQR rule.

    Args:
    -----
    df (pd.DataFrame): Dataset to be cleaned.

    Returns:
    --------
    clean_df (pd.DataFrame): Dataset with outliers removed.
    """
    # Get numerical variables
    num_cols = [col for col in df.columns if df[col].dtypes in ("int64", "float64")]

    # Create clean dataframe
    clean_df = df.copy()

    # Apply IQR Rule to each variable
    for col in num_cols:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers under variable
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    return clean_df

clean_df = remove_outliers(df)
clean_df.info()


for col in cat_cols:
    # Count number of categories in col
    cat_counts = df[col].value_counts(normalize=True).reset_index()

    # Create columns for counts
    cat_counts.columns = [col, "Frequency"]
    
    # Plot bar plot
    plt.figure(figsize=(14,6))
    sns.barplot(data=cat_counts, x=col, y="Frequency", color="skyblue", edgecolor="black")
    plt.show()


num_cols = [col for col in df.columns if df[col].dtypes in ("int64", "float64")]

for col in num_cols:
    # Plot bar plot
    plt.figure(figsize=(14,6))
    sns.boxplot(data=df, x=col, color="skyblue")
    plt.show()


df.head()


# Plot boxplots by fertilizer
for col in num_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="Fertilizer Name", y=col, data=df, palette="Set2")
    plt.title(f"Boxplot of {col} by Fertilizer Name")
    plt.tight_layout()
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, hue="Fertilizer Name", data=df, palette="Set2")
    plt.title(f"Count of {col} by Fertilizer Name")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


sns.pairplot(df[num_cols + ["Fertilizer Name"]].sample(20000, random_state=42), 
             hue="Fertilizer Name", corner=True, plot_kws={"alpha":0.6})
plt.show()

