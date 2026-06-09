# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetic defaults
sns.set(style="whitegrid", palette="Set2")
plt.rcParams['figure.figsize'] = (16, 10)

import warnings
warnings.filterwarnings('ignore')




# Load the dataset
train_path = "/kaggle/input/playground-series-s5e10/train.csv"
df = pd.read_csv(train_path)

# Preview the first 5 rows
df.head()


# Shape of the dataset
print(f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns.")

# Data types and non-null counts
df.info()


# Check for missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    print("Columns with missing values:\n", missing)
else:
    print("âœ… No missing values found.")

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")


# Describe numerical columns
df.describe()


# Identify categorical columns
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

# Show unique values for each categorical column
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts(normalize=True).round(3), '\n')


num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, bins=30, ax=axes[i], color="skyblue")
    axes[i].set_title(f"Distribution of {col}", fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

# Hide last empty plot if odd number
fig.delaxes(axes[-1])
plt.tight_layout()
plt.suptitle("ğŸ“Š Distribution of Numerical Features", fontsize=16, y=1.02)
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()
cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

for i, col in enumerate(cols):
    sns.boxplot(x=col, y='accident_risk', data=df, ax=axes[i], palette="Set3")
    axes[i].set_title(f"{col} vs Accident Risk", fontsize=12)

plt.tight_layout()
plt.suptitle("ğŸ“¦ Boxplots of Features vs Accident Risk", fontsize=16, y=1.02)
plt.show()


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
            'road_signs_present', 'public_road', 'holiday', 'school_season']

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, ax=axes[i], order=df[col].value_counts().index)
    axes[i].set_title(f"Count Plot: {col}")
    axes[i].tick_params(axis='x', rotation=30)

# Remove empty subplot if needed
if len(cat_cols) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.suptitle("ğŸ§¾ Categorical Feature Distributions", fontsize=18, y=1.02)
plt.show()


# Correlation matrix
corr = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("ğŸ”— Correlation Heatmap", fontsize=16)
plt.show()


target_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(target_cols):
    sns.boxplot(x=col, y='accident_risk', data=df, ax=axes[i], palette="Pastel1")
    axes[i].set_title(f"Accident Risk by {col}")
    axes[i].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.suptitle("ğŸ�¯ Target Variable vs Categorical Features", fontsize=18, y=1.02)
plt.show()


# Generate some basic insights
print("âœ… Key Observations:")

# Distribution observations
if df['accident_risk'].skew() > 1:
    print("- Accident risk is right-skewed; most samples have low risk.")

# Relationship to num_reported_accidents
print("- Accident risk appears correlated with number of reported accidents.")

# Feature balance
print("- Features like 'road_type', 'lighting', and 'weather' have multiple balanced categories.")

# Check boolean distribution
for col in df.select_dtypes('bool'):
    print(f"- {col}: {df[col].mean():.2f} True ratio")


# Feature engineering suggestions:
suggestions = {
    "road_type": "One-hot encode or label encode",
    "curvature": "Consider binning curvature into categories (e.g., flat, moderate, sharp)",
    "time_of_day": "Group into broader categories if needed (e.g., day vs night)",
    "weather": "Create a binary feature: is_adverse_weather",
    "num_reported_accidents": "Log-transform if skewed",
    "interaction": "Create interaction terms like curvature Ã— speed_limit",
}

# Display suggestions
for feature, idea in suggestions.items():
    print(f"{feature}: {idea}")

