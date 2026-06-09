# import relevant libraries
import pandas as pd
import math

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


# drop id column
train_df = train_df.drop('id', axis=1)


train_df.shape


train_df.head()


print(train_df['poutcome'].value_counts(normalize=True))


train_df.describe()


train_df.info()


import seaborn as sns
import matplotlib.pyplot as plt


# Check for missing values
missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0]
if missing_values.values.any():
    plt.figure(figsize=(12,6))
    sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()
else:
    print("No missing values found.")


# columns as their data types
columns = list(train_df.columns)
numeric_columns = []
categorical_columns = []
for column in columns:
    dtype = train_df[column].dtype
    # code for separting numeric and object dtype columns
    if dtype in ['int64','float64']:
        numeric_columns.append(column)
    else:
        categorical_columns.append(column)

# let's see
print("Numeric Columns:", len(numeric_columns))
print(numeric_columns)
print("\nCategorical Columns:",len(categorical_columns))
print(categorical_columns)


# for categorical columns
for category in categorical_columns:
    value_counts = train_df[category].value_counts(dropna=False)
    print(value_counts)
    print("Uniques:",len(value_counts))
    print()


# for numerical columns
discrete_columns = []
continuous_columsn = []
for col in numeric_columns:
    dtype = train_df[col].dtype
    is_discrete = train_df[col].nunique() < 100
    if is_discrete:
        discrete_columns.append(col)
    else:
        continuous_columsn.append(col)
    print(f"{col}: {'Discrete' if is_discrete else 'Continuous'}")

print("\nDiscrete Columns:", discrete_columns)
print("\nContinuous Columns:", continuous_columsn)


# number of uniques
for d_col in discrete_columns:
    print(train_df[d_col].nunique(),"uniques in",d_col)


print(train_df['y'].describe())
print(train_df['y'].value_counts(normalize=True))


sns.countplot(x=train_df['y'])
plt.title("Class Distribution")  
plt.show()


# histogram for numeric columns
num_plots = len([col for col in numeric_columns if col != 'y'])
n_cols = 2
n_rows = math.ceil(num_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False)  # Create subplot grid

axes = axes.flatten()
plot_idx = 0
for num_cols in numeric_columns:
    if num_cols == 'y':
        continue
    sns.histplot(data=train_df, x=num_cols, bins=25, kde=True, color='purple', ax=axes[plot_idx])
    axes[plot_idx].set_title(f"Analysis of {num_cols}")
    plt.xlabel(num_cols)
    plt.ylabel("Count")
    plot_idx += 1

for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# bar charts for categorical columns
num_plots = len(categorical_columns)
n_cols = 2
n_rows = math.ceil(num_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False)  # Create subplot grid

axes = axes.flatten()
plot_idx = 0
for cat_cols in categorical_columns:
    if cat_cols == 'month':
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        sns.countplot(data=train_df, x='month', order=month_order, palette='viridis', ax=axes[plot_idx])
    else:
        sns.countplot(data=train_df, x=cat_cols, order=train_df[cat_cols].value_counts().index, palette='viridis', ax=axes[plot_idx])
    axes[plot_idx].set_title(f"Analysis of {cat_cols}")
    plt.xlabel(cat_cols)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plot_idx += 1

for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# density plot (KDE + histogram)
num_plots = len([col for col in numeric_columns if col != 'y'])
n_cols = 2
n_rows = math.ceil(num_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False)  # Create subplot grid

axes = axes.flatten()
plot_idx = 0
for num_cols in numeric_columns:
    if num_cols == 'y':
        continue
    sns.kdeplot(data=train_df, x=num_cols, hue='y', fill=True, common_norm=True, ax=axes[plot_idx])
    axes[plot_idx].set_title(f"{num_cols} distribution by y")
    plot_idx += 1

# remove empty subplots
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# stacked box plot of all categorical features with target feature
num_plots = len(categorical_columns)
n_cols = 2
n_rows = math.ceil(num_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False)

axes = axes.flatten()
plot_idx = 0
for cat_cols in categorical_columns:
    sns.histplot(data=train_df, x=cat_cols, hue='y', hue_order=[1,0], palette=['salmon', 'skyblue'], multiple='stack', shrink=0.5, ax=axes[plot_idx])
    axes[plot_idx].set_title(f"{cat_cols} distribution by y")
    plt.xlabel("Feature")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plot_idx += 1

# remove empty subplots
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# bar plot analysis for comparing median per class
num_plots = len([col for col in numeric_columns if col != 'y'])
n_cols = 2
n_rows = math.ceil(num_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False)

axes = axes.flatten()
plot_idx = 0
for num_cols in numeric_columns:
    if num_cols == 'y':
        continue
    sns.barplot(data=train_df, x='y', y=num_cols, estimator='median', ci=None, palette=['salmon', 'skyblue'], ax=axes[plot_idx])
    axes[plot_idx].set_title(f"Median of {num_cols} by y")
    plot_idx += 1

# remove empty subplots
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# bar plot analysis for comparing mean per class
num_plots = len([col for col in numeric_columns if col != 'y'])
n_cols = 2
n_rows = math.ceil(num_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False)

axes = axes.flatten()
plot_idx = 0
for num_cols in numeric_columns:
    if num_cols == 'y':
        continue
    sns.barplot(data=train_df, x='y', y=num_cols, estimator='mean', ci=None, palette=['salmon', 'skyblue'], ax=axes[plot_idx])
    axes[plot_idx].set_title(f"Mean of {num_cols} by y")
    plot_idx += 1

# remove empty subplots
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# Correlation analysis for numeric features
import numpy as np
corr_matrix = train_df[numeric_columns].corr()
print(corr_matrix)

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


# Outlier detection using boxplots for numeric columns
num_plots = len([col for col in numeric_columns if col != 'y'])
n_cols = 2
n_rows = math.ceil(num_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False)

axes = axes.flatten()
plot_idx = 0
for num_cols in numeric_columns:
    if num_cols == 'y':
        continue
    sns.boxplot(data=train_df, x='y', y=num_cols, ax=axes[plot_idx])
    axes[plot_idx].set_title(f"Boxplot for {num_cols}")
    plt.xlabel(num_cols)
    plot_idx += 1

# remove empty subplots
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# Unique values and cardinality
print('Unique values in categorical columns:')
for col in categorical_columns:
    print(f'{col}: {train_df[col].nunique()}')

print('\nUnique values in numeric columns:')
for col in numeric_columns:
    unique_vals = train_df[col].nunique()
    print(f'{col}: {unique_vals} (discrete)' if unique_vals < 20 else f'{col}: {unique_vals} (continuous)')

