import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


df.sample(5)


df.columns


df.info()


import seaborn as sns
import matplotlib.pyplot as plt


# 1. Summary Statistics
numerical_column = "Time_spent_Alone"
print(f"Summary Statistics for {numerical_column}:")
print(df[numerical_column].describe())


# 2. Check for missing values
print(f"\nMissing Values in {numerical_column}:")
print(df[numerical_column].isnull().sum())


# 3. Distribution Plot (Histogram + KDE)
plt.figure(figsize=(10, 6))
sns.histplot(df[numerical_column], kde=True, color='skyblue', bins=10)
plt.title(f'Distribution of {numerical_column}')
plt.xlabel(numerical_column)
plt.ylabel('Frequency')
plt.show()


# 4. Boxplot for Outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df[numerical_column], color='lightgreen')
plt.title(f'Boxplot of {numerical_column}')
plt.xlabel(numerical_column)
plt.show()


categorical_column = "Stage_fear"


# 1. Count of each category (including NaN)
print(f"Count of each category in {categorical_column}:")
print(df[categorical_column].value_counts(dropna=False))


# 2. Percentage of missing values (NaN)
missing_percentage = df[categorical_column].isnull().mean() * 100
print(f"\nPercentage of missing values in {categorical_column}: {missing_percentage:.2f}%")


# 3. Barplot of category counts (with NaN as a separate category)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=categorical_column, order=df[categorical_column].value_counts().index, palette='Set2')
plt.title(f'Category Counts for {categorical_column}')
plt.ylabel('Frequency')
plt.xlabel(categorical_column)
plt.show()


# 4. Pie chart to show distribution of categories (with NaN as a separate category)
plt.figure(figsize=(8, 6))
df[categorical_column].value_counts(dropna=False).plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', n_colors=len(df[categorical_column].unique())))
plt.title(f'Distribution of {categorical_column}')
plt.ylabel('')
plt.show()


# 1. Summary Statistics
numerical_column = "Social_event_attendance"
print(f"Summary Statistics for {numerical_column}:")
print(df[numerical_column].describe())


# 2. Check for missing values
print(f"\nMissing Values in {numerical_column}:")
print(df[numerical_column].isnull().sum())


df[df[numerical_column].isnull()]


# 3. Distribution Plot (Histogram + KDE)
plt.figure(figsize=(10, 6))
sns.histplot(df[numerical_column], kde=True, color='skyblue', bins=10)
plt.title(f'Distribution of {numerical_column}')
plt.xlabel(numerical_column)
plt.ylabel('Frequency')
plt.show()


# 4. Boxplot for Outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df[numerical_column], color='lightgreen')
plt.title(f'Boxplot of {numerical_column}')
plt.xlabel(numerical_column)
plt.show()


numerical_column = "Going_outside"
# 1. Summary Statistics
print(f"Summary Statistics for {numerical_column}:")
print(df[numerical_column].describe())


# 2. Check for missing values
print(f"\nMissing Values in {numerical_column}:")
print(df[numerical_column].isnull().sum())



# 3. Distribution Plot (Histogram + KDE)
plt.figure(figsize=(10, 6))
sns.histplot(df[numerical_column], kde=True, color='skyblue', bins=10)
plt.title(f'Distribution of {numerical_column}')
plt.xlabel(numerical_column)
plt.ylabel('Frequency')
plt.show()


df["Going_outside"].value_counts()


# 4. Boxplot for Outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df[numerical_column], color='lightgreen')
plt.title(f'Boxplot of {numerical_column}')
plt.xlabel(numerical_column)
plt.show()


# Choose the categorical column you want to analyze (for example 'Gender')
categorical_column = 'Drained_after_socializing'

# 1. Count of each category (including NaN)
print(f"Count of each category in {categorical_column}:")
print(df[categorical_column].value_counts(dropna=False))


# 2. Percentage of missing values (NaN)
missing_percentage = df[categorical_column].isnull().mean() * 100
print(f"\nPercentage of missing values in {categorical_column}: {missing_percentage:.2f}%")


# 3. Barplot of category counts (with NaN as a separate category)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=categorical_column, order=df[categorical_column].value_counts().index, palette='Set2')
plt.title(f'Category Counts for {categorical_column}')
plt.ylabel('Frequency')
plt.xlabel(categorical_column)
plt.show()


# 4. Pie chart to show distribution of categories (with NaN as a separate category)
plt.figure(figsize=(8, 6))
df[categorical_column].value_counts(dropna=False).plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', n_colors=len(df[categorical_column].unique())))
plt.title(f'Distribution of {categorical_column}')
plt.ylabel('')
plt.show()


# Choose the categorical column you want to analyze (for example 'Gender')
categorical_column = 'Friends_circle_size'

# 1. Count of each category (including NaN)
print(f"Count of each category in {categorical_column}:")
print(df[categorical_column].value_counts(dropna=False))


# 2. Percentage of missing values (NaN)
missing_percentage = df[categorical_column].isnull().mean() * 100
print(f"\nPercentage of missing values in {categorical_column}: {missing_percentage:.2f}%")


# 3. Barplot of category counts (with NaN as a separate category)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=categorical_column, order=df[categorical_column].value_counts().index, palette='Set2')
plt.title(f'Category Counts for {categorical_column}')
plt.ylabel('Frequency')
plt.xlabel(categorical_column)
plt.show()


# 4. Pie chart to show distribution of categories (with NaN as a separate category)
plt.figure(figsize=(8, 6))
df[categorical_column].value_counts(dropna=False).plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', n_colors=len(df[categorical_column].unique())))
plt.title(f'Distribution of {categorical_column}')
plt.ylabel('')
plt.show()


df["Post_frequency"].describe()


df["Post_frequency"].value_counts()


categorical_column = 'Post_frequency'

# 1. Count of each category (including NaN)
print(f"Count of each category in {categorical_column}:")
print(df[categorical_column].value_counts(dropna=False))


# 2. Percentage of missing values (NaN)
missing_percentage = df[categorical_column].isnull().mean() * 100
print(f"\nPercentage of missing values in {categorical_column}: {missing_percentage:.2f}%")


# 3. Barplot of category counts (with NaN as a separate category)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=categorical_column, order=df[categorical_column].value_counts().index, palette='Set2')
plt.title(f'Category Counts for {categorical_column}')
plt.ylabel('Frequency')
plt.xlabel(categorical_column)
plt.show()


# 4. Pie chart to show distribution of categories (with NaN as a separate category)
plt.figure(figsize=(8, 6))
df[categorical_column].value_counts(dropna=False).plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', n_colors=len(df[categorical_column].unique())))
plt.title(f'Distribution of {categorical_column}')
plt.ylabel('')
plt.show()


# Choose the categorical column you want to analyze (for example 'Gender')
categorical_column = 'Personality'

# 1. Count of each category (including NaN)
print(f"Count of each category in {categorical_column}:")
print(df[categorical_column].value_counts(dropna=False))


# 3. Barplot of category counts (with NaN as a separate category)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=categorical_column, order=df[categorical_column].value_counts().index, palette='Set2')
plt.title(f'Category Counts for {categorical_column}')
plt.ylabel('Frequency')
plt.xlabel(categorical_column)
plt.show()


!pip install sweetviz
import sweetviz as sv

# Create a report
report = sv.analyze(df)

# Show the report
report.show_html('sweetviz_report.html')




