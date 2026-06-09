import pandas as pd

train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


print("Train DataFrame Info:")
train_df.info()
print("\nTest DataFrame Info:")
test_df.info()

print("\nTrain DataFrame Columns:")
display(train_df.columns)
print("\nTest DataFrame Columns:")
display(test_df.columns)

print("\nTrain DataFrame Head:")
display(train_df.head())
print("\nTest DataFrame Head:")
display(test_df.head())


print("Descriptive statistics for numerical columns in train_df:")
display(train_df.describe())

print("\nDescriptive statistics for categorical columns in train_df:")
display(train_df.describe(include='object'))

print("\nDescriptive statistics for numerical columns in test_df:")
display(test_df.describe())

print("\nDescriptive statistics for categorical columns in test_df:")
display(test_df.describe(include='object'))


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate missing values
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

# Calculate percentage of missing values
train_missing_percent = (train_missing / len(train_df)) * 100
test_missing_percent = (test_missing / len(test_df)) * 100

print("Missing values in train_df:")
display(train_missing)
print("\nPercentage of missing values in train_df:")
display(train_missing_percent)

print("\nMissing values in test_df:")
display(test_missing)
print("\nPercentage of missing values in test_df:")
display(test_missing_percent)

# Visualize missing values with heatmaps
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap for train_df')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(test_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap for test_df')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical columns
numerical_cols_train = train_df.select_dtypes(include=['float64', 'int64']).columns
numerical_cols_test = test_df.select_dtypes(include=['float64', 'int64']).columns

# Plot for numerical columns in train_df
print("Plotting for train_df numerical columns:")
for col in numerical_cols_train:
    if col != 'id': # Exclude 'id' column as it's an identifier
        plt.figure(figsize=(18, 5))

        # Histogram
        plt.subplot(1, 3, 1)
        sns.histplot(data=train_df, x=col, kde=True)
        plt.title(f'Histogram of {col} (train_df)')

        # Box Plot
        plt.subplot(1, 3, 2)
        sns.boxplot(data=train_df, x=col)
        plt.title(f'Box Plot of {col} (train_df)')

        # Density Plot
        plt.subplot(1, 3, 3)
        sns.kdeplot(data=train_df, x=col)
        plt.title(f'Density Plot of {col} (train_df)')

        plt.tight_layout()
        plt.show()

# Plot for numerical columns in test_df
print("\nPlotting for test_df numerical columns:")
for col in numerical_cols_test:
    if col != 'id': # Exclude 'id' column as it's an identifier
        plt.figure(figsize=(18, 5))

        # Histogram
        plt.subplot(1, 3, 1)
        sns.histplot(data=test_df, x=col, kde=True)
        plt.title(f'Histogram of {col} (test_df)')

        # Box Plot
        plt.subplot(1, 3, 2)
        sns.boxplot(data=test_df, x=col)
        plt.title(f'Box Plot of {col} (test_df)')

        # Density Plot
        plt.subplot(1, 3, 3)
        sns.kdeplot(data=test_df, x=col)
        plt.title(f'Density Plot of {col} (test_df)')

        plt.tight_layout()
        plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Identify categorical columns
categorical_cols_train = train_df.select_dtypes(include='object').columns
categorical_cols_test = test_df.select_dtypes(include='object').columns

# Plot for categorical columns in train_df
print("Plotting for train_df categorical columns:")
for col in categorical_cols_train:
    print(f"\nFrequency table for {col} (train_df):")
    display(train_df[col].value_counts(dropna=False)) # Include NaN in counts

    plt.figure(figsize=(8, 5))
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index, palette='viridis')
    plt.title(f'Frequency Distribution of {col} (train_df)')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Plot for categorical columns in test_df
print("\nPlotting for test_df categorical columns:")
for col in categorical_cols_test:
    print(f"\nFrequency table for {col} (test_df):")
    display(test_df[col].value_counts(dropna=False)) # Include NaN in counts

    plt.figure(figsize=(8, 5))
    sns.countplot(data=test_df, x=col, order=test_df[col].value_counts().index, palette='viridis')
    plt.title(f'Frequency Distribution of {col} (test_df)')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical and categorical columns in train_df
numerical_cols_train = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_train = train_df.select_dtypes(include='object').columns.tolist()

# Remove 'id' from numerical columns and 'Personality' from categorical columns for plotting against target
if 'id' in numerical_cols_train:
    numerical_cols_train.remove('id')
if 'Personality' in categorical_cols_train:
    categorical_cols_train.remove('Personality')

# 2. Create scatter plots for numerical features vs. 'Personality'
print("Creating scatter plots for numerical features vs. Personality:")
for col in numerical_cols_train:
    plt.figure(figsize=(8, 5))
    # Use stripplot for categorical target and numerical feature to show distribution
    sns.stripplot(data=train_df, x='Personality', y=col, jitter=True, alpha=0.5)
    plt.title(f'Relationship between {col} and Personality')
    plt.xlabel('Personality')
    plt.ylabel(col)
    plt.show()

# 3. Generate box plots for categorical features vs. 'Personality'
print("\nGenerating box plots for categorical features vs. Personality:")
for col in categorical_cols_train:
    plt.figure(figsize=(8, 5))
    # Use boxplot for numerical target (implicitly handled by seaborn) and categorical feature
    # Note: We are plotting categorical features against a categorical target. Box plot is not ideal for this.
    # A better approach is to analyze frequency distribution of the categorical feature for each target category.
    # Let's use countplot for visualizing the distribution of categorical features within each Personality type.
    sns.countplot(data=train_df, x=col, hue='Personality', palette='viridis')
    plt.title(f'Distribution of {col} by Personality')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# 4. Calculate and visualize the correlation matrix for numerical features
print("\nCalculating and visualizing the correlation matrix for numerical features:")
numerical_train_df = train_df.select_dtypes(include=[np.number]).drop(columns=['id'])
correlation_matrix = numerical_train_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features in train_df')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Identify the numerical columns in the test_df DataFrame, excluding the 'id' column.
numerical_cols_test = test_df.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numerical_cols_test:
    numerical_cols_test.remove('id')

# 2. Identify the categorical columns in the test_df DataFrame.
categorical_cols_test = test_df.select_dtypes(include='object').columns.tolist()

print("Numerical columns in test_df (excluding 'id'):", numerical_cols_test)
print("Categorical columns in test_df:", categorical_cols_test)

# 3. Generate scatter plots to explore pairwise relationships between numerical features in test_df.
print("\nGenerating scatter plots for pairwise numerical features in test_df:")
sns.pairplot(test_df[numerical_cols_test].dropna()) # Drop NaN for pairplot
plt.suptitle('Pairwise Scatter Plots of Numerical Features in test_df', y=1.02)
plt.show()

# 4. Calculate the correlation matrix for the numerical features in test_df.
numerical_test_df = test_df[numerical_cols_test]
correlation_matrix_test = numerical_test_df.corr()

print("\nCorrelation Matrix of Numerical Features in test_df:")
display(correlation_matrix_test)

# 5. Create a heatmap to visualize the correlation matrix of numerical features in test_df, including annotations.
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_test, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features in test_df')
plt.show()

# 6. Analyze the count distributions or cross-tabulations of categorical features to understand their relationships if necessary.
# For categorical features, we can look at the frequency distributions already generated in the univariate analysis.
# Cross-tabulation can be done if we want to see relationships between two categorical features.
print("\nAnalyzing relationships between categorical features (using cross-tabulation):")
if len(categorical_cols_test) >= 2:
    for i in range(len(categorical_cols_test)):
        for j in range(i + 1, len(categorical_cols_test)):
            col1 = categorical_cols_test[i]
            col2 = categorical_cols_test[j]
            print(f"\nCross-tabulation of {col1} and {col2}:")
            display(pd.crosstab(test_df[col1], test_df[col2], dropna=False))
elif len(categorical_cols_test) == 1:
    print("Only one categorical column found. Cross-tabulation requires at least two.")
else:
    print("No categorical columns found in test_df.")


import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical columns present in both train and test datasets
numerical_cols = numerical_train_df.columns.intersection(numerical_test_df.columns)

print("Plotting overlaid histograms for numerical features:")

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=train_df, x=col, color='skyblue', label='Train', kde=True, stat="density", common_norm=False)
    sns.histplot(data=test_df, x=col, color='salmon', label='Test', kde=True, stat="density", common_norm=False)
    plt.title(f'Distribution of {col} (Train vs Test)')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Identify categorical columns present in both train and test datasets
categorical_cols_train = train_df.select_dtypes(include='object').columns
categorical_cols_test = test_df.select_dtypes(include='object').columns
categorical_cols = categorical_cols_train.intersection(categorical_cols_test)

print("\nPlotting count plots for categorical features:")

for col in categorical_cols:
    train_counts = train_df[col].value_counts(dropna=False)
    test_counts = test_df[col].value_counts(dropna=False)

    categories = train_counts.index.union(test_counts.index)
    train_counts = train_counts.reindex(categories, fill_value=0)
    test_counts = test_counts.reindex(categories, fill_value=0)

    x = range(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], train_counts.values, width, label='Train', color='skyblue')
    plt.bar([i + width/2 for i in x], test_counts.values, width, label='Test', color='salmon')

    plt.ylabel('Count')
    plt.title(f'Distribution of {col} (Train vs Test)')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

