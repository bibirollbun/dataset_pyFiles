# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


dataset = [
    (df_train, "train"),
    (df_test, "test")
]

for df, name in dataset:
    print(f"There is {df.shape[0]} rows and {df.shape[1]} columns in the {name} dataset.")
    sum_data_duplicates = df.duplicated().sum()
    print(f"Duplicated fields in {name} dataset: {sum_data_duplicates}")


df_train.info()


df_test.info()


df_train.head()


df_test.head()


for df, name in dataset:
    print(f"\n--- {name.upper()} Dataset ---")
    total_rows = df.shape[0]
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    
    if not null_columns.empty:
        print("Columns with null values (count | %):")
        # Sort columns by null count (or percentage) in descending order
        sorted_null_columns = null_columns.sort_values(ascending=True)
        for col, count in sorted_null_columns.items():
            percent = (count / total_rows) * 100
            print(f"- {col}: {count} nulls ({percent:.2f}%)")
    else:
        print("No columns with null values.")


summary_dict = {col: df[col].describe() for col in null_columns.index}

# To display the summaries:
for col, summary in summary_dict.items():
    print(f"Summary for column: {col}")
    print(summary)
    print("\n")


plt.hist(df_train["Premium Amount"], bins=50, edgecolor='black')  # edgecolor adds bar borders
plt.title('Distribution of Premium Amounts')  # Fixed title to match the data
plt.xlabel('Premium Amount')
plt.ylabel('Frequency')
plt.show()


stats.probplot(df_train["Premium Amount"], dist="norm", plot=plt)
plt.title('Normal Q-Q plot Sale Prices')
plt.xlabel('Theoretical quantiles')
plt.ylabel('Ordered Values')
plt.grid(True)
plt.show()


from sklearn.preprocessing import PowerTransformer, QuantileTransformer

# Initialize transformers
transformers = {
    'Box-Cox': PowerTransformer(method='box-cox', standardize=True),
    'Yeo-Johnson': PowerTransformer(method='yeo-johnson', standardize=True),
    'Quantile': QuantileTransformer(output_distribution='normal')
}

# Apply transformations and store results
transformed_data = {
    'Original': df_train["Premium Amount"]
}
for name, transformer in transformers.items():
    transformed_data[name] = transformer.fit_transform(df_train[["Premium Amount"]]).flatten()

# Plot histograms
plt.figure(figsize=(18, 6))
for i, (name, data) in enumerate(transformed_data.items(), 1):
    plt.subplot(1, 4, i)
    plt.hist(data, bins=50, edgecolor='black')
    plt.title(f"{name} Data" if name == 'Original' else f"{name} Transformed")
    plt.xlabel("Premium Amount" if name == 'Original' else "Transformed Value")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot Q-Q plots
plt.figure(figsize=(18, 6))
for i, (name, data) in enumerate(transformed_data.items(), 1):
    plt.subplot(1, 4, i)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot: {name} Data' if name == 'Original' else f'Q-Q Plot: {name} Transformed')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Ordered Values')
    plt.grid(True)
plt.tight_layout()
plt.show()


df_train["Premium Amount"].describe()


columns_numerical_values = df_train.select_dtypes("number").drop(columns=["Premium Amount", "id"])
columns_numerical_values.count()


for col in columns_numerical_values.columns:
    plt.figure(figsize=(12, 5))
    
    # Left subplot: Histogram
    plt.subplot(1, 2, 1)
    plt.hist(columns_numerical_values[col].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Histogram for {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    
    # Right subplot: Boxplot with colors
    plt.subplot(1, 2, 2)
    bp = plt.boxplot(columns_numerical_values[col].dropna(), vert=False, patch_artist=True)
    plt.title(f"Boxplot for {col}")
    plt.xlabel(col)
    
    # Customize the boxplot colors
    for box in bp['boxes']:
        box.set(facecolor='lightgreen', color='darkgreen')
    for whisker in bp['whiskers']:
        whisker.set(color='blue', linestyle='--')
    for cap in bp['caps']:
        cap.set(color='red')
    for median in bp['medians']:
        median.set(color='orange')
    
    plt.tight_layout()
    plt.show()


columns_object_values = df_train.select_dtypes("object")
columns_object_values.count()


for col in columns_object_values.columns:
    # Get value counts for each category
    counts = columns_object_values[col].value_counts()
    
    # Plot a bar chart of the value counts
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar", color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

