# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

# List available files in the competition directory
competition_path = "/kaggle/input/"
print("Available datasets:")
print(os.listdir(competition_path))

# Check the folder name for this competition
competition_folders = os.listdir(competition_path)
for folder in competition_folders:
    print(f"\nFiles in {folder}:")
    print(os.listdir(os.path.join(competition_path, folder)))



import pandas as pd

# Define the path
data_path = "/kaggle/input/stanford-rna-3d-folding/"

# Load train data
train_sequences = pd.read_csv(data_path + "train_sequences.csv")
train_labels = pd.read_csv(data_path + "train_labels.csv")

# Load test data
test_sequences = pd.read_csv(data_path + "test_sequences.csv")

# Display first few rows
print("Train Sequences:")
display(train_sequences.head())

print("\nTrain Labels:")
display(train_labels.head())

print("\nTest Sequences:")
display(test_sequences.head())

# Check for missing values
print("\nMissing Values in Train Sequences:")
print(train_sequences.isnull().sum())

print("\nMissing Values in Train Labels:")
print(train_labels.isnull().sum())



# Drop rows with missing 3D coordinate values
train_labels_clean = train_labels.dropna()

# Drop rows with missing sequences
train_sequences_clean = train_sequences.dropna()

print(f"Remaining Rows in Train Labels: {len(train_labels_clean)}")
print(f"Remaining Rows in Train Sequences: {len(train_sequences_clean)}")



# Extract target_id from ID column
train_labels['target_id'] = train_labels['ID'].apply(lambda x: "_".join(x.split("_")[:2]))

# Merge datasets
train_data = pd.merge(train_sequences, train_labels, on="target_id")

# Display merged dataset
print(train_data.head())



import matplotlib.pyplot as plt
import seaborn as sns

# Plot sequence lengths
train_sequences['seq_length'] = train_sequences['sequence'].apply(len)
sns.histplot(train_sequences['seq_length'], bins=30, kde=True)
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.title("Distribution of RNA Sequence Lengths")
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load train sequences
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')

# Compute sequence lengths
train_sequences["seq_length"] = train_sequences["sequence"].apply(len)

# Summary statistics
print(train_sequences["seq_length"].describe())

# Boxplot for outliers
plt.figure(figsize=(10, 5))
sns.boxplot(x=train_sequences["seq_length"])
plt.title("Boxplot of RNA Sequence Lengths")
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")

# Add a column for sequence length
train_sequences["seq_length"] = train_sequences["sequence"].apply(len)

# Cap sequences at 1000
train_sequences["seq_length_capped"] = train_sequences["seq_length"].apply(lambda x: min(x, 1000))

# Log-transform the sequence lengths (adding 1 to avoid log(0))
train_sequences["log_seq_length"] = np.log1p(train_sequences["seq_length_capped"])

# Plot the transformed distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_sequences["log_seq_length"], kde=True, bins=50)
plt.xlabel("Log Sequence Length")
plt.ylabel("Count")
plt.title("Log-Transformed Distribution of RNA Sequence Lengths")
plt.show()




# Assuming df['Sequence_Length'] contains RNA sequence lengths
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # 25th percentile
    Q3 = df[column].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Example Usage
df = pd.DataFrame({'Sequence_Length': [3, 22, 39, 86, 100, 4298, 50, 70, 900, 2500]})
outliers, lb, ub = detect_outliers_iqr(df, 'Sequence_Length')

print("Lower Bound:", lb)
print("Upper Bound:", ub)
print("Outliers:\n", outliers)


df_cleaned = df[(df['Sequence_Length'] >= -945.625) & (df['Sequence_Length'] <=1687.375)]

# Plot the new distribution
sns.histplot(df_cleaned['Sequence_Length'], bins=50, kde=True)
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.title("Distribution After Removing Outliers")
plt.show()



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Apply log transformation (adding 1 to avoid log(0))
df["Log_Sequence_Length"] = np.log1p(df["Sequence_Length"])

# Plot the distribution
plt.figure(figsize=(7,5))
sns.histplot(df["Log_Sequence_Length"], kde=True, bins=50)
plt.xlabel("Log Sequence Length")
plt.ylabel("Count")
plt.title("Log-Transformed Distribution After Removing Outliers")
plt.show()





