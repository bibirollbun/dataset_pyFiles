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


import pandas as pd

# File paths
file1_path = '/kaggle/input/drw-submission-test/submission_ensemble.csv'
file2_path = '/kaggle/input/offline-drw-nn/submission_c9f4be847068_0.447745.csv'
file3_path = '/kaggle/input/greedy21/submission_54ab3041cda338de.csv'
file4_path = '/kaggle/input/105-feature-node/submission_node.csv'
file5_path = '/kaggle/input/105mlp/submission (1).csv'
file6_path = '/kaggle/input/mldl-nn-arch-search/submission_ensemble(1).csv'

# Weights (must sum to 1.0)
weight1 = 0.89  # Weight for submission_ensemble.csv
weight2 = 0.01  # Weight for submission_c9f4be847068_0.447745.csv
weight3 = 0.07  # Weight for submission_54ab3041cda338de.csv
weight4 = 0.01  # Weight for submission_node.csv
weight5 = 0.01  # Weight for submission (1).csv
weight6 = 0.01  # Weight for submission_ensemble(1).csv

# Verify weights sum to 1
print(f"Sum of weights: {weight1 + weight2 + weight3 + weight4 + weight5 + weight6}")

# Load the CSV files
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)
df3 = pd.read_csv(file3_path)
df4 = pd.read_csv(file4_path)
df5 = pd.read_csv(file5_path)
df6 = pd.read_csv(file6_path)

# Check if all dataframes have the same structure
print(f"Shape of file 1: {df1.shape}")
print(f"Shape of file 2: {df2.shape}")
print(f"Shape of file 3: {df3.shape}")
print(f"Shape of file 4: {df4.shape}")
print(f"Shape of file 5: {df5.shape}")
print(f"Shape of file 6: {df6.shape}")
print(f"Columns in file 1: {df1.columns.tolist()}")
print(f"Columns in file 2: {df2.columns.tolist()}")
print(f"Columns in file 3: {df3.columns.tolist()}")
print(f"Columns in file 4: {df4.columns.tolist()}")
print(f"Columns in file 5: {df5.columns.tolist()}")
print(f"Columns in file 6: {df6.columns.tolist()}")

# Assuming all files have the same structure with an ID column and prediction columns
# Identify the ID column (usually the first column)
id_col = df1.columns[0]

# Identify numeric columns to average (all columns except the ID column)
numeric_cols = [col for col in df1.columns if col != id_col]

# Create a copy of the first dataframe to store results
result_df = df1.copy()

# Apply weighted average to numeric columns
for col in numeric_cols:
    result_df[col] = (df1[col] * weight1 + 
                      df2[col] * weight2 + 
                      df3[col] * weight3 + 
                      df4[col] * weight4 + 
                      df5[col] * weight5 + 
                      df6[col] * weight6)

# Save the result
output_path = 'weighted_average_submission.csv'
result_df.to_csv(output_path, index=False)
print(f"\nWeighted average saved to: {output_path}")

# Display first few rows of the result
print("\nFirst 5 rows of the weighted average:")
print(result_df.head())

