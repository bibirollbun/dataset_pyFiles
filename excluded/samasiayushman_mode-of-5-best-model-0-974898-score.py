import pandas as pd

# Load the datasets
csv1 = pd.read_csv('/kaggle/input/bestcsv/submission (8).csv')
csv2 = pd.read_csv('/kaggle/input/private/submission (10).csv')
csv3 = pd.read_csv('/kaggle/input/private/submission (11).csv')
csv4 = pd.read_csv('/kaggle/input/private/submission_cat.csv')
csv5 = pd.read_csv('/kaggle/input/private/submission_lgb.csv')

# Extract the first column (IDs) and headers
id_column = csv1.iloc[:, 0]  # The ID column
headers = csv1.columns       # The headers

# Extract the categorical parts (excluding the first row and first column)
categorical1 = csv1.iloc[:, 1:]  # Exclude the ID column
categorical2 = csv2.iloc[:, 1:]
categorical3 = csv3.iloc[:, 1:]
categorical4 = csv4.iloc[:, 1:]
categorical5 = csv5.iloc[:, 1:]

# Combine the datasets
combined = pd.concat([categorical1, categorical2, categorical3, categorical4, categorical5], axis=1)

# Compute the mode (most frequent category) across datasets
result = combined.mode(axis=1)[0]  # [0] ensures the first mode is taken in case of ties

# Reconstruct the original DataFrame
result_df = pd.DataFrame(result.values, columns=headers[1:], index=id_column.index)
result_df.insert(0, headers[0], id_column)  # Add the ID column back

# Save the result to a new CSV file
result_df.to_csv('submission.csv', index=False)

