!cp /kaggle/input/stanford-rna-3d-folding/sample_submission.csv /kaggle/working/data.csv


!cp /kaggle/input/manual-submission/submission.csv /kaggle/working/test.csv


import pandas as pd

# Load CSV into a DataFrame
df = pd.read_csv('/kaggle/working/data.csv')

# Show the first 5 rows
print(df.head())
print(len(df))



# Load CSV into a DataFrame
df2 = pd.read_csv('/kaggle/working/test.csv')

# Show the first 5 rows
print(df2.head())
print(len(df2))


num = 2515
# Get the first 500 rows of df
first_num_ids = df.iloc[:num]['ID']

# Find matching rows in df2
df2_matches = df2[df2['ID'].isin(first_num_ids)]

# Replace rows in df where ID matches, only within the first 500 rows
for i in df.index[:num]:
    row_id = df.at[i, 'ID']
    match = df2[df2['ID'] == row_id]
    if not match.empty:
        df.loc[i] = match.iloc[0]
    else:
        print(row_id)



print(df.head())


df.to_csv('/kaggle/working/submission.csv', index=False)


!rm /kaggle/working/data.csv


!rm /kaggle/working/test.csv

