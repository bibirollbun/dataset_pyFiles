import os
import pandas as pd
import glob

# check for all submission.csv files from notebook input
submission_files = glob.glob('/kaggle/input/**/submission.csv', recursive=True)

#parsing all prediction files into one dataframe
main_df = pd.read_csv(submission_files[0])
for file in submission_files[0:]:
    df = pd.read_csv(file)
    second_col_name = df.columns[1]  # Get the second column name
    # Use the filename to make the column name unique
    new_col_name = os.path.basename(os.path.dirname(file)) + "_" + second_col_name
    main_df[new_col_name] = df[second_col_name]

main_df=main_df.set_index('id',drop=True)
main_df=main_df.iloc[:,1:]

import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns

#get all possible pairs of column indices
col_indices = range(len(main_df.columns))
pairs = list(combinations(col_indices, 2))

#correlation matrix
corr_matrix = main_df.corr()

# correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0.8, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

#plots for each pair
for x_idx, y_idx in pairs:
    plt.figure()
    plt.scatter(main_df.iloc[:, x_idx], main_df.iloc[:, y_idx])
    plt.xlabel(main_df.columns[x_idx])
    plt.ylabel(main_df.columns[y_idx])

plt.show()

