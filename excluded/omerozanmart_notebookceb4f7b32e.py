import pandas as pd
import os

# 1. Define the path to your uploaded file
# Based on your screenshot, your dataset is named 'tpcafa5-icafa5'
input_path = "/kaggle/input/tcafa5-icafa5-new-dataset/tcafa5-icafa5.csv"

# 2. Read the file
# We assume it is a CSV since the extension is .csv
df = pd.read_csv(input_path)

# 3. Save it to the Working Directory as a TSV
# The competition specifically requires the name 'submission.tsv'
output_path = "/kaggle/working/submission.tsv"
df.to_csv(output_path, sep='\t', index=False)

print(f"Success! File saved to {output_path}")

