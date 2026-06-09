import numpy as np
import pandas as pd
import glob

# Path to all CSVs you want to average
csv_files = glob.glob("/kaggle/input/temp1984259/*.csv")

print(f"Found {len(csv_files)} files:")
for f in csv_files:
    print(" -", f)

# Read and sort all CSVs by 'id'
dfs = [pd.read_csv(f).sort_values('id').reset_index(drop=True) for f in csv_files]

# Use the column that contains predictions (auto-detected)
pred_col = [c for c in dfs[0].columns if c != 'id'][0]

# Stack all prediction columns into an array
all_preds = np.column_stack([df[pred_col].values for df in dfs])

# Take mean prediction across all models
mean_preds = np.mean(all_preds, axis=1)

# Create submission
submission = pd.DataFrame({
    'id': dfs[0]['id'],
    pred_col: mean_preds
})

# Save final averaged submission
submission.to_csv("submission.csv", index=False)

print("Averaged submission saved as 'submission.csv'")





