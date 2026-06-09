import pandas as pd
import numpy as np

# Paths
path_A = "/kaggle/input/airr-ml-25-eda-convert-dataset-to-parquet/submission.csv"
path_B = "/kaggle/input/airr-ml-25-naive-baseline-with-xgboost-pca/submission.csv"

# Load submissions
sub_A = pd.read_csv(path_A)
sub_B = pd.read_csv(path_B)

# Ensure the same number of rows
assert len(sub_A) == len(sub_B), "Submissions must have the same number of rows."

# Set seed for reproducibility
np.random.seed(42)

# Define the change range
min_change = 0.01
max_change = 0.15

# Generate random changes for each row in the 'label_positive_probability'
change_pct = np.random.uniform(min_change, max_change, len(sub_B))  # Array of random percentages
sign_change = np.random.choice([-1, 1], size=len(sub_B))  # Randomly pick +1 or -1 for each row

# Calculate the change to apply to each row
change = sign_change * sub_A["label_positive_probability"].values * change_pct

# Apply the change to sub_B
sub_B["label_positive_probability"] += change

# Clip values to be between 0 and 1
sub_B["label_positive_probability"] = np.clip(sub_B["label_positive_probability"], 0, 1)

# Save the final blended submission
sub_B.to_csv("submission.csv", index=False)

print("Blended submission saved. Each row's probability changed and clipped to range [0, 1].")


