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
import numpy as np

# File paths
files_info = {
    "subothers3": "/kaggle/input/road-accident-sub-files/subothers3.csv",
    "submission24": "/kaggle/input/road-accident-sub-files/submission24.csv",
    "subothers2": "/kaggle/input/road-accident-sub-files/subothers2.csv",
    "submission32": "/kaggle/input/road-accident-sub-files/submission32.csv"
}

# --- Define Weights and Subtraction Factor ---
mean_files = ["submission32", "subothers2", "subothers3"]
mean_weights = [1.50, 1.25, 1.25] 

# --- Ensemble Calculation (Weighted Mean Only) ---

weighted_sum = None
mean_weight_sum = sum(mean_weights)
ID_COLUMN_NAME = "ID" 

for file_name, weight in zip(mean_files, mean_weights):
    file_path = files_info[file_name]
    df = pd.read_csv(file_path)
    
    # Check/Set ID column name on the first file read
    if weighted_sum is None:
        if ID_COLUMN_NAME not in df.columns and 'id' in df.columns:
            ID_COLUMN_NAME = "id"
        elif ID_COLUMN_NAME not in df.columns:
             print(f"ðŸš¨ WARNING: Could not find '{ID_COLUMN_NAME}' or 'id' column in {file_name}.")
             
        # 1. Initialize the base Series (weighted sum) and the final DataFrame
        weighted_sum = df["accident_risk"] * weight
        final_df = df[[ID_COLUMN_NAME]].copy() 
    else:
        # Add to the weighted sum
        weighted_sum += df["accident_risk"] * weight

# 2. Normalize the sum to get the weighted mean
if mean_weight_sum > 0:
    final_df["accident_risk"] = weighted_sum / mean_weight_sum
else:
    raise ValueError("The weights for the weighted mean must sum to a value greater than zero.")


# 3. Round the final result ðŸŽ¯
# Round the 'accident_risk' values to FOUR decimal points
final_df["accident_risk"] = final_df["accident_risk"].round(4) 

# --- Final Steps ---
# Save final submission
final_df.to_csv("submission.csv", index=False)

print("Final submission.csv created! 'accident_risk' values are rounded to 4 decimal points.")
print("\n--- Result Head (Weighted Mean & Rounded) ---")
print(final_df.head())

