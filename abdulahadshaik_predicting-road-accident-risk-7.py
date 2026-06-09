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
files = [
    "/kaggle/input/road-accident-sub-files/subothers.csv",
    "/kaggle/input/road-accident-sub-files/submission22.csv",
    "/kaggle/input/road-accident-sub-files/submission23.csv",
    "/kaggle/input/road-accident-sub-files/submission24.csv"
]

# Example weights (change as needed, must sum > 0)
weights = [0.7, 0.02, 0.03, 0.05]

# Load first file as base
final_df = pd.read_csv(files[0]).copy()
final_df["accident_risk"] = final_df["accident_risk"] * weights[0]

# Add weighted values from remaining files
for i in range(1, len(files)):
    df = pd.read_csv(files[i])
    final_df["accident_risk"] += df["accident_risk"] * weights[i]

# Normalize if weights don't sum to 1
weight_sum = sum(weights)
if weight_sum != 1:
    final_df["accident_risk"] /= weight_sum

# Save final submission
final_df.to_csv("submission.csv", index=False)

print("Final submission.csv created!")

print(final_df.head())

