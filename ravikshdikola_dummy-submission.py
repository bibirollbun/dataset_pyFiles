import pandas as pd
sub = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")
sub


import numpy as np
for i in range(1, 6):
    sub[f"x_{i}"] = np.random.uniform(0, 10)
    sub[f"y_{i}"] = np.random.uniform(0, 10)
    sub[f"z_{i}"] = np.random.uniform(0, 10)


sub


sub.to_csv("submission.csv", index=False)

