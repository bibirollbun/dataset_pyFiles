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
import glob

path = "/kaggle/input/submissions-for-blending"

files = glob.glob(os.path.join(path, "*.csv"))
files


dfs = []
for i, f in enumerate(files):
    df = pd.read_csv(f)

    pred_col = df.columns[-1]
    df = df.rename(columns={pred_col: f"pred_{i}"})

    dfs.append(df)

dfs


blend_df = dfs[0][["user_hash", "pred_0"]].copy()

for i in range(1, len(dfs)):
    blend_df = blend_df.merge(dfs[i][["user_hash", f"pred_{i}"]])


blend_df.head()


pred_col = [c for c in blend_df.columns if c.startswith("pred_")]
p = blend_df[pred_col].values


import scipy.stats as ss
# rank average
ranked = np.zeros_like(p)
 
for i, col in enumerate(pred_col):
    ranked[:, i] = ss.rankdata(blend_df[col]) / len(blend_df)

blend_df["prediction"] = ranked.mean(axis=1)
blend_df[["user_hash", "prediction"]].to_csv("blend_rank_avg.csv", index=False)


blend_df["min_avg"] = p.min(axis=1)
blend_df["max_avg"] = p.max(axis=1)
blend_df["mid_avg"] = (blend_df["min_avg"] + blend_df["max_avg"]) / 2

blend_df[["user_hash", "prediction"]].to_csv("blend_minmax_mid.csv", index=False)


# WEIGHTED BLEND USING LINEAR REGRESSION (best)

from sklearn.linear_model import LinearRegression
target = p.mean(axis=1)

reg = LinearRegression(positive=True)
reg.fit(p, target)
weights_opt = reg.coef_ / reg.coef_.sum()

print("\nOptimized weights (Linear Regression):")
for c, w in zip(pred_col, weights_opt):
    print(f"{c}: {w:.4f}")

blend_df["prediction"] = p.dot(weights_opt)
blend_df[["user_hash", "prediction"]].to_csv("submission.csv", index=False)




