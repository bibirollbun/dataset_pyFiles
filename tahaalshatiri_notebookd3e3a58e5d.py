# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# https://www.kaggle.com/code/tahaalshatiri/notebook7922746d36?scriptVersionId=238846740
# all subs here are from different version of this notebook with is pastprocessing of colab notebooks that i will share


import pandas as pd
from collections import Counter

# Load CSV files
df1 = pd.read_csv("/kaggle/input/qwen-final-ensemble/submission  version 6.csv")
df2 = pd.read_csv("/kaggle/input/qwen-final-ensemble/submission version 4.csv")
df3 = pd.read_csv("/kaggle/input/qwen-2-5-14b/submission (8).csv")

# Merge on 'id'
merged = df1.merge(df2, on="id", suffixes=('_1', '_2'))
merged = merged.merge(df3, on="id")
merged.rename(columns={"label": "label_3"}, inplace=True)

# Apply hard voting
def vote(row):
    votes = [row['label_1'], row['label_2'], row['label_3']]
    return Counter(votes).most_common(1)[0][0]

merged['label'] = merged.apply(vote, axis=1)

# Save final predictions
final_df = merged[['id', 'label']]
final_df.to_csv("ensemble_voted.csv", index=False)



final_df

