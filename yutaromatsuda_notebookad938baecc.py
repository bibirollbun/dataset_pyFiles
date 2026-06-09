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

# 1. CSVを読み込み
df = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

# 2. 確認（最初の5行）
print(df.head())

# 3. 別名で保存（workingディレクトリに保存される）
df.to_csv("sales_train_copy.csv", index=False)

print("sales_train_copy.csv を保存しました！")

