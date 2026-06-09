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
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")

print("------- Train Data -------")
print(train.info())
print(train.shape)
print(train.head())

print("\n------- Test Data -------")
print(test.info())
print(test.shape)
print(test.head())


df_numeric = train.drop(columns=["Soil Type", "Crop Type", "Fertilizer Name"] , axis=1)
corr = df_numeric.corr()

sns.heatmap(corr, cmap="coolwarm", annot=True)



fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(16, 24))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for i, col in enumerate(df_numeric.columns):
    sns.barplot(x="Fertilizer Name", y=col, data=train, ax=axes[i, 0], hue="Soil Type")
    axes[i, 0].set_title(f"{col} vs Fertilizer Name colored by Soil Type")
    sns.barplot(x="Fertilizer Name", y=col, data=train, ax=axes[i, 1], hue="Crop Type")
    axes[i, 1].set_title(f"{col} vs Fertilizer Name colored by Crop Type")
    sns.move_legend(axes[i, 0], "upper left", bbox_to_anchor=(1, 1))
    sns.move_legend(axes[i, 1], "upper left", bbox_to_anchor=(1, 1))


plt.show()


fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 24))
plt.subplots_adjust(hspace=0.4)

for i, col in enumerate(df_numeric.columns):
    sns.boxplot(x="Fertilizer Name", y=col, data=train, ax=axes[i])
    axes[i].set_title(f"{col} vs Fertilizer Name")

plt.show()

