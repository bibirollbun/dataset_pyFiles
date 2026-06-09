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


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(train["Calories"], kde=True)
plt.title("Distribution of Calories")



sns.boxplot(x="Sex", y="Calories", data=train)



# Feature Correlation with Target
train_corr = train.copy()
train_corr["Calories"] = np.log1p(train_corr["Calories"])
corr = train_corr.corr(numeric_only=True)
sns.heatmap(corr[["Calories"]].sort_values(by="Calories", ascending=False), annot=True)



#Duration vs Calories
sns.scatterplot(x="Duration", y="Calories", hue="Sex", data=train)



#Body Temperature or Heart Rate
sns.lmplot(x="Heart_Rate", y="Calories", data=train, hue="Sex")


