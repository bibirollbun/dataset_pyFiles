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


train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
train.head()
test.head()


train.isnull().sum() #check any NaN value
train.duplicated().sum() #check any duplicated data



import matplotlib.pyplot as plt
import seaborn as sns



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))


ax1.hist(train["Age"], bins=20, edgecolor="black", color="skyblue")
ax1.set_title("Age Distribution", fontsize=14)
ax1.set_xlabel("Age")
ax1.set_ylabel("Frequency")


sex_counts = train["Sex"].value_counts()
ax2.bar(sex_counts.index, sex_counts.values, color=["lightcoral", "lightblue"])
ax2.set_title("Sex Distribution", fontsize=14)
ax2.set_xlabel("Sex")
ax2.set_ylabel("Count")


smoke_counts = train["SmokingStatus"].value_counts()
ax3.bar(smoke_counts.index, smoke_counts.values, color=["orange", "green", "blue"])
ax3.set_title("Smoking Status", fontsize=14)
ax3.set_xlabel("Smoking Status")
ax3.set_ylabel("Count")


plt.tight_layout()
plt.show()



outliers = train[(train['FVC'] < 1000) | (train['FVC'] > 5000)]
outliers



fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# FVC distribution
ax[0].hist(train["FVC"], bins=50, edgecolor="black")
ax[0].set_title("FVC Distribution")
ax[0].set_xlabel("FVC")
ax[0].set_ylabel("Frequency")


ax[1].scatter(train["FVC"], train["Percent"], alpha=0.6)
ax[1].set_title("FVC vs Percent")
ax[1].set_xlabel("FVC")
ax[1].set_ylabel("Percent")

fig.tight_layout()
plt.show()



#allocate patients into each age group
bins = [50,60,65,70,80,90]
labels = ["50-59", "60-65", "65-70", "71-80", "81-90"]
train['AgeGroup'] = pd.cut(train["Age"], bins=bins, labels=labels, right=False, include_lowest=True)

#find the mean of fvc in each age group
avg = train.groupby("AgeGroup",observed=True)["FVC"].mean()

#plot
avg.plot(figsize=(16,10));
plt.xlabel("Age Group")
plt.ylabel("The mean of FVC")
plt.title("Average FVC by Age Group")




g50_59  = train[train['AgeGroup'] == "50-59"]
g60_65  = train[train['AgeGroup'] == "60-65"]
g65_70  = train[train['AgeGroup'] == "65-70"]
g71_80  = train[train['AgeGroup'] == "71-80"]
g81_90  = train[train['AgeGroup'] == "81-90"]
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,6))

ax = ax.flatten()


ax[0].boxplot(g50_59['FVC'])
ax[0].set_title("50-59", fontsize=8)

ax[1].boxplot(g60_65['FVC'])
ax[1].set_title("60-65", fontsize=8)

ax[2].boxplot(g65_70['FVC'])
ax[2].set_title("65-70", fontsize=8)

ax[3].boxplot(g71_80['FVC'])
ax[3].set_title("71-80", fontsize=8)

ax[4].boxplot(g81_90['FVC'])
ax[4].set_title("81-90", fontsize=8)


fig.delaxes(ax[5])








