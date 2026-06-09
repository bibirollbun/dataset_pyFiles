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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_data


test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_data


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.barplot(x="Color", y="Price", data=train_data, estimator=lambda x: sum(x)/len(x))
plt.title("Average Price by Color")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,5))
train_data["Price"].hist(bins=60)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=train_data, x='Brand', order=train_data['Brand'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Number of Backpacks by Brand")
plt.show()

