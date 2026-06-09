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

train = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



train.info()
train.describe()
train.isnull().sum()



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
sns.histplot(train['Premium Amount'], bins=50, kde=True)
plt.title("Distribution of Insurance Premium")
plt.xlabel("PremiumAmount")
plt.show()




num_cols = [
    'Age', 'Annual Income', 'Number of Dependents',
    'Health Score', 'Previous Claims', 'Vehicle Age',
    'Credit Score', 'Insurance Duration'
]
cat_cols = [
    'Gender', 'Marital Status', 'Education Level', 'Occupation',
    'Location', 'Policy Type', 'Customer Feedback', 'Smoking Status',
    'Exercise Frequency', 'Property Type'
]



from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="mean")
train[num_cols] = num_imputer.fit_transform(train[num_cols])



cat_imputer = SimpleImputer(strategy="most_frequent")
train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])



from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
train[cat_cols] = encoder.fit_transform(train[cat_cols])



train.info()
train.describe()
train.isnull().sum()





