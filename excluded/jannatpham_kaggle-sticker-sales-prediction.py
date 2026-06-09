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


testdata = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
traindata = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")


traindata.info()


traindata.isnull().sum()


country = traindata["country"].value_counts()
print(country)


product = traindata['product'].value_counts()
print(product)


rows_nulls = traindata[traindata['num_sold'].isnull()]
print(rows_nulls['num_sold'])


import matplotlib.pyplot as plt
import seaborn as sns


plt.hist(traindata['num_sold'])


plt.figure(figsize=(8, 6))
plt.scatter(traindata['country'], traindata['num_sold'], color='blue', alpha=1)
plt.title('Scatter Plot of Country vs. Num_sold')
plt.xlabel('Country')
plt.ylabel('Number of sold')
plt.grid()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(traindata['product'], traindata['num_sold'], color='green', alpha=0.7)
plt.title('Scatter Plot of Product vs. Num_sold')
plt.xlabel('Product')
plt.ylabel('Number of sold')
plt.grid()
plt.show()


traindata.describe()
traindata = traindata.dropna()


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



x = traindata['id']
y = traindata['num_sold']
X_test = testdata['id']

x = np.array(x).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)


model = LinearRegression()
model.fit(x, y)



y_predict = model.predict(X_test)




