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
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error 
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt 
import numpy as np




z = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv")
z


z


z.isnull().sum()


for i in z:
    if(z[i].isnull().sum() > 0):
        z = z[z[i].notna()]
z


z.isnull().sum()


z.corr(numeric_only = True)


z.columns


z["amount_new_house_transactions"].value_counts()


z["amount_new_house_transactions"].dtype


sns.heatmap(z.corr(numeric_only = True), cmap = "coolwarm", annot = True)


z.corr(numeric_only = True)["amount_new_house_transactions"].sort_values(ascending = False)



b = z.copy()
for i in b:
    if(b[i].dtype == "object"):
        b.drop([i], axis = 1, inplace = True)


b


X = b.copy()
X.drop(["amount_new_house_transactions"], axis = 1, inplace = True)
Y = b["amount_new_house_transactions"]


x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 100)


y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)


n = RandomForestRegressor()
n.fit(x_train, y_train)


y_predict_train = n.predict(x_train)
r2_score(y_true = y_train, y_pred = y_predict_train)


mean_squared_error(y_true = y_train, y_pred = y_predict_train)


(mean_absolute_percentage_error(y_true = y_train, y_pred = y_predict_train))*100


y_predict_test = n.predict(x_test)
r2_score(y_true = y_test, y_pred = y_predict_test)


mean_squared_error(y_true = y_test, y_pred = y_predict_test)


(mean_absolute_percentage_error(y_true = y_test, y_pred = y_predict_test))*100

