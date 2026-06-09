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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt 


z = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")


z


z.shape


z.size


z.dtypes


z.ndim


z.describe(include = "all")


z.isnull().sum()


for i in z :
    if (z[i].dtype == "object"):
        z = pd.get_dummies(z, columns = [i], drop_first = True)
    
    


z.columns


z["loan_paid_back"]


for i in z:
    if(z[i].dtype == "bool"):
        z[i] = z[i].astype(float)
z


z.dtypes


X = z.copy()
X.drop(["id", "loan_paid_back"], axis = 1, inplace = True)
Y = z["loan_paid_back"]


x_train = X
y_train = Y


y_train = np.array(y_train).reshape(-1, 1)


n = RandomForestClassifier()
n.fit(x_train, y_train)


y_predict_train = n.predict_proba(x_train)[:, 1]
roc_curve(y_true = y_train, y_score = y_predict_train)


z5 = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


z5


for i in z5 :
    if (z5[i].dtype == "object"):
        z5 = pd.get_dummies(z5, columns = [i], drop_first = True)
    


for i in z5:
    if(z5[i].dtype == "bool"):
        z5[i] = z5[i].astype(float)
z5


X1 = z5.copy()
X1.drop(["id"], axis = 1, inplace = True)


x_test = X1


y_predict_test = n.predict_proba(x_test)[0:, 1]


z5["loan_paid_back"] = y_predict_test


result = z5[["id", "loan_paid_back"]]



result


result.to_csv("/kaggle/working/submit_file.csv", index = False)




