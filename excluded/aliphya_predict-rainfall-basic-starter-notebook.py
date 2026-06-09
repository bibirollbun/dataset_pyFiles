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


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.head(10)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler,LabelEncoder



f = plt.figure(figsize=(10,10))
f.add_subplot(221)
sns.violinplot(data=train,y="maxtemp",x="rainfall",palette = "rainbow")
f.add_subplot(222)
sns.violinplot(data=train,y="sunshine",x="rainfall",palette = "coolwarm")
f.add_subplot(223)
sns.boxplot(data=train,y="humidity",x="rainfall",palette = "dark")
f.add_subplot(224)
sns.boxplot(data=train,y="dewpoint",x="rainfall",palette = "Set1")


f = plt.figure(figsize=(10,10))
f.add_subplot(221)
sns.violinplot(data=train,y="cloud",x="rainfall",palette = "rainbow")
f.add_subplot(222)
sns.violinplot(data=train,y="winddirection",x="rainfall",palette = "coolwarm")
f.add_subplot(223)
sns.boxplot(data=train,y="windspeed",x="rainfall",palette = "dark")
f.add_subplot(224)
sns.boxplot(data=train,y="temparature",x="rainfall",palette = "Set1")


f = plt.figure(figsize=(10,10))
f.add_subplot(121)
sns.violinplot(data=train,y="pressure",x="rainfall",palette = "rainbow")
f.add_subplot(122)
sns.violinplot(data=train,y="mintemp",x="rainfall",palette = "coolwarm")


selected_columns = ['sunshine','humidity', 'cloud','pressure']
tr = train[selected_columns]
te = test[selected_columns]
tr_y = train['rainfall']
tr.head(5)


model = xg.XGBClassifier()
scaler = StandardScaler()
tr_scaled = scaler.fit_transform(tr)
te_scaled = scaler.transform(te)
X_tr,X_te,y_tr,y_te = train_test_split(tr,tr_y,test_size=0.25)
model.fit(X_tr,y_tr)


predictions = model.predict(X_te)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_te,predictions)


final_predictions = model.predict(te)



submission['rainfall'] = final_predictions


submission['rainfall'].value_counts()


submission.to_csv('submission.csv',index=False)

