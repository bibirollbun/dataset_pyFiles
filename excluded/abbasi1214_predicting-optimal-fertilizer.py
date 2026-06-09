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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train
test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test
submission=pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission


train.head()
train.info()
train['Fertilizer Name'].value_counts


train.isnull().sum()
train.duplicated()
train.duplicated().sum()
sns.countplot(x="Fertilizer Name",data=train)
plt.title("Count plot for Train data")
plt.legend()
plt.show()


train_encoded=pd.get_dummies(train,drop_first=True).astype(int)
train_encoded


X=train_encoded.drop(columns=['Fertilizer Name_14-35-14','Fertilizer Name_17-17-17','Fertilizer Name_20-20','Fertilizer Name_28-28','Fertilizer Name_DAP','Fertilizer Name_Urea'])
y=train['Fertilizer Name']


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)


y_pred=model.predict(X_test)
y_pred
print(classification_report(y_test,y_pred))


test_encoded=pd.get_dummies(test,drop_first=True)


test_preds = model.predict(test_encoded)
submission['Fertilizer'] = test_preds
submission.to_csv('submission.csv', index=False)


