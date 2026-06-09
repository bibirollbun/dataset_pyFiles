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


d1=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


d1.info()


d1.head()


d1.isnull().sum()


d1.describe()


d2=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


d2.info()


d2.head()


d2.isnull().sum()


d2.describe()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.histplot(d1['Calories'], bins=30)
plt.title('Distribution of Calories')
plt.xlabel('Calories')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Calories', data=d1)
plt.title('Average Calories by Sex')
plt.show()


d1['Sex'].value_counts()


d1['Calories'].value_counts()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1['Sex']=le.fit_transform(d1['Sex'])
d2['Sex']=le.transform(d2['Sex'])


d1['BMI'] = d1['Weight'] / (d1['Height'] / 100) ** 2
d2['BMI'] =d2['Weight'] / (d2['Height'] / 100) ** 2


d1.head()


d2.head()


plt.figure(figsize=(15,10))
sns.heatmap(d1.corr(),annot=True,cmap='Blues')
plt.show()


X = d1.drop(['Calories', 'id'], axis=1)
y =d1['Calories']
X_test =d2.drop(['id'], axis=1)


X.shape


X_test.shape


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)


X_train.shape


X_val.shape


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='median')
X_train = si.fit_transform(X_train)
X_val = si.transform(X_val)
X_test = si.transform(X_test)
s=StandardScaler()
X_train=s.fit_transform(X_train)
X_val=s.transform(X_val)
X_test=s.transform(X_test)


from catboost import CatBoostRegressor
x= CatBoostRegressor(n_estimators=1000, learning_rate=0.1,loss_function='RMSE',random_state=42)
x.fit(X_train,y_train)


X_train


X_test


X_val


from sklearn.metrics import mean_squared_log_error
x1=np.maximum(0,x.predict(X_val))


np.sqrt(mean_squared_log_error(y_val,x1))


x2=np.maximum(0,x.predict(X_test))
submission = pd.DataFrame({
    'id': d2['id'],
    'Calories': x2
})
submission.to_csv('submission.csv', index=False)




