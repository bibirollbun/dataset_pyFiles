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
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# 1-dropna bilan nanlarni yo'qotgandim bo'lmadi boshidan yozdim keyin
# Yana shu yerda aytib ketaman columnslarda no odatiy qiymatlar ko'p ammo...
# ...ularni olib tashasam test.csv da ham ular bor ekan shuning uchun ularga tegmadim... 
# ...Ularni olib tashasam model nabarot yamon natija ko'rsatdi


df_train=pd.read_csv("/kaggle/input/multiclassificationtask/train.csv", index_col='id')
df_train.head()


df_train.shape


df_train.info()
# NaN qiymatlar bor


df_train.describe()
# Har bir ustunni tekshirish kerak yamon qiymatlar bor


df_train.isnull().sum()


# df_train=df_train.dropna()
# df_train.shape


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['N_Days'])
plt.show()
# 8000 dan tepasi kerakmas


# df_train=df_train[df_train['N_Days']<8000]
# plt.figure(figsize=(10,5))
# sns.scatterplot(df_train['N_Days'])
# plt.show() # Yaxshi


df_train['Age']=df_train['Age']//365.25
plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Age'])
plt.show()
# 100 dan tepasi va 20dan pasti kerakmas


# df_train=df_train[(df_train['Age']<100) & (df_train['Age']>20)]
# plt.figure(figsize=(10,5))
# sns.scatterplot(df_train['Age'])
# plt.show()


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Bilirubin'])
plt.show()
# Narmalniy


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Cholesterol'])
plt.show()
# bularam yamonmas


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Albumin'])
plt.show()   


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Copper'])
plt.show()    #  Yaxshi


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Alk_Phos'])
plt.show()    #  Yaxshi


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['SGOT'])
plt.show()   
# 500 dan kattalarini olib tashayman


# df_train=df_train[df_train['SGOT']<500]
# plt.figure(figsize=(10,5))
# sns.scatterplot(df_train['SGOT'])
# plt.show()    #  Yaxshi


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Tryglicerides'])
plt.show()
# 500 dan tepasini olib tashlayman   


# df_train=df_train[df_train['Tryglicerides']<500]
# plt.figure(figsize=(10,5))
# sns.scatterplot(df_train['Tryglicerides'])
# plt.show()    #  Yaxshi


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Platelets'])
plt.show() 
# 600 dan tepasi kerakmas  


# df_train=df_train[df_train['Platelets']<600]
# plt.figure(figsize=(10,5))
# sns.scatterplot(df_train['Platelets'])
# plt.show()    #  Yaxshi


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Prothrombin'])
plt.show()    
# 15 dan pasti kerak


# df_train[df_train['Prothrombin']>15]


# df_train=df_train[df_train['Prothrombin']<15]
# plt.figure(figsize=(10,5))
# sns.scatterplot(df_train['Prothrombin'])
# plt.show()    #  Yaxshi


plt.figure(figsize=(10,5))
sns.scatterplot(df_train['Stage'])
plt.show()    #  Yaxshi


df_train['Status'].value_counts()
# Bu yerda Y adashib tushib ketgan shekilli


df_train=df_train[df_train['Status']!='Y']


df_train['Status']=df_train['Status'].replace({'C':0, 'CL':1, 'D':2})
df_train.sample(5)


train,  test = train_test_split(df_train, train_size=0.9, random_state=42)
x_train = train.drop(columns='Status')
y_train=train['Status'].copy()
x_test = test.drop(columns='Status')
y_test = test['Status'].copy()


num_col=x_train.describe().columns
num_col


cat_col=['Drug', 'Sex',	'Ascites',	'Hepatomegaly',	'Spiders', 'Edema']


num_pipline=Pipeline([
  ('imputer1', SimpleImputer(strategy='median')),
  ('standartla', StandardScaler())
])
cat_pipline=Pipeline([
  ('imputer2', SimpleImputer(strategy='most_frequent')),
  ('encoder', OneHotEncoder())
])


pipline=ColumnTransformer([
  ("sonlar", num_pipline, num_col), 
  ('katigoryalar', cat_pipline, cat_col)
])


models={"Tree_model":DecisionTreeClassifier(random_state=42), "RF_model":RandomForestClassifier(random_state=42), "XGBC_model":XGBClassifier(random_state=42)}


for name, model in models.items():
  full_pipline=Pipeline([
  ('data', pipline),
  (name, model)
          ])
  full_pipline.fit(x_train, y_train)
  y_predict=full_pipline.predict_proba(x_test)
  print(name, log_loss(y_test, y_predict))


# Eng yaxshi model XGBC_model ekan


# testlash
y_predict=full_pipline.predict_proba(x_test)
y_predict=np.clip(y_predict, 1e-15, 1 - 1e-15)
log_loss(y_test, y_predict)


df_test=pd.read_csv('/kaggle/input/multiclassificationtask/test.csv', index_col='id')
df_test.head()


df_test['Age']=df_test['Age']//365.25


df_test.describe()
# test ustunlarida ham xatoliklar borku


df_train.describe()


df_test.isnull().sum()


natija=full_pipline.predict_proba(df_test)
natija=np.clip(natija,1e-15, 1 - 1e-15)


df=pd.DataFrame({"id": df_test.index, 'Status_C':natija[:, 0], 'Status_CL':natija[:, 1], "Status_D": natija[:, 2]})
df


df.to_csv('submission.csv', index=False)


df_train.shape




