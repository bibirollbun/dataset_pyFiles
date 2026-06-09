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
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve


df_train=pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv', index_col='id')
df_train


df_test=pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv', index_col='id')
df_test


df_train.info()
# Bundan NaN qiymatlar yo'qligini sonli ustunlar sonli ekanligini ko'rishimiz mumkin


df_train.describe()
# Bunda ko'rinadiki Balance ustunini bir tekshirish kerak. Va Exited ustuni ham bir tamonga og'ib ketgan


plt.figure(figsize=(10,6))
sns.scatterplot(df_train['Balance'])
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(x=df_train['Exited'])
plt.show()
# Ha bu yerda katta og'ish bor


df_train['Exited'].value_counts()


df_train=df_train.drop(columns=['Surname', 'CustomerId'])
df_train.head()


# df_0 = df_train[df_train['Exited'] == 0].sample(3000, random_state=42)
# df_1 = df_train[df_train['Exited'] == 1]
# df_train = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)  # aralashtiramiz
# df_train.head()    #og'ib ketishni yo'qotish uchun


train, test = train_test_split(df_train, test_size=0.1, random_state=42)
x_train=train.drop(columns='Exited')
y_train=train['Exited'].copy()
x_test=test.drop(columns='Exited')
y_test=test['Exited'].copy()


num_cols=['CreditScore',	'Age', 	'Tenure',	'Balance', 'NumOfProducts',	'IsActiveMember',	'EstimatedSalary']
cat_cols=['Geography', 'Gender']


preprocessor=ColumnTransformer([('standartla', StandardScaler(), num_cols),
             ('raqamla', OneHotEncoder(handle_unknown='ignore'), cat_cols)])
# 4- kamchilik to'g'irlandi yani onehotencoderni ishlatdim


full_pipline=Pipeline([
  ('preprocessor', preprocessor),
  ('model', RandomForestClassifier(random_state=42))
])
# 5- kamchilik to'g'irlandi ya'ni umumiy pipline yaratdim


full_pipline.fit(x_train, y_train)


y_predict=full_pipline.predict_proba(x_test)[:, 1]
y_predict
# 3-kamchilik yo'qotildi


roc_auc_score(y_test, y_predict)
# 2- kamchilik to'g'irlandi


fpr,tpr,thresholds=roc_curve(y_test, y_predict)
plt.figure(figsize=(10,5))
plt.plot(fpr, tpr, linestyle='--')
plt.show()


df_test.head()


df_test=df_test.drop(columns=['Surname', 'CustomerId'])
df_test.head()


predict=full_pipline.predict_proba(df_test)[:, 1]
predict
# 1- kamchilik to'g'irlandi


Natija=pd.DataFrame({"id": df_test.index, "Exited": predict})
Natija.to_csv("Submission.csv", index=False)




