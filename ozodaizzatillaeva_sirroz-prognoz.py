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


train = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
train.head()


train.info()


trainId = train['id']
train = train.drop(columns=['id'])

kategorik = ['Drug', 'Ascites', 'Hepatomegaly', 'Spiders']
for i in kategorik:
    train[i] = train[i].fillna('Unknown')

numerik = ['Cholesterol', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
for i in numerik:
    train[i] = train[i].fillna(train[i].median())

train.info()


# target ustunimizni numerikka o'tkazamiz

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train['Status'] = encoder.fit_transform(train['Status'])


# Kategorik ustunlarga sun'iy tartib berilmasligi uchun LabelEncoder emas, OneHotEncoderdan foydalanamiz

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

kategorik = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

encoder = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, kategorik),
        ('num', 'passthrough', numerik)
    ]
)


train['Status'].value_counts()


# Demak, bizda 1 dona begona qiymat bor, adashib kiritilgan bo'lsa kerak
# chunki Status ustunimizda faqat 3 xil qiymat (C, CL, D) bo'lishi kerak

# Shu qatorni tashlab yubiramiz

train = train[train['Status'] != 3]

train['Status'].value_counts()


# Endi train test setga ajratsak bo'ladi

from sklearn.model_selection import train_test_split

X = train.drop(columns=['Status'])
y = train['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=7)


from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

XGB_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=7
    ))
])

XGB_pipeline.fit(X_train, y_train)

# Baholash

evaluation = XGB_pipeline.predict_proba(X_test)

print("Log Loss:", log_loss(y_test, evaluation))



# tayyorlash

test = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
testID = test['id']
test = test.drop(columns=['id'])

for i in kategorik:
    test[i] = test[i].fillna('Unknown')

for i in numerik:
    test[i] = test[i].fillna(train[i].median()) 


# bashorat qilish
test_predict = XGB_pipeline.predict_proba(test)

# submission
submission = pd.DataFrame(test_predict, columns=['Status_C', 'Status_CL', 'Status_D'])
submission.insert(0, 'id', testID)

# ehtimolliklarni cheklash
submission[['Status_C', 'Status_CL', 'Status_D']] = submission[['Status_C', 'Status_CL', 'Status_D']].clip(1e-15, 1 - 1e-15)


submission.to_csv('submission.csv', index=False)

submission.head()

