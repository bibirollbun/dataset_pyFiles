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


train = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
train.head()


train.info()


# test setimizni ham load qilib olamiz
test = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')


# bizga Customerid va surname ustunlari exited ni 
# aniqlashimizga yordam bermaydi, ularni tashlab yuborsak bo'ladi

train = train.drop(columns=['CustomerId', 'Surname'])


# submission imiz uchun test setdagi Customerid larni saqlab qo'yamiz

testId = test['CustomerId']


# test setimizdan ham Customerid va surname ustunlarini olib tashlaymiz

test = test.drop(columns=['CustomerId', 'Surname'])


# bizda 2 ta ustunimiz - geography va gender categorical ustun ekan
# ya'ni encode qilishni talab qiladi

from sklearn.preprocessing import LabelEncoder

for i in ['Geography', 'Gender']:
    encoder = LabelEncoder()
    train[i] = encoder.fit_transform(train[i])
    test[i] = encoder.fit_transform(test[i])


# bizda test set alohida faylda bor. lekin uni oxirgi tekshiruv uchun olib qo'yamiz
# ungacha train faylimizni train-test setlarga ajratib, modelimizni baholaymiz

from sklearn.model_selection import train_test_split

X = train.drop('Exited', axis=1)
y = train['Exited']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=7)


# birinchi model uchun Random Forestdan foydalanamiz. Nega?
# - LR ga o'xshab ustunlarda 0-1 oraligini talab qilmaydi
# - datasetimiz juda katta emas - overfittingni oldini olish uchun RF yaxshi varinat
# - boshlanishiga oson va tez
# - exited ni predict qilishimizda qaysi ustunlar muhim ekanligini ko'rsatadi

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=7)
model.fit(X_train, y_train)


# baholab ko'ramiz

from sklearn.metrics import roc_auc_score

y_pred = model.predict_proba(X_test)[:,1]
print('Auc Roc:', roc_auc_score(y_test, y_pred))


# qaysi ustunlar muhimligini bilish uchun tayyor internetdan olingan koddan foydalandim:

import matplotlib.pyplot as plt

importances = model.feature_importances_
feat_names = X.columns
plt.barh(feat_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.show()


# XGBOOST MODEL TRAININH

import xgboost as xgb

xgbmodel = xgb.XGBClassifier(
    max_depth=5,
    n_estimators=400,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.5,
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=7
)

xgbmodel.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=True)


y_xgb_pred = model.predict_proba(X_test)[:,1]
print('AUC ROC: ', roc_auc_score(y_test, y_xgb_pred))


testProbs = model.predict_proba(test)[:,1]
submission = pd.DataFrame({
    'id': testId,
    'Exited': testProbs
})
submission.to_csv('submission.csv', index=False)
submission.head()


def ketish_ehtimoli(probability):
    if probability <= 0.20:
        return 'Kam'
    elif probability <= 0.60:
        return "O'rtacha"
    else:
        return 'Katta'

submission['KetishEhtimoli'] = submission['Exited'].apply(ketish_ehtimoli)
submission.head()

