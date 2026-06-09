# Kerakli kutubxonalarni yuklab olish
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Ma'lumotlarni pandas kutubxonasi orqali train_dataset.csv faylidan o'qib olamiz
df = pd.read_csv('/kaggle/input/aviakompaniya/train_dataset.csv', index_col='id')
df.head()


# Arrival Delay in Minutes dagi null larni o'rtacha qiymat bilan to'ldirib olamiz
df.fillna(df["Arrival Delay in Minutes"].mean(), inplace=True)

# Null qiymatlar qolmadi tekshirib olamiz
df.isnull().sum()


# ColumnTransformer uchun number ustunlarini ajratib olamiz
number_columns = df.drop(['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction'], axis=1)


# Ma'lumotlarni distributsiyani ko'rish
%matplotlib inline
number_columns.hist(bins=50, figsize=(20, 15))
plt.show()


#Transformer yasash object columnlar uchun OneHotEncoder preprocessingni,number ustunlari uchun StandardScaler preprocessingni qo'llaymiz
cat_attr = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
column_transformer = ColumnTransformer([
    ('standard_scaler', StandardScaler(), list(number_columns)),
    ('one_hot_encoder', OneHotEncoder(), cat_attr)
])


#DataFramedan labelni ajratib olamiz
X = df.drop('satisfaction', axis=1)
Y = df['satisfaction'].copy()


#DataFrameni X_train, X_test, y_train, y_test larga ajratib olamiz
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#Train ma'lumotlarni o'qitish uchun tayyorlash
X_train_prepared = column_transformer.fit_transform(X_train)

#Test ma'lumotlarni test qilish uchun tayyorlash
X_test_prepared = column_transformer.transform(X_test)


#Support Vector Machine modelini o'qitish
SVM_model = svm.SVC(kernel='linear', gamma='scale', C=0.1, probability=True)
SVM_model.fit(X_train_prepared, Y_train)


# accuracy_score hisoblash
X_test_predict = SVM_model.predict(X_test_prepared)
tas = accuracy_score(Y_test, X_test_predict)
print("accuracy_score:", tas * 100)

#roc_auc_score hisoblash
X_test_predict_proba = SVM_model.predict_proba(X_test_prepared)[:, 1]
roc_auc = roc_auc_score(Y_test, X_test_predict_proba)
print(f"roc_auc: {roc_auc:.4f}")


#RandomForestClassifier modelini o'qitish
RFC_model = RandomForestClassifier(n_estimators=500, random_state=42)

# RandomForestClassifier modelni o‘qitamiz
RFC_model.fit(X_train_prepared, Y_train)


# accuracy_score hisoblash
X_test_predict = RFC_model.predict(X_test_prepared)
tas = accuracy_score(Y_test, X_test_predict)
print("accuracy_score:", tas * 100)

Y_pred_prob = RFC_model.predict_proba(X_test_prepared)[:, 1]

# ROC AUC hisoblash
roc_auc = roc_auc_score(Y_test, Y_pred_prob)
print(f"roc_auc: {roc_auc:.4f}")


#Modelni sinab ko'ramiz buning uchun yangi test datani yuklab olamiz
test_df = pd.read_csv('/kaggle/input/aviakompaniya/test_dataset.csv', index_col="id")

#Ma'lumotlarni modelga o'qitish uchun tayyorlaymiz
test_df = test_df.drop(columns=['satisfaction'], errors='ignore')
test_df.fillna(test_df["Arrival Delay in Minutes"].mean(), inplace=True)
test_df_prepared = column_transformer.transform(test_df)

#Predict qilamiz
test_predict = RFC_model.predict(test_df_prepared)

# Olingan natijani submission.csv fayliga yuklab olamiz
predict_df = pd.DataFrame({'id': test_df.index, 'satisfaction': test_predict})
predict_df.to_csv("/kaggle/working/submission.csv", index=False)

