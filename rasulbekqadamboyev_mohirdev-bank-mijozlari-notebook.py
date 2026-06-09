import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import os

# 1. Kaggle notebookdagi ma'lumotlar jildining asosiy yo'li
# Musobaqa nomi "binaryclassificationwithabankchurndataset"
base_path = '/kaggle/input/binaryclassificationwithabankchurndataset/'

# 2. Har bir faylning to'liq yo'lini aniqlaymiz
train_file_path = os.path.join(base_path, 'train.csv')
test_file_path = os.path.join(base_path, 'test.csv')
submission_file_path = os.path.join(base_path, 'sample_submission.csv')

# 3. Fayllarni o'qib, siz so'ragan o'zgaruvchilarga saqlaymiz
df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)
df_sub = pd.read_csv(submission_file_path)

# 4. Hammasi to'g'ri yuklanganini tekshirish uchun
print("--- df_train (O'qitish ma'lumotlari) ---")
print(df_train.head())
print("\n")

print("--- df_test (Test ma'lumotlari) ---")
print(df_test.head())
print("\n")

print("--- df_sub (Namuna submission fayli) ---")
print(df_sub.head())


# ma'lumotlarni tekshirish
df_train.head()


# drop customer id and surname
df_train.drop(['CustomerId', 'Surname'], axis=1, inplace=True)
df_test.drop(['CustomerId', 'Surname'], axis=1, inplace=True)


# labeling geography and gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['Geography'] = le.fit_transform(df_train['Geography'])
df_train['Gender'] = le.fit_transform(df_train['Gender'])
df_test['Geography'] = le.fit_transform(df_test['Geography'])
df_test['Gender'] = le.fit_transform(df_test['Gender'])


# correlation Exited
df_train.corr()['Exited'].abs().sort_values(ascending=False)


# creating good train set
cor = ['Age', 'NumOfProducts', 'IsActiveMember', 'Gender','Balance','Exited']
train = df_train[cor]
# train_set test_set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train, test_size=0.1, random_state=42)
x_train = train_set.drop('Exited', axis=1)
y_train = train_set['Exited']
x_test = test_set.drop('Exited', axis=1)
y_test = test_set['Exited']
cor1 = ['Age', 'NumOfProducts', 'IsActiveMember', 'Gender','Balance']
TEST = df_test[cor1] # for submission file
TEST_ID = df_test['id'] # for submission file


# standart scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
TEST = sc.transform(TEST) # for submission file


#XGB
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=150, max_depth=2,random_state=42)
xgb.fit(x_train, y_train)

# predict probablities
y_pred = xgb.predict_proba(TEST)[:,1]


# saving y_pred in csv file
y_pred_df = pd.DataFrame(y_pred, columns=['Exited'])

# columns id and Exited
y_pred_df['id'] = TEST_ID
y_pred_df = y_pred_df[['id', 'Exited']]

# saving csv file
y_pred_df.to_csv('sample_submission.csv', index=False)




