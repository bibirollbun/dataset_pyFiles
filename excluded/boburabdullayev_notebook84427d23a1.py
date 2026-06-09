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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Ma'lumotlarni yuklash
train_data = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
test_data = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')

# Kategorik o'zgaruvchilarni kodlash
le_geography = LabelEncoder()
le_gender = LabelEncoder()
train_data['Geography'] = le_geography.fit_transform(train_data['Geography'])
train_data['Gender'] = le_gender.fit_transform(train_data['Gender'])
test_data['Geography'] = le_geography.transform(test_data['Geography'])
test_data['Gender'] = le_gender.transform(test_data['Gender'])

# Yo'qolgan qiymatlarni to'ldirish
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

# Feature Engineering
train_data['Balance_to_Salary_Ratio'] = train_data['Balance'] / (train_data['EstimatedSalary'] + 1e-6)
test_data['Balance_to_Salary_Ratio'] = test_data['Balance'] / (test_data['EstimatedSalary'] + 1e-6)

# Xususiyatlarni tanlash
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
            'Balance_to_Salary_Ratio']
X = train_data[features]
y = train_data['Exited']

# Ma'lumotlarni normalizatsiya qilish
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data[features])

# Ma'lumotlarni bo'lish
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelni o'qitish
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Bashorat qilish
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model accuracy: {accuracy * 100:.2f}%")

# Test ma'lumotlari uchun bashorat
test_predictions = model.predict_proba(test_data_scaled)[:, 1]

# Natijani faylga saqlash
submission = pd.DataFrame({'id': test_data['id'], 'Exited': test_predictions})
submission.to_csv('submission_logistic.csv', index=False)

# Izohlar:
# Kaggle foydalanuvchi nomi: boburabdullayev
# Foydalanilgan algoritm: Logistic Regression
# Model aniqligi: {accuracy * 100:.2f}%


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Ma'lumotlarni yuklash
train_data = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
test_data = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')

# Kategorik o'zgaruvchilarni kodlash
le_geography = LabelEncoder()
le_gender = LabelEncoder()
train_data['Geography'] = le_geography.fit_transform(train_data['Geography'])
train_data['Gender'] = le_gender.fit_transform(train_data['Gender'])
test_data['Geography'] = le_geography.transform(test_data['Geography'])
test_data['Gender'] = le_gender.transform(test_data['Gender'])

# Yo'qolgan qiymatlarni to'ldirish
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

# Feature Engineering
train_data['Balance_to_Salary_Ratio'] = train_data['Balance'] / (train_data['EstimatedSalary'] + 1e-6)
test_data['Balance_to_Salary_Ratio'] = test_data['Balance'] / (test_data['EstimatedSalary'] + 1e-6)

# Xususiyatlarni tanlash
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
            'Balance_to_Salary_Ratio']
X = train_data[features]
y = train_data['Exited']

# Ma'lumotlarni normalizatsiya qilish
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data[features])

# Ma'lumotlarni bo'lish
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelni o'qitish
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Bashorat qilish
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model accuracy: {accuracy * 100:.2f}%")

# Test ma'lumotlari uchun bashorat
test_predictions = model.predict_proba(test_data_scaled)[:, 1]

# Natijani faylga saqlash
submission = pd.DataFrame({'id': test_data['id'], 'Exited': test_predictions})
submission.to_csv('submission_random_forest.csv', index=False)

# Izohlar:
# Kaggle foydalanuvchi nomi: boburabdullayev
# Foydalanilgan algoritm: Random Forest
# Model aniqligi: {accuracy * 100:.2f}%


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Ma'lumotlarni yuklash
train_data = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
test_data = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')

# Kategorik o'zgaruvchilarni kodlash
le_geography = LabelEncoder()
le_gender = LabelEncoder()
train_data['Geography'] = le_geography.fit_transform(train_data['Geography'])
train_data['Gender'] = le_gender.fit_transform(train_data['Gender'])
test_data['Geography'] = le_geography.transform(test_data['Geography'])
test_data['Gender'] = le_gender.transform(test_data['Gender'])

# Yo'qolgan qiymatlarni to'ldirish
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

# Feature Engineering
train_data['Balance_to_Salary_Ratio'] = train_data['Balance'] / (train_data['EstimatedSalary'] + 1e-6)
test_data['Balance_to_Salary_Ratio'] = test_data['Balance'] / (test_data['EstimatedSalary'] + 1e-6)

# Xususiyatlarni tanlash
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
            'Balance_to_Salary_Ratio']
X = train_data[features]
y = train_data['Exited']

# Ma'lumotlarni normalizatsiya qilish
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data[features])

# Ma'lumotlarni bo'lish
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modellarni tayyorlash
xgb_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)
)

rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)

# Ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model)
], voting='soft')

# Modelni o'qitish
ensemble_model.fit(X_train, y_train)

# Bashorat qilish
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model accuracy: {accuracy * 100:.2f}%")

# Test ma'lumotlari uchun bashorat
test_predictions = ensemble_model.predict_proba(test_data_scaled)[:, 1]

# Natijani faylga saqlash
submission = pd.DataFrame({'id': test_data['id'], 'Exited': test_predictions})
submission.to_csv('submission_ensemble.csv', index=False)

# Izohlar:
# Kaggle foydalanuvchi nomi: boburabdullayev
# Foydalanilgan algoritm: Ensemble (XGBoost + Random Forest)
# Model aniqligi: {accuracy * 100:.2f}%


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Ma'lumotlarni yuklash
train_data = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
test_data = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')

# Kategorik o'zgaruvchilarni kodlash
le_geography = LabelEncoder()
le_gender = LabelEncoder()
train_data['Geography'] = le_geography.fit_transform(train_data['Geography'])
train_data['Gender'] = le_gender.fit_transform(train_data['Gender'])
test_data['Geography'] = le_geography.transform(test_data['Geography'])
test_data['Gender'] = le_gender.transform(test_data['Gender'])

# Yo'qolgan qiymatlarni to'ldirish
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

# Feature Engineering
train_data['Balance_to_Salary_Ratio'] = train_data['Balance'] / (train_data['EstimatedSalary'] + 1e-6)
test_data['Balance_to_Salary_Ratio'] = test_data['Balance'] / (test_data['EstimatedSalary'] + 1e-6)

# Xususiyatlarni tanlash
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
            'Balance_to_Salary_Ratio']
X = train_data[features]
y = train_data['Exited']

# Ma'lumotlarni normalizatsiya qilish
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data[features])

# Ma'lumotlarni bo'lish
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelni o'qitish
model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)
)
model.fit(X_train, y_train)

# Bashorat qilish
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model accuracy: {accuracy * 100:.2f}%")

# Test ma'lumotlari uchun bashorat
test_predictions = model.predict_proba(test_data_scaled)[:, 1]

# Natijani faylga saqlash
submission = pd.DataFrame({'id': test_data['id'], 'Exited': test_predictions})
submission.to_csv('submission_xgboost.csv', index=False)

# Izohlar:
# Kaggle foydalanuvchi nomi: boburabdullayev
# Foydalanilgan algoritm: XGBoost
# Model aniqligi: {accuracy * 100:.2f}%

