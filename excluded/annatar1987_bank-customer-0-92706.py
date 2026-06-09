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
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/bank-customer-churn-prediction-challenge/train.csv")
test = pd.read_csv("/kaggle/input/bank-customer-churn-prediction-challenge/test.csv")
submission = pd.read_csv("/kaggle/input/bank-customer-churn-prediction-challenge/sample_submission.csv")


# Сохраняем ID для финального сабмита
test_ids = test['id']


# Целевая переменная
y = train['Exited']
X = train.drop(columns=['id', 'Exited'])
X_test = test.drop(columns=['id'])


# Объединение для совместной предобработки
full = pd.concat([X, X_test], axis=0)


# Кодирование категориальных признаков
for col in full.select_dtypes(include='object').columns:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))


# Делим обратно
X = full.iloc[:len(X), :]
X_test = full.iloc[len(X):, :]


# Разделение на train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
#model = LogisticRegression(max_iter=1000)
#model.fit(X_train, y_train)


# Оценка
#y_pred = model.predict(X_val)
#print(classification_report(y_val, y_pred))


# Предсказания и сабмит
#preds = model.predict(X_test)
#submission['Exited'] = preds
#submission.to_csv("submission.csv", index=False)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=5)
print(f"Baseline RMSE: {scores.mean():.4f}")

# Обучение и предсказание
model.fit(X, y)
preds = model.predict(X_test)

# Готовим submission
submission['Exited'] = preds
submission.to_csv("submission.csv", index=False)




