# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import log_loss
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 1. Завантаження даних
train = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv")
test = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv")


# 2. Обробка фіч
X = train.drop(['id', 'target'], axis=1)
y = train['target']
X_test = test.drop(['id'], axis=1)


# Перетворюємо ціль на числову
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# Розділення на train/val
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded)



# Модель SVM
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)


# Ймовірності
y_val_proba = model.predict_proba(X_val)




# Log Loss
y_val_bin = pd.get_dummies(y_val)
loss = log_loss(y_val_bin, y_val_proba)
print(f"Validation Log Loss: {loss:.4f}")




# Submission
y_test_proba = model.predict_proba(X_test_scaled)
submission = pd.DataFrame(y_test_proba, columns=label_encoder.classes_)
submission.insert(0, 'id', test['id'])
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

