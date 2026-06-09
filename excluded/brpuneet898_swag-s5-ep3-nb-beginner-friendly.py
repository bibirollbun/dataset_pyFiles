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

train_dataset = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


test_dataset['winddirection'].fillna(test_dataset['winddirection'].mean(), inplace=True)


print(train_dataset.isnull().sum())
print(test_dataset.isnull().sum())


# print("---------Training Dataset----------")
# print(type(train_dataset))
# print(train_dataset.head())
# print(train_dataset.info())
# print("---------Testing Dataset-----------")
# print(type(test_dataset))
# print(test_dataset.head())
# print(test_dataset.info())


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
# import tensorflow as tf
# from tensorflow.keras import layers, models


# X = train_dataset.drop(columns=["id", "day", "rainfall"])  
# y = train_dataset["rainfall"]


X_train = train_dataset.drop(columns=["id", "day", "rainfall"]) 
y_train = train_dataset["rainfall"]


scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.fit_transform(X_train)


# X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)


X_test = test_dataset.drop(columns=["id", "day"])
X_test_scaled = scaler.transform(X_test)


# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)


# X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
# X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)


# model = models.Sequential([
#     layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
#     layers.MaxPooling1D(),
#     layers.Conv1D(128, 3, activation='relu'),
#     layers.MaxPooling1D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(1, activation='sigmoid')  
# ])


# import xgboost as xgb

# model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)


# from catboost import CatBoostClassifier

# model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, cat_features=[], verbose=0)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)


# from lightgbm import LGBMClassifier

# model = LGBMClassifier(n_estimators=1000, learning_rate=0.1, max_depth=-1)


# from sklearn.svm import SVC

# model = SVC(probability=True, kernel='rbf', C=1, gamma='scale')


# from sklearn.ensemble import GradientBoostingClassifier

# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)


# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))


# model.fit(X_train, y_train)


model.fit(X_train_scaled, y_train)


# model.fit(X_train_scaled, y_train, epochs=45, batch_size=64)


# y_pred_prob = model.predict(X_val)


# y_pred_prob = model.predict_proba(X_val)[:, 1]


y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] 


# auc_score = roc_auc_score(y_val, y_pred_prob)

# print(f"AUC Score: {auc_score}")


# y_pred_prob = model.predict(X_test_scaled)


submission = pd.DataFrame({
    "id": test_dataset["id"],
    "rainfall": y_pred_prob.flatten() 
})


submission.head()


print(submission.isnull().sum())


submission.to_csv("submission.csv", index=False)

