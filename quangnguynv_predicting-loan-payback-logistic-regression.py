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


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train.drop(columns='id', inplace=True)
train.head()


train['loan_paid_back'].describe


test.head()


train.isna().sum()


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Tách nhãn (target)
y = train['loan_paid_back']
train = train.drop(columns='loan_paid_back')
test_id = test['id']

# Phân loại cột
numeric = train.select_dtypes(exclude='object').columns
categoric = train.select_dtypes(include='object').columns

# Mã hóa one-hot nếu có cột phân loại
if len(categoric) > 0:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(train[categoric])

    train_encoded = pd.DataFrame(
        encoder.transform(train[categoric]),
        columns=encoder.get_feature_names_out(categoric),
        index=train.index
    )

    test_encoded = pd.DataFrame(
        encoder.transform(test[categoric]),
        columns=encoder.get_feature_names_out(categoric),
        index=test.index
    )

    train = pd.concat([train[numeric], train_encoded], axis=1)
    test = pd.concat([test[numeric], test_encoded], axis=1)

# Đảm bảo không có giá trị NaN
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler();
train[numeric] = scaler.fit_transform(train[numeric])
test[numeric] = scaler.transform(test[numeric])


from sklearn.model_selection import train_test_split

X = train

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

model = LogisticRegression(
    max_iter=5000,
    penalty='l2',
    C=1e-3,
)
model.fit(X_train, y_train)

y_prob_val = model.predict_proba(X_val)[:, 1]

threshold = 0.8
y_pred = (y_prob_val >= threshold).astype(int)

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)

# ROC
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


model.fit(X, y)
y_prob = model.predict_proba(test)[:, 1]

prediction = (y_prob >= threshold).astype(int)

submission = pd.DataFrame({
    'id': test_id,
    'loan_paid_back': prediction
})
submission.to_csv('submission.csv', index=False)


submission.head()

