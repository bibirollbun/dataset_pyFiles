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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
print(df.columns.tolist())


df = df.drop(columns=['id'])

binary_columns = ['Stage_fear', 'Drained_after_socializing']
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

le = LabelEncoder()
df['Personality'] = le.fit_transform(df['Personality'])  # Extrovert=1, Introvert=0

X = df.drop(columns=['Personality'])
y = df['Personality']


import seaborn as sns
import matplotlib.pyplot as plt

corr = X.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


X = X.drop(columns=["Drained_after_socializing"])

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.4f}".format(accuracy))


from lightgbm import LGBMClassifier

model = LGBMClassifier(random_state=42, n_estimators=100)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

from sklearn.metrics import accuracy_score
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

test_df = test_df.drop(columns=['id'])

binary_columns = ['Stage_fear', 'Drained_after_socializing']
for col in binary_columns:
    test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

test_df = test_df.drop(columns=["Drained_after_socializing"])

test_imputed = imputer.transform(test_df)
test_scaled = scaler.transform(test_imputed)


test_preds = model.predict(test_scaled)
test_preds_label = le.inverse_transform(test_preds)


test_ids = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')['id']

submission = pd.DataFrame({
    'id': test_ids,
    'Personality': test_preds_label
})

submission.to_csv('submission.csv', index=False)
print("File is successfully submitted as submission.csv")

