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


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col = 0)
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col=0)



train.head()


test.head()


from sklearn.preprocessing import LabelEncoder


train = pd.get_dummies(train, columns=['Soil Type', 'Crop Type'])


le = LabelEncoder()
train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])  # Save encoder


train.head()


X = train.drop('Fertilizer Name', axis=1)
y = train['Fertilizer Name']



X.head()


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100)




model.fit(X_train, y_train)



y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt

feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.show()


# Apply the same preprocessing used in training
test_encoded = pd.get_dummies(test, columns=['Soil Type', 'Crop Type'])

# Ensure test columns match training columns
missing_cols = set(X_train.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0

test_encoded = test_encoded[X_train.columns]  # Align column order



y_pred = model.predict(test_encoded)



fertilizer_names = le.inverse_transform(y_pred)



submission = pd.DataFrame({
    'id': test.index,  # or test['id'] if available
    'Fertilizer Name': fertilizer_names
})
submission.to_csv('submission.csv', index=False)




