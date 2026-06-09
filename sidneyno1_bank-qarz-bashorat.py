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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Fayllarni o‘qiymiz
train_df = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv")
test_df = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv")

# Foydasiz ustunlarni olib tashlaymiz
train_df = train_df.drop(columns=['id', 'CustomerId', 'Surname'])

# Kategorik ustunlarni kodlaymiz
cat_cols = ['Geography', 'Gender']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le


X = train_df.drop(columns=['Exited'])
y = train_df['Exited']

#Trening va validatsiyaga ajratamiz
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#RandomForestClassifierdan foydalanimiz
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# predict qilamzz
y_pred = model.predict(X_valid)
print(classification_report(y_valid, y_pred, target_names=['Stayed (0)', 'Exited (1)'], zero_division=0))


