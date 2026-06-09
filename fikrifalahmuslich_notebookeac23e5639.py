# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/train.csv')
test = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/test.csv')
sample_submission = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/sample_submission.csv')



print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.head())


target_col = 'price'  # <-- change this if needed
X = train.drop(columns=[target_col])
y = train[target_col]


cat_cols = X.select_dtypes(include='object').columns  # get object (string) columns

for col in cat_cols:
    combined = pd.concat([X[col], test[col]], axis=0).astype(str)

    le = LabelEncoder()
    le.fit(combined)

    X[col] = le.transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))



imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test = pd.DataFrame(imputer.transform(test), columns=test.columns)



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Identify non-numeric columns
non_numeric_cols = X.select_dtypes(include='object').columns
print("Dropping these text columns:", list(non_numeric_cols))

# Drop them
X = X.drop(columns=non_numeric_cols)
test = test.drop(columns=non_numeric_cols)



# Use only 5000 samples to speed up during development
X_sample = X_train.sample(5000, random_state=42)
y_sample = y_train.loc[X_sample.index]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_sample, y_sample)




# Predict on validation set
val_preds = model.predict(X_val)

# Evaluate accuracy
from sklearn.metrics import accuracy_score
print("Validation Accuracy:", accuracy_score(y_val, val_preds))



# Predict on test set
test_preds = model.predict(test)

# Create submission DataFrame
submission = sample_submission.copy()
submission[target_col] = test_preds

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


