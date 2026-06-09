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


train_data  = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')


train_data.info()


for col in test_data.columns:
    mode_test = test_data[col].mode()[0]
    mode_train = train_data[col].mode()[0]
    test_data[col] = test_data[col].fillna(mode_test)
    train_data[col] = train_data[col].fillna(mode_train)


print(test_data.isna().sum(), "\n")
print(train_data.isna().sum())





from sklearn.calibration import LabelEncoder


labelEncoder = LabelEncoder()
cat_cols = list(train_data.select_dtypes(include=['object']).columns.difference(['Personality']))

# encoding common column in both train.csv and test.csv data
for col_name in cat_cols:
    train_data[col_name]=labelEncoder.fit_transform(train_data[col_name]).astype(int)
    test_data[col_name]=labelEncoder.transform(test_data[col_name]).astype(int)

# ecoding target column in train data only
train_data['Personality'] = labelEncoder.fit_transform(train_data['Personality'])


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train, test = train_test_split(train_data, test_size=0.2)
X = train.iloc[:, 0:-1]
y = train.Personality

model = LogisticRegression(penalty=None, random_state=42)
model.fit(X, y)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


X_test = test.iloc[:, 0:-1]
y_test = test.Personality #target column

preds = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, preds))
print("Recall: ", recall_score(y_test, y_pred=preds))
print("Precision: ", precision_score(y_test, y_pred=preds),
print("F1: ", f1_score(y_test, preds)))


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

_ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)


data = confusion_matrix(y_test, preds)

index = ['Actual Negative', 'Actual Positive']
columns = ['Predicted Negative', 'Predicted Positive']

# Base crosstab
df = pd.DataFrame(data, index=index, columns=columns)

# Add totals
df['Row Total'] = df.sum(axis=1)
df.loc['Column Total'] = df.sum()

df



test_data.head()


test_preds = model.predict(test_data)

# Mapping 0 & 1 to Extrovert & Introvert
label_map = {0: 'Extrovert', 1: 'Introvert'}
test_preds_mapped = [label_map[pred] for pred in test_preds]

ids = test_data.index

output = pd.DataFrame({'id': ids,
				   'Personality': test_preds_mapped})
output.to_csv('submission.csv', index=False)

output

