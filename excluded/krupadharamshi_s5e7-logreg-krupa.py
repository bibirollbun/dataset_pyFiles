# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd 
# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head()


train.tail()


train.shape


test.shape


train.info()


train.describe()


train.isnull().sum()


test.isnull().sum()


train['Personality'].value_counts()


sns.countplot(data=train, x='Personality')
plt.title('Personality Distribution')
plt.show()


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency']


for col in num_cols:
    median = train[col].median()
    train[col] = train[col].fillna(median)
    test[col] = test[col].fillna(median)


cat_cols = ['Stage_fear', 'Drained_after_socializing']


for col in cat_cols:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    test[col] = test[col].fillna(mode)


print(train.isnull().sum())
print(test.isnull().sum())


# Encode binary Yes/No columns
binary_map = {'Yes': 1, 'No': 0}
train['Stage_fear'] = train['Stage_fear'].map(binary_map)
train['Drained_after_socializing'] = train['Drained_after_socializing'].map(binary_map)
test['Stage_fear'] = test['Stage_fear'].map(binary_map)
test['Drained_after_socializing'] = test['Drained_after_socializing'].map(binary_map)

# Encode target column
target_map = {'Introvert': 0, 'Extrovert': 1}
train['Personality'] = train['Personality'].map(target_map)


train[['Stage_fear', 'Drained_after_socializing', 'Personality']].head()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 
            'Friends_circle_size', 'Post_frequency']


X = train[features]
y = train['Personality']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_val)


print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))


test_preds = model.predict(test[features])


submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_preds
})


submission['Personality'] = submission['Personality'].map({0: 'Introvert', 1: 'Extrovert'})


submission.to_csv('submission.csv', index=False)

