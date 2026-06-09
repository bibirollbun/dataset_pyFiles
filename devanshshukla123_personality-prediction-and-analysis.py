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


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train



import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
# from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=1500, learning_rate=0.01, depth=5)

binary_cols = ['Stage_fear', 'Drained_after_socializing']
train['Alone_Friends_ratio'] = train['Time_spent_Alone'] / (train['Friends_circle_size'] + 1)
test['Alone_Friends_ratio'] = test['Time_spent_Alone'] / (test['Friends_circle_size'] + 1)

train['Social_minus_Alone'] = train['Social_event_attendance'] - train['Time_spent_Alone']
test['Social_minus_Alone'] = test['Social_event_attendance'] - test['Time_spent_Alone']

# Binning
train['Alone_bin'] = pd.qcut(train['Time_spent_Alone'], q=4, labels=False)
test['Alone_bin'] = pd.qcut(test['Time_spent_Alone'], q=4, labels=False)

# Frequency encoding for binary columns
for col in binary_cols:
    freq = train[col].value_counts() / len(train)
    train[col+'_freq'] = train[col].map(freq)
    test[col+'_freq'] = test[col].map(freq)
for col in binary_cols:
    train[col] = train[col].fillna('MISSING')
    test[col] = test[col].fillna('MISSING')


encoders = {}
for col in binary_cols:
    le = LabelEncoder()
   
    all_values = pd.concat([train[col], test[col]]).unique()
    le.fit(all_values)
    encoders[col] = le
    
    
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

num_imputer = SimpleImputer(strategy='median')
train[numeric_cols] = num_imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = num_imputer.transform(test[numeric_cols])

y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})
X = train.drop(['id', 'Personality'], axis=1)

# model = RandomForestClassifier()
#     max_depth=10, 
#     min_samples_split=10, 
#     n_estimators=100,
#     random_state=42)
model.fit(X, y)


if 'Personality' in test.columns:
    X_test = test.drop(['id', 'Personality'], axis=1)
else:
    X_test = test.drop(['id'], axis=1)

test_preds = model.predict(X_test)
test['Personality_pred'] = test_preds
test['Personality_pred'] = test['Personality_pred'].map({0: 'Introvert', 1: 'Extrovert'})

submission = test[['id', 'Personality_pred']].rename(columns={'Personality_pred': 'Personality'})
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Submission file created successfully!")
print("\nSample of predictions:")
print(submission.head())

train_preds = model.predict(X)
print(f"\nModel accuracy on training data: {accuracy_score(y, train_preds):.6f}")












final_model = CatBoostClassifier(iterations=2000,learning_rate=0.01,loss_function = 'MultiClass',depth=8,cat_features=['Drained_after_socializing','Stage_fear'],verbose=100,random_seed=42)
final_model.fit(X, y)
preds = final_model.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y, train_preds))


import matplotlib.pyplot as plt
from xgboost import plot_importance

plot_importance(model)
plt.show()


