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


test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_df.head()


test_df.info()


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train_df.head()


train_df.corr()


train_df.info()


train_df.columns


import seaborn as sns

# each boxplot i saw the data is not normalize, it has so many outliers
sns.boxplot(train_df['age'])


# # normalizing age column 

# q1 = train_df['age'].quantile(.25)
# q3 = train_df['age'].quantile(.75)

# iqr = q3 - q1

# lb_age = q1 - 1.5 * iqr
# ub_age = q3 + 1.5 * iqr
# print(lb_age, ub_age)

# train_df = train_df[(train_df['age']>=lb_age) & (train_df['age']<=ub_age)]


sns.boxplot(train_df['age'])


sns.histplot(train_df['age'], bins=30, kde=True)


# # normalizing age column - Test DF

# q1 = test_df['age'].quantile(.25)
# q3 = test_df['age'].quantile(.75)

# iqr = q3 - q1

# lb_age = q1 - 1.5 * iqr
# ub_age = q3 + 1.5 * iqr
# print(lb_age, ub_age)

# test_df = test_df[(test_df['age']>=lb_age) & (test_df['age']<=ub_age)]


total_null = train_df.isnull().sum()
total_null


# drop colns
cleaned_train_df = train_df[['id', 'age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan','y']]
cleaned_test_df = test_df[['id', 'age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan']]


cleaned_train_df.info()


cleaned_train_df = pd.get_dummies(cleaned_train_df, columns=['job','marital','education','default','housing','loan'])
cleaned_test_df = pd.get_dummies(cleaned_test_df, columns=['job','marital','education','default','housing','loan'])


# feature selection
X = cleaned_train_df.drop(columns=['y'])
y = cleaned_train_df['y']


X


cleaned_train_df.head()


# train_val_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# model = LogisticRegression()

# model = RandomForestClassifier(n_estimators=400, max_depth=5, bootstrap=True)

model = GradientBoostingClassifier(n_estimators=400, max_depth=5)


model.fit(X_train, y_train)


# y_preds = model.predict(X_val)

y_preds = model.predict_proba(X_val)[:, 1]


from sklearn.metrics import accuracy_score, roc_auc_score

# print(accuracy_score(y_val, y_preds))

print(roc_auc_score(y_val, y_preds))


y_test_preds = model.predict_proba(cleaned_test_df)[:,1]


submission = pd.DataFrame({
    'id': test_df['id'],
    'y': y_test_preds
})

# 2. Save the submission file to the Kaggle output path
submission.to_csv('/kaggle/working/submission.csv', index=False)


submissions_df = pd.read_csv('/kaggle/working/submission.csv')
submissions_df

