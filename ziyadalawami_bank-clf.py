import pandas as pd
import numpy as np


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submit = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

train.head()


train.info()


train.isnull().sum()


test.isnull().sum()


train.contact.value_counts()


# train.marital.value_counts()


print(train.shape)
print(test.shape)


train = train.replace({"unknown":np.nan})
test = test.replace({"unknown":np.nan})


train.isnull().mean() * 100


train.isna().sum()


# label encoding
edu = {
    'primary':1,
    'secondary':2,
    'tertiary':3,
}

default = {
    'yes':1,
    'no':0,
}

tele = {
    'cellular':1,
    'telephone':2,
}

month_ordinal_map = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
}


train = pd.get_dummies(train, columns=["job"], drop_first=False)
test = pd.get_dummies(test, columns=["job"], drop_first=False)

train = pd.get_dummies(train, columns=["marital"], drop_first=False)
test = pd.get_dummies(test, columns=["marital"], drop_first=False)

train['education'] = train['education'].map(edu)
test['education'] = test['education'].map(edu)

train['default'] = train['default'].map(default)
test['default'] = test['default'].map(default)

train['housing'] = train['housing'].map(default)
test['housing'] = test['housing'].map(default)

train['loan'] = train['loan'].map(default)
test['loan'] = test['loan'].map(default)

train['contact'] = train['contact'].map(tele)
test['contact'] = test['contact'].map(tele)

train['month'] = train['month'].map(month_ordinal_map)
test['month'] = test['month'].map(month_ordinal_map)


# drop cols
train.drop(columns=['poutcome', 'id'], inplace=True)
test.drop(columns=['poutcome'], inplace=True)


train.dtypes


train.columns


test.columns


X_train = train.drop(columns='y')
y_train = train['y']

X_test = test


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

model = XGBClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test.drop(columns='id'))


X_test.columns


submit.head()


# Prepare submission DataFrame
submission = pd.DataFrame({
    'id': X_test['id'],
    'y': y_pred
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")




