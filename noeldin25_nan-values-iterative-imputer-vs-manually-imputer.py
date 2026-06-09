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


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score,train_test_split 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test.head()


train_IterativeImputer = train
test_IterativeImputer = test


train.info()


train.drop('id', axis=1).describe()


for col in train.columns:
    if train[col].dtype == 'object':
        print(f'{col}: {train[col].unique()}')



fig, axes = plt.subplots(1, 2, figsize=(16, 6))

msno.matrix(train, ax=axes[0])
axes[0].set_title('Train Data Missing Values')

msno.matrix(test, ax=axes[1])
axes[1].set_title('Test Data Missing Values')

plt.show()


null_counts = train.isnull().sum()
null_percent = train.isnull().mean() * 100

missing_train = pd.DataFrame({
    'Columns': null_counts.index,
    'Counts': null_counts.values,
    'Percentage': null_percent.values
})

missing_train


null_counts = test.isnull().sum()
null_percent = test.isnull().mean() * 100

missing_test = pd.DataFrame({
    'Columns': null_counts.index,
    'Counts': null_counts.values,
    'Percentage': null_percent.values
})

missing_test


plt.figure(figsize=(8, 6))
corr = train.drop('id', axis=1).select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f')
plt.tick_params('x', rotation=45)
plt.show()



train = train.fillna({
    'Stage_fear': 'No_stage',
    'Drained_after_socializing': 'No_social_events'
})

train[['Stage_fear', 'Drained_after_socializing']].isna().sum()


test = test.fillna({
    'Stage_fear': 'No_stage',
    'Drained_after_socializing': 'No_social_events'
})

test[['Stage_fear', 'Drained_after_socializing']].isna().sum()


def impute_Nan_based_on_col(df, imputed_col, related_col):
    nan_before = df[[imputed_col]].isna().sum()
    
    df['bins'] = pd.qcut(
        df[related_col],
        q=[0, 0.20, 0.40, 0.60, 0.80, 1.00],
        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    )

    df[imputed_col] = df[imputed_col]\
        .fillna(df.groupby('bins')[imputed_col].transform('median'))
    
    df = df.drop('bins', axis=1)
    
    nan_after = df[[imputed_col]].isna().sum()

    print("Nans Before:", nan_before)
    print("Nans After:", nan_after)

    return df


print('Train:')
train = impute_Nan_based_on_col(train,'Time_spent_Alone', 'Social_event_attendance')

print('\nTest:')
test = impute_Nan_based_on_col(test, 'Time_spent_Alone', 'Social_event_attendance')


print('Train:')
train = impute_Nan_based_on_col(train, 'Time_spent_Alone', 'Going_outside')

print('\nTest:')
test = impute_Nan_based_on_col(test, 'Time_spent_Alone', 'Going_outside')


print('Train:')
train = impute_Nan_based_on_col(train,'Social_event_attendance', 'Going_outside')

print('\nTest:')
test = impute_Nan_based_on_col(test, 'Social_event_attendance', 'Going_outside')


print('Train:')
train = impute_Nan_based_on_col(train,'Social_event_attendance', 'Friends_circle_size')

print('\nTest:')
test = impute_Nan_based_on_col(test, 'Social_event_attendance', 'Friends_circle_size')


print('Train:')
train = impute_Nan_based_on_col(train,'Social_event_attendance', 'Post_frequency')

print('\nTest:')
test = impute_Nan_based_on_col(test, 'Social_event_attendance', 'Post_frequency')


print('Train:')
train = impute_Nan_based_on_col(train,'Going_outside', 'Post_frequency')

print('\nTest:')
test = impute_Nan_based_on_col(test, 'Going_outside', 'Post_frequency')


print('Train:')
train = impute_Nan_based_on_col(train,'Going_outside', 'Friends_circle_size')

print('\nTest:')
test = impute_Nan_based_on_col(test, 'Going_outside', 'Friends_circle_size')


print('Train:')
train = impute_Nan_based_on_col(train,'Friends_circle_size', 'Time_spent_Alone')

print('\nTest:')
test = impute_Nan_based_on_col(test, 'Friends_circle_size', 'Time_spent_Alone')


print('Train:')
train = impute_Nan_based_on_col(train,'Post_frequency', 'Going_outside')

print('\nTest:')
test = impute_Nan_based_on_col(test, 'Post_frequency', 'Going_outside')


null_counts = train.isnull().sum()
null_percent = train.isnull().mean() * 100

missing_train = pd.DataFrame({
    'Columns': null_counts.index,
    'Counts': null_counts.values,
    'Percentage': null_percent.values
})

missing_train


null_counts = test.isnull().sum()
null_percent = test.isnull().mean() * 100

missing_test = pd.DataFrame({
    'Columns': null_counts.index,
    'Counts': null_counts.values,
    'Percentage': null_percent.values
})

missing_test


le_target = LabelEncoder()
train['Personality'] = le_target.fit_transform(train['Personality'])


for col in ['Stage_fear', 'Drained_after_socializing']:
    le = LabelEncoder()
    train[f"{col}_encoder"] = le.fit_transform(train[col])
    test[f"{col}_encoder"] = le.transform(test[col]) 



train = train.drop(['Stage_fear', 'Drained_after_socializing'], axis=1)
test = test.drop(['Stage_fear', 'Drained_after_socializing'], axis=1)


train.info()


X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']


xgb = XGBClassifier(
    max_depth=4,         
    learning_rate=0.01,   
    n_estimators=1000,    
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0 
)

cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=0
)

lgbm = LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=0
)


voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lgbm', lgbm)
    ],
    voting='soft' 
)


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(voting_clf, X, y, cv=cv, scoring='accuracy')

print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Std Dev: {scores.std():.4f}")


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency']

imp = IterativeImputer(random_state=42)
train_IterativeImputer[features] = imp.fit_transform(train_IterativeImputer[features])
test_IterativeImputer[features] = imp.transform(test_IterativeImputer[features])



train_IterativeImputer = train_IterativeImputer.fillna({
    'Stage_fear': 'No_stage',
    'Drained_after_socializing': 'No_social_events'
})

train_IterativeImputer[['Stage_fear', 'Drained_after_socializing']].isna().sum()


test_IterativeImputer = test_IterativeImputer.fillna({
    'Stage_fear': 'No_stage',
    'Drained_after_socializing': 'No_social_events'
})

test_IterativeImputer[['Stage_fear', 'Drained_after_socializing']].isna().sum()


null_counts = train_IterativeImputer.isnull().sum()
null_percent = train_IterativeImputer.isnull().mean() * 100

missing_train_IterativeImputer = pd.DataFrame({
    'Columns': null_counts.index,
    'Counts': null_counts.values,
    'Percentage': null_percent.values
})

missing_train_IterativeImputer


le_target = LabelEncoder()
train_IterativeImputer['Personality'] = le_target.fit_transform(train_IterativeImputer['Personality'])


for col in ['Stage_fear', 'Drained_after_socializing']:
    le = LabelEncoder()
    train_IterativeImputer[f"{col}_encoder"] = le.fit_transform(train_IterativeImputer[col])
    test_IterativeImputer[f"{col}_encoder"] = le.transform(test_IterativeImputer[col]) 



train_IterativeImputer = train_IterativeImputer.drop(['Stage_fear', 'Drained_after_socializing'], axis=1)
test_IterativeImputer = test_IterativeImputer.drop(['Stage_fear', 'Drained_after_socializing'], axis=1)


X = train_IterativeImputer.drop(['id', 'Personality'], axis=1)
y = train_IterativeImputer['Personality']


xgb = XGBClassifier(
    max_depth=4,         
    learning_rate=0.01,   
    n_estimators=1000,    
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0 
)

cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=0
)

lgbm = LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=0
)


voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lgbm', lgbm)
    ],
    voting='soft' 
)


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(voting_clf, X, y, cv=cv, scoring='accuracy')

print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Std Dev: {scores.std():.4f}")




