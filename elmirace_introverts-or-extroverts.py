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


# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# Show the shape of each file
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Submission shape:", submission.shape)

# Show the first few rows of the train dataset
train.head()


train.columns


# Check for missing values
train.isnull().sum()

# Check the distribution of the target
train['Personality'].value_counts(normalize=True)

# Check datatypes and summary
train.info()
train.describe()


numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']


# For train
for col in numeric_cols:
    train[col].fillna(train[col].median(), inplace=True)

for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)


from sklearn.preprocessing import LabelEncoder

# Dictionary to save encoders
encoders = {}

# Fit on train data, transform train
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = train[col].astype(str)  # convert to string for safety
    le.fit(train[col])
    train[col] = le.transform(train[col])
    encoders[col] = le  # save encoder

# Transform test data using the saved encoders
for col in categorical_cols:
    test[col] = test[col].astype(str)
    le = encoders[col]
    # Handle unseen labels by mapping unknown categories to a default value
    # For LabelEncoder, no built-in way to handle unseen, so do manual mapping:
    
    test[col] = test[col].map(lambda s: s if s in le.classes_ else 'Missing')
    
    # Now, update encoder classes_ if you added 'Missing' category
    if 'Missing' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'Missing')

    test[col] = le.transform(test[col])


numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

for col in numeric_cols:
    median = train[col].median()
    train[col].fillna(median, inplace=True)
    test[col].fillna(median, inplace=True)


# Create a feature: ratio of alone time to social attendance (avoid division by zero)
train['Alone_vs_Social_ratio'] = train['Time_spent_Alone'] / (train['Social_event_attendance'] + 1)

# Bin a continuous feature
train['Time_spent_Alone_binned'] = pd.qcut(train['Time_spent_Alone'], q=4, labels=False, duplicates='drop')


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

X = train.drop(columns=['id', 'Personality'])
le_target = LabelEncoder()
y = le_target.fit_transform(train['Personality'])


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV

xgb_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgb_cv = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_params,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

xgb_cv.fit(X, y)
print("Best XGBoost Params:", xgb_cv.best_params_)
print("Best XGBoost Score:", xgb_cv.best_score_)


from sklearn.ensemble import RandomForestClassifier

rf_base = RandomForestClassifier(random_state=42)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [8, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_cv = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_params,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

rf_cv.fit(X, y)
print("Best RF Params:", rf_cv.best_params_)
print("Best RF Score:", rf_cv.best_score_)


import lightgbm as lgb

lgb_base = lgb.LGBMClassifier(random_state=42)

lgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

lgb_cv = GridSearchCV(
    estimator=lgb_base,
    param_grid=lgb_params,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

lgb_cv.fit(X, y)
print("Best LGB Params:", lgb_cv.best_params_)
print("Best LGB Score:", lgb_cv.best_score_)


from catboost import CatBoostClassifier

cat_base = CatBoostClassifier(verbose=0, random_state=42)

cat_params = {
    'iterations': [100, 200],
    'learning_rate': [0.05, 0.1],
    'depth': [4, 6]
}

cat_cv = GridSearchCV(
    estimator=cat_base,
    param_grid=cat_params,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

cat_cv.fit(X, y)
print("Best CatBoost Params:", cat_cv.best_params_)
print("Best CatBoost Score:", cat_cv.best_score_)


from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_cv.best_estimator_),
        ('rf', rf_cv.best_estimator_),
        ('lgb', lgb_cv.best_estimator_),
        ('cat', cat_cv.best_estimator_)
    ],
    voting='soft',  # averages predicted probabilities
    n_jobs=-1
)

ensemble.fit(X, y)


X_test = test.drop(columns=['id'])
# Apply same feature engineering to X_test
X_test['Alone_vs_Social_ratio'] = X_test['Time_spent_Alone'] / (X_test['Social_event_attendance'] + 1)
X_test['Time_spent_Alone_binned'] = pd.qcut(X_test['Time_spent_Alone'], q=4, labels=False, duplicates='drop')

preds = ensemble.predict(X_test)
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': le_target.inverse_transform(preds)
})
submission.to_csv('submission.csv', index=False)

