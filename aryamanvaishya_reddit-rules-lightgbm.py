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


# Load the training data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

# Basic info
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Preview the data
train_df.head()


train_df['rule_violation'].value_counts(normalize=True)


print("Unique training rules:", train_df['rule'].nunique())
print("Unique test rules:", test_df['rule'].nunique())


train_rules = set(train_df['rule'].unique())
test_rules = set(test_df['rule'].unique())

print("Unseen test rules:", len(test_rules - train_rules))


train_df['body_len'] = train_df['body'].apply(lambda x: len(str(x).split()))
train_df['body_len'].describe()


rule_violation_by_rule = train_df.groupby('rule')['rule_violation'].mean().sort_values()
rule_violation_by_rule.plot(kind='barh', figsize=(10, 10), title='Violation rate per rule')


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine body + rule text as input
train_df['text'] = train_df['body'] + " [SEP] " + train_df['rule']

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['rule_violation'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
val_preds = lr.predict_proba(X_val_tfidf)[:, 1]
print("Logistic Regression AUC:", roc_auc_score(y_val, val_preds))


# from sklearn.linear_model import Ridge
# from sklearn.metrics import roc_auc_score

# ridge = Ridge(alpha=1.0)
# ridge.fit(X_train_tfidf, y_train)
# val_preds_ridge = ridge.predict(X_val_tfidf)

# print("Ridge Regression AUC:", roc_auc_score(y_val, val_preds_ridge))


import lightgbm as lgb

# Define datasets
lgb_train = lgb.Dataset(X_train_tfidf, label=y_train)
lgb_val = lgb.Dataset(X_val_tfidf, label=y_val)

# Params
params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt'
}

# Use callbacks for early stopping
callbacks = [lgb.early_stopping(stopping_rounds=10)]

# Train
lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_val],
    valid_names=["val"],
    callbacks=callbacks
)


val_preds_lgb = lgb_model.predict(X_val_tfidf, num_iteration=lgb_model.best_iteration)
from sklearn.metrics import roc_auc_score
print("LightGBM AUC:", roc_auc_score(y_val, val_preds_lgb))


# Combine body and rule just like training
test_df['input'] = test_df['body'] + ' [SEP] ' + test_df['rule']
X_test_tfidf = vectorizer.transform(test_df['input'])

# Predict
test_preds = lgb_model.predict(X_test_tfidf, num_iteration=lgb_model.best_iteration)

# Prepare submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_preds
})

submission.to_csv('submission.csv', index=False)




