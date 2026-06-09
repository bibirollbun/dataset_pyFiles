# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost as xgb


train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')


train.info()


train.head()


text_cols = train.select_dtypes(include=['object']).columns.tolist()
text_cols


train['combined_text'] = (
    "comment: " + train['body'] + 
    " [SEP] rule: " + train['rule'] + 
    " [SEP] positive examples: " + train['positive_example_1'] + ", " + train['positive_example_2'] + 
    " [SEP] negative examples: " + train['negative_example_1'] + ", " + train['negative_example_2']
)
train.head()


X_train = train['combined_text']
y_train = train['rule_violation']


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.33, random_state=42
)


print("Defining Pipeline -----")

xgb_pipeline = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=100000)),
    ('xgb', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        n_estimators=4000,
        learning_rate=0.01,
        max_depth=6,
        subsample = 0.8,
        random_state=42
    ))
])

xgb_pipeline


xgb_pipeline.fit(X_train, y_train)

print("Pipeline training complete.")


test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')


test['combined_text'] = (
    "comment: " + test['body'] + 
    " [SEP] rule: " + test['rule'] + 
    " [SEP] positive examples: " + test['positive_example_1'] + ", " + test['positive_example_2'] + 
    " [SEP] negative examples: " + test['negative_example_1'] + ", " + test['negative_example_2']
)


X_test = test['combined_text']


y_test_pred_proba = xgb_pipeline.predict_proba(X_test)[:, 1]


def submit(y_test_pred_proba):
    subm_df = pd.DataFrame(
        {
            'row_id': test['row_id'],
            'rule_violation': y_test_pred_proba
        }
    )

    subm_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully!")


submit(y_test_pred_proba)

