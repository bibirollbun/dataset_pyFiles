import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", index_col = 'id')


train.head()


train.columns


train.isna().sum()
#so there are no missings


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


cat_features = [col for col in train.columns if train[col].dtype == 'object']
print(cat_features)


X = train.drop('diagnosed_diabetes', axis=1)
y = train['diagnosed_diabetes']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)


y_train


model = CatBoostClassifier(
    iterations=1000,
    cat_features=cat_features,
    verbose=100
)


model.fit(X_train, y_train, eval_set=(X_val, y_val))


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


from catboost import Pool
test_pool = Pool(test, cat_features=cat_features)  # Use your cat_features list
test_preds = model.predict_proba(test_pool)[:, 1]


submit = pd.DataFrame({'id': test['id'],
                       'diagnosed_diabetes': test_preds})

submit.to_csv('submission.csv', index=False)
print("Submission saved successfully!")

