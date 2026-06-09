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


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

train = train.fillna(0)
test = test.fillna(0)

target = train['loan_paid_back']
features = train.drop(['loan_paid_back', 'id'], axis = 1)
upd_test = test.drop('id', axis = 1)

encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
cat_cols = features.select_dtypes(include='object').columns.tolist()
encoder.fit(pd.concat([features[cat_cols], upd_test[cat_cols]], axis = 0))

encoded_train = encoder.transform(features[cat_cols])
encoded_test = encoder.transform(upd_test[cat_cols])

encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_cols), index=features.index)
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols), index=upd_test.index)

features_final = pd.concat([features.drop(cat_cols, axis=1).reset_index(drop=True), encoded_train_df.reset_index(drop=True)], axis=1)
upd_test_final = pd.concat([upd_test.drop(cat_cols, axis=1).reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)

X_train, X_val, y_train, y_val = train_test_split(features_final, target, test_size=0.2, random_state=42)

model = model = lgb.LGBMClassifier(n_estimators=1800, learning_rate=0.03, max_depth=8, num_leaves= 120)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:,1]
print('Validation ROC AUC:', roc_auc_score(y_val, y_pred))

test_preds = model.predict_proba(upd_test_final)[:,1]



submission = pd.DataFrame({
    'id': test['id'],  
    'loan_paid_back': test_preds  
})


submission.to_csv('submission.csv', index=False)

