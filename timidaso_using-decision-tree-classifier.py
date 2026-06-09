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


import gzip
import shutil

input_path = '/kaggle/input/avazu-ctr-prediction/train.gz'
output_path = '/kaggle/working/train.csv'

with gzip.open(input_path, 'rb') as f_in:
    with open(output_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Extraction complete! File saved as:", output_path)



n_rows = 300000

# Load 300,000 rows from the compressed train.gz file
df = pd.read_csv("/kaggle/working/train.csv", nrows=n_rows)

# Preview the data
df.head()


Y = df['click'].values


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model'], axis=1).values
# X = df.drop(columns=["id", "click"])


enc = OneHotEncoder(handle_unknown= 'ignore')



# from sklearn.compose import make_column_selector
# categorical_selector = make_column_selector(dtype_include='object')
# categorical_columns = categorical_selector(X)

# print("Categorical Columns:", categorical_columns)


# 80/20 split for training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



enc_X_train =  enc.fit_transform(X_train)
print(enc_X_train[0])


enc_X_test = enc.transform(X_test)


decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)
parameters = {'max_depth': [3, 10, None]}


grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')


grid_search.fit(X_train, Y_train)



print(grid_search.best_params_)


decision_tree_best = grid_search.best_estimator_
prob = decision_tree_best.predict_proba(X_test)[:, 1]


from sklearn.metrics import roc_auc_score, log_loss
auc = roc_auc_score(Y_test, prob)
loss = log_loss(Y_test, prob)

print(f"AUC Score: {auc:.4f}")
print(f"Log Loss: {loss:.4f}")

