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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# read the train data
df=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')


df.head()


df.info()


df.describe().T


#dropping "id" column as this is not a feature
df1=df.drop(['ID'],axis=1)


df1.columns


target = df1['efs']


target.head()


X=df1.drop(['efs','efs_time'], axis=1)
X.head()


X = pd.get_dummies(X,drop_first = False)
# Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in X.columns}
new_n_list = list(new_names.values())
# [LightGBM] Feature appears more than one time.
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
X = X.rename(columns=new_names)
X.head()


X.info()


X.columns


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, auc
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


#create train/test split
X_train,X_test,y_train,y_test = train_test_split(X,target, test_size=0.10, random_state=101)


#create parameters grid for LGBM. add parameter choices to pick the best using gridsearch cross-validation
param_grid = {
    "objective": ["binary"],
    "boosting_type":["gbdt"],
    'metric':['auc'],
    "random_state": [42],
    'learning_rate': [0.003],
    'n_estimators': [1000],
    'num_leaves':[31],
    'max_depth': [20],
    'verbosity':[-1],
    'error_score':["raise"]
     }
lgbm_model = LGBMClassifier()
grid = GridSearchCV(lgbm_model,param_grid,cv=5)
grid.fit(X_train,y_train)
grid_pred = grid.predict(X_test)
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
grid.best_params_


test_df=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
test_df1=test_df.drop(['ID'], axis=1)
test_df1 = pd.get_dummies(test_df1,drop_first = False)
# Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in test_df1.columns}
new_n_list = list(new_names.values())
# [LightGBM] Feature appears more than one time.
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
test_df1 = test_df1.rename(columns=new_names)
test_df1.info()


# ensure all the columns in the feature dataframe used for training are present in the test dataset
for col in X.columns:
    if col in test_df1.columns:
        pass
    else:
        test_df1[col] = np.nan


test_df1.head()


# preparing the submission files
submission = pd.DataFrame()
submission['ID'] = test_df['ID']

submission['prediction'] = grid.predict_proba(test_df1)[:,1]
file_name = 'submission.csv'
submission.to_csv(file_name, index=False)
submission

