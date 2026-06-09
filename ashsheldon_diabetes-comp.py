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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

print('Setup complete')


X_full=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv',index_col='id')

y=X_full.diagnosed_diabetes
X_full=X_full.drop('diagnosed_diabetes',axis=1)

X_test_full=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
X_test_id=X_test_full.id
X_test = X_test_full.drop('id',axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_full,y, test_size=0.2, random_state=1, stratify=y)
X_train.columns.isna().sum()
print('Class Imbalance: ', y_train.value_counts(normalize=True)*100)
X_full.head()


cols=X_train.columns
obj_cols = [col for col in X_train.columns if X_train[col].dtype=='object']
num_cols = list(set(cols)-set(obj_cols))
print('The numerical cols are :',num_cols)
print('The categorical cols are :',obj_cols)



num_trans = Pipeline(steps=[('num_trans',SimpleImputer(strategy='mean')),
                            ('scaler', StandardScaler())])                            

cat_trans = Pipeline(steps=[('cat_imp', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])

pre_pro = ColumnTransformer(transformers=[('num', num_trans, num_cols), 
                                       ('cat', cat_trans, obj_cols)])



model=LogisticRegression(penalty='l2', C=10, solver='lbfgs', max_iter=1000, n_jobs=-1, random_state=1)
my_pipe = Pipeline(steps=[('preprocess', pre_pro),
                          ('model', model)])



my_pipe.fit(X_train, y_train)
val_preds = my_pipe.predict_proba(X_val)[:,1]


auc_roc=roc_auc_score(y_val, val_preds)
print("Validation ROC AUC: ",auc_roc*100)


my_pipe.fit(X_full, y)

fin_preds = my_pipe.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
  "id": X_test_id,
  "diagnosed_diabetes": fin_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()

