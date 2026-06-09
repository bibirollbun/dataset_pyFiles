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
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.naive_bayes import GaussianNB


train = pd.read_csv(r'/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e8/sample_submission.csv')


num_columns = [col for col in train.select_dtypes(np.number).columns if col not in ['y','id']]
cat_columns = [col for col in train.select_dtypes('object').columns if col not in ['y','id']]


X = train[[col for col in train.columns if col not in ('id','y')]]
y = train['y']


col_transformer = ColumnTransformer(
    remainder = 'passthrough',
    transformers = [('num',StandardScaler(), num_columns),
                   ('cat', OneHotEncoder(),cat_columns)]
)


cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)


pipe_lr = Pipeline([
    ('col_transformer', col_transformer),
    ('logre',LogisticRegression())
])


pipe_rfc = Pipeline([
    ('col_transformer', col_transformer),
    ('rfc',RandomForestClassifier())
])


pipe_gauss = Pipeline([
    ('col_transformer', col_transformer),
    ('gau',GaussianNB())
])


estimators = [
    ('lr_stack',pipe_lr),
    ('rf_stack',pipe_rfc)
    #('gauss_stack',pipe_gauss)
]


stack_cls = StackingClassifier(
    estimators = estimators,
    final_estimator = XGBClassifier(use_label_encoder = False,
                                    n_jobs=-1,
                                   eval_metric = 'auc'),
    cv = cv
)


param_grid_stack = {
    'lr_stack__logre__C': [0.01, 0.1, 1, 10],   
    'lr_stack__logre__max_iter': [100,500,1000],
    'rf_stack__rfc__n_estimators': [200,500,1000],
    'rf_stack__rfc__max_depth': [3,5,7],
    'final_estimator__n_estimators': [100, 200, 500],
    'final_estimator__min_samples_split': [2,4,5],
    'final_estimator__min_samples_leaf': [2,4,5],
    'final_estimator__max_leaf_nodes': [4,15,50]
}


grid_stack = RandomizedSearchCV(
    estimator = stack_cls,
    param_distributions = param_grid_stack,
    scoring = 'roc_auc',
    verbose = 4,
    n_iter = 3
)


grid_stack.fit(X,y)


pd.DataFrame(grid_stack.cv_results_)


print(f"Best params {grid_stack.best_params_}")
print(f"Best_model : {grid_stack.best_estimator_}")
print(f"Best score: {grid_stack.best_score_}")


y_pred = grid_stack.best_estimator_.predict_proba(test[[col for col in test.columns if col not in ('id')]])[:,1]


submission = pd.DataFrame({'id':test['id'], 'y':y_pred})


submission.to_csv('submission.csv', index = False)


submission

