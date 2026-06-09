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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score


train = pd.read_csv(r'/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e8/sample_submission.csv')


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


num_columns = [col for col in train.select_dtypes(np.number).columns if col not in ['y','id']]
cat_columns = [col for col in train.select_dtypes('object').columns if col not in ['y','id']]


col_transformer = ColumnTransformer(
    remainder = 'passthrough',
    transformers = [('num',StandardScaler(), num_columns),
                   ('cat', OneHotEncoder(),cat_columns)]
)


pipe = Pipeline([
    ('col_transformer', col_transformer),
    ('logre',LogisticRegression())
])


param_grid = {
    'logre__C': [0.01, 0.1, 1, 10, 100],   
    'logre__penalty': ['l2','elasticnet'],              
    'logre__max_iter': [100,500,1000]
}


cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)


grid_search = GridSearchCV(
    estimator = pipe,
    param_grid = param_grid,
    cv = cv,
    scoring = 'roc_auc'
)


X = train[[col for col in train.columns if col not in ('id','y')]]
y = train['y']


grid_search.fit(X,y)


pd.DataFrame(grid_search.cv_results_)


print(f"Best params {grid_search.best_params_}")
print(f"ROE entire dataset: {grid_search.score(X,y)}")
print(f"Best_model : {grid_search.best_estimator_}")
print(f"Best score: {grid_search.best_score_}")


y_pred = grid_search.predict_proba(test[[col for col in test.columns if col not in ('id')]])[:,1]


submission = pd.DataFrame({'id':test['id'], 'y':y_pred})


submission.to_csv('submission.csv', index = False)


submission

