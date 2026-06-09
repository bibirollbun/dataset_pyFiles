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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.model_selection import KFold,cross_val_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


train=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
X=train.drop(['id','BeatsPerMinute'],axis=1)
y=train['BeatsPerMinute']
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X)
param_grid = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.001, 0.01, 0.1]
}


sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
grid_search = GridSearchCV(
    estimator=sgd,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',  # RMSE scoring
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y)

print("Best Params:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_) 


best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled,y)


sub=pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
test=test.drop('id',axis=1)
scaler=MinMaxScaler()
test=scaler.fit_transform(test)

y_pred=best_model.predict(test)
sub['BeatsPerMinute']=y_pred


sub.to_csv("submission.csv", index=False)

sub.head()




