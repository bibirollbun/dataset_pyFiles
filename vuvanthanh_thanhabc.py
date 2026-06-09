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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import catboost
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import FunctionTransformer, Normalizer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator

import eli5
from eli5.catboost import explain_weights


import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/flight-delays-spring-2018/flight_delays_train.csv')
test = pd.read_csv('/kaggle/input/flight-delays-spring-2018/flight_delays_test.csv')

print(train.shape)
train.head()


train.dep_delayed_15min.replace({'N':0, 'Y':1}, inplace=True)
train.rename(columns={'dep_delayed_15min':'y'}, inplace=True)
train.y.value_counts(normalize=True)


#make additional features
class MakeCols(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        pass

    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['DepHour'] = pd.cut(X.DepTime.astype(int), 
                              bins=[t for t in range(0,2700,100)], 
                              labels=[str(h) for h in range(1,27)])
        X['Route'] = X['Origin'] + "_" + X['Dest']
        X['Carrier_Origin'] = X['UniqueCarrier'] + "_" + X['Origin']
        X['Carrier_Dest'] = X['UniqueCarrier'] + "_" + X['Dest']
        X['Carrier_DepHour'] = X['UniqueCarrier'] + '_' + X['DepHour'].astype('str')
        X['Route_DepHour'] = X['Route'] +  '_' + X['DepHour'].astype('str')
        
        return X


train_ = MakeCols().fit_transform(train)
test_ = MakeCols().fit_transform(test)

train_.head()


X_train = train_.drop(['y'], axis=1).copy()
y_train = train_['y'].copy()
X_test = test_.copy()


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, 
                                            train_size=0.8, shuffle=True, 
                                            stratify=y_train, 
                                            random_state=33)

cat_features = ['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest','DepHour', 'Route', 'Carrier_Origin', 'Carrier_Dest','Carrier_DepHour', 'Route_DepHour']


cb_grid = CatBoostClassifier(eval_metric='AUC', 
                             cat_features=cat_features, 
                             early_stopping_rounds=40, 
                             learning_rate=0.1, 
                             depth=5, 
                             l2_leaf_reg=5, 
                             rsm=1.0)
cb_grid.fit(X_tr, y_tr, eval_set=(X_val, y_val), 
            cat_features=cat_features, 
            use_best_model=True, 
            verbose=True)


!pip install shap


X = train_.drop(['y'], axis=1).copy()
print(X.shape)
X.head()


print(cb_grid.feature_names_)


import shap

explainer = shap.TreeExplainer(cb_grid)
shap_values = explainer.shap_values(X)


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :])
# Lệnh này giải thích tại sao mô hình đưa ra dự đoán 
# cho một mẫu cụ thể bằng cách phân rã đóng góp của từng feature.


shap.summary_plot(shap_values, X_train )
# Trong toàn bộ tập X_train, feature nào ảnh hưởng nhiều nhất đến 
# dự đoán của mô hình, và ảnh hưởng theo chiều hướng nào?


#eli5
import eli5
from eli5.sklearn import PermutationImportance

my_set = PermutationImportance( cb_grid , random_state=34).fit(X_val,y_val)
eli5.show_weights(my_set, feature_names = X_val.columns.tolist())

