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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve


train_data = pd.read_csv(os.path.join(dirname, 'train.csv'))
test_data = pd.read_csv(os.path.join(dirname, 'test.csv'))
train, test = train_data.copy(), test_data.copy()


train_df.columns


class dropfeatureselector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        X_dropped = X.drop(self.variables, axis = 1)
        return X_dropped
    
class simpleimputercustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables, strategy):
        self.variables = variables
        self.strategy = strategy
        self.imp = SimpleImputer(missing_values=np.nan, strategy=self.strategy)
    def fit(self, X, y = None):
        X_ = X.loc[:,self.variables]
        self.imp.fit(X_)
        return self
    def transform(self, X):
        X_ = X.loc[:,self.variables]
        X_transformed = pd.DataFrame(self.imp.transform(X_), columns= self.variables)
        X.drop(self.variables, axis= 1, inplace=True)
        X[self.variables] = X_transformed[self.variables].values
        return X

class featureselector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X.loc[:,self.variables]

class temparature(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x_ = x.copy()
        x_["average_temparature"] = round((x_["maxtemp"] + x_["mintemp"] + x_["temparature"])/3, 1)
        return x_.drop(columns=["maxtemp", "mintemp", "temparature"], axis=1)


class suppress(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x_ = x.copy()
        for var in self.variables:
            Q1 = x_[var].quantile(0.25)
            Q3 = x_[var].quantile(0.75)
            IQR = Q3 - Q1


            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            x_.loc[x_[var] < lower_bound, var] = lower_bound
            x_.loc[x_[var] > upper_bound, var] = upper_bound
        return x_


class seasonofdate(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        winter = [12, 1, 2]
        spring = [3, 4, 5]
        summer = [6, 7, 8]
        x_ = x.copy()
        x_["month"] = x_["day"].apply(lambda x: datetime.strptime(str(x), "%j").month)
        x_["season"] = x_["month"].apply(lambda x: 1 if x in winter \
                                        else (2 if x in spring\
                                        else(3 if x in summer else 4)))
        
        return x_
    

class seasonofmonth(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, x, y=None):
        return self

class OneHotEncodercustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.ohe = OneHotEncoder(drop='first', handle_unknown = 'ignore')
    def fit(self, X, y = None):
        X_ = X.loc[:,self.variables]
        self.ohe.fit(X_)
        return self
    def transform(self, X):
        X_ = X.loc[:,self.variables]
        X_transformed =  pd.DataFrame(self.ohe.transform(X_).toarray(), columns= self.ohe.get_feature_names_out())
        X.drop(self.variables, axis= 1, inplace=True)
        X[self.ohe.get_feature_names_out()] = X_transformed[self.ohe.get_feature_names_out()].values
        return X


x = train.drop('rainfall', axis=1)
y = train['rainfall']
x.columns


num_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']


drop_features = dropfeatureselector(variables=['id'])
num_features = featureselector(variables=num_cols)
median_imputer = simpleimputercustom(variables=num_cols, strategy='median')
suppress_ = suppress(variables=num_cols)
temparature_ = temparature()
scaler = StandardScaler()



num_preprocess = Pipeline(
    steps=[('drop_features',drop_features),
           ('num_features_select', num_features),
           ('imputer', median_imputer),
           ('suppres_features', suppress_),
           ('temp', temparature_)])


cat_cols = ['day']


cat_features = featureselector(variables=cat_cols)
season = seasonofdate()
freq_imputer = simpleimputercustom(variables=cat_cols, strategy='most_frequent')


cat_preprocess = Pipeline(
    steps=[('cat_feature_select', cat_features),
           ('imputer_cat', freq_imputer),
           ('season', season)])


class ModelSwitcher(BaseEstimator):
    
    def __init__(self, estimator = LogisticRegression()):
        self.estimator = estimator
    
    def fit(self, x, y=None, **kwargs):
        self.estimator.fit(x, y)
        return self
    
    def predict(self, x, y=None):
        return self.estimator.predict(x)
    
    def predict_proba(self, x):
        return self.estimator.predict_proba(x)
    
    def score(self, x):
        return self.estimator.score(x, y)


combined_preprocessing = FeatureUnion([
    ('numericals', num_preprocess),
    ('categoricals', cat_preprocess),
])


complete_pipeline = Pipeline([
        ('preprocessing', combined_preprocessing),
        ('StandardScaler', scaler),
        ('Model Training', ModelSwitcher())
    ])
display(complete_pipeline)


X_train, X_valid, y_train, y_valid = train_test_split(x, y, 
    test_size=0.2, random_state=42)


complete_pipeline.fit(X_train, y_train)
pred = complete_pipeline.predict_proba(X_valid)
roc_auc_score(y_valid, pred[:, 1])


from sklearn.svm import SVC
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier


param_range = [0.1, 1, 10]
param_range_fl = [1.0, 0.5]
knn_range = list(range(1, 31))
grid_params_lr = [{'model__penalty': ['l1', 'l2'],
                   'model__C': np.logspace(-4,4,50),
                   'model__solver': ['liblinear', 'saga']}]
grid_params_svm = [{'model__kernel': ['linear', 'rbf'],
                    'model__C': param_range,
                    'model__gamma': [1, 0.1, 0.01, 0.001, 0.0001]}]
grid_params_lgbm = [{'model__num_leaves': [5, 20, 31],
                    'model__learning_rate': [0.05, 0.1, 0.2],
                    'model__n_estimators': [50, 100, 150]}]

grid_params_cat = [{'model__iterations': [100, 200],
                    'model__learning_rate': [0.01, 0.1],
                    'model__depth': [3, 6]}]
jobs = -1

#models = {'lr': {'model': LogisticRegression(),
#                 'params': grid_params_lr},
#          'scv': {'model': SVC(probability=True),
#                  'params': grid_params_svm},
#          'cat': {'model': CatBoostClassifier(),
#                  'params': grid_params_cat},
#          'lgbm': {'model': lgb.LGBMClassifier(),
#                   'params': grid_params_lgbm}}

models = {'lr': {'model': LogisticRegression(),
                 'params': grid_params_lr}}


best_acc = 0.0
best_model = ''

for model in models.keys():
    print(model)
    complete_pipeline = Pipeline([
        ('preprocessing', combined_preprocessing),
        ('StandardScaler', scaler),
        ('model', models[model]['model'])])
    gcv = GridSearchCV(complete_pipeline, models[model]['params'], scoring='roc_auc', cv=10)
    gcv.fit(X_train, y_train)
    pred = gcv.best_estimator_.predict_proba(X_valid)
    acc = roc_auc_score(y_valid, pred[:, 1])
    print(acc)
    if acc > best_acc:
        best_model = gcv
        best_acc = acc
best_model


best_model.best_params_


best_acc



test


test.isna().sum()


predict = best_model.predict(test)


submission = pd.DataFrame()
submission['id'] = test['id']
submission['rainfall'] = predict
submission


submission.to_csv('submission.csv', index=False)




