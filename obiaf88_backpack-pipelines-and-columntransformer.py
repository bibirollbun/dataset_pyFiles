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


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion	 									
from sklearn.preprocessing import OneHotEncoder , StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score,RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
from sklearn import linear_model
warnings.filterwarnings("ignore")


train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')



train.drop('id', axis = 1 , inplace = True)
train_extra.drop('id', axis = 1 , inplace = True)


train = pd.concat([train,train_extra], axis = 0)


pd.DataFrame((train.isnull().sum()/train.shape[0]), index = train.columns, columns = ['Percentage missing'])


pd.DataFrame((test.isnull().sum()/test.shape[0]), index = test.columns, columns = ['Percentage missing'])


categorical_columns = train.select_dtypes(include = ['object']).columns
numerical_columns = [col for  col in train.select_dtypes(include = ['number']).columns if col not in ('Price') ]


numerical_columns


categorical_columns


train.head()


X = train[[col for col in train.columns if col not in ('Price')]]
y = train['Price']


X.shape, y.shape


cat_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
        ('onehot', OneHotEncoder(sparse_output = False,handle_unknown = "ignore"))
    ]
     )



cat_preprocessor = ColumnTransformer(transformers = 
                                     [('cat', cat_transformer, categorical_columns)],
                                     remainder = 'drop')


cat_preprocessor


num_transformer = ColumnTransformer([('comp_imputer', SimpleImputer(strategy = 'most_frequent'),['Compartments']),
                                    ('weight_imputer', SimpleImputer(strategy = 'mean'),['Weight Capacity (kg)'])], remainder = 'drop')


num_preprocessor = Pipeline([('num_transformer', num_transformer), ('scaler', StandardScaler())])


num_preprocessor


preprocessing = ColumnTransformer([
        ('cat', cat_preprocessor, categorical_columns),
        ('num',num_preprocessor, numerical_columns )
])


preprocessing


model = Pipeline([('preprocessing', preprocessing),('regr', xgb.XGBRegressor( eval_metric='rmsle'))])


model


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


X_train.shape, X_test.shape, y_train.shape,y_test.shape


param_grid = {'regr__max_depth':[6,10],
             "regr__n_estimators": [250,300]
             }


random_grid_search = RandomizedSearchCV(model, param_grid, cv = 3, n_jobs=2, verbose=1, random_state=0, n_iter=5, scoring = 'neg_root_mean_squared_error').fit(X_train, y_train)

print(f"Best parameters are: {random_grid_search.best_params_}" )


print(random_grid_search.score(X_test, y_test))


random_grid_search.best_estimator_


model_regr = Pipeline([('preprocessing', preprocessing),('rfr', RandomForestRegressor( n_estimators=100
                                                                                     ))])


model_regr.fit(X_train, y_train)


test_prediction = test[[col for col in test.columns if col not in ('id')]]


test_prediction.head(2)


pred_xgb = random_grid_search.best_estimator_.predict(test_prediction)

# Create the submission DataFrame
submission_xgb = pd.DataFrame({
    'id': test['id'],         
    'Price': pred_xgb      
})


submission_xgb.head(4)


pred_rf = model_regr.predict(test_prediction)

# Create the submission DataFrame
submission_rf = pd.DataFrame({
    'id': test['id'],         
    'Price': pred_rf      
})



submission_rf.head()


submission = submission_xgb.merge(submission_rf, on = 'id')


submission['Price'] = (submission['Price_x'] +  submission['Price_y'])/2
submission.drop(['Price_x','Price_y'], axis = 1 , inplace = True)


submission.head(2)


# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)


