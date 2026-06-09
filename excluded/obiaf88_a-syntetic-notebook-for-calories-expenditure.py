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


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures,StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_log_error


#importing data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.head()


cat_columns = train.select_dtypes('object').columns
num_columns = [col for col in train.select_dtypes('number').columns if col not in ('id','Calories')]


print(f"Numerical columns : {num_columns} \n Categorical columns : {cat_columns}")


num_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('scaler',StandardScaler())
    
])


preprocessing = ColumnTransformer([
    ('cat_preprocessig',OneHotEncoder(),cat_columns),
    ('num_preprocessing', num_pipeline,num_columns)
])


preprocessing


# we need to take the log of target variable because eval metric has log 
X = train[[col for col in train.columns if col not in ('id','Calories')]]
y = np.log(train['Calories'])


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)


# the output of preprocessing.fit_transform is an array but you can transform it in a dataframe
X_train_preprocessed = pd.DataFrame(preprocessing.fit_transform(X_train), columns = preprocessing.get_feature_names_out())


X_train_preprocessed.head()


xgb_regressor=xgb.XGBRegressor(eval_metric='rmsle')


param_grid = {"max_depth":    [3,5,10],
              "n_estimators": [500,700,1000],
              "learning_rate":  [0.1, 0.015]}


best_xgb_regressor  = GridSearchCV(xgb_regressor, param_grid, cv=5).fit(X_train_preprocessed, y_train).best_estimator_
y_test_pred = np.clip(best_xgb_regressor.predict(preprocessing.fit_transform(X_test)), a_min=0,a_max = None)


print(f"Root Mean Squared Log Error: {np.sqrt(mean_squared_log_error(y_test,y_test_pred))}")


submission = pd.DataFrame({
'id': test['id'],
'Calories': np.exp(np.clip(best_xgb_regressor.predict(preprocessing.fit_transform(test[[col for col in test.columns if col not in 'id']])), a_min = 0, a_max = None))
})


submission.head(2)


submission.to_csv('submission.csv', index=False)
print("Submission created")

