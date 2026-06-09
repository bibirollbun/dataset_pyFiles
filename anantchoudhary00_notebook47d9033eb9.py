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


TRAIN_PATH = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH  = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"


import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import optuna













traind = pd.read_csv(TRAIN_PATH)
testd = pd.read_csv(TEST_PATH)

numeric_features_for_outliers = traind.select_dtypes(include=np.number).columns.drop('CORRUCYSTIC_DENSITY')

for column in numeric_features_for_outliers:
    Q1 = traind[column].quantile(0.25)
    Q3 = traind[column].quantile(0.75)
    IQR = Q3 - Q1
    lbound = Q1 - 1.5 * IQR
    ubound = Q3 + 1.5 * IQR

    # Use .clip() to cap the outliers in the column
    traind[column] = traind[column].clip(lower=lbound, upper=ubound)
print("Outlier handling complete.")





features = traind.drop('CORRUCYSTIC_DENSITY', axis=1)
target = traind['CORRUCYSTIC_DENSITY']


numericfeatures = features.select_dtypes(include=np.number).columns.tolist()
categoricalfeatures = features.select_dtypes(include='object').columns.tolist()



numerictransformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categoricaltransformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerictransformer, numericfeatures),
        ('cat', categoricaltransformer, categoricalfeatures)
    ],
    remainder='passthrough'
)





def objective(trial):
  
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    
  
    score = cross_val_score(pipeline, features, target, cv=3, scoring='neg_root_mean_squared_error')
    
    return np.mean(score)








target.dropna(inplace=True)
features = features.loc[target.index]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) 

bestparams = study.best_params
print(f" Best hyperparameters found: {bestparams}")


finalmodel = xgb.XGBRegressor(**bestparams, random_state=42, n_jobs=-1, objective='reg:squarederror')

finalpipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', finalmodel)])



finalpipeline.fit(features, target)

predictions = finalpipeline.predict(testd)




submission = pd.DataFrame({
    'LOCAL_IDENTIFIER': testd['LOCAL_IDENTIFIER'],
    'CORRUCYSTIC_DENSITY': predictions
})

submission.to_csv('submission.csv', index=False)



pd.read_csv('submission.csv').head()




