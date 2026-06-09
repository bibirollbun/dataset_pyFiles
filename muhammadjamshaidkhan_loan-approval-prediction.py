# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()



# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

playground_series_s4e10_path = kagglehub.competition_download('playground-series-s4e10')

print('Data source import complete.')



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


print(playground_series_s4e10_path,'Train')


Train = pd.read_csv(os.path.join(playground_series_s4e10_path, 'train.csv'))
Test =  pd.read_csv(os.path.join(playground_series_s4e10_path, 'test.csv'))
sample =  pd.read_csv(os.path.join(playground_series_s4e10_path, 'sample_submission.csv'))



Train.info()


Test.info()


combine = pd.concat([Train,Test], axis = 0 , ignore_index=True)


combine.info()


df = pd.get_dummies(combine, columns= ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], drop_first=True )


df.head()


df.info()


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
param = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]

}

RS =  RandomizedSearchCV(
    estimator= XGBClassifier(),
    param_distributions= param,
    n_iter= 30,
    cv= 5,
    n_jobs= -1
)


# Drop rows with missing values in the target variable
df_cleaned = df.dropna(subset=['loan_status'])

feature =  df_cleaned.drop(columns= ['loan_status'])


target =  df_cleaned['loan_status']


from sklearn.model_selection import train_test_split

RANDOM_STATE = 55 # Define RANDOM_STATE
X_train,X_val,y_train,y_val = train_test_split(feature, target, train_size=0.8, random_state= RANDOM_STATE)


print(f'Train:{len(X_train)}')
print(f'Train:{len(y_val)}')


Random_search = RS.fit(X_train,y_train)


Random_search


print(f'best parameter:{Random_search.best_params_}')


print(f'best parameter:{Random_search.best_score_}')


from xgboost import XGBClassifier

model = XGBClassifier(
    subsample=0.8,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    colsample_bytree=1.0,
    random_state=55
)

model.fit(X_train,y_train)

pred = model.predict(X_val)




from sklearn.metrics import accuracy_score, classification_report

print(f'accuracy:{accuracy_score(y_val, pred)}')
print(f'accuracy:{classification_report(y_val, pred)}')


X_test = df[df['loan_status'].isna()].drop(columns=['loan_status'])


predictions = model.predict(X_test)


submission_df = pd.DataFrame({'id': Test['id'], 'loan_status': predictions})


submission_df.to_csv('submission.csv', index=False)




