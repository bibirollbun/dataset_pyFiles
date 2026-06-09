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


data=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')
data.head()
xtrain=data.drop(['target'],axis=1)
ytrain=data['target']



datat=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')
datat.head()
xtest=datat.drop(['id'],axis=1)


from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Base models with optimized parameters
models = [
    ('et', ExtraTreesRegressor(n_estimators=300, max_depth=30, min_samples_split=5, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, objective='reg:squarederror'))
]

# Final estimator with tuned parameters
final_estimator = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)

# Stacking model
stacked_model = StackingRegressor(estimators=models, final_estimator=final_estimator, cv=5, n_jobs=-1)

# Fit the stacking model
stacked_model.fit(xtrain, ytrain)

# Predict and evaluate
y_pred = stacked_model.predict(xtest)




output = pd.DataFrame({
    'id': datat.id,
    'target': y_pred
})



d=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/sample_submission.csv')
ytest=d.drop(['id'],axis=1)


from sklearn.metrics import r2_score
r2_rf = r2_score(ytest, y_pred)
r2_rf


output.to_csv('submission.csv',index=False)



