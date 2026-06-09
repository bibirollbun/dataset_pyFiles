# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install lunax


from lunax.data_processing import *
df_train = load_data('./train.csv') # or df = load_data('train.parquet')
target = 'Fertilizer Name'
df_train = preprocess_data(df_train,target) # data pre-processing, including missing value handling, feature encoding, feature scaling


encoded_values, unique_values = pd.factorize(df_train['Fertilizer Name'])
df_train['Fertilizer Name'] = encoded_values


X_train, X_val, y_train, y_val = split_data(df_train, target)


df_test = load_data('./test.csv')
df_test = preprocess_data(df_test)


from lunax.models import xgb_clf # or xgb_reg, lgbm_reg, lgbm_clf, cat_clf, cat_reg
from lunax.hyper_opt import OptunaTuner
tuner = OptunaTuner(n_trials=10,model_class="XGBClassifier") # Hyperparameter optimizer, n_trials is the number of optimization times
# or "XGBRegressor", "LGBMRegressor", "LGBMClassifier" , "CatClassifier", "CatRegressor"
results = tuner.optimize(X_train, y_train, X_val, y_val)
best_params = results['best_params']
model = xgb_clf(best_params)
model.fit(X_train, y_train)


model.evaluate(X_val, y_val)


df_sub=load_data('./sample_submission.csv')


y_test = model.predict(df_test)
df_sub[target] = unique_values[y_test]


df_sub.to_csv('submission.csv',index=False)

