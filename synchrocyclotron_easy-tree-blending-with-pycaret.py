!pip install pycaret


import pycaret
pycaret.__version__


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.head()


from pycaret.regression import *
automl = setup(train, target = 'Calories')


best = compare_models(include = ['xgboost', 'lightgbm', 'catboost'], sort = 'MSE', n_select = 3)


final = blend_models(best)


y_pred = predict_model(final, data = test)


y_pred


sub = y_pred[['id', 'prediction_label']]
sub.rename(columns={'prediction_label' : 'Calories'})


sub.to_csv('submission.csv', index=False)
sub.head()

