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


!pip install xgbtune


%load_ext cudf.pandas


import pandas as pd       
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgbtune import tune_xgb_model

import warnings
warnings.filterwarnings('ignore')

seed = 42


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

train = pd.concat([train, train_extra], axis=0, ignore_index=True)

train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


cat_cols=train.select_dtypes(include='object').columns.tolist()
TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

features = test.columns.tolist()

for col in features:
    TE.fit(train[col], train['Price'])
    train[col] = TE.transform(train[col])
    test[col] = TE.transform(test[col])


# Feature engineering 
def feature_engineering(df):
    na_columns = ['Material', 'Style', 'Brand', 'Size', 'Waterproof', 'Color', 'Laptop Compartment']
    nan_flags = {col: 'NaN' for col in na_columns}
    df.fillna(nan_flags, inplace=True)
    for col in na_columns:
        df[f'_NaN_{col.replace(" ", "_")}'] = (df[col] == 'NaN').astype(int)

    df['weight/compartment'] = df['Weight Capacity (kg)'] / df['Compartments']
    df['_7_NaNs'] = df[[f'_NaN_{col.replace(" ", "_")}' for col in na_columns]].sum(axis=1)

    for cat in cat_cols:
        df[f'{cat}_wc'] = df[cat] / 1000 + df['Weight Capacity (kg)']
        df[f'{cat}_cmp'] = df[cat] / 100 + df['Compartments']
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)



X_train = train.drop('Price', axis = 1)
y_train = train['Price']


params = {'eval_metric': 'rmsle', 'tree_method': 'hist', 'device': 'cuda'}

params, round_count = tune_xgb_model(params, X_train, y_train)


dtrain = xgb.DMatrix(X_train, label=y_train)
final_model = xgb.train(params, dtrain, num_boost_round=round_count)

dtest = xgb.DMatrix(test)
y_pred = final_model.predict(dtest)


xgb.plot_importance(final_model, importance_type='gain', max_num_features=10)
plt.title('Feature Importance (Top 10)')
plt.show()



sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub = pd.DataFrame({"id":sub.id, "Price":y_pred})
sub.to_csv('submission.csv', index=False)


sub.head()

