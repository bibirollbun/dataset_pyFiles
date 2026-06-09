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


data_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
data_train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


data_train = pd.concat([data_train, data_train_extra])
len(data_train)


def missing_filler(data, col_cat):
    for col in col_cat:
        data[col] = data[col].fillna('none')
    return data


col_cat = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color']


data_train = missing_filler(data_train, col_cat)


data_train = pd.get_dummies(data_train, col_cat)
data_train.head()


pip install autogluon


from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator


predictor = TabularPredictor(label="Price", eval_metric='rmse').fit(
    train_data=data_train,
    presets="best_quality",  # Ensures strong models
#    num_bag_folds=5,          # Enables ensembling via bagging
    feature_generator=IdentityFeatureGenerator(),  # No transformations
#    time_limit=3600,
    auto_stack=False         # No automatic deep stacking (like ANN but with ML models)
)


leaderboard = predictor.leaderboard(data=None)  # Shows performance on validation data
print(leaderboard)


data_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


data_test = missing_filler(data_test, col_cat)


data_test = pd.get_dummies(data_test, col_cat)
data_test.head()


predictions = predictor.predict(data_test)


df_res = pd.DataFrame({'id':data_test['id'], 'Price':predictions})
df_res.to_csv('/kaggle/working/submission.csv', index=False)
df_res

