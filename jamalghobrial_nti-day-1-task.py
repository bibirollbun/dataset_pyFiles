!pip -q install "pycaret==3.3.2" "scikit-learn==1.4.2"


# pip install --upgrade scikit-learn


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import *
import pycaret
from pycaret.regression import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/terrain-prices-reggression/train.csv")
df_test = pd.read_csv("/kaggle/input/terrain-prices-reggression/test.csv")


df_train


df_train.info()


df_train.drop(labels = "id", axis = 1, inplace = True)


df_train["location_type"].unique()


le = LabelEncoder()
df_train['location_type'] = le.fit_transform(df_train['location_type'])
df_train['zoning_code'] = le.fit_transform(df_train['zoning_code'])
df_train['land_use'] = le.fit_transform(df_train['land_use'])
df_test['location_type'] = le.fit_transform(df_test['location_type'])
df_test['zoning_code'] = le.fit_transform(df_test['zoning_code'])
df_test['land_use'] = le.fit_transform(df_test['land_use'])


df_test.head()


cor = df_train.corr()
sns.heatmap(cor)


df_train.head()


session = pycaret.regression.setup(data = df_train, target = -1, train_size = 0.8, categorical_features = ["location_type", "zoning_code","land_use"],
                                ignore_features = None, remove_multicollinearity= False,
                                   pca = False, pca_method = 'linear',
                                   feature_selection= False, feature_selection_method= 'classic',
                                   n_features_to_select = 0.2,use_gpu = True)


# best = pycaret.regression.compare_models()





et = create_model('et')
bagged_dt = ensemble_model(et, method = 'Bagging')


df_test.drop("id", axis = 1, inplace = True)


y_preds = predict_model(bagged_dt, data = df_test)


y_preds


submissions = pd.read_csv("/kaggle/input/terrain-prices-reggression/sample_submission.csv")
submissions["target"] = y_preds["prediction_label"]
submissions


submissions.to_csv("submission.csv", index = False)

