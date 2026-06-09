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


pip install pandas-summary


#Core Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import random
import warnings
from scipy import stats

#Visualization Libraries 

import matplotlib.pyplot as plt
import seaborn as sns

#machine Learning Libraries 

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder ,FunctionTransformer
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import make_scorer,accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier ,HistGradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,IsolationForest
from sklearn.compose import ColumnTransformer


df_train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')



df_train



df_sub



df_train.info()


df_train.isnull().sum()


pip install ydata-profiling


from ydata_profiling import ProfileReport
profile = ProfileReport(df_train)
profile.to_notebook_iframe()



# 3.1 Select the categorical and numerical columns


df_train.info()


categorical = df_train.select_dtypes(include = ['object']).columns


categorical


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ['Geography', 'Gender']:
    df_train[col] = le.fit_transform(df_train[col])


df_train['Geography']


df_train['Gender']





df_train = df_train.drop(columns =['Surname','id'])


df_train


X = df_train.drop(columns=['Exited'])
y = df_train['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)




