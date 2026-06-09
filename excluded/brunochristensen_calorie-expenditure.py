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


from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, Normalizer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate


#Let's explore some data!
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col = "id")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col = "id")


#750k entries
train.shape


#Numerical features with with one catagorical feature
train.head()


num_cols = ['Age','Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
cat_cols = ['Sex']


#No missing values, so no need for imputation.
train.info(verbose=True)


train.describe()


train[num_cols].hist(bins=50, figsize = (10,10))


X = train.drop(columns=['Calories'])
y = train['Calories']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#I'm curious if I can throw together a quick XGBoost pipeline and score a relatively low RMSLE
preprocessor = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), cat_cols),
    ('normalizer', Normalizer(), ['Body_Temp']),
    ('scaler', StandardScaler(), num_cols),
])

xgb = XGBRegressor(n_estimators=2000, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb)
])


pipeline.fit(X_train,y_train)


y_pred = pipeline.predict(X_test)


def RMSLE(y_true: list, y_pred: list) -> float:
    n = len(y_true)
    msle = np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
    return msle


RMSLE(y_test, y_pred)


X, X_test= train[num_cols + cat_cols], test[num_cols + cat_cols]
y = train["Calories"]


pipeline.fit(X, y)
preds = pipeline.predict(X_test)

for i in range(len(preds)):
    preds[i] = abs(preds[i])

sub = pd.DataFrame()
sub["id"] = test.index
sub["Calories"] = preds
sub.to_csv("/kaggle/working/submission.csv", index = False)

