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


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df


import seaborn as sns
import matplotlib.pyplot as plt


print(df.shape)
print(df.describe())
print(df.info())


print(df['num_reported_accidents'].max())
print(df.columns)


for col in df.columns:
    if(df[col].nunique() >10) :
        print(col, ":" , df[col].nunique())
    else:
        print(col, ":", df[col].unique())


numeric_columns = ['curvature', 'accident_risk']
categoric_columns = ['road_type', 'num_lanes', 'lighting', 'weather', 'road_signs_present', 'public_road','num_reported_accidents', 'time_of_day', 'holiday', 'school_season']


df.isnull().sum()


sns.boxplot(df["curvature"])


sns.displot(df['curvature'])


sns.heatmap(pd.crosstab(df['time_of_day'], df['weather']), annot=True, cmap='coolwarm')


# fig, ax = plt.subplots(len(categoric_columns), 1)
# for i,col in enumerate(categoric_columns):
#     sns.countplot(df[col], ax=ax[i])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


order = [
    ['rural', 'urban', 'highway'],
    ['daylight', 'dim', 'night'],
    ['morning', 'afternoon', 'evening']
]


from sklearn.preprocessing import FunctionTransformer


logtrf = FunctionTransformer(func=np.log1p)
trfs = ColumnTransformer(
    transformers = [
        ('trf1', OrdinalEncoder(categories=order), ['road_type', 'lighting', 'time_of_day']),
        ('trf2', OneHotEncoder(), ['weather']),
        ('trf3', FunctionTransformer(func=np.log1p), ['curvature'])
    ],
    remainder='passthrough'
)


from sklearn.model_selection import train_test_split


X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
X.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


X_train = trfs.fit_transform(X_train)


X_test = trfs.fit_transform(X_test)


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)


trfs


from sklearn.metrics import mean_squared_error
y_pred = dtr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(y_test.std())
rmse


from xgboost import XGBRegressor
xgb = XGBRegressor(
    max_depth=12,
    learning_rate=0.01,
    n_estimators=500,
    random_state=666,
    eval_metric= "rmse"
)

xgb.fit(X_train, y_train,)
y_pred = xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("ytest std = ", y_test.std())
print("rmse = ", rmse)


from sklearn.pipeline import Pipeline, make_pipeline


pipe1 = Pipeline([('trf1', trfs), ('XGB', xgb)])
pipe1


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

trf2 = ColumnTransformer(
    transformers=[
        ('scaler', scaler, [2,3])
    ],
    remainder='passthrough'
)
pipe2 = Pipeline([('trf1', trfs), ('scaler', trf2), ('XGB', xgb)])
pipe2


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# pipe1.fit(X_train, y_train)
# y_pred = pipe1.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = mse ** 0.5
# print("ytest std = ", y_test.std())
# print("rmse = ", rmse)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipe2.fit(X_train, y_train)
y_pred = pipe2.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("ytest std = ", y_test.std())
print("rmse = ", rmse)


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
r2


r2_train = r2_score(y_train, pipe2.predict(X_train))
r2_test = r2_score(y_test, pipe2.predict(X_test))
print(r2_train, r2_test)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_df


predictions = pipe2.predict(test_df)


predictions_df = pd.DataFrame(predictions)
predictions_df['id'] = test_df['id']
predictions_df.columns = ['accident_risk', 'id']
predictions_df = predictions_df[['id', 'accident_risk']]
predictions_df


predictions_df.to_csv("submission.csv", index=False)




