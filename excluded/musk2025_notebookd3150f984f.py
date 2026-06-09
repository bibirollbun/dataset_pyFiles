import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
id_ = df_test.id
df_train.head()


y_train = df_train.iloc[:, -1]
df_train.drop("accident_risk", axis = 1, inplace = True)
df_train.drop("id", axis = 1, inplace = True)
df_test.drop("id", axis = 1, inplace = True)

df_train.head()


df_train.info()


df_test.info()


columns_name = df_train.columns.tolist()
object_columns = df_train.select_dtypes(include = "object").columns.tolist()
numerical_columns = df_train.select_dtypes(exclude = "object").columns.tolist()


for column in object_columns:
    print(column, " : ",df_train[column].unique())


for column in numerical_columns:
    print(column, " : ",df_train[column].describe(),"\n","\n")


from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(sparse_output= False)
df_train_transform = one_hot.fit_transform(df_train[object_columns])
encode_column = one_hot.get_feature_names_out(object_columns)
train = pd.DataFrame(df_train_transform, columns=encode_column)

df_train.drop(object_columns, axis= 1, inplace= True)
train_final = pd.concat([df_train, train], axis = 1)
train_final.index = df_train.index


df_test_transform = one_hot.fit_transform(df_test[object_columns])
test = pd.DataFrame(df_test_transform, columns=encode_column)

df_test.drop(object_columns, axis= 1, inplace= True)
test_final = pd.concat([df_test, test], axis = 1)
test_final.index = df_test.index


train_final


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,12))

sns.heatmap(train_final.corr(), annot= True)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split

x_train, x_test, Y_train, y_test = train_test_split(train_final, y_train, train_size= 0.75, random_state=42)

Logistic_model = LinearRegression()
random_model = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=1.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=42,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
)
XGB_model = XGBRFRegressor()
LGB_model = LGBMRegressor(
    boosting_type  = 'gbdt',
    num_leaves= 31,
    max_depth = -1,
    learning_rate = 0.1,
    n_estimators= 100,
    subsample_for_bin = 200000,
    min_split_gain= 0.0,
    min_child_weight= 0.001,
    min_child_samples = 20,
    subsample = 1.0,
    subsample_freq = 0,
    colsample_bytree = 1.0,
    reg_alpha = 0.0,
    reg_lambda = 0.0,
    importance_type = 'split',
) 


models = [Logistic_model, random_model, XGB_model,LGB_model]
for model in models :
    model.fit(x_train, Y_train)
    y_pred = model.predict(x_train)
    print(model, mean_squared_error(Y_train, y_pred))
    test_pred = model.predict(x_test)
    print(model, mean_squared_error(y_test, test_pred))


test_predict = models[3].predict(test_final)


submission = pd.DataFrame({"id" : id_, "accident_risk": test_predict})
submission.to_csv("submission.csv", index = False)




