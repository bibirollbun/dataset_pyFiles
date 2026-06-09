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


train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


print(train.shape)
print(test.shape)


train.head()


train.columns


train.nunique()


num_cols=train.select_dtypes(include=['int64','float']).columns.tolist()
cat_cols=train.select_dtypes(include=['object']).columns.tolist()
bool_cols=train.select_dtypes(include=['boolean']).columns.tolist()
print("cat_cols - ",cat_cols)
print("num_cols - ",num_cols)
print("bool_cols - ",bool_cols)


for i in cat_cols:
    print(train[i].value_counts())
    print("----")


from sklearn.model_selection import train_test_split

X_train,X_oob,y_train,y_oob=train_test_split(train.drop(columns=['accident_risk'],axis=1),train['accident_risk'],test_size=0.33,random_state=42)


print(X_train.shape)
print(X_oob.shape)
print(y_train.shape)
print(y_oob.shape)


from sklearn.preprocessing import OneHotEncoder

ohe_encoder=OneHotEncoder(sparse_output=False) # Non-zero values only 
encoded_array=ohe_encoder.fit_transform(X_train[cat_cols])
encoded_cols=ohe_encoder.get_feature_names_out(cat_cols)
encoded_df=pd.DataFrame(encoded_array,columns=encoded_cols,index=X_train.index)
encoded_df.head()


train_df=pd.concat([X_train.drop(columns=cat_cols),encoded_df],axis=1)
train_df.head()


encoded_bool=X_train[bool_cols].astype(int)
encoded_df1=pd.concat([train_df.drop(columns=bool_cols),encoded_bool],axis=1)
encoded_df1.head()


# check for cat variables
encoded_df1.select_dtypes(object).sum()


train_data=encoded_df1


print(train_data.shape,y_train.shape)


# OOB data transformations
oob_cat_cols=X_oob.select_dtypes(include='object').columns.tolist()
oob_bool_cols=X_oob.select_dtypes(include='bool').columns.fillna(0).tolist()
# Test data transformations
test_cat_cols=test.select_dtypes(include='object').columns.tolist()
test_bool_cols=test.select_dtypes(include='bool').columns.tolist()


# one hot encoding 
oob_data_ohe=ohe_encoder.transform(X_oob[oob_cat_cols])
oob_data_bool=X_oob[oob_bool_cols].astype(int)
oob_data_ohe_cols=ohe_encoder.get_feature_names_out(oob_cat_cols)
oob_data_ohe_df=pd.DataFrame(oob_data_ohe,columns=oob_data_ohe_cols,index=X_oob.index)
print(oob_data_ohe_df.shape)
print(oob_data_ohe_df.head())


test_data_ohe=ohe_encoder.transform(test[test_cat_cols])
test_data_bool=test[test_bool_cols].astype(int)
test_data_ohe_cols=ohe_encoder.get_feature_names_out(test_cat_cols)
test_data_ohe_df=pd.DataFrame(test_data_ohe,columns=test_data_ohe_cols,index=test.index)
test_data_ohe_df.head()


pd.concat([X_oob,oob_data_ohe_df],axis=1).head()


print(X_oob.shape)
pd.concat([X_oob.drop(columns=oob_cat_cols, axis=1), oob_data_ohe_df], axis=1).shape


# Boolean encoding
oob_df1=pd.concat([X_oob.drop(columns=oob_cat_cols),oob_data_ohe_df],axis=1)
oob_df=pd.concat([oob_df1.drop(columns=oob_bool_cols),oob_df1[oob_bool_cols].astype(int)],axis=1)
# oob_df.drop(columns='id',axis=1,inplace=True)


test_df1=pd.concat([test.drop(columns=test_cat_cols),test_data_ohe_df],axis=1)
test_df=pd.concat([test_df1.drop(columns=test_bool_cols),test_df1[test_bool_cols].astype(int)],axis=1)
test_df.drop(columns='id',axis=1,inplace=True)


print(oob_df.shape,oob_df1.shape,y_oob.shape,oob_data_ohe_df.shape)
print(test_data_ohe_df.shape,test.shape)


oob_df.fillna(0,inplace=True)
test_df.fillna(0,inplace=True)


train_data.shape


X_oob.shape


test_df.shape


oob_df.shape


oob_df.head()





from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


models_to_check={'LinearRegression':LinearRegression(),
                 'DecisionTreeRegressor':DecisionTreeRegressor(max_depth=10,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        random_state=42),
                 'RandomForestRegressor':RandomForestRegressor(n_estimators=100,
                        max_depth=10,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1),
                 'GradientBoostingRegressor':GradientBoostingRegressor(n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        subsample=0.8,
                        random_state=42),
                 'AdaBoostRegressor':AdaBoostRegressor(n_estimators=100,
                        learning_rate=0.1,
                        random_state=42),
                 'BaggingRegressor':BaggingRegressor(n_estimators=100,
                        random_state=42,
                        n_jobs=-1),
                 'ExtraTreesRegressor':ExtraTreesRegressor(
                         n_estimators=100,
                        max_depth=10,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                 ),
                 'XGBRegressor':XGBRegressor(
                     n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                 ),
                 'SVR':SVR(kernel='rbf',
                    C=1.0,
                    epsilon=0.1),
                 'KNeighborsRegressor':KNeighborsRegressor(n_neighbors=5,
                    weights='uniform',
                    n_jobs=-1)
                 }


oob_df.head()


for name,model in models_to_check.items() :
    model.fit(train_data,y_train)
    print("------")
    print(name)
    y_oob_pred=model.predict(oob_df)
    mse=mean_squared_error(y_oob_pred,y_oob)
    r2 = r2_score(y_oob, y_oob_pred)
    print("mse - ",mse)
    print("r2 - ",r2)




