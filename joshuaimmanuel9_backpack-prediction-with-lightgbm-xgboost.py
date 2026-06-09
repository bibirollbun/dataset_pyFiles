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


import pandas as pd


df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


df_train.head()


df_test.head()


df_extra.head()


df = pd.concat([df_train, df_extra], ignore_index=True)


df_extra.shape


df.head()


df.info()


df.describe()


df['Size'].unique()


df.describe(include='object')


df.drop(columns=['id'],inplace=True)


df.isnull().sum(axis = 0)


# df.dropna(inplace=True)


for column in ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']:
    mode_value = df[column].mode()[0] 
    df[column] = df[column].fillna(mode_value)

for column in ['Compartments', 'Weight Capacity (kg)', 'Price']:
    mean_value = df[column].mean()
    df[column] = df[column].fillna(mean_value)


df.isna().sum(axis=0)


from sklearn.preprocessing import LabelEncoder

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
df['Size_encoded'] = df['Size'].map(size_mapping)
le = LabelEncoder()
df['Material_encoded'] = le.fit_transform(df['Material'])
one_hot_columns = ['Color', 'Brand', 'Style', 'Laptop Compartment', 'Waterproof']
df_encoded= pd.get_dummies(df, columns=one_hot_columns, drop_first=False)
df_final = pd.concat([df, df_encoded], axis=1)


size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
df_test['Size_encoded'] = df_test['Size'].map(size_mapping)
le = LabelEncoder()
df_test['Material_encoded'] = le.fit_transform(df_test['Material'])
one_hot_columns = ['Color', 'Brand', 'Style', 'Laptop Compartment', 'Waterproof']
df_test_encoded= pd.get_dummies(df_test, columns=one_hot_columns, drop_first=False)
df_test_final = pd.concat([df_test, df_test_encoded], axis=1)



df_final = df_final.loc[:, ~df_final.columns.duplicated()]


df_test_final = df_test_final.loc[:, ~df_test_final.columns.duplicated()]


df_test=df_test_final


df= df_final


df.columns


df_test.columns


from category_encoders import TargetEncoder

encoder = TargetEncoder()

categorical_columns = ['Brand', 'Material', 'Size', 'Style', 'Color']

df_train_encoded =df.copy()
df_train_encoded[categorical_columns] = encoder.fit_transform(df[categorical_columns],df['Price'])
df_test_encoded = df_test.copy()
df_test_encoded[categorical_columns] = encoder.transform(df_test[categorical_columns])

print("Encoded Test Data:\n", df_test_encoded)


df_train_encoded


df_train_encoded['Laptop Compartment'] = df_train_encoded['Laptop Compartment'].map({'Yes': 1, 'No': 0})
df_test_encoded['Laptop Compartment'] = df_test_encoded['Laptop Compartment'].map({'Yes': 1, 'No': 0})


df_train_encoded['Waterproof'] = df_train_encoded['Waterproof'].map({'Yes': 1, 'No': 0})
df_test_encoded['Waterproof'] = df_test_encoded['Waterproof'].map({'Yes': 1, 'No': 0})


df_test_encoded.drop(columns=['id'],inplace=True)


df_test_encoded.head()


y = df_train_encoded['Price']
df_train_encoded.drop(columns=['Price'],inplace=True)
X=df_train_encoded


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_train_encoded = pd.DataFrame(scaler.fit_transform(df_train_encoded), columns=df_train_encoded.columns)
df_test_encoded = pd.DataFrame(scaler.transform(df_test_encoded), columns=df_test_encoded.columns)


from sklearn.model_selection import train_test_split


X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)


import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


params = {
    'objective': 'reg:squarederror',   
    'tree_method': 'gpu_hist',        
    'eval_metric': 'rmse',           
    'learning_rate': 0.1,             
    'max_depth': 10,                   
    'min_child_weight': 1,            
    'subsample': 0.8,                 
    'colsample_bytree': 0.8,          
    'lambda': 1,                     
    'alpha': 0                       
}



xgb_model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])


y_pred = xgb_model.predict(dtest)



rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))



X_pred_final=df_test_encoded
X_pred_final


dtest_fin = xgb.DMatrix(X_pred_final)
fin_pred = xgb_model.predict(dtest_fin)


fin_pred.mean()


fin_pred.max()


fin_pred.min()


result_df = pd.DataFrame({
    'id': df_test['id'],
    'Price': fin_pred
})

result_df.to_csv('predictions_xgb.csv', index=False)





!pip install lightgbm 


import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)


params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'device_type': 'gpu',
    'min_data_in_leaf': 20,  
    'max_depth': 5,          
    'learning_rate': 0.1,    
    'num_leaves': 31,        
    'feature_fraction': 0.9, 
}

gbm = lgb.train(params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, test_data],
                valid_names=['train', 'eval'])


fin_pred_lgbm = gbm.predict(X_pred_final)


fin_pred_lgbm.max()


fin_pred_lgbm.min()


fin_pred


result_df = pd.DataFrame({
    'id': df_test['id'],
    'Price': fin_pred
})

result_df.to_csv('predictions_lgbm.csv', index=False)

