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


train_data=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
submission_data=pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_data.head(5)


train_extra.shape



len(train_data.columns==train_extra.columns)



extra_cols=set(train_data.columns)-set(test_data.columns)
missing_cols=set(test_data.columns)-set(train_data.columns)


extra_colsintrain=set(train_data.columns)-set(train_extra.columns)
missing_colsintrain=set(train_extra.columns)-set(train_data.columns)


extra_colsintrain
missing_colsintrain


new_train_data=pd.concat([train_data,train_extra],axis=0)


new_train_data.shape


new_train_data.info()


new_train_data.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns


sns.barplot(x='Brand',y='Price',data=new_train_data)


plt.hexbin(new_train_data['Weight Capacity (kg)'], 
           new_train_data['Price'], 
           gridsize=50, cmap='Blues', alpha=0.8)

plt.xlabel('Weight Capacity (kg)')
plt.ylabel('Price')
plt.colorbar(label='Density')
plt.show()


new_train_data['Weight Capacity (kg)'].describe()


numeric_cols=['Compartments','Weight Capacity (kg)']
for i in range(len(numeric_cols)):
    q1 = new_train_data[numeric_cols[i]].quantile(0.25)  
    q2 = new_train_data[numeric_cols[i]].quantile(0.50)  
    q3 = new_train_data[numeric_cols[i]].quantile(0.75)  
    IQR=q3-q1
    new_train_data=new_train_data[(new_train_data[numeric_cols[i]] < (q3 + 1.5 * IQR)) & 
                     (new_train_data[numeric_cols[i]] > (q1 - 1.5 * IQR))]
new_train_data


from sklearn.model_selection import train_test_split
y=new_train_data.Price
ids=new_train_data.id
new_train_data.drop(columns=['Price','id'],inplace=True)
X=new_train_data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.33, random_state=42)



OrdinalCols=['Size','Laptop Compartment','Waterproof']
numericcols=set(X_train.select_dtypes(exclude='object').columns)-set(OrdinalCols)
numericcols
catagorical_cols=set(X_train.select_dtypes(include='object').columns)-set(OrdinalCols)
catagorical_cols


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
OrdinalPipe=Pipeline([
    ('imputeOrd',SimpleImputer(strategy='most_frequent')),
    ('ordinalenc',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))])

NumericPipe=Pipeline([('imputenumeric',SimpleImputer(strategy='mean'))])

CatagoricalPipe=Pipeline([('imputeCat',SimpleImputer(strategy='most_frequent')),
                 ('OHE',OneHotEncoder(handle_unknown='ignore'))])

ct = ColumnTransformer(
    [("ord",OrdinalPipe,OrdinalCols),
     ("numer",NumericPipe,list(numericcols)),
     ("catagor",CatagoricalPipe,list(catagorical_cols))])



from sklearn import set_config
ct
transformed_train=ct.fit_transform(X_train)
transformed_valid=ct.transform(X_valid)


X_train_new=pd.DataFrame(transformed_train,columns=ct.get_feature_names_out())
X_valid_new=pd.DataFrame(transformed_valid,columns=ct.get_feature_names_out())


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error




regr=XGBRegressor(n_estimators=1000,learning_rate=0.1,eval_metric="rmse",early_stopping_rounds=5)
regr.fit(X_train_new,y_train,eval_set=[(X_valid_new, y_valid)],verbose=True)


y_pred = regr.predict(X_valid_new)



test_ids=test_data.id
test_data.drop(columns=['id'],axis=1,inplace=True)
transformed_test_data=ct.transform(test_data)



test_data_new=pd.DataFrame(transformed_test_data,columns=ct.get_feature_names_out())


y_pred=regr.predict(test_data_new)


submission_data.columns


final_data=pd.DataFrame({'id':test_ids,'Price':y_pred})
final_data.to_csv('submissionbackpackk.csv',index=False)


final_data













