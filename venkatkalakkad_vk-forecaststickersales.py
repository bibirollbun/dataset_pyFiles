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


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
 
from sklearn.metrics import mean_absolute_percentage_error



sales_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")


sales_df.columns


sales_df.head()


for col in sales_df.columns:
    print (sales_df[col].unique())


sales_df.isna().sum()


sales_df.dropna(inplace=True)


sales_df.isna().sum()


sns.histplot(data=sales_df[sales_df["product"] == 'Holographic Goose'], x="num_sold", kde='true')
plt.show()


sales_df['date'] = pd.to_datetime(sales_df['date'])
sales_df['new_date'] = sales_df['date']
sales_df["quarter"] = sales_df["new_date"].dt.quarter.astype('object')
sales_df['month'] = sales_df['new_date'].dt.month.astype('object')
sales_df['day'] = sales_df['new_date'].dt.day.astype('object')
sales_df.set_index('date', inplace=True)

 
train_data = sales_df[:'2015-12-31'] 
test_data = sales_df['2016-01-01':]

X_train = train_data.drop('num_sold', axis=1)
y_train = train_data['num_sold']

X_test = test_data.drop('num_sold', axis=1)
y_test = test_data['num_sold']


train_data.shape, test_data.shape


X_train = X_train.drop("id", axis=1)
X_test= X_test.drop("id",axis =1 )


cat_col = []
for col in train_data.columns :
    if train_data[col].dtype == 'O':
        cat_col.append(col)
cat_col


#Let us use out of the box catboost for start. Now then we can see how hyperparameter tuning can help
# why? I do not want to be bothered by categoricals!!

reg = cb.CatBoostRegressor(verbose=0)
reg.fit(X_train,y_train,cat_features=cat_col)


predictions = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
 
r2 = r2_score( predictions,y_test)
r2,mse,rmse


 

mape = mean_absolute_percentage_error(y_test,predictions,)
print(mape) 


 
reg = cb.CatBoostRegressor(loss_function="MAE",iterations=220,learning_rate=0.11,depth=16, l2_leaf_reg=10)


reg.fit(X_train,y_train,cat_features=cat_col)


predictions = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
 
r2 = r2_score( predictions,y_test)
r2,mse,rmse,mae


 

mape = mean_absolute_percentage_error(y_test, predictions)
print(mape, "with hyperparameter tuning") 



X_train.head()


#Now let us submit with this model itself. 

sub_df  = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sub_df1 = sub_df.drop("id",axis=1)
sub_df1['date'] = pd.to_datetime(sub_df1['date'])
sub_df1['new_date'] = sub_df1['date']
sub_df1["quarter"] = sub_df1["new_date"].dt.quarter.astype('object')
sub_df1['month'] = sub_df1['new_date'].dt.month.astype('object')
sub_df1['day'] = sub_df1['new_date'].dt.day.astype('object')
sub_df1['new_date'] = sub_df1['date']


sub_df1.set_index('date', inplace=True)

sub_df1.head()



new_pred = reg.predict(sub_df1)
new_pred


submission = pd.DataFrame({
    'id': sub_df['id'],
    'num_sold': new_pred
})


submission.to_csv('submission.csv', index=False)

