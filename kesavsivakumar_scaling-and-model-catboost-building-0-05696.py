# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn import preprocessing


df_train = pd.read_csv(r"/kaggle/input/playground-series-s5e5/train.csv")


df_train.info()


df_train.describe()


df_train['Sex'] = df_train['Sex'].replace('male',0)
df_train['Sex'] = df_train['Sex'].replace('female',1)



# Increase the chunk size to avoid rendering issues with complex plots
plt.rcParams['agg.path.chunksize'] = 10000

# Optionally adjust the path simplification threshold for performance
plt.rcParams['path.simplify_threshold'] = 0.2

#plt.plot(df_train['Age'].values, df_train['Calories'], color='red')


df_train['Sex'].value_counts()


df_train


Y = df_train['Calories'].values


df_train.drop(['id','Calories'],axis =1,inplace = True)



#Normalizing Input features
names = df_train.columns
scaler = preprocessing.RobustScaler()
scaled_df = scaler.fit_transform(df_train)

#X = df_train.values


scaled_df ,scaled_df.shape


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


X_train,X_val,Y_train,Y_val = train_test_split(scaled_df,Y,test_size =0.2)


## changing target to logarthmic 

Y_train_log = np.log1p(Y_train)
Y_val_log = np.log1p(Y_val)



X_train.shape


model=CatBoostRegressor(iterations=10000, depth=10, learning_rate=0.01, loss_function='RMSE')


model.fit(X_train,Y_train_log ,plot=True,use_best_model=True,eval_set = (X_val,Y_val_log))


df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


id_ = df_test['id'].values
df_test.drop(['id'],axis =1 ,inplace =True)



df_test['Sex'] = df_test['Sex'].replace('male',0)
df_test['Sex'] = df_test['Sex'].replace('female',1)


df_test


X_test = scaler.transform(df_test)
#X_test = df_test.values


X_test


y_pred_log = model.predict(X_test)


y_pred = np.expm1(y_pred_log)


y_pred_log,y_pred


y_pred.min() ,y_pred.max()


fin_dict = {'id':id_,'Calories':y_pred}
find_df = pd.DataFrame(fin_dict)


find_df.to_csv('output.csv',index=False)




