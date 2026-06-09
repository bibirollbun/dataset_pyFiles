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


test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')



train.head(10)


train.info()


train.describe()


df_processed = train.copy()
df_processed


df_processed = df_processed.drop('id',axis = 1)


bool_cols = df_processed.select_dtypes(include='bool').columns


bool_cols


for col in bool_cols:
    df_processed[col]= df_processed[col].astype('int')
    test[col]=test[col].astype('int')


test.dtypes


df_processed.dtypes



df_processed.head(5
                 )


import seaborn as sns
import matplotlib.pyplot as plt


object_list = df_processed.select_dtypes(include='object')


for col in object_list:
    print(df_processed[col].unique(),col)


df_processed['weather'].value_counts()


df_processed.groupby(['weather'])[['accident_risk','num_reported_accidents']].agg(mean_value=('accident_risk','mean'),sum_value=('num_reported_accidents','sum'))





plot_list =['num_lanes','curvature']


sns.boxplot(df_processed[plot_list])


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


for col in object_list:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    test[col]= le.fit_transform(test[col])
    # For LightGBM, we explicitly cast to 'category' for better performance
    test[col]=test[col].astype('category')
    df_processed[col] = df_processed[col].astype('category')


test.head()


df_processed.head()


df_processed['num_reported_accidents'].value_counts()


p99 = df_processed['num_reported_accidents'].quantile(0.99)
df_processed['num_reported_accidents'] = np.where(df_processed['num_reported_accidents'] > p99, p99, 
    df_processed['num_reported_accidents']
)


x= df_processed.drop('accident_risk',axis = 1)
y= df_processed['accident_risk']


import lightgbm as lgb


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


lgbm = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=1000, 
                         learning_rate=0.05, random_state=42, n_jobs=-1, 
                         categorical_feature=list(object_list))


lgbm.fit(X_train, y_train,  # Train on the training portion
         eval_set=[(X_test, y_test)],  # Monitor performance on the test/validation portion
         eval_metric='rmse',
         callbacks=[lgb.early_stopping(100, verbose=False)])


y_pred =lgbm.predict(X_test)


from sklearn.metrics import r2_score


print('rms', r2_score(y_test,y_pred))


import math


print('mean:',math.sqrt(mean_squared_error(y_test,y_pred)))


tests =  test.drop('id',axis=1)


tests


final_output = lgbm.predict(tests)


final_output = np.round(final_output,3)


test.columns


df_final = pd.DataFrame({'id':test['id'],'accident_risk':final_output})


df_final.to_csv('Road Accident Prediction.csv',index=False)

