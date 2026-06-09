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


import numpy as np
import pandas as pd
df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


df_test


import seaborn as sns
df['BMI'] = df['Weight']/np.square(df['Height']/100)
df_test['BMI'] = df_test['Weight']/np.square(df_test['Height']/100)
sns.heatmap(df.select_dtypes(exclude = 'object').corr(), cmap = 'RdBu_r', annot = True)


features = ['Duration','Heart_Rate','Body_Temp','BMI','Age']
X = df[features]
y = df['Calories']


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X[features] = ss.fit_transform(X[features])
df_test[features]  = ss.transform(df_test[features])





from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


xgb = XGBRegressor( colsample_bytree = 1.0,  learning_rate = 0.1, max_depth = 7, n_estimators = 300, subsample = 1.0)
xgb.fit(X_train,y_train)



y_pred = xgb.predict(X_test)
y_pred = np.clip(y_pred, 0, None)
from sklearn.metrics import mean_squared_log_error

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print("RMSLE:", rmsle)





sub_df=pd.DataFrame()

sub_df['id']=submission['id']

sub_df['Calories'] = xgb.predict(df_test[features])


sub_df[['id','Calories']].to_csv('submission.csv', index=False)


df_test[features]

