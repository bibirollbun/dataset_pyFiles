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


train_df=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")


train_df.head()


train_df.columns


train_df.shape


train_df.describe()


train_df.info()


train_df.value_counts()


train_df.isnull().sum()


X=train_df.drop("BeatsPerMinute",axis=1)
Y=train_df["BeatsPerMinute"]


X


Y


from sklearn.model_selection import train_test_split


X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,random_state=42)


from sklearn.ensemble import RandomForestRegressor


random_model = RandomForestRegressor(
    n_estimators=50,    # reduce trees (default = 100)
    max_depth=15,       # limit tree depth
    random_state=42,
    n_jobs=-1           # use all CPU cores
)



#Training the model
random_model.fit(X_train,Y_train)


random_val_predict=random_model.predict(X_val)


random_val_predict.shape


Y_val.shape


from sklearn.metrics import mean_squared_error, mean_absolute_error


import numpy as np


print("Mean Squared Error : ",mean_squared_error(random_val_predict,Y_val))
print("Mean Absolute Error : ",mean_absolute_error(random_val_predict,Y_val))
print("Root Mean Squared Error : ",np.sqrt(mean_squared_error(random_val_predict,Y_val)))


from xgboost import XGBRegressor


xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


## Training :
xgb_model.fit(X_train,Y_train)


xgb_val_predict=xgb_model.predict(X_val)


print("Mean Squared Error : ",mean_squared_error(xgb_val_predict,Y_val))
print("Mean Absolute Error : ",mean_absolute_error(xgb_val_predict,Y_val))
print("Root Mean Squared Error : ",np.sqrt(mean_squared_error(xgb_val_predict,Y_val)))


combine_val_predict=(random_val_predict+xgb_val_predict)/2


test_df=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


test_df.head()


random_predict=random_model.predict(test_df)


random_predict[:10]


xgb_predict=xgb_model.predict(test_df)


combined_predict=(random_predict+xgb_predict)/2


combined_predict[:10]


test_df.columns


submission = pd.DataFrame({
    "id": test_df["id"],            # 'Id' column from test.csv
    "BeatsPerMinute": combined_predict
})

# Save as CSV with header
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")
























































