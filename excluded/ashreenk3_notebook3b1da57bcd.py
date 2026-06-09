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


train_data=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv",index_col="id")
test_data=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv",index_col="id")


train_data.info()


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
train_data["Sex"]=encoder.fit_transform(train_data["Sex"])
test_data["Sex"] = encoder.transform(test_data["Sex"])


#BUILD DECISION TREE REGRESSOR TO PREDICT THE CALORIES BURNED BASED ON FEATURES
from sklearn.tree import DecisionTreeRegressor
features = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
X=train_data[features]
y=train_data["Calories"]
DecisionTreeModel=DecisionTreeRegressor(random_state=42)
DecisionTreeModel.fit(X,y)
predictions=DecisionTreeModel.predict(X)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, predictions)


#BUILD RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
features=["Sex","Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]
X=train_data[features]
y=train_data["Calories"]
RandomForestModel=RandomForestRegressor(n_estimators=100, random_state=42)
RandomForestModel.fit(X,y)
predictions = RandomForestModel.predict(X)
mse = mean_squared_error(y, predictions)
print("Mean Squared Error:", mse)


#BUILD LINEAR REGRESSOR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
LinearModel=LinearRegression()
LinearModel.fit(X,y)
linear_prediction=LinearModel.predict(X)
linear_mse = mean_squared_error(y, linear_prediction)

print("Linear Regression - Mean Squared Error:", linear_mse)



#BUILD LGBMREGRESSOR
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
LGBMModel = lgb.LGBMRegressor(n_estimators=100, random_state=42)
LGBMModel.fit(X, y)
lgbm_predictions = LGBMModel.predict(X)
lgbm_mse = mean_squared_error(y, lgbm_predictions)
print("LightGBM Regressor - Mean Squared Error:", lgbm_mse)



submission = pd.DataFrame()
submission["id"] = test_data.index
features = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
submission["Calories"] = LinearModel.predict(test_data[features])
submission.head()


submission = pd.DataFrame()
submission["id"] = test_data.index
features = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
submission["Calories"] = RandomForestModel.predict(test_data[features])
submission.head()


submission = pd.DataFrame()
submission["id"] = test_data.index
features = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
submission["Calories"] = LGBMModel.predict(test_data[features])
submission.head()


submission.to_csv("submission_LGBMModel.csv",index=False,header=True)


submission.to_csv("submission_LinearModel.csv",index=False,header=True)


submission.to_csv("submission_RandomForestModel.csv",index=False,header=True)

