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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train


train.shape


train.info()


train.describe()


train.Sex.value_counts()


numeric=['Age','Duration','Weight','Height','Heart_Rate','Body_Temp']
categorical=['Sex']


# numeric
num_pipeline = Pipeline([
    ('scaler',StandardScaler())
])

# categorical
cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder())
])
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, numeric),
    ('cat', OneHotEncoder(), categorical)
])


train.drop('id',axis=1,inplace=True)


X=train.drop('Calories',axis=1)
y=train['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# RMSLE: 0.0673566410629442 degree=5


from sklearn.preprocessing import PolynomialFeatures

# Step 1: Transform with full_pipeline first
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)  # Use transform, NOT fit_transform

# Step 2: Apply PolynomialFeatures to both sets
poly_features_1 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly_1 = poly_features_1.fit_transform(X_train_prepared)
X_test_poly_1 = poly_features_1.transform(X_test_prepared)

# Step 3: Train and predict
LR = linear_model.LinearRegression()
LR.fit(X_train_poly_1, y_train)
yhat_1 = LR.predict(X_test_poly_1)


y_test_clipped = np.maximum(0, y_test)
predict_clipped_2= np.maximum(0, yhat_1)
rmsle_2 = np.sqrt(mean_squared_log_error(y_test_clipped, predict_clipped_2))
print("RMSLE:", rmsle_2)
#RMSLE: 0.06665558533187595 degree=3


RF=RandomForestClassifier()
RF.fit(X_train_prepared,y_train)
y_predict=RF.predict(X_test_prepared)
y_test_clipped_3 = np.maximum(0, y_test)
predict_clipped_4= np.maximum(0, y_predict)
rmsle_3 = np.sqrt(mean_squared_log_error(y_test_clipped_3, predict_clipped_4))
print("RMSLE:", rmsle_3)


test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test


test_id=test.id.copy()
test.drop('id',axis=1)


test_prepared=full_pipeline.transform(test)


# Assuming poly_features was already fit on training data
test_poly = poly_features_1.transform(test_prepared)



y_test_pred = LR.predict(test_poly)
y_test_pred = np.maximum(0, y_test_pred)  # Replace negative values with 0



submission_colory1=pd.DataFrame({
    'id':test_id,
    'Predict':y_test_pred
})
submission_colory1.to_csv('submission_colory1.csv',index=False)


