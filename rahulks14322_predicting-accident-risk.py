# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
Sample=pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
train=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


## The data of train is ##
train.head()


## The data of test is ##
test.head()


## The data of Sample is ##
Sample.head()


## clenning the data ##
train.head()#i remove the id column..



## creating the feature for the model ##

X=train.drop(columns=['id','accident_risk'])
y=train['accident_risk']
X_test = test.drop(columns=['id'], axis=1)


# Define feature types
categorical = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean = ['road_signs_present', 'public_road', 'holiday', 'school_season']
numeric = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


#Preprocessor
preprocessor = ColumnTransformer([
    ('categ', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ('bool', 'passthrough', boolean),
    ('num', 'passthrough', numeric)
])


# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
## now its time see the len of the variable ##
#x_train.shape=(414203, 12)
#x_test.shape=(103551, 12)
#y_train.shape=(414203,)
#y_test.shape=(103551,)


# Define Model
model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor',LinearRegression())
                 ])


## For these i will use the model called Decision tree. it is best for predictind the accurency ans..

model.fit(X_train, y_train) 


y_pred=model.predict(X_val)
y_pred


model.score(X_val,y_val)


model.predict(X_test)



# Calculate metrics
import numpy as np
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print("RMSE: ", rmse)
print("R2: ", r2)


y_pred2 = model.predict(X_test)
Sample['accident_risk'] = y_pred2
Sample.to_csv('submission.csv', index=False)
print(Sample.head())

