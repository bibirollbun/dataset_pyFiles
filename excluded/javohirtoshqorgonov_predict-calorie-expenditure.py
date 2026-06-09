import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error


train=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
train.head(2)


test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test.head()


sample=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sample.head()


X = train.drop(columns=['Calories'])
y = train['Calories']


categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])



pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestRegressor())
])


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

# model qurish
pipeline.fit(X_train, y_train)


y_pred=pipeline.predict(X_test)
RMSE=mean_squared_error(y_pred,y_test)
RMSE


pred=pipeline.predict(test)


sample.head(2)


sample["Calories"] = pred
sample


sample.to_csv("submission.csv", index=False)




