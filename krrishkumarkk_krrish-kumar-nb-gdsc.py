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
import numpy as np
TRAIN_PATH=r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH=r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"


# Load and clean the training data
df = pd.read_csv(TRAIN_PATH)  
df.describe
df.head
# number of missing values in each coulumn 
value_count_by_column=(df.isnull().sum())
print(value_count_by_column)

#columns that are null
cols_null=[col for col in df.columns if df[col].isnull().any()]
print(cols_null)



from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Drop rows where the target is NaN in actual training data
df.drop(df[df['CORRUCYSTIC_DENSITY'].isna()].index, inplace=True)

# Define the target column
target_col = 'CORRUCYSTIC_DENSITY'

# Select numeric columns, then drop the target column to get feature columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target_col).tolist()

# Save the feature list for later use
features_used = num_cols.copy()

# Separate features and target
X = df[features_used]
y = df[target_col]

# Impute missing values in features 
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data to test model performance
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train the model 
#Currently I have trained the model with elastic net ml model which gave me one of the best results
#for using any other model.. I just change the model name and get their import statement
model = ElasticNet()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f" RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f" MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# Train the model on the full data
model.fit(X_imputed, y)



# Load the test data 
df_test = pd.read_csv(TEST_PATH)  

# Select the same feature columns as training
X_test = df_test[features_used]

# Impute missing values using the same imputer
X_test_imputed = imputer.transform(X_test)

# Make predictions
y_test_pred = model.predict(X_test_imputed)




#Using pipeline instead of train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

model = ElasticNet()

my_pipeline=Pipeline(steps=[('preprocessor',SimpleImputer()),('model',model)])

#can use fit for training 
#using cross val score for cross validation to improve accuracy and model quality

score=-1*cross_val_score(my_pipeline,X,y,cv=5, scoring="neg_mean_squared_error")

print(score)
print(score.mean())


# this technique will be used with predicted dataset that are predicted by the models 
#let them be y_pred1 and y_pred2
import pandas as pd


y_pred1 = pd.DataFrame({'num1': [1, 2, 3, 4], 'num2': [9, 10, 11, 12]})
y_pred2 = pd.DataFrame({'num1': [5, 6, 7, 8], 'num2': [13.14, 15, 16, 17]})


y_pred1_sampled = y_pred1*0.3
y_pred2_sampled = y_pred2*0.7

# Concatenate the sampled DataFrames
mixed = y_pred1_sampled+ y_pred2_sampled
print(mixed)




#Linear Regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=100,random_state=42)
#also have max leaf node as a parameter

#Ridge Regressor
from sklearn.linear_model import Ridge
model=Ridge()

#Lasso Regressor
from sklearn.linear_model import Lasso
model=Lasso()

#Dummy/baseline Regressor 
from sklearn.dummy import DummyRegressor
model=DummyRegressor(strategy='mean')


#ExtraTressRegressor
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()

#KNN Regressor
from sklearn.neighbors import KNeighborsRegressor
model=KNeighborsRegressor()

#Gausian Regressor
from sklearn.gaussian_process import GaussianProcessRegressor
model=GaussianProcessRegressor()


#XGB Regressor
from xgboost import XGBRegressor
model=XGBRegressor(n_estimators=500,learning_rate=0.05)
#we can use other parameters for tuning with XGBoost regressor while training like 
#early_stopping_rounds and n_jobs


# Create the submission file 
submission = pd.DataFrame({
    'LOCAL_IDENTIFIER': df_test['LOCAL_IDENTIFIER'].astype(int),
    'CORRUCYSTIC_DENSITY': y_test_pred.astype(float)
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✅ Submission file created!")
print(submission.head())

