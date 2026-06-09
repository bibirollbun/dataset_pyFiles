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


# Library Importation :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For Machine Learning :
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xg

# Pre-processing :
from sklearn.preprocessing import LabelEncoder

# For Neural Network :
import tensorflow as tf

# For Model Evaluation :
from sklearn.metrics import mean_squared_error

# Removing the warnings :
import warnings
warnings.filterwarnings("ignore")


# Importing the Data :
path_train = "/kaggle/input/playground-series-s5e2/train.csv"
path_test = "/kaggle/input/playground-series-s5e2/test.csv"
# Storing Data :
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)


df_train.head()


df_test.head()


df_train.shape


df_test.shape


for_id = pd.read_csv(path_test)
for_id


# Dropping 'id' column :
df_train.drop("id",axis=1,inplace=True)
df_test.drop("id",axis=1,inplace=True)


print(df_train.shape)
print(df_test.shape)


# Checking the missing values :
df_train.isna().sum()


more_miss = []
less_miss = []
for i in df_train.columns:
    print("Column {0} have {1}% missing data.\n".format(i,(df_train[i].isna().sum()/df_train.shape[0])*100))


cat_col = []
num_col = []
for i in df_train.columns:
    if df_train[i].dtype == 'O':
        cat_col.append(i)
    else:
        num_col.append(i)


cat_col


num_col


# Filling missing values by Mode :
for i in cat_col:
    df_train[i] = df_train[i].fillna(df_train[i].mode()[0])


from sklearn.impute import KNNImputer
# Initialize KNN Imputer with k=3
knn_imputer = KNNImputer(n_neighbors=3)

# Impute only the 'Weight' column
df_train[['Weight Capacity (kg)']] = knn_imputer.fit_transform(df_train[['Weight Capacity (kg)']])


df_train.isna().sum()


# Now checking the duplicate columns :
duplicate_count = df_train.duplicated().sum()
print(f"Number of duplicated rows: {duplicate_count}")


df_train.head()


cat_col


num_col


from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()
le6 = LabelEncoder()
le7 = LabelEncoder()

df_train['Brand']= le1.fit_transform(df_train['Brand'])
df_train['Material']= le2.fit_transform(df_train['Material'])
df_train['Size']= le3.fit_transform(df_train['Size'])
df_train['Laptop Compartment']= le4.fit_transform(df_train['Laptop Compartment'])
df_train['Waterproof']= le5.fit_transform(df_train['Waterproof'])
df_train['Style']= le6.fit_transform(df_train['Style'])
df_train['Color']= le7.fit_transform(df_train['Color'])


# Train - Test Split :
X = df_train.drop('Price',axis=1)
y = df_train['Price']


X.shape


y.shape


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Linear Regression Model :
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


y_pred = model.predict(X_test)


y_train_pred = model.predict(X_train)


print("Test : ",model.score(X_test,y_test))
print("Train : ",model.score(X_train,y_train))


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, y_train_pred)**0.5)
print(mean_squared_error(y_test, y_pred)**0.5)


# Parameter Tuning using Grid Search : 
from sklearn.model_selection import GridSearchCV
Model_Final = LinearRegression()
n_jobs = [1,2,3,4]
fit_intercept = [True, False]
copy_X = [True, False]
positive = [True, False]

random_grid = {'n_jobs': n_jobs,
               'fit_intercept': fit_intercept,
               'copy_X': copy_X,
               'positive': positive}

print(random_grid)


Sub_grid = GridSearchCV(estimator=Model_Final, param_grid=random_grid, cv=3, verbose=2, n_jobs=4)
Sub_grid.fit(X_train, y_train)


print("Best Parameters : ",Sub_grid.best_params_)
print("Best Score : ",Sub_grid.best_score_)


# Building Final Model and Submission : 
Model_Submit = LinearRegression(copy_X = True, fit_intercept = True, n_jobs = 1, positive = True)


# Importing test data : 
test_path = "/kaggle/input/playground-series-s5e2/test.csv"
test_data = pd.read_csv(test_path)
test_data.head()


# Missing value :
for i in cat_col:
    test_data[i] = test_data[i].fillna(test_data[i].mode()[0])


test_data[['Weight Capacity (kg)']] = knn_imputer.fit_transform(test_data[['Weight Capacity (kg)']])


# Encoding the data :
test_data['Brand']= le1.transform(test_data['Brand'])
test_data['Material']= le2.transform(test_data['Material'])
test_data['Size']= le3.transform(test_data['Size'])
test_data['Laptop Compartment']= le4.transform(test_data['Laptop Compartment'])
test_data['Waterproof']= le5.transform(test_data['Waterproof'])
test_data['Style']= le6.transform(test_data['Style'])
test_data['Color']= le7.transform(test_data['Color'])


test_data.isna().sum()


test_df = pd.read_csv(test_path)


test_data.drop('id',axis=1,inplace=True)


Model_Submit.fit(X,y)
y_pred = Model_Submit.predict(test_data)


# Submission : 
output = pd.DataFrame({'id': test_df.id,
                       'Machine failure': y_pred})
output.to_csv('submission.csv', index=False)


# Checking the Submission :
sub_path = "/kaggle/working/submission.csv"
sub = pd.read_csv(sub_path)


sub







