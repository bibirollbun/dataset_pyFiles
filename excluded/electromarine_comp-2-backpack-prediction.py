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


#importing important libraries
import matplotlib.pyplot as plt


df=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df.head()


100*df.isnull().mean()


object_col=[]
for col in df.columns:
    if df[col].dtype=="O":
        object_col.append(col)
print(object_col) 



#Handling the missing values in categorical features by imputing the mode (most frequest class)

df[object_col]=df[object_col].apply(lambda x: x.fillna(x.mode()[0]) if x.isna else x)


# Checkingthe missing percentage again
100*df.isnull().mean()


#Handling the missing row values in ordinal feature by imputing mean value of that feature
df["Weight Capacity (kg)"]=df["Weight Capacity (kg)"].fillna(df["Weight Capacity (kg)"].median())


# Checking the missing percentage again
100*df.isnull().mean()


df.describe()


# Creating a dummy variable for the categorical variables and dropping the first one.
for col in object_col:
    dummies = pd.get_dummies(df[col], dtype=int, drop_first=True, prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[col], inplace=True)

df.head(-1)


# Putting feature variables to X
X = df.drop('Price',axis=1)

# Putting target variable to y
y = df['Price']


from sklearn.model_selection import train_test_split


np.random.seed(0)
X_train,X_test,y_train,y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 100)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error


rf=RandomForestRegressor(random_state=100,n_jobs=-1)


#Create the parameter grid based on the results of random search 
params = {
    'max_depth': [1, 2, 6,8, 10],
    'min_samples_leaf': [50,100],
    'max_features': [10,16,19],#inc 13
    'n_estimators': [10,30] #inc 40
}


#Define the custom scoring function for RMSE
rmse_scorer = make_scorer(mean_squared_error, squared=False)


# Instantiate the grid search model
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=rf, 
                           param_grid=params,
                           cv=4, 
                           verbose=-1, 
                           scoring = rmse_scorer)


grid_search.fit(X_train,y_train)


# Use the best estimator from GridSearchCV
rf_best = grid_search.best_estimator_
rf_best


 from yellowbrick.regressor import PredictionError


# Create the prediction error plot using the best estimator
visualizer = PredictionError(rf_best)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()  # Finalize and render the figure


#------------ 1st model R-squared:0.048


df_test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
df_test


100*df_test.isnull().mean()


object_col=[]
for col in df_test.columns:
    if df_test[col].dtype=="O":
        object_col.append(col)
print(object_col) 


#Handling the missing values in categorical features by imputing the mode (most frequest class)

df_test[object_col]=df_test[object_col].apply(lambda x: x.fillna(x.mode()[0]) if x.isna else x)


100*df_test.isnull().mean()


#Handling the missing row values in ordinal feature by imputing mean value of that feature
df_test["Weight Capacity (kg)"]=df_test["Weight Capacity (kg)"].fillna(df_test["Weight Capacity (kg)"].median())


100*df_test.isnull().mean()


# Creating a dummy variable for the categorical variables and dropping the first one.
for col in object_col:
    dummies = pd.get_dummies(df_test[col], dtype=int, drop_first=True, prefix=col)
    df_test = pd.concat([df_test, dummies], axis=1)
    df_test.drop(columns=[col], inplace=True)

df_test.head(-1)


predictions = grid_search.predict(df_test)


# Creating a DataFrame with 'id' and 'price'
submission_df = pd.DataFrame({
    'id': df_test['id'],  
    'Price': predictions
})

# Saving the DataFrame as an .csv file
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())




