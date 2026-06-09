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
X_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
X_test.head()


df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df 


df.describe()


df.isnull().sum()


missing_percentage= df.isnull().mean()*100
#filter col with more than 30$ or 50 %
col_with_missing_30= missing_percentage[missing_percentage>3.0]

#display the col with more than 30$ misssing value
print('col_with_missing_30')
print( col_with_missing_30)


df=df.drop(columns ={  'id'} , axis=1)
df


#since its only one col. SimpleImputer used
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
numeric_cols =  {'num_sold'}
from sklearn.impute import SimpleImputer
imp = SimpleImputer( strategy="median")
df['num_sold'] = imp.fit_transform(df[['num_sold']])


df.head()


df.dtypes


df. groupby(['country','product']).size()  


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])
df.head()


X_test_id = X_test.id
X_test =X_test.drop(columns ={  'id'} , axis=1)
label_encoder = LabelEncoder()
for column in X_test.select_dtypes(include=['object']).columns:
    X_test[column] = label_encoder.fit_transform(X_test[column])
X_test.head()


y_train = df['num_sold']
X_train =df.drop(columns ='num_sold' , axis=1)


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Initialize and train the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)


submission_df = pd.DataFrame({ 'id': X_test_id,  'num_sold':y_pred_dt})  


# Save the DataFrame as a CSV file without the index column
submission_df.to_csv('submission.csv', index=False)




