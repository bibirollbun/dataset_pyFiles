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


#import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder



df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")



df_train.head(5)



# Save the id from test data
test_ids = df_test["id"].copy()
df_test.drop("id", inplace=True, axis=1)
df_train.drop("id", inplace=True, axis=1)



df_train.shape



#Show the quantity of null value
df_train.isnull().sum()


categorical_columns = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]



#Impute
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df_train[column] = le.fit_transform(df_train[column].astype(str))
    df_test[column] = le.transform(df_test[column].astype(str))
    label_encoders[column] = le


imputer_cat = KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
df_train[categorical_columns] = imputer_cat.fit_transform(df_train[categorical_columns])
df_test[categorical_columns] = imputer_cat.transform(df_test[categorical_columns])


for column in categorical_columns:
    df_train[column] = label_encoders[column].inverse_transform(df_train[column].astype(int))
    df_test[column] = label_encoders[column].inverse_transform(df_test[column].astype(int))


numeric_columns = df_train.select_dtypes(include=[np.number]).columns
numeric_columns = [col for col in numeric_columns if col != "Price"]

imputer_num = KNNImputer(n_neighbors=5, weights="distance")  # K-Mean
df_train[numeric_columns] = imputer_num.fit_transform(df_train[numeric_columns])
df_test[numeric_columns] = imputer_num.transform(df_test[numeric_columns])



#spilt the data to train & test
X = df_train.drop("Price", axis =1)
y = df_train["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#pip catboost
!pip install catboost


#pose the model
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


cat_features = ["Brand" , "Material" , "Size" ,"Laptop Compartment", "Waterproof" , "Style", "Color"]
ctb = CatBoostRegressor(iterations=1000 , depth=3 , learning_rate=0.1 , task_type="GPU"  , cat_features=cat_features , verbose = 0)

ctb.fit(X_train, y_train)

y_pred = ctb.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)



test_prediction = ctb.predict(df_test)
output = pd.DataFrame({'id': test_ids, 'Price': test_prediction})
output.to_csv('submission.csv', index=False)





