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


!pip install -U scikit-learn
!pip install xgboost


from sklearn.metrics import root_mean_squared_log_error

# define a function to get the metrics of predictions
def accuracy_final_score(y_true, y_pred):
    score = root_mean_squared_log_error(y_true, y_pred)
    print("Number of samples: ", len(y_true))
    print("Root Mean Squared Logarithmic Error: ", score)
    return score


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data
input_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")

# Extract features and target
X = input_df[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
y = input_df['Calories']

# Convert categorical 'Sex' column to binary
X["IsMaleSex"] = (X["Sex"] == "male").astype(int)
X = X.drop("Sex", axis=1)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
X_scaler = MinMaxScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)


X_train_scaled


X_train.describe()


y_train.describe()


y_train.to_numpy()


# apply linear regression
from sklearn.linear_model import LinearRegression, Ridge
model2 = Ridge(alpha=10)
model = LinearRegression()

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_val_scaled)

y_pred[y_pred < -1] = 0
accuracy_final_score(y_val.to_numpy(), y_pred)


from xgboost import XGBRegressor
# create an xgboost regression model
model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_val_scaled)

accuracy_final_score(y_val.to_numpy(), y_pred)


test_set = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test_set = test_set.set_index("id")
test_set["IsMaleSex"] = (test_set["Sex"] == "male").astype(int)
test_set = test_set.drop("Sex", axis=1)
test_set = test_set[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'IsMaleSex']]
X_test_scaled = X_scaler.transform(test_set)
test_set["Calories"] = model.predict(X_test_scaled)
final_pred = test_set[["Calories"]]


final_pred[final_pred["Calories"]<0] = 0


final_pred = final_pred.reset_index()


print(final_pred.head())
final_pred.to_csv("test_predictions.csv", index=False)




