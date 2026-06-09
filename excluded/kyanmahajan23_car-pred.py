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
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
data

data[['year', 'month', 'date']] = data['date'].str.split('-', expand=True)
data[['date', 'time']] = data['date'].str.split(' ', expand =True)
test[['year', 'month', 'date']] = test['date'].str.split('-', expand=True)
test[['date', 'time']] = test['date'].str.split(' ', expand =True)
data

print(data)


plates = data['plate'];

code  = [];
other=[]
for plate in plates:
    i= len(plate)-1;
    
    while i>0 and plate[i].isdigit():
        i = i-1

    code.append(plate[i+1:])
    other.append(plate[:i+1])


# print(code, other)


# print(data)
data['un']= other
data['code'] = code
data = data.drop('plate', axis = 1)
print(data)
        


plates = test['plate'];

code  = [];
other=[]
for plate in plates:
    i= len(plate)-1;
    
    while i>0 and plate[i].isdigit():
        i = i-1

    code.append(plate[i+1:])
    other.append(plate[:i+1])


# print(code, other)


# print(data)
test['un']= other
test['code'] = code
test = test.drop('plate', axis = 1)
print(test)


data[['h','m', 's']] = data['time'].str.split(':', expand =True)
test[['h','m', 's']] = test['time'].str.split(':', expand =True)
data = data.drop('time', axis = 1)
test = test.drop('time', axis=1)

print(data)
print(test)




# data_new = data.drop('time', axis =1 );

# data_new['plate']= data_new['plate'].str[-2:]
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Fit the encoder and transform the data
data['un'] = label_encoder.fit_transform(data['un'])
# print(data])

# print(code)
label_encoder = LabelEncoder()

# Fit the encoder and transform the data
test['un'] = label_encoder.fit_transform(test['un'])


data_new = data.apply(pd.to_numeric, errors='coerce')
# test_new = test.drop('time', axis =1 );
# test_new['plate']= test_new['plate'].str[-2:]

test_new = test.apply(pd.to_numeric, errors='coerce')
print(data_new)
print(test_new)





idd = test_new.iloc[:, 0].to_numpy()

idd




data_new


data_new.corr()


data_new.columns


# cols = ['id', 'plate', 'date', 'price', 'year', 'month', 'h', 'm', 's']
# import matplotlib.pyplot as plt
# import seaborn as sns

# for col in cols:
#     sns.scatterplot(x= data_new[col], y= data_new['price'])
#     plt.show()




# class_cols = ['year', 'month']

# for cols in class_cols:
#     sns.barplot(x= data_new[cols], y= data_new['price'])
#     plt.show()


# class_cols = ['year', 'month']

# for cols in class_cols:
#     sns.boxplot(x= data_new[cols], y= data_new['price'])
#     plt.show()
# test_new = test_new.drop(['h', 'm', 's', 'month', 'date'], axis =1)
data_new


# data_new = data_new.drop(['h', 'm', 's', 'month', 'date'], axis =1)
# test_new = test_new.drop(['h', 'm', 's', 'month', 'date'], axis =1)
# from sklearn.preprocessing import StandardScaler

# # Assuming `data_new` is a 2D array-like structure (e.g., a NumPy array or Pandas DataFrame)
# sr = StandardScaler()

# # Fit the scaler to the data and then transform it

# data_new_ = sr.fit_transform(data_new)
# test_new_ = sr.transform(test_new)
# # Print the scaled data
# # print(data_new_scaled)

# # print(pd.DataFrame(data_new_scaled).columns)

# data_new_ = pd.DataFrame(data_new_)
# test_new_ =  pd.DataFrame(test_new_)
print(data_new_*2)
print(test_new_)



from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
model = DecisionTreeRegressor(random_state=42)

# Train the model

x_train, x_test, y_train, y_test = train_test_split(data_new_.drop(1, axis =1), data_new_[1], test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(r2_score(y_test, y_pred))
print(f'Mean Squared Error: {mse}')

# pd.DataFrame(data_new_scaledd).apply(type)


# data_new_scaledd = data_new_scaledd.astype('float64')


data_s = data_new_
data_s
Q1 = data_s.quantile(0.25)
Q3 = data_s.quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Define bounds to filter out outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers by filtering rows that are outside the bounds
filtered_data = data_s[~((data_s < lower_bound) | (data_s > upper_bound)).any(axis=1)]

# Print the filtered dataset (after removing outliers)
print(filtered_data)


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
model = DecisionTreeRegressor(random_state=42)

# Train the model

x_train, x_test, y_train, y_test = train_test_split(filtered_data.drop(1, axis =1),filtered_data[1], test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(r2_score(y_test, y_pred))
print(f'Mean Squared Error: {mse}')


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


X_train, X_test, y_train, y_test = train_test_split(filtered_data.drop(1, axis =1),filtered_data[1], test_size=0.2, random_state=42)
# Define the parameter grid
param_grid_dtree = {
    'max_depth': [None, 10, 20, 30, 40, 50],  # Depth of the tree (None means nodes are expanded until all leaves are pure)
    'min_samples_split': [2, 5, 10, 20],     # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4, 10],       # Minimum number of samples required to be at a leaf node
    'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
  
  
}

# Initialize the Decision Tree Classifier
dtree = DecisionTreeRegressor(random_state=42)

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid_dtree, scoring='r2',cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-validation Accuracy: {r2_score}")

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy on the test set




dtree = DecisionTreeRegressor(max_depth= 10, max_features= None, min_samples_leaf= 4, min_samples_split= 20)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

print(r2_score(y_test,y_pred))


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Initialize the XGBoost Regressor
xgboost_model = xgb.XGBRegressor(random_state=42)

# Set up the parameter grid (adjust this as needed)
param_grid_xgb = {
    'max_depth': [3, 6, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [10],
  
}

# Set up GridSearchCV with 5-fold cross-validation and r2 as the scoring metric
grid_search_xgb = GridSearchCV(estimator=xgboost_model, param_grid=param_grid_xgb, scoring='r2', cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the training data
grid_search_xgb.fit(X_train, y_train)

# Get the best parameters and score
best_params_xgb = grid_search_xgb.best_params_
best_score_xgb = grid_search_xgb.best_score_

print(f"Best Parameters for XGBoost: {best_params_xgb}")
print(f"Best Cross-validation R² for XGBoost: {best_score_xgb}")

# Get the best model from GridSearchCV
best_model_xgb = grid_search_xgb.best_estimator_

# Predict on the test set
y_pred_xgb = best_model_xgb.predict(X_test)

# Evaluate the model's performance on the test set using R²
test_r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"Test R² for XGBoost: {test_r2_xgb}")



import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
xgboost = xgb.XGBRegressor(learning_rate=0.1, max_depth=10, n_estimators=200, random_state=42)
xgboost.fit(X_train,y_train)
y_pred_ = xgboost.predict(X_test)

# Evaluate the model's performance on the test set using R²
test_r2_xgb = r2_score(y_test, y_pred_)
print(f"Test R² for XGBoost: {test_r2_xgb}")



# print(X_test)
# # print(test_new_.drop(1, axis =1))
# test_new_ = test_new_.drop(1, axis =1)
y_pred_final = xgboost.predict(test_new_)

# y_pred_final

concatenated_array = np.concatenate((idd.reshape(-1, 1), y_pred_final.reshape(-1, 1)), axis=1)

print(concatenated_array)



# np.concatenate([idd, y_pred_final])

final_Sub = pd.DataFrame(concatenated_array, columns = ['id','price' ])

print(final_Sub)

final_Sub.apply(pd.to_numeric)






final_Sub['id'] = final_Sub['id'].astype(np.int32)
print(final_Sub*2)


final_Sub.to_csv('/kaggle/working/output.csv', index=False)


