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


# importing all the necessary libraries

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


data_train= pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
data_train.head()


data_train.shape


data_train.info()


data_train.isnull().sum()


data_train['Brand']= data_train['Brand'].fillna('Not mentioned')
data_train['Material']= data_train['Material'].fillna('Not mentioned')
data_train['Size']= data_train['Size'].fillna('Not mentioned')
data_train['Laptop Compartment']= data_train['Laptop Compartment'].fillna('Not mentioned')
data_train['Waterproof']= data_train['Waterproof'].fillna('Not mentioned')
data_train['Style']= data_train['Style'].fillna('Not mentioned')
data_train['Color']= data_train['Color'].fillna('Not mentioned')
data_train['Weight Capacity (kg)']= data_train['Weight Capacity (kg)'].fillna(data_train['Weight Capacity (kg)'].mean())


plt.figure(figsize=(10, 5), facecolor= 'lightgrey')
sns.distplot(data_train['Weight Capacity (kg)'])
plt.show()


fig, axes= plt.subplots(1,2, figsize=(12, 6), facecolor= 'lightgrey')

sns.boxplot(data_train['Weight Capacity (kg)'], ax= axes[0])
axes[0].set_title('Weight Capacity')

sns.boxplot(data_train['Compartments'], ax= axes[1])
axes[1].set_title('Compartments')


fig, ax= plt.subplots(1, 2, figsize=(12, 6), facecolor= 'grey')
data_train['Waterproof'].value_counts().plot(kind= 'pie', autopct='%.1f', ax= ax[0], colors=['lightblue', 'lightcoral', 'blue'])
ax[0].set_title('Waterproof Distribution', color= 'white')

data_train['Laptop Compartment'].value_counts().plot(kind= 'pie', autopct='%.1f', ax= ax[1], colors=['lightblue', 'lightcoral', 'blue'])
ax[1].set_title('Laptop Compartment Distribution', color= 'white')


fig, axes= plt.subplots(3, 2, figsize=(12, 18), facecolor= 'lightgrey')

sns.countplot(x= 'Brand', data= data_train, ax= axes[0, 0], palette='Blues')
axes[0, 0].set_title('Brand Distribution')

sns.countplot(x= 'Material', data= data_train, ax= axes[0, 1], palette='Greens')
axes[0, 1].set_title('Material Distribution')

sns.countplot(x= 'Size', data= data_train, ax= axes[1, 0], palette='Reds')
axes[1, 0].set_title('Size Distribution')

sns.countplot(x= 'Style', data= data_train, ax= axes[1, 1], palette='Purples')
axes[1, 1].set_title('Style Distribution')

sns.countplot(x= 'Color', data= data_train, ax= axes[2, 0], palette='Oranges')
axes[2, 0].set_title('Color Distribution')

fig.delaxes(axes[2,1])


num_data= data_train.select_dtypes(include= ['number'])

correlation= num_data.corr()
sns.heatmap(correlation, fmt= '.1f', cbar= True, square= True, annot= True, annot_kws={'size':8}, cmap= 'Blues')


data_train= pd.get_dummies(data_train, columns= ['Style', 'Color', 'Size', 'Material', 'Brand', 'Laptop Compartment', 'Waterproof'], drop_first= True, dtype= int)
data_train.head()


X= data_train.drop(columns=['Price'], axis= 1)
Y= data_train['Price']
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, random_state= 2)


xgb= XGBRegressor(
    n_estimators= 1500,
    learning_rate= 0.01,
    early_stopping_rounds= 100
)
xgb.fit(X_train, Y_train, eval_set=[(X_test, Y_test)])
Y_pred= xgb.predict(X_test)


param_grid= {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 250, 500],
    'max_depth': [2, 4, 6]
}

grid_search= GridSearchCV(xgb, param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, Y_train, eval_set= [(X_test, Y_test)])


improved_xgb= grid_search.best_estimator_


rmse= np.sqrt(mean_squared_error(Y_test, Y_pred))
mae= mean_absolute_error(Y_test, Y_pred)
r2= r2_score(Y_test, Y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squeared Error: {rmse}')
print(f'R2 Score: {r2}')


print('Best Parameters:', grid_search.best_params_)
print('Best RMSE Score:', -grid_search.best_score_)


data_test= pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
data_test.head()


data_test['Brand']= data_test['Brand'].fillna('Not mentioned')
data_test['Material']= data_test['Material'].fillna('Not mentioned')
data_test['Size']= data_test['Size'].fillna('Not mentioned')
data_test['Laptop Compartment']= data_test['Laptop Compartment'].fillna('Not mentioned')
data_test['Waterproof']= data_test['Waterproof'].fillna('Not mentioned')
data_test['Style']= data_test['Style'].fillna('Not mentioned')
data_test['Color']= data_test['Color'].fillna('Not mentioned')
data_test['Weight Capacity (kg)']= data_test['Weight Capacity (kg)'].fillna(data_train['Weight Capacity (kg)'].mean())


data_test= pd.get_dummies(data_test, columns= ['Style', 'Color', 'Size', 'Material', 'Brand', 'Laptop Compartment', 'Waterproof'], drop_first= True, dtype= int)


data_result= improved_xgb.predict(data_test.values)


submission_file= pd.DataFrame({
    'id': data_test['id'],
    'Price': data_result
})


submission_file


submission_file.to_csv('submission.csv', index= False)
print('SUBMISSION SUCCESSFUL')

