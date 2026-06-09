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
from sklearn.preprocessing import StandardScaler

# Load the correct train CSV
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

# Apply StandardScaler to selected columns
scaler = StandardScaler()
train[['Episode_Length_minutes', 'Host_Popularity_percentage']] = scaler.fit_transform(
    train[['Episode_Length_minutes', 'Host_Popularity_percentage']]
)




print(train.isnull().sum())


print(train.describe())




import warnings
warnings.filterwarnings('ignore')  



print(train.head())  



from sklearn.metrics import mean_squared_error
import numpy as np


y_pred = [1, 2, 3, 4, 5]
y_true = [1, 2, 2, 4, 5]


mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')




print(train.columns)



import seaborn as sns
import matplotlib.pyplot as plt



sns.histplot(train['Listening_Time_minutes'], kde=True)
plt.title('Distribution of Listening Time')
plt.show()




train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median(), inplace=True)



print(train.isnull().sum())




train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median(), inplace=True)



['Genre', 'Publication_Day', 'Episode_Sentiment']



# Check the columns in the 'train' DataFrame
print(train.columns)



X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Dummy data
X = pd.DataFrame(np.random.rand(1000, 20), columns=[f'feature_{i}' for i in range(20)])
y = pd.Series(np.random.rand(1000))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print(X_test.shape)    # (200, 20)
print(y_test.shape)    # (200,)
print(y_pred.shape)    # (200,)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')



from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)

print(f'Random Forest MSE: {rf_mse}')
print(f'Random Forest RMSE: {rf_rmse}')



from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')



model = LinearRegression()
model.fit(X_train, y_train)



import pandas as pd

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

print(train.columns)
print(test.columns)





