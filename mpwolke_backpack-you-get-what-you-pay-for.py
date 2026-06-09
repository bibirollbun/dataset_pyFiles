# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.info()


train.tail(3)


train.duplicated().sum()


cols_fillna = ["Brand", "Material","Size","Laptop Compartment", "Waterproof", "Style",
               "Color"]

# replace 'NaN' with 'None' in these columns
for col in cols_fillna:
    train[col].fillna('None',inplace=True)
    test[col].fillna('None',inplace=True)


train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].fillna("0")
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].fillna("0")


pd.option_context('mode.use_inf_as_na', True)
sns.displot(train['Price']);


train['Brand'].value_counts().plot(kind = 'barh', color='r', title= 'Backpack Brand');


sns.barplot(x = train['Brand'], y = train['Price'])
plt.xticks(rotation='vertical')
plt.show();


train['Style'].value_counts().plot(kind = 'barh', color='r', title='Backpacks Styles');


sns.barplot(x=train['Waterproof'], y = train['Price']);


#Gaurab Baral https://www.kaggle.com/code/itzgauurab/worse-than-baseline

from sklearn.preprocessing import LabelEncoder

categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
# Label Encoding for categorical columns (you can also use OneHotEncoding for a larger dataset)
le = LabelEncoder()
for col in categorical_cols:
    train[col] = le.fit_transform(train[col])
X = train[['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']]
y = train['Price']


X = train.drop(['Price'],axis=1)
Y = train['Price'].values
#X = X.select_dtypes(exclude=['object'])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


SGDreg = SGDRegressor()
SGDreg.fit(X_train, y_train)


pred = SGDreg.predict(X_train)
sgd_mse = mean_squared_error(y_train, pred)
sgd_rmse = np.sqrt(sgd_mse)
sgd_rmse


param_grid = {
    'alpha': 10.0 ** -np.arange(1, 7),
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    'max_iter': [1000, 5000, 10000]
}

grid_search = GridSearchCV(SGDreg, param_grid)
grid_search.fit(X_train, y_train)
print("Best score: " + str(grid_search.best_score_))


Knn = KNeighborsRegressor()
Knn.fit(X_train, y_train)


pred = Knn.predict(X_train)
k_mse = mean_squared_error(y_train, pred)
k_rmse = np.sqrt(k_mse)
k_rmse


param_grid = {'n_neighbors': np.arange(1, 12, 2),
              'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(Knn, param_grid)
grid_search.fit(X_train, y_train)
print("Best score: " + str(grid_search.best_score_))


final_model = grid_search.best_estimator_
final_pred = final_model.predict(X_test)
final_pred = final_pred.tolist()
for pred in range(0, len(final_pred)):
    print("Predicition: " + str(round(final_pred[pred], 2)) + " Actual: " + str(y_test[pred]))


fig_dims = (20, 10)
fig, ax = plt.subplots(figsize=fig_dims)
ax.scatter(y_test, final_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

