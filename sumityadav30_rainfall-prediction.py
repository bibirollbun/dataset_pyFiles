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

df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df.head()


df.info()


df.columns


df[['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']][df['rainfall'] == 1].hist(bins= 50)


df[['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']][df['rainfall'] == 0].hist(bins= 50)


import matplotlib.pyplot as plt

# plotting correlation matrix
plt.imshow(df.corr(), cmap='Blues')

# adding colorbar
plt.colorbar()


# extracting variable names
variables = []
for i in df.corr().columns:
    variables.append(i)

# Adding labels to the matrix
plt.xticks(range(len(df.corr())), variables, rotation=45, ha='right')
plt.yticks(range(len(df.corr())), variables)

# Display the plot
plt.show()


df.corr()


df.columns


import pandas as pd
from datetime import datetime



# Convert day of year to month
df['month'] = df['day'].apply(lambda x: datetime(2024, 1, 1).replace(day=1) + pd.to_timedelta(x - 1, unit='D'))
df['month'] = df['month'].dt.month
df.head()


df_processed = df.drop(['temparature','id','day' ], axis = 1)

df_processed.head()


df.day


X = df_processed.drop(columns='rainfall', axis=1)
Y = df_processed['rainfall']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV


X_test


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score





param_grid = {
    'iterations': [500, 1000],  
    'learning_rate': [0.01, 0.1],  
    'depth': [4, 6],  
    'l2_leaf_reg': [1, 3, None],  
    'bootstrap_type': ['Bayesian', 'Bernoulli', 'No']
}


catboost = CatBoostClassifier(verbose=0, task_type="CPU") 

grid_search = GridSearchCV(catboost, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, Y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)





best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(Y_test, y_pred))




