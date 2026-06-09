# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #для визуализации
import seaborn as sns #для визуализации

from sklearn import linear_model #линейные моделиё
from sklearn import tree #деревья решений
from sklearn import ensemble #ансамбли
from sklearn import metrics #метрики
from sklearn import preprocessing #предобработка
from sklearn.model_selection import train_test_split #сплитование выборки

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/bioresponse/train.csv')
df_train.head(3)
df_train.isnull().sum() # проверка на наличие пропусков


df_train.describe()


df_test = pd.read_csv('/kaggle/input/bioresponse/test.csv')
df_test.head(3)


# Проверка классов на сбалансированность
sns.countplot(data = df_train, x = 'Activity');



X = df_train.drop(columns = 'Activity') # матрица наблюдений X 
y = df_train.iloc[:,0] # вектор ответов y


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42, test_size = 0.2)


# Создаем объект класса логистическая регрессия
lr = linear_model.LogisticRegression(max_iter = 50)
# Обучаем модель
lr.fit(X_train, y_train)
y_test_pred = lr.predict(X_test)
print('f1 score на тестовом наборе {:.2f}'.format(metrics.f1_score(y_test, y_test_pred)))


# Для тех же данных построим модель случайного леса


#Создаем объект класса дерево решений
rf = ensemble.RandomForestClassifier(random_state = 42)
#Обучаем модель
rf.fit(X_train, y_train)
#Выводим значения метрики 
y_test_pred_rf = rf.predict(X_test)
print('Test f1: {:.2f}'.format(metrics.f1_score(y_test, y_test_pred_rf)))


from sklearn.model_selection import GridSearchCV #импортируем библиотеку

param_grid = {'penalty' : ['l2', 'none'],
              'solver' : ['lbfgs', 'saga']
}
# Подберем гиперпараметры применительно к модели логистической регрессии
grid_search = GridSearchCV(
     estimator=linear_model.LogisticRegression(
        random_state=42,
        max_iter= 50
    ),
    param_grid=param_grid, 
    cv=5, 
    n_jobs = -1
)
%time grid_search.fit(X_train, y_train)
y_test_pred = grid_search.predict(X_test)
print('f1_score на тестовом наборе: {:.2f}'.format(metrics.f1_score(y_test, y_test_pred)))
print("Наилучшие значения гиперпараметров: {}".format(grid_search.best_params_))



from sklearn.model_selection import RandomizedSearchCV

param_distributions = {'penalty': ['l2', 'none'] ,
              'solver': ['lbfgs', 'sag'],
               'C': list(np.linspace(0.01, 1, 10, dtype=float))}

random_search = RandomizedSearchCV(
    estimator=linear_model.LogisticRegression(random_state=42, max_iter=50), 
    param_distributions=param_distributions, 
    cv=5, 
    n_iter = 10, 
    n_jobs = -1
)

%time random_search.fit(X_train, y_train)
y_test_pred = random_search.predict(X_test)
print('f1_score на тестовом наборе: {:.2f}'.format(metrics.f1_score(y_test, y_test_pred)))
print("Наилучшие значения гиперпараметров: {}".format(random_search.best_params_))







