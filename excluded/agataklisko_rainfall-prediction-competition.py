# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, RocCurveDisplay, classification_report, f1_score, get_scorer_names
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn


data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_file = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_file['winddirection'] = test_file['winddirection'].fillna(test_file['winddirection'].mean())


data


data.info()


data.describe()


test_file['winddirection'] = test_file['winddirection'].fillna(test_file['winddirection'].mean())


data.rename({'temparature' : 'temperature'}, axis =1, inplace =True)
test_file.rename({'temparature' : 'temperature'}, axis =1, inplace =True)


day_counts = data['day'].value_counts().to_frame()
day_counts[day_counts['count']!=6].value_counts()


for i in data.index:
   if (i+1 - data.loc[i, 'day'])%365 != 0:
       data.loc[i, 'day'] = np.where((i+1)%365 ==0, 365, ((i+1)%365) )

data['day'].value_counts()


test_file['day'].value_counts()


data['month'] = pd.cut(data['day'], bins = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366], labels = np.arange(1, 13))
test_file['month'] = pd.cut(test_file['day'], bins = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366], labels = np.arange(1, 13))


data['winddirection'].nunique()


bins = [0, 11, 33, 56, 78, 101, 123, 146, 168, 191, 213, 236, 258, 281, 303, 326, 348, 360]
directions = ['N','NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
data['wind_dir'] = pd.cut(data['winddirection'], bins = bins, labels = directions, ordered = False)
data.drop(['winddirection', 'day'], axis = 1, inplace = True)
#
test_file['wind_dir'] = pd.cut(test_file['winddirection'], bins = bins, labels = directions, ordered = False)
test_file.drop(['winddirection', 'day'], axis = 1, inplace = True)



data['pressure_ch'] = data['pressure'] - data['pressure'].shift(1)
data['dewpoint_ch'] = data['dewpoint'] - data['dewpoint'].shift(1)
data['cloud_ch'] = data['cloud'] - data['cloud'].shift(1)
data['sunshine_ch'] = data['sunshine'] - data['sunshine'].shift(1)
data['cloud_ch'].fillna(data['cloud_ch'].mean(), inplace = True)
data['sunshine_ch'] = data['sunshine_ch'].fillna(data['sunshine_ch'].mean())
data['dewpoint_ch'] = data['dewpoint_ch'].fillna(data['dewpoint_ch'].mean())
data['pressure_ch'] = data['pressure_ch'].fillna(data['pressure_ch'].mean())
data


test_file['pressure_ch'] = test_file['pressure'] - test_file['pressure'].shift(1)
test_file['dewpoint_ch'] = test_file['dewpoint'] - test_file['dewpoint'].shift(1)
test_file['cloud_ch'] = test_file['cloud'] - test_file['cloud'].shift(1)
test_file['sunshine_ch'] = test_file['sunshine'] - test_file['sunshine'].shift(1)
test_file['cloud_ch'].fillna(test_file['cloud_ch'].mean(), inplace = True)
test_file['sunshine_ch'] = test_file['sunshine_ch'].fillna(test_file['sunshine_ch'].mean())
test_file['pressure_ch'] = test_file['pressure_ch'].fillna(test_file['pressure_ch'].mean())


colors = ['gold', 'steelblue']
sns.countplot(x = 'rainfall', data = data, palette=colors)
plt.xlabel('rainfall')
plt.ylabel('# of observations')
plt.show()


sns.countplot(x='month', hue = 'rainfall', data = data, palette=colors)
plt.show()


sns.countplot(x = data.wind_dir, hue = data['rainfall'], palette = colors)
plt.xticks(rotation = 90)
plt.title('Rainfall vs wind direction')
plt.show()


fig, axes = plt.subplots(7, 2, figsize = (24,16) )
axes = axes.flatten()[:-1]
for loc, ax in enumerate(axes):
    sns.lineplot(x = 'id', y = data.drop(['id', 'rainfall', 'month', 'wind_dir'], axis = 1).columns[loc], data = data, hue = 'rainfall', ax=ax, palette = colors)
plt.show()


correlation = data.drop(['id', 'wind_dir'], axis =1).corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(correlation, annot = True, ax = ax)


fig, axes = plt.subplots(7, 2, figsize = (12,30) )
axes = axes.flatten()[:-1]
for loc, ax in enumerate(axes):
    sns.boxplot(y = data.drop(['id', 'rainfall', 'wind_dir', 'month'], axis = 1).columns[loc], x = 'rainfall', data = data, ax = ax, palette = colors)
plt.show()


fig, axes = plt.subplots(7, 2, figsize = (12,30) )
axes = axes.flatten()[:-1]
for loc, ax in enumerate(axes):
    sns.histplot(x = data.drop(['id', 'month', 'rainfall', 'wind_dir'], axis = 1).columns[loc], data = data, ax = ax)
plt.show()


from scipy.stats import skew
cols = []
skews = []
for col in data.select_dtypes(include = ['float']).columns:
    cols.append(col)
    skews.append(skew(data[col]))

ax = sns.barplot(x = cols, y = np.abs(skews))
ax.set_ylabel('skewness')
plt.xticks(rotation = 90)
plt.show()


data_encoded = pd.get_dummies(data, columns = ['wind_dir', 'month'])
data_encoded.drop(['id'], axis = 1, inplace = True)



y = data_encoded[['rainfall']]
X = data_encoded.drop('rainfall', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 22, shuffle = True, stratify = y)


lr = LogisticRegression(max_iter = 5000, class_weight = 'balanced')
poly = PolynomialFeatures()
ss = StandardScaler()
lr_pipe = Pipeline(steps = [('poly', poly),
                            ('ss', ss),
                            ('lr', lr)])
lr_params = [{'poly__degree':[1,2],
           'lr__penalty' : ['l1', 'l2'],
           'lr__C' : [0.05, 0.072, 0.075, 0.1],
            'lr__solver' : ['lbfgs','newton-cg','liblinear','saga']
          }]

lr_search = GridSearchCV(lr_pipe, lr_params, cv = 5, scoring = 'f1_weighted')
#grid search is commented out to avoid time consuming fitting every time the notebook is running
#lr_search.fit(X_train, y_train.values.ravel())
#lr_search.best_params_


#model with hyperparamteters found in the grid
lr_grid = LogisticRegression(max_iter = 5000, class_weight = 'balanced', C=0.075,
                   penalty='l1', solver='saga')

poly_grid = PolynomialFeatures(degree = 1)
ss_grid = StandardScaler()
lr_pipe_grid = Pipeline(steps = [('poly', poly_grid),
                            ('ss', ss_grid),
                            ('lr', lr_grid)])
lr_pipe_grid.fit(X_train, y_train.values.ravel())





lr_pipe_grid.score(X_train, y_train)


def model_eval(model, X_test, y_test):
    """prints roc auc score and classification report of the fitted model"""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]

    roc_auc = roc_auc_score(y_test, proba)
    class_report = classification_report(y_test, pred)
    print(f'roc_auc_score = {roc_auc} ')
    print(class_report)



print('Metrics of the test set:')
model_eval(lr_pipe_grid, X_test, y_test)


print('Metrics of the train set:')
model_eval(lr_pipe_grid, X_train, y_train)


dt1 = DecisionTreeClassifier(max_depth=1, random_state=42)
dt2 = DecisionTreeClassifier(max_depth=2, random_state=42)
dt3 = DecisionTreeClassifier(max_depth=3, random_state=42)
adb = AdaBoostClassifier(random_state=42)

ada_param_grid = [{'n_estimators': np.arange(50, 230, 30),
               'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
               'estimator':[dt1, dt2, dt3]}]
grid_ada = GridSearchCV(adb, param_grid = ada_param_grid, cv = 4,scoring = 'f1_weighted')
grid_ada.fit(X_train, y_train.values.ravel())
grid_ada.best_params_


grid_ada.score(X_train, y_train)


model_eval(grid_ada, X_test, y_test)


ss = StandardScaler()
param_grid_svc = {'kernel': ['linear', 'rbf','poly'] , 
              'C':[0.5, 1, 5, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'degree' : [1,2,3,4,5]}
svc=SVC(probability=True, random_state = 22)

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
grid_svc = GridSearchCV(svc, param_grid = param_grid_svc, cv = 4, scoring = 'f1_weighted')
#grid_svc.fit(X_train_scaled, y_train.values.ravel())
#grid_svc.best_params_


#training SVC model with params found in the grid
svc_grid = SVC(C=5, degree=1, gamma=1, kernel='linear', probability=True, random_state = 22)

svc_grid.fit(X_train_scaled, y_train)
model_eval(svc_grid, X_test_scaled, y_test)

