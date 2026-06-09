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


# importing the used packages and functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)


# reading the datas

df_train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
df_trainLabels = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)
df_test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)
print(f"{df_train.shape}")
print(f"{df_trainLabels.shape}")
print(f"{df_test.shape}")


df_train.head()


df_trainLabels.head()


# Flattening the labels

df_trainLabels = np.ravel(df_trainLabels)
df_trainLabels


# train-test split for the algos

X_train, X_test, y_train, y_test = train_test_split(df_train, df_trainLabels, test_size=0.33, random_state=42)

# Normalized datas for the SVC algo

normalized = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train_norm = normalized.transform(X_train)
X_test_norm = normalized.transform(X_test)


# Basic logistic regression to claim base score

logistic = LogisticRegression()
logistic.fit(X_train, y_train)

reg_pred = logistic.predict(X_test)

reg_score = accuracy_score(y_test, reg_pred)
reg_prob = logistic.predict_proba(X_test)

print(f"The first 30 predicted classes: {reg_pred[:30]}")
print(f"The first 30 actual classes:    {y_test[:30]}")
print(f"Prediction score: {reg_score:.3f}")
print(f"Prediction probability: {reg_prob[0, 0]:.3f}")

c = 0
for pred, test in zip(reg_pred, y_test):
    if pred != test:
        c += 1

print(f"Total failures between prediction and test: {c}")


# Basic KNN Classification to claim base score

knn_modell = KNeighborsClassifier(n_jobs=-1)
knn_modell.fit(X_train, y_train)

knn_pred = knn_modell.predict(X_test)
knn_score = accuracy_score(y_test, knn_pred)

print(f"The first 30 predicted classes: {knn_pred[:30]}")
print(f"The first 30 actual classes:    {y_test[:30]}")
print(f"Prediction score: {knn_score:.3f}")

c = 0
for pred, test in zip(knn_pred, y_test):
    if pred != test:
        c += 1

print(f"Total failures between prediction and test: {c}")


# Basic SVC model to claim base score

svc = SVC(class_weight='balanced')
svc.fit(X_train_norm, y_train)

svc_pred = svc.predict(X_test_norm)
svc_score = accuracy_score(y_test, svc_pred)

print(f"The first 30 predicted classes: {svc_pred[:30]}")
print(f"The first 30 actual classes:    {y_test[:30]}")
print(f"Prediction score: {svc_score:.3f}")

c = 0
for pred, test in zip(svc_pred, y_test):
    if pred != test:
        c += 1

print(f"Total failures between prediction and test: {c}")


# Basic decesion tree classification to claim base score

best_n_splits = 0
best_depth = 0
best_tree_score = 0

for n_splits in range(2, 15):
    tree_cross_validation = KFold(n_splits=n_splits,
                                  shuffle=True,
                                  random_state=42)
    print(f"n_splits: {n_splits}")

    for depth in range(1,15):
        tree_classifier = DecisionTreeClassifier(
            max_depth=depth, random_state=42)
        if tree_classifier.fit(X_train, y_train).tree_.max_depth < depth:
            break
        tree_score = np.mean(cross_val_score(tree_classifier,
                                             X_train, y_train,
                                             scoring='accuracy',
                                             cv=tree_cross_validation))
        
        if tree_score > best_tree_score:
            best_tree_score = tree_score
            best_n_splits = n_splits
            best_depth = depth
            
        print(f"Depth: {depth:2} Score: {abs(tree_score):.3f}")

print(f"\nBest params: n_splits: {best_n_splits} max_depth: {best_depth} best score: {best_tree_score:.3f}")


models = pd.DataFrame({'Models':['Logistic Regression', 'KNN', 'SVC', 'Decesion Tree'],
                       'Scores':[reg_score, knn_score, svc_score, best_tree_score]})
models = models.sort_values(by='Scores', ascending=False, ignore_index=True)
models


sns.barplot(x='Scores', y='Models', data=models)


# First optimizing only for n_neighbors param

best_score = 0
best_param = 0

for n_neigh in range(1,50):
    knn_modell = KNeighborsClassifier(n_neighbors=n_neigh, n_jobs=-1)
    knn_modell.fit(X_train, y_train)

    knn_pred = knn_modell.predict(X_test)
    knn_score = accuracy_score(y_test, knn_pred)

    if knn_score > best_score:
        best_score = knn_score
        best_param = n_neigh

print(f"Best score: {best_score:.3f}")
print(f"Best param: {best_param}")


best_knn = KNeighborsClassifier(n_neighbors=best_param, n_jobs=-1)
best_knn.fit(X_train, y_train)
best_knn_pred = best_knn.predict(X_test)

c = 0
for pred, test in zip(best_knn_pred, y_test):
    if pred != test:
        c += 1

print(f"First 30 KNN predictions: {knn_pred[:30]}")
print(f"First 30 actual classes:  {y_test[:30]}")
print(f"Total failures between prediction and test: {c}")


grid_knn_modell = KNeighborsClassifier(n_jobs=-1)

param_grid = {'n_neighbors': [1, 3, 5, 7, 10, 25, 50, 100],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan', 'cosine', 'minkowski']}

scoring_metric = 'neg_mean_squared_error'
search = GridSearchCV(
    estimator=grid_knn_modell, param_grid=param_grid,
    scoring=scoring_metric, n_jobs=-1, refit=True,
    return_train_score=True, cv=10)
search.fit(X_train, y_train)

print(f"Best parameters: {search.best_params_}")
best_score = abs(search.best_score_)
print(f"Cross-val mean squared error of the best parameters: {best_score:.3f}")


grid_knn_best = KNeighborsClassifier(n_neighbors=7, metric='euclidean', weights='uniform')
grid_knn_best.fit(X_train, y_train)
grid_knn_pred = grid_knn_best.predict(X_test)
grid_knn_score = accuracy_score(y_test, grid_knn_pred)

c = 0
for pred, test in zip(grid_knn_pred, y_test):
    if pred != test:
        c += 1

print(f"Prediction score: {grid_knn_score:.3f}")
print(f"First 30 KNN predictions: {grid_knn_pred[:30]}")
print(f"First 30 actual classes:  {y_test[:30]}")
print(f"Total failures between prediction and test: {c}")


knn_modell = KNeighborsClassifier(n_neighbors=best_param, n_jobs=-1)
scoring = cross_val_score(knn_modell, X=X_train, y=y_train,
                          cv=10, scoring='neg_mean_squared_error',
                          n_jobs=-1)
base_score = np.mean(np.abs(scoring))
print(f"Base score with the best parameter: {base_score:.3f}")


selector_knn_modell = KNeighborsClassifier(n_jobs=-1)
selector = SequentialFeatureSelector(
    estimator=selector_knn_modell,
    direction='backward',
    cv=3,
    scoring='neg_mean_squared_error',
    n_features_to_select=14
)

selector.fit(X_train, y_train)
feature_mask = selector.support_
selected = [feature for feature, support in zip(X_train.columns, feature_mask) if support]
print(f"Selected features: {selected}")


# Let's see the base score for this model

scoring_metric = 'neg_mean_squared_error'
scores = cross_val_score(
    selector_knn_modell, X=X_train.loc[:, feature_mask], y=y_train,
    cv=10, scoring=scoring_metric, n_jobs=-1)
base_score = np.mean(np.abs(scores))
print(f"Base score with default parameters: {base_score:.3f}")


# Now the grid search give new params

search = GridSearchCV(
    estimator=selector_knn_modell, param_grid=param_grid,
    scoring=scoring_metric, n_jobs=-1, refit=True,
    return_train_score=True, cv=10)
search.fit(X_train.loc[:, feature_mask], y_train)


print(f"Best parameters: {search.best_params_}")
best_score = abs(search.best_score_)
print(f"Cross-val mean squared error of the best parameters: {best_score:.3f}")


# Let's use this new params for the KNN model

knn_modell = KNeighborsClassifier(n_neighbors=7, metric='cosine', weights='uniform', n_jobs=-1)
knn_modell.fit(X_train.loc[:, feature_mask], y_train)
knn_score = knn_modell.score(X_test.loc[:, feature_mask], y_test)
knn_pred = knn_modell.predict(X_test.loc[:, feature_mask])
print(f"KNN modell score with selected best correlated features: {knn_score:.3f}")


c = 0
for pred, test in zip(knn_pred, y_test):
    if pred != test:
        c += 1

print(f"First 30 KNN predictions: {knn_pred[:30]}")
print(f"First 30 actual classes:  {y_test[:30]}")
print(f"Total failures between prediction and test: {c}")


# Time to use the model for the test data

final_knn_model = KNeighborsClassifier(n_jobs=-1)
final_selector = SequentialFeatureSelector(
    estimator=final_knn_model,
    direction='backward',
    cv=3,
    scoring='neg_mean_squared_error',
    n_features_to_select=14
)

final_selector.fit(df_train, df_trainLabels)
feature_mask = selector.support_
selected = [feature for feature, support in zip(df_train.columns, feature_mask) if support]
print(f"Selected features: {selected}")


scoring_metric = 'neg_mean_squared_error'
scores = cross_val_score(
    final_knn_model, X=df_train.loc[:, feature_mask], y=df_trainLabels,
    cv=10, scoring=scoring_metric, n_jobs=-1)
base_score = np.mean(np.abs(scores))
print(f"Base score with default parameters: {base_score:.3f}")


final_knn_modell = KNeighborsClassifier(n_neighbors=7, metric='euclidean', weights='uniform', n_jobs=-1)
final_knn_modell.fit(df_train.loc[:, feature_mask], df_trainLabels)
final_knn_score = final_knn_modell.score(df_train.loc[:, feature_mask], df_trainLabels)
final_knn_pred = final_knn_modell.predict(df_test.loc[:, feature_mask])
print(f"KNN modell score with selected best correlated features: {final_knn_score:.3f}")
print(f"The first 30 predicted classes: {final_knn_pred[:30]}")


data = {'Solution':final_knn_pred, 'Id': [i for i in range(1,9001)]}
solution = pd.DataFrame(data=data)
solution = solution.set_index('Id')


# The solution as a csv file

solution.to_csv('solution.csv')




