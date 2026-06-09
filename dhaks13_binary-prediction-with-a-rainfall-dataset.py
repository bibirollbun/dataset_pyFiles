# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train_data.head()


train_data.drop('id',axis=1,inplace=True)
train_data.head()


y = train_data['rainfall']
X = train_data.drop('rainfall',axis=1)

print(X.head())
print(y.head())

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# from sklearn.preprocessing import StandardScaler

# normalizer = StandardScaler()
# X_scaled = normalizer.fit_transform(X)



# Define hyperparameter grids for each model
rf_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [3, 5, 10],
    'max_depth': [50, 100, 150, 200, None],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

dt_param_grid = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 3, 5],
    'criterion': ['gini', 'entropy']
}

gnb_param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

ab_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'algorithm': ['SAMME', 'SAMME.R']
}

lr_param_grid = {
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l1', 'l2'],
    'max_iter': [100, 200, 300],
    'C': [0.01, 0.1, 1.0, 10]
}

# Define SVC model
svc = SVC()

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}





# Define RSCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

# Fit model
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Test performance
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Best parameters:", grid_search.best_params_)
print("Test Accuracy:", accuracy)


print("Running RF")
rf = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(rf, rf_param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
print("RF fitted")


print("Running DT")
dt = DecisionTreeClassifier(random_state=42)
dt_random = RandomizedSearchCV(dt, dt_param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
dt_random.fit(X_train, y_train)
print("DT fitted")


print("Running GBN")
gnb = GaussianNB()
gnb_random = RandomizedSearchCV(gnb, gnb_param_grid, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
gnb_random.fit(X_train, y_train)
print("GBN fitted")


print("Running AB")
ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), random_state=42)
ab_random = RandomizedSearchCV(ab, ab_param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
ab_random.fit(X_train, y_train)
print("AB fitted")


print("Running LR")
lr = LogisticRegression(random_state=42)
lr_random = RandomizedSearchCV(lr, lr_param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
lr_random.fit(X_train, y_train)
print("LR fitted")


print("Best Parameters for Random Forest:", rf_random.best_params_)
print("Best Parameters for Decision Tree:", dt_random.best_params_)
print("Best Parameters for Gaussian Naive Bayes:", gnb_random.best_params_)
print("Best Parameters for AdaBoost:", ab_random.best_params_)
print("Best Parameters for Logistic Regression:", lr_random.best_params_)


# Create the best SVC model with optimal parameters
svc_best = SVC(**grid_search.best_params_)

svc_best.fit(X,y)


# model defining
rf_best = RandomForestClassifier(**rf_random.best_params_, random_state=42)
dt_best = DecisionTreeClassifier(**dt_random.best_params_, random_state=42)
gnb_best = GaussianNB(**gnb_random.best_params_)
ab_best = AdaBoostClassifier(**ab_random.best_params_, random_state=42)
lr_best = LogisticRegression(**lr_random.best_params_, random_state=42)

# Train models
rf_best.fit(X, y)
dt_best.fit(X, y)
gnb_best.fit(X, y)
ab_best.fit(X, y)
lr_best.fit(X, y)


test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

test_data.head()


ids = test_data['id']
predict_data = test_data.drop('id',axis=1)
predict_data.head()


predict_data.isna().sum()
predict_data['winddirection']=predict_data['winddirection'].fillna(predict_data['winddirection'].mean())


predict_data.head()


models = [svc_best,rf_best,dt_best,gnb_best,ab_best,lr_best]
test_preds = []
for model in models:
    test_preds.append(model.predict(predict_data))


test_pred = test_preds[0]
for pred in test_preds[1::]:
    test_pred += (pred)


output = []
for test in test_pred:
    if(test/6>0.8):
        output.append(1)
    else:
        output.append(0)
test_preds=np.array(output)


# Submission
submission = pd.DataFrame({'id': test_data['id'], 'rainfall': test_preds})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'.")


submission

