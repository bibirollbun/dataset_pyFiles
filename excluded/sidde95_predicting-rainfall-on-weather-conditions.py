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


train_df = pd.read_csv('/kaggle/input/helping-hand/train.csv')
test_df = pd.read_csv('/kaggle/input/helping-hand/test.csv')
sample_df = pd.read_csv('/kaggle/input/helping-hand/test.csv')


train_df.sample(5)


test_df.sample(5)


sample_df.sample(3)


print(f"There are {train_df.shape[0]} rows and {train_df.shape[1]} columns")



train_df.info()


train_df.isnull().sum()


# Dropping the id columns
train_df = train_df.drop('id', axis = 1)


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train_df.sample()


def boxplot(data):
    num_cols = train_df.select_dtypes(exclude = 'O').columns
    for i, j in enumerate(num_cols):
        plt.subplot(len(num_cols)//3+1, 3, i+1 )
        sns.boxplot(data = data, x = j)
        plt.title(f"{j} plot")
    plt.tight_layout()

plt.figure(figsize = (15, 10))
boxplot(train_df)


def kdeplot(data):
    num_cols = train_df.select_dtypes(exclude = 'O').columns
    for i, j in enumerate(num_cols):
        plt.subplot(len(num_cols)//3+1, 3, i+1 )
        sns.kdeplot(data = data, x = j)
        plt.title(f"{j} plot")
    plt.tight_layout()

plt.figure(figsize = (15, 10))
kdeplot(train_df)


plt.figure(figsize = (15, 8))
sns.heatmap(train_df.corr(), annot = True)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = train_df.drop('rainfall', axis = 1)
y = train_df.rainfall

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 9)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {
          "Logistic Regression": LogisticRegression(),
          "Decision Tree": DecisionTreeClassifier(),
          "Random Forest": RandomForestClassifier(),
          "AdaBoost": AdaBoostClassifier(),
          "Gradient Boost": GradientBoostingClassifier(),
          "XGBoost": XGBClassifier(),
          "KNN": KNeighborsClassifier()
         }

result_list = []
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    ar_score = accuracy_score(y_test, y_pred)
    pre_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1])

    metrics_dict = {
    'model name': model_name,
    'accuracy score': ar_score,
    'precision score': pre_score,
    'recall score': rec_score,
    'f1-score': f_score,
    'roc_auc_score': roc_score 
    }

    result_list.append(metrics_dict)

metric_df = pd.DataFrame(result_list)
metric_df


metric_df.sort_values(by = ['accuracy score', 'roc_auc_score'], ascending = False)


from sklearn.model_selection import GridSearchCV, StratifiedKFold

lg_params = {
    'penalty': ['l1', 'l2'],       # type of regularization
    'C': [0.01, 0.1, 1, 10, 100],  # inverse regularization strength
    'solver': ['liblinear', 'saga']
}

gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.01],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gridcv_models = [
    ("Gradient Boost Classifier", GradientBoostingClassifier(), gb_params),
    ("Logistic Regression", LogisticRegression(), lg_params)
]


for model_name, model, params in gridcv_models:
    gridcv = GridSearchCV(estimator= model, cv = cv, scoring = 'roc_auc', verbose= 1, n_jobs = -1, param_grid= params)
    gridcv.fit(X_train_scaled, y_train)
    
    print(f"---------------{model_name}---------------")
    print(gridcv.best_params_)
    print(gridcv.best_estimator_)
    print(roc_auc_score(y_test, gridcv.predict_proba(X_test_scaled)[:,1]))


test_df.sample(5)


test_df.shape


test_df.isnull().sum()


test_df.winddirection = test_df.winddirection.fillna(test_df.winddirection.mean())


test_df_copy = test_df
test_df_copy = test_df_copy.drop('id', axis = 1)


# Data Preprocessing
test_scaled_df = scaler.transform(test_df_copy)

# Prediction on test df
model= LogisticRegression(C=0.1, solver='liblinear', penalty= 'l2')
model.fit(X_train_scaled, y_train)

print(f"Roc-Auc Score: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1])}")
y_pred = model.predict_proba(test_scaled_df)[:,1]
y_pred[:5]


submission = pd.DataFrame({'id': test_df.id,
             'rainfall': y_pred})
submission


submission.to_csv('submission.csv', index = False)
print("Sucessfully saved!")




