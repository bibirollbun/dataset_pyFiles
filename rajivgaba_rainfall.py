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


train_file = "/kaggle/input/playground-series-s5e3/train.csv"
test_file = "/kaggle/input/playground-series-s5e3/test.csv"


train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

train_data.head()


train_data.info()


train_data.isnull().mean()*100


train_data.describe()


y = train_data.pop("rainfall")
X = train_data


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=45)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


X_train.drop(columns=['id','day'], axis=1, inplace=True)


X_test.drop(columns=['id','day'], axis=1, inplace=True)


X_train.columns


scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train)
scaled_test = scaler.transform(X_test)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


X_train.shape[1]





# Model building using tensorflow

import tensorflow as tf

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=10, activation = "relu"))

ann.add(tf.keras.layers.Dense(units=10, activation = "relu"))

tf.keras.layers.BatchNormalization(),

ann.add(tf.keras.layers.Dense(units=8, activation = "relu"))
ann.add(tf.keras.layers.Dense(units=8, activation = "relu"))

ann.add(tf.keras.layers.Dense(units=1, activation = "sigmoid"))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2)

ann.predict(X_test)

test_cols = X_test.columns

prediction_data = test_data[test_cols]

prediction_result = ann.predict(prediction_data)

prediction_result

test_data['id'].shape

prediction_result[:,0]


# model building using XGB

# X_train.shape, X_test.shape, y_train.shape, y_test.shape, prediction_data.shape


'''
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

xgb_cfl = xgb.XGBClassifier(n_jobs = -1,objective = 'binary:logistic')

params = {
        'n_estimators' : [100, 200, 500, 750], # no of trees 
        'learning_rate' : [0.01, 0.02, 0.05, 0.1, 0.25],  # eta
        'min_child_weight': [1, 5, 7, 10],
        'gamma': [0.1, 0.5, 1, 1.5, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 10, 12]
        }

folds = 5

param_comb = 800

random_search = RandomizedSearchCV(xgb_cfl, 
    param_distributions=params, 
    n_iter=param_comb, 
    scoring='accuracy', 
    n_jobs=-1, cv=5, verbose=3, random_state=56
    )

random_search.fit(X_train, y_train)

random_search.best_estimator_

print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best accuracy for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ )
print('\n Best hyperparameters:')
print(random_search.best_params_)

best_params = random_search.best_params_

best_params = {
    'subsample': [0.6],
    'n_estimators': [750],
    'min_child_weight': [7],
    'max_depth': [10],
    'learning_rate': [0.05],
    'gamma': [5],
    'colsample_bytree': [0.8]
}

%%time 

param_comb = 800

random_search = RandomizedSearchCV(xgb_cfl, 
    param_distributions=best_params, 
    n_iter=param_comb, 
    scoring='accuracy', 
    n_jobs=-1, cv=5, verbose=3, random_state=56
    )

a = random_search.fit(X_train, y_train)

xgb_predictions_hpt = random_search.predict_proba(X_test)
accuracy_score(y_test, xgb_predictions_hpt)

xgb_predictions_hpt


prediction_result = random_search.predict(prediction_data)
prediction_result

prediction_result = random_search.predict_proba(prediction_data)
prediction_result
'''


results_df = pd.DataFrame(
    {
    "id": test_data['id'],
    "rainfall": prediction_result[:,0]
    }
)

results_df.head()


# Submission file

results_df['rainfall'] = results_df['rainfall'].apply(lambda x: str(x)[:3])

results_df = results_df.fillna("0")

results_df ['rainfall'] = results_df['rainfall'].apply(lambda x: x if x != "nan" else 0.0)

results_df[results_df['rainfall'] == "nan"]

results_df.to_csv("ann_pred.csv", index=False)

