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


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt


# display the train set
train_csv = pd.read_csv('/kaggle/input/smoker-binary-class/train.csv')
train_csv.head()


# missing values -> None
train_csv.isnull().sum()


# test data for final prediction
test_csv = pd.read_csv('/kaggle/input/smoker-binary-class/test.csv')
test_csv.head()


# missing values? i guess not
test_csv.isnull().values.any()


# how the final .csv file should 
sub_csv = pd.read_csv('/kaggle/input/smoker-binary-class/sample_submission.csv')
sub_csv.head()


# age by smoking value
plt.figure()
plt.hist(train_csv[train_csv.smoking == 0]['age'], bins=20, alpha=0.5, label='Non-smokers')
plt.hist(train_csv[train_csv.smoking == 1]['age'], bins=20, alpha=0.5, label='Smokers')
plt.legend()
plt.title('Age Histogram by Smoking Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# weight vs waist circumference by smoking status
plt.figure()
plt.scatter(train_csv[train_csv.smoking == 0]['weight(kg)'], train_csv[train_csv.smoking == 0]['waist(cm)'], marker='o', label='Non-smokers')
plt.scatter(train_csv[train_csv.smoking == 1]['weight(kg)'], train_csv[train_csv.smoking == 1]['waist(cm)'], marker='x', label='Smokers')
plt.legend()
plt.title('Weight vs Waist by Smoking Status')
plt.xlabel('Weight (kg)')
plt.ylabel('Waist (cm)')
plt.show()


# correlation matrix heatmap
corr = train_csv.drop('id', axis=1).corr()
plt.figure()
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.index)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()


# separate the ids, features and labels
train_ids = train_csv['id']
test_ids = test_csv['id']
y = train_csv['smoking']
X = train_csv.drop(['id', 'smoking'], axis=1)
X_test = test_csv.drop('id', axis=1)


# split data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)


# looking for the best LGBM params
base_lgb = LGBMClassifier(random_state=42)
param_dist = {
    'n_estimators':      [100, 200, 300, 500, 700, 1000, 1500],
    'num_leaves':        [15, 31, 50, 75, 100, 150],
    'max_depth':         [3, 5, 7, 10, 15],
    'learning_rate':     [0.0001, 0.001, 0.01, 0.1],
    'subsample':         [0.3, 0.6, 0.8, 1.0],
    'colsample_bytree':  [0.3, 0.6, 0.8, 1.0],
    'reg_alpha':         [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    'reg_lambda':        [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    'scale_pos_weight':  [1, 2, 3, 5, 7, 10]
}

random_search = RandomizedSearchCV(
    estimator = base_lgb,
    param_distributions = param_dist,
    scoring = 'roc_auc',
    cv = 10,
    n_iter = 100,    
    random_state = 42,
    n_jobs = -1
)

random_search.fit(X_train, y_train)

print("Best parameters found:")
print(random_search.best_params_)
print(f"Best ROC AUC: {random_search.best_score_:.4f}")


# lgb = LGBMClassifier(random_state = 42, max_depth = 5)
# lgb.fit(X_train, y_train)
lgb = random_search.best_estimator_


y_pred_proba = lgb.predict_proba(X_valid)[:,1]
y_pred = lgb.predict(X_valid)

print(f"ROC_AUC Score %4f" % roc_auc_score(y_valid, y_pred_proba))
print(f"Accuracy Score %.2f" % accuracy_score(y_valid, y_pred))


res = lgb.predict_proba(X_test)[:, 1]

final_sub = pd.DataFrame({'id': test_ids, 'smoking': res})
final_sub.to_csv('submission.csv', index = False)
final_sub.head()




