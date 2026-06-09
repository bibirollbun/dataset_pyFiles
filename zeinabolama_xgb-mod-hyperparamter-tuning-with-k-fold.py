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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from scipy import stats

from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from sklearn.pipeline import Pipeline
import os, glob, math, cv2, gc, logging, warnings, random

import lightgbm as lgb
import catboost
from sklearn.metrics import *
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

import shap
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
sample_submission = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")


feature_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


train.head(3)


feature_dict.head()


# Using feature_dict to select categorical and numerical features
categorical_features = feature_dict.query("type == 'Categorical'")['variable'].tolist()
numerical_features = feature_dict.query("type == 'Numerical'")['variable'].tolist()

# we separate target variables efs and efs_time
categorical_features.remove('efs')
numerical_features.remove('efs_time')

print("Num Categorical Features:", len(categorical_features))
print("Num Numerical Features:", len(numerical_features))


# Remove ID column from train and test datasets
train = train.drop('ID', axis=1)
test = test.drop('ID', axis=1)



feature_cols = test.columns.tolist()
features = feature_cols
target_col = 'efs'


plots_per_figure = 15

for i in range(0, len(categorical_features), plots_per_figure):
    plt.figure(figsize=(30, 15))
    for j, col in enumerate(categorical_features[i:i+plots_per_figure]):
        plt.subplot(3, 5, j+1)
        sns.countplot(data=train, x=col, hue=target_col, palette="Set2")
        plt.title(col)
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=45)  
        plt.legend(title=target_col, loc='upper right')  
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(20, 10))
sns.heatmap(train[numerical_features+[target_col]].corr(), annot=True)
plt.show()


folds = 4
train['kfold'] = -1  

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X=train, y=train[target_col])):
    train.loc[val_idx, 'kfold'] = fold
  

print(train.kfold.value_counts())


preds = np.zeros(len(test)) 
new_df = pd.DataFrame({'preds': preds})
new_df


feature_cols = categorical_features + numerical_features


for col in train.select_dtypes(include=['object']).columns:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')



learning_rate = [0.05 , 0.1]
max_depth = [6 , 8]
n_estimators = [300 , 500]



# all permutaions of learning_rate, max_depth, n_estimators to create params map

from itertools import product

params_map = []
for lr, md, ne in product(learning_rate, max_depth, n_estimators):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': lr,
        'max_depth': md,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'n_estimators': ne,
        'device': 'cuda',
        'tree_method': 'hist',  # Faster with categorical data
    }
    params_map.append(params)
len(params_map)



import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from itertools import product


folds = 4
scores = []
feature_importances = pd.DataFrame()
feature_importances['feature'] = feature_cols

for params in params_map:
    fold_scores = []
    for fold in range(folds):
      # Splitting the data into training and validation sets
      x_train = train[train.kfold != fold].copy()
      x_valid = train[train.kfold == fold].copy()
      x_test = test[feature_cols].copy()

      y_train = x_train[target_col]
      y_valid = x_valid[target_col]

      x_train = x_train[feature_cols]
      x_valid = x_valid[feature_cols]

      # XGBoost DMatrix with categorical support enabled
      train_data = xgb.DMatrix(data=x_train, label=y_train, enable_categorical=True)
      valid_data = xgb.DMatrix(data=x_valid, label=y_valid, enable_categorical=True)
      test_data = xgb.DMatrix(data=x_test, enable_categorical=True)



      # Training the model
      clf = xgb.train(
          params,
          train_data,
          num_boost_round=params['n_estimators'],
          evals=[(train_data, 'train'), (valid_data, 'valid')],
          verbose_eval=100,
          
      )

      # Predictions
      preds_train = clf.predict(train_data)
      preds_valid = clf.predict(valid_data)

      # Evaluate AUC
      train_auc = roc_auc_score(y_train, preds_train)
      valid_auc = roc_auc_score(y_valid, preds_valid)

      print(f"| Fold {fold + 1} | train AUC: {train_auc:.5f} | valid AUC: {valid_auc:.5f} |")
      print("|--------|----------------|----------------|")

      fold_scores.append(valid_auc)
    scores.append(np.mean(fold_scores))
    print(f"Average AUC for params {params}: {np.mean(fold_scores)}")

best_params_index = np.argmax(scores)
print(f"Best hyperparameters: {params_map[best_params_index]} with AUC: {scores[best_params_index]}")


#  Train bet model on the entire test data
best_params = params_map[best_params_index]

# Initialize predictions
final_predictions = np.zeros(len(test))

# Train with best hyperparameters on the entire dataset
x_train = train[feature_cols]
y_train = train[target_col]
x_test = test[feature_cols]

train_data = xgb.DMatrix(data=x_train, label=y_train, enable_categorical=True)
test_data = xgb.DMatrix(data=x_test, enable_categorical=True)

clf = xgb.train(
    best_params,
    train_data,
    num_boost_round=best_params['n_estimators'],  # Use the best n_estimators
    verbose_eval=100
)

final_predictions = clf.predict(test_data)


sample_submission


sample_submission['prediction'] = final_predictions
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head()

