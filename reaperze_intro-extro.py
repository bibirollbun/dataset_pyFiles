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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier , BaggingClassifier , ExtraTreesClassifier,GradientBoostingClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


import torch
from torch.utils.data import Dataset ,DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


df_test


print("Shape of training set is: ",df_train.shape)
print('Shape of test set is: ',df_test.shape)


# Columns with missing values
print('Number of Missing Values in the following Training set features: ')
print(df_train.isnull().sum())

print('Number of Missing Values in the following Test set features: ')
df_test.isnull().sum()


df_train


# PERFORMING LABELENCODING
le = LabelEncoder()

# Encode 'Stage_fear' (preserving NaN)
mask_stage_fear_train = df_train['Stage_fear'].notnull()
df_train.loc[mask_stage_fear_train, 'Stage_fear'] = le.fit_transform(df_train.loc[mask_stage_fear_train, 'Stage_fear'])


mask_stage_fear_test = df_test['Stage_fear'].notnull()
df_test.loc[mask_stage_fear_test, 'Stage_fear'] = le.fit_transform(df_test.loc[mask_stage_fear_test, 'Stage_fear'])

# Encode 'Drained_after_socializing' (preserving NaN)
le2_tr = LabelEncoder()
mask_drained_tr = df_train['Drained_after_socializing'].notnull()
df_train.loc[mask_drained_tr, 'Drained_after_socializing'] = le2_tr.fit_transform(df_train.loc[mask_drained_tr, 'Drained_after_socializing'])


le2_t = LabelEncoder()
mask_drained_t = df_test['Drained_after_socializing'].notnull()
df_test.loc[mask_drained_t, 'Drained_after_socializing'] = le2_t.fit_transform(df_test.loc[mask_drained_t, 'Drained_after_socializing'])


# SEPARATING FEATURES CONTAINING MISSING VALUES

feature_na = [feature for feature in df_train.columns if df_train[feature].isnull().sum() > 0]

print('Training Data')

for feature in feature_na:
    print("{} : {} % missing values".format(feature,np.round(df_train[feature].isnull().mean(),4)))
print('-----------------------------------------------------')
print('Test Data')
for feature in feature_na:
    print("{} : {} % missing values".format(feature,np.round(df_test[feature].isnull().mean(),4)))


# HANDLING THE MISSING VALUES 

cols = ['Stage_fear' ,'Drained_after_socializing']
for col in feature_na:
    le = LabelEncoder()

    
    mask_tr = df_train[col].notnull()
    df_train.loc[mask_tr, col] = le.fit_transform(df_train.loc[mask_tr, col])
    df_train[col] = df_train[col].astype(float)
    df_train[col] = df_train[col].fillna(df_train[col].mean())

    mask_t = df_test[col].notnull()
    df_test.loc[mask_t, col] = le.fit_transform(df_test.loc[mask_t, col])
    df_test[col] = df_test[col].astype(float)
    df_test[col] = df_test[col].fillna(df_test[col].mean())
    
    
for col in cols:
    df_train[col] = (df_train[col] > 0.5).astype(int)
    df_test[col] = (df_test[col] > 0.5).astype(int)

# ENCODING THE TARGET CLASS
le_y1 = LabelEncoder()
le_y2 = LabelEncoder()
le_y1.fit(df_train['Personality'])
le_y2.fit(df_test['Personality'])
df_train['Personality'] = le_y1.transform(df_train['Personality'])
df_test['Personality'] = le_y2.transform(df_test['Personality'])



# GENERAL INFORMATION REGARDING DATASET

print('Summary of Training Data: ')
print(df_train.describe())
print('----------------------------------------------------------------------------')
print('Summary of Testing Data')
print(df_test.describe())


# HISTOGRAM PLOT FOR ALL FEATURES 

for feature in df_train.drop(columns = ['id','Stage_fear','Drained_after_socializing']):
    plt.figure(figsize=(6,4))
    sns.histplot(df_train[feature], bins=20, kde=True)
    plt.title('Histogram of {}'.format(feature))
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()



# BOX PLOT FOR ALL FEATURES 

for feature in df_train.drop(columns = ['id','Stage_fear','Drained_after_socializing']):
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df_train[feature])
    plt.title('Boxplot of {}'.format(feature))
    plt.show()



X = df_train.drop(columns = ['id','Personality'])
y = df_train['Personality']

X_train,X_val , y_train,y_val = train_test_split(X,y ,test_size = 0.2 , random_state =42)
X_test = df_test.drop(columns = ['id','Personality'])
y_test = df_test['Personality']


X_train


X_val


X_test


 # NORMALIZATION OF TRAINING AND VALIDATION DATASET

ss = StandardScaler()
ss.fit(X_train)

X_train  = ss.transform(X_train)
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)


print("Shape of X_train is: ",X_train.shape)
print("Shape of X_test is: ",X_test.shape)
print("Shape of X_val is: ",X_val.shape)


def objective(trial):
    # Choose classifier type
    classifier_name = trial.suggest_categorical(
        'classifier',
        [
            'DecisionTree', 'RandomForest', 'Bagging',
            'ExtraTrees', 'GradientBoosting', 'XGBoost', 'LogisticRegression'
        ]
    )

    # Common possible hyperparameters
    max_depth = trial.suggest_int('max_depth', 1, 10)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    
    # Classifier-specific instantiation and hyperparameters
    if classifier_name == 'DecisionTree':
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif classifier_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=2
        )
    elif classifier_name == 'Bagging':
        model = BaggingClassifier(
            n_estimators=n_estimators,
            random_state=2
        )
    elif classifier_name == 'ExtraTrees':
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=2
        )
    elif classifier_name == 'GradientBoosting':
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=2
        )
    elif classifier_name == 'XGBoost':
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=2
        )
    elif classifier_name == 'LogisticRegression':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        model = LogisticRegression(
            solver=solver,
            penalty=penalty,
            random_state=2
        )
    else:
        raise ValueError("Unknown classifier")
    
    # Fit and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    return score



# OPTIMIZATION OF HYPERPARAMETERS USING OPTUNA 

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)



# FEATCHING THE BEST PARAMETERS OBTAINED FROM OPTUNA

results = study.trials_dataframe()

best_params_per_classifier = {}

for clf in ['DecisionTree', 'RandomForest', 'Bagging', 'ExtraTrees', 
            'GradientBoosting', 'XGBoost', 'LogisticRegression']:
    clf_trials = results[results['params_classifier'] == clf]
    if not clf_trials.empty:
        best_trial = clf_trials.sort_values('value', ascending=False).iloc[0]
        # Collect all columns that start with 'params_'
        params = {col.replace('params_', ''): best_trial[col]
                  for col in results.columns if col.startswith('params_')}
        best_params_per_classifier[clf] = params
    else:
        print(f"No Optuna trials found for classifier: {clf}")



# SETTING MODEL FOR DIFFERENT CLASSIFIERS

classifiers = {}

for name, params in best_params_per_classifier.items():
    if name == 'DecisionTree':
        classifiers[name] = DecisionTreeClassifier(max_depth=params['max_depth'])
    elif name == 'RandomForest':
        classifiers[name] = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=2
        )
    elif name == 'Bagging':
        classifiers[name] = BaggingClassifier(
            n_estimators=params['n_estimators'],
            random_state=2
        )
    elif name == 'ExtraTrees':
        classifiers[name] = ExtraTreesClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=2
        )
    elif name == 'GradientBoosting':
        classifiers[name] = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=2
        )
    elif name == 'XGBoost':
        classifiers[name] = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=2
        )
    elif name == 'LogisticRegression':
        classifiers[name] = LogisticRegression(
            solver=params.get('solver', 'liblinear'),
            penalty=params.get('penalty', 'l1'),
            random_state=2
        )



# PREDICTION AND EVALUATION ON VALIDATION DATASET

predictions_val = {}
print("Accuracy on Validation dataset")
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    predictions_val[name] = preds
    print('The Accuracy for {} is {}'.format(name,accuracy_score(predictions_val[name],y_val)))



# PREDICTION AND EVALUATION ON TEST DATASET
predictions_test = {}
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions_test[name] = preds
    print('The Accuracy for {} is {}'.format(name,accuracy_score(predictions_test[name],y_test)))


prediction_test = pd.DataFrame(predictions_test)



bagging_predictions = ['Extrovert' if x == 0 else 'Introvert' for x in predictions_test['Bagging']]



submission_final = pd.DataFrame({
    'Id': test_ids,
    'label': bagging_predictions    # or whichever model you choose
})
submission_final.to_csv("submission.csv", index=False)

