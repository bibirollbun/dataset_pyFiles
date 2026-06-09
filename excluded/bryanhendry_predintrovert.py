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


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import optuna
import xgboost as xgb
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head()


train.isna().sum()


train.shape


train.describe()


train.info()


categorical_features = ['Stage_fear', 'Drained_after_socializing']
features_with_NaN = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']


ordinal_encoder = OrdinalEncoder()

train[categorical_features] = ordinal_encoder.fit_transform(train[categorical_features])
test[categorical_features] = ordinal_encoder.transform(test[categorical_features])


imputer = KNNImputer(n_neighbors=1)

train[features_with_NaN] = imputer.fit_transform(train[features_with_NaN])
test[features_with_NaN] = imputer.transform(test[features_with_NaN])


train.info()


train.isna().sum()


train.head()


train['Time_spent_Alone']=train['Time_spent_Alone'].astype(int)
test['Time_spent_Alone']=test['Time_spent_Alone'].astype(int)


X=train.drop(['Personality','id'],axis=1)

y_t=train['Personality']
y=le.fit_transform(y_t)
y


X_test=test.drop('id',axis=1)


scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

x_scale_test = scaler.transform(X_test)


def objective_xgb(trial):
    params = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
    }

    model = xgb.XGBClassifier(**params, use_label_encoder=False)
    scores = cross_val_score(model, X_scale, y, cv=3, scoring='accuracy')
    return scores.mean()

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=50)

print("Best XGBoost parameters:", study_xgb.best_params)


def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_scale, y, cv=3, scoring="accuracy")
    return scores.mean()

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=50)

print("Best RandomForest parameters:", study_rf.best_params)


xgb_model = xgb.XGBClassifier(**study_xgb.best_params)
    
''''max_depth': 7, 'learning_rate': 0.07523960631118792, 'n_estimators': 453, 'gamma': 3.888615329560457, 'min_child_weight': 2, 'subsample': 0.5593363336264556, 'colsample_bytree': 0.941688103994737, 'lambda': 0.2984868135612588, 'alpha': 0.08515570583426206'''


rf_model = RandomForestClassifier(**study_rf.best_params)

'''n_estimators=126,
                                  max_depth=11,
                                  min_samples_split=15,
                                  min_samples_leaf=16,
                                  max_features='log2')'''

# Initialize the VotingClassifier (Majority Voting)
voting_model = VotingClassifier(
    estimators=[('xgb', xgb_model),('RandomForestClassifier', rf_model)],
    voting='soft'    
)

# Initialize Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

accuracies=[]
for train_index, test_index in skf.split(X_scale, y):
    # Split the dataset into training and testing sets for each fold
    X_train, X_test = X_scale[train_index], X_scale[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the Voting Classifier model
    voting_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = voting_model.predict(X_test)
    
    # Evaluate the accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Output the average accuracy across all folds
print(f"Average Accuracy across {n_splits} folds: {np.mean(accuracies):.4f}")

# Final Classification Report (using the whole dataset, but typically, you may compute on the final model)
voting_model.fit(X_scale, y)  # Fit on the whole data for final model
final_predictions = voting_model.predict(X_scale)
print("Final Classification Report on Entire Dataset:\n", classification_report(y, final_predictions))


final_predictions = voting_model.predict(x_scale_test)
final_predictions


pred=le.inverse_transform(final_predictions)
pred


output = pd.DataFrame({'id': test.id, 'Personalities': pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


sub=pd.read_csv("submission.csv")
sub




