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



train=pd.read_csv("/kaggle/input/clear-data/train.csv")
test=pd.read_csv("/kaggle/input/clear-data/test.csv")


train.isnull().sum()


train.head()


train.shape


train.describe()


train.isnull().sum()


train.info()


'''train=train.drop('id',axis=1)
train1=train1.drop('id',axis=1)'''


train.shape


train.drop_duplicates(inplace=True)



train.shape


train.describe()


train.head()


train['Time_spent_Alone']=train['Time_spent_Alone'].astype(int)
test['Time_spent_Alone']=test['Time_spent_Alone'].astype(int)


'''# Social Engagement Score (interaction term)
train['Social_score'] = (train['Social_event_attendance'] + train['Going_outside'] + train['Friends_circle_size'])
# Introvert-Tendency Proxy
train['Introvert_score'] = (train['Time_spent_Alone'] - train['Social_score'])
train['Inp']=train['Introvert_score']-train['Post_frequency']
train['set']=train['Social_event_attendance']-train['Time_spent_Alone']
train['In_ex']=train['Stage_fear']+train['Drained_after_socializing']

# Social Engagement Score (interaction term)
test['Social_score'] = (test['Social_event_attendance'] + test['Going_outside'] + test['Friends_circle_size'])
# Introvert-Tendency Proxy
test['Introvert_score'] = (test['Time_spent_Alone'] - test['Social_score'])
test['Inp']=test['Introvert_score']-test['Post_frequency']
test['set']=test['Social_event_attendance']-test['Time_spent_Alone']
test['In_ex']=test['Stage_fear']+test['Drained_after_socializing']'''


'''def team(a):
    if a<-1:
        return 0
    else:
        return 1
train['team']=train['set'].apply(team)
test['team']=test['set'].apply(team)'''
        


test.head()


train.head()


#categorical_cols=['Stage_fear','Drained_after_socializing']
# One-hot encode and update the original DataFrame
#train = pd.get_dummies(train, columns=categorical_cols)
#train
#test = pd.get_dummies(test, columns=categorical_cols)
#test


X=train.drop(['Personality','id'],axis=1)

y_t=train['Personality']



y=le.fit_transform(y_t)
y


x=test.drop('id',axis=1)


scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

x_scale = scaler.transform(x)




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



import xgboost as xgb
# Initialize models
xgb_model = xgb.XGBClassifier(**study_xgb.best_params)
    
'''max_depth= 4,
    learning_rate= 0.15483589148235946,
    n_estimators= 350,
    gamma= 1.9580710041703155,
    min_child_weight= 8,
    subsample=0.9863208156915935,
    colsample_bytree= 0.7423945009147257,
    reg_alpha= 1.7398742129657665,
    reg_lambda= 6.3942304254815605
)'''


rf_model = RandomForestClassifier(**study_rf.best_params)

'''n_estimators=126,
                                  max_depth=11,
                                  min_samples_split=15,
                                  min_samples_leaf=16,
                                  max_features='log2')'''

# Initialize the VotingClassifier (Majority Voting)
voting_model = VotingClassifier(
    estimators=[('xgb', xgb_model),
                
               ('RandomForestClassifier', rf_model)
               ],
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



accuracy_score(y,final_predictions)


final_predictions = voting_model.predict(x_scale)
final_predictions



pred=le.inverse_transform(final_predictions)
pred


output = pd.DataFrame({'id': test.id, 'Personalities': pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


sub=pd.read_csv("submission.csv")
sub

