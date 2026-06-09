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



train=pd.read_csv("/kaggle/input/clear-data/train.csv")
test=pd.read_csv("/kaggle/input/clear-data/test.csv")


#train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
#test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train.head()


numerical_features = train.select_dtypes(include=['number']).columns
categorical_cols = train.select_dtypes(exclude=['number']).columns

train[numerical_features] = train[numerical_features].fillna(train[numerical_features].mean())

for col in categorical_cols:
    if train[col].isnull().any():
        train[col] = train[col].fillna(train[col].mode()[0])


numerical_features = test.select_dtypes(include=['number']).columns
categorical_cols = test.select_dtypes(exclude=['number']).columns

test[numerical_features] = test[numerical_features].fillna(test[numerical_features].mean())

for col in categorical_cols:
    if test[col].isnull().any():
        test[col] = test[col].fillna(test[col].mode()[0])


train.drop_duplicates(inplace=True)


train['Time_spent_Alone']=train['Time_spent_Alone'].astype(int)
test['Time_spent_Alone']=test['Time_spent_Alone'].astype(int)


# Social Engagement Score (interaction term)
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
test['In_ex']=test['Stage_fear']+test['Drained_after_socializing']


def team(a):
    if a<-1:
        return 0
    else:
        return 1
train['team']=train['set'].apply(team)
test['team']=test['set'].apply(team)
        


categorical_cols=['Stage_fear','Drained_after_socializing']
# One-hot encode and update the original DataFrame
train = pd.get_dummies(train, columns=categorical_cols)
train
test = pd.get_dummies(test, columns=categorical_cols)
test


X=train.drop(['Personality'],axis=1)

y_t=train['Personality']



y=le.fit_transform(y_t)
y


scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

x_scale = scaler.transform(test)


import optuna
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset (you can replace this with your own)
#data = load_breast_cancer()
#X, y = data.data, data.target

def objective(trial):
    # Define hyperparameters to search
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0)
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(xgb.XGBClassifier(**param, use_label_encoder=False), X_scale, y, cv=cv, scoring="accuracy")

    return scores.mean()



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print("  Best hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")



# Initialize models
xgb_model = xgb.XGBClassifier(max_depth= 4,
    learning_rate= 0.15483589148235946,
    n_estimators= 350,
    gamma= 1.9580710041703155,
    min_child_weight= 8,
    subsample=0.9863208156915935,
    colsample_bytree= 0.7423945009147257,
    reg_alpha= 1.7398742129657665,
    reg_lambda= 6.3942304254815605,
    )


rf_model = RandomForestClassifier(n_estimators=126,
                                  max_depth=11,
                                  min_samples_split=15,
                                  min_samples_leaf=16,
                                  max_features='log2')

# Initialize the VotingClassifier (Majority Voting)
voting_model = VotingClassifier(
    estimators=[('xgb', xgb_model),
                
               ('RandomForestClassifier', rf_model)
               ],
    voting='hard'  
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

