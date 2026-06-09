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

from sklearn.model_selection import cross_val_score, StratifiedKFold


from catboost import CatBoostClassifier




train=pd.read_csv("/kaggle/input/clear-data/train.csv")
test=pd.read_csv("/kaggle/input/clear-data/test.csv")


train=train.drop('id',axis=1)


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


for feature in ['Stage_fear','Drained_after_socializing']:
    train[feature]=le.fit_transform(train[feature])
    test[feature]=le.fit_transform(test[feature])


train['Time_spent_Alone']=train['Time_spent_Alone'].astype(int)
test['Time_spent_Alone']=test['Time_spent_Alone'].astype(int)


# Social Engagement Score (interaction term)
train['Social_score'] = (train['Social_event_attendance'] + train['Going_outside'] + train['Friends_circle_size'])
# Introvert-Tendency Proxy
train['Introvert_score'] = (train['Time_spent_Alone'] - train['Social_score'])
train['Inp']=train['Introvert_score']-train['Post_frequency']
train['set']=train['Social_event_attendance']-train['Time_spent_Alone']
train['In_ex']=train['Stage_fear']+train['Drained_after_socializing']
train['tsd']=train['Time_spent_Alone']+train['Stage_fear']+train['Drained_after_socializing']

# Social Engagement Score (interaction term)
test['Social_score'] = (test['Social_event_attendance'] + test['Going_outside'] + test['Friends_circle_size'])
# Introvert-Tendency Proxy
test['Introvert_score'] = (test['Time_spent_Alone'] - test['Social_score'])
test['Inp']=test['Introvert_score']-test['Post_frequency']
test['set']=test['Social_event_attendance']-test['Time_spent_Alone']
test['In_ex']=test['Stage_fear']+test['Drained_after_socializing']
test['tsd']=test['Time_spent_Alone']+test['Stage_fear']+test['Drained_after_socializing']


def team(a):
    if a<-1:
        return 0
    else:
        return 1
train['team']=train['set'].apply(team)
test['team']=test['set'].apply(team)
        


def intro(a):
    if a>0:
        return 1
    else:
        return 0
train['intro']=train['In_ex'].apply(intro)
test['intro']=test['In_ex'].apply(intro)



def tsd_filter(a):
    if a==5:
        return 0
    elif a<5:
        return 0
    else:
        return 1
train['tsd_filter']=train['tsd'].apply(tsd_filter)
test['tsd_filter']=test['tsd'].apply(tsd_filter)



numerical_features = train.select_dtypes(include=['number']).columns


correlation_matrix = train[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


'''categorical_cols=['Stage_fear','Drained_after_socializing']
# One-hot encode and update the original DataFrame
train = pd.get_dummies(train, columns=categorical_cols)
train
test = pd.get_dummies(test, columns=categorical_cols)
test'''


X=train.drop(['Personality'],axis=1)

y_t=train['Personality']



y=le.fit_transform(y_t)
y


x=test.drop('id',axis=1)


scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

x_scale = scaler.transform(x)


X_train, X_valid, y_train, y_valid = train_test_split(X_scale, y, test_size=0.2, random_state=42)


import xgboost as xgb

def objective_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)
best_params_xgb = study_xgb.best_params


from sklearn.ensemble import RandomForestClassifier

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=50)
best_params_rf = study_rf.best_params



def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'verbose': 0
    }

    # Some bootstrap types need additional params
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)
study_catboost = optuna.create_study(direction='maximize')
study_catboost.optimize(objective_catboost, n_trials=50)
best_params_catboost = study_catboost.best_params






rf_model=RandomForestClassifier(**study_rf.best_params)
# Initialize Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# List to store the accuracies for each fold
accuracies = []

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X_scale, y):
    # Split the dataset into training and testing sets for each fold
    X_train, X_test = X_scale[train_index], X_scale[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the Voting Classifier model
    rf_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
# Output the average accuracy across all folds
print(f"Average Accuracy across {n_splits} folds: {np.mean(accuracies):.4f}")

# Final Classification Report (using the whole dataset, but typically, you may compute on the final model)
rf_model.fit(X_scale, y)  # Fit on the whole data for final model
final_predictions = rf_model.predict(X_scale)
print("Final Classification Report on Entire Dataset:\n", classification_report(y, final_predictions))
rf_model.fit(X_scale,y)
rf_labels=rf_model.predict(x_scale)


cat_model=CatBoostClassifier(**study_catboost.best_params)
# Initialize Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# List to store the accuracies for each fold
accuracies = []

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X_scale, y):
    # Split the dataset into training and testing sets for each fold
    X_train, X_test = X_scale[train_index], X_scale[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the Voting Classifier model
    cat_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = cat_model.predict(X_test)
    
    # Evaluate the accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
# Output the average accuracy across all folds
print(f"Average Accuracy across {n_splits} folds: {np.mean(accuracies):.4f}")

# Final Classification Report (using the whole dataset, but typically, you may compute on the final model)
cat_model.fit(X_scale, y)  # Fit on the whole data for final model
final_predictions = cat_model.predict(X_scale)
print("Final Classification Report on Entire Dataset:\n", classification_report(y, final_predictions))
cat_model.fit(X_scale,y)
cat_labels=cat_model.predict(x_scale)


xgb_model=xgb.XGBClassifier(**study_xgb.best_params)
# Initialize Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# List to store the accuracies for each fold
accuracies = []

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X_scale, y):
    # Split the dataset into training and testing sets for each fold
    X_train, X_test = X_scale[train_index], X_scale[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the Voting Classifier model
    xgb_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)
    
    # Evaluate the accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
# Output the average accuracy across all folds
print(f"Average Accuracy across {n_splits} folds: {np.mean(accuracies):.4f}")

# Final Classification Report (using the whole dataset, but typically, you may compute on the final model)
xgb_model.fit(X_scale, y)  # Fit on the whole data for final model
final_predictions = xgb_model.predict(X_scale)
print("Final Classification Report on Entire Dataset:\n", classification_report(y, final_predictions))
xgb_model.fit(X_scale,y)
xgb_labels=xgb_model.predict(x_scale)


avg_labels = mode([rf_labels,
                   xgb_labels,
                   cat_labels
                  ], axis=0)[0].flatten()
avg_labels


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def optimize_xgb(X_scale, y, n_trials=25):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scale, y, cv=cv, scoring='roc_auc')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params



from sklearn.ensemble import RandomForestClassifier

def optimize_rf(X_scale, y, n_trials=25):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        }
        model = RandomForestClassifier(**params, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scale, y, cv=cv, scoring='roc_auc')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params



from catboost import CatBoostClassifier

def optimize_cat(X_scale, y, n_trials=25):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': 0
        }
        model = CatBoostClassifier(**params, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scale, y, cv=cv, scoring='roc_auc')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params



best_xgb = optimize_xgb(X_scale, y, n_trials=30)
print("Best XGBoost params:", best_xgb)

best_rf = optimize_rf(X_scale, y, n_trials=30)
print("Best Random Forest params:", best_rf)

best_cat = optimize_cat(X_scale, y, n_trials=30)
print("Best CatBoost params:", best_cat)



xgb = XGBClassifier(**best_xgb)
rf = RandomForestClassifier(**best_rf, random_state=42)
cat = CatBoostClassifier(**best_cat, random_state=42)



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification

# Generate synthetic binary classification data
#X, y = make_classification(n_samples=1000, n_features=20, 
#                         n_informative=10, n_redundant=5, 
#                         random_state=42)

# Initialize models
#xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
#rf = RandomForestClassifier(n_estimators=100, random_state=42)
#cat = CatBoostClassifier(verbose=0, random_state=42)

# Meta model
meta_model = LogisticRegression()

# Create empty arrays to hold OOF predictions
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds_xgb = np.zeros((X.shape[0],))
oof_preds_rf = np.zeros((X.shape[0],))
oof_preds_cat = np.zeros((X.shape[0],))

# Final test predictions (optional)
# test_preds = np.zeros((X_test.shape[0],))  # if you have a test set

# Stacking loop
for train_idx, val_idx in skf.split(X_scale, y):
    X_train, X_val = X_scale[train_idx], X_scale[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train base models
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Predict validation fold
    oof_preds_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
    oof_preds_rf[val_idx] = rf.predict_proba(X_val)[:, 1]
    oof_preds_cat[val_idx] = cat.predict_proba(X_val)[:, 1]

# Stack predictions into meta-feature matrix
X_meta = np.vstack((oof_preds_xgb, oof_preds_rf, oof_preds_cat)).T

# Train meta model
meta_model.fit(X_meta, y)

# Evaluate ensemble
final_preds = meta_model.predict(X_meta)
final_probs = meta_model.predict_proba(X_meta)[:, 1]

print("✅ Accuracy:", accuracy_score(y, final_preds))
print("✅ ROC AUC:", roc_auc_score(y, final_probs))
print("✅ F1 Score:", f1_score(y, final_preds))



print("✅ Accuracy:", accuracy_score(y, final_preds))
print("✅ ROC AUC:", roc_auc_score(y, final_probs))
print("✅ F1 Score:", f1_score(y, final_preds))


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models=None, meta_model=None, n_folds=5):
        self.base_models = base_models or [
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            RandomForestClassifier(n_estimators=100, random_state=42),
            CatBoostClassifier(verbose=0, random_state=42)
        ]
        self.meta_model = meta_model or LogisticRegression()
        self.n_folds = n_folds
        self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    def fit(self, X_scale, y):
        self.base_models_ = [list() for _ in self.base_models]
        oof_preds = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in self.skf.split(X, y):
                cloned_model = self._clone_model(model)
                X_train, y_train = X_scale[train_idx], y[train_idx]
                X_val = X_scale[val_idx]

                cloned_model.fit(X_train, y_train)
                oof_preds[val_idx, i] = cloned_model.predict_proba(X_val)[:, 1]
                self.base_models_[i].append(cloned_model)

        self.meta_model.fit(oof_preds, y)
        return self

    def predict_proba(self, X_scale):
        meta_features = np.column_stack([
            np.mean([model.predict_proba(X_scale)[:, 1] for model in models], axis=0)
            for models in self.base_models_
        ])
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X_scale):
        return self.predict_proba(X_scale)[:, 1] > 0.5

    def _clone_model(self, model):
        return model.__class__(**model.get_params())




from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Create dummy dataset
#X, y = make_classification(n_samples=1000, n_features=20, 
 #                          n_informative=10, n_redundant=5, 
#                           random_state=42)

# Scale (if needed)
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_scale)

# Train ensemble
ensemble_model = StackingEnsemble()
ensemble_model.fit(X_scale, y)

# Predict on the same or new data
y_pred = ensemble_model.predict(x_scale)
#y_prob = ensemble_model.predict_proba(X_scaled)[:, 1]

# Evaluate
#print("✅ Accuracy:", accuracy_score(y, y_pred))
#print("✅ ROC AUC:", roc_auc_score(y, y_prob))
#print("✅ F1 Score:", f1_score(y, y_pred))



y_pred


import pandas as pd

# Convert y_pred to Series
pred = pd.Series(y_pred).map({True: 'Introvert', False: 'Extrovert'})
pred





#pred=y_pred.map({True:'Introvert',False:'Extrovert'})


#avg_labels=meta_model.predict(x_scale)
#avg_labels


#pred=le.inverse_transform(avg_labels)
#pred


output = pd.DataFrame({'id': test.id, 'Personalities': pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


sub=pd.read_csv("submission.csv")
sub

