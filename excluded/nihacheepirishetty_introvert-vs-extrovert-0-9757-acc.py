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
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import optuna
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
#from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np




train=pd.read_csv("/kaggle/input/clear-data/train.csv")
test=pd.read_csv("/kaggle/input/clear-data/test.csv")


train.head()


test.head()


train.isnull().sum()


test.isnull().sum()


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


train.isnull().sum()


test.isnull().sum()


for feature in numerical_features:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=train[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {train[feature].skew():.2f}")
    print(f"Number of Missing Values: {train[feature].isnull().sum()}")


correlation_matrix = train[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


train.columns


train.info()


train['Time_spent_Alone']=train['Time_spent_Alone'].astype(int)
test['Time_spent_Alone']=test['Time_spent_Alone'].astype(int)


train.info()


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
train.head()


numerical_features = train.select_dtypes(include=['number']).columns


correlation_matrix = train[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


train.describe().T


train.info()


X=train.drop(['Personality','id'],axis=1)

y=train['Personality']


y_t=le.fit_transform(y)
y_t


x=test.drop(['id'],axis=1)


scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

x_scale = scaler.fit_transform(x)




X_train, X_valid, y_train, y_valid = train_test_split(X_scale, y_t, test_size=0.2, random_state=42)


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



'''from sklearn.svm import SVC

def objective_svm(trial):
    params = {
        'C': trial.suggest_float('C', 1e-2, 100, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    }

    model = SVC(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)

study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective_svm, n_trials=50)
best_params_svm = study_svm.best_params'''



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



import lightgbm as lgb

def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 7, 127),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0)
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=50)
best_params_lgb = study_lgb.best_params



'''from sklearn.linear_model import LogisticRegression

def objective_logreg(trial):
    params = {
        'C': trial.suggest_float('C', 1e-3, 100, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
    }

    # Avoid incompatible combinations
    if params['penalty'] == 'l1' and params['solver'] == 'saga':
        return 0.0  # skip incompatible combination

    model = LogisticRegression(**params, max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)

study_logreg = optuna.create_study(direction='maximize')
study_logreg.optimize(objective_logreg, n_trials=50)
best_params_logreg = study_logreg.best_params'''



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







# ========== Train XGBoost ==========
xgb = XGBClassifier(**best_params_xgb)
'''use_label_encoder=False, eval_metric='logloss',gamma=0,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=3,
    subsample=1.0,
    n_estimators=500,        
    tree_method='hist',      
    n_jobs=-1,               
    verbosity=1 ,
    enable_categorical=True# Optional: show training progress
)'''
#xgb_model.fit(X_scale,y_t)
# ========== Train LightGBM ==========
lgbm= LGBMClassifier(**best_params_lgb)
    
'''learning_rate=0.05,        
    num_leaves=31,              
    max_depth=-1,              
    min_child_samples=20,       
    subsample=0.8,              
    colsample_bytree=0.8,       
    reg_alpha=0.0,              
    reg_lambda=0.0,             
    n_estimators=100,          
    random_state=42 ) '''

#lgb_model.fit(X_scale,y_t)
#svc_model=SVC(**best_params_svm)
'''kernel='rbf',
            C=1.0,
            probability=True,  # Required for predict_proba
            random_state=42)'''
#svc_model.fit(X_scale,y_t)
cat=CatBoostClassifier(**best_params_catboost)
'''iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=0,
        random_seed=42,
        early_stopping_rounds=50
    )'''
#cat_model.fit(X_scale,y_t)

rf=RandomForestClassifier(**best_params_rf)
'''n_estimators=126,
    max_depth=11,
    min_samples_split=15,
    min_samples_leaf=16,
max_features='log2')'''
#rf_model.fit(X_scale,y_t)
#lg_model=LogisticRegression(**best_params_logreg)
#lg_model.fit(X_scale,y_t)


'''# ========== Predict Probabilities ==========
xgb_labels = xgb_model.predict(x_scale)
lgb_labels = lgb_model.predict(x_scale)
rf_labels=rf_model.predict(x_scale)
#svc_labels=svc_model.predict(x_scale)
cat_labels=cat_model.predict(x_scale)
#lg_labels=lg_model.predict(x_scale)

avg_labels = mode([xgb_labels, 
                   lgb_labels,
                   rf_labels,
                  #svc_labels,
                  cat_labels
                  #lg_labels
                  ], axis=0)[0].flatten()'''

# ========== Average the Probabilities ==========
#avg_probs = (xgb_probs + lgb_probs) / 2


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
#from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np



# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#

# === Option 2: Stacking Classifier ===
base_learners = [
    ('xgb', xgb),
    ('lgbm', lgbm),
    ('cat', cat),
    ('rf', rf)
]

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=True  # Optionally use original features with base learner predictions
)

print("\n=== StackingClassifier Cross-Validation ===")
stacking_scores = []

for train_idx, test_idx in skf.split(X_scale, y_t):
    X_train, X_test = X_scale[train_idx], X_scale[test_idx]
    y_train, y_test = y_t[train_idx], y_t[test_idx]
    stacking_clf.fit(X_train, y_train)
    preds = stacking_clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    stacking_scores.append(acc)

print("Stacking Accuracy per Fold:", stacking_scores)
print("Stacking Average Accuracy:", np.mean(stacking_scores))



stacking_clf.fit(X_scale,y_t)





avg_labels=stacking_clf.predict(x_scale)
avg_labels


pred=le.inverse_transform(avg_labels)
pred


output = pd.DataFrame({'id': test.id, 'Personalities': pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")



sub=pd.read_csv("submission.csv")
sub.head()

