!pip install catboost



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


import os
wdir=os.getcwd()
import matplotlib.pyplot as plt
path=wdir+'/loan/train.csv'
path2=wdir+'/loan/test.csv'
train = pd.read_csv("/kaggle/input/dataset/train.csv")
test=pd.read_csv("/kaggle/input/dataset/test.csv")


train.head()



train.info()


def preprocess_data(train, test):
    # Encoding categorical variables
    train['person_home_ownership'].replace({'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}, inplace=True)
    
    loan_intent_dummies = pd.get_dummies(train['loan_intent'], prefix='loan_intent')
    train = pd.concat([train, loan_intent_dummies], axis=1)
    train= train.drop('loan_intent',axis=1)
    
    train['loan_grade'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}, inplace=True)
    train['cb_person_default_on_file'].replace({'N': 0, 'Y': 1}, inplace=True)
    
    test['person_home_ownership'].replace({'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}, inplace=True)
    
    loan_intent_dummies = pd.get_dummies(test['loan_intent'], prefix='loan_intent')
    test = pd.concat([test, loan_intent_dummies], axis=1)
    test= test.drop('loan_intent',axis=1)
    
    test['loan_grade'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}, inplace=True)
    test['cb_person_default_on_file'].replace({'N': 0, 'Y': 1}, inplace=True)

    # Handling missing values and converting data types
    test['person_emp_length'] = test['person_emp_length'].astype(int)
    test['loan_int_rate'] = (test['loan_int_rate'] * 100).astype(int)
    test['loan_percent_income'] = (test['loan_percent_income'] * 100).astype(int)

    train['person_emp_length'] = train['person_emp_length'].astype(int)
    train['loan_int_rate'] = (train['loan_int_rate'] * 100).astype(int)
    train['loan_percent_income'] = (train['loan_percent_income'] * 100).astype(int)
    
    return train, test


train, test = preprocess_data(train, test)


train.head()



X = train.drop(columns=['loan_status'])
y = train['loan_status']


catboost_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.08114394459649094,
    'iterations': 1000,
    'depth': 6,
    'random_strength': 0,
    'l2_leaf_reg': 0.7047064221215757,
    'task_type': 'CPU',
    'random_seed': 42,
    'verbose': False
}


cv = StratifiedKFold(5, shuffle=True, random_state=0)
cv_splits = cv.split(X, y)


%%time
scores = []
test_preds = []
print(scores)



X_test_pool = Pool(test, cat_features=X.columns.values)



%%time
for i, (train_idx, val_idx) in enumerate(cv_splits):
    model = CatBoostClassifier(**catboost_params)
    
    X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
    y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
    
    X_train_pool = Pool(X_train_fold, y_train_fold, cat_features=X.columns.values)
    X_valid_pool = Pool(X_val_fold, y_val_fold, cat_features=X.columns.values)

    model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=False, early_stopping_rounds=200)
    
    val_pred = model.predict_proba(X_valid_pool)[:, 1]
    score = roc_auc_score(y_val_fold, val_pred)
    scores.append(score)
    
    print(f'Fold {i + 1} roc_auc_score: {score}')
    
    test_pred = model.predict_proba(X_test_pool)[:, 1]
    test_preds.append(test_pred)
    



print(f'Cross-validated roc_auc_score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max roc_auc_score: {np.max(scores):.3f}')
print(f'Min roc_auc_score: {np.min(scores):.3f}')


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='g', label='ROC AUC score')
plt.title('ROC AUC Score Across Folds')
plt.xlabel('Fold Number')
plt.ylabel('ROC AUC Score')
plt.xticks(range(1, len(scores) + 1))
plt.grid(True)
plt.legend()
plt.show()


final_pred_test = np.mean(test_preds, axis=0)

submission = test[['id']].copy()
submission['loan_status'] = final_pred_test  


print(np.shape(submission))



submission.to_csv('loancatb.csv', index=False)

