# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier

import joblib

import warnings 
warnings.filterwarnings('ignore')


data_tr = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
data_tr.head(3)


data_tr.shape


data_te = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
data_te.head(3)


data_te.shape


data = pd.concat([data_tr,data_te],axis=0)
data


data.replace('unknown', np.nan, inplace=True)


sns.heatmap(data.isnull())


data.drop(['poutcome','contact'],axis =1 ,inplace = True)


data.isnull().sum()


data.info()


data.describe().T


imputer = SimpleImputer(strategy = 'most_frequent')

data[['job','education']] = imputer.fit_transform(data[['job','education']])


categorical_cols = data.select_dtypes(include=['object']).columns
categorical_cols


for i in categorical_cols :
    print(i)
    print(data[i].unique())
    print('\n')


encoder = OrdinalEncoder()

data[categorical_cols] = encoder.fit_transform(data[categorical_cols])
#data.drop(categorical_cols,inplace=True,axis=1)
data


data_te=data[data['y'].isnull()]
data_tr=data[~data['y'].isnull()]


data_tr.head(3)


data_tr.shape


data_te.drop('y',axis = 1,inplace = True)
data_te.head(3)


data_te.shape


X = data_tr.drop('y',axis = 1)
y = data_tr['y']

X.shape , y.shape


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)


# XGBClassifier

param_dist_xgb = {
    'n_estimators': [100, 200, 300, 400,500],
    'max_depth': [3, 5, 7, 10,13, 15],
    'learning_rate': [0.01, 0.05, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'min_child_weight': [1, 5, 7],
    'gamma': [0, 1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],     
    'reg_lambda': [1, 1.5, 2.0]     
}

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

xgb = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist',       
    verbosity=0,
    random_state=42
)

random_search_xgb = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist_xgb,
    n_iter=10,                
    cv=skf,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy'
)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

random_search_xgb.fit(X_train, y_train_encoded)


best_xgb = random_search_xgb.best_estimator_
print("Best Parameters:\n", random_search_xgb.best_params_)

y_pred_xgb = best_xgb.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test_encoded, y_pred_xgb))


# CatBoostClassifier

param_dist_cat = {
    'iterations': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'depth': [3, 5, 7, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128],
    'random_strength': [1, 5, 10],
    'bagging_temperature': [0, 0.5, 1],
    'scale_pos_weight': [1, 2, 3],  
    'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS']
}

X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

cat_model = CatBoostClassifier(verbose=0, random_state=42, loss_function='Logloss')

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_cat = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=param_dist_cat,
    n_iter=5,
    scoring='f1_weighted',
    n_jobs=-1,
    cv=skf,
    verbose=1,
    random_state=42
)

random_search_cat.fit(
    X_train_cat,
    y_train_cat,
    eval_set=(X_val_cat, y_val_cat),
    early_stopping_rounds=30
)


best_cat = random_search_cat.best_estimator_
y_pred_cat = best_cat.predict(X_test)

print("Best Parameters:\n", random_search_cat.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred_cat))


xgb_best = XGBClassifier(**random_search_xgb.best_params_)
cat_best = CatBoostClassifier(**random_search_cat.best_params_, verbose=0)

estimators = [
    ('lr',LogisticRegression()),
    ('xgb', xgb_best),
    ('cat', cat_best)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

stack.fit(X, y)


accuracy_score(y_train,stack.predict(X_train)) , accuracy_score(y_test,stack.predict(X_test))


test_predict = stack.predict(data_te)
test_predict


Submission = pd.DataFrame({'id':data_te['id'],'y':test_predict})
Submission


Submission.to_csv('submission.csv',index = False)

