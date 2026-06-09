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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics 
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric


# Loading the dataset
df_train = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
df_test = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
df_sub = pd.read_csv('/kaggle/input/multiclassificationtask/sample_submission.csv')


df_train.columns


df_train.shape


df_train.info()


df_train.describe()


null = df_train.isnull().sum().sum()
print(f'Null Count in Train: {null}')


duplicates = df_train.duplicated().sum()
print(f'Duplicates: {duplicates}')


df_train.head()


df_test.head()


df_sub.head()


df_train.shape,df_test.shape,df_sub.shape


df_train.isnull().sum()


df_train = df_train.drop(['id'],axis=1)


#1. Drug ustunidan nan qiymatlarni tashlab yuboramiz
# df_train = df_train.dropna(subset=['Drug'])


df_train.info()


df_train['Status'].value_counts()


df_train = df_train[df_train['Status'] != 'Y']


df_train['Status'].value_counts()


# split dataset X, y
X = df_train.drop('Status', axis=1)
y = df_train['Status']
y


encoder = LabelEncoder()
y = encoder.fit_transform(y)


y[:100]


class_labels = encoder.classes_
class_labels


X.info()


# Pipline
cat_attributes = X.select_dtypes(include=['object']).columns.to_list()
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore')),
])


num_cols = X.select_dtypes(include=['float64']).columns.to_list()
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer([
    ('categorical', cat_pipeline, cat_attributes),
    ('numerical', num_pipeline, num_cols)
])


X_prepared = preprocessor.fit_transform(X)
X_prepared


X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)


# contain all estimators in the function
def evaluate_model(y_test, y_pred, y_proba, name):
    # Model estimation 
    print(f"\n--- {name} ---")
    print(f"Model accuracy: {metrics.accuracy_score(y_test,y_pred)*100:.1f}%")

    print("Log Loss:", metrics.log_loss(y_test, y_proba, labels=[0, 1, 2]))
    print(f"Classification Report:\n{metrics.classification_report(y_test, y_pred, zero_division=0)}")
    print('='*50)
    
    # confusion matrix
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True,fmt="g")
    plt.show()


# SVC
SVC_model = SVC(kernel='linear', probability=True, decision_function_shape='ovo')
SVC_model.fit(X_train, y_train)

y_pred = SVC_model.predict(X_test)
y_proba = SVC_model.predict_proba(X_test)
evaluate_model(y_test, y_pred, y_proba, 'SVC')


# Logistic regression
LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(X_train, y_train)

y_pred = LR_model.predict(X_test)
y_proba = LR_model.predict_proba(X_test)
evaluate_model(y_test, y_pred, y_proba, 'Logistic regression')


# Random Forest
RF_model = RandomForestClassifier(n_estimators=200, random_state=42)
RF_model.fit(X_train, y_train)

y_pred = RF_model.predict(X_test)
y_proba = RF_model.predict_proba(X_test)
evaluate_model(y_test, y_pred, y_proba, 'Random Forest')


# XGBClassifier
XG_model = XGBClassifier()
XG_model.fit(X_train, y_train)

y_pred = XG_model.predict(X_test)
y_proba = XG_model.predict_proba(X_test)
evaluate_model(y_test, y_pred, y_proba, 'XGBClassifier')


# LGBMClassifier
lgbParams = {'n_estimators': 1000,
             'max_depth': 25, 
             'learning_rate': 0.025,
             'min_child_weight': 3.43,
             'min_child_samples': 216, 
             'subsample': 0.782,
             'subsample_freq': 4, 
             'colsample_bytree': 0.29, 
             'num_leaves': 21}
LG_model = LGBMClassifier(**lgbParams)
LG_model.fit(X_train, y_train)

y_pred = LG_model.predict(X_test)
y_proba = LG_model.predict_proba(X_test)
evaluate_model(y_test, y_pred, y_proba, 'LGBMClassifier')


CAT_model = CatBoostClassifier(eval_metric='AUC',learning_rate=0.022,iterations=1000)
CAT_model.fit(X_train, y_train)

y_pred = CAT_model.predict(X_test)
y_proba = CAT_model.predict_proba(X_test)
evaluate_model(y_test, y_pred, y_proba, 'CAT_model')


# KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors=22)
KNN_model.fit(X_train, y_train)

y_pred = KNN_model.predict(X_test)
y_proba = KNN_model.predict_proba(X_test)
evaluate_model(y_test, y_pred, y_proba, 'KNeighborsClassifier')


df_test.head(10)


ids = df_test['id']


X_submission = df_test


X_submission = X_submission.drop(['id'],axis=1)


X_submission.head(5)


test_df_prepared = preprocessor.transform(X_submission)


y_probs = LG_model.predict_proba(test_df_prepared)
y_probs


y_probs_clipped = np.clip(y_probs, 1e-15, 1 - 1e-15)
y_probs_clipped


y_probs_clipped[:, 0]


#Label encoder orqali ketmaketlik C=0, CL=1 va D=2 vormatida bo'ladi
submission = pd.DataFrame(y_probs_clipped, columns=[f'Status_{cls}' for cls in class_labels])
submission['id'] = df_test['id']
submission = submission[['id'] + [f'Status_{cls}' for cls in class_labels]]

submission.head(10)


submission.to_csv('LG3_submission.csv', index=False)

