# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df_train.head()


df_train.describe()


df_train.info()


df_train.isnull().sum()


df_test.isnull().sum()


### Dropping id col
df_train = df_train.drop('id', axis = 1)
df_test = df_test.drop('id', axis = 1)


df_train.head()


df_train.describe().T


len(df_train), df_train.columns


df_train.head()


cat_cols = df_train.select_dtypes(exclude=[np.number]).columns
num_cols = df_train.select_dtypes(exclude = 'object').columns


num_cols


cat_cols


for col in cat_cols:
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df_train, x=col, hue='Personality')
    plt.title(f'{col} count by Personality')
    plt.show()


for col in num_cols:
    df_train[col] = df_train[col].fillna(df_train[col].mean())
    df_test[col] = df_test[col].fillna(df_train[col].mean())


corr = df_train[num_cols].corr()
sns.heatmap(corr)
plt.show()


### new feature based on strong correlation
df_train['Social_activity_score'] = (df_train['Friends_circle_size'] + df_train['Post_frequency'] + df_train['Social_event_attendance'])
df_test['Social_activity_score'] = (df_test['Friends_circle_size'] + df_test['Post_frequency'] + df_test['Social_event_attendance'])


df_train.isnull().sum()


for col in cat_cols:
    # mode_val = df_train[col].mode()[0]
    # df_train[col] = df_train[col].fillna(mode_val)
    df_train[col] = df_train[col].fillna('Unknown')
    if col != 'Personality':
        # df_test[col] = df_test[col].fillna(mode_val)
        df_test[col] = df_test[col].fillna('Unknown')


df_train.isnull().sum(), df_test.isnull().sum()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['Stage_fear', 'Drained_after_socializing']:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])


df_train['Personality'].value_counts()


## Encoding the class
df_train['Personality'] = df_train['Personality'].map({'Extrovert': 0, 'Introvert': 1})


y = df_train['Personality']
X = df_train.drop('Personality', axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


space = {
    'max_depth': hp.choice('max_depth', range(3, 12)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', range(50, 500)),
    'gamma': hp.uniform('gamma', 0, 5),
    'min_child_weight': hp.uniform('min_child_weight', 1, 10),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
}



def objective(params):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        **params
    )
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    return {'loss': -score, 'status': STATUS_OK}



trials = Trials()

xgb_best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,  
    trials=trials,
)



xgb_best


best_params = {
    'max_depth': [3,4,5,6,7,8,9,10,11][xgb_best['max_depth']],
    'n_estimators': [i for i in range(50, 500)][xgb_best['n_estimators']],
    'learning_rate': xgb_best['learning_rate'],
    'gamma': xgb_best['gamma'],
    'min_child_weight': xgb_best['min_child_weight'],
    'subsample': xgb_best['subsample'],
    'colsample_bytree': xgb_best['colsample_bytree'],
}

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **best_params)
xgb_model.fit(X_train, y_train)



y_pred = xgb_model.predict(X_test)


print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()


space = {
    'n_estimators': hp.choice('n_estimators', range(50, 301, 10)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 1.0),
    'max_depth': hp.choice('max_depth', range(1, 11)),
}


def objective(params):
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=params['max_depth']),
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        random_state=42
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    return {'loss': -score, 'status': STATUS_OK}



trials = Trials()
ada_best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials,
)


ada_best['n_estimators'] = list(range(50, 301, 10))[ada_best['n_estimators']]
ada_best['max_depth'] = list(range(1, 11))[ada_best['max_depth']]
print(best)


adaboost_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=ada_best['max_depth']),
    n_estimators=ada_best['n_estimators'],
    learning_rate=ada_best['learning_rate'],
    random_state=42
)

adaboost_model.fit(X_train, y_train)



y_pred = adaboost_model.predict(X_test)


print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Extrovert", "Introvert"]).plot()


test_id = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')['id']


# test_labels = final_model.predict(df_test)
test_labels = xgb_model.predict(df_test)


test_labels = ['Extrovert' if pred == 0 else 'Introvert' for pred in test_labels]
test_labels[:5]


submission = pd.DataFrame({'id' : test_id, 'Personality' : test_labels})
submission.to_csv('submission.csv', index = False)




