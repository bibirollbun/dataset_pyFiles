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


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df.head()


df.info()


df.drop(columns=['id'], inplace=True)


df.dropna(inplace=True)


df['Drained_after_socializing'].value_counts()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


cat_col = df.select_dtypes(include='object').columns.tolist()
print(cat_col)


for col in cat_col:
    df[col] = le.fit_transform(df[col])


df.head()


df.isnull().sum()


df.info()


test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
test.info()


sub_id = test['id']


print(sub_id)


test.drop(columns=['id'], inplace=True)


test['Time_spent_Alone'] = test['Time_spent_Alone'].fillna(test['Time_spent_Alone'].mean())


test['Stage_fear'] = test['Stage_fear'].fillna(test['Stage_fear'].mode()[0])
test['Drained_after_socializing'] = test['Drained_after_socializing'].fillna(test['Drained_after_socializing'].mode()[0])


test['Stage_fear'].value_counts()


test['Stage_fear'] = le.fit_transform(test['Stage_fear'])
test['Drained_after_socializing'] = le.fit_transform(test['Drained_after_socializing'])


test.info()


print(cat_col)


num_col = test.select_dtypes(include='float64').columns.tolist()
print(num_col)


for col in num_col:
    test[col] = test[col].fillna(test[col].mean())


test.info()


test.dropna(inplace=True)


test.info()


X = df.drop(columns=['Personality'])
y = df['Personality']


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)


y_train.shape


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


class ML_Model:
    def __init__(self, x, y, model):
        self.x = x
        self.y = y
        self.model = model

    def train(self):
        self.model.fit(self.x, self.y)

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, test_x, test_y):
        pred = self.model.predict(test_x)
        print(f"Accuracy: {accuracy_score(test_y, pred)}")
        print(f"Precision: {precision_score(test_y, pred, average='binary')}")
        print(f"Recall: {recall_score(test_y, pred, average='binary')}")
        print(f"Confusion Matrix:\n{confusion_matrix(test_y, pred)}")


lr = ML_Model(X_train,y_train,LogisticRegression())
dt = ML_Model(X_train,y_train,DecisionTreeClassifier())
rf = ML_Model(X_train,y_train,RandomForestClassifier(n_estimators = 500))
knn = ML_Model(X_train,y_train,KNeighborsClassifier())


import optuna
def objective(trial):
    # Tham số cho XGBoost
    xgb_params = {
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
        'gamma': trial.suggest_float('xgb_gamma', 0.0, 3.0),
        'max_depth': trial.suggest_int('xgb_max_depth', 6, 15),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.1, 2.0),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.1, 2.0),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
        'objective': 'multi:softprob',
        'num_class': len(set(y_train)),  # nếu là multi-class
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1
    }


    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 1000),
        'criterion': trial.suggest_categorical('rf_criterion', ["gini", "entropy", "log_loss"]),
        'max_depth': trial.suggest_int('rf_max_depth', 6, 15),
        'min_samples_split': trial.suggest_float('rf_min_samples_split', 0.0, 1.0),
        'max_features': trial.suggest_categorical('rf_max_features', ["sqrt", "log2", None]),
        'random_state': 42,
        'n_jobs': -1
    }


    w_rf = trial.suggest_float("weight_rf", 0.1, 2.0)
    w_xgb = trial.suggest_float("weight_xgb", 0.1, 2.0)

    xgb_model = XGBClassifier(**xgb_params)
    rf_model = RandomForestClassifier(**rf_params)

    voting_clf = VotingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model)],
        voting="soft",
        weights=[w_rf, w_xgb],
        n_jobs=-1
    )

    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# Tạo study và chạy tối ưu
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_trial.params)
print("Best score:", study.best_value)


lr.train()


dt.train()
rf.train()
knn.train()


rf_params = {
    'n_estimators': 626,
    'criterion': 'log_loss',
    'max_depth': 13,
    'min_samples_split': 0.8321156372191643,
    'max_features': 'log2',
    'random_state': 42,
    'n_jobs': -1
}
xgb_params = {
    'learning_rate': 0.07297004610144715,
    'gamma': 1.7253356371825426,
    'max_depth': 7,
    'min_child_weight': 4,
    'subsample': 0.8714485363200863,
    'colsample_bytree': 0.42344206351773384,
    'reg_lambda': 0.7342401441632284,
    'reg_alpha': 0.5546660715967193,
    'n_estimators': 271,
    'objective': 'multi:softprob',
    'num_class': len(set(y_train)),
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'n_jobs': -1
}

rf_model = RandomForestClassifier(**rf_params)
xgb_model = XGBClassifier(**xgb_params)

voting_clf = VotingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model)],
        voting="soft",
        weights=[1.3281795373334073, 1.5235020381631403],
        n_jobs=-1
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)


lr.evaluate(X_test, y_test)


dt.evaluate(X_test, y_test)


rf.evaluate(X_test, y_test)


knn.evaluate(X_test, y_test)


test.info()


print(lr.predict(test))


print(dt.predict(test))
print(rf.predict(test))
print(knn.predict(test))


sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
sub.info()


test.info()


sub_id


preds = voting_clf.predict(test)  # NumPy array

pred_df = pd.DataFrame(preds, columns=['Personality'])

# ID column
sub_id_df = sub_id.reset_index(drop=True).to_frame(name='id')

# Combine both
final_df = pd.concat([sub_id_df, pred_df], axis=1)
final_df.head()


perso = {0:'Extrovert',1:'Introvert'}
final_df['Personality'] = final_df['Personality'].map(perso)
final_df.head()


final_df[['id', 'Personality']].to_csv("submit_fixed.csv", index=False)


final_df.to_csv("/kaggle/working/submit.csv")

