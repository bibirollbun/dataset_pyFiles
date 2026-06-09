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


df_train =pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.info(),df_test.info()


df_train.head(3),df_test.head(3)


df_train.isnull().sum(),df_test.isnull().sum()


 for col in df_train.columns:
        if df_train[col].dtype == 'object':  # Categorical column
            df_train[col].fillna(df_train[col].mode()[0], inplace=True)
        else:  # Numeric column
            df_train[col].fillna(df_train[col].mean(), inplace=True)


 for col in df_test.columns:
        if df_test[col].dtype == 'object':  # Categorical column
            df_test[col].fillna(df_test[col].mode()[0], inplace=True)
        else:  # Numeric column
            df_test[col].fillna(df_test[col].mean(), inplace=True)


df_train.isnull().sum(),df_test.isnull().sum()


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
encoder.fit(df_train[['stage_fear', 'drained_after_socializing']])

# Transform both
df_train[['stage_fear', 'drained_after_socializing']] = encoder.transform(df_train[['stage_fear', 'drained_after_socializing']])
df_test[['stage_fear', 'drained_after_socializing']] = encoder.transform(df_test[['stage_fear', 'drained_after_socializing']])


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_train['personality'] = le.fit_transform(df_train['personality'])


df_train.head(3)


col_to_scale=['time_spent_alone','social_event_attendance','going_outside','friends_circle_size','post_frequency']


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit the scaler on train data and transform both train and test
df_train[col_to_scale] = scaler.fit_transform(df_train[col_to_scale])
df_test[col_to_scale] = scaler.transform(df_test[col_to_scale])


df_train.head(3)


X=df_train.drop(columns=['personality'])
y=df_train['personality']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2, random_state =42,stratify=y)


import optuna
import xgboost as xgb
from sklearn.metrics import classification_report,accuracy_score
def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


best_params = study.best_params


final_model = xgb.XGBClassifier(
    **best_params,
    objective='binary:logistic',
    use_label_encoder=False,
    verbosity=0
)

final_model.fit(X_train, y_train)


from sklearn.metrics import classification_report,accuracy_score

val_preds = final_model.predict(X_test)
print("Validation Accuracy:", accuracy_score(y_test, val_preds))
print(classification_report(y_test, val_preds))


test_probs = final_model.predict(df_test)


test_labels = le.inverse_transform(test_probs)


test_labels


submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': test_labels
})
submission.to_csv('submission.csv', index=False)



submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

