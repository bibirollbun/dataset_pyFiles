# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train_df['source'] = 'Train'
test_df['source'] = 'Test'
df = pd.concat([train_df, test_df], ignore_index=True)
#Combining datasets to apply preprocessing steps uniformly


df.isna().sum()


df.head(5)


df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
num_cols=df[['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']]
cat_cols=df[['Stage_fear','Drained_after_socializing']]


for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())
for col in cat_cols:
    mode_value = df[col].mode(dropna=True)[0]  
    df[col] = df[col].fillna(mode_value)
#Try KNN imputer if low accuracy


df['Total_social_activity']=df['Going_outside']+df['Social_event_attendance'] #Total hours the person put in for socialising
df['Loner'] = ((df['Friends_circle_size'] <= 3) & (df['Post_frequency'] <= 2)).astype(int) #New feature which determines if a person is a loner
float_cols = df.select_dtypes(include='float').columns
df[float_cols] = df[float_cols].astype(int)



train_processed = df[df['source'] == 'Train'].drop(columns='source')
train_processed['Personality'] = train_processed['Personality'].map({'Introvert': 0, 'Extrovert': 1})
test_processed = df[df['source'] == 'Test'].drop(columns='source')


from sklearn.model_selection import train_test_split, KFold
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

X=train_processed.drop(columns=['Personality','id'])
y=train_processed['Personality']
cat_cols = X.select_dtypes(include='object').columns.tolist()


import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'iterations': 500,
        'random_state': 42,
        'verbose': 0,
        'cat_features': cat_cols
    }

    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X, y, scoring='accuracy', cv=3).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best Trial:")
print(study.best_trial.params)


best_params = study.best_trial.params
best_params.update({
    'cat_features': cat_cols,
    'verbose': 100,
    'random_state': 42
})

final_model = CatBoostClassifier(**best_params)
final_model.fit(X, y)


X_test=test_processed.drop(columns=['Personality','id'])


test_preds = final_model.predict(X_test)


label = {0: 'Introvert', 1: 'Extrovert'}
test_preds = [label[p] for p in test_preds]
submission = pd.DataFrame({
    'id': test_df['id'], 
    'Personality': test_preds
})
submission.to_csv('submission.csv', index=False)





