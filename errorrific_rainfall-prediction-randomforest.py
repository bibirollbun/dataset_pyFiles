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


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


df.sample(1)


df = df.drop(columns=['id','day'])


from sklearn.model_selection import train_test_split
X = df.drop(columns=['rainfall'])
y = df['rainfall']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=23)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


import optuna
from sklearn.model_selection import cross_val_score


def objective(trail):
    n_estimators = trail.suggest_int("n_estimators",50,300)
    max_depth = trail.suggest_int("max_depth",2,30)
    min_samples_split = trail.suggest_int("min_samples_split",2,10)
    min_samples_leaf = trail.suggest_int("min_samples_leaf",1,10)
    max_features = trail.suggest_categorical("max_features",["sqrt","log2"])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf = min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)




# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_test.fillna(df_test.mean(), inplace = True)


df_test.shape


df_test.isnull().sum()


test_id = df_test['id']


df_test = df_test.drop(columns=["id","day"])


predictions = best_model.predict(df_test)


prob_class_1 = best_model.predict_proba(df_test)[:, 1]


prob_class_1


submission = pd.DataFrame({
    "id": test_id, 
    "rainfall": prob_class_1 
})

# Save submission file
submission.to_csv("submission.csv", index=False)




