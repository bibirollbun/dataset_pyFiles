# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             classification_report)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import optuna
from sklearn.ensemble import RandomForestClassifier
import warnings

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
warnings.filterwarnings("ignore")


sample_data_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
train_data_path = '/kaggle/input/playground-series-s5e7/train.csv'
test_data_path = '/kaggle/input/playground-series-s5e7/test.csv'

sample_sub = pd.read_csv(sample_data_path)
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)


simple_imputer_mean = SimpleImputer(strategy="mean")
simple_imputer_constant = SimpleImputer(strategy="constant", fill_value="unknown")
ordinal_encoder = OrdinalEncoder(categories=[["No", "unknown", "Yes"], ["No", "unknown", "Yes"]])
ordinal_encoder2 = OrdinalEncoder(categories=[["Introvert", "Extrovert"]])


numeric_cols = train_data.select_dtypes(include=['float']).columns.to_list()
cat_cols = train_data.select_dtypes(include=['object']).columns.to_list()


train_data[numeric_cols] = simple_imputer_mean.fit_transform(train_data[numeric_cols])
test_data[numeric_cols] = simple_imputer_mean.transform(test_data[numeric_cols])
train_data[cat_cols[:2]] = simple_imputer_constant.fit_transform(train_data[cat_cols[:2]])
test_data[cat_cols[:2]] = simple_imputer_constant.transform(test_data[cat_cols[:2]])


train_data[cat_cols[:2]] = ordinal_encoder.fit_transform(train_data[cat_cols[:2]])
test_data[cat_cols[:2]] = ordinal_encoder.transform(test_data[cat_cols[:2]])
train_data["Personality_encoded"] = ordinal_encoder2.fit_transform(train_data[["Personality"]])


X = train_data.drop(["id", 'Personality', 'Personality_encoded'], axis=1)
y = train_data['Personality_encoded']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=44)


rf_model = RandomForestClassifier(random_state=44)


rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("="*10,f" Report for Random Forest Model ", "="*10, end="\n\n")
print(f"Accuracy: {acc}\n")
print(classification_report(y_test, y_pred, target_names=["Introvert", "Extrovert"]))


from sklearn.model_selection import cross_val_score


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 44,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score


study = optuna.create_study(direction="maximize")


study.optimize(objective, n_trials=50)


print("En iyi parametreler:", study.best_params)


best_model = RandomForestClassifier(**study.best_params)
best_model.fit(X, y)


test_data_X = test_data.drop("id", axis=1)


rf_predictions = best_model.predict(test_data_X)


id_ = test_data["id"]
data = pd.DataFrame(
    {
        "id": id_,
        "enc_Personality": rf_predictions
    }
)
data.head()


data["Personality"] = ordinal_encoder2.inverse_transform(data[["enc_Personality"]]).flatten()


data_to_submit = data.drop("enc_Personality", axis=1)


data_to_submit.head()


data_to_submit.to_csv('submission.csv', index=False)




