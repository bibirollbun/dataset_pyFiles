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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from numpy.random import randn, rand, seed
from statistics import mean

import warnings
warnings.filterwarnings('ignore')

seed(42)


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

train.drop(columns=['id'], inplace=True)
test_id = test['id']
test.drop(columns=['id'], inplace=True)


train_duplicates = train.duplicated().sum()
test_duplicates = test.duplicated().sum()

print(f"Train duplicates: {train_duplicates}")
print(f"Test duplicates: {test_duplicates}")


train.head()


train.info()


train["Personality"] = train["Personality"].map({"Introvert": 0, "Extrovert": 1})

def impute_numerical(df):
    df["Time_spent_Alone"].fillna(df["Time_spent_Alone"].mean(), inplace=True)
    df["Social_event_attendance"].fillna(df["Social_event_attendance"].mean(), inplace=True)
    df["Going_outside"].fillna(df["Going_outside"].mean(), inplace=True)
    df["Friends_circle_size"].fillna(df["Friends_circle_size"].mean(), inplace=True)
    df["Post_frequency"].fillna(df["Post_frequency"].mean(), inplace=True)
    return df

def impute_categorical(df):
    df["Stage_fear"].fillna(df["Stage_fear"].mode()[0], inplace=True)
    df["Drained_after_socializing"].fillna(df["Drained_after_socializing"].mode()[0], inplace=True)
    return df

#def generate_features(df):
    #df["Total_social_score"] = (
        #df["Social_event_attendance"] +
        #df["Going_outside"] +
        #df["Friends_circle_size"]
    #)
    #df["Alone_ratio"] = df["Time_spent_Alone"] / (df["Post_frequency"] + df["Going_outside"] + 1)
    #return df

train = impute_numerical(train)
train = impute_categorical(train)
#train = generate_features(train)

test = impute_numerical(test)
test = impute_categorical(test)
#test = generate_features(test)


TE = TargetEncoder()
CATS = ["Stage_fear", "Drained_after_socializing"]

for col in CATS:
    train[f"TE_{col}"] = TE.fit_transform(train[col], train["Personality"])
    test[f"TE_{col}"] = TE.transform(test[col])

train.drop(columns=CATS, inplace=True)
test.drop(columns=CATS, inplace=True)



X = train.drop(columns=["Personality"])
y = train["Personality"]
X_test = test


def objective(X, y, cfg):
    eta, max_depth, subsample = cfg
    max_depth = int(max_depth)

    model = XGBClassifier(
        objective='multi:softmax', 
        num_class=len(np.unique(y)),
        eta=eta,
        max_depth=max_depth,
        subsample=subsample,
        tree_method='gpu_hist',    
        predictor='gpu_predictor',
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return mean(scores)


def step(cfg, step_size):
    eta, max_depth, subsample = cfg

    new_eta = np.clip(eta + randn() * step_size, 0.01, 0.5)
    new_max_depth = int(np.clip(max_depth + randn() * step_size * 10, 3, 15))
    new_subsample = np.clip(subsample + randn() * step_size, 0.5, 1.0)

    return [new_eta, new_max_depth, new_subsample]

def hillclimbing(X, y, objective, n_iter, step_size):
    solution = [rand()*0.4 + 0.05, rand()*10 + 3, rand()*0.5 + 0.5]
    solution_eval = objective(X, y, solution)

    for i in range(n_iter):
        candidate = step(solution, step_size)
        candidate_eval = objective(X, y, candidate)
        if candidate_eval >= solution_eval:
            solution, solution_eval = candidate, candidate_eval
            print(f'>{i}, cfg={solution}, acc={solution_eval:.5f}')
    return solution, solution_eval



n_iter = 50
step_size = 0.1

best_cfg, best_score = hillclimbing(X, y, objective, n_iter, step_size)

print("Best config:", best_cfg)
print("Accuracy:", best_score)


eta, max_depth, subsample = best_cfg
max_depth = int(max_depth)

best_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    eta=eta,
    max_depth=max_depth,
    subsample=subsample,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

best_model.fit(X, y)



y_pred = best_model.predict(X_test)


sub['Personality'] = y_pred
sub['Personality'] = sub['Personality'].map({0: "Introvert", 1: "Extrovert"})
sub.to_csv("submission.csv", index=False)


sub.head()

