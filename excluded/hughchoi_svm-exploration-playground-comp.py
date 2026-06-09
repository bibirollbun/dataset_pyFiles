# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass

train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
train = train.sample(frac=0.01, random_state=42)

teste = test.copy()
y = train["BeatsPerMinute"]

train = train.drop(["id","AudioLoudness","VocalContent" ,"AcousticQuality" ,"InstrumentalScore" , "LivePerformanceLikelihood" ,"BeatsPerMinute" ], axis = 1)
testa = teste.drop(["id","AudioLoudness","VocalContent" ,"AcousticQuality" ,"InstrumentalScore" , "LivePerformanceLikelihood"], axis = 1)

train_X, test_X, train_y, test_y = train_test_split(train,y ,test_size = 0.2, random_state = 42)

pipeline = Pipeline([
    ("preprocessing" , StandardScaler()),
    ("SVM" , SVR())
])

param_dist = {
    'SVM__kernel': ['rbf'],
    'SVM__C': loguniform(1e-1, 1e3),
    'SVM__epsilon': uniform(0.01, 1.0),
    'SVM__gamma': loguniform(1e-4, 1e-1)
}

ran_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=3,       
    cv=3,           
    n_jobs=-1,      
    random_state=42
)
ran_search.fit(train_X, train_y)

pred_y = ran_search.predict(train_X)

features_to_plot = ['RhythmScore', 'MoodScore', 'Energy']

best_svr = ran_search.best_estimator_

fig, ax = plt.subplots(figsize=(12, 4), ncols=len(features_to_plot))
display = PartialDependenceDisplay.from_estimator(
    best_svr,
    train_X,
    features=features_to_plot,
    ax=ax
)

fig.suptitle("Partial Dependence Plots for SVR")
plt.tight_layout()
plt.show()

final_y = ran_search.predict(testa)

submission = pd.DataFrame({
    "Id": test["id"],
    "BeatsPerMinute": final_y
})

submission.to_csv("submission.csv", index=False)





