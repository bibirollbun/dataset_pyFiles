import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df.head()


df.describe()


df.info()


df.isna().sum()


df.corr


x = df.drop(["rainfall"],axis = "columns")
y = df.rainfall


from sklearn.model_selection import train_test_split
x_train , x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)


len(x_test)


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import GridSearchCV


model_para ={
    "svm":{ "model":svm.SVC(),
           "para":{
               "C":[1,10,20],
               "kernel":["rbf","linear"],
               "gamma":["auto","scale"]
           } 
    },
    "RandomForestClassifier":{
        "model":RandomForestClassifier(),
        "para":{
            "n_estimators":[10,20,40,50,60,70,100]
        }
    },
    "LogisticRegression" :{
        "model":LogisticRegression(solver="liblinear"),
        "para":{
            "C":[1,10,20]
        }
    },
    "DecisionTreeClassifier":{
        "model":DecisionTreeClassifier(),
        "para":{"max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    }
}


scores = []
for model_name , mp in model_para.items():
    clf = GridSearchCV(mp["model"],mp["para"],cv=5,return_train_score =False,n_jobs=-1)
    clf.fit(x_train,y_train)

    scores.append({
        'model': model_name,
        'best_score': clf.best_score_ ,
        'best_params': clf.best_params_
    })
df1 = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df1


model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)


model.score(x_test,y_test)


model.predict(x_test)


model2 = RandomForestClassifier(n_estimators=100)
model2.fit(x,y)


df.columns


test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test_df.head()


print(test_df.isnull().sum())


test_df.fillna(test_df.median(), inplace=True)


model2.predict(test_df)


model2.predict_proba(test_df)


test_preds = model2.predict_proba(test_df)[:, 1] 


output = pd.DataFrame({'id': test_df.id,
                       'rainfall': test_preds})
output.to_csv('submission.csv', index=False)

