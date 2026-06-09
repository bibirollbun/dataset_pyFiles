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


train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols=train.select_dtypes(include=['int','float']).columns.tolist()


categorical_cols


for i in categorical_cols:
    train[i]=train[i].fillna(train[i].mode()[0])
for i in numerical_cols:
    train[i]=train[i].fillna(train[i].mean())
train=pd.get_dummies(train,drop_first=True)


train


categorical_cols.remove("Personality")


print(categorical_cols)


for i in categorical_cols:
    test[i]=test[i].fillna(test[i].mode()[0])
for i in numerical_cols:
    test[i]=test[i].fillna(test[i].mean())
test=pd.get_dummies(test,drop_first=True)
test=test.drop("id",axis=1)


X=train.drop(["id","Personality_Introvert"],axis=1)
y=train["Personality_Introvert"]




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)




from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "XGBClassifier":XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
}


accuracy_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    accuracy_scores[name] = acc
    print(f"{name}: {acc:.2f}%")


model=XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)


Personality=model.predict(test)


Personality = [ "Introvert" if p == True else "Extrovert" for p in Personality ]
sample_submission["Personality"]=Personality


output = pd.DataFrame({'id': sample_submission.id, 'Personality': Personality})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


sample_submission

