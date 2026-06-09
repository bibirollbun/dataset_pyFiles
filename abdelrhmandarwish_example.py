import pandas as pd
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
id_num=test['id']


train.drop(columns='id',inplace=True)
test.drop(columns='id',inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


X=train.drop(columns=['y'])
y=train['y']
X_test=test


X_train, X_val, y_train, y_val= train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


X_train.dtypes


model=LogisticRegression(max_iter=1000)


le=LabelEncoder()


cat_col=X_train.select_dtypes(include="object")
for col in cat_col :
    X_train[col]=le.fit_transform(X_train[col])
X_train


cat_col=X_val.select_dtypes(include="object")
for col in cat_col :
    X_val[col]=le.fit_transform(X_val[col])
X_val


cat_col=X_test.select_dtypes(include="object")
for col in cat_col :
    X_test[col]=le.fit_transform(X_test[col])
X_test


clf=DecisionTreeClassifier(criterion="entropy")


clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)



accuracy = metrics.accuracy_score(y_val, y_pred[0:225000])
print(f"Accuracy: {accuracy:.2f}")


submission = pd.DataFrame({
    "id": id_num ,       
    "y": y_pred         
})


submission.to_csv("submission.csv", index=False)



submission


!ls

