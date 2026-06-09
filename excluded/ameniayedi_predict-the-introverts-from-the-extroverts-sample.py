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


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import missingno

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split


df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")



df.head()


df.info()


df.shape


df.isna().sum().sort_values()


missingno.matrix(df)


df.describe()


df.columns


for col in df.columns:
    print(f'Unique values for the {col} column : \n{df[col].unique()}')


df['Personality'].value_counts()


sns.countplot(data=df,x='Personality')
plt.show()


df['Time_spent_Alone'].value_counts().sort_values()


df[['Time_spent_Alone','Personality']].groupby('Personality',as_index=False).mean()


sns.barplot(data=df,y='Time_spent_Alone',x='Personality')
plt.show()


for col in list(df.columns)[1:7]:
    sns.countplot(data=df,x=col,hue='Personality')
    plt.show()


X=df.drop(columns=['Personality'],axis=1)


y=df['Personality']


X.head()


y


le = LabelEncoder()
y= le.fit_transform(y)


y


num_cols=list(X.select_dtypes('float').columns)


for col in num_cols:
    median=X[col].median()
    print(f"The median {col} value is :",median)
    X[col].fillna(median,inplace=True)


X.isna().sum()


bool_cols=list(X.select_dtypes('object').columns)


bool_cols


for col in bool_cols:
    mode=X[col].mode()[0]
    print(f"The median {col} value is :",mode)
    X[col].fillna(mode,inplace=True)


X.isna().sum()


print("Number of duplicated values in the train dataset:" ,X.duplicated().sum())


X["Stage_fear"]=X["Stage_fear"].map({"No":0,"Yes":1})
X["Drained_after_socializing"]=X["Drained_after_socializing"].map({"No":0,"Yes":1})


X.info()


scaler = StandardScaler()
X[['Time_spent_Alone', 'Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']]= scaler.fit_transform(X[['Time_spent_Alone', 'Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']])


X=X.drop(columns=['id'],axis=1)


X.isna().sum()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print("X_train shape : ",X_train.shape)
print("y_train shape : ",y_train.shape)
print("X_test shape : ",X_test.shape)
print("y_test shape : ",y_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
accuracy_log=round(logreg.score(X_train,y_train)*100,2)
accuracy_log


svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
accuracy_svc=round(svc.score(X_train,y_train)*100,2)
accuracy_svc


knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_knn=round(knn.score(X_train,y_train)*100,2)
accuracy_knn


gaussian=GaussianNB()
gaussian.fit(X_train,y_train)
y_pred=gaussian.predict(X_test)
accuracy_gaussian=round(gaussian.score(X_train,y_train)*100,2)
accuracy_gaussian


perceptron=Perceptron()
perceptron.fit(X_train,y_train)
y_pred=perceptron.predict(X_test)
accuracy_perceptron=round(perceptron.score(X_train,y_train)*100,2)
accuracy_perceptron


l_svc=LinearSVC()
l_svc.fit(X_train,y_train)
y_pred=l_svc.predict(X_test)
accuracy_l_svc=round(l_svc.score(X_train,y_train)*100,2)
accuracy_l_svc


sgd=SGDClassifier()
sgd.fit(X_train,y_train)
y_pred=sgd.predict(X_test)
accuracy_sgd=round(sgd.score(X_train,y_train)*100,2)
accuracy_sgd


decision_tree=DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
y_pred=decision_tree.predict(X_test)
accuracy_decision_tree=round(decision_tree.score(X_train,y_train)*100,2)
accuracy_decision_tree


random_forest=RandomForestClassifier()
random_forest.fit(X_train,y_train)
y_pred=random_forest.predict(X_test)
accuracy_random_forest=round(random_forest.score(X_train,y_train)*100,2)
accuracy_random_forest


catboost=CatBoostClassifier()
catboost.fit(X_train,y_train)
y_pred=catboost.predict(X_test)
accuracy_catboost=round(catboost.score(X_train,y_train)*100,2)
accuracy_catboost


model_perf=pd.DataFrame({
                        "Model": ["Support Vector Machines","KNN","Logistic Regression","Random Forest","Naive Bayes","Perceptron","Stochastic Gradient Decent","Linear SVC","Decision Tree","CatBoost"],
                         'Score' : [accuracy_svc,accuracy_knn,accuracy_log,accuracy_random_forest,accuracy_gaussian,accuracy_perceptron,accuracy_sgd,accuracy_l_svc,accuracy_decision_tree,accuracy_catboost]
                        })


model_perf.sort_values("Score",ascending=False)


classifiers=[LogisticRegression(), SVC(), KNeighborsClassifier(), 
GaussianNB(), Perceptron(), LinearSVC(), SGDClassifier() ,
DecisionTreeClassifier(), RandomForestClassifier(), CatBoostClassifier()]
len(classifiers)


cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,X_train,y_train,scoring='accuracy',cv=10))


cv_results


cv_mean=[]
cv_std=[]
for cv_result in cv_results:
    cv_mean.append(cv_result.mean())
    cv_std.append(cv_result.std())


cv_res=pd.DataFrame({
            "Cross Validation Mean" : cv_mean,
            "Cross Validation Std" : cv_std,
            "Algorithm": ["Logistic Regression","Support Vector Machines","KNN","Naive Bayes","Perceptron","Linear SVC","Stochastic Gradient Decent","Decision Tree","Random Forest","CatBoost"]   
})
cv_res.sort_values(by="Cross Validation Mean",ascending=False,ignore_index=True)


classifiers


sns.barplot(x="Cross Validation Mean",y="Algorithm",data=cv_res.sort_values(by="Cross Validation Mean", ascending=False))
plt.ylabel("Algorithm")
plt.title("Cross Validation Scores")
plt.show()


from sklearn.model_selection import GridSearchCV 

param_grid = {'C': [0.1, 1, 10, 100, 1000], 
			'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
			'kernel': ['rbf']} 

grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3) 
 
grid.fit(X_train, y_train)


print("Best parameters : ",grid.best_params_)
print("Best estimator : ", grid.best_estimator_)


svc=SVC(C=100, gamma=0.001, kernel= 'rbf')


svc.fit(X_train,y_train)


y_pred=svc.predict(X_test)
acc_svc=round(accuracy_score(y_test,y_pred)*100,2)
print(acc_svc)


cross_val_score(svc,X_train,y_train,scoring='accuracy',cv=10).mean()


y_pred




