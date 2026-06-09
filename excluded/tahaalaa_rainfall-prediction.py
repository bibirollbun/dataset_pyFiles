import numpy as np 
import pandas as pd 
from sklearn.svm import LinearSVC 
from sklearn.model_selection import train_test_split


data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
data.head()


X = data.drop(columns=['rainfall','id'])
y = data['rainfall']

#splitting data 
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=38,test_size=.2)


#splitting data 
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=38,test_size=.2)


#select model and train data 
model = LinearSVC(max_iter=10000,random_state=38)
model.fit(X_train,y_train)


score = model.score(X_test,y_test)
score




