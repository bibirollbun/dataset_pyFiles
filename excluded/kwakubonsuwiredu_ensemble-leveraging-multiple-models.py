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


data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
data.head(5)


!pip install xgboost
from sklearn.model_selection import train_test_split

# import models
from xgboost import XGBRFClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Final model
from sklearn.svm import SVC, SVR


X = data.drop('rainfall', axis=1)
y = data[['rainfall']]
X_train, X_test, y_train, y_test = train_test_split(X,y)


# Training models
from sklearn.metrics import accuracy_score

# 1. XGBoost
model_1 = XGBRFClassifier(tree_method="hist", eval_metric=accuracy_score).fit(X=X_train, y = y_train)

# 2. AdaBoost
model_2 = AdaBoostClassifier().fit(X_train, y_train)

# 3. Stacking
model_3 = RidgeClassifier().fit(X_train, y_train)

# 4. Random Forest
model_4 = RandomForestClassifier().fit(X_train, y_train)

# 5. KNN
model_5 = KNeighborsClassifier().fit(X_train, y_train)

# 6. Linear Regression
model_6 = LinearRegression().fit(X_train, y_train)

# Multilayered Preceptron Learning
model_7 = MLPClassifier().fit(X_train, y_train)


preds1 = pd.DataFrame(model_1.predict(X_test))
preds2 = pd.DataFrame(model_2.predict(X_test))
preds3 = pd.DataFrame(model_3.predict(X_test))
preds4 = pd.DataFrame(model_4.predict(X_test))
preds5 = pd.DataFrame(model_5.predict(X_test))
preds6 = pd.DataFrame(model_6.predict(X_test))
preds7 = pd.DataFrame(model_7.predict(X_test))



x=0
for prediction in [preds1,preds2,preds3,preds4,preds5,preds6,preds7]:
    x+=1
    try:
        print(f"Accuracy of Model {x} is {accuracy_score(prediction, y_test)}")
    except Exception as e:
        print(e)


preds1 = pd.DataFrame(model_1.predict(X_test))
preds2 = pd.DataFrame(model_2.predict(X_test))
preds3 = pd.DataFrame(model_3.predict(X_test))
preds4 = pd.DataFrame(model_4.predict(X_test))
preds5 = pd.DataFrame(model_5.predict(X_test))
preds6 = pd.DataFrame(model_6.predict(X_test))
preds7 = pd.DataFrame(model_7.predict(X_test))
data_final1 = pd.concat([preds1,preds2,preds3,preds4,preds5,preds6,preds7], axis = 1)
data_final1


train1 = pd.DataFrame(model_1.predict(X_train))
train2 = pd.DataFrame(model_2.predict(X_train))
train3 = pd.DataFrame(model_3.predict(X_train))
train4 = pd.DataFrame(model_4.predict(X_train))
train5 = pd.DataFrame(model_5.predict(X_train))
train6 = pd.DataFrame(model_6.predict(X_train))
train7 = pd.DataFrame(model_7.predict(X_train))
data_final = pd.concat([train1,train2,train3,train4,train5,train6,train7], axis = 1)
data_final


model_final = SVC(kernel ='poly')
model_final.fit(data_final, y_train)


accuracy_score(model_final.predict(data_final1),y_test)


#Finalising Models

model_1 = XGBRFClassifier(tree_method="hist", eval_metric=accuracy_score).fit(X=X, y = y)

# 2. AdaBoost
model_2 = AdaBoostClassifier().fit(X, y)

# 3. Stacking
model_3 = RidgeClassifier().fit(X, y)
# 4. Random Forest
model_4 = RandomForestClassifier().fit(X, y)

# 5. KNNy
model_5 = KNeighborsClassifier().fit(X, y)

# 6. Linear Regression
model_6 = LinearRegression().fit(X, y)

# Multilayered Preceptron Learning
model_7 = MLPClassifier().fit(X, y)


# SVM
model_final = SVR(kernel = 'poly')

preds1 = pd.DataFrame(model_1.predict(X))
preds2 = pd.DataFrame(model_2.predict(X))
preds3 = pd.DataFrame(model_3.predict(X))
preds4 = pd.DataFrame(model_4.predict(X))
preds5 = pd.DataFrame(model_5.predict(X))
preds6 = pd.DataFrame(model_6.predict(X))
preds7 = pd.DataFrame(model_7.predict(X))
data_final1 = pd.concat([preds1,preds2,preds3,preds4,preds5,preds6,preds7], axis = 1)


model_final.fit(data_final1, y)
data_final1


test1 = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv',index_col='id')
from sklearn.impute import KNNImputer
test = KNNImputer().fit_transform(test)


preds1 = pd.DataFrame(model_1.predict(test))
preds2 = pd.DataFrame(model_2.predict(test))
preds3 = pd.DataFrame(model_3.predict(test))
preds4 = pd.DataFrame(model_4.predict(test))
preds5 = pd.DataFrame(model_5.predict(test))
preds6 = pd.DataFrame(model_6.predict(test))
preds7 = pd.DataFrame(model_7.predict(test))
data_final1 = pd.concat([preds1,preds2,preds3,preds4,preds5,preds6,preds7], axis = 1)

prediction = model_final.predict(data_final1)
csv_dat = pd.DataFrame()

csv_dat['id'] = test1['id']
csv_dat['rainfall'] = prediction
csv_dat.to_csv('submission.csv', index=False)

