!pip install xgboost


#import all files

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import seaborn as sns


train = pd.read_csv(r"C:\Users\mandeep\Downloads\playground-series-s5e8\train.csv")


#Train Data Set
train


#Drop ID column
train.drop(columns=['id'], axis=1, inplace=True)


#Train Data Set
train


train.info()


#Counting values of column "y"
train['y'].value_counts()


#Convert values

train['job'] = train['job'].replace("management", 1)
train['job'] = train['job'].replace("blue-collar", 2)
train['job'] = train['job'].replace("technician", 3)
train['job'] = train['job'].replace("admin.", 4)
train['job'] = train['job'].replace("services", 5)
train['job'] = train['job'].replace("retired", 6)
train['job'] = train['job'].replace("self-employed", 7)
train['job'] = train['job'].replace("entrepreneur", 8)
train['job'] = train['job'].replace("unemployed", 9)
train['job'] = train['job'].replace("housemaid", 10)
train['job'] = train['job'].replace("student", 11)
train['job'] = train['job'].replace("unknown", 12)

train['marital'] = train['marital'].replace("married", 1)
train['marital'] = train['marital'].replace("single", 2)
train['marital'] = train['marital'].replace("divorced", 3) 

train['education'] = train['education'].replace("primary", 1)
train['education'] = train['education'].replace("secondary", 2)
train['education'] = train['education'].replace("tertiary", 3)
train['education'] = train['education'].replace("unknown", 4)

train['default'] = train['default'].replace("yes", 1)
train['default'] = train['default'].replace("no", 2)

train['housing'] = train['housing'].replace("yes", 1)
train['housing'] = train['housing'].replace("no", 2)

train['loan'] = train['loan'].replace("no", 2)
train['loan'] = train['loan'].replace("yes", 1)

train['contact'] = train['contact'].replace("cellular", 1)
train['contact'] = train['contact'].replace("unknown", 2)
train['contact'] = train['contact'].replace("telephone", 3) 

train['month'] = train['month'].replace("jan", 1)
train['month'] = train['month'].replace("feb", 2)
train['month'] = train['month'].replace("mar", 3)
train['month'] = train['month'].replace("apr", 4)
train['month'] = train['month'].replace("may", 5)
train['month'] = train['month'].replace("jun", 6)
train['month'] = train['month'].replace("jul", 7)
train['month'] = train['month'].replace("aug", 8)
train['month'] = train['month'].replace("sep", 9)
train['month'] = train['month'].replace("oct", 10)
train['month'] = train['month'].replace("nov", 11)
train['month'] = train['month'].replace("dec", 12)

train['poutcome'] = train['poutcome'].replace("success", 1)
train['poutcome'] = train['poutcome'].replace("failure", 2)
train['poutcome'] = train['poutcome'].replace("unknown", 3)
train['poutcome'] = train['poutcome'].replace("other", 4)


#Train Data Set after conversion
train


test = pd.read_csv(r"C:\Users\mandeep\Downloads\playground-series-s5e8\test.csv")


#Test Data Set
test


#Copy of Test Data Set
test_copy = test.copy()


test_copy


#Drop ID column
test.drop(columns=['id'], axis=1, inplace=True)


#Test Data Set
test


test.info()


#Convert values

test['job'] = test['job'].replace("management", 1)
test['job'] = test['job'].replace("blue-collar", 2)
test['job'] = test['job'].replace("technician", 3)
test['job'] = test['job'].replace("admin.", 4)
test['job'] = test['job'].replace("services", 5)
test['job'] = test['job'].replace("retired", 6)
test['job'] = test['job'].replace("self-employed", 7)
test['job'] = test['job'].replace("entrepreneur", 8)
test['job'] = test['job'].replace("unemployed", 9)
test['job'] = test['job'].replace("housemaid", 10)
test['job'] = test['job'].replace("student", 11)
test['job'] = test['job'].replace("unknown", 12)

test['marital'] = test['marital'].replace("married", 1)
test['marital'] = test['marital'].replace("single", 2)
test['marital'] = test['marital'].replace("divorced", 3)

test['education'] = test['education'].replace("primary", 1)
test['education'] = test['education'].replace("secondary", 2)
test['education'] = test['education'].replace("tertiary", 3)
test['education'] = test['education'].replace("unknown", 4)

test['default'] = test['default'].replace("yes", 1)
test['default'] = test['default'].replace("no", 2)

test['housing'] = test['housing'].replace("yes", 1)
test['housing'] = test['housing'].replace("no", 2)

test['loan'] = test['loan'].replace("yes", 1)
test['loan'] = test['loan'].replace("no", 2)

test['contact'] = test['contact'].replace("cellular", 1)
test['contact'] = test['contact'].replace("unknown", 2)
test['contact'] = test['contact'].replace("telephone", 3)

test['month'] = test['month'].replace("jan", 1)
test['month'] = test['month'].replace("feb", 2)
test['month'] = test['month'].replace("mar", 3)
test['month'] = test['month'].replace("apr", 4)
test['month'] = test['month'].replace("may", 5)
test['month'] = test['month'].replace("jun", 6)
test['month'] = test['month'].replace("jul", 7)
test['month'] = test['month'].replace("aug", 8)
test['month'] = test['month'].replace("sep", 9)
test['month'] = test['month'].replace("oct", 10)
test['month'] = test['month'].replace("nov", 11)
test['month'] = test['month'].replace("dec", 12)

test['poutcome'] = test['poutcome'].replace("success", 1)
test['poutcome'] = test['poutcome'].replace("failure", 2)
test['poutcome'] = test['poutcome'].replace("unknown", 3)
test['poutcome'] = test['poutcome'].replace("other", 4)


#Test Data Set after conversion
test


X_train = train.iloc[:, :-1]
X_train


Y_train = train.y
Y_train


X_test = test.iloc[:, :]
X_test


len(X_train)


len(Y_train)


len(X_test)


model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=3)


model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)


Y_pred


probabilities_native = 1 / (1 + np.exp(-Y_pred))
print("\nSample probabilities:")
print(probabilities_native[:5])


probabilities_native


len(X_test)


len(Y_pred) 


X_test_with_predictions = test_copy.copy()


X_test_with_predictions['predictions'] = Y_pred


X_test_with_predictions['prob'] = probabilities_native


X_test_with_predictions


X_test_with_predictions.to_csv(r"C:\Users\mandeep\Downloads\playground-series-s5e8\Final_submission.csv", index=False)


#I can't find the accuracy of a model because the test I got doesn't have 'y' column with which I can't able to match my 'Y_pred' values.

