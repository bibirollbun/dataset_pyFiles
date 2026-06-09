import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
Table = []


#read the training dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s3e12/train.csv')
#read the test dataset
test_df = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')


#drop column id
train_df.drop('id',axis = 1,inplace= True)

#drop nulls
train_df.dropna(inplace = True)
test_df.dropna(inplace = True)

#reset index
train_df.reset_index(drop=True,inplace=True)
test_df.reset_index(drop=True,inplace=True)



#feature engineering
X = train_df.iloc[:,:6]
y = train_df[['target']]

#split data
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)


# Import the library
from sklearn.naive_bayes import MultinomialNB

# Initialize and fit the model
naive_model = MultinomialNB().fit(X_train, y_train.values.ravel())

# Predict on the test data
y_pred = naive_model.predict(X_test)

# Optionally, evaluate the model (e.g., using accuracy score)
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

Table.append(['Naive Bayes',naive_model.score(X_test, y_test)])



#import the library
from sklearn.linear_model import LogisticRegression

#initialize and fit
lr = LogisticRegression(max_iter = 5000)
lr.fit(X_train,y_train.values.ravel())

#test
y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

Table.append(['Logistic Regression', lr.score(X_test, y_test)])


#using SGDCLassifier

from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier()
sgd_model.fit(X_train,y_train.values.ravel())

#predict
y_pred = sgd_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

Table.append(['SGDClassifier', sgd_model.score(X_test, y_test)])


#import the necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

#initialize and fit
knn_model = KNeighborsClassifier(algorithm = "brute",n_jobs = -1)
knn_model.fit(X_train,y_train.values.ravel())

#predict on the test sample
y_pred = knn_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

Table.append(['KNN', knn_model.score(X_test, y_test)])


#Import the necessary libraries
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report

#Initialize and fit
svm = LinearSVC(C=0.0001, max_iter=6000) 
svm.fit(X_train,y_train.values.ravel())

#predict on test sample
y_pred = svm.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
Table.append(['SVM', svm.score(X_test, y_test)])


#Import the necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report

#Initialize and fit
tree = DecisionTreeClassifier(min_samples_split = 10,max_depth = 3)
tree.fit(X_train,y_train.values.ravel())

#predict on test sample
y_pred = tree.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
Table.append(['Decision Tree', tree.score(X_test, y_test)])


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report

bg = BaggingClassifier(DecisionTreeClassifier(min_samples_split = 10,max_depth = 3),max_samples = 0.5,max_features = 1.0,n_estimators = 10)
bg.fit(X_train,y_train.values.ravel())

#predict on test sample
y_pred = bg.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
Table.append(['Bagging Decision Tree', bg.score(X_test, y_test)])



from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2),n_estimators = 100,learning_rate = 0.5)
adb.fit(X_train,y_train.values.ravel())

#predict on test sample
y_pred = adb.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
Table.append(['Boosting Decision Tree', adb.score(X_test, y_test)])


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators = 100)
gbc.fit(X_train,y_train.values.ravel())

#predict on test sample
y_pred = gbc.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
Table.append(['Gradient Boosting Decision Tree', gbc.score(X_test, y_test)])


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 300,max_depth = 3)
rf.fit(X_train,y_train.values.ravel())
#predict on test sample
y_pred = rf.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
Table.append(['Random Forest', rf.score(X_test, y_test)])


from tabulate import tabulate
print(tabulate(Table, headers=["Model","Score"], tablefmt='fancy_outline') )


#fine tuning the decision tree classifier
from sklearn.model_selection import GridSearchCV

dtc = DecisionTreeClassifier()

#define parameter grid
param_grid = [{'min_samples_split': [5, 10, 15, 20], 'max_depth': [3, 6, 9, 12]}]

grid_search = GridSearchCV(estimator=dtc,
                          param_grid=param_grid,
                          scoring="accuracy",
                          cv=5,
                          return_train_score=True)

# fit the grid search
grid_search.fit(X_train, y_train.values.ravel())


# get the best estimator
dtc_tuned = grid_search.best_estimator_

# fit the estimator
dtc_tuned.fit(X_train, y_train.values.ravel())
print("score on test: "  + str(dtc_tuned.score(X_test, y_test)))


result = dtc_tuned.predict(X_test)

# Create submission DataFrame with 'id' and 'target'
submission = pd.DataFrame({
    'target': result
})

# Format date
datestamp = '{:%Y_%m_%d}'.format(datetime.date.today())

# Save to CSV
filename = f"{datestamp}_submission.csv"
submission.to_csv(filename, index=False)

print(f"✅ Submission saved as: {filename}")

