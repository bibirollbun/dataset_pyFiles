import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)
train_labels = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)


print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Train Labels Shape:', train_labels.shape)


train.head()


train.info()


train.describe()


train_labels.head()



train_labels.info()
train_labels.describe()


#Split
# train labels is a DataFrame so we will convert it to 1D array at first
y = train_labels.values.ravel()


X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=42)
print("Training set:", X_train.shape)
print("Validation set:", X_valid.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)


lr_pred = lr_model.predict(X_valid)


#accuracy
lr_acc = accuracy_score(y_valid, lr_pred)
print("LR Accuracy:", lr_acc)


full_model = LogisticRegression(max_iter=1000)
full_model.fit(train, y)


test_pred = full_model.predict(test)
print(test_pred.shape)


submission = pd.DataFrame({
    'Id': np.arange(1, len(test_pred) + 1),
    'Solution': test_pred
})
submission.to_csv('submission_LogisticRegression.csv', index=False)
submission.head()


from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1, gamma='scale') #using capital C so that model could be more accurate
svm_model.fit(X_train, y_train)


svm_pred = svm_model.predict(X_valid)
svm_acc = accuracy_score(y_valid, svm_pred)


print("SVM Accuracy:", svm_acc)


fullSVM_model = SVC(kernel='rbf', C=1, gamma='scale')
fullSVM_model.fit(train, y)
SVMtest_pred = fullSVM_model.predict(test)
print(SVMtest_pred.shape)
submission = pd.DataFrame({
    'Id': np.arange(1, len(SVMtest_pred) + 1),
    'Solution': SVMtest_pred
})
submission.to_csv('submission.csv', index=False)
submission.head()


from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)


rf_pred = rf_model.predict(X_valid)
rf_acc = accuracy_score(y_valid, rf_pred)


print("Random Forest Accuracy:", rf_acc)


fullRF_model = RandomForestClassifier(n_estimators=300, random_state=42)
fullRF_model.fit(train, y)
RFtest_pred = fullRF_model.predict(test)
print(RFtest_pred.shape)
submission = pd.DataFrame({
    'Id': np.arange(1, len(RFtest_pred) + 1),
    'Solution': RFtest_pred
})
submission.to_csv('submission_RF.csv', index=False)
submission.head()


from sklearn.neighbors import KNeighborsClassifier


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)


knn_pred = knn_model.predict(X_valid)
knn_acc = accuracy_score(y_valid, knn_pred)


print("KNN Accuracy:", knn_acc)


fullKNN_model = KNeighborsClassifier(n_neighbors=5)
fullKNN_model.fit(train, y)
KNNtest_pred = fullKNN_model.predict(test)
print(KNNtest_pred.shape)
submission = pd.DataFrame({
    'Id': np.arange(1, len(KNNtest_pred) + 1),
    'Solution': KNNtest_pred
})
submission.to_csv('submission_KNN.csv', index=False)
submission.head()


print("LR Accuracy:", lr_acc)
print("SVM:", svm_acc)
print("Random Forest:", rf_acc)
print("KNN:", knn_acc)

