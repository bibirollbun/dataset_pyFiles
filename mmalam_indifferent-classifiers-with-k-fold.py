import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')


df.head()


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df = pd.get_dummies(data=df, columns=['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade'],drop_first=True)
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_test = pd.get_dummies(data=df_test, columns=['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade'],drop_first=True)
df.head()


X = df.drop(columns=['loan_paid_back'])
y = df['loan_paid_back']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


X_test_sub = df_test.reindex(columns=X.columns, fill_value=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_sub = sc.transform(X_test_sub)


result_comparison = pd.DataFrame(columns=['Model', 'Cross Validation Mean Accuracy', 'Cross Validation Standard Deviation', 'Test Data Accuracy', 'Test Data Precision', 'Test Data Recall', 'Test Data Specificity' ])


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))


precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100
print("Precision: {:.2f} %".format(precision))
recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
print("Recall: {:.2f} %".format(recall))
specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100
print("Specificity: {:.2f} %".format(specificity))



new_row = {
    'Model': 'Logistic Regression',
    'Cross Validation Mean Accuracy':accuracies.mean()*100,
    'Cross Validation Standard Deviation': accuracies.std()*100,
    'Test Data Accuracy': accuracy_score(y_test, y_pred)*100,
    'Test Data Precision': precision,
    'Test Data Recall': recall,
    'Test Data Specificity': specificity
}

# Append using pd.concat
result_comparison = pd.concat(
    [result_comparison, pd.DataFrame([new_row])],
    ignore_index=True
);


y_test_pred = classifier.predict(X_test_sub)

submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_test_pred
})

submission.to_csv("Logistic_Regression.csv", index=False)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100
print("Precision: {:.2f} %".format(precision))
recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
print("Recall: {:.2f} %".format(recall))
specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100
print("Specificity: {:.2f} %".format(specificity))

new_row = {
    'Model': 'Decision Tree',
    'Cross Validation Mean Accuracy':accuracies.mean()*100,
    'Cross Validation Standard Deviation': accuracies.std()*100,
    'Test Data Accuracy': accuracy_score(y_test, y_pred)*100,
    'Test Data Precision': precision,
    'Test Data Recall': recall,
    'Test Data Specificity': specificity
}

# Append using pd.concat
result_comparison = pd.concat(
    [result_comparison, pd.DataFrame([new_row])],
    ignore_index=True
);


y_test_pred = classifier.predict(X_test_sub)

submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_test_pred
})

submission.to_csv("Decision_Tree.csv", index=False)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100
print("Precision: {:.2f} %".format(precision))
recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
print("Recall: {:.2f} %".format(recall))
specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100
print("Specificity: {:.2f} %".format(specificity))

new_row = {
    'Model': 'Random Forest',
    'Cross Validation Mean Accuracy':accuracies.mean()*100,
    'Cross Validation Standard Deviation': accuracies.std()*100,
    'Test Data Accuracy': accuracy_score(y_test, y_pred)*100,
    'Test Data Precision': precision,
    'Test Data Recall': recall,
    'Test Data Specificity': specificity
}

result_comparison = pd.concat(
    [result_comparison, pd.DataFrame([new_row])],
    ignore_index=True
);


y_test_pred = classifier.predict(X_test_sub)

submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_test_pred
})

submission.to_csv("Random_Forest.csv", index=False)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100
print("Precision: {:.2f} %".format(precision))
recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
print("Recall: {:.2f} %".format(recall))
specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100
print("Specificity: {:.2f} %".format(specificity))

new_row = {
    'Model': 'Naive Bayes',
    'Cross Validation Mean Accuracy':accuracies.mean()*100,
    'Cross Validation Standard Deviation': accuracies.std()*100,
    'Test Data Accuracy': accuracy_score(y_test, y_pred)*100,
    'Test Data Precision': precision,
    'Test Data Recall': recall,
    'Test Data Specificity': specificity
}

result_comparison = pd.concat(
    [result_comparison, pd.DataFrame([new_row])],
    ignore_index=True
);


y_test_pred = classifier.predict(X_test_sub)

submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_test_pred
})

submission.to_csv("GaussianNB.csv", index=False)


    !pip install xgboost


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100
print("Precision: {:.2f} %".format(precision))
recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
print("Recall: {:.2f} %".format(recall))
specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100
print("Specificity: {:.2f} %".format(specificity))

new_row = {
    'Model': 'XGBoost',
    'Cross Validation Mean Accuracy':accuracies.mean()*100,
    'Cross Validation Standard Deviation': accuracies.std()*100,
    'Test Data Accuracy': accuracy_score(y_test, y_pred)*100,
    'Test Data Precision': precision,
    'Test Data Recall': recall,
    'Test Data Specificity': specificity
}

result_comparison = pd.concat(
    [result_comparison, pd.DataFrame([new_row])],
    ignore_index=True
);


y_test_pred = classifier.predict(X_test_sub)

submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_test_pred
})

submission.to_csv("xgboost.csv", index=False)


  !pip install catboost


from catboost import CatBoostClassifier
classifier = CatBoostClassifier()

classifier.fit(X_train, y_train, 
                 eval_set=(X_train, y_train),
                 verbose=False)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100
print("Precision: {:.2f} %".format(precision))
recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
print("Recall: {:.2f} %".format(recall))
specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100
print("Specificity: {:.2f} %".format(specificity))

new_row = {
    'Model': 'CatBoost',
    'Cross Validation Mean Accuracy':accuracies.mean()*100,
    'Cross Validation Standard Deviation': accuracies.std()*100,
    'Test Data Accuracy': accuracy_score(y_test, y_pred)*100,
    'Test Data Precision': precision,
    'Test Data Recall': recall,
    'Test Data Specificity': specificity
}

result_comparison = pd.concat(
    [result_comparison, pd.DataFrame([new_row])],
    ignore_index=True
);


y_test_pred = classifier.predict(X_test_sub)

submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_test_pred
})

submission.to_csv("catboost.csv", index=False)


 !pip install lightgbm


from lightgbm import LGBMClassifier
classifier = LGBMClassifier(n_jobs=-1)
classifier.fit(X_train, y_train);

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

precision = cm[0,0]/(cm[0,0]+cm[1,0]) * 100
print("Precision: {:.2f} %".format(precision))
recall = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
print("Recall: {:.2f} %".format(recall))
specificity = cm[1,1]/(cm[1,1]+cm[1,0]) * 100
print("Specificity: {:.2f} %".format(specificity))

new_row = {
    'Model': 'Light GBM',
    'Cross Validation Mean Accuracy':accuracies.mean()*100,
    'Cross Validation Standard Deviation': accuracies.std()*100,
    'Test Data Accuracy': accuracy_score(y_test, y_pred)*100,
    'Test Data Precision': precision,
    'Test Data Recall': recall,
    'Test Data Specificity': specificity
}

result_comparison = pd.concat(
    [result_comparison, pd.DataFrame([new_row])],
    ignore_index=True
);


y_test_pred = classifier.predict(X_test_sub)

submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_test_pred
})

submission.to_csv("light_gbm.csv", index=False)


result_comparison




