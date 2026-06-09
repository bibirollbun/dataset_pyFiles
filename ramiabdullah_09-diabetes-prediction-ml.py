from IPython.display import Image


Image(filename='/kaggle/input/diabetes-prediction-image/Diabetes_Prediction_Image.jpg')


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('/kaggle/input/diabetes-prediction-from-medicalrecords/train.csv')
df_test = pd.read_csv('/kaggle/input/diabetes-prediction-from-medicalrecords/test.csv')


df_train.head()


df_test.head(5)


df_train.info()


df_test.info()


df_train.describe().T


df_train.columns


df_train.shape


df_train = df_train.drop(columns=['Id'])
df_test = df_test.drop(columns=['Id'])


corr = df_train.corr()
corr


plt.figure(figsize=(14, 7))
sns.heatmap(corr, annot=True, cmap='cividis')
plt.title('Correlation Matrix')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,7))

sns.histplot(data=df_train,
             x='Glucose',
             hue='Outcome',
             kde=True,
             palette='viridis',
             alpha=0.6,
             multiple="dodge")

plt.title('Distribution of Glucose Levels by Outcome')
plt.xlabel('Glucose Level')
plt.ylabel('Frequency')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,7))

sns.histplot(data=df_train,
             x='BMI',
             hue='Outcome',
             kde=True,
             color='viridis',
             alpha=0.6,
             multiple="dodge")

plt.title('Distribution of BMI by Outcome')
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Frequency')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,7))

sns.histplot(data=df_train,
             x='Age',
             hue='Outcome',
             kde=True,
             palette='viridis',
             alpha=0.6,
             multiple="dodge")

plt.title('Distribution of Age by Outcome')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,7))

sns.histplot(data=df_train,
             x='Insulin',
             hue='Outcome',
             kde=True,
             palette='viridis',
             alpha=0.6,
             multiple="dodge")

plt.title('Distribution of Insulin Levels by Outcome')
plt.xlabel('Insulin Level')
plt.ylabel('Frequency')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,7))

sns.scatterplot(data=df_train,
                x='SkinThickness',
                y='Insulin',
                hue='Outcome',
                palette='viridis',
                alpha=0.7,
                size='SkinThickness',
                sizes=(20, 200))

plt.title('Relationship between Skin Thickness and Insulin')
plt.xlabel('Skin Thickness')
plt.ylabel('Insulin Level')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,7))

sns.scatterplot(data=df_train,
                x='Glucose',
                y='Insulin',
                hue='Outcome',
                palette='viridis',
                alpha=0.7,
                size='Glucose',
                sizes=(20, 200))

plt.title('Relationship between Glucose and Insulin')
plt.xlabel('Glucose Level')
plt.ylabel('Insulin Level')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,7))

sns.scatterplot(data=df_train,
                x='BMI',
                y='DiabetesPedigreeFunction',
                hue='Outcome',
                palette='viridis',
                alpha=0.7,
                size='BMI',
                sizes=(20, 200))

plt.title('Relationship between BMI and Diabetes Pedigree Function')
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Diabetes Pedigree Function')
plt.show()


from sklearn.model_selection import train_test_split

X = df_train.drop('Outcome' , axis = 1)
y = df_train['Outcome']

X_train , X_val , y_train , y_val = train_test_split(X,y,test_size= 0.3 , random_state= 42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler


X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


scaling_test = scaler.transform(df_test)


from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression()
LogReg


param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200],
    'class_weight': [None, 'balanced'],
    'tol': [1e-4]
             }


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV (LogReg , param_grid , cv=5)
grid_search


grid_search.fit(X_train , y_train)


print('Best Parameters :' , grid_search.best_params_)


best_LogReg = grid_search.best_estimator_
best_LogReg


best_LogReg.fit(X_train , y_train)


LogReg_predict = best_LogReg.predict(X_val)
LogReg_predict


from sklearn.metrics import accuracy_score

LogReg_Accuracy = accuracy_score(y_val, LogReg_predict)
print (f'LogisticRegression Accuracy-score is : {LogReg_Accuracy:.2f}')


from sklearn.metrics import precision_score

LogReg_Precision = precision_score(y_val, LogReg_predict)
print (f'LogisticRegression Precision-score is : {LogReg_Precision:.2f}' )


from sklearn.metrics import recall_score

LogReg_Recall = recall_score(y_val, LogReg_predict)
print (f'LogisticRegression Recall-score is : {LogReg_Recall:.2f} ')


from sklearn.metrics import f1_score

LogReg_F1 = f1_score(y_val, LogReg_predict)
print(f'LogisticRegression F1-score is :{LogReg_F1:.2f}')


from sklearn.metrics import roc_curve

LogReg_fpr , LogReg_tpr , _ = roc_curve(y_val, LogReg_predict)
LogReg_fpr , LogReg_tpr , _


from sklearn.metrics import auc

LogReg_auc = auc(LogReg_fpr ,LogReg_tpr)
print(f'AUC: {LogReg_auc:.2f}')


from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()
DTC


param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'max_features': [None, 'sqrt', 'log2', 0.5]
             }


from sklearn.model_selection import GridSearchCV

grid_searsh = GridSearchCV(DTC , param_grid , cv=5)
grid_searsh


grid_searsh.fit(X_train,y_train)


print('Best Parameters :' , grid_searsh.best_params_)


best_DTC = grid_searsh.best_estimator_
best_DTC


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(best_DTC, X_train, y_train, cv=5)
print("Cross-Validation Accuracy Scores:", cv_scores)


best_DTC.fit(X_train,y_train)


DTC_predict =best_DTC.predict(X_val)


from sklearn.metrics import accuracy_score

DTC_Accuracy = accuracy_score(y_val, DTC_predict)
print(f'Decision Tree Accuracy-score is: {DTC_Accuracy:.2f}')


from sklearn.metrics import precision_score

DTC_Precision = precision_score(y_val, DTC_predict, average='weighted')
print(f'Decision Tree Precision-score is: {DTC_Precision:.2f}')


from sklearn.metrics import recall_score

DTC_Recall = recall_score(y_val, DTC_predict, average='weighted')
print(f'Decision Tree Recall-score is: {DTC_Recall:.2f}')


from sklearn.metrics import f1_score

DTC_F1 = f1_score(y_val, DTC_predict, average='weighted')
print(f'Decision Tree F1-score is: {DTC_F1:.2f}')


from sklearn.metrics import roc_curve, auc

DTC_fpr, DTC_tpr, _ = roc_curve(y_val, DTC_predict)


DTC_auc = auc(DTC_fpr, DTC_tpr)
print(f'AUC: {DTC_auc:.2f}')


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()
RFC


param_grid = {
     'n_estimators' : [50, 100],
     'max_depth' : [None, 5, 10, 15, 20],
     'max_features' : [None, 'sqrt', 'log2'],
     'min_samples_split': [2, 5, 10]
             }


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(RFC, param_grid, cv=5 ,n_jobs=-1, verbose=1)
grid_search


grid_search.fit(X_train, y_train)


print("Best Parameters:", grid_search.best_params_)


best_RFC = grid_search.best_estimator_
best_RFC


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(best_RFC, X, y, cv=5)
print("Cross-Validation Accuracy Scores:", cv_scores)


best_RFC.fit(X_train,y_train)


RFC_predict = best_RFC.predict(X_val)


from sklearn.metrics import accuracy_score

RFC_Accuracy = accuracy_score(y_val, RFC_predict)
print(f'Random Forest Accuracy-score is: {RFC_Accuracy:.2f}')


from sklearn.metrics import precision_score

RFC_Precision = precision_score(y_val, RFC_predict, average='weighted')
print(f'Random Forest Precision-score is: {RFC_Precision:.2f}')


from sklearn.metrics import recall_score

RFC_Recall = recall_score(y_val, RFC_predict, average='weighted')
print(f'Random Forest Recall-score is: {RFC_Recall:.2f}')


from sklearn.metrics import f1_score

RFC_F1 = f1_score(y_val, RFC_predict, average='weighted')
print(f'Random Forest F1-score is: {RFC_F1:.2f}')


from sklearn.metrics import roc_curve, auc

RFC_fpr, RFC_tpr, _ = roc_curve(y_val, RFC_predict)


RFC_auc = auc(RFC_fpr, RFC_tpr)
print(f'AUC: {RFC_auc:.2f}')


model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [LogReg_Accuracy, DTC_Accuracy, RFC_Accuracy],
    'Precision': [LogReg_Precision, DTC_Precision, RFC_Precision],
    'Recall': [LogReg_Recall, DTC_Recall, RFC_Recall],
    'F1-Score': [LogReg_F1, DTC_F1, RFC_F1],
    'AUC': [LogReg_auc, DTC_auc, RFC_auc]
})


model_comparison = model_comparison.sort_values(by='Accuracy', ascending=False)

model_comparison


from sklearn.ensemble import VotingClassifier

VOT = VotingClassifier(estimators=[('logreg', best_LogReg),('rf', best_RFC)],
                              voting='soft',
                              weights=[2, 1])


VOT_model = VOT.fit(X_train, y_train)


VOT_predict = VOT_model.predict(X_val)


from sklearn.metrics import accuracy_score

VOT_Accuracy = accuracy_score(y_val, VOT_predict)
print(f'Voting Classifier Accuracy-score is: {VOT_Accuracy:.2f}')


from sklearn.metrics import precision_score

VOT_Precision = precision_score(y_val, VOT_predict, average='weighted')
print(f'Voting Classifier Precision-score is:{VOT_Precision:.2f}')


from sklearn.metrics import recall_score

VOT_Recall = recall_score(y_val, VOT_predict, average='weighted')
print(f'Voting Classifier Recall-score is: {VOT_Recall:.2f}')


from sklearn.metrics import f1_score

VOT_F1 = f1_score(y_val, VOT_predict, average='weighted')
print(f'Voting Classifier F1-score is: {VOT_F1:.2f}')


from sklearn.metrics import roc_curve, auc

VOT_fpr, VOT_tpr, _ = roc_curve(y_val, VOT_predict)


VOT_auc = auc(VOT_fpr, VOT_tpr)
print(f'Voting Classifier AUC: {VOT_auc:.2f}')


df_test.columns


predictions = VOT_model.predict(scaling_test)


sampel_submission = pd.read_csv('/kaggle/input/diabetes-prediction-from-medicalrecords/sample_submission.csv')
sampel_submission


sampel_submission['Outcome'] = predictions


sampel_submission.to_csv('submission.csv', index=False)

