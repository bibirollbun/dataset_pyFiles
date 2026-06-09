import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split



train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.info()


test.info()


print('Train\n\n',train.isnull().sum())
print('-------------------')
print('Test\n\n' , test.isnull().sum())


test['winddirection'].fillna(test['winddirection'].mode()[0], inplace=True)


train.describe()


train.head()


train['day'].unique()


plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()



train['month'] = ((train['day'] - 1) // 30 + 1).clip(upper=12)
train['season'] = train['month'] % 12 // 3 + 1
test['month'] = ((test['day'] - 1) // 30 + 1).clip(upper=12)
test['season'] = test['month'] % 12 // 3 + 1



train.drop(['day'] , axis=1 , inplace=True)
test.drop(['day'] , axis=1 , inplace=True)


train['Diff_Temp'] = train['maxtemp'] - train['mintemp']
test['Diff_Temp'] = test['maxtemp'] - test['mintemp']


train.drop(['maxtemp' , 'mintemp' , 'temparature' , 'dewpoint'] , axis=1 , inplace=True)
test.drop(['maxtemp' , 'mintemp' , 'temparature' , 'dewpoint'] , axis=1 , inplace=True)


sns.countplot(x= 'rainfall' , data=train)


train.groupby('season')['rainfall'].mean().plot(kind='line', figsize=(12, 6), title="Target Rate by season of Year")



train.groupby('month')['rainfall'].mean().plot(kind='line', figsize=(12, 6), title="Target Rate by month of Year")



train[['month', 'rainfall']].groupby('month').mean().rolling(window=7).mean().plot()



sns.boxplot(x='season', y='rainfall', data=train)



sns.pairplot(train, hue='rainfall')



fingure = plt.figure(figsize=(15, 10))
sns.boxplot(train)


X = train.drop(train[['rainfall' , 'id']], axis=1)
y = train['rainfall']


from imblearn.over_sampling import SMOTE
over_sampling = SMOTE()
X , y = over_sampling.fit_resample(X , y)



x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
Random_Forest = RandomForestClassifier()
Random_Forest.fit(x_train , y_train)
y_pred = Random_Forest.predict(x_test)
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))


acc = Random_Forest.score(x_test, y_test)
print(acc)


from sklearn.tree import DecisionTreeClassifier
Decision_Tree = DecisionTreeClassifier()
Decision_Tree.fit(x_train , y_train)
y_pred = Decision_Tree.predict(x_test)
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))


from sklearn.linear_model import LogisticRegression
Logistic_Regression = LogisticRegression()
Logistic_Regression.fit(x_train , y_train)
y_pred = Logistic_Regression.predict(x_test)
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(x_train , y_train)
y_pred = XGB.predict(x_test)
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))


accuracy = XGB.score(x_test, y_test)
print(accuracy)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['No Rainfall', 'Rainfall'],
            yticklabels=['No Rainfall', 'Rainfall'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest')
plt.show()



from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



model = RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=100)


model.fit(X,y)


X_test = test.drop(columns=['id'])
preds = model.predict(X_test)



submission = pd.DataFrame({
    'id': test['id'],
    'target': preds
})
submission.to_csv('submission.csv', index=False)





