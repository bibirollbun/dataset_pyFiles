#import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#read the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


#drop id column from the train_df
train_df = train_df.drop('id',axis = 1)


#drop the null values in the train and test dataset
train_df.dropna()
test_df.dropna()


train_df = train_df.drop('day',axis = 1)


#get the attributes and the target variable separately
X = train_df.iloc[:,:-1]
y = train_df.iloc[:,-1]
print(X)


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)


#do feature scaling here
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use same scaler on test data


from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(max_iter = 5000)


lr.fit(X_train,y_train)


#make pred and check accuracy
from sklearn.metrics import accuracy_score
y_pred = lr.predict(X_test)
print("Accuracy is :",accuracy_score(y_test,y_pred))


from sklearn.svm import LinearSVC


svc = LinearSVC(C = 0.0001,max_iter = 6000)
svc.fit(X_train,y_train)


#make pred and check accuracy
y_pred = svc.predict(X_test)
print("Accuracy :",accuracy_score(y_test,y_pred))


#finetune the SVM
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced'],
    'penalty': ['l2']  # 'l1' works only with solver='liblinear'
}

svc = LinearSVC(dual=False, random_state=42)

# Grid search with 5-fold CV
grid_search = GridSearchCV(linear_svc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_linear_svc = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)



svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print("Validation Accuracy:", accuracy_score(y_test, y_pred))


from sklearn.impute import SimpleImputer

# Impute missing values with the mean (use same strategy as training)
imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent'
X_test_imputed = imputer.fit_transform(test_df)  # If test_df is the raw data


X_test_imputed = X_test_imputed[:,2:]


#do feature scaling here
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_test_imputed = scaler.fit_transform(X_test_imputed)


y_pred = svc.predict(X_test_imputed)


results = pd.DataFrame({
    'id': test_df['id'],  # Preserve original IDs
    'rainfall_probability': y_pred  # Your model's predictions
})


results.to_csv('rainfall_predictions.csv', index=False)

