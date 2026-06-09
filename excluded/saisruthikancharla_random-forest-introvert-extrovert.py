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


train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.concat([train,test])


data.isnull().sum()


data.info()


data['Time_spent_Alone']=data['Time_spent_Alone'].fillna(data['Time_spent_Alone'].median())
data['Social_event_attendance']=data['Social_event_attendance'].fillna(data['Social_event_attendance'].median())
data['Going_outside']=data['Going_outside'].fillna(data['Going_outside'].median())
data['Friends_circle_size']=data['Friends_circle_size'].fillna(data['Friends_circle_size'].median())
data['Post_frequency']=data['Post_frequency'].fillna(data['Post_frequency'].median())

data['Stage_fear']=data['Stage_fear'].fillna(data['Stage_fear'].mode()[0])
data['Drained_after_socializing']=data['Drained_after_socializing'].fillna(data['Drained_after_socializing'].mode()[0])


data.isnull().sum()


train = data[~data['Personality'].isna()].copy()
test = data[data['Personality'].isna()].copy()


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
train['Drained_after_socializing']=encoder.fit_transform(train['Drained_after_socializing'])
train['Stage_fear']=encoder.fit_transform(train['Stage_fear'])
test['Drained_after_socializing']=encoder.transform(test['Drained_after_socializing'])
test['Stage_fear']=encoder.transform(test['Stage_fear'])


train['Personality']=train['Personality'].map({'Extrovert':0,'Introvert':1})


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


x,y=train.iloc[:,:-1],train.iloc[:,-1]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


test.iloc[:,:-1]=scaler.transform(test.iloc[:,:-1])


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


models={
    "Logisitic Regression":LogisticRegression(),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "Gradient Boost":GradientBoostingClassifier(),
    "Adaboost":AdaBoostClassifier(),
    "Xgboost":XGBClassifier()
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(x_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred) 
    model_train_precision = precision_score(y_train, y_train_pred) 
    model_train_recall = recall_score(y_train, y_train_pred)


    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred)
    model_test_precision = precision_score(y_test, y_test_pred) 
    model_test_recall = recall_score(y_test, y_test_pred) 
    print(list(models.keys())[i])     
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))  
    print('----------------------------------')
        
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))

    
    print('='*35)
    print('\n')


from sklearn.model_selection import cross_val_score
for name, model in models.items():
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} accuracy Score: {scores.mean():.4f}")


rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "max_features": [3,5,7,9],
    "min_samples_split": [2, 10]
}

gradient_params = {
    "loss": ['log_loss'],
    "criterion": ['friedman_mse'],
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "min_samples_split": [2, 10]
}

logistic_params = {
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10]
}

randomcv_models=[
    ('RF',RandomForestClassifier(),rf_params),
    ('logistic',LogisticRegression(),logistic_params),
    ('GradientBoost',GradientBoostingClassifier(),gradient_params)
]


import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import RandomizedSearchCV
model_params={}
for name,model,params in randomcv_models:
    rsc=RandomizedSearchCV(estimator=model,param_distributions=params,n_iter=5,cv=3,verbose=2,n_jobs=-1)
    rsc.fit(x_train,y_train)
    model_params[name]=rsc.best_params_


for model_name in model_params:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_params[model_name])


models={
    
    "Random Forest":RandomForestClassifier(n_estimators=50,min_samples_split=10,max_features=7,max_depth=10),
    "logistic":LogisticRegression(penalty='l2',C=0.01),
    'GradientBoost':GradientBoostingClassifier(n_estimators=50,min_samples_split=10,max_depth=10,loss='log_loss',criterion='friedman_mse')
}


for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(x_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    model_train_accuracy = accuracy_score(y_train, y_train_pred) 
    model_train_precision = precision_score(y_train, y_train_pred)
    model_train_recall = recall_score(y_train, y_train_pred) 
    


    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred) 
    model_test_precision = precision_score(y_test, y_test_pred) 
    model_test_recall = recall_score(y_test, y_test_pred) 
     #Calculate Roc


    print(list(models.keys())[i])
       
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    

    
    print('='*35)
    print('\n')


rf=RandomForestClassifier(n_estimators=50,min_samples_split=10,max_features=7,max_depth=10)
rf.fit(x_train,y_train)
y_test_pred=rf.predict(test.iloc[:,:-1])


submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality'] = pd.Series(y_test_pred).map({0: 'Extrovert', 1: 'Introvert'})
submission.to_csv('my_submission.csv', index=False)

