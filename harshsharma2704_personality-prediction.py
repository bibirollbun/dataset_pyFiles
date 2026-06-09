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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


data.head()


print('\n training data')
print(data.isna().mean()*100)

print('\n test data')
print(test.isna().mean()*100)


print('training data:',data.duplicated().sum())
print('test data:',test.duplicated().sum())



test.head()


data.info()


# data['Stage_fear']=pd.to_numeric(data['Stage_fear'],errors='coerce')
# data['Drained_after_socializing']=pd.to_numeric(data['Drained_after_socializing'],errors='coerce')

# test['Stage_fear']=pd.to_numeric(test['Stage_fear'],errors='coerce')
# test['Drained_after_socializing']=pd.to_numeric(test['Drained_after_socializing'],errors='coerce')



# data['Introvert_Score'] = data['Time_spent_Alone'] + data['Stage_fear'] + data['Drained_after_socializing']
# data['Extrovert_Score'] = data['Social_event_attendance'] + data['Going_outside'] + data['Friends_circle_size'] + data['Post_frequency']
# data['Net_Social_Score'] = data['Extrovert_Score'] - data['Introvert_Score']


# test['Introvert_Score'] = test['Time_spent_Alone'] + test['Stage_fear'] + test['Drained_after_socializing']
# test['Extrovert_Score'] = test['Social_event_attendance'] + test['Going_outside'] + test['Friends_circle_size'] + test['Post_frequency']
# test['Net_Social_Score'] = test['Extrovert_Score'] - test['Introvert_Score']



X = data.drop(columns=['id','Personality'])
y= data['Personality']


num_cols =X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object','category']).columns


for i in num_cols:
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    sns.kdeplot(x=X[i])
    plt.subplot(122)
    sns.boxplot(x=X[i])


from sklearn.preprocessing import PowerTransformer
num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='mean')),
    ('power',PowerTransformer(method='yeo-johnson'))
])

cat_pipe =Pipeline ([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])

preprocessing  =ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])


X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_valid_enc = le.fit_transform(y_valid)


model_xgb = Pipeline([
    ('preprocess',preprocessing),
    ('algo',XGBClassifier(learning_rate=0.1,max_depth=3,gamma=2,sampling_method='uniform',subsample=0.7,reg_lambda=5,reg_alpha=3))
])
model_xgb.fit(X_train,y_train_enc)
y_pred = model_xgb.predict(X_valid)
print('Validation score',accuracy_score(y_valid_enc,y_pred))

y_pred_train = model_xgb.predict(X_train)
print('Training score',accuracy_score(y_train_enc,y_pred_train))


model_cat =Pipeline([
    ('preprocess',preprocessing),
    ('algo',CatBoostClassifier(n_estimators=120,max_depth=12,l2_leaf_reg=8,verbose=False))
])
model_cat.fit(X_train,y_train)
y_pred = model_cat.predict(X_valid)
print('validation score',accuracy_score(y_valid,y_pred))

y_pred_train = model_cat.predict(X_train)
print('training score',accuracy_score(y_train,y_pred_train))



from sklearn.metrics import classification_report
print(classification_report(y_valid,y_pred))


model_lgb = Pipeline([
    ('preprocess',preprocessing),
    ('algo',LGBMClassifier(n_estimators=1500,max_depth=15,learning_rate=0.65,reg_lambda=2,reg_alpha=3,verbose=-1))
])
model_lgb.fit(X_train,y_train)
y_pred = model_lgb.predict(X_valid)
print('validation score',accuracy_score(y_valid,y_pred))

y_pred_train = model_lgb.predict(X_train)
print('training score',accuracy_score(y_train,y_pred_train))


model_rf =Pipeline([
    ('preprocess',preprocessing),
    ('algo',RandomForestClassifier(n_estimators=500,class_weight='balanced',max_depth=12,criterion='log_loss',oob_score=True,bootstrap=True,min_samples_leaf=10,min_samples_split=20))
]) 

model_rf.fit(X_train,y_train)
y_pred = model_rf.predict(X_valid)
print('validation score',accuracy_score(y_valid,y_pred))

y_pred_train = model_rf.predict(X_train)
print('training score',accuracy_score(y_train,y_pred_train))


model_rf.named_steps['algo'].oob_score_


from sklearn.ensemble import VotingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
estimators = [
    ('lgbm',LGBMClassifier(learning_rate=0.05,max_depth=3,n_estimators=1200,verbose=-1)),
    ('xgb',AdaBoostClassifier(learning_rate=0.04,n_estimators=200,estimator=DecisionTreeClassifier(max_depth=3)))
]

model_vote = Pipeline([
    ('preprocess',preprocessing),
    ('algo',VotingClassifier(estimators = estimators,n_jobs=-1))
])

model_vote.fit(X_train,y_train)
y_pred  = model_vote.predict(X_valid)
print('validation score',accuracy_score(y_valid,y_pred))

y_pred_train = model_vote.predict(X_train)
print('training score',accuracy_score(y_train,y_pred_train))


from sklearn.metrics import classification_report
print(classification_report(y_valid,y_pred))


y_pred_test = model_vote.predict(test)
output = pd.DataFrame({'id':test['id'],'Personality':y_pred_test})
output.to_csv('submission.csv',index=False)

