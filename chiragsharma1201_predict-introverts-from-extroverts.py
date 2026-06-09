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


d=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")



d.replace([np.inf, -np.inf], np.nan, inplace=True) 


d.head()


d.shape


d.describe()


d.columns


d.index


d.isnull().sum()


d.info()


d['Personality'].value_counts().plot(kind='bar') # check personality 


d1=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


d1.replace([np.inf, -np.inf], np.nan, inplace=True)


d1.head()


d1.shape


d1.columns


d1.describe()


d1.isnull().sum()


numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing'] # numerical and catgeorial data columns


for i in numerical_cols:
    d[i].fillna(d[i].median())
    d1[i].fillna(d[i].median()) # fill numerical column with median (null one)


for i in categorical_cols:
    d[i].fillna(d[i].mode()[0])
    d1[i].fillna(d[i].mode()[0]) # fill categorical column with mode (null one)


from sklearn.preprocessing import LabelEncoder
l= {} # apply label encoding on categorical data columns
for i in categorical_cols + ['Personality']:
    l[i]= LabelEncoder()
    d[i] = l[i].fit_transform(d[i])


for i in ['Stage_fear', 'Drained_after_socializing']:
    d1[i] = l[i].transform(d1[i]) 


features=['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing', 'Friends_circle_size','Post_frequency']


ans='Personality'


d.head()


d['Stage_fear'].value_counts().plot(kind='bar')


d['Drained_after_socializing'].value_counts().plot(kind='bar')


import seaborn as sns 
c=d[features].corr()
sns.heatmap(c,annot=True,cmap='coolwarm') # heatmap to check relation b/w each other for various columns 


d.groupby('Personality')['Time_spent_Alone'].size()


X=d[features] # create input ans output variable for result 
y=d[ans]
X_val = d1[features]


from sklearn.model_selection import train_test_split,RandomizedSearchCV
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


X_train.shape # after train_test_split analyze your shape 


X_test.shape


from catboost import CatBoostClassifier # apply catboost classifier for easily enchance model performance for structured data (catgeorical mostly)
x= CatBoostClassifier(verbose=0,cat_features=categorical_cols)


param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [200, 500, 1000],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0, 0.5, 1],
    'border_count': [32, 64, 128],
    'subsample': [0.6, 0.8, 1.0] # performed hypertuning by using various parameters 
}


from sklearn.model_selection import StratifiedKFold
r= RandomizedSearchCV( # apply randomized search cv here 
    x,
    param_distributions=param_dist,
    n_iter=20,  
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1,random_state=2
)


r.fit(X_train,y_train,eval_set=(X_test, y_test)) # fit your data 


y_pred=r.predict(X_test) # predict  personality to check introverts from extroverts 


y_pred1= r.best_estimator_


y_pred


y_pred1


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) # accuracy score 


val_pred= r.predict(X_val)
val_pred_labels = l['Personality'].inverse_transform(val_pred) # inverse transform means 1 0 form again converted to Introvert Extrovert 



submission = pd.DataFrame({
    'id': d1['id'],
    'Personality': val_pred_labels # submission 
})


submission.to_csv('submission.csv', index=False)


submission.shape


submission.head()


submission['Personality'].value_counts() # conclusion is that 1555 people are introvert and 4620 are extrovert out of 6175 

