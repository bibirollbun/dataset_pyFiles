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


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


numeric_cols_train = train.select_dtypes(include=np.number).columns
medians_train = train[numeric_cols_train].median()
train[numeric_cols_train] = train[numeric_cols_train].fillna(medians_train)
numeric_cols_test = test.select_dtypes(include=np.number).columns
medians_test = test[numeric_cols_test].median()
test[numeric_cols_test] = test[numeric_cols_test].fillna(medians_test)
le=LabelEncoder()
train['Stage_fear']=le.fit_transform(train['Stage_fear'])
train['Personality']=le.fit_transform(train['Personality'])
train['Drained_after_socializing']=le.fit_transform(train['Drained_after_socializing'])
test['Stage_fear']=le.fit_transform(test['Stage_fear'])
test['Drained_after_socializing']=le.fit_transform(test['Drained_after_socializing'])


from sklearn.model_selection import train_test_split
y = train['Personality']
x = train.drop(['id', 'Personality'], axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=42)


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import StackingClassifier
dt_params = {
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 4, 6, 8],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
dt_grid.fit(x, y)
best_dt = dt_grid.best_estimator_
print("ðŸŒ³ En iyi DecisionTree:", dt_grid.best_params_)
from sklearn.ensemble import RandomForestClassifier
rf_params = {
    'n_estimators': [50, 100,120], 
    'max_depth': [3, 4, 6],    
    'min_samples_split': [2, 5, 10],   
    'min_samples_leaf': [1, 2, 4],    
    'bootstrap': [True, False]        
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rf_grid.fit(x, y)  
best_rf = rf_grid.best_estimator_
print("En iyi parametreler:", rf_grid.best_params_)




from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

base_models = [
    ('dt', best_dt),   
    ('rf', best_rf)]

stackmodel = StackingClassifier(
    estimators=base_models,
    final_estimator=RidgeClassifier(),
    passthrough=False,
    cv=5  
)





from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
accuracy_lst = []

skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    print(f"\nâœ… Fold {fold + 1} iÅŸleniyor...")
    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    stackmodel.fit(x_train,y_train)
    y_pred_label = stackmodel.predict(x_val)
    acc_score = accuracy_score(y_val, y_pred_label)
    accuracy_lst.append(acc_score)


acc_score


test_features = test.drop(columns=['id'])
prediction = stackmodel.predict(test_features)




mapping_dict = {0: 'Extrovert', 1: 'Introvert'}
prediction_string = np.array([mapping_dict[val] for val in prediction])
prediction_string


submission=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission.head()



submission['id']=test['id']
submission['Personality']=prediction_string
submission.to_csv("submission.csv", index=False)





