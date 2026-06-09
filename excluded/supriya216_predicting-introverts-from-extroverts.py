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


df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df.head()


test.tail()


df['stage_fear']=np.where(df['Stage_fear']=='No',0,1)
df['drain']=np.where(df['Drained_after_socializing']=='No',0,1)
df['personality']=np.where(df['Personality']=='Introvert',1,0)


df.head()


test['stage_fear']=np.where(test['Stage_fear']=='No',0,1)
test['drain']=np.where(test['Drained_after_socializing']=='No',0,1)



X=df.drop(columns=['id','Personality','Stage_fear','Drained_after_socializing','personality'])
y=df['personality']


from sklearn.model_selection import  train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from sklearn.feature_selection import SelectKBest, mutual_info_classif

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feature_select', SelectKBest(mutual_info_classif, k=4)),
    ('scaler', StandardScaler()),
    ('svc', LinearSVC(C=3.7554,max_iter=10000))
])



from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV


# Parameter space to search
param_dist = {
    'svc__C': uniform(loc=0.01, scale=10),  # C in range [0.01, 10]
}

# Perform search
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,  # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1  # Uses all CPU cores
)


pipeline.fit(X_train,y_train)


y_pred=pipeline.predict(X_valid)


y_valid.value_counts()


print(classification_report(y_pred,y_valid))


X_test=test.drop(columns=['Stage_fear','Drained_after_socializing','id'])


y_test=pipeline.predict(X_test)


y_test


label_map = {0: "Extrovert", 1: "Introvert"}
predicted_labels = [label_map[val] for val in y_test]



submission1 = pd.DataFrame({
    "id": test["id"],
    "Personality": predicted_labels  
})



submission1['Personality'].value_counts()


submission1.to_csv("submission1.csv", index=False)


from lightgbm import LGBMClassifier


pipeline1 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feature_select', SelectKBest(mutual_info_classif, k=6)),
    ('scaler', StandardScaler()),  # Optional for tree models, can be skipped
    ('lgbm', LGBMClassifier(
      
        class_weight='balanced',  # helpful for class imbalance
        random_state=42,
        subsample=0.7, 
        num_leaves= 63,
        n_estimators= 100, min_child_samples= 20, learning_rate= 0.01, colsample_bytree=0.6
    ))
])


pipeline1.fit(X_train,y_train)


y_pred1=pipeline1.predict(X_valid)


print("âœ… Accuracy:", accuracy_score(y_valid, y_pred1))
print("\nðŸ“‹ Classification Report:\n", classification_report(y_valid, y_pred1))


y_test2=pipeline1.predict(X_test)


label_map = {0: "Extrovert", 1: "Introvert"}
predicted_labels1 = [label_map[val] for val in y_test2]


submission2 = pd.DataFrame({
    "id": test["id"],
    "Personality": predicted_labels1  
})



submission2['Personality'].value_counts()


submission2.to_csv("submission2.csv", index=False)




