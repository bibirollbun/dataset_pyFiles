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


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.info()


train.shape


display(train.head(10))


train.describe()


train.isnull().sum()


from sklearn.preprocessing import LabelEncoder

train = train.drop(columns=['id'])
testing_data = test.drop(columns=['id'])

#just for the object type data
train['Stage_fear'] = train['Stage_fear'].fillna('Missing')
train['Drained_after_socializing'] = train['Drained_after_socializing'].fillna('Missing')
train['Personality'] = train['Personality'].fillna('Missing')

#for testing data also
testing_data['Stage_fear'] = testing_data['Stage_fear'].fillna('Missing')
testing_data['Drained_after_socializing'] = testing_data['Drained_after_socializing'].fillna('Missing')


num_cols = train.select_dtypes(include='number').columns
for col in num_cols:
    train[col] = train[col].fillna(train[col].mean())
    testing_data[col] = testing_data[col].fillna(testing_data[col].mean())


print(testing_data.head())
print(train.head(10))


categorical_col = ['Stage_fear','Drained_after_socializing','Personality']
le_dict={}
for col in categorical_col:
    le = LabelEncoder()
    try:
        train[col] = le.fit_transform(train[col].astype(str))
        if col in testing_data.columns:
            testing_data[col] = le.transform(testing_data[col].astype(str))
        le_dict[col] = le
    except Exception as e:
        print(f"Error encoding {col}: {str(e)}")
        continue


testing_data.head()


train.head()


train.isnull().sum()


print(testing_data.shape)
print(test.shape)


corr = train.select_dtypes(include=np.number).corr()
sns.heatmap(corr,annot =True,fmt=".2f")


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus


X = train.drop(columns=['Personality'])
Y = train['Personality']


X_train , X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.25,random_state = 42)


dt = DecisionTreeClassifier(criterion = "gini", max_depth=5, random_state=42)


dt.fit(X_train,Y_train)


predictions = dt.predict(X_test)
accuracy = accuracy_score(Y_test,predictions)
print(f'accuracy score is {accuracy}')


from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(X_train,Y_train)
xgb_predictions = xgb.predict(X_test)
xgb_accuracy = accuracy_score(Y_test,xgb_predictions)
print(xgb_predictions)
print(f'accuracy score for xgboost : {xgb_accuracy}')


#dtc_predict = dt.predict(testing_data)
xgb_predict = xgb.predict(testing_data)
submission = pd.DataFrame({
    'id':test['id'],
    'Personality': le.inverse_transform(xgb_predict)
})

print(submission)
submission.to_csv("submission.csv",index=False)

