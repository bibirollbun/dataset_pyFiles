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


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train


pap = test['id']
#Drop id as it is of no use
test.drop('id',axis = 1,inplace = True)
train.drop('id',axis = 1,inplace = True)


#Check for null values
train.isnull().sum()


y = train['Personality']
X = train.drop('Personality',axis = 1)
#Split data into training and test set
import seaborn as sns
#Here I have drawn all the plots of the columns of training set
for i in X.columns:
    sns.displot(X[i])
num_cols = [x for x in train if train[x].dtype!='object']    



#Fill the missing values using median in numeric and most frequent in categorical columns

from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(strategy = 'median')
for i in num_cols:
    train[i] = Imputer.fit_transform(train[[i]])
    test[i] = Imputer.fit_transform(test[[i]])
x = test.shape[0]    

Impute = SimpleImputer(strategy = 'most_frequent')    
for i in train.columns:
    if train[i].dtype=='object' and train[i].isnull().sum()>0:
        train[i] = Impute.fit_transform(train[[i]]).reshape(18524,)
        test[i] = Impute.fit_transform(test[[i]]).reshape(x,)


test


train


#Apply log transformation to Time SPENT ALONE to normalize it as it is skewed
def func(X):
    if X<2.2:
        return 0
    elif X<4.4:
        return 1
    elif X<6.6:
        return 2
    elif X<8.8:
        return 3
    elif X<11:
        return 4
    else:
        return 5
def func1(Y):
    if Y=='Extrovert':
        return 0
    else:
        return 1
        
#Also here i have encoded Personality in 0 and 1 for easy understanding

train['Personality'] = train['Personality'].apply(func1)   
train['Time_spent_Alone'] = train['Time_spent_Alone'].apply(func)
test['Time_spent_Alone'] = test['Time_spent_Alone'].apply(func)


#Then I have target encoded the categorical variables as it has high correlation with the target column

from sklearn.preprocessing import TargetEncoder
encoder = TargetEncoder()
train['Stage_fear'] = encoder.fit_transform(train[['Stage_fear']],train['Personality'])
test['Stage_fear'] = encoder.transform(test[['Stage_fear']])
train['Drained_after_socializing'] = encoder.fit_transform(train[['Drained_after_socializing']],train['Personality'])
test['Drained_after_socializing'] = encoder.transform(test[['Drained_after_socializing']])
train


#Feature engineering to high corr features
train['total'] = train['Stage_fear']+train['Drained_after_socializing']
test['total'] = test['Stage_fear']+test['Drained_after_socializing']


X = train['Personality']
y = train.drop('Personality',axis = 1)
from lightgbm import LGBMClassifier as LGBM
model = LGBM(n_estimators = 1000,learning_rate = 0.03,random_state = 42)
model.fit(y,X)


oupu = model.predict(test)


data = pd.DataFrame({'id' : pap , 'Personality' : oupu})
def fu(X):
    if X==0:
        return 'Extrovert'
    else:
        return 'Introvert'
data['Personality'] = data['Personality'].apply(fu)        
data.to_csv('submission.csv',index = False)


data


train['Time_spent_Alone']




