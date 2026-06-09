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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train=pd.read_csv('/kaggle/input/spam-emails12345/train.csv')
train.head()


train.info()


train = train.drop(['CustomerId','Surname'], axis=1)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['Gender']=le.fit_transform(train['Gender'])


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
encoded_data=ohe.fit_transform(train[['Geography']]).toarray()

encoded_train=pd.DataFrame(encoded_data,columns=ohe.get_feature_names_out(['Geography']))

train=pd.concat([train.drop('Geography',axis=1),encoded_train],axis=1)                                                                         


train.info()


X=train.drop('Exited',axis=1)
y=train['Exited']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=10,shuffle=True)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print('accuracy=',accuracy)


test=pd.read_csv('/kaggle/input/spam-emails12345/test.csv')
test.head()


test.info()


test = test.drop(['CustomerId','Surname'], axis=1)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
test['Gender']=le.fit_transform(test['Gender'])


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
encoded_data=ohe.fit_transform(test[['Geography']]).toarray()

encoded_test=pd.DataFrame(encoded_data,columns=ohe.get_feature_names_out(['Geography']))

test=pd.concat([test.drop('Geography',axis=1),encoded_test],axis=1)                                                                         


test.info()


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X,y)

test = test.reindex(columns=X.columns, fill_value=0)

y_pred=model.predict(test)


sample=pd.read_csv('/kaggle/input/spam-emails12345/sample_submission.csv')
sample.head()


submission = pd.DataFrame({
    'id': test['id'],
    'Exited': y_pred
})


submission.to_csv("submission.csv", index=False)




