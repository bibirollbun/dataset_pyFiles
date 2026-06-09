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


from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('../input/predict-who-is-more-influential-in-a-social-network/train.csv')
test = pd.read_csv('../input/predict-who-is-more-influential-in-a-social-network/test.csv')


train.head()


test.head()


print(train.shape, test.shape)


print('Null Values in Train:',train.isna().values.any())
print('Null Values in Test:',test.isna().values.any())


X = train.drop(columns=['Choice'])
y = train['Choice']


rf = RandomForestClassifier(n_estimators=100,random_state= 24)
rf.fit(X,y)


y_hat = rf.predict(test)


sub = pd.DataFrame({"Id": list(range(1,len(y_hat)+1)),
                         "Choice":y_hat})


sub.to_csv('submission.csv',index=False)




