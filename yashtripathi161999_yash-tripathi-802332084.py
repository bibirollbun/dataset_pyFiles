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
train = pd.read_csv('/kaggle/input/thapar-summer-school-2025-hack-ii/train.csv')
test = pd.read_csv('/kaggle/input/thapar-summer-school-2025-hack-ii/test.csv')
train.head()
# test.head()



X=train.drop(['yield', 'id'], axis=1)
y=train['yield']
X_test=test.drop('id', axis=1)



# from sklearn.ensemble import RandomForestRegressor


# model=RandomForestRegressor(random_state=42)
# model.fit(X,y)
# predctions=model.predict(X_test)





from sklearn.linear_model import LinearRegression
lr_model=LinearRegression()
lr_model.fit(X,y)
lr_pred=model.predict(X_test)




submission=pd.DataFrame({
    'id':test['id'],
    'yield':predctions
})
submission.to_csv('Yash_Tripathi_802332084.csv',index=False)
submission.head()
print("Submission file created successfully!")




