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


dataset=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
nan_row_count = dataset.isna().any(axis=1).sum()
print(nan_row_count)
dataset=dataset.dropna()
nan_row_count = dataset.isna().any(axis=1).sum()
print(nan_row_count)


X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])


print(X)


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X,y)


import pandas as pd
DATA=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


X_test=DATA.iloc[:,1:].values


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X_test[:,0]=le.fit_transform(X_test[:,0])


classifier.predict(X_test)


Predicted_Calorie=pd.DataFrame()
Predicted_Calorie['id']=DATA['id']
Predicted_Calorie['calories']=classifier.predict(X_test)


Predicted_Calorie.to_csv('Predicted_Calorie1.csv',index=False)




