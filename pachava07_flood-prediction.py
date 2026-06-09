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

train = pd.read_csv("/kaggle/input/playground-series-s4e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e5/test.csv")


print(train.shape)
print(test.shape)


train.info()


train.head(5)


train.describe()


train.describe().T


train.isnull().mean()





from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


x = train.drop(['FloodProbability','id'], axis= 1)
y = train['FloodProbability']


len(x.columns)


x_train, x_valid , y_train, y_valid = train_test_split(x,y,test_size= 0.3,random_state = 42)


model = LinearRegression()


model.fit(x_train,y_train)


y_pred = model.predict(x_valid)


print('r2:', r2_score(y_valid,y_pred))


print('mean:',mean_squared_error(y_valid,y_pred))


train.columns


test.columns


test['id']


tests =test.drop('id',axis = 1)


final_output = model.predict(tests)
final_output = np.round(final_output,2)


final_output_df = pd.DataFrame({'id': test['id'],'FloodProbability': final_output})


final_output_df.to_csv('floods.csv',index=False)

