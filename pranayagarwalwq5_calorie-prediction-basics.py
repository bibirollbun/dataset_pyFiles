# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

import warnings
warnings.filterwarnings("ignore")



path1 = '/kaggle/input/playground-series-s5e5/train.csv'
path2 = '/kaggle/input/playground-series-s5e5/test.csv'

train_data = pd.read_csv(path1)

test_data = pd.read_csv(path2)


train_data['Sex'] = train_data['Sex'].map({'male':'0','female':'1'}).astype(float)
test_data['Sex'] = test_data['Sex'].map({'male':'0','female':'1'}).astype(float)


from ydata_profiling import ProfileReport

report = ProfileReport(train_data,title='Data Report')

report


train_data


from sklearn.model_selection import train_test_split

X = train_data.drop(['id','Calories'],axis=1)
y = train_data['Calories']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train1 = sc.fit_transform(X_train)
X_val1 = sc.transform(X_val)


from sklearn.linear_model import LinearRegression
from sklearn import metrics

for name,method in [('Linear regression', LinearRegression())]: 
    method.fit(X_train1,y_train)
    predict = method.predict(X_val1)

print('\nOriginal Model')
print('\nMethod: {}'.format(name))   

#Coefficents
print('\nIntercept: {:.2f}'.format(float(method.intercept_)))
coeff_table=pd.DataFrame(np.transpose(method.coef_),train_data.drop(['id','Calories'],axis=1).columns,columns=['Coefficients'])
print(coeff_table)
    
#R2,MAE,MSE and RMSE
print('\nR2: {:.2f}'.format(metrics.r2_score(y_val,predict)))
adjusted_r_squared = 1-(1-metrics.r2_score(y_val,predict))*(len(y)-1)/(len(y)-X.shape[1]-1)
print('Adj_R2: {:0.2f}'.format(adjusted_r_squared))
print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(y_val, predict)))  
print('Mean Squared Error: {:.2f}'.format(metrics.mean_squared_error(y_val, predict)))  
print('Root Mean Squared Error: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_val, predict)))) 


X_test = test_data.drop(columns=['id'])

test_preds = method.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'calories_burned': test_preds
})

# Save submission file
submission.to_csv('submission.csv', index=False)

