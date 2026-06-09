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
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 


sample_submission = pd.read_csv('/kaggle/input/pascmlsig/sample_submission.csv')
sample_submission.head()


train_data = pd.read_csv('/kaggle/input/pascmlsig/train.csv')
train_data.head()


test_data = pd.read_csv('/kaggle/input/pascmlsig/test.csv')
test_data.head()


train_data.info()


train_data.isnull().sum()


train_data.shape


x = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]



X_train,X_test,y_train,y_test  = train_test_split(x,y, random_state= 42, test_size=0.2)


# Before standardized the data display as

plt.figure(figsize=(15, 6))  # Increase figure size

plt.subplot(1, 2, 1)
X_train.plot.kde(ax=plt.gca())  # Use current subplot axis
plt.title("X_train KDE")

plt.subplot(1, 2, 2)
X_test.plot.kde(ax=plt.gca())   # Use current subplot axis
plt.title("X_test KDE")

plt.tight_layout()  # Call this after plotting
plt.show()



# Now perfrom the standarlization tecchnique to it

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
test_data = sc.fit_transform(test_data)


# After standardized the data display as
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
plt.figure(figsize=(15, 6))  # Increase figure size

plt.subplot(1, 2, 1)
X_train_df.plot.kde(ax=plt.gca())  # Use current subplot axis
plt.title("X_train KDE")

plt.subplot(1, 2, 2)
X_test_df.plot.kde(ax=plt.gca())   # Use current subplot axis
plt.title("X_test KDE")

plt.tight_layout()  # Call this after plotting
plt.show()



le = LinearRegression()
le.fit(X_train,y_train)
y_pred_test = le.predict(X_test)
y_pred_train = le.predict(X_train)


pd.DataFrame(y_pred_test)


pd.DataFrame(y_pred_train)


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
Model_R2_Score_test  = r2_score(y_test,y_pred_test)
Model_R2_Score_train  = r2_score(y_train,y_pred_train)
Model_Mean_Squared_Error_test = mean_squared_error(y_test,y_pred_test)
Model_Mean_Squared_Error_train = mean_squared_error(y_train,y_pred_train)
Model_Mean_absolute_Error_test = mean_absolute_error(y_test,y_pred_test)
Model_Mean_absolute_Error_train = mean_absolute_error(y_train,y_pred_train)


print ('Model R2 Testing Score: ', Model_R2_Score_test)
print ('Model R2 Training Score: ', Model_R2_Score_train)
print ('Model MSE Testing Score: ', Model_Mean_Squared_Error_test)
print ('Model MAE Testing Score: ', Model_Mean_absolute_Error_test)
print ('Model MSE Training Score: ', Model_Mean_Squared_Error_train)
print ('Model MAE Training Score: ', Model_Mean_absolute_Error_train)


if Model_Mean_Squared_Error_train < Model_Mean_Squared_Error_test or Model_R2_Score_train > Model_R2_Score_test:
    if abs(Model_R2_Score_train - Model_R2_Score_test) > 0.1:
        print('Model may be overfitting to the training data')
    else:
        print('Model Performs reasonably well, but check slight overfitting')
elif Model_Mean_Squared_Error_train > Model_Mean_Squared_Error_test and Model_R2_Score_train < Model_R2_Score_test:
    print('Model may be underfitting, Consider increasing model complexity')
else:
    print(' Model has balanced fit on both training and testing data') 

