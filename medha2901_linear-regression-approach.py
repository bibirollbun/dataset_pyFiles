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
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


##Loading the data

train_set = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/train.csv')
train_set.head()
train_set.info()     


test_set = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/test.csv')
test_set.head()
test_set.info()


sample_submission = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/sample_submission.csv')
sample_submission.info()


##DATA PREPROCESSING
#Checking for outliers

for col in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']:
    sns.boxplot(x=train_set[col])
    plt.title(f"Boxplot of {col}")
    plt.show()



#Capping outliers in f1, f2, f3, f4, f5, f7 and f8

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Cap the values at the lower and upper bounds
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

for col in ['f1', 'f2', 'f3', 'f4', 'f5', 'f7', 'f8']:
    train = cap_outliers(train_set, col)



for col in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']:
    sns.boxplot(x=train[col])
    plt.title(f"Boxplot of {col} After Handling Outliers")
    plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = train_set.copy() 

# Scale the features 
scaled_values = scaler.fit_transform(train_set.iloc[:, 1:])  # Scale all features except 'target'
train_scaled.iloc[:, 1:] = scaled_values  # Overwrite columns with scaled float values



##Standardizing the training set
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = train_set.copy()
train_scaled.iloc[:, 1:] = scaler.fit_transform(train_set.iloc[:, 1:])


##Normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_normalized = train_set.copy()
train_normalized.iloc[:, 1:] = scaler.fit_transform(train_set.iloc[:, 1:])



from sklearn.model_selection import train_test_split

X = train_set.drop(columns=['target']) 
y = train_set['target']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print("R^2 Score:", r2_score(y_val, y_pred))



test_scaled = test_set.copy()  # Apply the same scaling as training data
test_scaled.iloc[:, 1:] = scaler.transform(test_set.iloc[:, 1:])  # Assuming test data doesn't have 'target'

test_predictions = model.predict(test_scaled.iloc[:, 1:])
test_set['target'] = test_predictions



submission = test_set[['id', 'target']]
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")





