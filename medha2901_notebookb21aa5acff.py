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


train_set = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/train.csv')
train_set.info()     


test_set = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/test.csv')
test_set.info()     


train_set = train_set.astype({col: 'float64' for col in train_set.select_dtypes(include=['int64', 'float64']).columns})
test_set = test_set.astype({col: 'float64' for col in test_set.select_dtypes(include=['int64', 'float64']).columns})


print(train_set.dtypes)
print(test_set.dtypes)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score



print(train_set.isnull().sum())
print(test_set.isnull().sum())


# In train_set, dropping only target column
X = train_set.drop(columns=['target'])  
y = train_set['target'] 

# In test_set, drop the 'id' column
X_test = test_set.drop(columns=['id'])  


# Scaling features
scaler = StandardScaler()
X = scaler.fit_transform(X) 
X_test = scaler.transform(X_test)  


# Initializing Ridge Regression with cross-validation
ridge_model = Ridge(alpha=1.0) 

# Training the model
ridge_model.fit(X, y)


# Predicting on test set
test_predictions = ridge_model.predict(X_test)


# Splitting training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Training on training split
ridge_model.fit(X_train, y_train)

# Validating on validation split
val_predictions = ridge_model.predict(X_val)
r2 = r2_score(y_val, val_predictions)
print("Validation R² Score:", r2)



# submission file
submission = pd.DataFrame({
    'id': test_set['id'], 
    'target': test_predictions  
})

submission.to_csv('submission.csv', index=False)





