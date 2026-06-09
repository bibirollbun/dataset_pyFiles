# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/train.csv')
test  = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/test.csv')
lookup = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/feature_lookup.csv')#Loading Data
print(train.columns)
print(test.columns)

print(train.head())
print(train.info())
print(train.describe())#Simple Exploratory Data Analysis

X = train.drop(['ID', 'relationship_probability'], axis=1)
y = train['relationship_probability']#seprate features and target columns

X = pd.get_dummies(X)

X = X.fillna(0)
#Handle Missing Values

model = RandomForestRegressor(
    n_estimators=600,
    random_state=42,
    max_features='sqrt'
)

model.fit(X,y)#Training Model

test_ids = test['ID']
X_test = test.drop(['ID'], axis=1)
X_test = pd.get_dummies(X_test)
X_test = X_test.fillna(0)
#Test Preprocessing(Ensuring Test Data is in same format as Train data)

X_test = X_test.reindex(columns=X.columns,fill_value=0)#Ensuring Test has same columns as Train

predictions = model.predict(X_test)

predictions = np.clip(predictions, 0, 100)

submission = pd.DataFrame({
    'ID': test_ids,
    'relationship_probability': predictions
})#Creating Submissions

submission.to_csv('submission.csv',index=False)
print("Submission File Created")



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




