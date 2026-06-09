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
        import itertools
import warnings
import random
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import os
from sklearn.preprocessing import LabelEncoder

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train.head()


#Preparing Data
X=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen','Potassium', 'Phosphorous']
Y=['Fertilizer Name']
train['Soil Type']=train['Soil Type'].astype("category") #Making soil and crop type categorical
train['Crop Type']=train['Crop Type'].astype("category")
train['Fertilizer Name']=train['Fertilizer Name'].astype("category")
le = LabelEncoder()

#Converts fertilizer names to integer labels
train['Fertilizer Name Encoded'] = le.fit_transform(train['Fertilizer Name']) 


#Creating and Training model
model = XGBClassifier(
    enable_categorical=True,
    tree_method="hist"  
)
model.fit(train[X], train['Fertilizer Name Encoded']) 


#Preparing test set
test['Soil Type'] = test['Soil Type'].astype('category')
test['Crop Type'] = test['Crop Type'].astype('category')
X_test = test[X]


# Predict top 3
probs = model.predict_proba(X_test)
top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # top 3 in descending order

# Converting integer labels back to fertilizer names
preds = [' '.join(le.inverse_transform(row)) for row in top_3]

# Creating submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': preds
})
submission.to_csv('submission.csv', index=False)

