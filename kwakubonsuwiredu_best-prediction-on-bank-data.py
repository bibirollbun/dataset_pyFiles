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


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train_o = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep = ';')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


features = list(train.drop(['id','y'], axis =1).columns)
target = 'y'


from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

categorical = [col for col in features if train[col].dtype == 'object']
categorical


label_enc = LabelEncoder()
scaler = StandardScaler()
cat_enc = OrdinalEncoder()

train[categorical] = cat_enc.fit_transform(train[categorical])
train_o[categorical] = cat_enc.transform(train_o[categorical])

train_o[target] = label_enc.fit_transform(train_o[target])

## Scaled Data
train_norm = pd.DataFrame()
train_o_norm = pd.DataFrame()

train_norm[features] = scaler.fit_transform(train[features])
train_o_norm[features] = scaler.transform(train_o[features])


from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors = 5)


knn.fit(train_norm[features], train[target])


from sklearn.metrics import accuracy_score as ac



print("KNN score:", ac(train_o[target], knn.predict(train_o_norm[features])))



train_full = pd.concat([train,train_o], ignore_index = True)
train_full


train_full.drop('id', axis=1, inplace = True)
train_full


knn = KNeighborsClassifier(n_neighbors=5)

scaler = StandardScaler()
train_full[features] = scaler.fit_transform(train_full[features])

knn.fit(train_full[features], train_full[target])


test.head()


test[categorical] = cat_enc.transform(test[categorical])
test[features] = scaler.transform(test[features])


predictions = knn.predict_proba(test[features])


final_predictions = predictions[:,1]


submission = pd.DataFrame({
    'id': test['id'],
    'y': final_predictions
})

submission.to_csv('submission.csv', index = False)

