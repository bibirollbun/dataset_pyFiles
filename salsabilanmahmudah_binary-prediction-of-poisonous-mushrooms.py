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


# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn import metrics 
from sklearn import preprocessing # Import for preprocessing modules
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef


train_data = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
subs_data = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')


train_data.head()


train_data.columns


train_data.info()


train_data_no_id = train_data.drop(columns = ['id'])
test_data_no_id = test_data.drop(columns = ['id'])


train_data_missing = train_data_no_id.isna().mean() * 100
test_data_missing = test_data_no_id.isna().mean() * 100

print("Percentage missing value in Train Data")
print(train_data_missing)

print("\n Percentage missing value in Test Data")
print(test_data_missing)


tr_data_missing = test_data_no_id.isna().sum() 
te_data_missing = test_data_no_id.isna().sum() 

print("Percentage missing value in Train Data")
print(tr_data_missing)

print("\n Percentage missing value in Test Data")
print(te_data_missing)


# Dropping missing value in the dataset that > 10%
index_missing_train_data = train_data_missing[train_data_missing > 10].index
index_missing_test_data = test_data_missing[test_data_missing > 10].index

train_data_used = train_data.drop(columns = index_missing_train_data)
test_data_used = test_data.drop(columns = index_missing_test_data)


train_data_used


test_data_used


def knn_impute(df, n_neighbors=5):   
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_encoded), columns=df_encoded.columns)
    for col in df.select_dtypes(include='object').columns:
        df_imputed[col] = df_imputed[col].round().astype(int).map(
            dict(enumerate(df[col].astype('category').cat.categories)))
    return df_imputed


df_train_imputed = knn_impute(train_data_used, n_neighbors=5)
df_train_imputed


df_test_imputed = knn_impute(test_data_used, n_neighbors=5)
df_test_imputed


cat_cols_train = df_train_imputed.select_dtypes(include=['object']).columns
cat_cols_train = cat_cols_train[cat_cols_train != 'class']

cat_cols_test = df_test_imputed.select_dtypes(include=['object']).columns

ordinal_encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

df_train_imputed[cat_cols_train] = ordinal_encoder.fit_transform(df_train_imputed[cat_cols_train].astype(str))
df_test_imputed[cat_cols_test] = ordinal_encoder.fit_transform(df_test_imputed[cat_cols_test].astype(str))


cat_cols_test


df_train_imputed.head()


df_test_imputed.head()


#Separate data for X_train and y_train
X = df_train_imputed.drop(['class'], axis=1)
y = df_train_imputed['class']


# 70% training dataset and 30% test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# creating a RF classifier
clf_rf = RandomForestClassifier(n_estimators = 100)  
clf_rf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred_rf = clf_rf.predict(X_test)
 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred_rf))


# # import support vector classifier 
# # "Support Vector Classifier"
# from sklearn.svm import SVC 
# clf = SVC(kernel='linear') 
# clf.fit(X_train, y_train)

# # performing predictions on the test dataset
# y_pred_svm = clf.predict(X_test)


# data = list(zip(X_train, y_train))
# knn = KNeighborsClassifier(n_neighbors=1)

# knn.fit(data, classes)

# # performing predictions on the test dataset
# y_pred_knn = knn.predict(X_test)


print('MCC for every algorithm')
print('MCC for Random Forest :', matthews_corrcoef(y_test, y_pred_rf))
# print('MCC for Support Vector Machine' + matthews_corrcoef(y_test, y_pred_svm))
# print('MCC for K-Nearest Neighbor' + matthews_corrcoef(y_test, y_pred_knn))


pred_test_data_rf = clf_rf.predict(df_test_imputed)


submission = test_data[['id']]
submission['Random Forest'] = pred_test_data_rf


submission

