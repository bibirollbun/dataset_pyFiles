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


df_train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


df_train.info()


df_test.info()


df_train.nunique()


df_test.nunique()


df_train.isna().sum()


df_test.isna().sum()


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_train_num = df_train.select_dtypes(include = numerics)
df_train_num.columns


df_test_num = df_test.select_dtypes(include = numerics)
df_test_num.columns


# df_train_num.isna().sum()


df_train_cate = df_train.drop(df_train_num.columns,axis = 1)
df_train_cate.columns


df_test_cate = df_test.drop(df_test_num.columns,axis = 1)
df_test_cate.columns


# df_train_cate.isna().sum()


# df_train_num.isna().sum()


# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt
# df = sns.load_dataset('iris')

# sns.boxplot( y=df_train["Premium Amount"], data = df_train_num )
plt.figure(figsize = (14, 10), dpi = 72)
plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.5, hspace = 0.3)

plt.subplot(331)
sns.boxplot(y = df_train["Age"])
plt.subplot(332)
sns.boxplot(y = df_train["Annual Income"])
plt.subplot(333)
sns.boxplot(y = df_train["Number of Dependents"])
plt.subplot(334)
sns.boxplot(y = df_train["Health Score"])
plt.subplot(335)
sns.boxplot(y = df_train["Previous Claims"])
plt.subplot(336)
sns.boxplot(y = df_train["Vehicle Age"])
plt.subplot(337)
sns.boxplot(y = df_train["Credit Score"])
plt.subplot(338)
sns.boxplot(y = df_train["Insurance Duration"])
plt.subplot(339)
sns.boxplot(y = df_train["Premium Amount"])


from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
def fillna(data,col):
    df_train_num1 = pd.DataFrame(data[col])
    impute = IterativeImputer(estimator = RandomForestRegressor(), random_state = 0)
    impute.fit(df_train_num1)
    # display(pd.DataFrame(impute.transform(df_train_num1)))
    data[col] = pd.DataFrame(impute.transform(df_train_num1))
    return data[col]


for i in df_train_num:
    if i == 'id':
        continue
    df_train[i] = fillna(df_train_num,i)


df_train_num


df_train


for i in df_test_num:
    if i == 'id':
        continue
    df_test[i] = fillna(df_test_num,i)


df_test


print(df_test['Customer Feedback'].mode())
print(df_test['Marital Status'].mode())
print(df_test['Occupation'].mode())


df_test['Customer Feedback'] = df_test['Customer Feedback'].fillna('Average')
df_test['Marital Status'] = df_test['Marital Status'].fillna('Single')
df_test['Occupation'] = df_test['Occupation'].fillna('Employed')


print(df_train['Customer Feedback'].mode())
print(df_train['Marital Status'].mode())
print(df_train['Occupation'].mode())


df_train['Customer Feedback'] = df_train['Customer Feedback'].fillna('Average')
df_train['Marital Status'] = df_train['Marital Status'].fillna('Single')
df_train['Occupation'] = df_train['Occupation'].fillna('Employed')


df_train.isna().sum()


df_test.isna().sum()


df_test_cate.isna().sum()


df_train_num = df_train.select_dtypes(include = numerics)
# df_train_num.columns
df_test_num = df_test.select_dtypes(include = numerics)
# df_test_num.columns
df_train_cate = df_train.drop(df_train_num.columns,axis = 1)
# df_train_cate.columns
df_test_cate = df_test.drop(df_test_num.columns,axis = 1)
# df_test_cate.columns


for i in df_train_cate:
    enc_nom_1 = (df_train_cate.groupby(i).size()) / len(df_train_cate)
    # print(enc_nom_1)
    df_train[i] = df_train_cate[i].apply(lambda x : enc_nom_1[x])
df_train


for i in df_test_cate:
    enc_nom_1 = (df_test_cate.groupby(i).size()) / len(df_test_cate)
    # print(enc_nom_1)
    df_test[i] = df_test_cate[i].apply(lambda x : enc_nom_1[x])
df_test


trainX = df_train.drop('Premium Amount',axis = 1)
trainY = df_train['Premium Amount']
testX = df_test


from sklearn import svm, ensemble, metrics
forest = ensemble.RandomForestRegressor(n_estimators = 20) 
# max_features 特徵數、max_depth 樹的深度
forest.fit(trainX, trainY)
predictions = forest.predict(testX)
#metrics.accuracy_score(testY, predictions)
predictions = pd.DataFrame(predictions)
predictions


ans = pd.DataFrame(testX['id'])
ans['Premium Amount'] = predictions
ans.to_csv("submission_forest_fillnadifferent.csv",index = False)

