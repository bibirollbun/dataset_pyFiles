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
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.tree import plot_tree,DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer,OneHotEncoder,OrdinalEncoder,FunctionTransformer,PowerTransformer


df_train=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_submission=pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


df_train.head(3)


df_train.drop(columns=['id'],inplace=True) # droping the id column because it's not that useful
df_train.head(1)


# checking the number of null, duplicated and unique values present in our dataset
def preprocessing(dataframe):
    print(dataframe.isnull().sum())
    print('Total no of Duplicate Values: ',dataframe.duplicated().sum())
    print('Unique Values: ',dataframe.nunique())


preprocessing(df_train)


# Separating the numerical columns present in our dataset & storing it in "num_col" variable as pandas dataframe
num_col=pd.DataFrame()
for i in df_train.columns:
    if df_train[i].dtype!=object:
        num_col[i]=df_train[i]

num_col.head(2)


# Separating the categorical columns present in our dataset & storing it in "cat_col" variable as pandas dataframe
cat_col=pd.DataFrame()
for i in df_train.columns:
    if df_train[i].dtype==object:
        cat_col[i]=df_train[i]

cat_col.head(2)


# Plotting "Distplot" to see if the distribution of a column normal or skewed
plt.figure(figsize=(20,12))
plt.subplot(2,4,1)
sns.distplot(num_col['age'])

plt.subplot(2,4,2)
sns.distplot(num_col['balance'])

plt.subplot(2,4,3)
sns.distplot(num_col['day'])

plt.subplot(2,4,4)
sns.distplot(num_col['duration'])

plt.subplot(2,4,5)
sns.distplot(num_col['campaign'])

plt.subplot(2,4,6)
sns.distplot(num_col['pdays'])

plt.subplot(2,4,7)
sns.distplot(num_col['previous'])


num_col.skew() # Skewness of numerical columns quantity wise


# Plotting "Boxplot" to see if there are outliers in our numerical columns or not
plt.figure(figsize=(20,12))
plt.subplot(2,4,1)
sns.boxplot(num_col['age'])

plt.subplot(2,4,2)
sns.boxplot(num_col['balance'])

plt.subplot(2,4,3)
sns.boxplot(num_col['day'])

plt.subplot(2,4,4)
sns.boxplot(num_col['duration'])

plt.subplot(2,4,5)
sns.boxplot(num_col['campaign'])

plt.subplot(2,4,6)
sns.boxplot(num_col['pdays'])

plt.subplot(2,4,7)
sns.boxplot(num_col['previous'])


# Plotting 'Histplot' to see the distribution of numbers/values among classes
plt.figure(figsize=(20,12))
plt.subplot(3,3,1)
sns.histplot(cat_col['job'])

plt.subplot(3,3,2)
sns.histplot(cat_col['marital'])

plt.subplot(3,3,3)
sns.histplot(cat_col['education'])

plt.subplot(3,3,4)
sns.histplot(cat_col['default'])

plt.subplot(3,3,5)
sns.histplot(cat_col['housing'])

plt.subplot(3,3,6)
sns.histplot(cat_col['loan'])

plt.subplot(3,3,7)
sns.histplot(cat_col['contact'])

plt.subplot(3,3,8)
sns.histplot(cat_col['month'])

plt.subplot(3,3,9)
sns.histplot(cat_col['poutcome'])


cat_col['default'].value_counts() # To show you how big the gap is among the default classes


# Droping the default column from both the df_train dataset as well as cat_col dataset
df_train.drop(columns=['default'],inplace=True)
cat_col.drop(columns=['default'],inplace=True)


#sns.pairplot(num_col)


df_train.head(3)


# To confirm if the num_col contains the negatice values
for col in num_col.columns:
    if (num_col[col] <  0).any():
        print(f"Column '{col}' has negative values")
    else:
        print(f"Column '{col}' has no negative values")


trf = ColumnTransformer([
    ("KBins", KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile'), [0,4,8,10,11,12,13]), # Transforming continuous values of a column to discrete values(also reduce outleirs affect to some extent)
    ("OneHot", OneHotEncoder(drop='first', sparse_output=False, dtype='int64'), [1,5,6,7,14]), # Transforming the categorical values(having no order b/w them) to numeric values
    ("Ordina", OrdinalEncoder(), [2,3,9]), # Transforming the categorical values(having some sort of order b/w them) to numeric values
("FunctionTransformer", FunctionTransformer(func=np.log1p), [0,8,10,11,13]), # For right skewed numerical columns
("PowerTransformer", PowerTransformer(method='yeo-johnson',standardize=True ), [4,12]) # For  the columns having negative values in them

])



X1=df_train.drop(columns=['y'])
y1=df_train['y']
X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.2,random_state=43)
print('Training Data Shape: ',X_train.shape)
print('Testning Data Shape: ',X_test.shape)



lr=LogisticRegression(penalty='l2',max_iter=150,solver='saga',random_state=43,n_jobs=-1,l1_ratio=0.75)
dt=DecisionTreeClassifier(splitter='best',max_depth=45)
rf=RandomForestClassifier(n_estimators=250, max_depth=45,n_jobs=-1,max_features=0.25,bootstrap=True)


pipe=Pipeline([
               ('trf',trf),
               ('lr',lr)])
pipe.fit(X_train,y_train)


roc_auc_score(y_test,pipe.predict(X_test))


pipe1=Pipeline([
               ('trf',trf),
               ('dt',dt)])
pipe1.fit(X_train,y_train)


roc_auc_score(y_test,pipe1.predict(X_test))


pipe2=Pipeline([
               ('trf',trf),
               ('rf',rf)])
pipe2.fit(X_train,y_train)


roc_auc_score(y_test,pipe2.predict(X_test))



submission=pd.DataFrame({
    'id':df_test['id'],
    'y':pipe2.predict(df_test)
})

submission.to_csv('submission.csv',index=False)


df=pd.read_csv('submission.csv')
df




