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


pip install pandas-summary


#Core Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import random
import warnings
from scipy import stats

#Visualization Libraries 

import matplotlib.pyplot as plt
import seaborn as sns

#machine Learning Libraries 

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder ,FunctionTransformer
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import make_scorer,accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier ,HistGradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,IsolationForest
from sklearn.compose import ColumnTransformer


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df_train


df_train.info()


df_train.isnull().sum()


pip install ydata-profiling


from ydata_profiling import ProfileReport
profile = ProfileReport(df_train)
profile.to_notebook_iframe()


categorical = df_train.select_dtypes(include = ['object']).columns
numerical = df_train.select_dtypes(include = ['float64']).columns



categorical


numerical


df_train = df_train.drop(columns =['id'])





for col in categorical:
    print(df_train[col].value_counts().head(5))
    print(f"Missing value  : {df_train[col].isnull().sum()}")
    print("_"*40)


for col in categorical:
    missing_percent = df_train[col].isnull().mean() * 100
    print(f"{col} → Missing: {missing_percent:.2f}%")



# Approach: Impute with mode (most frequent) for Cetrgorical data
for col in categorical:
    mode_val = df_train[col].mode()[0]
    df_train[col].fillna(mode_val, inplace=True)



df_train[categorical].isnull().sum()



for col in numerical:
    print(df_train[col].value_counts().head(5))
    print(f"Missing value  : {df_train[col].isnull().sum()}")
    print("_"*40)


for col in numerical:
    missing_percent = df_train[col].isnull().mean() * 100
    print(f"{col} → Missing: {missing_percent:.2f}%")



for col in numerical:
    mode_val = df_train[col].mode()[0]
    df_train[col].fillna(mode_val, inplace=True)



for col in numerical:
    missing_percent = df_train[col].isnull().mean() * 100
    print(f"{col} → Missing: {missing_percent:.2f}%")



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ['Stage_fear', 'Personality' ,'Drained_after_socializing']:
    df_train[col] = le.fit_transform(df_train[col])


df_train


print(df_train.skew())


# Fix Skewness


# applying the Box-Cox Transformation for those columns which are Skewness > +1 or < -1 = Highly skewed
# Skewness between ±0.5 = Almost symmetric → can ignore



from scipy.stats import boxcox

cols = ['Stage_fear', 'Time_spent_Alone', 'Drained_after_socializing']

for col in cols:
    df_train[col], _ = boxcox(df_train[col] + 1) 



print(df_train.skew())


corr_matrix = df_train.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()



plt.figure(figsize = (10,6 ))
sns.boxplot(data = df_train)
plt.title("Box plot for outlier Detection")
plt.show()




plt.figure(figsize = (10,6))
plt.hist(df_train ,bins= 10 , edgecolor = 'black')
plt.title("Histogram for outlier Detection")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()




isolation_forest  = IsolationForest(contamination = 0.04 ,random_state =42)
isolation_forest.fit(df_train)


scores = isolation_forest.decision_function(df_train)

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, edgecolor='black')
plt.title("Isolation Forest Decision Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()


# label -1 or 1 . if there will be outlier its 1 otherwise -1
outlier_label =isolation_forest.fit_predict(df_train)


outlier_label


non_outlier = outlier_label!=-1
non_outlier.sum()


have_outlier = outlier_label==-1
have_outlier.sum()



df_train = df_train[non_outlier]


df_train




isolation_forest  = IsolationForest(contamination = 0.04 ,random_state =42)
isolation_forest.fit(df_train)


scores = isolation_forest.decision_function(df_train)

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, edgecolor='black')
plt.title("Isolation Forest Decision Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()


X = df_train.drop(columns=['Personality'])
y = df_train['Personality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


X_train



X_test

