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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score



train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=0)
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col=0)


train.head()


test.head()


sns.pairplot(train)


# missing value
train.isnull().sum()


train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].mean(), inplace=True) # Replace NaN in 'col1' with 0
test['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].mean(), inplace=True) # Replace NaN in 'col1' with 0


train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean(), inplace=True) # Replace NaN in 'col1' with 0
test['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean(), inplace=True) # Replace NaN in 'col1' with 0


#replace missing ad number with zero
train['Number_of_Ads'].fillna(0, inplace=True) # Replace NaN in 'col1' with 0
test['Number_of_Ads'].fillna(0, inplace=True) # Replace NaN in 'col1' with 0


test.isnull().sum()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Genre_code'] = le.fit_transform(train['Genre'])
test['Genre_code'] = le.transform(test['Genre'])


train['Episode_Sentiment_code'] = le.fit_transform(train['Episode_Sentiment'])
test['Episode_Sentiment_code'] = le.transform(test['Episode_Sentiment'])


train['Episode_Num'] = train['Episode_Title'].str[8:].astype(int)
test['Episode_Num'] = test['Episode_Title'].str[8:].astype(int)
test.head()


train['Is_Weekend'] = 0
train.loc[train['Publication_Day'].isin(['Saturday', 'Sunday']),'Is_weekend'] = 1
test['Is_Weekend'] = 0
test.loc[test['Publication_Day'].isin(['Saturday', 'Sunday']),'Is_weekend'] = 1


#col_keep = ['Genre_code','Episode_Sentiment_code','Number_of_Ads','Is_Weekend','Episode_Num',
 #          'Guest_Popularity_percentage','Host_Popularity_percentage','Episode_Length_minutes']
col_keep = ['Episode_Num',
           'Guest_Popularity_percentage','Host_Popularity_percentage','Episode_Length_minutes']


sample_train = train.sample(60000)
X = sample_train.loc[:,sample_train.columns.isin(col_keep)]
y = sample_train.Listening_Time_minutes


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Apply random forest
rf = RandomForestRegressor(n_estimators=500, random_state=42)
#rf = RandomForestRegressor(n_estimators=300, random_state=42)

#lgr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_valid)


mean_squared_error(y_valid,y_pred), r2_score(y_valid,y_pred)


# Feature Importance analysis
feature_importance = rf.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X_train.columns,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("RF Feature Importance")
plt.gca().invert_yaxis()  
plt.show()




X_test=test.loc[:,test.columns.isin(col_keep)]



X_test_scaled = scaler.transform(X_test)
y_test_pred = rf.predict(X_test_scaled)
#y_test_pred = lr_reg.predict(X_test_scaled)


test['Listening_Time_minutes'] = y_test_pred
#test.loc[test.Listening_Time_minutes<0,'Listening_Time_minutes'] = 0
test.describe()


test[['Listening_Time_minutes']].to_csv('submission.csv')




