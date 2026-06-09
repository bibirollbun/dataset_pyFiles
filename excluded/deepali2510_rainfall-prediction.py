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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression



train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')



print(train.info())
print(train.describe())
print(train.isnull().sum())



train.hist(figsize=(12,8),bins=30)
plt.show()



sns.countplot(x=train['rainfall'])
plt.title("Rainfall Distribution")
plt.show()



corr_matrix = train.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()



train['temp_range'] = train['maxtemp'] - train['mintemp']
test['temp_range'] = test['maxtemp'] - test['mintemp']
train['dewpoint_spread'] = train['temparature'] - train['dewpoint']
test['dewpoint_spread'] = test['temparature'] - test['dewpoint']
train['relative_humidity'] = train['humidity']*(train['temparature']/100)
test['relative_humidity'] = test['humidity']*(test['temparature']/100)
train.fillna(train.median(),inplace=True)
test.fillna(test.median(),inplace=True)



x = train.drop(columns=['id','rainfall'])
y = train['rainfall']
x_train, x_valid, y_train, y_valid = train_test_split(x,y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_valid)
print("Logistics Regression Accuracy:", accuracy_score(y_valid, y_pred))



rf_model = RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_valid)
print("Random Forest Accuracy:", accuracy_score(y_valid, y_pred))



importances = rf_model.feature_importances_
feature = x.columns
plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=feature)
plt.title("Feature Importance")
plt.show()



test_features = test.drop(columns=['id'])


xgb_model = XGBClassifier(n_estimators=200,learning_rate=0.05, max_depth=5,random_state=42)
xgb_model.fit(x_train,y_train)
y_pred = xgb_model.predict(x_valid)
print("XGBoost Accuracy:", accuracy_score(y_valid, y_pred))
print("Train columns:", train.columns)
print("Test columns:", test.columns)



test_probs = xgb_model.predict_proba(test.drop(columns=['id'],errors='ignore'))[:,1]
submission = pd.DataFrame({'id':test['id'],'rainfall': test_probs})
submission.to_csv('submission.csv',index=False)
print("Submission file saved")


