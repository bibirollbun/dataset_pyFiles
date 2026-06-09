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
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


train=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.head()


test.head()


print("train shape:", train.shape)
print("test shape:", test.shape)


train.isnull().sum()


train.duplicated().sum()


train.info()


train.describe()


sns.countplot(x='diagnosed_diabetes',
              data=train)
plt.show()


grid = sns.FacetGrid(train,
                    col='diagnosed_diabetes',
                    row='smoking_status', 
                    aspect=1.6)

grid.map(plt.hist,
        'gender',
        alpha=.5,
        bins=20)

grid.add_legend()


grid = sns.FacetGrid(train,
                     col='diagnosed_diabetes',
                     row='systolic_bp',
                     aspect=1.6)

grid.map(plt.hist,
         'gender',
         alpha=.5,
         bins=20)

grid.add_legend()


sns.histplot(data=train,
             x='age',
             hue='diagnosed_diabetes',
             bins=30,
             kde=True)

plt.show()


le=LabelEncoder()
for col in train.select_dtypes(include=['object']).columns:
    train[col]=le.fit_transform(train[col])
    test[col]=le.transform(test[col])


X= train[['family_history_diabetes',
        'physical_activity_minutes_per_week',
        'age',
        'triglycerides',
        'bmi',
        'ldl_cholesterol',
        'cardiovascular_history',
        'diet_score',
        'hdl_cholesterol',
        'heart_rate']]

y= train['diagnosed_diabetes']


train_X, val_X, train_y, val_y = train_test_split(X,
                                                  y,
                                                  test_size=0.2,
                                                  random_state=42)


gnb=GaussianNB()

gnb.fit(train_X,
        train_y)

val_pred = gnb.predict(val_X)

print("GaussianNB Accuracy:",accuracy_score(val_y, val_pred))


rf=RandomForestClassifier(n_estimators=100,
                           random_state=42)

rf.fit(train_X,
       train_y)

val_pred = rf.predict(val_X)
print("Random Forest Accuracy:", accuracy_score(val_y, val_pred))


xgb=XGBClassifier(n_estimators=500,
                  random_state=42)

xgb.fit(train_X,
        train_y)

val_pred = xgb.predict(val_X)

print(f"Tuned XGB Accuracy: {accuracy_score(val_y, val_pred):.2f}")

print("Confusion Matrix:\n",
       confusion_matrix(val_y, val_pred))


feat_importances=pd.Series(xgb.feature_importances_,
                            index=train_X.columns)

feat_importances.plot(kind='barh')
plt.show()


# xgb=XGBClassifier()
# param_grid={
#     'n_estimators':[100,200,300,500],
#     'max_depth':[3,5,7,10],
#     'learning_rate':[0.01,0.1,0.2,0.3],
#     'subsample':[0.6,0.8,1.0],
#     'colsample_bytree':[0.6,0.8,1.0]
# }
# grid_search=GridSearchCV(estimator=xgb, 
#                          param_grid=param_grid, 
#                          cv=3, 
#                          n_jobs=-2, 
#                          verbose=2)

# grid_search.fit(train_X,
#                 train_y)

# best_xgb=grid_search.best_estimator_
# val_pred = best_xgb.predict(val_X)
# print("Tuned XGB Accuracy:", accuracy_score(val_y, val_pred))


# Ensure test data uses the same features as training
result = xgb.predict(test[X.columns])
sample_submission = pd.DataFrame({
    "id": test.id,
    "submission": result
})
sample_submission.head()


print("Saving submission file...\n")
sample_submission.to_csv('submission.csv', index=False) 
print("✅ submission.csv created successfully!")

