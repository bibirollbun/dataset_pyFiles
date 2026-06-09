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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer


df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


df.shape


df.columns


df.isnull().sum()


cat_cols = df.select_dtypes(include=['object']).drop(columns=['diagnosed_diabetes','id'], errors='ignore').columns.tolist()

num_cols = df.select_dtypes(exclude=['object']).columns.tolist()


df['diagnosed_diabetes'].value_counts()


df[df.diagnosed_diabetes==1].describe(include=['object'])


df[df.diagnosed_diabetes==0].describe(include=['object'])


# plt.figure(figsize=(20,25))

# for i,col in enumerate(cat_cols):
#     plt.subplot(8,2,i+1)
#     sns.countplot(data=df,x=col,palette='Set2',hue=col)
#     plt.title(f"Distribution of {col}")
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(20,25))

# for i,col in enumerate(num_cols):
#     plt.subplot(16,2,i+1)
#     sns.histplot(data=df,x=col,kde=True)
#     plt.title(f"Distribution of {col}")
# plt.tight_layout()
# plt.show()


y = df['diagnosed_diabetes']
X = df.drop(columns=['diagnosed_diabetes','id'])

X_train,X_valid ,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=42)


encoder = OneHotEncoder()
preprocessor = ColumnTransformer(
    transformers = [
        ('cat',OneHotEncoder(handle_unknown='ignore'),
        cat_cols)
    ],
    remainder = 'passthrough'
    
)

X_train_transform = preprocessor.fit_transform(X_train)
X_valid_transform = preprocessor.transform(X_valid)



rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(random_state=42)
cat_model= CatBoostClassifier(random_state= 42,verbose=False)
lgb_model= LGBMClassifier(random_state=42,verbose=0)


rf_model.fit(X_train_transform,y_train)
xgb_model.fit(X_train_transform,y_train)
cat_model.fit(X_train_transform,y_train)
lgb_model.fit(X_train_transform,y_train)







rf_p = rf_model.predict(X_valid_transform)
xgb_p = xgb_model.predict(X_valid_transform)
cat_p = cat_model.predict(X_valid_transform)
lgb_p = lgb_model.predict(X_valid_transform)


roc_rf = roc_auc_score(y_valid,rf_p)
roc_xgb = roc_auc_score(y_valid,xgb_p)
roc_cat = roc_auc_score(y_valid,cat_p)
roc_lgb = roc_auc_score(y_valid,lgb_p)

print(roc_rf)
print(roc_xgb)
print(roc_cat)
print(roc_lgb)


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


test_new =test.drop(columns=['id'])
test_transform = preprocessor.transform(test_new)  


test_proba = cat_model.predict_proba(test_transform)[:, 1]



submission =pd.DataFrame({
    'id' : test['id'],
'diagnosed_diabetes' : test_proba
})
submission.to_csv('submission.csv',index=False)




