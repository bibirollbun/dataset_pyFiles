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


df=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


df.head()


# pip install xgboost



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier






y = df['diagnosed_diabetes']
X = df.drop(columns=['diagnosed_diabetes', 'id','alcohol_consumption_per_week','education_level','income_level','ethnicity'])


cat_cols = X.select_dtypes(include=['object']).columns

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)



preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))



import matplotlib.pyplot as plt
from xgboost import plot_importance

plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=15)
plt.show()



#load the testing data 
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
df_test = df_test[X_train.columns]

cat_cols = df_test.select_dtypes(include=['object']).columns

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_test[col] = le.fit_transform(df_test[col])
    encoders[col] = le

probs = model.predict_proba(df_test)[:,1]


pd.DataFrame({
    "id":range(700000 , 700000+ df_test.shape[0]),
    "diagnosed_diabetes":probs
}).to_csv("submission.csv",index=False)


probs


model.predict_proba(df_test)


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
# sample_submission.diagnosed_diabetes = probs


sample_submission.to_csv("submission.csv")







