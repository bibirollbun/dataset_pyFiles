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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from xgboost import XGBClassifier


train_df = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/test.csv')
sample_submission = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/sampleSubmission.csv')


train_df.head()


y = train_df["target"]
X = train_df.drop(columns=["target", "id"])  
X_test = test_df.drop(['id'], axis=1)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded)


model = XGBClassifier(
    learning_rate=0.06426980635477918,
    max_depth=7,
    n_estimators=379,
    subsample= 0.8070259980080767,
    colsample_bytree=0.8166031869068445,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

y_val_probs = model.predict_proba(X_val)  

loss = log_loss(y_val, y_val_probs, labels=list(range(9)))  
print(f"Log Loss: {loss:.4f}")

\
test_probs = model.predict_proba(X_test_scaled)  


submission = pd.DataFrame(test_probs, columns=sample_submission.columns[1:])  



submission.insert(0, 'id', test_df['id'])


submission.to_csv('submission.csv', index=False)
print("✅ Submission file created: submission.csv")

