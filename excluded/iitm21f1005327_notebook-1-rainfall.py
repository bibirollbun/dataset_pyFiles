import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


test.isnull().sum()


test.fillna(0,inplace=True)


X = train.iloc[:,:-1]
y = train.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
scaled_test = ss.transform(test)


model = AdaBoostClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


score = roc_auc_score(y_pred,y_test)
print(score)


y_pred_prob = model.predict_proba(scaled_test)[:,1]


submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_pred_prob
})

submission.to_csv("submission.csv", index=False)

