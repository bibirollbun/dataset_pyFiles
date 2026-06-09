# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

dtr = train
dts = test




# Step 1: Identifying and filling missing values in train dataset
missing = train.isnull().sum()
missing = missing[missing > 0]  # Get only columns with missing values
print(missing)

for col in missing.index:
    if train[col].dtype == 'object':
        train[col] = train[col].fillna(train[col].mode()[0])  # Fill missing with mode
    else:
        train[col] = train[col].fillna(train[col].median())  # Fill missing with median

# Step 2: Identifying and filling missing values in test dataset
missing_1 = test.isnull().sum()
missing_1 = missing_1[missing_1 > 0]  # Get only columns with missing values
print(missing_1)

for col in missing_1.index:
    if test[col].dtype == 'object':
        test[col] = test[col].fillna(test[col].mode()[0])  # Fill missing with mode
    else:
        test[col] = test[col].fillna(test[col].median())  # Fill missing with median



for col in train.columns:
    if train[col].dtype == 'object':
        encode = LabelEncoder()
        train[col] = encode.fit_transform(train[col])
        if col in test.columns:
            test[col] = encode.transform(test[col])



X = train.drop(['id','Personality'],axis=1)
y = train['Personality']

X_test_final = test.drop("id", axis=1)
# X.head(), y.head()


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


model1 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model1.fit(X_train, y_train)




# y_val_pred = model.predict(X_val)
# print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))



y_valp= model1.predict(X_val)
accuracy = accuracy_score(y_val, y_valp)
print("Validation Accuracy:", accuracy)


model1.fit(X,y)


y_pred_test = model1.predict(X_test_final)
y_pred_labels = encode.inverse_transform(y_pred_test)

print(y_pred_labels)


submission = pd.DataFrame({
    "id":test['id'],
    "Personality": y_pred_labels
})
submission.to_csv("/kaggle/working/submission.csv", index=False)



submission.head()


# import IPython
# from IPython.display import FileLink

# # Save the submission file
# submission.to_csv("submission.csv", index=False)

# # Display clickable download link
# FileLink("submission.csv")



from xgboost import XGBClassifier

model1 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model1.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_valp= model1.predict(X_val)
accuracy = accuracy_score(y_val, y_valp)
print("Validation Accuracy:", accuracy)




import matplotlib.pyplot as plt

feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind='barh')
plt.title("Top Features")
plt.show()


