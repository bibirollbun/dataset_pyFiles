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

train_df = pd.read_csv('/kaggle/input/thapar-kaggle-hack-v02/train.csv')
test_df = pd.read_csv('/kaggle/input/thapar-kaggle-hack-v02/test.csv')
sample_submission = pd.read_csv('/kaggle/input/thapar-kaggle-hack-v02/sample_submission.csv')

print(train_df.head())

print(test_df.head())

print(sample_submission.head())



X_train = train_df.drop(columns=['id', 'target'])
y_train = train_df['target']  
X_test = test_df.drop(columns=['id'])

print(X_train.shape, y_train.shape, X_test.shape)  



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.countplot(x=train_df['target'], palette='viridis')
plt.title("Class Distribution of Target")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()



# Summary: 
print(train_df.describe())  



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
sns.heatmap(train_df.iloc[:, 1:].corr(), cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df.drop(columns=['id', 'target']))
X_test_scaled = scaler.fit_transform(test_df.drop(columns=['id']))


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, train_df['target'], test_size=0.2, random_state=42, stratify=train_df['target']
)



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=len(train_df['target'].unique()),
    eval_metric="mlogloss",
    use_label_encoder=False,
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.5
)
xgb.fit(X_train_scaled, train_df['target'])


y_val_pred = xgb.predict(X_val)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")



xgb.fit(X_train_scaled, train_df['target'])



y_test_pred = xgb.predict(X_test_scaled)



submission = pd.DataFrame({
    'id': test_df['id'],  
    'target': y_test_pred  
})

submission.to_csv("submission.csv", index=False)





