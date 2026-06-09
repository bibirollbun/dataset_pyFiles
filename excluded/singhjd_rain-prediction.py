# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


print("Train df shape:", train_df.shape)
print("Test df shape:", test_df.shape)


train_df.head()


train_df.isna().sum()


test_df.head()


test_df.isna().sum()


test_df['winddirection']=test_df['winddirection'].fillna(test_df['winddirection'].mean())


test_df.isna().sum()


test_ids=test_df['id']


train_df.info()


sns.heatmap(train_df.corr(), cmap="Blues")
plt.show()


features=[]
for i in train_df.columns:
    if(i=="id" or i=="rainfall" or i=="day"):
        continue
    else:
        features.append(i)

print(features)
print(len(features))


train_df.drop(columns=['id', 'day'], inplace=True)
test_df.drop(columns=['id', 'day'], inplace=True)


X=train_df.drop(columns=['rainfall'])
y=train_df['rainfall']


scaler=MinMaxScaler()

X_scaled=scaler.fit_transform(X)


test_df_scaled=scaler.transform(test_df)


X_train, X_test, y_train, y_test=train_test_split(X_scaled, y, test_size=0.2)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


log_model=LogisticRegression()


log_model.fit(X_train, y_train)


log_model_pred=log_model.predict(X_test)


print(classification_report(y_test, log_model_pred))


prediction1=log_model.predict(test_df)


submission1=pd.DataFrame({"id": test_ids, "rainfall": prediction1})
submission1.to_csv("submission1", index=False)
print("submission1 file created")


""

