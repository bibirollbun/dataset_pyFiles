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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train.head()


print("Train Shape :", train.shape)
print("Test Shape :",test.shape)


train.columns


print ("\nMissing values :  ", train.isnull().sum().values.sum())


train.info()


((train.isnull().sum())*100)/train.shape[0]


print("Unique Values : \n", train.nunique())


train['Time_spent_Alone'].unique()


for col in train.columns:
    print(f"Unique values in {col} :", train[col].unique())


mean = train['Time_spent_Alone'].mean()
train['Time_spent_Alone'] = train['Time_spent_Alone'].fillna(mean)


numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency']
binary_cols = ['Stage_fear', 'Drained_after_socializing']


test.isnull().sum()


for col in numeric_cols:
    train.fillna({col: train[col].mean()}, inplace=True)
    test.fillna({col: train[col].mean()}, inplace=True)


for col in binary_cols:
    mode = train[col].mode()[0]
    train.fillna({col : mode}, inplace = True)
    test.fillna({col:mode},inplace = True)
    # Convert Yes/No to 1/0
    train[col] = train[col].map({'Yes': 1, 'No': 0})
    test[col] = test[col].map({'Yes': 1, 'No': 0})


train.head(3)


le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])  # Extrovert â†’ 0, Introvert â†’ 1


train.columns


features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency']
X = train[features]
y = train['Personality']


X.head(3)


y.head(3)


X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state = 42
)


model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred


val_accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {val_accuracy:.7f}")


test_preds = model.predict(test[features])
test_preds


test['Personality'] = le.inverse_transform(test_preds)
test['Personality']


sns.countplot(data=train, x=le.inverse_transform(y))
plt.title("Personality Distribution")
plt.show()


sns.boxplot(data=train, x=le.inverse_transform(y), y='Time_spent_Alone')
plt.title("Time Spent Alone by Personality")
plt.show()


sns.kdeplot(data=train, x='Friends_circle_size', hue=le.inverse_transform(y), fill=True)
plt.title("Friends Circle Size Distribution")
plt.show()


train_corr = train.copy()
plt.figure(figsize=(10, 6))
sns.heatmap(train_corr.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()


test_preds = model.predict(test[features])
test['Personality'] = le.inverse_transform(test_preds)  # convert back to string labels


submission = test[['id', 'Personality']]
submission.to_csv("submission3.csv", index=False)
print("submission3.csv ready to upload!")




