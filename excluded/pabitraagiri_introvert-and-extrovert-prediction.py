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
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original_data= pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")


train


def inspect_features(df, name):
    print(f"ğŸ”� {name} Set Feature Overview:")
    df_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        "Duplicate Values":df.duplicated().sum()
    })
    display(df_info.sort_values(by='Missing Values', ascending=False))


inspect_features(train, 'Train')

inspect_features(test_data, 'Test')

inspect_features(original_data, 'Original')


train


def fill_value(df,name):
    df.fillna({
        "Stage_fear":df['Stage_fear'].mode()[0],
        "Time_spent_Alone":df["Time_spent_Alone"].mean(),
        "Social_event_attendance": df['Social_event_attendance'].mean(),
        "Going_outside" : df["Going_outside"].mean(),
        "Drained_after_socializing":df["Drained_after_socializing"].mode()[0],
        "Friends_circle_size":df["Friends_circle_size"].mean(),
        "Post_frequency":df['Post_frequency'].mean()
        },inplace=True)

fill_value(train,'train')
fill_value(original_data,'Original')
fill_value(test_data,'Test')


fig, axes = plt.subplots(2, 2, figsize=(10, 20))

plots_info = [
    ("Social_event_attendance", axes[0][0]),
    ("Going_outside", axes[0][1]),
    ("Friends_circle_size",axes[1][0]),
    ("Post_frequency",axes[1][1])
]

for col,ax in plots_info:
    col_info = train[col].value_counts()
    ax.hist(train[col])
    ax.set_xlabel(col)
    ax.set_ylabel("count")


plt.show()


fig, axes = plt.subplots(1,2,figsize=(10,10))
counts = train['Stage_fear'].value_counts()
axes[0].bar(counts.index,counts.values)
counts = train['Drained_after_socializing'].value_counts()
axes[1].bar(counts.index,counts.values)
	


fig, axes = plt.subplots(2,3,figsize=(20,15))

plots_info = [
    ("Social_event_attendance", axes[0][0]),
    ("Stage_fear", axes[0][1]),
    ("Going_outside", axes[0][2]),
    ("Drained_after_socializing",axes[1][0]),
    ("Friends_circle_size",axes[1][1]),
    ("Post_frequency",axes[1][2])
]
for col,ax in plots_info:
    ctab = pd.crosstab(train["Personality"], train[col])
    ctab.plot(kind="bar", ax=ax),
    ax.set_title(f"{col} by Personality")
    ax.set_xlabel("Personality")
    ax.set_ylabel("Count")
    ax.legend(title=col)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


ohe = OneHotEncoder()
X = train.iloc[:, :-1]   
y = train.iloc[:, -1] 
X_encoder= ohe.fit_transform(X)



X_train,X_test,y_train,y_test = train_test_split(X_encoder,y,test_size=0.2,random_state=42)


rf= RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)


y_pred = rf.predict(X_test)


print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nclassification report:\n",classification_report(y_test,y_pred))
print("\n confusion matrix:\n",confusion_matrix(y_test,y_pred))


best_threshold = 0.5 
probs = rf.predict_proba(X_test)[:,1]
predictions = (probs>= best_threshold).astype(int)


submission = pd.DataFrame({'id': id, "Personality": predictions})

# Converting 1s back to Extrovert and 0s back to Introvert
submission['Personality'].replace({1: 'Extrovert', 0: 'Introvert'}, inplace=True)
submission.to_csv('submission.csv', index=False)










































