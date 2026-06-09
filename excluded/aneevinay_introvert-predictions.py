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
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


df.shape


df.head()


df.describe()


df.info()


df.isnull().sum()


df.fillna({
    'Time_spent_Alone': 0.0,
    'Stage_fear': 'No',
    'Social_event_attendance': 0.0,
    'Going_outside': 0.0,
    'Drained_after_socializing': 'No',
    'Friends_circle_size': 0.0,
    'Post_frequency': 0.0
}, inplace=True)


df['Personality'].value_counts()


plt.figure(figsize=(12, 6))
sns.countplot(x='Personality', data=df)
plt.title('Personality Class Distribution')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.show()
plt.savefig('Personality class distribution.png')


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(10, 6))

for col in num_cols:
    sns.kdeplot(df[col], label=col, fill=True, alpha=0.3)

plt.title("Distribution of Numerical Features")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Distribution of numerical features.png')


group_means = df.groupby('Personality')[num_cols].mean()
group_means


group_means.plot(kind='bar', figsize=(10,6))
plt.title('Group Differences in Personality Traits')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.legend(title='Features', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
plt.savefig('Group_differences_in_personality_traits.png')


corr = df[num_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()


columns = ['Stage_fear', 'Drained_after_socializing'] 

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  
encoded_data = ohe.fit_transform(df[columns])

encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(columns), index=df.index)

df = pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)



X=df.drop(['Personality','id'],axis=1)
y=df['Personality']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'GaussianNB': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)               
    y_pred = model.predict(X_test)            
    accuracy = accuracy_score(y_test, y_pred) 
    print(f'{name}: Accuracy = {accuracy:.4f}')


model=GaussianNB()
model.fit(X_train,y_train)


test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


test_id = test['id']


test.fillna({
    'Time_spent_Alone': 0.0,
    'Stage_fear': 'No',
    'Social_event_attendance': 0.0,
    'Going_outside': 0.0,
    'Drained_after_socializing': 'No',
    'Friends_circle_size': 0.0,
    'Post_frequency': 0.0
}, inplace=True)


columns = ['Stage_fear', 'Drained_after_socializing']


encoded = ohe.transform(test[columns])

encoded_test = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(columns), index=test.index)

test = pd.concat([test.drop(columns, axis=1), encoded_test], axis=1)



y_preds = model.predict(test.drop('id', axis=1)) 

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': y_preds
})

submission.to_csv('submission.csv', index=False)




