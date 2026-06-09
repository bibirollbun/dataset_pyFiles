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

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

from warnings import filterwarnings
filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


print("Train shape:", train.shape)


train.head()


train.tail()


print("="*30)
print("| 'Missing values in dataset':|")
print("="*30)
print(train.isnull().sum())


train.duplicated().sum().sum()


train.info()


train.describe().T


sns.countplot(x="loan_paid_back", data=train)
plt.title("Loan Payment Status")
plt.xlabel("Loan Paid Back (1=Yes, 0=No)")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
counts = train["marital_status"].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title("marital_status Distribution")

plt.subplot(1,2,2)
sns.countplot(x=train["marital_status"])
plt.title("marital_status Distribution")
plt.show()


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
counts = train["gender"].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title("gender Distribution")

plt.subplot(1,2,2)
sns.countplot(x=train["gender"],hue=train["loan_paid_back"])
plt.title("gender wise loan_paid back")
plt.show()


counts = train["education_level"].value_counts()

sns.barplot(x='education_level', y='loan_paid_back', data=train, order=counts.index, palette='Set2')
plt.title("Loan Payment Status by education_level")
plt.xlabel("education_level")
plt.ylabel("Loan Paid Back (mean)")
plt.ylim(0, 1)
plt.show()

print("="*50)
print("\t\t**Insight**: \n  Bachelor students payback less than others")
print("="*50)


counts = train["employment_status"].value_counts()

sns.barplot(x='employment_status', y='loan_paid_back', data=train, order=counts.index, palette='Set2')
plt.title("Loan Payment Status by Employment Status")
plt.xlabel("Employment Status")
plt.ylabel("Loan Paid Back (mean)")
plt.ylim(0, 1)
plt.show()

print("="*50)
print("\t\t**Insight**: \n\tEmployed and Retired payback_loan \n\tBecause they have a source of income")
print("="*50)


plt.figure(figsize=(10, 6))

sns.histplot(data=train, x='loan_amount', hue='loan_paid_back', kde=True, bins=30)
plt.title("Distribution of Loan Amount by Loan Payment Status")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()


# annual_income distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=train, x='annual_income', hue='loan_paid_back', kde=True, bins=30)
plt.title("Distribution of Annual Income by Loan Payment Status")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(17, 6))

plt.subplot(1,2,1)
sns.histplot(data=train, x='annual_income', hue='loan_paid_back', kde=True, bins=30)
plt.title("Distribution of Annual Income by Loan Payment Status")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
sns.histplot(data=train, x='annual_income', hue='loan_paid_back', kde=True, bins=30,log_scale=True)
plt.title("Distribution of Annual Income by Loan Payment Status")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(10, 6))
boxplot = train.boxplot(column=['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate'])
plt.title("Box Plot for Loan Amount, Annual Income, and Interest Rate")
plt.ylabel("Values")
plt.show()


column=['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate']

train = train.copy()

for col in column:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train = train[(train[col] >= lower_bound) & (train[col] <= upper_bound)]

# Create the box plot
plt.figure(figsize=(10, 6))
graph = train.boxplot(column=column)
plt.title("Box Plot after Removing Outliers")
plt.ylabel("Values")
plt.show()


le = LabelEncoder()

# find object columns that exist in both train and test
obj_cols = train.select_dtypes(include='object').columns

for col in obj_cols:
    train[col]=le.fit_transform(train[[col]])


x=train.drop(columns=["id","loan_paid_back",'debt_to_income_ratio','credit_score','loan_purpose','grade_subgrade'],axis=1)
y=train["loan_paid_back"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred1=model.predict(x_test)

print("accuracy_score: ",accuracy_score(y_test,y_pred1))
print("f1_score",f1_score(y_test,y_pred1))
print("confusion_matrix: \n",confusion_matrix(y_test,y_pred1))


rf=RandomForestClassifier(random_state=42,n_estimators=10,n_jobs=-2)
rf.fit(x_train,y_train)
y_pred2=rf.predict(x_test)

print("accuracy_score: ",accuracy_score(y_test,y_pred2))
print("f1_score",f1_score(y_test,y_pred2))
print("confusion_matrix:\n ",confusion_matrix(y_test,y_pred2))


xgb=XGBClassifier(n_jobs=-2,random_state=42)
xgb.fit(x_train,y_train)
y_pred3=xgb.predict(x_test)

print("accuracy_score: ",accuracy_score(y_test,y_pred3))
print("f1_score",f1_score(y_test,y_pred3))
print("confusion_matrix: \n",confusion_matrix(y_test,y_pred3))


models = {
    "DecisionTreeClassifier": y_pred1,
    "RandomForestClassifier": y_pred2,
    "XGBClassifier": y_pred3
}


print('''
==========================================
 PLOT 1: Confusion Matrix (Graph of Each)
==========================================
      ''')
# We create a figure with 3 subplots side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, preds) in enumerate(models.items()):
    cm = confusion_matrix(y_test, preds)
    acc = accuracy_score(y_test, preds)
    
    # Create Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    
    # Labels
    axes[i].set_title(f"{name}\nAccuracy: {acc:.2f}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.show()


print(''' 
    ==========================================
       PLOT 2: Accuracy Comparison
    ==========================================
''')
accuracies = [accuracy_score(y_test, p) for p in models.values()]
names = list(models.keys())

plt.figure(figsize=(8, 5))
bars = plt.bar(names, accuracies, color=['#4CAF50', '#FF9800', '#2196F3'])

plt.title("Model Performance Comparison")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1.1) # Set y-axis limit to be slightly above 1 for text

# Add numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, 
             f"{yval:.2f}", ha='center', fontweight='bold')

plt.show()


# pickle to save model
import pickle
with open('loan_payback_model.pkl', 'wb') as file:
    pickle.dump(xgb, file)

