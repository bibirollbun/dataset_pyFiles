import pandas as pd
pd.set_option('display.max_columns',100)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import normalize, scale
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df.head()


df.shape


df.isnull().sum()


df.describe().T


df['Target'].unique()


for col in df.columns:
    print(f'{col} has {df[col].nunique()} values')


plt.figure(figsize=(30,25))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='viridis');


order = df['Target'].value_counts().index
ax = sns.countplot(x=df['Target'], order=order)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)


order = df['Marital status'].value_counts().index
ax = sns.countplot(x=df['Marital status'], order=order)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)


sns.countplot(x="Marital status", hue="Target", data=df, palette="viridis")
plt.title("Marital Status vs. Student Outcome");


plt.figure(figsize=(15,10))
order = df['Nacionality'].value_counts().index
ax = sns.countplot(x=df['Nacionality'], order=order)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)


sns.histplot(df["Age at enrollment"], bins=15, kde=True, color="teal")
plt.title("Distribution of Age at Enrollment");


sns.boxplot(x="Target", y="Admission grade", data=df, palette="Set2")
plt.title("Admission Grade Distribution by Target Category");


sns.countplot(x="Tuition fees up to date", hue="Target", data=df, palette="coolwarm")
plt.title("Tuition Fees Payment vs. Student Outcome");


sns.boxplot(x="Target", y="Curricular units 1st sem (grade)", data=df, palette="Set1")
plt.title("1st Semester Grade Distribution by Target");


x=df.drop(['Target','id'],axis=1)
y=df[['Target']]


from sklearn.preprocessing import normalize, scale 
x_new=scale(x) 
x=pd.DataFrame(x_new,columns=x.columns)


def classification_algo(x, y, confusion_mtr=False, classification_rpt=False):
    g = GaussianNB()
    b = BernoulliNB()
    l = LogisticRegression()
    d = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    h = GradientBoostingClassifier()
    k = KNeighborsClassifier()
    
    algos = [g, b, l, d, rf, h, k]
    algo_names = ['Gaussian NB', 'Bernoulli NB', 'Logistic Regression', 
                  'Decision Tree Classifier', 'Random Forest Classifier', 
                  'Gradient Boosting Classifier', 'KNeighbors Classifier']

    accuracy = []
    confusion = []
    classification = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Prepare a DataFrame to store results
    result = pd.DataFrame(columns=['Accuracy Score', 'Confusion Matrix', 'Classification Report'], 
                          index=algo_names)

    labels = sorted(y["Target"].unique())

    for algo in algos:
        p = algo.fit(x_train, y_train).predict(x_test)
        accuracy.append(accuracy_score(y_test, p))
        confusion.append(confusion_matrix(y_test, p, labels=labels))
        classification.append(classification_report(y_test, p))

    # Store results
    result['Accuracy Score'] = accuracy
    result['Confusion Matrix'] = confusion
    result['Classification Report'] = classification

    # Sort results by accuracy
    r_table = result.sort_values('Accuracy Score', ascending=False)
    
    if confusion_mtr:
        for index, row in r_table.iterrows():
            confusion_mat = np.array(row['Confusion Matrix'])
            print(f"Confusion Matrix of {index}")
            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_mat, annot=True, fmt="d", 
                        xticklabels=labels, yticklabels=labels, cmap="Blues",cbar=None)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.show()
    
    if classification_rpt:
        for index, row in r_table.iterrows():
            print(f"Classification Report of {index}:")
            print(row['Classification Report'])

    return r_table[['Accuracy Score']]


classification_algo(x,y,confusion_mtr=True,classification_rpt=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
GBC= GradientBoostingClassifier()
model = GBC.fit(x_train, y_train)


import joblib
joblib.dump(model, 'best_model.pkl')


df_test=pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


submission1=pd.DataFrame({
    'id':df_test['id']
})


df_test.drop('id',axis=1,inplace=True)


df_test_new=scale(df_test) 
df_test=pd.DataFrame(df_test_new,columns=df_test.columns)


predictions=model.predict(df_test)


submission1['Target']=predictions


submission1.to_csv("submission.csv", index=False)


x=df.drop(['Target','id'],axis=1)
y=df[['Target']]


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y['Target'])


x_train , x_test, y_train, y_test =train_test_split(x,y,test_size=.2,random_state=42)


model2=Sequential()
model2.add(Dense(8, activation='relu'))
model2.add(Dense(32,activation='relu')) 
model2.add(Dense(64,activation='relu')) 
model2.add(Dense(128,activation='relu'))
model2.add(Dense(64,activation='relu'))
model2.add(Dense(32,activation='relu'))
model2.add(Dense(3,activation='softmax'))
model2.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


x_train.shape,y_train.shape


history=model2.fit(x_train,y_train,epochs=100,validation_split=.20,verbose=0)


predictions=model2.predict(x_test)
predictions_labels = np.argmax(predictions, axis=1)
accuracy_score(predictions_labels,y_test)


df_test=pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


submission2=pd.DataFrame({
    'id':df_test['id']
})


df_test.drop('id',axis=1,inplace=True)


predictions2=model2.predict(df_test)
class_labels = ['Graduate', 'Dropout', 'Enrolled']
predictions_labels = [class_labels[i] for i in np.argmax(predictions2, axis=1)]


submission2['Target']=predictions_labels


submission2.to_csv("submission.csv", index=False)

