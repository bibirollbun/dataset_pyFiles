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


df=pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')


df.head()


df.shape


df.isnull().sum()


df.info()


df.describe().T


df['NObeyesdad'].unique()


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(numeric_only=True),annot=True);


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0], color='blue')
axes[0].set_title("Age Distribution")

sns.histplot(df['Height'], bins=20, kde=True, ax=axes[1], color='green')
axes[1].set_title("Height Distribution")

sns.histplot(df['Weight'], bins=20, kde=True, ax=axes[2], color='red')
axes[2].set_title("Weight Distribution")

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 5))
sns.boxplot(x='NObeyesdad', y='Weight', data=df, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Weight Distribution Across Obesity Levels")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x='NObeyesdad', data=df, palette="viridis", order=df['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Obesity Level Distribution")
plt.show()


plt.figure(figsize=(8, 5))
sns.violinplot(x='NObeyesdad', y='FAF', data=df, palette="muted")
plt.xticks(rotation=45)
plt.title("Physical Activity Across Obesity Levels")
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(x="MTRANS", hue="NObeyesdad", data=df, palette="Paired")
plt.xticks(rotation=45)
plt.title("Transportation Mode by Obesity Level")
plt.show()


x=df.drop(['id','NObeyesdad'],axis=1)
y=df[['NObeyesdad']]


x=pd.get_dummies(x,drop_first=True)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb

def classification_algo(x, y, confusion_mtr=False, classification_rpt=False):
    g = GaussianNB()
    b = BernoulliNB()
    l = LogisticRegression()
    d = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    h = GradientBoostingClassifier()
    k = KNeighborsClassifier()
    lgbm = LGBMClassifier(verbose=-1)  

    algos = [g, b, l, d, rf, h, k, lgbm]
    algo_names = ['Gaussian NB', 'Bernoulli NB', 'Logistic Regression', 
                  'Decision Tree Classifier', 'Random Forest Classifier', 
                  'Gradient Boosting Classifier', 'KNeighbors Classifier',
                  'LightGBM Classifier'] 

    accuracy = []
    confusion = []
    classification = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Prepare a DataFrame to store results
    result = pd.DataFrame(columns=['Accuracy Score', 'Confusion Matrix', 'Classification Report'], 
                          index=algo_names)

    labels = sorted(y["NObeyesdad"].unique())

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
                        xticklabels=labels, yticklabels=labels, cmap="Blues")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.show()
    
    if classification_rpt:
        for index, row in r_table.iterrows():
            print(f"Classification Report of {index}:")
            print(row['Classification Report'])

    return r_table[['Accuracy Score']]


classification_algo(x,y,confusion_mtr=True,classification_rpt=True)


x=df.drop(['id','NObeyesdad'],axis=1)
y=df[['NObeyesdad']]
x=pd.get_dummies(x,drop_first=True)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
lgbm = LGBMClassifier(verbose=-1)  
model=lgbm.fit(x_train,y_train)
predictions=model.predict(x_test)
score=accuracy_score(y_test,predictions)
score


import joblib
joblib.dump(model, 'classification_model.pkl')


df_test=pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')


submission=pd.DataFrame({
    'id':df_test['id']
})


df_test=df_test.drop(['id'],axis=1)


df_test=pd.get_dummies(df_test,drop_first=True)
df_test = df_test.reindex(columns=x.columns, fill_value=0)


predictions=model.predict(df_test)


submission['NObeyesdad']=predictions


submission.to_csv('submission.csv',index=False)

