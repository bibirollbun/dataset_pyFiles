import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')



Data=pd.read_csv('Merged File.csv')


Data.head(5)


Data.tail(5)


Data.shape


Data.columns


Data.info()


Data.describe()


Data.isnull().sum()


Data.duplicated().sum()


Data.nunique()


plt.figure(figsize=(20,12))

sns.pairplot(Data)


# Catgeorical and Numerical Features 

Cat_columns=[feature for feature in Data.columns if Data[feature].dtypes=='object']
Num_columns=[feature for feature in Data.columns if Data[feature].dtypes!='object']


Cat_columns


Num_columns


Data[Cat_columns].head(5)


for i in Cat_columns:
    ax=sns.countplot(Data, x=i)
    for container in ax.containers:
        ax.bar_label(container)
    plt.xlabel(i)
    plt.show()


Data[Num_columns].head(5)


plt.pie(Data['Stage_fear'].value_counts(), labels=['No', 'Yes'], colors=['blue', 'violet'], explode=[0,0.2], shadow=True, autopct='%1.1f')
plt.show()


Data['Stage_fear'].value_counts()


plt.pie(x=Data['Drained_after_socializing'].value_counts(), labels=['No', 'Yes'], colors=['orange', 'blue'], explode=[0, 0.2], shadow=True, autopct='%1.1f')
plt.show()


plt.pie(x=Data['Personality'].value_counts(), labels=['Introvert', 'Extrovert'], colors=['pink', 'green'], autopct='%1.1f', explode=[0, 0.2], shadow=True)
plt.show()


Data['Personality'].value_counts()


Data['Drained_after_socializing'].value_counts()


for i in Num_columns:
    sns.distplot(x=Data[i])
    plt.xlabel(i)
    plt.show()


for i in Num_columns:
    sns.histplot(x=Data[i], kde=True)
    plt.xlabel(i)
    plt.show()


for i in Num_columns:
    sns.histplot(x=Data[i], kde=True, hue=Data['Personality'])
    plt.xlabel(i)
    plt.show()


for i in Cat_columns:
    ax=sns.countplot(Data, x=i, hue=Data['Personality'])
    for container in ax.containers:
        ax.bar_label(container)
    plt.xlabel(i)
    plt.show()


for i in Num_columns:
    sns.scatterplot(Data[i])
    plt.xlabel(i)
    plt.show()


for i in Num_columns:
    sns.boxplot(x=Data[i])
    plt.xlabel(i)
    plt.show()


sns.heatmap(Data[Num_columns].corr(), fmt='.2f', linewidth=0.2, annot=True)


Data.corr()


Data.isnull().sum()


Data.info()


for feature in Num_columns:
    if Data[feature].isnull().any():   # only if NaN exists
        median_value = Data[feature].median()
        Data[feature].fillna(median_value, inplace=True)


for feature in Cat_columns:
    if Data[feature].isnull().any():   # only if NaN exists
        mode_value = Data[feature].mode()[0]
        Data[feature].fillna(mode_value, inplace=True)


Data.isnull().sum()


Data.isnull()


Data.corr()


for i in Cat_columns:
    print(Data[i].value_counts())
    


Data['Personality']=Data['Personality'].replace({'Introvert':0, 'Extrovert':1})


Data['Personality'].value_counts()


X=Data.drop('Personality', axis=1)
y=Data['Personality']


X


y


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=23)


X_train.shape, y_train.shape


X_test.shape, y_test.shape


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


Scaler=StandardScaler()
OHE=OneHotEncoder(drop='first')


preprocessor=ColumnTransformer([
    ('OneHotEncoder', OHE, ['Stage_fear', 'Drained_after_socializing']),
    ('StandardScaler', Scaler, ['id', 'Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Friends_circle_size'])
])


preprocessor


X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)


X_train


X_test


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score


def model_evaluate(true, predicted):
    accuracy=accuracy_score(true, predicted)
    f1=f1_score(true, predicted)
    precision=precision_score(true, predicted)
    recall=recall_score(true, predicted)
    roc_auc=roc_auc_score(true, predicted)
    return accuracy, f1, precision, recall, roc_auc


models={
    'Logistic Regression' : LogisticRegression(),
    'Naive Bayes' : BernoulliNB(),
    'Decision Tree Classifier' : DecisionTreeClassifier(),
    'Random Forest Classifier' : RandomForestClassifier(),
    'Gradient Boosting Classifier' : GradientBoostingClassifier(),
    'AdaBoost Classifier' : AdaBoostClassifier(),
    'KNN' : KNeighborsClassifier(),
    'SVM' : SVC()
    
}    


models


model_list=[]
acc_list=[]
model_objs=[]

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)

    model_train_acc, model_train_f1, model_train_precision, model_train_recall, model_train_roc=model_evaluate(y_train, y_train_pred)
    model_test_acc, model_test_f1, model_test_precision, model_test_recall, model_test_roc=model_evaluate(y_test, y_test_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print("Model Performance for Training Data")
    print(" - Accuracy Score : {:.4f}".format(model_train_acc))
    print(" - F1 Score : {:.4f}".format(model_train_f1))
    print(" - Precision Score : {:.4f}".format(model_train_precision))
    print(" - Recall Score : {:.4f}".format(model_train_recall))
    print(" - Roc Auc Score : {:.4f}".format(model_train_roc))


    print("-------------------------------------------------------------------")

    print("Model Performance for Testing Data")
    print(" - Accuracy Score : {:.4f}".format(model_test_acc))
    print(" - F1 Score : {:.4f}".format(model_test_f1))
    print(" - Precision Score : {:.4f}".format(model_test_precision))
    print(" - Recall Score : {:.4f}".format(model_test_recall))
    print(" - Roc Auc Score : {:.4f}".format(model_test_roc))

    model_objs.append(model)


    acc_list.append(model_test_acc)

    print("\n")
    print("$"*45)
    print("\n")


pd.DataFrame(list(zip(model_list, acc_list)), columns=['Model Name', 'Accuracy Score']).sort_values(by=["Accuracy Score"],ascending=False)































