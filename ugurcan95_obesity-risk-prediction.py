import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)


df=pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')


df.head()


df.shape


df.info()


df.isnull().sum()


plt.figure(figsize=(17,10))
corr = df.drop('id',axis=1).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='NObeyesdad', y='Age')
plt.title('Age Distribution by Obesity Classification')
plt.xlabel('Obesity Classification')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Gender', hue='NObeyesdad')
plt.title('Obesity Classification by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Obesity Classification')
plt.show()


sns.countplot(data=df, x='family_history_with_overweight', hue='NObeyesdad')
plt.title('Obesity Classification by Family History of Overweight')
plt.xlabel('Family History of Overweight')
plt.ylabel('Count')
plt.legend(title='Obesity Classification')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Height', y='Weight', hue='NObeyesdad', alpha=0.7)
plt.title('Height vs. Weight by Obesity Classification')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(title='Obesity Classification')
plt.show()


plt.figure(figsize=(8, 6))
df['NObeyesdad'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Obesity Classification')
plt.ylabel('')
plt.show()


sns.countplot(data=df, x='SMOKE', hue='NObeyesdad')
plt.title('Obesity Classification by Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.legend(title='Obesity Classification')
plt.show()


sns.pairplot(df, hue='NObeyesdad', vars=['Age', 'Height', 'Weight', 'CH2O'])
plt.title('Pair Plot of Numerical Features by Obesity Classification')
plt.show()


df.info()


def print_unique_values(df):
    for column in df.select_dtypes(include=['object']).columns:
        print(f"{column}: {df[column].unique()}\n")

print_unique_values(df)


def convert_to_numeric(df):
    gender_mapping = {'Male': 1, 'Female': 0}
    family_history_mapping = {'yes': 1, 'no': 0}
    FAVC_mapping = {'yes': 1, 'no': 0}
    CAEC_mapping = {'Sometimes': 1, 'Frequently': 2, 'no': 0, 'Always': 3}
    SMOKE_mapping = {'no': 0, 'yes': 1}
    SCC_mapping = {'no': 0, 'yes': 1}
    CALC_mapping = {'Sometimes': 1, 'no': 0, 'Frequently': 2, 'Always':3}
    MTRANS_mapping = {'Public_Transportation': 0, 'Automobile': 1, 'Walking': 2, 'Motorbike': 3, 'Bike': 4}

    df['Gender'] = df['Gender'].map(gender_mapping)
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map(family_history_mapping)
    df['FAVC'] = df['FAVC'].map(FAVC_mapping)
    df['CAEC'] = df['CAEC'].map(CAEC_mapping)
    df['SMOKE'] = df['SMOKE'].map(SMOKE_mapping)
    df['SCC'] = df['SCC'].map(SCC_mapping)
    df['CALC'] = df['CALC'].map(CALC_mapping)
    df['MTRANS'] = df['MTRANS'].map(MTRANS_mapping)

    return df


df = convert_to_numeric(df)


NObeyesdad_mapping = {
    'Overweight_Level_II': 2,
    'Normal_Weight': 0,
    'Insufficient_Weight': 1,
    'Obesity_Type_III': 5,
    'Obesity_Type_II': 4,
    'Overweight_Level_I': 3,
    'Obesity_Type_I': 6
}

df['NObeyesdad'] = df['NObeyesdad'].map(NObeyesdad_mapping)


x=df.drop(['id','NObeyesdad'],axis=1)
y=df[['NObeyesdad']]


def algo_test(x, y):
    models = [
        MultinomialNB(),
        BernoulliNB(),
        GaussianNB(),
        LogisticRegression(max_iter=1000),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        KNeighborsClassifier(),
        XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    ]

    algorithms = [
        "MultinomialNB", "BernoulliNB", "GaussianNB", "LogisticRegression",
        "DecisionTreeClassifier", "RandomForestClassifier",
        "AdaBoostClassifier", "GradientBoostingClassifier", "KNeighborsClassifier",
        "XGBClassifier"
    ]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    results = []

    for model, name in zip(models, algorithms):
        model.fit(x_train, y_train)
        predict = model.predict(x_test)

        cm = confusion_matrix(y_test, predict)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        report = classification_report(y_test, predict, output_dict=True)
        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        metrics = {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Accuracy', ascending=False)

    return results_df


algo_test(x, y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier()
model.fit(x_train, y_train)

predict = model.predict(x_test)

accuracy = accuracy_score(y_test, predict)
report = classification_report(y_test, predict)

print(f"Accuracy: {accuracy}, \nReport:\n {report}")


with open('obesity_risk.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


test_df.CALC.value_counts()


test_df = convert_to_numeric(test_df)


pred_x=test_df.drop(['id'],axis=1)


predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['id'] = test_df['id']
submision['NObeyesdad'] = predictions


reverse_NObeyesdad_mapping = {v: k for k, v in NObeyesdad_mapping.items()}
submision['NObeyesdad'] = submision['NObeyesdad'].map(reverse_NObeyesdad_mapping)


submision.head()


submision.to_csv('submission.csv', index=False)

