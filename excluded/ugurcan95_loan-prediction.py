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


df=pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')


df.head()


df.shape


df.info()


df.isnull().sum()


plt.figure(figsize=(17,10))
corr = df.drop('id',axis=1).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='person_home_ownership', hue='loan_status', palette='Set2')
plt.title('Loan Status by Home Ownership')
plt.xlabel('Home Ownership Status')
plt.ylabel('Count')
plt.legend(title='Loan Status')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='loan_status', y='loan_amnt', palette='Set3')
plt.title('Loan Amount Distribution by Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Loan Amount')
plt.show()


plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x='loan_grade', y='loan_int_rate', palette='muted')
plt.title('Interest Rate Distribution by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Interest Rate')
plt.show()


sns.pairplot(df, vars=['person_age', 'person_income', 'loan_amnt'], hue='loan_status', palette='Set1')
plt.title('Pair Plot of Age, Income, and Loan Amount')
plt.show()


plt.figure(figsize=(8, 6))
df['loan_status'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Loan Status Classification')
plt.ylabel('')
plt.show()


df.info()


def print_unique_values(df):
    for column in df.select_dtypes(include=['object']).columns:
        print(f"{column}: {df[column].unique()}\n")

print_unique_values(df)


def convert_to_numeric(df):
    homeOwners_mapping = {
        'RENT': 0,
        'OWN': 1,
        'MORTGAGE': 2,
        'OTHER': 3
        }
    loanIntent_mapping = {
        'EDUCATION': 0,
        'MEDICAL': 1,
        'PERSONAL': 2,
        'VENTURE': 3,
        'DEBTCONSOLIDATION': 4,
        'HOMEIMPROVEMENT': 5
        }
    loanGrade_mapping = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6
        }
    defaultOnFile_mapping = {
        'N': 0,
        'Y': 1
        }

    df['person_home_ownership'] = df['person_home_ownership'].map(homeOwners_mapping)
    df['loan_intent'] = df['loan_intent'].map(loanIntent_mapping)
    df['loan_grade'] = df['loan_grade'].map(loanGrade_mapping)
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(defaultOnFile_mapping)

    return df


df = convert_to_numeric(df)


x=df.drop(['id','loan_status'],axis=1)
y=df[['loan_status']]


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

model = XGBClassifier()
model.fit(x_train, y_train)

predict = model.predict(x_test)

accuracy = accuracy_score(y_test, predict)
report = classification_report(y_test, predict)

print(f"Accuracy: {accuracy}, \nReport:\n {report}")


with open('loan_prediction.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


test_df = convert_to_numeric(test_df)


pred_x=test_df.drop(['id'],axis=1)


predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['id'] = test_df['id']
submision['loan_status'] = predictions


submision.head()


submision.to_csv('submission.csv', index=False)

