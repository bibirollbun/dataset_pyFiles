# Importing Libraries
import pandas as pd
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# Loading datasets
df_train = pd.read_csv('/kaggle/input/loan-approval/train.csv')
df_test = pd.read_csv('/kaggle/input/loan-approval/test.csv')


df_train.head(5)


print('----------Displaying Columns----------')
df_train.columns


df_train.info()


df_test.info()


df_train.describe()


df_train.isnull().sum()


# Selecting columns with categorical data types (object columns like strings or categories)
categorical_columns = df_train.select_dtypes(include=['object']).columns
# Selecting columns with numerical data types (excluding object columns, so these are numbers like int or float)
numerical_columns = df_train.select_dtypes(exclude=['object']).columns


# Lets see uniqure values of categorical columns
for column in categorical_columns:
  num_unique = df_train[column].nunique()
  print(f'{column} has {num_unique} unique values')


categorical_columns.tolist()


print('Numerical Columns')
numerical_columns.tolist()


test_ids = df_test['id']
target_column = 'loan_status'

df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)


# Histogram plot
numerical_columns_to_plot = ['person_age', 'person_income', 'loan_amnt']
plt.figure(figsize=(10,10))
for i, column in enumerate(numerical_columns_to_plot):
  plt.subplot(3, 1, i+1)
  sns.histplot(data=df_train, x=column, kde=False)


# Different Count Plots
categorical_columns_to_plot = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for i, column in enumerate(categorical_columns_to_plot):
  plt.figure(figsize=(5,5))
  sns.countplot(data=df_train, x=column)
  plt.title(f'Count Plot of {column}')
  xticklabels = plt.xticks(rotation=45)


# Count Plots
for column in categorical_columns_to_plot:
  plt.figure(figsize=(5,5))
  pd.crosstab(df_train[column], df_train[target_column]).plot(kind='bar', stacked=True)
  plt.title(f'Stacked bar plot of {column}, and Target')
  plt.xlabel(column)
  plt.ylabel('Counts')
  plt.show()


# Pie Plot
class_counts = df_train[target_column].value_counts().sort_index()
labels = ['Loan not approved', 'Loan approved']
plt.figure(figsize=(5,5))
plt.pie(class_counts, labels=labels, autopct='%.2f', startangle=90)
plt.title('Pie Chart of Target Variable')
plt.show()


# Heat Map
corr = df_train.corr(numeric_only=True)
plt.figure(figsize=(8,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Heat Map')
plt.show()


# Encoding categorical variables in the training and testing datasets using LabelEncoder to convert them into numerical values for model compatibility.
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
label_encoder = LabelEncoder()
for column in categorical_columns:
  df_train[column] = label_encoder.fit_transform(df_train[column])
for column in categorical_columns:
  df_test[column] = label_encoder.fit_transform(df_test[column])


x = df_train.drop(target_column, axis=1)
y = df_train[target_column]


# Splitting dataset into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)


# Training Logistic Regression Model on Data
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


# Training Random Forest model on Data
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train, y_train)


# Training gradient boosting model on data
from sklearn.ensemble import GradientBoostingClassifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(x_train, y_train)


# Testing Logistic Regression
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = log_reg.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Testing Random Forest Model
y_pred = rf_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Test Gradient Boosting Model
y_pred = gb_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Gradient Boosting Accuracy: {accuracy}')


sub = pd.read_csv('/kaggle/input/loan-approval/sample_submission.csv')
sub[target_column] = rf_classifier.predict(df_test)
sub.to_csv('submission.csv', index=False)




