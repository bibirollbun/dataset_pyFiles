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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test.head()


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df.head()


##****EDA (Exploratory Data Analysis)*****
print("EDA (Exploratory Data Analysis)")

#Load data and check basic info (shape, dtypes, memory usage)
print("Data Types")
print(df.dtypes)
print("DF Shape")
print(df.shape)
print("Memory used by Each Column")
print(df.memory_usage(deep=True))

#Examine target variable distribution and class balance
print(df.value_counts(['Personality']))
print(df['Personality'].value_counts(normalize=True)*100)
proportions = df['Personality'].value_counts(normalize=True)*100
proportions.plot(kind='bar', color=['red','blue'])
plt.ylabel('Percentage')
plt.title('Personality Class Distribution')
plt.show()

#Check for missing values per column
print("Values Missing for each Column")
print(df.isnull().sum())
print("% of values Missing for each Column")
print(round((df.isnull().sum()/df.count())*100))

#Analyze numerical features (distributions, outliers, correlations)
print("Distribution")
df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()
print("outliers")
df.boxplot(figsize=(10,8))
plt.tight_layout()
plt.show()
print("Correlations")
corr = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr, cmap='coolwarm')
plt.tight_layout()
plt.show()

#Analyze categorical features (unique values, frequencies)
print(df.select_dtypes(include='object').apply(lambda x : x.value_counts()))


#****Data Preprocessing****
print("Data Preprocessing")

#Handle missing values (imputation strategy)
def handle_missing_values(df):
    for col in df.select_dtypes(include='number').columns:
        df.fillna({col: df[col].mean()}, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df.fillna({col:df[col].mode()}, inplace=True)

handle_missing_values(df)
handle_missing_values(df_test)

#Encode categorical variables (Label/One-hot/Ordinal encoding)
def encode(df):
    le = LabelEncoder() #used in Yes/No Scenarios
    df['Stage_fear'] = le.fit_transform(df['Stage_fear'])
    df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])

encode(df)
encode(df_test)

#Handle outliers (remove/cap/transform)
#no outliers detected in this case

#Scale numerical features (StandardScaler/MinMaxScaler)
print("No High numerical data, so scaling not necessary, only dropping and saving id column, for later use, as it has vast data, and can skew the MODEL Training")
ids = df['id']
df = df.drop('id', axis=1)
ids_test = df_test['id']
x_test_final = df_test.drop('id', axis=1)

#Split features and target variables
X = df.drop('Personality', axis=1) #features
y = df['Personality'] #axis


#Ensure train/test consistency in preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Apply same transformations (no .fit!) for test data
X_test_final_scaled = scaler.transform(x_test_final)


#****FEATURE ENGINEERING****

#Create interaction features (multiply/divide related features)
def create_features(df):
    df['friends_per_event'] = df['Friends_circle_size'] / (df['Social_event_attendance'] + 1)
    df['social_interaction'] = df['Friends_circle_size'] * df['Social_event_attendance']
    df['Alone_x_Post'] = df['Time_spent_Alone'] * df["Post_frequency"]

create_features(df)
create_features(df_test)
#Generate polynomial features if needed
#Do it yourself

#Create binning/bucketing for continuous variables
#Not neccessary as we have already scaled

#Extract date/time components if applicable
print("No Date/Time columns in the dataset")


#****Model Selection****

#Choose appropriate algorithm family (classification/regression)
print("Since we're predicting Personality (e.g., Introvert vs Extrovert):")
print("Use a Classification algorithm")
print("Because:\n- Output is categorical\n- You’re assigning a class label")

#Start with simple baseline models (LogisticRegression/LinearRegression)
print("Going with Logistic Regression")


#****Model Training & Validation****
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

#Train baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

#Evaluate using appropriate metrics
print("accuracy score = ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label='Extrovert'))
print("Recall:", recall_score(y_test, y_pred, pos_label='Extrovert'))
print("F1 Score:", f1_score(y_test, y_pred, pos_label='Extrovert'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Implement hyperparameter tuning (GridSearch/RandomSearch)
#DIY

#Check for overfitting (train vs validation scores)
print("F1 Score Test:", f1_score(y_test, y_pred, pos_label='Extrovert'))
print("F1 Score Train:", f1_score(y_train, y_train_pred, pos_label='Extrovert'))
print("High train, low val → Overfitting")
print("Similar scores → Good generalization")
print("Both low → Underfitting")


#****Prediction & Submission****

#Preprocess test data using same pipeline
#Done above check all the tasks done on df_test

final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train, y_train)

y_pred_final = final_model.predict(X_test_final_scaled)
submission = pd.DataFrame({
    'id': ids_test,
    'Personality': y_pred_final
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")




