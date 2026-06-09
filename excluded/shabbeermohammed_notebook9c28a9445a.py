import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from google.colab import drive



df = pd.read_csv("/kaggle/input/carnival-risk-analytics-challenge/train.csv")
df.head()


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Annual Income'].fillna(df['Annual Income'].median(), inplace=True)
df['Marital Status'].fillna('Single', inplace=True)
df['Number of Children'].fillna(0, inplace=True)
df['Education Level'].fillna(df['Education Level'].mode()[0], inplace=True)
df['Occupation'].fillna('Unemployed', inplace=True)
df['Health Score'].fillna(df['Health Score'].median(), inplace=True)
df['Location'].fillna(df['Location'].mode()[0], inplace=True)
df['Policy Type'].fillna('Basic', inplace=True)
df['Previous Claims'].fillna(0, inplace=True)
df['Vehicle Age'].fillna(df['Vehicle Age'].median(), inplace=True)
df['Credit Score'].fillna(df['Credit Score'].median(), inplace=True)
df['Insurance Duration'].fillna(df['Insurance Duration'].median(), inplace=True)
df['Customer Feedback'].fillna('Unknown', inplace=True)
df['Smoking Status'].fillna('No', inplace=True)
df['Exercise Frequency'].fillna('Rarely', inplace=True)
df['Property Type'].fillna(df['Property Type'].mode()[0], inplace=True)


df.isnull().sum()


df['Marital Status'] = df['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
df['Education Level'] = df['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
df['Occupation'] = df['Occupation'].map({'Unemployed': 0, 'Self-Employed': 1, 'Employed': 2})
df['Location'] = df['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
df['Policy Type'] = df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
df['Customer Feedback'] = df['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2, 'Unknown': 3})
df['Smoking Status'] = df['Smoking Status'].map({'No': 0, 'Yes': 1})
df['Exercise Frequency'] = df['Exercise Frequency'].map({'Rarely': 0, 'Monthly': 1, 'Weekly': 2, 'Daily' : 3})
df['Property Type'] = df['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})


df.head()


df['Income_per_Child'] = df['Annual Income'] / (df['Number of Children'] + 1)
df['Claims_per_Year'] = df['Previous Claims'] / (df['Insurance Duration'] + 1)
df['Credit_per_Income'] = df['Credit Score'] / (df['Annual Income'] + 1)
df['Health_to_Age'] = df['Health Score'] / (df['Age'] + 1)
df['Vehicle_Risk'] = df['Vehicle Age'] * (1 + df['Previous Claims'])

x = df[["Income_per_Child", "Claims_per_Year", "Marital Status", "Credit_per_Income", "Education Level", "Occupation", "Health_to_Age", "Location", "Policy Type", "Vehicle_Risk", "Credit Score", "Insurance Duration", "Customer Feedback", "Smoking Status", "Exercise Frequency", "Property Type"]]
scaler = StandardScaler()
x = scaler.fit_transform(x)

y = df[["Premium Amount"]]


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5, random_state=42)

model = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    random_state=42,
    tree_method='hist'
)
model.fit(x_train, y_train.values.ravel())


y_pred = model.predict(x_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Root Mean Squared Error:", rmse)


df['Premium Amount'].describe()


test = pd.read_csv("/kaggle/input/carnival-risk-analytics-challenge/test.csv")
test.head()


test['Age'] = test['Age'].fillna(test['Age'].median())
test['Annual Income'] = test['Annual Income'].fillna(test['Annual Income'].median())
test['Marital Status'] = test['Marital Status'].fillna('Single')
test['Number of Children'] = test['Number of Children'].fillna(0)
test['Education Level'] = test['Education Level'].fillna(test['Education Level'].mode()[0])
test['Occupation'] = test['Occupation'].fillna('Unemployed')
test['Health Score'] = test['Health Score'].fillna(test['Health Score'].median())
test['Location'] = test['Location'].fillna(test['Location'].mode()[0])
test['Policy Type'] = test['Policy Type'].fillna('Basic')
test['Previous Claims'] = test['Previous Claims'].fillna(0)
test['Vehicle Age'] = test['Vehicle Age'].fillna(test['Vehicle Age'].median())
test['Credit Score'] = test['Credit Score'].fillna(test['Credit Score'].median())
test['Insurance Duration'] = test['Insurance Duration'].fillna(test['Insurance Duration'].median())
test['Customer Feedback'] = test['Customer Feedback'].fillna('Unknown')
test['Smoking Status'] = test['Smoking Status'].fillna('No')
test['Exercise Frequency'] = test['Exercise Frequency'].fillna('Rarely')
test['Property Type'] = test['Property Type'].fillna(test['Property Type'].mode()[0])


test['Marital Status'] = test['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
test['Education Level'] = test['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
test['Occupation'] = test['Occupation'].map({'Unemployed': 0, 'Self-Employed': 1, 'Employed': 2})
test['Location'] = test['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
test['Policy Type'] = test['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
test['Customer Feedback'] = test['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2, 'Unknown': 3})
test['Smoking Status'] = test['Smoking Status'].map({'No': 0, 'Yes': 1})
test['Exercise Frequency'] = test['Exercise Frequency'].map({'Rarely': 0, 'Monthly': 1, 'Weekly': 2, 'Daily' : 3})
test['Property Type'] = test['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})
test.head()


test['Income_per_Child'] = test['Annual Income'] / (test['Number of Children'] + 1)
test['Claims_per_Year'] = test['Previous Claims'] / (test['Insurance Duration'] + 1)
test['Credit_per_Income'] = test['Credit Score'] / (test['Annual Income'] + 1)
test['Health_to_Age'] = test['Health Score'] / (test['Age'] + 1)
test['Vehicle_Risk'] = test['Vehicle Age'] * (1 + test['Previous Claims'])

x_test = test[["Income_per_Child", "Claims_per_Year", "Marital Status", "Credit_per_Income", "Education Level", "Occupation", "Health_to_Age", "Location", "Policy Type", "Vehicle_Risk", "Credit Score", "Insurance Duration", "Customer Feedback", "Smoking Status", "Exercise Frequency", "Property Type"]]
scaler = StandardScaler()
x_test = scaler.fit_transform(x_test)


y_pred_test = model.predict(x_test)


submission = pd.DataFrame({"id": test["id"], "Premium Amount": y_pred_test.flatten()})
submission.to_csv("/kaggle/working/submission.csv", index=False)

