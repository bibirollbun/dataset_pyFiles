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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# Load the training dataset
train_data = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv')


# Data preprocessing for training data
# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le


# Define features and target variable
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']


# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train an XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)



# Validate the model
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.2f}")



# Load the test dataset
solution = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv')


for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        # Build mapping from classes
        mapping = dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))
        # Map safely, use -1 for unknown labels
        solution[col] = solution[col].map(mapping).fillna(-1).astype(int)



# Select features for prediction
X_test = solution.drop(columns=['Unnamed: 0', 'id'], errors='ignore')


# Handle missing values in test data
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# Make predictions
solution['satisfaction'] = model.predict(X_test)



# Map predictions back to original labels
solution['satisfaction'] = label_encoders['satisfaction'].inverse_transform(solution['satisfaction'])


 # Rename the 'id' column to 'ID' and save the predictions to Submission.csv
solution.rename(columns={'id': 'ID'}, inplace=True)
solution[['ID', 'satisfaction']].to_csv("submission.csv", index=False)
solution.head()


#data analysis
#######
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv')


# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "id"])
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)


# Ensure Age is numeric and clean
df = df[pd.to_numeric(df["Age"], errors="coerce").notnull()]
df["Age"] = df["Age"].astype(int)


# Create Age Groups
df["Age Group"] = pd.cut(df["Age"], bins=[0, 18, 30, 45, 60, 90],
                         labels=["<18", "18-30", "31-45", "46-60", "60+"])


#Q1. What is the overall satisfaction rate?
#Q2. Did women report more satisfaction than men?
#Q3. Was travel class a key factor in satisfaction?
#Q4. Was age a key factor in satisfaction?
#Q5. Were loyal customers more satisfied than first-timers?
#Q6. Was flight distance related to satisfaction?
#Q7. Did any service scores correlate strongly with satisfaction?
#Q8. What combinations guaranteed satisfaction?


# Q1: Overall Satisfaction Rate
df["satisfaction"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Q1: Overall Satisfaction Rate")
plt.ylabel("")
plt.show()


# Q2: Satisfaction by Gender
sns.countplot(data=df, x="Gender", hue="satisfaction")
plt.title("Q2: Satisfaction by Gender")
plt.show()


# Q3: Satisfaction by Travel Class
sns.countplot(data=df, x="Class", hue="satisfaction")
plt.title("Q3: Satisfaction by Travel Class")
plt.show()


# Q4: Satisfaction by Age Group
sns.countplot(data=df, x="Age Group", hue="satisfaction")
plt.title("Q4: Satisfaction by Age Group")
plt.show()


# Q5: Satisfaction by Customer Type
sns.countplot(data=df, x="Customer Type", hue="satisfaction")
plt.title("Q5: Customer Loyalty vs Satisfaction")
plt.show()


# Q6: Flight Distance vs Satisfaction
sns.boxplot(data=df, x="satisfaction", y="Flight Distance")
plt.title("Q6: Flight Distance vs Satisfaction")
plt.show()


# Q7: Correlation Heatmap (Satisfaction encoded)
df_numeric = df.select_dtypes(include=['int64', 'float64']).copy()
df_numeric["satisfaction"] = (df["satisfaction"] == "satisfied").astype(int)
corr = df_numeric.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Q7: Correlation Heatmap with Satisfaction")
plt.show()


# Q8: Combined Effect of Class on Satisfaction
sns.countplot(data=df, x="Class", hue="satisfaction", palette="Set2")
plt.title("Q8: Combined Effect of Class on Satisfaction")
plt.show()


# Q9: Online Boarding Score vs Satisfaction
sns.boxplot(data=df, x="satisfaction", y="Online boarding")
plt.title("Q9: Online Boarding Score vs Satisfaction")
plt.show()

