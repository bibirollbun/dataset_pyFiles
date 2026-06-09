import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Load training dataset
df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# View the first few rows
df.head()



# Drop only if columns exist
df.drop(columns=[col for col in ['Unnamed: 0', 'id'] if col in df.columns], inplace=True)



df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(0)
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)



# Keep only valid satisfaction responses
df = df[df['satisfaction'].isin(['satisfied', 'neutral or dissatisfied'])].copy()

# Create a binary target column
df['satisfaction_binary'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})



# Age Group Bins
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 120], labels=['Young', 'Adult', 'Middle-aged', 'Senior'])

# Delay Flag
df['IsDelayed'] = ((df['Departure Delay in Minutes'] > 10) | (df['Arrival Delay in Minutes'] > 10)).astype(int)

# Comfort Index (average of comfort-related ratings)
df['Comfort_Index'] = df[['Seat comfort', 'Leg room service', 'Cleanliness']].mean(axis=1)

# Digital Experience Index
df['Digital_Exp'] = df[['Inflight wifi service', 'Ease of Online booking', 
                        'Online boarding', 'Checkin service']].mean(axis=1)

# Combined Feature: Class + Type of Travel
df['Class_Travel'] = df['Class'] + " | " + df['Type of Travel']



df.info()
df.head()



# Count of satisfied vs dissatisfied passengers
satisfaction_counts = df['satisfaction_binary'].value_counts().sort_index()

# Plot
plt.figure(figsize=(6,4))
sns.barplot(x=satisfaction_counts.index.map({0: 'Not Satisfied', 1: 'Satisfied'}), 
            y=satisfaction_counts.values, palette='viridis')

plt.title("Overall Satisfaction Rate")
plt.ylabel("Number of Passengers")
plt.xlabel("Satisfaction")
plt.tight_layout()
plt.show()

# Print percentages
satisfaction_percentage = satisfaction_counts / satisfaction_counts.sum() * 100
print("Satisfaction Rate:\n", satisfaction_percentage.round(2))


plt.figure(figsize=(8,4))
sns.countplot(data=df, x='Gender', hue='satisfaction_binary', palette='Set2')
plt.title("Satisfaction Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title='Satisfaction', labels=['Not Satisfied', 'Satisfied'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(data=df, x='Age_Group', hue='satisfaction_binary', palette='Set1')
plt.title("Satisfaction Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title='Satisfaction', labels=['Not Satisfied', 'Satisfied'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(data=df, x='Customer Type', hue='satisfaction_binary', palette='coolwarm')
plt.title("Satisfaction by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Count")
plt.legend(title='Satisfaction', labels=['Not Satisfied', 'Satisfied'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(data=df, x='Class', hue='satisfaction_binary', palette='pastel')
plt.title("Satisfaction by Travel Class")
plt.xlabel("Travel Class")
plt.ylabel("Count")
plt.legend(title='Satisfaction', labels=['Not Satisfied', 'Satisfied'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(data=df, x='Type of Travel', hue='satisfaction_binary', palette='Set3')
plt.title("Satisfaction by Type of Travel")
plt.xlabel("Type of Travel")
plt.ylabel("Count")
plt.legend(title='Satisfaction', labels=['Not Satisfied', 'Satisfied'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(data=df, x='IsDelayed', hue='satisfaction_binary', palette='RdYlGn')
plt.title("Satisfaction vs Flight Delay Status")
plt.xlabel("Flight Delayed (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.legend(title='Satisfaction', labels=['Not Satisfied', 'Satisfied'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='satisfaction_binary', y='Flight Distance', palette='coolwarm')
plt.title("Flight Distance vs Satisfaction")
plt.xlabel("Satisfaction (0 = Not Satisfied, 1 = Satisfied)")
plt.ylabel("Flight Distance")
plt.tight_layout()
plt.show()

comfort_features = ['Seat comfort', 'Leg room service', 'Cleanliness']

plt.figure(figsize=(10,6))
df.groupby('satisfaction')[comfort_features].mean().T.plot(kind='bar', colormap='Paired')
plt.title('Average Comfort Ratings by Satisfaction')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.legend(title='Satisfaction')
plt.tight_layout()
plt.show()

digital_features = ['Inflight wifi service', 'Ease of Online booking', 'Online boarding', 'Checkin service']

plt.figure(figsize=(10,6))
df.groupby('satisfaction')[digital_features].mean().T.plot(kind='bar', colormap='Set2')
plt.title('Average Digital Experience Ratings by Satisfaction')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.legend(title='Satisfaction')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
order = df.groupby('Class_Travel')['satisfaction_binary'].mean().sort_values().index
sns.barplot(data=df, x='Class_Travel', y='satisfaction_binary', order=order, palette='mako')
plt.title('Satisfaction Rate by Class and Type of Travel')
plt.xlabel('Class | Type of Travel')
plt.ylabel('Average Satisfaction')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()





from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Reload training data for modeling
train_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Encode categorical variables
label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

# Drop unnecessary columns
X = train_df.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_df['satisfaction']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")



# Load test data
test_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Encode test data categorical columns
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        test_df[col] = label_encoders[col].transform(test_df[col])

# Prepare test features
X_test = test_df.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Predict
test_df['satisfaction'] = model.predict(X_test)

# Decode predictions back to original labels
test_df['satisfaction'] = label_encoders['satisfaction'].inverse_transform(test_df['satisfaction'])

# ✅ Create submission file
test_df.rename(columns={'id': 'ID'}, inplace=True)
submission = test_df[['ID', 'satisfaction']]
submission.to_csv("submission.csv", index=False)

# Confirm it's saved
import os
print("✅ submission.csv created:", 'submission.csv' in os.listdir())

# Preview
submission.head()


