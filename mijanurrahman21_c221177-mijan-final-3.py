import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier  # Changed model

# Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Data preprocessing for training data
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

# Train a LightGBM Classifier (better than Random Forest)
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.2f}")

# Load the test dataset
solution = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Preprocess the test dataset
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        solution[col] = label_encoders[col].transform(solution[col])

# Select features for prediction
X_test = solution.drop(columns=['Unnamed: 0', 'id'], errors='ignore')

# Handle missing values in test data
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Make predictions
solution['satisfaction'] = model.predict(X_test)

# Map predictions back to original labels
solution['satisfaction'] = label_encoders['satisfaction'].inverse_transform(solution['satisfaction'])

# Optionally display the first few results
print(solution[['satisfaction']].head())



# Check for missing values
print("Missing values in training data:\n", train_data.isnull().sum())

# Drop unnecessary columns (like unnamed index or ID columns)
train_data = train_data.drop(columns=['Unnamed: 0', 'id'], errors='ignore')

# Fill missing values using mean (you can also use median or mode)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
train_data_imputed = train_data.copy()
numeric_cols = train_data_imputed.select_dtypes(include=['float64', 'int64']).columns

train_data_imputed[numeric_cols] = imputer.fit_transform(train_data_imputed[numeric_cols])

# Confirm no more missing values
print(train_data_imputed.isnull().sum())



from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
categorical_cols = train_data_imputed.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_data_imputed[col] = le.fit_transform(train_data_imputed[col])
    label_encoders[col] = le

# Final cleaned and processed data
print(train_data_imputed.head())



import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=train_data, x='satisfaction')
plt.title("Satisfaction Distribution")
plt.show()



sns.countplot(data=train_data, x='Gender', hue='satisfaction')
plt.title("Gender vs Satisfaction")
plt.show()



import numpy as np

plt.figure(figsize=(12, 8))
sns.heatmap(train_data_imputed.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()



sns.boxplot(data=train_data, x='satisfaction', y='Flight Distance')
plt.title("Flight Distance vs Satisfaction")
plt.show()


