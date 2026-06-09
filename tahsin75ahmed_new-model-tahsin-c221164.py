import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

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

# Train a Random Forest Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.2f}")

# Load the test dataset
solution = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Preprocess the test dataset
# Encode categorical variables
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



# Rename the 'id' column to 'ID' and save the predictions to Submission.csv
solution.rename(columns={'id': 'ID'}, inplace=True)
solution[['ID', 'satisfaction']].to_csv("submission.csv", index=False)


solution.head()


import pandas as pd

# Load dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Show first few rows
print(train_data.head())

# Check basic info
print(train_data.info())

# Check for missing values
print("\nMissing Values:\n", train_data.isnull().sum())


encoded = train_data.copy()
from sklearn.preprocessing import LabelEncoder
for col in categorical_columns:
    le = LabelEncoder()
    encoded[col] = le.fit_transform(encoded[col])
plt.figure(figsize=(12,8))
sns.heatmap(encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



encoded = train_data.copy()
from sklearn.preprocessing import LabelEncoder
for col in categorical_columns:
    le = LabelEncoder()
    encoded[col] = le.fit_transform(encoded[col])


plt.figure(figsize=(12,8))
sns.heatmap(encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns.drop(['id'])

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='satisfaction', y=col, data=train_data)
    plt.title(f'{col} vs Satisfaction')
    plt.tight_layout()
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=train_data, x='satisfaction')
plt.title('Satisfaction Distribution')
plt.show()


encoded = train_data.copy()
from sklearn.preprocessing import LabelEncoder
for col in categorical_columns:
    le = LabelEncoder()
    encoded[col] = le.fit_transform(encoded[col])

plt.figure(figsize=(12,8))
sns.heatmap(encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns.drop(['id'])

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='satisfaction', y=col, data=train_data)
    plt.title(f'{col} vs Satisfaction')
    plt.tight_layout()
    plt.show()


sns.countplot(data=train_data, x='Gender', hue='satisfaction')
plt.title("Gender vs Satisfaction")
plt.show()

sns.countplot(data=train_data, x='Class', hue='satisfaction')
plt.title("Class vs Satisfaction")
plt.show()


import missingno as msno
msno.matrix(train_data)
plt.show()

