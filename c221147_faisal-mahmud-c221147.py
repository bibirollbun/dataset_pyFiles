import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Display the first few rows of the training data
print("Training Data Head:")
display(train_data.head())

# Display the data types of the training data
print("\nTraining Data Info:")
display(train_data.info())




# Load the test dataset
test_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")

# Display the first few rows of the test data
print("\nTest Data Head:")
display(test_data.head())

# Display the data types of the test data
print("\nTest Data Info:")
display(test_data.info())


# Check for missing values in the training data
print("\nTraining Data Missing Values:")
display(train_data.isnull().sum())

# Display descriptive statistics for the training data
print("\nTraining Data Description:")
display(train_data.describe())
# Check for missing values in the test data
print("\nTest Data Missing Values:")
display(test_data.isnull().sum())

# Display descriptive statistics for the test data
print("\nTest Data Description:")
display(test_data.describe())


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Handle missing values in 'Arrival Delay in Minutes' using the mean
imputer = SimpleImputer(strategy='mean')
train_data['Arrival Delay in Minutes'] = imputer.fit_transform(train_data[['Arrival Delay in Minutes']])
test_data['Arrival Delay in Minutes'] = imputer.transform(test_data[['Arrival Delay in Minutes']])

# Encode categorical variables in the training data
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Encode categorical variables in the test data using the same encoders
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        test_data[col] = label_encoders[col].transform(test_data[col])



# Display the first few rows and info of the preprocessed training data
print("Preprocessed Training Data Head:")
display(train_data.head())
print("\nPreprocessed Training Data Info:")
display(train_data.info())

# Display the first few rows and info of the preprocessed test data
print("\nPreprocessed Test Data Head:")
display(test_data.head())
print("\nPreprocessed Test Data Info:")
display(test_data.info())


X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Classifier model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")


X_test = test_data.drop(columns=['Unnamed: 0', 'id'], errors='ignore')

# Make predictions on the test data
test_predictions = model.predict(X_test)

# Map predictions back to original labels
test_predictions_labels = label_encoders['satisfaction'].inverse_transform(test_predictions)

# Create the submission DataFrame
submission_df = pd.DataFrame({'ID': test_data['id'], 'satisfaction': test_predictions_labels})

# Save the submission file
submission_df.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created successfully!")
display(submission_df.head())


import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style('whitegrid')

# Visualize the distribution of the target variable 'satisfaction'
plt.figure(figsize=(6, 4))
sns.countplot(x='satisfaction', data=train_data, palette='viridis', hue='satisfaction')
plt.title('Distribution of Customer Satisfaction')
plt.xlabel('Satisfaction (0: Neutral or Dissatisfied, 1: Satisfied)')
plt.ylabel('Count')
plt.show()


# Visualize the distribution of 'Customer Type'
plt.figure(figsize=(6, 4))
sns.countplot(x='Customer Type', data=train_data, palette='viridis', hue='Customer Type')
plt.title('Distribution of Customer Type')
plt.xlabel('Customer Type (0: Loyal Customer, 1: disloyal Customer)')
plt.ylabel('Count')
plt.show()


# Visualize the relationship between 'Type of Travel' and 'satisfaction'
plt.figure(figsize=(6, 4))
sns.countplot(x='Type of Travel', hue='satisfaction', data=train_data, palette='viridis')
plt.title('Customer Satisfaction by Type of Travel')
plt.xlabel('Type of Travel (0: Business travel, 1: Personal Travel)')
plt.ylabel('Count')
plt.show()


# Visualize the distribution of 'Class'
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=train_data, palette='viridis', hue='Class')
plt.title('Distribution of Class')
plt.xlabel('Class (0: Business, 1: Eco, 2: Eco Plus)')
plt.ylabel('Count')
plt.show()


# Visualize the relationship between 'Class' and 'satisfaction'
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', hue='satisfaction', data=train_data, palette='viridis')
plt.title('Customer Satisfaction by Class')
plt.xlabel('Class (0: Business, 1: Eco, 2: Eco Plus)')
plt.ylabel('Count')
plt.show()


# Visualize the distribution of 'Age'
plt.figure(figsize=(8, 5))
sns.histplot(data=train_data, x='Age', bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# Visualize the relationship between 'Age' and 'satisfaction'
plt.figure(figsize=(8, 5))
sns.boxplot(x='satisfaction', y='Age', data=train_data, palette='viridis', hue='satisfaction')
plt.title('Customer Satisfaction by Age')
plt.xlabel('Satisfaction (0: Neutral or Dissatisfied, 1: Satisfied)')
plt.ylabel('Age')
plt.show()

