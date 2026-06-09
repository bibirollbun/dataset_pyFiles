import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

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

# Train an XGBoost Classifier (improved accuracy)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
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


solution.head(10)


train_data.columns


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# To show plots in notebook
%matplotlib inline


# Load the dataset
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Display the first few rows
df.head()


print("ðŸ§¹ Checking for missing values:")
print(df.isnull().sum())


# Structure of the dataset
print("ðŸ“Š Dataset Info:")
print(df.info())

# Dataset shape
print(f"\nTotal rows: {df.shape[0]}, Total columns: {df.shape[1]}")


print("ðŸ“ˆ Summary of numerical columns:")
print(df.describe())


print("ðŸŽ¯ Satisfaction class distribution:")
print(df['satisfaction'].value_counts())

# Visualize class distribution
sns.countplot(data=df, x='satisfaction')
plt.title("Satisfaction Class Distribution")
plt.show()


sns.countplot(data=df, x='Gender', hue='satisfaction')
plt.title("Satisfaction by Gender")
plt.show()


sns.countplot(data=df, x='Customer Type', hue='satisfaction')
plt.title("Satisfaction by Customer Type")
plt.xticks(rotation=15)
plt.show()


sns.countplot(data=df, x='Class', hue='satisfaction')
plt.title("Satisfaction by Class")
plt.show()


sns.boxplot(data=df, x='satisfaction', y='Age')
plt.title("Age vs Satisfaction")
plt.show()

