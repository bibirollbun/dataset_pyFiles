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

# Train an XGBoost Classifier
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
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

df = pd.read_csv('submission.csv')
print("Number of rows and columns:", df.shape)
print("Column names:", df.columns.tolist())


import pandas as pd

# Load the CSV file
df = pd.read_csv('submission.csv')

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Print all columns for the first few rows
print(df.head())


print("Data types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())


if 'Customer Type' in df.columns:
    print("Customer Type distribution:\n", df['Customer Type'].value_counts(dropna=False))
else:
    print("'Customer Type' column not found.")

if 'Class' in df.columns:
    print("\nClass distribution:\n", df['Class'].value_counts(dropna=False))
else:
    print("'Class' column not found.")


num_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
num_cols = [col for col in num_cols if col in df.columns]
if num_cols:
    print(df[num_cols].describe())
else:
    print("No specified numerical columns found.")


if 'Class' in df.columns and 'Flight Distance' in df.columns:
    print(df.groupby('Class')['Flight Distance'].mean())
else:
    print("Required columns for this analysis are missing.")


service_cols = [
    'Inflight wifi service', 'Food and drink', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
service_cols = [col for col in service_cols if col in df.columns]
if 'Customer Type' in df.columns and service_cols:
    print(df.groupby('Customer Type')[service_cols].mean())
else:
    print("Required columns for this analysis are missing.")


if 'Class' in df.columns and 'Gender' in df.columns:
    print(pd.crosstab(df['Class'], df['Gender']))
else:
    print("Required columns for this analysis are missing.")


if 'Departure Delay in Minutes' in df.columns:
    print("Departure delays > 30 min:", (df['Departure Delay in Minutes'] > 30).sum())
else:
    print("'Departure Delay in Minutes' column not found.")

if 'Arrival Delay in Minutes' in df.columns:
    print("Arrival delays > 30 min:", (df['Arrival Delay in Minutes'] > 30).sum())
else:
    print("'Arrival Delay in Minutes' column not found.")


if 'Type of Travel' in df.columns and 'Age' in df.columns:
    print(df.groupby('Type of Travel')['Age'].mean())
else:
    print("Required columns for this analysis are missing.")


print("Number of duplicate rows:", df.duplicated().sum())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load your dataset
df = pd.read_csv('your_data.csv')
# Count plot for satisfaction
sns.countplot(x='satisfaction', data=df)
plt.title("Distribution of Passenger Satisfaction")
plt.xlabel("Satisfaction Level")
plt.ylabel("Count")
plt.show()

# Print value counts
print(df['satisfaction'].value_counts())
# Select numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
correlation = numeric_df.corr()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

sns.countplot(x='Gender', hue='satisfaction', data=df)
plt.title("Satisfaction by Gender")
plt.show()
sns.countplot(x='Type of Travel', hue='satisfaction', data=df)
plt.title("Satisfaction by Type of Travel")
plt.show()
sns.countplot(x='Class', hue='satisfaction', data=df)
plt.title("Satisfaction by Class")
plt.show()

sns.boxplot(x='satisfaction', y='Inflight wifi service', data=df)
plt.title("Inflight WiFi Rating by Satisfaction")
plt.show()

