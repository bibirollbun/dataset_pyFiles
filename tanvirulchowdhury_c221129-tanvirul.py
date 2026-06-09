import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Encode categorical variables and save encoders
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Define features and target variable
# Drop 'Unnamed: 0' and 'id' if present, adapt as needed
drop_cols = ['Unnamed: 0', 'id']
for col in drop_cols:
    if col in train_data.columns:
        train_data.drop(columns=[col], inplace=True)

X = train_data.drop(columns=['satisfaction'])
y = train_data['satisfaction']

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='median')  # median strategy to match your previous fillna median
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(classification_report(y_val, y_pred))

# Load the test dataset
test_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")

# Encode categorical variables in test data using saved label encoders
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in test_data.columns and col in label_encoders:
        test_data[col] = label_encoders[col].transform(test_data[col])

# Keep IDs for submission, remove unnecessary columns before prediction
ids = test_data['id'] if 'id' in test_data.columns else test_data.index
drop_cols_test = ['Unnamed: 0', 'id']
for col in drop_cols_test:
    if col in test_data.columns:
        test_data.drop(columns=[col], inplace=True)

# Handle missing values in test data
X_test = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

# Make predictions
test_pred = model.predict(X_test)

# Convert predictions back to original labels
test_pred_labels = label_encoders['satisfaction'].inverse_transform(test_pred)

# Prepare submission dataframe
submission = pd.DataFrame({
    'ID': ids,
    'satisfaction': test_pred_labels
})

# Save submission file
submission.to_csv("submission.csv", index=False)

