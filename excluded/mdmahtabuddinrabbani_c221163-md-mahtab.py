import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Define features and target
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'], errors='ignore')
y = train_data['satisfaction']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.2f}")

# Load the test dataset
solution = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Encode categorical variables in test data
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        solution[col] = label_encoders[col].transform(solution[col])

# Prepare test features
X_test = solution.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Predict satisfaction
solution['satisfaction'] = model.predict(X_test)

# Convert predictions back to original labels
solution['satisfaction'] = label_encoders['satisfaction'].inverse_transform(solution['satisfaction'])

# Save or display final solution
solution.to_csv("submission.csv", index=False)


# Rename the 'id' column to 'ID' and save the predictions to Submission.csv
solution.rename(columns={'id': 'ID'}, inplace=True)
solution[['ID', 'satisfaction']].to_csv("submission.csv", index=False)


solution.head(15)

