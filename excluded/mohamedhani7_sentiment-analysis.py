import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import json
import pandas as pd

# Load training data
train_data = []
with open('/kaggle/input/depi-r-2-emotion-analysis/train.json', 'r') as f:
    for line in f:
        train_data.append(json.loads(line))

# Load validation data
val_data = []
with open('/kaggle/input/depi-r-2-emotion-analysis/validation.json', 'r') as f:
    for line in f:
        val_data.append(json.loads(line))

# Load test data
test_data = []
with open('/kaggle/input/depi-r-2-emotion-analysis/test.json', 'r') as f:
    for line in f:
        test_data.append(json.loads(line))

# Convert to pandas DataFrame
train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)
test_df = pd.DataFrame(test_data)

# Display the first few rows of the training data
print(train_df.head())


print("Training Data Shape:", train_df.shape)
print("Validation Data Shape:", val_df.shape)
print("Test Data Shape:", test_df.shape)

print("\nTraining Data Columns:", train_df.columns)
print("\nSample Training Data:\n", train_df.head())


import pandas as pd

# Load JSON Lines files
train_df = pd.read_json('/kaggle/input/depi-r-2-emotion-analysis/train.json', lines=True)
val_df = pd.read_json('/kaggle/input/depi-r-2-emotion-analysis/validation.json', lines=True)
test_df = pd.read_json('/kaggle/input/depi-r-2-emotion-analysis/test.json', lines=True)

# Display the first few rows of the training data
print(train_df.head())


import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply cleaning to all datasets
train_df['text'] = train_df['text'].apply(clean_text)
val_df['text'] = val_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train = vectorizer.fit_transform(train_df['text'])

# Transform validation and test data
X_val = vectorizer.transform(val_df['text'])
X_test = vectorizer.transform(test_df['text'])

# Labels
y_train = train_df['label']
y_val = val_df['label']


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))


from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))


test_predictions = model.predict(X_test)  




# Assuming test_predictions contains your model's predictions
submission_df = pd.DataFrame({
    'ID': test_df['id'],  # Use the correct column name for IDs
    'Label': test_predictions  # Your model's predictions
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

