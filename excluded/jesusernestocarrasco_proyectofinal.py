import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

print("Libraries loaded successfully")


!pip install seaborn matplotlib


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Display the shapes
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Checking the first 5 rows
display(train_df.head())


print("--- Info ---")
train_df.info()

print("\n--- Missing Values ---")
print(train_df.isnull().sum())


# Combine train and test to ensure the LabelEncoder sees all possible categories
target = 'Personality'
all_data = pd.concat([train_df.drop(columns=[target]), test_df], axis=0)

# Identify text columns
cat_cols = all_data.select_dtypes(include=['object']).columns
print(f"Categorical columns found: {list(cat_cols)}")

# Initialize Encoder
le = LabelEncoder()

# Apply encoding
for col in cat_cols:
    # Handle missing values first (fill with 'Unknown')
    train_df[col] = train_df[col].fillna('Unknown')
    test_df[col] = test_df[col].fillna('Unknown')

    # Fit encoder on all data combined
    le.fit(pd.concat([train_df[col], test_df[col]]))

    # Transform
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Fill numeric missing values with the mean
num_cols = train_df.select_dtypes(include=['number']).columns
train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].mean())
test_df[num_cols] = test_df[num_cols].fillna(test_df[num_cols].mean())

print("Preprocessing complete. Data is now numeric and tiddy")
display(train_df.head())


# Define Features and Targets
X = train_df.drop(columns=[target, 'id'])
y = train_df[target]
X_test = test_df.drop(columns=['id'])

# Split 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {X_train.shape[0]} rows, Validating on {X_val.shape[0]} rows.")


# Initialize and train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training complete.")


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Get Feature Importances
importances = model.feature_importances_
feature_names = X.columns

# 2. Create a DataFrame to organize them
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# 3. Sort by importance (highest on top)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 4. Plot
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15)) # Top 15 features
plt.title('Top 15 Features determining Introvert vs Extrovert')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.show()


val_preds = model.predict(X_val)
score = accuracy_score(y_val, val_preds)

print(f"Validation Accuracy: {score:.4}")


# Predict on the competition set data
final_preds = model.predict(X_test)

# Create DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': final_preds
})

# Quick check of the submission format
print(submission.head())

# Save
submission.to_csv('submission.csv', index=False)
print("submission.csv saved!")




