import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


DATA_PATH = "/kaggle/input/playground-series-s5e7/"


train_df = pd.read_csv(DATA_PATH + "train.csv")
test_df = pd.read_csv(DATA_PATH + "test.csv")


print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# Glimpse of data
train_df.head()


# Null value check
train_df.isnull().sum()


# Target distribution
sns.countplot(x='Personality', data=train_df)
plt.title("Target Class Distribution")
plt.show()


# Pairplot for numeric features vs target
sns.pairplot(train_df, hue='Personality')


# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Numerical Feature Correlation")
plt.show()


X = train_df.drop("Personality", axis=1)
y = train_df["Personality"]

test_ids = test_df['id']
X = X.drop('id', axis=1)
test_df_processed = test_df.drop('id', axis=1)


categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

print(f"\nCategorical Features identified: {list(categorical_features)}")
print(f"Numerical Features identified: {list(numerical_features)}")


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('passthrough', 'passthrough') 
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nTarget classes mapping: {list(le.classes_)}")


model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])


# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("\nTraining model...")
# Fit the pipeline (it will first preprocess X_train, including imputation, then train the classifier)
model_pipeline.fit(X_train, y_train)
print("Model training complete.")


y_val_pred = model_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {accuracy:.4f}")


print("\nMaking predictions on the test set...")
test_predictions_encoded = model_pipeline.predict(test_df_processed)

# Convert numerical predictions back to original labels
test_predictions_labels = le.inverse_transform(test_predictions_encoded)
print("Predictions made.")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

# Plot
plt.figure(figsize=(6, 4))
disp.plot(cmap='Blues')
plt.title("ğŸ”� Confusion Matrix on Validation Set")
plt.show()


# Convert back to class labels
val_preds_labels = le.inverse_transform(y_val_pred)
true_labels = le.inverse_transform(y_val)

# Compare prediction vs actual
plt.figure(figsize=(10, 5))

# Actual
plt.subplot(1, 2, 1)
sns.countplot(x=true_labels)
plt.title("Actual Personality Distribution")

# Predicted
plt.subplot(1, 2, 2)
sns.countplot(x=val_preds_labels)
plt.title("Predicted Personality Distribution")

plt.tight_layout()
plt.show()


submission_df = pd.DataFrame({'id': test_ids, 'Personality': test_predictions_labels})
submission_df.to_csv('Introsubmission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())

