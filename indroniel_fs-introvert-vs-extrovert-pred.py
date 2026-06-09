# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the datasets
# Make sure to upload 'train_introvert.csv' and 'test_introvert.csv' to your Colab environment
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Display the first few rows of the training data
print("Train Data Head:")
print(train_df.head())

# Display information about the training data
print("\nTrain Data Info:")
print(train_df.info())

# Check for missing values in the training data
print("\nMissing values in Train Data:")
print(train_df.isnull().sum())

# Display descriptive statistics for numerical columns in the training data
print("\nDescriptive Statistics for Train Data:")
print(train_df.describe())

# Display the first few rows of the test data
print("\nTest Data Head:")
print(test_df.head())

# Display information about the test data
print("\nTest Data Info:")
print(test_df.info())

# Check for missing values in the test data
print("\nMissing values in Test Data:")
print(test_df.isnull().sum())

# Display descriptive statistics for numerical columns in the test data
print("\nDescriptive Statistics for Test Data:")
print(test_df.describe())


# Set up the style for plots
sns.set_style("whitegrid")

# Plot the distribution of the target variable 'Personality'
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Personality')
plt.title('Distribution of Personality Types in Training Data')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.show()

# Plot the distribution of categorical features
categorical_cols_eda = ['Stage_fear', 'Drained_after_socializing']

plt.figure(figsize=(14, 6))
for i, col in enumerate(categorical_cols_eda):
    plt.subplot(1, 2, i + 1)
    sns.countplot(data=train_df, x=col, hue='Personality', palette='viridis')
    plt.title(f'Distribution of {col} by Personality')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the distribution of numerical features
numerical_cols_eda = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
plt.figure(figsize=(18, 12))
for i, col in enumerate(numerical_cols_eda):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data=train_df, x=col, kde=True, hue='Personality', palette='coolwarm')
    plt.title(f'Distribution of {col} by Personality')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Separate target variable from features
X = train_df.drop('Personality', axis=1)
y = train_df['Personality']

# Identify numerical and categorical columns
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_features = ['Stage_fear', 'Drained_after_socializing']

# Preprocessing pipelines for numerical and categorical features
# Numerical pipeline: Impute missing values with median, then scale using StandardScaler
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute missing values with the most frequent value (mode), then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
# This allows applying different transformers to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep 'id' column as is, as it's not a feature for prediction
)

# Encode the target variable 'Personality' from text ('Introvert', 'Extrovert') to numerical (0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Check the mapping of encoded labels to understand which number corresponds to which personality type
print(f"Personality mapping: {list(label_encoder.classes_)}")


# Define various classification models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'), # Good baseline model
    'Random Forest': RandomForestClassifier(random_state=42), # Ensemble method, generally robust
    'Gradient Boosting': GradientBoostingClassifier(random_state=42) # Another powerful ensemble method
}

# Create a pipeline for each model
# Each pipeline first preprocesses the data and then applies the classifier
pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# Train and evaluate models using cross-validation
# Cross-validation helps in getting a more reliable estimate of model performance
results = {}
for name, pipeline in pipelines.items():
    print(f"\n--- Training {name} ---")
    # Perform 5-fold cross-validation, setting n_jobs=1 to avoid pickling issues
    scores = cross_val_score(pipeline, X, y_encoded, cv=5, scoring='accuracy', n_jobs=1) # Changed n_jobs from -1 to 1
    results[name] = scores
    print(f"{name} Cross-validation Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# Select the best performing model based on cross-validation results
# For this example, we'll assume Random Forest is chosen due to its general good performance.
# In a real scenario, you would choose based on the 'np.mean(scores)' values.
best_model_name = 'Random Forest'
best_pipeline = pipelines[best_model_name]

# Train the best model on the full training data
# This step is crucial before making predictions on unseen test data
best_pipeline.fit(X, y_encoded)
print(f"\n{best_model_name} trained on full training data.")

# Evaluate the model on the training set (for sanity check)
# This gives an idea of how well the model learned the training data.
# Note: This is not the true generalization performance, which is better estimated by cross-validation.
y_train_pred = best_pipeline.predict(X)
print(f"\n{best_model_name} Training Accuracy: {accuracy_score(y_encoded, y_train_pred):.4f}")
print(f"\nClassification Report for {best_model_name} (Training Data):\n")
print(classification_report(y_encoded, y_train_pred, target_names=label_encoder.classes_))


# Example using GridSearchCV for Random Forest
# Uncomment and run this section if you want to optimize the chosen model's parameters.

param_grid = {
    'classifier__n_estimators': [100, 200, 300], # Number of trees in the forest
    'classifier__max_features': ['sqrt', 'log2'], # Number of features to consider when looking for the best split
    'classifier__max_depth': [4, 6, 8, None] # Maximum depth of the tree (None means unlimited)
}
grid_search = GridSearchCV(pipelines['Random Forest'], param_grid, cv=3, n_jobs=1, verbose=2, scoring='accuracy') # Changed n_jobs to 1
grid_search.fit(X, y_encoded)

print(f"\nBest parameters for Random Forest: {grid_search.best_params_}")
print(f"Best cross-validation score for Random Forest: {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_ # Update best_pipeline with the tuned model


# Make predictions on the test data
test_predictions_encoded = best_pipeline.predict(test_df)

# Decode the predictions back to original labels
test_predictions_personality = label_encoder.inverse_transform(test_predictions_encoded)

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_predictions_personality
})

# Display the first few rows of the submission file
print("\nPrediction Output (first 10 rows):")
print(submission_df.head(10))

# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)
print("\nPredictions saved to 'submission.csv'")

print("\n--- End of Notebook ---")

