import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# Load the datasets into pandas DataFrames
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


print("Train DataFrame head:")
display(train_df.head())

print("\nTest DataFrame head:")
display(test_df.head())

print("\nTrain DataFrame info:")
train_df.info()

print("\nTest DataFrame info:")
test_df.info()

print("\nTrain DataFrame describe:")
display(train_df.describe())

print("\nTest DataFrame describe:")
display(test_df.describe())


print("\nMissing values in Train DataFrame:")
display(train_df.isnull().sum())

print("\nMissing values in Test DataFrame:")
display(test_df.isnull().sum())

print("\nDistribution of 'Personality' in Train DataFrame:")
personality_counts = train_df['Personality'].value_counts()
display(personality_counts)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.barplot(x=personality_counts.index, y=personality_counts.values)
plt.title('Distribution of Personality')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.show()


numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Exclude 'id' and 'Personality' from numerical columns for plotting
numerical_cols.remove('id')
if 'Personality' in numerical_cols:
    numerical_cols.remove('Personality')

n_cols = 3
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

plt.figure(figsize=(15, n_rows * 5))

for i, col in enumerate(numerical_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(train_df[col].dropna(), kde=True, color='skyblue', label='Train')
    sns.histplot(test_df[col].dropna(), kde=True, color='lightcoral', label='Test')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()


categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
# Exclude 'Personality' as it's the target variable and already analyzed
if 'Personality' in categorical_cols:
    categorical_cols.remove('Personality')

for col in categorical_cols:
    print(f"\nUnique values and counts for '{col}' in Train DataFrame:")
    display(train_df[col].value_counts())

    print(f"\nUnique values and counts for '{col}' in Test DataFrame:")
    display(test_df[col].value_counts())



# Handle missing values: fill numerical with median, categorical with mode
for col in train_df.columns:
    if train_df[col].dtype in ['float64', 'int64']:
        train_df[col].fillna(train_df[col].median(), inplace=True)
    elif train_df[col].dtype == 'object' and col != 'Personality':
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)

for col in test_df.columns:
    if test_df[col].dtype in ['float64', 'int64']:
        test_df[col].fillna(test_df[col].median(), inplace=True)
    elif test_df[col].dtype == 'object':
        test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# Identify categorical columns for encoding
categorical_cols_to_encode = ['Stage_fear', 'Drained_after_socializing']

# One-hot encode categorical features
train_df_encoded = pd.get_dummies(train_df, columns=categorical_cols_to_encode, drop_first=True)
test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols_to_encode, drop_first=True)

# Separate target variable and encode
X_train = train_df_encoded.drop(['id', 'Personality'], axis=1)
y_train = train_df_encoded['Personality'].map({'Introvert': 0, 'Extrovert': 1})
X_test = test_df_encoded.drop('id', axis=1)

# Align columns after one-hot encoding
# This is crucial to ensure both dataframes have the same columns before training
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols]

print("Missing values in train_df after handling:")
display(X_train.isnull().sum().sum())
print("\nMissing values in test_df after handling:")
display(X_test.isnull().sum().sum())
print("\nShape of X_train after preprocessing:")
display(X_train.shape)
print("\nShape of X_test after preprocessing:")
display(X_test.shape)
print("\nShape of y_train after preprocessing:")
display(y_train.shape)


X_train['Time_Social_Interaction'] = X_train['Time_spent_Alone'] * X_train['Social_event_attendance']
X_test['Time_Social_Interaction'] = X_test['Time_spent_Alone'] * X_test['Social_event_attendance']

X_train['Social_Go_Outside_Interaction'] = X_train['Social_event_attendance'] * X_train['Going_outside']
X_test['Social_Go_Outside_Interaction'] = X_test['Social_event_attendance'] * X_test['Going_outside']

X_train['Friends_Post_Interaction'] = X_train['Friends_circle_size'] * X_train['Post_frequency']
X_test['Friends_Post_Interaction'] = X_test['Friends_circle_size'] * X_test['Post_frequency']

print("Shape of X_train after adding new features:")
display(X_train.shape)
print("\nShape of X_test after adding new features:")
display(X_test.shape)


# Instantiate the model
model = XGBClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)


# Make predictions on the training data
y_train_pred = model.predict(X_train)

# Calculate evaluation metrics
accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)

# Print the evaluation metrics
print(f"Accuracy on training data: {accuracy:.4f}")
print(f"Precision on training data: {precision:.4f}")
print(f"Recall on training data: {recall:.4f}")
print(f"F1-score on training data: {f1:.4f}")


print("""
Discussion on training data evaluation:

The evaluation metrics on the training data (Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}) show a very high level of performance. While these metrics indicate that the model fits the training data well, they do not provide a reliable measure of how the model will perform on unseen data. Evaluating the model solely on the training data can lead to an overestimation of its generalization ability, especially if the model has overfitted to the training set.

To get a more accurate assessment of the model's performance on new data and to detect potential overfitting, it is essential to evaluate the model on a separate validation set or use cross-validation techniques. This will provide a more realistic estimate of how well the model is likely to perform in a real-world scenario or on the test data provided for the competition.
""".format(accuracy=accuracy, precision=precision, recall=recall, f1=f1))


# Make predictions on the test data
test_predictions_numeric = model.predict(X_test)

# Convert numerical predictions back to original categorical labels
# 0 -> Introvert, 1 -> Extrovert
test_predictions_categorical = ['Extrovert' if pred == 1 else 'Introvert' for pred in test_predictions_numeric]

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_categorical})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")


# Create a countplot of the predicted 'Personality' distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=submission_df, palette='viridis')
plt.title('Distribution of Predicted Personality Types in Submission File')
plt.xlabel('Predicted Personality Type')
plt.ylabel('Count')
plt.show()


cm = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Introvert', 'Extrovert'], yticklabels=['Introvert', 'Extrovert'])
plt.title('Confusion Matrix for Training Data Predictions')
plt.xlabel('Predicted Personality Type')
plt.ylabel('Actual Personality Type')
plt.show()

