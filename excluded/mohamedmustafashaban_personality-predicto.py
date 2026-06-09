import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# Load the datasets into pandas DataFrames
df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')



print("\nFirst 5 rows of train.csv:")
display(df_train.head())


print("\nFirst 5 rows of test.csv:")
display(df_test.head())


print("First 5 rows of sample_submission.csv:")
display(df_sample_submission.head())



# Print the statistical summary of the training data
print("Statistical summary of training data:")
display(df_train.describe())


# Print information about the training data (data types and non-null counts)
print("\nInformation about training data:")
df_train.info()


# Print the count of missing values for each column in the training data
print("\nMissing values in training data:")
display(df_train.isnull().sum())



# Print the statistical summary of the test data
print("\nStatistical summary of test data:")
display(df_test.describe())


# Print information about the test data (data types and non-null counts)
print("\nInformation about test data:")
df_test.info()



# Print the count of missing values for each column in the test data
print("\nMissing values in test data:")
display(df_test.isnull().sum())


# Print the data types of columns in the training data
print("Data types of columns in df_train:")
display(df_train.dtypes)


# Print the data types of columns in the training data
print("Data types of columns in df_train:")
display(df_train.dtypes)


# Print the number of unique values in each column of the training data
print("\nNumber of unique values in each column of df_train:")
display(df_train.nunique())



# Print the number of unique values in each column of the test data
print("\nNumber of unique values in each column of df_test:")
display(df_test.nunique())


# Print the unique values in categorical columns of the training data
print("\nUnique values in categorical columns of df_train:")
for col in df_train.select_dtypes(include='object').columns:
    print(f"\nUnique values in '{col}':")
    display(df_train[col].unique())


X = df_train.drop(['id', 'Personality'], axis=1).copy()
y = df_train['Personality'].copy()


numerical_features = X.select_dtypes(include=np.number).columns
imputer_numerical = SimpleImputer(strategy='mean')
X[numerical_features] = imputer_numerical.fit_transform(X[numerical_features])



categorical_features = X.select_dtypes(include='object').columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
X[categorical_features] = imputer_categorical.fit_transform(X[categorical_features])



X = pd.get_dummies(X, columns=categorical_features, drop_first=True)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)




label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_test = df_test.drop('id', axis=1).copy()


X_test[numerical_features] = imputer_numerical.transform(X_test[numerical_features])



X_test[categorical_features] = imputer_categorical.transform(X_test[categorical_features])



X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)



train_cols = X.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_test.drop(c, axis=1, inplace=True)


X_test = X_test[train_cols]


X_test_scaled = scaler.transform(X_test)


X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)



print("Processed Training Features (X_scaled_df):")
display(X_scaled_df.head())
print("\nInformation about Processed Training Features:")






X_scaled_df.info()
print("\nProcessed Test Features (X_test_scaled_df):")
display(X_test_scaled_df.head())
print("\nInformation about Processed Test Features:")
X_test_scaled_df.info()



print("\nEncoded Training Target (y_encoded):")
display(y_encoded[:5])


df_train_processed = X_scaled_df.copy()
df_train_processed['Personality'] = y


# 1. Bar plot for the distribution of 'Personality'
# Purpose: Visualize the class distribution of the target variable 'Personality' to check for imbalance after processing.
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=df_train_processed, palette='viridis')
plt.title('Distribution of Personality Types (Processed Data)', fontsize=14)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# 2. Distribution plot for a numerical variable (e.g., 'Time_spent_Alone')
# Purpose: Understand the distribution of a representative numerical feature after scaling.
plt.figure(figsize=(10, 6))
sns.histplot(df_train_processed['Time_spent_Alone'], kde=True, color='skyblue')
plt.title('Distribution of Time Spent Alone (Processed Data)', fontsize=14)
plt.xlabel('Time Spent Alone (Scaled)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()



# 3. Pair plot for a subset of numerical variables (after scaling)
# Purpose: Visualize pairwise relationships and distributions of a few key numerical variables.
# Selecting a subset for clarity as pair plot can be dense with many features.
subset_numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size']
plt.figure(figsize=(12, 10))
sns.pairplot(df_train_processed[subset_numerical_cols + ['Personality']], hue='Personality', diag_kind='kde', palette='viridis')
plt.suptitle('Pair Plot of Selected Numerical Variables by Personality (Processed Data)', y=1.02, fontsize=16)
plt.show()


# 4. Bar plots for encoded categorical columns (if any are still meaningful after one-hot encoding and dropping first)
# After one-hot encoding and dropping the first column, the resulting columns are binary (0 or 1).
# Visualizing their distribution might not be as insightful as the original categorical columns.
# However, we can check the distribution of the encoded 'Yes' categories.
encoded_categorical_cols = [col for col in df_train_processed.columns if col.endswith('_Yes')]
if encoded_categorical_cols:
    print("\nDistribution of encoded categorical features ('_Yes' columns):")
    for col in encoded_categorical_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, data=df_train_processed, palette='plasma')
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col.replace('_', ' '), fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Personality', y='Social_event_attendance', data=df_train_processed, palette='plasma')
plt.title('Social Event Attendance by Personality Type', fontsize=14)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Social Event Attendance (Scaled)', fontsize=12)
plt.show()



pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_df) 
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', alpha=0.6)
plt.title('PCA of Training Data Colored by Personality', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
handles, _ = scatter.legend_elements()
legend_labels = label_encoder.inverse_transform(np.unique(y_encoded))
plt.legend(handles, legend_labels, title="Personality")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Personality', y='Social_event_attendance', data=df_train_processed, palette='plasma')
plt.title('Social Event Attendance by Personality Type', fontsize=14)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Social Event Attendance (Scaled)', fontsize=12)
plt.show()
plt.show()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_encoded, test_size=0.25, random_state=42)



# Initialize and train the models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    try:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except ValueError:
        roc_auc = "Not defined for this model"

    results[name] = {
        "accuracy": accuracy,
        "classification_report": report,
        "roc_auc": roc_auc
    }
    print(f"Finished training and evaluating {name}.")



# Print the results for each model
print("\n--- Model Evaluation Results ---")
for name, metrics in results.items():
    print(f"\nModel: {name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}") # Format accuracy for better readability
    print(f"Classification Report:\n{metrics['classification_report']}")
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}" if isinstance(metrics['roc_auc'], float) else f"ROC AUC Score: {metrics['roc_auc']}")
    print("-" * 30)


best_model = models["Random Forest"]
test_predictions_encoded = best_model.predict(X_test_scaled_df)
test_predictions_original_labels = label_encoder.inverse_transform(test_predictions_encoded)
submission_df = pd.DataFrame({'id': df_test['id'], 'Personality': test_predictions_original_labels})
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
display(submission_df.head())




