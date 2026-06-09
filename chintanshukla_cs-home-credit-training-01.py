# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


df_application_train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")


# Check the shape of the dataset
print("DataFrame Shape:", df_application_train.shape)


df_application_train.head()


df_application_train.info()


df_application_train.describe()


df_application_train.describe(include="object")


# Replace infinite values with NaN before handling missing values
df_application_train.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Infinite values replaced with NaN.")


df_application_train.head()


categorical_candidates = df_application_train.select_dtypes(include=['object']).columns.tolist()
print(categorical_candidates)


# Step 2: Identify numeric columns with low unique values (possible categorical)
low_unique_counts = df_application_train.nunique()
numeric_categoricals = low_unique_counts[
    (low_unique_counts < 20) & (df_application_train.dtypes != 'object')].index.tolist()
print(numeric_categoricals)


# Step 3: Combine all categorical columns
final_categorical_columns = categorical_candidates + numeric_categoricals
print(final_categorical_columns)


# Step 4: Convert detected columns to 'category' dtype
if final_categorical_columns:
    df_application_train[final_categorical_columns] = df_application_train[final_categorical_columns].astype("category")
    print(f"Converted categorical columns: {final_categorical_columns}")
else:
    print("No categorical columns detected in this dataset. No dtype conversion needed.")


# Step 5: Manual Verification
print("\nChecking final dtype distribution:")
print(df_application_train.dtypes.value_counts())


from collections import Counter

# Normalize dtype names to strings
dtype_names = df_application_train.dtypes.apply(lambda x: str(x))
print(dtype_names.value_counts())


# Check for missing values
pd.set_option('display.max_rows', None) 
print("Missing values in each column:")
print(df_application_train.isnull().sum())
pd.reset_option('display.max_rows')  


# Define missing value thresholds
low_threshold = 1    # Less than 1% missing
moderate_threshold = 20  # Between 1% and 20% missing
high_threshold = 50   # More than 50% missing 

# Calculate missing value percentage for df_application_train
missing_percent = (df_application_train.isnull().sum() / len(df_application_train)) * 100  

# Display missing percentages sorted from highest to lowest
print("Missing Value Percentages:")
display(missing_percent[missing_percent > 0].sort_values(ascending=False).apply(lambda x: f"{x:.2f}%")) 


missing_percent[missing_percent > 0].sort_values(ascending=False).plot(kind='bar', figsize=(12,6))
plt.title("Missing Value Percentage by Column")
plt.ylabel("Percentage")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Identify columns to drop
columns_to_drop = missing_percent[missing_percent > high_threshold].index

# Drop columns
df_application_train.drop(columns=columns_to_drop, inplace=True)

# Print removed columns
print(f"Dropped {len(columns_to_drop)} columns with more than 50% missing values.")
print("Dropped columns:", list(columns_to_drop))


# Fill numeric columns with median
numeric_cols = df_application_train.select_dtypes(include=['int64', 'float64']).columns
df_application_train[numeric_cols] = df_application_train[numeric_cols].fillna(df_application_train[numeric_cols].median())

print("Filled numeric missing values with median.")


# Fill categorical columns with mode
categorical_cols = df_application_train.select_dtypes(include=['category']).columns
for col in categorical_cols:
    df_application_train[col] = df_application_train[col].fillna(df_application_train[col].mode()[0]) 

print("Filled categorical missing values with mode.")


print("Final Missing Values Check:")
print(df_application_train.isnull().sum().sum())


# Check for duplicates
print("Number of duplicate rows:", df_application_train.duplicated().sum())


# Display all columns and their data types
pd.set_option('display.max_rows', None) 
print("Updated Data Types for df_application_train:")
print(df_application_train.dtypes)
pd.reset_option('display.max_rows')  


df_application_train.to_csv("cleaned_application_train.csv")
df_application_train.to_pickle("cleaned_application_train.pkl")

print("Cleaned Application Train dataset saved successfully")


df_application_train['TARGET_numeric'] = df_application_train['TARGET'].astype(int)


for col in df_application_train.select_dtypes(include='category').columns:
    df_application_train[col] = pd.factorize(df_application_train[col])[0]


numeric_data = df_application_train.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()

# Correlation with TARGET
target_corr = corr_matrix['TARGET_numeric'].sort_values(ascending=False)
print("Top features positively correlated with TARGET:")
print(target_corr.head(10))

print("\nTop features negatively correlated with TARGET:")
print(target_corr.tail(10))


# Compute correlation matrix
corr_matrix = df_application_train.corr()

# Set up the plot
plt.figure(figsize=(16,12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)

# Customize labels and layout
plt.title('Correlation Matrix of Numeric Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



df_application_train.drop(columns='TARGET_numeric', inplace=True)


df_application_train.to_csv("application_train_processed.csv", index=False)
df_application_train.to_pickle("application_train_processed.pkl")

print("Application train dataset saved successfully")


df_application_train.info()


# Features and target
X = df_application_train.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df_application_train['TARGET']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# Initialize and train
# liblinear works well for binary classification
logreg = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')  
logreg.fit(X_train, y_train)


# Predict probabilities and classes
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

# Metrics
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Generate thresholds
thresholds = np.linspace(0, 1, 100)
accuracies = []

# Calculate accuracy for each threshold
for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    accuracies.append(acc)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, color='blue', linewidth=2)
plt.title('Accuracy vs Classification Threshold')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()



# Thresholds
thresholds = np.linspace(0.01, 0.99, 99)
precision_scores = []
recall_scores = []
f1_scores = []

# Calculate metrics for each threshold
for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    precision_scores.append(precision_score(y_test, y_pred_thresh, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred_thresh))
    f1_scores.append(f1_score(y_test, y_pred_thresh))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, label='Precision', color='blue')
plt.plot(thresholds, recall_scores, label='Recall', color='green')
plt.plot(thresholds, f1_scores, label='F1 Score', color='red')
plt.title('Precision, Recall, and F1 Score vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()



# Initialize with class_weight='balanced' to handle imbalance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)


# Predict
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Metrics
print("ROC AUC Score:", roc_auc_score(y_test, y_proba_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)
print("Top Features:\n", top_features)


# Use predicted probabilities from Random Forest
thresholds = np.linspace(0.01, 0.99, 99)
precision_scores = []
recall_scores = []
f1_scores = []

# Calculate metrics for each threshold
for thresh in thresholds:
    y_pred_thresh = (y_proba_rf >= thresh).astype(int)
    precision_scores.append(precision_score(y_test, y_pred_thresh, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred_thresh))
    f1_scores.append(f1_score(y_test, y_pred_thresh))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, label='Precision', color='blue')
plt.plot(thresholds, recall_scores, label='Recall', color='green')
plt.plot(thresholds, f1_scores, label='F1 Score', color='red')
plt.title('Precision, Recall, and F1 Score vs Threshold (Random Forest)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()


