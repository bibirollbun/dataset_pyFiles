# Install TabPFN
!pip install tabpfn 
!pip install tabpfn-extensions


# Kaggle Rainfall Prediction - Playground Series S5E3 using TabPFNClassifier with Feature Engineering & k-Fold CV
# Author: Aaron Isom

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score,  train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import seaborn as sns

# Load Data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')

# Display first few rows
display(train_df.head(10))
display("Train Shape", train_df.shape)
display("\nTest Shape", test_df.shape)

# Describe the data
display(train_df.describe())

# Display information about dtypes and missing values
display("Train Data Info:", train_df.info())

# Check target distribution
display("Target Distribution:", train_df['rainfall'].value_counts(normalize=True))



# Feature Engineering Function
# Depracted since adding new features adds no value to the model performance.

def feature_engineering(df):
    df = df.copy()

    # Interaction Features
    df['cloud_humidity'] = df.cloud + df.humidity
    df['cloud_humidity_sunshine'] = df.cloud + df.humidity + df.sunshine
    df['cloud_sunshine'] = df.cloud * df.sunshine
    df['humidity_sunshine'] = df.humidity * df.sunshine
    df['temp_diff'] = df['maxtemp'] - df['mintemp']

    # Time-Based Features (Lag & Differences) - Only apply if column exists
    time_features = ['pressure', 'maxtemp', 'mintemp', 'humidity']  # Removed 'temperature'
    for c in time_features:
        if c in df.columns:  # Check if column exists
            for gap in [1]:
                df[c + f"_shift{gap}"] = df[c].shift(gap)
                df[c + f"_diff{gap}"] = df[c].diff(gap)

    # Additional Features - Apply only if relevant columns exist
    if 'pressure' in df.columns and 'humidity' in df.columns:
        df['humidity_pressure_ratio'] = df.humidity / (df.pressure + 1e-6)  # Prevent division by zero
    if 'maxtemp' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df.maxtemp * df.humidity
    if 'cloud' in df.columns and 'maxtemp' in df.columns:
        df['cloud_temp_interaction'] = df.cloud * df.maxtemp
    if 'maxtemp' in df.columns and 'sunshine' in df.columns:
        df['temp_sunshine_ratio'] = df.maxtemp / (df.sunshine + 1e-6)

    return df

# Apply feature engineering
# train_df = feature_engineering(train_df)
# test_df = feature_engineering(test_df)

# Fill NaNs caused by shifting operations
# train_df.fillna(train_df.median(), inplace=True)
# test_df.fillna(test_df.median(), inplace=True)


# Separate features and target
X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']

# Standardize numerical features (important for TabPFN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_df)
display(X_scaled)

# Split the train/test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# Check class counts
class_counts = y.value_counts()
print("\nClass Distribution:\n", class_counts)

# Compute imbalance ratio
imbalance_ratio = class_counts.min() / class_counts.max()
print(f"\nImbalance Ratio: {imbalance_ratio:.4f}")  # Closer to 0 means more imbalance

# Plot imbalances
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.xlabel("Class Labels")
plt.ylabel("Frequency")
plt.title("Class Distribution in Rainfall Training Data")
plt.xticks([0, 1], labels=["No Rain", "Rain"])  # Adjust labels if needed
plt.show()


# Try basline AutoTabPFN for comparison
clf = AutoTabPFNClassifier(device='auto', max_time=60)
clf.fit(X_train, y_train)

preds = clf.predict_proba(X_test)

print('ROC AUC: ',  roc_auc_score(y_test, preds[:,1]))


# Train on full dataset and predict on test set.
clf.fit(X, y)
test_preds = clf.predict_proba(X_test_scaled)[:, 1]

submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = test_preds
submission.to_csv('submission.csv', index=False) #Prepare submission file
print("Submission file saved!")
display(submission)


# Cross-validation KFold=5
# kf = KFold(5)
# cv1_accuracies = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
#     print(f"\nTraining Fold {fold + 1}...")

#     X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     # Initialize and train TabPFN Classifier
#     clf1 = TabPFNClassifier(random_state=42)
#     clf1.fit(X_train, y_train)

#     # Make predictions
#     y_val_pred = clf1.predict(X_val)
#     accuracy = accuracy_score(y_val, y_val_pred)
#     cv1_accuracies.append(accuracy)

#     print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

# # Display average cross-validation accuracy
# mean_cv_accuracy1 = np.mean(cv1_accuracies)
# print(f"\nKFold Mean Cross-Validation Accuracy: {mean_cv_accuracy1:.4f}")

# # StratifiedKFold
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv2_accuracies = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     # Train TabPFN
#     clf2 = TabPFNClassifier(device="cpu")  # Adjust to "cuda" if using GPU
#     clf2.fit(X_train, y_train)

#     # Evaluate
#     y_val_pred = clf2.predict(X_val)
#     accuracy = accuracy_score(y_val, y_val_pred)
#     cv2_accuracies.append(accuracy)
    
#     print(f"Fold {fold + 1} Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

# # Display average cross-validation accuracy
# mean_cv_accuracy2 = np.mean(cv2_accuracies)
# print(f"\nStratifiedKFold Mean Cross-Validation Accuracy: {mean_cv_accuracy2:.4f}")


# Remove old file(s)
# import os
# os.remove('/kaggle/working/submission.csv')

