import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-learn modules for model training and evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

# Configure visualization styles for academic presentation
%matplotlib inline
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Suppress warnings for clarity
import warnings
warnings.filterwarnings('ignore')

# Define file paths (specific to the Kaggle environment)
train_path = '/kaggle/input/playground-series-s5e3/train.csv'
test_path = '/kaggle/input/playground-series-s5e3/test.csv'
sample_submission_path = '/kaggle/input/playground-series-s5e3/sample_submission.csv'

# Load the datasets into DataFrames
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

# Display the first few records of the training dataset
print("Sample of Training Data:")
display(train_df.head())

# Print detailed information about the training dataset, including data types and non-null counts
print("Training Data Information:")
train_df.info()

# Assess missing values in the training dataset
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

# Generate descriptive statistics to summarize the central tendency, dispersion, and shape of the dataset
print("\nDescriptive Statistics:")
display(train_df.describe())


# Plot the distribution of the binary target variable 'rainfall'
plt.figure(figsize=(6, 4))
sns.countplot(x='rainfall', data=train_df, palette="viridis")
plt.title("Distribution of Rainfall (Target Variable)")
plt.xlabel("Rainfall (0 = No, 1 = Yes)")
plt.ylabel("Frequency")
plt.show()


# Identify numeric features (excluding 'rainfall' and 'id')
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('rainfall')
numeric_features.remove('id')

# Use a dynamic layout for the histograms to avoid layout errors
n_features = len(numeric_features)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

train_df[numeric_features].hist(bins=20, figsize=(15, 10), layout=(n_rows, n_cols), color='steelblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Create boxplots for each numeric feature to visually identify outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_features):
    plt.subplot(n_rows, n_cols, i+1)
    sns.boxplot(y=train_df[col], color="lightgreen")
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()


# Generate a correlation matrix heatmap to understand inter-feature relationships
plt.figure(figsize=(12, 10))
corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix Heatmap")
plt.show()


# Use pair plots to visualize pairwise relationships between features, colored by the target variable
sns.pairplot(train_df[numeric_features + ['rainfall']], hue='rainfall', palette="Set2", diag_kind='kde')
plt.suptitle("Pair Plot of Features", y=1.02)
plt.show()


# Impute missing values with the median value for each column in both training and test datasets
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

# Define the feature set X and the target variable y from the training data
X = train_df.drop(['rainfall', 'id'], axis=1)
y = train_df['rainfall']

# For the test data, preserve the 'id' column for final submission and remove it from the feature set
test_ids = test_df['id']
X_test = test_df.drop(['id'], axis=1)

# Apply standardization to the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# Split the data into training and validation subsets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model with an increased iteration limit to ensure convergence
log_reg = LogisticRegression(random_state=42, max_iter=1000)

# Train the logistic regression model on the training data
log_reg.fit(X_train, y_train)

# Predict the probabilities for the validation set
y_val_probs = log_reg.predict_proba(X_val)[:, 1]

# Compute the ROC AUC score for logistic regression on the validation set
roc_auc_lr = roc_auc_score(y_val, y_val_probs)
print("Logistic Regression - Validation ROC AUC: {:.4f}".format(roc_auc_lr))

# Perform 5-fold cross-validation on the full training set to validate the model's robustness
cv_scores_lr = cross_val_score(log_reg, X_scaled, y, cv=5, scoring='roc_auc')
print("Logistic Regression - Cross-Validated ROC AUC: {:.4f} ± {:.4f}".format(cv_scores_lr.mean(), cv_scores_lr.std()))


# Plot the ROC curve for logistic regression
fpr_lr, tpr_lr, _ = roc_curve(y_val, y_val_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_lr:.4f})", lw=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.show()


# Initialize and train the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10)
rf_model.fit(X_train, y_train)

# Predict probabilities on the validation set using the Random Forest model
y_val_probs_rf = rf_model.predict_proba(X_val)[:, 1]

# Compute the ROC AUC score for the Random Forest model
roc_auc_rf = roc_auc_score(y_val, y_val_probs_rf)
print("Random Forest - Validation ROC AUC: {:.4f}".format(roc_auc_rf))

# Perform 5-fold cross-validation on the Random Forest model
cv_scores_rf = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='roc_auc')
print("Random Forest - Cross-Validated ROC AUC: {:.4f} ± {:.4f}".format(cv_scores_rf.mean(), cv_scores_rf.std()))


# Plot the ROC curve for the Random Forest model
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_val_probs_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.4f})", lw=2, color="darkorange")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.show()


# Visualize feature importances as determined by the Random Forest model
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
plt.title("Feature Importances from Random Forest")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.show()


# Plot both ROC curves on a single figure for direct comparison
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_lr:.4f})", lw=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.4f})", lw=2, color="darkorange")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.show()


# Retrain the selected Random Forest model on the complete training dataset
rf_model.fit(X_scaled, y)

# Generate predicted probabilities for the test dataset
test_pred_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

# Create the submission DataFrame in the required format
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': test_pred_probs
})

# Save the submission file in CSV format
submission.to_csv('submission.csv', index=False)
print("Final submission file 'submission.csv' has been successfully created.")


from sklearn.calibration import calibration_curve

# Calculate calibration curve for logistic regression on the validation set
prob_true_lr, prob_pred_lr = calibration_curve(y_val, y_val_probs, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred_lr, prob_true_lr, marker='o', linewidth=1, label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve - Logistic Regression")
plt.legend()
plt.show()


# Generate confusion matrix and classification report for the Random Forest model
y_val_pred_rf = rf_model.predict(X_val)
cm = confusion_matrix(y_val, y_val_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("Classification Report - Random Forest:")
print(classification_report(y_val, y_val_pred_rf))

