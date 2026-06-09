import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from tqdm import tqdm  

# Set visualization style
sns.set_style("whitegrid")

# ---------------------------
# 1. Load & Preprocess Data
# ---------------------------

# Define file paths
TRAIN_FILES_PATTERN = "/kaggle/input/neo-bank-non-sub-churn-prediction/train_*.parquet"
TEST_FILE_PATH = "/kaggle/input/neo-bank-non-sub-churn-prediction/test.parquet"
SUBMISSION_SAMPLE_PATH = "/kaggle/input/neo-bank-non-sub-churn-prediction/sample_submission.csv"
 
# Output submission file names
SUBMISSION_FILE = "random_forest_submission.csv"
FINAL_SUBMISSION_FILE = "random_forest_with_postprocessing_submission.csv"

# Load all training data
train_files = glob.glob(TRAIN_FILES_PATTERN)
if not train_files:
    raise FileNotFoundError(f"No Parquet files found in path: {TRAIN_FILES_PATTERN}")

print(f"Found {len(train_files)} training files. Loading...")
data_frames = []
for file in tqdm(train_files, desc="Loading files"):
    data_frames.append(pd.read_parquet(file))

df = pd.concat(data_frames, ignore_index=True)

# Display basic dataset info
print("Training Data Info:")
df.info()

# Select relevant features for training
FEATURE_COLUMNS = [
    "bank_transfer_in", "bank_transfer_out", "crypto_in_volume", "crypto_out_volume", 
    "tenure", "complaints", "model_predicted_fraud"
]

# Check if 'churn_due_to_fraud' exists before using it
if "churn_due_to_fraud" in df.columns:
    df["churn"] = df["churn_due_to_fraud"].astype(int)  
else:
    raise KeyError("'churn_due_to_fraud' column not found in dataset.")

# Handle missing feature values by filling with 0
df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

# ---------------------------
# 2. Train/Test Split
# ---------------------------

X_train, X_val, y_train, y_val = train_test_split(
    df[FEATURE_COLUMNS], df["churn"], test_size=0.2, random_state=42, stratify=df["churn"]
)

# ---------------------------
# 3. Train Random Forest Model
# ---------------------------

model = RandomForestClassifier(
    n_estimators=200,  # Increase number of trees for stability
    max_depth=8,       # Increase depth for better feature interactions
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")
model.fit(X_train, y_train)

# ---------------------------
# 4. Evaluate Model Performance
# ---------------------------

y_val_pred_proba = model.predict_proba(X_val)[:, 1]  
y_val_pred = model.predict(X_val)  

logloss_score = log_loss(y_val, y_val_pred_proba)
roc_auc = roc_auc_score(y_val, y_val_pred_proba)
accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Log Loss: {logloss_score:.4f}")
print(f"Validation ROC-AUC: {roc_auc:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, y_val_pred_proba)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, color="blue", lw=2, label=f"Precision-Recall Curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
plt.show()

# Feature Importance Plot
feature_importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
feature_importances.sort_values(ascending=False).plot(kind="bar", title="Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# ---------------------------
# 5. Load Test Data & Generate Predictions
# ---------------------------

print("Loading test data...")
test = pd.read_parquet(TEST_FILE_PATH)

# Handle missing values
test[FEATURE_COLUMNS] = test[FEATURE_COLUMNS].fillna(0)

# Predict churn probabilities
print("Generating predictions on test data...")
test["churn"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

# ---------------------------
# 6. Rule-Based Post-Processing
# ---------------------------

if "churn_due_to_fraud" in test.columns:
    test.loc[test["churn_due_to_fraud"] == True, "churn"] = 1.0
    test.drop(columns=["churn_due_to_fraud"], inplace=True, errors="ignore")

# Load sample submission file
submission = pd.read_csv(SUBMISSION_SAMPLE_PATH)

# Ensure correct format
submission["churn"] = test["churn"]
submission.to_csv(FINAL_SUBMISSION_FILE, index=False)

print(f"Final submission saved: {FINAL_SUBMISSION_FILE}")
print(submission.head())

