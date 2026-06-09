# Core libraries
import os
import joblib
import pandas as pd

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sbn

# Scikit-learn utilities
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

# Create required directories
for folder in ["models", "outputs"]:
    os.makedirs(folder, exist_ok=True)


# Load the datasets
train_df = pd.read_csv("/kaggle/input/bank-customer-churn-prediction-challenge/train.csv")
test_df = pd.read_csv("/kaggle/input/bank-customer-churn-prediction-challenge/test.csv")

# Basic info about the data
print(f"Training set dimensions: {train_df.shape}")
print(f"Testing set dimensions:  {test_df.shape}\n")

# Peek at first few records
print("Preview of training data:")
display(train_df.head())

# Check for null values
print("\nCount of missing values per column:")
print(train_df.isna().sum())


# Encode categorical string features into numeric form
encoder = LabelEncoder()
categorical_cols = ["Gender", "Geography"]

for feature in categorical_cols:
    train_df[feature] = encoder.fit_transform(train_df[feature])
    test_df[feature] = encoder.transform(test_df[feature])


# Columns not needed for modeling
ignore_cols = ["id", "CustomerId", "Surname"]

# Split features and target
y_full = train_df["Exited"]
X_full = train_df.drop(ignore_cols + ["Exited"], axis=1)
X_eval = test_df.drop(ignore_cols, axis=1)

# Apply standard scaling
scaler_tool = StandardScaler()
X_full_scaled = scaler_tool.fit_transform(X_full)
X_eval_scaled = scaler_tool.transform(X_eval)

# Save the scaler for reuse
joblib.dump(scaler_tool, "models/std_scaler.pkl")


# Divide the dataset into training and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full_scaled, y_full, 
    test_size=0.2, 
    random_state=42,
    stratify=y_full
)


def fit_and_assess(clf, name):
    """Train a classifier, evaluate, save model & submission."""
    
    print(f"\n>>> Training: {name}")
    clf.fit(X_tr, y_tr)

    # Predictions on validation data
    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else val_preds

    # Evaluation metrics
    acc_val = accuracy_score(y_val, val_preds)
    f1_val = f1_score(y_val, val_preds)
    auc_val = roc_auc_score(y_val, val_probs)

    print(f"{name} | Acc: {acc_val:.4f} | F1: {f1_val:.4f} | AUC: {auc_val:.4f}")

    # Save trained model
    model_file = f"models/{name.replace(' ', '_')}.pkl"
    joblib.dump(clf, model_file)
    print(f"Model stored at {model_file}")

    # Predict on test set
    test_preds = clf.predict_proba(X_eval_scaled)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_eval_scaled)

    # Create submission file
    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "Exited": test_preds
    })
    output_file = f"outputs/{name.replace(' ', '_')}_submission.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"Submission ready at {output_file}")

    return {"model": name, "accuracy": acc_val, "f1": f1_val, "auc": auc_val}



# List of candidate models
candidate_models = [
    (RandomForestClassifier(n_estimators=100, random_state=10), "Random Forest"),
    (GradientBoostingClassifier(n_estimators=100, random_state=10), "Gradient Boosting"),
    (SVC(probability=True, random_state=10), "Support Vector Machine"),
    (LogisticRegression(max_iter=1000, solver="lbfgs"), "Logistic Regression")
]

# Collect evaluation metrics
eval_results = []
for clf, label in candidate_models:
    eval_results.append(fit_and_assess(clf, label))


# Convert evaluation outcomes into a DataFrame
metrics_df = pd.DataFrame(eval_results)

print("\n" + "="*80)
print("Model Performance Summary:")
print(metrics_df)

# Identify the top performer by AUC
top_model = metrics_df.sort_values("auc", ascending=False).iloc[0]
print("\n" + "="*80)
print("Best Model (by AUC):")
print(top_model)


# Map model names to their corresponding objects
model_dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=10),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=10),
    "Support Vector Machine": SVC(probability=True, random_state=10)
}

print("\n" + "="*80)
print(f"Performing cross-validation for: {top_model['model']}")

chosen_model = model_dict[top_model["model"]]
cv_auc_scores = cross_val_score(
    chosen_model, 
    X_full_scaled, 
    y_full, 
    cv=5, 
    scoring="roc_auc"
)

print("\n" + "="*80)
print("Fold-wise AUC scores:", cv_auc_scores)
print(f"\nMean AUC across folds: {cv_auc_scores.mean():.4f}")

