import pandas as pd
import numpy as np

# File paths
train_path = "/kaggle/input/playground-series-s5e8/train.csv"
test_path = "/kaggle/input/playground-series-s5e8/test.csv"


def wrangle(filepath):
    # Read dataset
    df = pd.read_csv(filepath, index_col="id")
    
    # âœ… Feature engineering
    df["questions"] = df["default"] + " " + df["housing"] + " " + df["contact"]
    df["status"] = df["job"] + " " + df["education"] + " " + df["marital"]
    df["intellect"] = df["job"] + " " + df["education"]
    
    df["min_duration_sin"] = np.sin(df["duration"] / 60)
    df["min_duration_cos"] = np.cos(df["duration"] / 60)
    
    df["date"] = df["day"].astype(str) + " " + df["month"]
    df["contacted_before"] = (df["pdays"] != -1).astype(int)
    df["balance_log"] = np.log1p(df["balance"].clip(lower=0))
    
    # Drop unused column
    df = df.drop(columns="pdays")
    
    # âœ… Quick summary
    print(f"Shape: {df.shape}")
    print("\nColumns:\n", df.columns.tolist())
    print("\nUnique values per column:\n", df.nunique())
    
    return df

# ðŸ“Œ Load datasets
train = wrangle(train_path)
test = wrangle(test_path)

# ðŸ“Œ Show first few rows
train.head()



# Distribution of target variable
import matplotlib.pyplot as plt
target_dist = train["y"].value_counts(normalize=True)

plt.bar(target_dist.index.astype(str), target_dist.values)
plt.xlabel("Target (y)")
plt.ylabel("Proportion")
plt.title("Target Distribution")
plt.show()



# Checking for Multicollinearity in the Numerical Dataset
import seaborn as sns
num_data = train.select_dtypes("number").drop(columns="y")
plt.figure(figsize=(8,5))
sns.heatmap(num_data.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()



# Correlation of numerical features with target 'y'
corr_with_target = abs(train.select_dtypes("number").corr()["y"]).sort_values()

corr_with_target.plot(kind="barh", figsize=(6,5))
plt.title("Correlation with Target (y)")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.show()



target = "y"
X = train.drop(columns=target)
y = train[target]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)



def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train, predict and evaluate a single model."""
    pipe = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        StandardScaler(),
        model
    )
    
    # Train
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    
    # ROC AUC
    roc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {roc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["0", "1"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    return pipe, roc



from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from category_encoders import OneHotEncoder
import matplotlib.pyplot as plt



def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    1) Build pipeline  2) Fit  3) Predict  4) Metrics + Confusion Matrix  5) Return
    """
    # 1) pipeline: one-hot encode categoricals -> model (no scaler needed for trees)
    pipe = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        model
    )

    # 2) train
    pipe.fit(X_train, y_train)

    # 3) predict class + probabilities/scores
    y_pred = pipe.predict(X_test)
    try:
        y_score = pipe.predict_proba(X_test)[:, 1]
    except AttributeError:
        # fallback for models without predict_proba
        y_score = pipe.decision_function(X_test)

    # 4) metrics
    roc = roc_auc_score(y_test, y_score)
    print(f"ROC AUC: {roc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 5) return fitted pipeline + score
    return pipe, roc



from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(random_state=42)
pipe, roc = evaluate_model(model, X_train, X_test, y_train, y_test)



# 1) build
pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    GradientBoostingClassifier(random_state=42)
)

# 2) fit
pipe.fit(X_train, y_train)

# 3) predict
y_pred = pipe.predict(X_test)
y_score = pipe.predict_proba(X_test)[:, 1]

# 4) metrics + plot
print("ROC AUC:", roc_auc_score(y_test, y_score))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()



# predict probabilities (1 = positive class)
test_pred = pipe.predict_proba(test)[:, 1]



sub = pd.DataFrame({
    "id": test.index,   # id from test.csv
    "y": test_pred      # predictions
})



sub.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")



sub

