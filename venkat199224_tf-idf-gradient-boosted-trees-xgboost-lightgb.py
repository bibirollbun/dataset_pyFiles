import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df  = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# Combine body + rule + subreddit into a single text field
train_df["text"] = train_df["body"] + " " + train_df["rule"] + " " + train_df["subreddit"]
test_df["text"]  = test_df["body"]  + " " + test_df["rule"]  + " " + test_df["subreddit"]


tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1,2))
X = tfidf.fit_transform(train_df["text"])
X_test = tfidf.transform(test_df["text"])
y = train_df["rule_violation"]

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# Define the model
model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    num_leaves=128,
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train with early stopping via callbacks
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)]
)


val_pred_proba = model.predict_proba(X_val)[:, 1]
val_pred = model.predict(X_val)



print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print("Precision:", precision_score(y_val, val_pred))
print("Recall:", recall_score(y_val, val_pred))
print("F1 Score:", f1_score(y_val, val_pred))
print("ROC AUC:", roc_auc_score(y_val, val_pred_proba))
print("\nClassification Report:\n", classification_report(y_val, val_pred))

# Confusion Matrix
cm = confusion_matrix(y_val, val_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Violation","Violation"],
            yticklabels=["No Violation","Violation"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


test_pred_proba = model.predict(X_test)

submission = pd.DataFrame({
    "row_id": test_df["row_id"], 
    "rule_violation": test_pred_proba
})
submission.to_csv("submission.csv", index=False)
submission.head()

