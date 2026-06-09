import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Step 1: Import libraries & load the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For model training and evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Load the datasets
train = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv")
test = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv")

# Quick look at the data
print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# Step 2: Data Exploration & Visualization

# Check for missing values
print("Missing values in train:\n", train.isnull().sum().sum())
print("Missing values in test:\n", test.isnull().sum().sum())

# Distribution of target classes
plt.figure(figsize=(10,5))
sns.countplot(x='target', data=train, order=sorted(train['target'].unique()))
plt.title("Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.show()

# Encode the target labels for modeling
le = LabelEncoder()
train['target_encoded'] = le.fit_transform(train['target'])

# Basic statistics
print(train.describe())


# Step 3: Data Preprocessing

from sklearn.preprocessing import StandardScaler

# Drop id column
X = train.drop(['id', 'target', 'target_encoded'], axis=1)
y = train['target_encoded']

# Scale the features (XGBoost doesn't require it but it's sometimes beneficial)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare test data (no label)
X_test = test.drop(['id'], axis=1)
X_test_scaled = scaler.transform(X_test)

# Split training data for local validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)


import xgboost as xgb
from sklearn.metrics import log_loss
import numpy as np

# Step 1: Define model parameters
params = {
    "objective": "multi:softprob",
    "num_class": 9,
    "eval_metric": ["mlogloss", "merror"],
    "learning_rate": 0.1,
    "max_depth": 10,
    "min_child_weight": 2,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "gamma": 0.2,
    "lambda": 1.0,
    "alpha": 0.5,
    "seed": 42
}


# Step 2: Convert training and validation data into DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Step 3: Cross-validation to find best number of boosting rounds
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    nfold=20,
    early_stopping_rounds=15,
    stratified=True,
    verbose_eval=True,
    as_pandas=True
)

# Get best iteration based on log loss
best_num_boost_round = cv_results['test-mlogloss-mean'].idxmin()
print("\nBest boosting round from CV:", best_num_boost_round)
print("Best test log loss from CV:", cv_results['test-mlogloss-mean'].min())

# Step 4: Train final model with best number of rounds
evals_result = {}
final_model = xgb.train(
    params,
    dtrain,
    num_boost_round=best_num_boost_round,
    evals=[(dtrain, "train"), (dval, "eval")],
    evals_result=evals_result,
    verbose_eval=False
)

# Step 5: Predict on validation set
y_val_pred = final_model.predict(dval)
val_log_loss = log_loss(y_val, y_val_pred)
print(f"Validation Log Loss: {val_log_loss:.5f}")

# Step 6: Accuracy on validation set
y_val_pred_class = np.argmax(y_val_pred, axis=1)
val_accuracy = np.mean(y_val_pred_class == y_val)
print(f"Validation Accuracy: {val_accuracy:.5f}")



# Step 5: Predict test set & prepare submission

# Convert scaled test data to DMatrix
dtest = xgb.DMatrix(X_test_scaled)

# Predict probabilities
y_test_pred = final_model.predict(dtest)


# Create DataFrame for submission
submission = pd.DataFrame(y_test_pred, columns=[f"Class_{i}" for i in range(1, 10)])
submission.insert(0, "id", test["id"])

# Save to CSV
submission.to_csv("xgboost_submission_10.csv", index=False)
print("Submission file saved as 'xgboost_submission.csv'")
submission=pd.read_csv("/kaggle/working/xgboost_submission_10.csv")
submission.head(10)


from sklearn.metrics import accuracy_score, precision_score, recall_score

# Get predicted classes (not probabilities)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Accuracy
accuracy = accuracy_score(y_val, y_val_pred_classes)
# Precision, Recall (macro = average across all classes)
precision = precision_score(y_val, y_val_pred_classes, average='macro')
recall = recall_score(y_val, y_val_pred_classes, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_val_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[f"Class_{i}" for i in range(1,10)],
            yticklabels=[f"Class_{i}" for i in range(1,10)])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# Step 6.3: Overfitting Analysis – Log Loss Curve (after retraining)

train_logloss = evals_result['train']['mlogloss']
eval_logloss = evals_result['eval']['mlogloss']

plt.figure(figsize=(10, 6))
plt.plot(train_logloss, label='Train Log Loss')
plt.plot(eval_logloss, label='Validation Log Loss')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('Log Loss Curve - Overfitting Check')
plt.legend()
plt.grid(True)
plt.show()





