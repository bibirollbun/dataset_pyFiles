# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


file_pattern = '/kaggle/input/neo-bank-non-sub-churn-prediction/train_*.parquet' 
all_files = glob.glob(file_pattern)
df = pd.concat([pd.read_parquet(file) for file in all_files], ignore_index=True)


df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
df['age'] = ((pd.Timestamp.now() - df['date_of_birth']).dt.days / 365).fillna(0).astype(int)


df['country'] = df['country'].fillna('unknown')
label_encoder = LabelEncoder()
df['country_encoded'] = label_encoder.fit_transform(df['country'])


features = [
    "bank_transfer_in", "bank_transfer_out", "crypto_in_volume", "crypto_out_volume",
    "tenure", "complaints", "age", "country_encoded", "bank_transfer_in_volume",
    "bank_transfer_out_volume", "crypto_in", "crypto_out", "atm_transfer_in", 
    "atm_transfer_out", "from_competitor"
]


# Prepare data for model training
X = df[features].fillna(0)
y = df['churn_due_to_fraud'].astype(int)

# Train-test split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8]
}

# Hyperparameter tuning using RandomizedSearchCV
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(model, param_grid, scoring='average_precision', cv=cv, n_iter=5, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)


best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]


print("Classification Report:\n", classification_report(y_test, y_pred))
average_precision = average_precision_score(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

print(f"Average Precision Score: {average_precision:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")


precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.show()


# Confusion matrix-based metrics
tn, fp, fn, tp = conf_matrix.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision_metric = tp / (tp + fp)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision_metric:.4f}")


# Create a submission DataFrame
submission_df = pd.DataFrame({
    'customer_id': df.loc[X_test.index, 'customer_id'], 
    'churn_due_to_fraud_pred': y_pred_proba  
})

submission_df.to_csv('submission.csv', index=False)

print("Submission CSV file has been created successfully!")




sub = pd.read_csv("/kaggle/input/neo-bank-non-sub-churn-prediction/sample_submission.csv")

# Clip values between 0.1 and 0.9 for churn column
sub['churn'] = np.clip(X_test['from_competitor'], 0.1, 0.9) 
# Save the modified submission file
sub.to_csv("sub.csv", index=False)

# Display the first few rows of the submission file
sub.head()


