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
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE, ADASYN



# Load dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')



# Drop unnecessary columns
train_df.drop(columns=['day'], inplace=True, errors='ignore')


# Define features (X) and target variable (y)
X = train_df.drop(columns=['rainfall'])  # Assuming 'rainfall' is the target variable
y = train_df['rainfall']


# Load dataset (Assume X and y are already loaded)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Handle Class Imbalance
smote = SMOTE(random_state=42)
adasyn = ADASYN(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_scaled, y_train)


# Define Classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(solver='liblinear', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=100, verbose=0, random_state=42)
}


# Function to Train and Evaluate Classifiers
def train_evaluate_classifier(name, classifier, X_train_resampled, y_train_resampled, method_name):
    print(f"\n--- Training {name} with {method_name} ---")
    classifier.fit(X_train_resampled, y_train_resampled)
    
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    auc = None
    try:
        y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    except AttributeError:
        pass  # Some models don't support predict_proba
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC: {auc if auc else 'N/A'}")
    
    return {
        'model_name': name,
        'resampling_method': method_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'trained_model': classifier
    }



# Train and Evaluate Models
results = []
for name, clf in classifiers.items():
    results.append(train_evaluate_classifier(name, clf, X_train_scaled, y_train, "Original Data"))
    results.append(train_evaluate_classifier(name, clf, X_train_smote, y_train_smote, "SMOTE"))
    results.append(train_evaluate_classifier(name, clf, X_train_adasyn, y_train_adasyn, "ADASYN"))


# Find Best Model Based on Minority Class F1-Score
class_counts = Counter(y_train)
minority_class_label = min(class_counts, key=class_counts.get)

best_result = None
best_f1_minority = -1

for result in results:
    model = result['trained_model']
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    f1_minority = report.get(str(minority_class_label), {}).get('f1-score', 0)
    
    if f1_minority > best_f1_minority:
        best_f1_minority = f1_minority
        best_result = result


# Display Best Model Information
if best_result:
    print("\n--- Best Model Based on Minority Class F1-Score ---")
    print(f"Model: {best_result['model_name']}")
    print(f"Resampling Method: {best_result['resampling_method']}")
    print(f"F1-Score (Minority Class): {best_f1_minority:.4f}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall: {best_result['recall']:.4f}")
    if best_result['auc'] is not None:
        print(f"AUC: {best_result['auc']:.4f}")
    
    best_model = best_result['trained_model']
else:
    print("No best model found.")


# Load and Preprocess Test Data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_df.fillna({'winddirection': test_df['winddirection'].mean()}, inplace=True)
test_df.drop(columns=['day'], inplace=True, errors='ignore')
X_test_final = scaler.transform(test_df)


# Make Predictions with Best Model
if best_model:
    y_new_test_pred = best_model.predict(X_test_final)
    print("\n--- Predictions on New Test Data using the Best Model ---")
    print(y_new_test_pred)
    
    submission = pd.DataFrame({'id': test_df['id'], 'rainfall': y_new_test_pred})
    submission.to_csv("submission.csv", index=False)
    print("Submission file saved as submission.csv")
else:
    print("\n No best model found, cannot make predictions on new test data.")


