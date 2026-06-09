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


import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, classification_report, 
    precision_recall_curve, confusion_matrix, 
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.feature_selection import SelectKBest, f_classif

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# Store test IDs for submission
test_ids = test['id']

# Data Preparation
def prepare_data(df):
    df = df.drop(columns=['id'])
    return df

train_df = prepare_data(train)
test_prepared = prepare_data(test)

# Define target (1: Extrovert, 0: Introvert)
target = 'Personality'
X = train_df.drop(columns=[target])
y = train_df[target].map({'Extrovert': 1, 'Introvert': 0})

# Check class distribution
print("Class Distribution:")
print(y.value_counts(normalize=True))

# Define features
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 
                     'Going_outside', 'Friends_circle_size', 'Post_frequency']
binary_features = ['Stage_fear', 'Drained_after_socializing']

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('binary', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='No')),
            ('encoder', OrdinalEncoder())
        ]), binary_features)
    ]
)

# Define class weights (inverse of class frequency)
class_weights = {1: 0.26, 0: 0.74}  # Extrovert: 74%, Introvert: 26%

# Initialize Models with Class Weighting
models = {
    "Logistic Regression (Balanced)": LogisticRegression(
        class_weight=class_weights,
        solver='saga',
        penalty='elasticnet',
        l1_ratio=0.5,
        max_iter=1000
    ),
    "Calibrated SVM": CalibratedClassifierCV(
        base_estimator=SVC(
            class_weight=class_weights,
            kernel='rbf',
            probability=False  # Calibration will handle probabilities
        ),
        method='sigmoid',
        cv=3
    ),
    "Random Forest (Balanced)": RandomForestClassifier(
        class_weight=class_weights,
        n_estimators=200,
        max_depth=7,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
    )
}

# Train-Validation Split (Stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Evaluate Models with Optimal Threshold Tuning
best_model = None
best_auc = 0
results = {}

for name, model in models.items():
    print(f"\n{'-'*50}")
    print(f"Training & Evaluating: {name}")
    print(f"{'-'*50}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', SelectKBest(f_classif, k='all')),  # Optional feature selection
        ('classifier', model)
    ])
    
    # Cross-Validation AUC
    cv_auc = cross_val_score(pipeline, X_train, y_train, 
                            cv=StratifiedKFold(5), 
                            scoring='roc_auc').mean()
    print(f"CV ROC-AUC: {cv_auc:.4f}")
    
    # Fit on full training data
    pipeline.fit(X_train, y_train)
    
    # Validation predictions
    y_probs = pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_probs)
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    
    # Find optimal threshold (maximizing F1)
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print("\nClassification Report at Optimal Threshold:")
    y_pred = (y_probs >= optimal_threshold).astype(int)
    print(classification_report(y_val, y_pred))
    
    # Store results
    results[name] = {
        'model': pipeline,
        'cv_auc': cv_auc,
        'val_auc': val_auc,
        'threshold': optimal_threshold
    }
    
    # Update best model
    if val_auc > best_auc:
        best_auc = val_auc
        best_model = pipeline
        best_threshold = optimal_threshold

# Visualize Best Model Performance
print("\n" + "="*50)
print(f"Best Model: {max(results, key=lambda x: results[x]['val_auc'])}")
print(f"Validation AUC: {best_auc:.4f}")
print("="*50)

# Plot Confusion Matrix and ROC Curve
y_val_pred = (best_model.predict_proba(X_val)[:, 1] >= best_threshold).astype(int)
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                             display_labels=['Introvert', 'Extrovert'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Optimal Threshold)")
plt.show()

RocCurveDisplay.from_estimator(best_model, X_val, y_val)
plt.title("ROC Curve")
plt.show()

# Train Best Model on Full Data & Generate Submission
print("\nRetraining best model on full data...")
best_model.fit(X, y)

# Predict probabilities on test set
test_probs = best_model.predict_proba(test_prepared)[:, 1]

# Convert probabilities to class predictions using optimal threshold
test_preds = (test_probs >= best_threshold).astype(int)

# Map 0/1 back to 'Introvert'/'Extrovert'
test_preds_labels = np.where(test_preds == 1, 'Extrovert', 'Introvert')

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': test_preds_labels
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file saved!")
print("\nSample submission:")
print(submission.head())




