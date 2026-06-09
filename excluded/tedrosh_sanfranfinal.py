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


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay



# Load data
train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
test = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip')


# Preprocessing
def preprocess(df):
    df['Dates'] = pd.to_datetime(df['Dates'])
    df['Year'] = df['Dates'].dt.year
    df['Month'] = df['Dates'].dt.month
    df['Day'] = df['Dates'].dt.day
    df['Hour'] = df['Dates'].dt.hour
    return df

train = preprocess(train)
test = preprocess(test)


# Encode target
le = LabelEncoder()
y_train = le.fit_transform(train['Category'])

# One-hot encode categorical features
def encode_features(df):
    day_dummies = pd.get_dummies(df['DayOfWeek'], prefix='Day')
    district_dummies = pd.get_dummies(df['PdDistrict'], prefix='District')
    numerical = df[['X', 'Y', 'Year', 'Month', 'Day', 'Hour']]
    return pd.concat([day_dummies, district_dummies, numerical], axis=1)

X_train = encode_features(train)
X_test = encode_features(test)


plt.figure(figsize=(12,6))
sns.countplot(y='Category', data=train, order=train['Category'].value_counts().index)
plt.title('Distribution of Crime Categories')
plt.xlabel('Number of Incidents')
plt.ylabel('Crime Category')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(x='Hour', data=train)
plt.title('Crimes by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Crimes')
plt.tight_layout()
plt.show()



plt.figure(figsize=(10,5))
sns.countplot(x='PdDistrict', data=train, order=train['PdDistrict'].value_counts().index)
plt.title('Crimes by Police District')
plt.xlabel('Police District')
plt.ylabel('Number of Crimes')
plt.tight_layout()
plt.show()



# Correlation matrix for numerical features
numerical_features = ['X', 'Y', 'Year', 'Month', 'Day', 'Hour']
corr_matrix = train[numerical_features].corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()



# Align test columns with train (fill missing with 0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train_split, y_train_split)


# XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_split, y_train_split)


# SVM with scaling and calibration
scaler = StandardScaler()
X_train_svm = X_train.copy()
X_train_svm[['X', 'Y', 'Year', 'Month', 'Day', 'Hour']] = scaler.fit_transform(
    X_train_svm[['X', 'Y', 'Year', 'Month', 'Day', 'Hour']]
)
X_test_svm = X_test.copy()
X_test_svm[['X', 'Y', 'Year', 'Month', 'Day', 'Hour']] = scaler.transform(
    X_test_svm[['X', 'Y', 'Year', 'Month', 'Day', 'Hour']]
)

svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1)
calibrated_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=3)
calibrated_svm.fit(X_train_svm, y_train)


# Random Forest Metrics
rf_pred = rf.predict(X_val_split)
print("Random Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_val_split, rf_pred):.4f}")
print(f"Precision (macro): {precision_score(y_val_split, rf_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_val_split, rf_pred, average='macro'):.4f}")
print(f"F1 Score (macro): {f1_score(y_val_split, rf_pred, average='macro'):.4f}")
print("Classification Report:\n", classification_report(y_val_split, rf_pred, target_names=le.classes_))

# XGBoost Metrics
xgb_pred = xgb_model.predict(X_val_split)
print("XGBoost Metrics:")
print(f"Accuracy: {accuracy_score(y_val_split, xgb_pred):.4f}")
print(f"Precision (macro): {precision_score(y_val_split, xgb_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_val_split, xgb_pred, average='macro'):.4f}")
print(f"F1 Score (macro): {f1_score(y_val_split, xgb_pred, average='macro'):.4f}")
print("Classification Report:\n", classification_report(y_val_split, xgb_pred, target_names=le.classes_))

# Calibrated SVM Metrics
svm_pred = calibrated_svm.predict(X_val_split)
print("Calibrated SVM Metrics:")
print(f"Accuracy: {accuracy_score(y_val_split, svm_pred):.4f}")
print(f"Precision (macro): {precision_score(y_val_split, svm_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_val_split, svm_pred, average='macro'):.4f}")
print(f"F1 Score (macro): {f1_score(y_val_split, svm_pred, average='macro'):.4f}")
print("Classification Report:\n", classification_report(y_val_split, svm_pred, target_names=le.classes_))



from sklearn.metrics import mean_squared_error
import numpy as np

# For each model, get predicted class labels
rf_pred = rf.predict(X_val_split)
xgb_pred = xgb_model.predict(X_val_split)
svm_pred = calibrated_svm.predict(X_val_svm)  # X_val_svm is your scaled validation set

# Compute RMSE for each model
rf_rmse = np.sqrt(mean_squared_error(y_val_split, rf_pred))
xgb_rmse = np.sqrt(mean_squared_error(y_val_split, xgb_pred))
svm_rmse = np.sqrt(mean_squared_error(y_val_split, svm_pred))

print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"XGBoost RMSE: {xgb_rmse:.4f}")
print(f"Calibrated SVM RMSE: {svm_rmse:.4f}")



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Random Forest Matrix
rf_pred = rf.predict(X_val_split)
ConfusionMatrixDisplay.from_predictions(
    y_val_split,
    rf_pred,
    display_labels=le.classes_,
    ax=axes[0],
    xticks_rotation=90,
    colorbar=False
)
axes[0].set_title('Random Forest')

# XGBoost Matrix
xgb_pred = xgb_model.predict(X_val_split)
ConfusionMatrixDisplay.from_predictions(
    y_val_split,
    xgb_pred,
    display_labels=le.classes_,
    ax=axes[1],
    xticks_rotation=90,
    colorbar=False
)
axes[1].set_title('XGBoost')

# Calibrated SVM Matrix
svm_pred = calibrated_svm.predict(X_val_split)
ConfusionMatrixDisplay.from_predictions(
    y_val_split,
    svm_pred,
    display_labels=le.classes_,
    ax=axes[2],
    xticks_rotation=90,
    colorbar=False
)
axes[2].set_title('Calibrated SVM')

# Formatting
plt.suptitle('Classifier Comparison: Confusion Matrices', y=1.02)
plt.tight_layout()
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Random Forest Matrix
rf_pred = rf.predict(X_val_split)
ConfusionMatrixDisplay.from_predictions(
    y_val_split,
    rf_pred,
    display_labels=le.classes_,
    ax=axes[0],
    xticks_rotation=90,
    colorbar=False
)
axes[0].set_title('Random Forest')

# XGBoost Matrix
xgb_pred = xgb_model.predict(X_val_split)
ConfusionMatrixDisplay.from_predictions(
    y_val_split,
    xgb_pred,
    display_labels=le.classes_,
    ax=axes[1],
    xticks_rotation=90,
    colorbar=False
)
axes[1].set_title('XGBoost')

# Calibrated SVM Matrix
svm_pred = calibrated_svm.predict(X_val_split)
ConfusionMatrixDisplay.from_predictions(
    y_val_split,
    svm_pred,
    display_labels=le.classes_,
    ax=axes[2],
    xticks_rotation=90,
    colorbar=False
)
axes[2].set_title('Calibrated SVM')

# Formatting
plt.suptitle('Classifier Comparison: Confusion Matrices', y=1.02)
plt.tight_layout()
plt.show()



# 1. Confusion Matrix Comparison
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for ax, (name, pred) in zip(axes, [('Random Forest', rf_pred), 
                                  ('XGBoost', xgb_pred), 
                                  ('Calibrated SVM', svm_pred)]):
    cm = confusion_matrix(y_val_split, pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                 display_labels=le.classes_)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_val_split, pred):.3f}')
    
plt.suptitle('Normalized Confusion Matrix Comparison', y=1.02)
plt.tight_layout()
plt.show()

# 2. Metric Comparison Bar Plot
metrics = {
    'Accuracy': accuracy_score,
    'F1-Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
model_names = ['Random Forest', 'XGBoost', 'Calibrated SVM']
predictions = [rf_pred, xgb_pred, svm_pred]

for ax, (metric_name, metric_fn) in zip(axes, metrics.items()):
    scores = [metric_fn(y_val_split, pred) for pred in predictions]
    ax.bar(model_names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylim(0, 1)
    ax.set_title(f'{metric_name} Comparison')
    ax.set_ylabel(metric_name)
    
    # Add value labels
    for i, score in enumerate(scores):
        ax.text(i, score + 0.02, f'{score:.3f}', ha='center')

plt.tight_layout()
plt.show()

# 3. Classification Report Table
print("Random Forest Classification Report:")
print(classification_report(y_val_split, rf_pred, target_names=le.classes_))

print("\nXGBoost Classification Report:")
print(classification_report(y_val_split, xgb_pred, target_names=le.classes_))

print("\nCalibrated SVM Classification Report:")
print(classification_report(y_val_split, svm_pred, target_names=le.classes_))


import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import numpy as np

# Create a copy of the validation data
X_val_svm = X_val_split.copy()

# Scale ONLY the numerical columns (same as during training)
numerical_cols = ['X', 'Y', 'Year', 'Month', 'Day', 'Hour']
X_val_svm[numerical_cols] = scaler.transform(X_val_split[numerical_cols])

# Now use this preprocessed validation data for SVM
svm_proba = calibrated_svm.predict_proba(X_val_svm)

# Binarize the validation labels for OvR ROC
label_binarizer = LabelBinarizer()
y_val_bin = label_binarizer.fit_transform(y_val_split)
n_classes = y_val_bin.shape[1]

# Get predicted probabilities for each model
rf_proba = rf.predict_proba(X_val_split)
xgb_proba = xgb_model.predict_proba(X_val_split)
# Get predicted probabilities for SVM
X_val_svm = X_val_split.copy()
X_val_svm[['X', 'Y', 'Year', 'Month', 'Day', 'Hour']] = scaler.transform(
    X_val_split[['X', 'Y', 'Year', 'Month', 'Day', 'Hour']]  # Scale only these columns
)
svm_proba = calibrated_svm.predict_proba(X_val_svm)  # Now works!


# Colors for ROC curves
colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown'])

# Function to plot multiclass ROC for a model
def plot_multiclass_roc(proba, title, ax):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label='Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
    ax.grid(alpha=.4)

# Create a figure with subplots for each model
fig, axes = plt.subplots(1, 3, figsize=(24, 12))

plot_multiclass_roc(rf_proba, 'Random Forest ROC', axes[0])
plot_multiclass_roc(xgb_proba, 'XGBoost ROC', axes[1])
plot_multiclass_roc(svm_proba, 'Calibrated SVM ROC', axes[2])

plt.tight_layout()
plt.show()



# Generate predictions
def submit_probs(model, X_test_data, filename):
    probs = model.predict_proba(X_test_data)
    submission = pd.DataFrame(probs, columns=le.classes_)
    submission.insert(0, 'Id', test['Id'])
    submission.to_csv(filename, index=False)

submit_probs(rf, X_test, 'submission_rf.csv')
submit_probs(xgb_model, X_test, 'submission.csv')
submit_probs(calibrated_svm, X_test_svm, 'submission_svm.csv')

