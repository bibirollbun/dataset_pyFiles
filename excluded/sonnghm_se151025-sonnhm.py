# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/smoking-binary-prediction-using-bio-signals/train.csv')
test = pd.read_csv('/kaggle/input/smoking-binary-prediction-using-bio-signals/test.csv')



print("First 5 rows of the train dataset:")
print(train.head())

print("\nGet train dataset info:")
print(train.info())

print("\nSummarize statistics:")
print(train.describe())

print("\nCheck for missing values:")
print(train.isnull().sum())



print("First 5 rows of the test dataset:")
print(test.head())

print("\nGet test dataset info:")
print(test.info())

print("\nSummarize statistics:")
print(test.describe())

print("\nCheck for missing values:")
print(test.isnull().sum())



#Separate features and target
X = train.drop(columns=['id', 'smoking'])
y = train['smoking']
test_ids = test['id']
X_test = test.drop(columns=['id'])



'''
Handle missing values (if any), datasets are clearly already in this contest 
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)
'''
#Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.fit_transform(X_test)



# Split training data for evaluation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)



#Train a model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)


# Predict on validation set
val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]


# Evaluate accuracy
accuracy = accuracy_score(y_val, val_preds)
print(f"Accuracy: {accuracy:.4f}")


# Confusion Matrix
cm = confusion_matrix(y_val, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-smoker", "Smoker"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()



# AUC-ROC
auc = roc_auc_score(y_val, val_probs)
print(f"AUC-ROC: {auc:.4f}")


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, val_probs)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# Classification Report
print("\nClassification Report:")
print(classification_report(y_val, val_preds))


test_preds = model.predict_proba(X_test_scaled)[:, 1]


#Create submission file
submission = pd.DataFrame({'id': test_ids, 'smoking': test_preds})
submission.to_csv('sample_submission.csv', index=False)

