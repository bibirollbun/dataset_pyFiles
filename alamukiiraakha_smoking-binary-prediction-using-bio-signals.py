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
train_data = pd.read_csv("/kaggle/input/smoking-binary-prediction-using-bio-signals/train.csv")
test_data = pd.read_csv("/kaggle/input/smoking-binary-prediction-using-bio-signals/test.csv")


train_data.head()


train_data.shape


test_data.shape


test_ids = test_data['id']


train_data.drop(columns = ["id"], inplace = True)


test_data.head()


x_test = test_data.drop(columns = ["id"], inplace = True)


train_data.describe()


train_data.info()


train_data.duplicated().sum()


train_data.isnull().sum()


train_data[['age']].boxplot()


train_data[['Cholesterol']].boxplot()


x = train_data.drop(columns=['smoking'])
y = train_data['smoking']


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,  random_state=42)


import xgboost as xgb
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)


# Train the model
model.fit(x_train, y_train)


# Predict on validation set
val_preds = model.predict(x_val)
val_probs = model.predict_proba(x_val)[:, 1]


# Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, val_preds)
print(f"Accuracy: {accuracy:.4f}")


from sklearn.svm import SVC
# Create SVM classifier
model2 = SVC(kernel='linear', C=1.0, random_state=42)


# Train the model
model2.fit(x_train, y_train)


# Predict on validation set
val_preds2 = model.predict(x_val)
val_probs2 = model.predict_proba(x_val)[:, 1]


# Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, val_preds2)
print(f"Accuracy: {accuracy:.4f}")


from sklearn.linear_model import LogisticRegression
# Create Logistic Regression classifier
model3 = LogisticRegression(max_iter=1000, random_state=42)



# Train the model
model3.fit(x_train, y_train)


# Predict on validation set
val_preds3 = model3.predict(x_val)
val_probs3 = model3.predict_proba(x_val)[:, 1]


# Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, val_preds3)
print(f"Accuracy: {accuracy:.4f}")


from sklearn.ensemble import GradientBoostingClassifier
#Train a model
model4 = GradientBoostingClassifier(random_state=42)
model4.fit(x_train, y_train)


# Predict on validation set
val_preds4 = model4.predict(x_val)
val_probs4 = model4.predict_proba(x_val)[:, 1]


# Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, val_preds4)
print(f"Accuracy: {accuracy:.4f}")


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, ConfusionMatrixDisplay


# Confusion Matrix
import matplotlib.pyplot as plt
cm = confusion_matrix(y_val, val_preds4)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-smoker", "Smoker"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


# AUC-ROC
auc = roc_auc_score(y_val, val_probs4)
print(f"AUC-ROC: {auc:.4f}")# AUC-ROC



# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, val_probs4)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()



test_preds = model4.predict_proba(test_data)[:, 1]


#Create submission file
submission = pd.DataFrame({'id': test_ids, 'smoking': test_preds})
submission.to_csv('submission.csv', index=False)




