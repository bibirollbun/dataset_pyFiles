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


import numpy as np # linear algebra
import pandas as pd


train = pd.read_csv('/kaggle/input/benin-national-ai-olympiad-selection/train.csv')
test = pd.read_csv('/kaggle/input/benin-national-ai-olympiad-selection/test.csv')
submission = pd.read_csv('/kaggle/input/benin-national-ai-olympiad-selection/sample_submission.csv')
train.head(3)


train.target.value_counts()


# dimensions
print('Train Set:', train.shape)
print('Test Set :', test.shape)


train.info(verbose=True, show_counts=True) # le type de chaque colonne


# La liste des features
features = train.columns.tolist()
print(features)


# On enleve ID et target de la liste
features.remove('id')
features.remove('target')


print(features)


# decrire les features
train[features].describe()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[features], train['target'], test_size=0.3, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc


model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


y_pred_proba = model.predict_proba(X_test)
y_pred_proba_positive_class = y_pred_proba[:, 1] # Probabilities for class 1
auc_score = roc_auc_score(y_test, y_pred_proba_positive_class)
print(f"\nAUC Score: {auc_score:.4f}")


import matplotlib.pyplot as plt
# --- 6. Calculate ROC Curve points ---
# The roc_curve function returns false positive rates (fpr), true positive rates (tpr), and thresholds.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_positive_class)

# Calculate the Area Under the Curve (AUC) using the fpr and tpr.
# This is an alternative way to get the AUC, same as roc_auc_score result.
roc_auc_from_curve = auc(fpr, tpr)
print(f"AUC calculated from ROC curve points: {roc_auc_from_curve:.4f}")


# --- 7. Plot ROC Curve (Optional) ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level (AUC = 0.5)') # Diagonal line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


test_proba = model.predict_proba(test[features])
test_proba_positive_class = test_proba[:, 1]


submission.head()


submission['target'] = test_proba_positive_class


submission.to_csv('submission.csv', index=False)




