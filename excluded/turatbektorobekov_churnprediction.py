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
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE



train_df = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/test.csv')
df = train_df.copy()
df.drop('Customer_ID', axis=1, inplace=True)


df = train_df.copy()
df.drop('Customer_ID', axis=1, inplace=True)



label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop('Churn', axis=1)
y = df['Churn']
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)


X_train, X_val, y_train, y_val = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)


model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)


y_val_proba = model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba > 0.5).astype(int)


print("Classification Report:\n")
print(classification_report(y_val, y_val_pred))


conf_matrix = confusion_matrix(y_val, y_val_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


roc_auc = roc_auc_score(y_val, y_val_proba)
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)


plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()


importances = model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df.sort_values(by='Importance', ascending=False, inplace=True)
plt.figure(figsize=(8,6))
sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature')
plt.title("Top 15 Feature Importances")
plt.show()


test_df_copy = test_df.copy()
test_df_copy.drop('Customer_ID', axis=1, inplace=True)

# Кодирование как в train
for col in test_df_copy.select_dtypes(include='object').columns:
    if col in label_encoders:
        le = label_encoders[col]
        test_df_copy[col] = le.transform(test_df_copy[col])
    else:
        # если новый unseen label — ставим 0 (или любое значение по умолчанию)
        test_df_copy[col] = 0


test_preds_proba = model.predict_proba(test_df_copy)[:, 1]
submission = pd.DataFrame({
    'Customer_ID': test_df['Customer_ID'],
    'Churn_Probability': test_preds_proba
})
submission.to_csv('submission.csv', index=False)

