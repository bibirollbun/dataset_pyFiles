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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc


submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
data_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
data_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


data_train.head()


data_train.info()


data_train.describe()


data_train.isna().sum()


data_train.duplicated().sum()


data_train['diagnosed_diabetes'].value_counts()


sns.countplot(x=data_train['diagnosed_diabetes'])
plt.show()


X = data_train.drop('diagnosed_diabetes', axis=1)
y = data_train['diagnosed_diabetes']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.head()


y_train


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Define which columns are categorical
categorical_cols = []
for label, content in X.items():
    if pd.api.types.is_object_dtype(content):
        categorical_cols.append(label)
print(categorical_cols)

# Create the transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ], 
    remainder='passthrough'  # Keeps numerical columns as they are
)

# Apply to your data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
data_test_transformed = preprocessor.transform(data_test)


X_train_transformed


X_test_transformed


data_test_transformed


plt.figure(figsize=(18,12))
sns.heatmap(data_train.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_transformed, y_train)


rf_y_preds = rf_clf.predict(X_test_transformed)
rf_y_preds


rf_accuracy = accuracy_score(y_test, rf_y_preds)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")


rf_confusion = confusion_matrix(y_test, rf_y_preds)
rf_confusion


rf_classification_rep = classification_report(y_test, rf_y_preds)
print(rf_classification_rep)


rf_Y_prob = rf_clf.predict_proba(X_test_transformed)[:,1]
rf_roc_auc = roc_auc_score(y_test, rf_Y_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_Y_prob)


plt.figure()
plt.plot(rf_fpr, rf_tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {rf_roc_auc:.2f})')
# plt.xlim([0.0,0.1])
# plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receive Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train_transformed, y_train)


gb_y_preds = gb_clf.predict(X_test_transformed)
gb_y_preds


gb_accuracy = accuracy_score(y_test, gb_y_preds)
print(f"Gradient Boosting Accuracy Score: {gb_accuracy:.2f}")


gb_confusion = confusion_matrix(y_test, gb_y_preds)
gb_confusion


gb_classification_rep = classification_report(y_test, gb_y_preds)
print(gb_classification_rep)


gb_Y_prob = gb_clf.predict_proba(X_test_transformed)[:,1]
gb_roc_auc = roc_auc_score(y_test, gb_Y_prob)
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_Y_prob)


plt.figure()
plt.plot(gb_fpr, gb_tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {gb_roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


y_preds = gb_clf.predict(data_test_transformed)
y_preds


submission.head()


submission['diagnosed_diabetes'] = y_preds


submission['diagnosed_diabetes'].value_counts()


submission.to_csv('Submission.csv', index=False)




