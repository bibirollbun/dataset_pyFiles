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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


data_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
data_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


data_train.head()


data_test.head()


data_train.info()


data_train.describe()


data_train.isna().sum()


data_test.isna().sum()


data_train.duplicated().sum()


data_test.duplicated().sum()


data_train['diagnosed_diabetes'].value_counts()


sns.countplot(x=data_train['diagnosed_diabetes'])
plt.show()


X = data_train.drop('diagnosed_diabetes', axis=1)
y = data_train['diagnosed_diabetes']


X.head()


y


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
X_train_transformed = preprocessor.fit_transform(X)
X_test_transformed = preprocessor.transform(data_test)


X_train_transformed


X_test_transformed


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_transformed)
X_test_scaled = scaler.transform(X_test_transformed)


X_train_scaled


X_test_scaled


plt.figure(figsize=(18,12))
sns.heatmap(data_train.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


model = LogisticRegression()
model.fit(X_train_scaled, y)


Y = model.predict(X_train_scaled) # For calculating accuracy


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc


accuracy = accuracy_score(y, Y)
print(f"Accuracy: {accuracy:.2f}")


confusion = confusion_matrix(y, Y)
confusion


classification_rep = classification_report(y, Y)
print(classification_rep)


Y_prob = model.predict_proba(X_train_scaled)[:,1]
roc_auc = roc_auc_score(y, Y_prob)
fpr, tpr, _ = roc_curve(y, Y_prob)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0,0.1])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receive Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


y_preds = model.predict(X_test_scaled)


y_preds


submission.head()


submission['diagnosed_diabetes'].value_counts()


submission['diagnosed_diabetes'] = y_preds


submission['diagnosed_diabetes'].value_counts()


submission.to_csv("Submission.csv", index=False)




