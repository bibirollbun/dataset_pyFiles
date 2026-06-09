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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")    # test data without 'y'
print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
print(train_data.head())
print("Columns:", train_data.columns.tolist())
print("Shape:", train_data.shape)
print(train_data.head())
label_encoder = LabelEncoder()
for col in train_data.select_dtypes(include=['object']).columns:
    if col != 'y':  # avoid target column
        label_encoder.fit(train_data[col])
        train_data[col] = label_encoder.transform(train_data[col])
        if col in test_data.columns:
            test_data[col] = label_encoder.transform(test_data[col])

X_train = train_data.drop(columns=['y'])
y_train = train_data['y']
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(test_data)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# --- Step 8: Create Submission File ---
submission = pd.DataFrame({
    "id": range(750000, 750000 + len(y_prob)),
    "y": y_prob
})

# --- Step 9: Save Submission ---
submission.to_csv("submission_bank_classification.csv", index=False)

print("\n✅ Submission file saved as 'submission.csv'")
print(submission.head())
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.countplot(x='y', data=train_data)
plt.title("Distribution of Target Variable (y)", fontsize=16)
plt.xlabel("Subscribed to Term Deposit? (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Number of Clients", fontsize=12)
# Adding annotations to the bars
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, ha='center', va='bottom')
plt.show()
plt.figure(figsize=(12, 8))
sns.countplot(y='job', data=train_data, order=train_data['job'].value_counts().index, palette='viridis')
plt.title('Distribution of Client Jobs', fontsize=16)
plt.xlabel('Number of Clients', fontsize=12)
plt.ylabel('Job Type', fontsize=12)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='education', data=train_data, order=train_data['education'].value_counts().index, palette='plasma')
plt.title('Distribution of Client Education Levels', fontsize=16)
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('Number of Clients', fontsize=12)
plt.show()


