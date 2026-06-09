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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.head()


print(train.shape)
print(test.shape)


train.info()


print("values of stage fear : ", train['Stage_fear'].unique())
print("values of drained after socializing : ", train['Drained_after_socializing'].unique())
print("values of personality : ", train['Personality'].unique())


train.describe()


train.describe(include=['object'])


for i in train.columns:
    print(i," : ",train[i].nunique())


train.isna().sum()


# Handle missing numeric values using median
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency']

for col in num_cols:
    median_value = train[col].median()
    train[col].fillna(median_value, inplace=True)
    test[col].fillna(median_value, inplace=True)

# Handle missing categorical values using mode
cat_cols = ['Stage_fear', 'Drained_after_socializing']

for col in cat_cols:
    mode_value = train[col].mode()[0]
    train[col].fillna(mode_value, inplace=True)
    test[col].fillna(mode_value, inplace=True)


train.isna().sum()


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


label_encoders = {}
for col in ['Stage_fear', 'Drained_after_socializing']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
train['Personality'] = target_encoder.fit_transform(train['Personality'])  # Introvert=0, Extrovert=1

# --- Step 5: Split Features and Target ---
X = train.drop(columns=['id', 'Personality'])
y = train['Personality']

# --- Step 6: Train/Validation Split ---
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 7: Scale Numerical Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(test.drop(columns=['id']))

# --- Step 8: Train Model ---
model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(X_train_scaled, y_train)

# --- Step 9: Evaluate ---
y_pred = model.predict(X_valid_scaled)
print("Validation Accuracy:", accuracy_score(y_valid, y_pred))
print("\nClassification Report:\n", classification_report(y_valid, y_pred))
sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# --- Step 10: Predict on Test Data ---
test_pred = model.predict(X_test_scaled)

# --- Step 11: Create Submission ---
submission = sample_submission.copy()
submission['Personality'] = target_encoder.inverse_transform(test_pred)
submission.to_csv("submission.csv", index=False)

print("✅ Submission file created successfully!")
submission.head()

