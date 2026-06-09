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


train=pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


train.head()


test.head()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ---------------------------
# 1️⃣ Load train and test data
# ---------------------------
train=pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')

# ---------------------------
# 2️⃣ Separate features and target
# ---------------------------
X = train.drop("loan_status", axis=1)
y = train["loan_status"]

# ---------------------------
# 3️⃣ Encode categorical columns
# ---------------------------
cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# ---------------------------
# 4️⃣ Scale numeric features
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.drop("id", axis=1))
test_scaled = scaler.transform(test.drop("id", axis=1))

# ---------------------------
# 5️⃣ Split for validation
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------
# 6️⃣ Train Random Forest model
# ---------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------------------------
# 7️⃣ Evaluate performance
# ---------------------------
y_pred_prob = model.predict_proba(X_val)[:, 1]
y_pred_class = (y_pred_prob >= 0.5).astype(int)

print("✅ Validation Accuracy:", accuracy_score(y_val, y_pred_class))
print("✅ ROC-AUC:", roc_auc_score(y_val, y_pred_prob))
print(classification_report(y_val, y_pred_class))

# ---------------------------
# 8️⃣ Predict probability on test data
# ---------------------------
test_pred_prob = model.predict_proba(test_scaled)[:, 1]  # probability of loan_status = 1

# ---------------------------
# 9️⃣ Save submission file (with probabilities)
# ---------------------------
submission = pd.DataFrame({
    "id": test["id"],
    "loan_status": test_pred_prob  # values between 0 and 1
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully with probabilities (0–1)!")





