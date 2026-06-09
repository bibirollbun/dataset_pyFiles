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
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")



print(train.shape)
print(train.columns)
train.head()

# Target variable
sns.countplot(x='y', data=train)
plt.title('Target Variable Distribution')


# Check for missing values
train.isnull().sum()

# Check for unique values per column
for col in train.columns:
    print(f"{col}: {train[col].nunique()}")



train.drop('id', axis=1, inplace=True)
test_ids = test['id']
test.drop('id', axis=1, inplace=True)


# ------------------ Replace 'unknown' if any ------------------
train.replace('unknown', np.nan, inplace=True)
test.replace('unknown', np.nan, inplace=True)



# Fill with mode for categorical values
for col in train.select_dtypes(include='object').columns:
    mode = train[col].mode()[0]
    train[col].fillna(mode, inplace=True)
    test[col].fillna(mode, inplace=True)



# ------------------ Encode Categorical ------------------
cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    le_dict[col] = le

# Encode binary columns
bin_cols = ['default', 'housing', 'loan']
for col in bin_cols:
    train[col] = train[col].map({'yes': 1, 'no': 0}).astype(int) if train[col].dtype == object else train[col]
    test[col] = test[col].map({'yes': 1, 'no': 0}).astype(int) if test[col].dtype == object else test[col]



# ------------------ Scale Numeric Features ------------------
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


# ------------------ Train-Test Split ------------------
X = train.drop('y', axis=1)
y = train['y']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



# ------------------ Model Training ------------------
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)



# ------------------ Validation Evaluation ------------------
val_preds = model.predict_proba(X_val)[:,1]
print("ROC AUC Score:", roc_auc_score(y_val, val_preds))


# Optional: Threshold classification
val_class = (val_preds > 0.5).astype(int)
print(classification_report(y_val, val_class))

# ------------------ Test Predictions ------------------
test_preds = model.predict_proba(test)[:, 1]
submission = sample_submission.copy()
submission['y'] = test_preds
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file created: submission.csv")


# -------- Imports --------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from lightgbm import LGBMClassifier
    lightgbm_available = True
except ImportError:
    lightgbm_available = False


# -------- Define Models --------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "KNN": KNeighborsClassifier()
}

if lightgbm_available:
    models["LightGBM"] = LGBMClassifier()

# -------- Train & Evaluate --------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, preds),
        "Recall (Class 1)": recall_score(y_val, preds),
        "Precision (Class 1)": precision_score(y_val, preds),
        "F1 Score": f1_score(y_val, preds),
        "ROC AUC": roc_auc_score(y_val, probs)
    })

# -------- Display Leaderboard --------
results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
print("\nğŸ�† Model Leaderboard:\n")
print(results_df.to_string(index=False))








