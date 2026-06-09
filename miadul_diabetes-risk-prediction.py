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


# Cell 1 — imports and basic setup
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


# calibration
from sklearn.calibration import CalibratedClassifierCV


# reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Cell 2 — load data
DATA_DIR = Path('/kaggle/input/playground-series-s5e12') 
train = pd.read_csv(DATA_DIR / '/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv(DATA_DIR / '/kaggle/input/playground-series-s5e12/test.csv')
sample_submission = pd.read_csv(DATA_DIR / '/kaggle/input/playground-series-s5e12/sample_submission.csv')


print('Train shape:', train.shape)
print('Test shape:', test.shape)
train.head()


# Cell 3 — quick checks
print(train.columns.tolist())
print('\nTarget distribution:')
print(train['diagnosed_diabetes'].value_counts(normalize=True))


# Basic numeric summary
train.describe().T


# Cell 4 — missing values
miss_train = train.isnull().mean().sort_values(ascending=False)
miss_test = test.isnull().mean().sort_values(ascending=False)
print('Top missing in train:\n', miss_train.head(15))
print('\nTop missing in test:\n', miss_test.head(15))


# Cell 5 — simple visualizations for important features
numeric_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ('id','diagnosed_diabetes')]


plt.figure(figsize=(12,6))
train[numeric_cols].hist(bins=30, figsize=(14,10))
plt.tight_layout()



target_col = 'diagnosed_diabetes' if 'diagnosed_diabetes' in train.columns else None

numeric_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = train.select_dtypes(include=['object','category']).columns.tolist()

if target_col in numeric_cols:
    numeric_cols.remove(target_col)

print("Numerical Columns:", numeric_cols)
print("Categorical Columns:", cat_cols)



for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=train[col])
    plt.title(f'Count Plot - {col}')
    plt.xticks(rotation=45)
    plt.show()



for col in cat_cols:
    plt.figure(figsize=(6,6))
    train[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'Pie Chart - {col}')
    plt.ylabel("")
    plt.show()



for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.kdeplot(train[col], fill=True)
    plt.title(f'KDE Plot - {col}')
    plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

train.head()



target_col = 'diagnosed_diabetes'

cat_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status'
]

num_cols = [col for col in train.columns if col not in cat_cols + [target_col]]

print("Categorical:", cat_cols)
print("Numerical:", num_cols)



label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

print("✅ Encoding completed")



scaler = MinMaxScaler()

train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

print("✅ Scaling completed")



X = train.drop(columns=[target_col])
y = train[target_col]

X_test_final = test.copy()

print("X shape:", X.shape)
print("y shape:", y.shape)



X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_val)

print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print(classification_report(y_val, y_pred_lr))



rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_val)

print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))



final_model = rf_model  # change to lr_model if needed
final_model.fit(X, y)



test_predictions = final_model.predict_proba(X_test_final)[:,1]



submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_predictions
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv file created")



cm = confusion_matrix(y_val, y_pred_rf)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


