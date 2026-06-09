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
import math
import warnings
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,  VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
df_save = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
df_train


df_test


X = df_train.drop('diagnosed_diabetes', axis=1)
y= df_train['diagnosed_diabetes']


columns_to_drop = ['id', 'screen_time_hours_per_day', 'income_level', 'education_level', 'employment_status']
df_train = df_train.drop(columns=columns_to_drop)

# Result check karo
print(df_train.head())


df_train.isnull().sum()


df_train.describe()


df_train.head()


from sklearn.preprocessing import LabelEncoder
columns = ['gender' , 'ethnicity' , 'education_level' , 'income_level', 'smoking_status','employment_status']
le = LabelEncoder()
for col in df_train.columns:
    df_train[col] = le.fit_transform(df_train[col])
df_train


df_train.hist(figsize=(16, 12), bins=12, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.show()


import seaborn as sns
# ---------------------------
# Step 4: Target variable analysis
# ---------------------------
target_col = 'diagnosed_diabetes'
sns.countplot(x=target_col, data=df_train)
plt.title('Distribution of Diagnosed Diabetes')
plt.show()


df_test


df_test.isnull().sum()


df_test


X = df_train.drop('diagnosed_diabetes', axis=1)
y= df_train['diagnosed_diabetes']


X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.2, random_state=42)


print(X_train.isna().sum())
print(X_test.isna().sum())



print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)



model_E = ExtraTreesClassifier()
model_E.fit(X_train, y_train)


importances = model_E.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(12, 8))
plt.title('Feature Importances (Extra Trees Classifier)', fontsize=14)
plt.barh(range(len(indices)), importances[indices], color='purple')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,ConfusionMatrixDisplay,classification_report
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary') 
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import xgboost as xgb

model_XGB = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model_XGB.fit(X_train, y_train)
y_pred = model_XGB.predict(X_test)



y_pred = model_XGB.predict(X_test)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,ConfusionMatrixDisplay,classification_report
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary') 
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


features = X_train.columns 
import lightgbm as lgb

model_lgb = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model_lgb.fit(X_train, y_train)
y_pred = model_lgb.predict(X_test)



y_pred = model_lgb.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary') 
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


X_test = df_test[features]  # same columns, same order

# Handle missing values (important!)
X_test = X_test.fillna(X_train.mean())  # safe approach



# Drop same columns as training
columns_to_drop = ['id', 'screen_time_hours_per_day', 'income_level', 'education_level', 'employment_status']
X_test_final = df_test.drop(columns=columns_to_drop)

# Encode categorical columns exactly as training
for col in ['gender', 'ethnicity', 'smoking_status']:
    X_test_final[col] = le.fit_transform(X_test_final[col])

# Fill any potential NaNs (safe)
X_test_final = X_test_final.fillna(X_train.mean())



# Using the best model (LightGBM in your case)
y_pred_final = model_lgb.predict(X_test_final)



submission = pd.DataFrame({
    "id": df_test["id"],
    "diagnosed_diabetes": y_pred_final
})

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(submission.shape)





