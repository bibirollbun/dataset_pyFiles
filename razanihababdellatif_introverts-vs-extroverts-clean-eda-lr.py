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


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
sns.set_palette("Set2")


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
dftrain = train.copy()
dftest = test.copy()


train


test


print("train shape:", train.shape)
print("test shape:", test.shape)



# ðŸ“Š Quick Data Representation

print("Training Data Preview (First 5 Rows):")
print(train.head())

print("Test Data Preview (First 5 Rows):")
print(test.head())


# ðŸ“‹ Dataset Information Summary

print("Training Data Info:")
train.info()

print("Test Data Info:")
test.info()


train.drop(columns=['Stage_fear','Drained_after_socializing']).describe()




test.drop(columns=['Stage_fear','Drained_after_socializing']).describe()


#  Checking for Missing Values in Each Dataset

print("Missing Values in Training Data:")
print(train.isna().sum())

print("Missing Values in Test Data:")
print(test.isna().sum())


#  Handle Missing Values 
# Fill numeric missing values with median
numeric_cols = ['Time_spent_Alone','Social_event_attendance',
                'Going_outside','Friends_circle_size','Post_frequency']
for col in numeric_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)


# Fill Stage_fear & Drained_after_socializing categorical missing values with mode
for col in ['Stage_fear','Drained_after_socializing']:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)


#  Encode Categorical Variables 
yes_no_cols = ['Stage_fear', 'Drained_after_socializing']
for col in yes_no_cols:
    train[col] = train[col].map({'Yes':1,'No':0})
    test[col] = test[col].map({'Yes':1,'No':0})


# Encode target
train['Personality'] = train['Personality'].map({'Introvert':0,'Extrovert':1})


print(train.isna().sum())



print(test.isna().sum())


# Check number of duplicate rows
duplicate_count = train.duplicated().sum()
print(f"Duplicate rows in train: {duplicate_count}")

# If duplicates exist
if duplicate_count > 0:
    train = train.drop_duplicates()
    print("Duplicates dropped!")


train.duplicated().sum()


# Check number of duplicate rows
duplicate_count = test.duplicated().sum()
print(f"Duplicate rows in train: {duplicate_count}")

# If duplicates exist
if duplicate_count > 0:
    test = test.drop_duplicates()
    print("Duplicates dropped!")


test.duplicated().sum()


# Univariate Analysis 
plt.figure(figsize=(10,5))
sns.countplot(x='Personality', data=train)
plt.title("Personality Distribution")
plt.show()


# Distribution plots for numeric features
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(train[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# Bivariate Analysis 
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='Personality', y=col, data=train)
    plt.title(f"{col} vs Personality")
    plt.show()


# Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


#  Split Data 
X = train.drop('Personality', axis=1)
y = train['Personality']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



#  Models 
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}


for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"{name} Accuracy: {acc:.4f}")


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)


val_preds = log_model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc:.4f}")


# Confusion Matrix
cm = confusion_matrix(y_val, val_preds)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Introvert','Extrovert'], yticklabels=['Introvert','Extrovert'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


# Classification Report
print("\nClassification Report:")
print(classification_report(y_val, val_preds, target_names=['Introvert','Extrovert']))



# Cross-validation for Robustness 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(log_model, X, y, cv=cv, scoring='accuracy')
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")



#  Train on Full Data & Predict Test
log_model.fit(X, y)
test_preds = log_model.predict(test)


submission = test[['id']].copy()
submission['Personality'] = test_preds



# Convert numeric predictions back to categorical
submission = pd.DataFrame({
    "id": test['id'],
    "Personality": np.where(test_preds == 1, "Extrovert", "Introvert")
})

submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")
print(submission.head())



X = train.drop('Personality', axis=1)
y = train['Personality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)


rf = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=20, scoring='accuracy', cv=5, random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
print("Best RF Params:", random_search.best_params_)



train_acc = accuracy_score(y_train, best_rf.predict(X_train))
val_acc = accuracy_score(y_val, best_rf.predict(X_val))
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_score = cross_val_score(best_rf, X_scaled, y, cv=cv, scoring='accuracy').mean()
print(f"Cross-validation Accuracy: {cv_score:.4f}")


