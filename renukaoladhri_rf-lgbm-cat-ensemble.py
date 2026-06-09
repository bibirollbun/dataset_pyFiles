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


df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


df.head()


df.info()


df.shape


df.isnull().sum()


print("Original shape:", df.shape)
df = df.dropna()
print("New shape after dropping nulls:", df.shape)



df.isna().sum()


df.info()


df['Stage_fear'].value_counts()


df['Drained_after_socializing'].value_counts()


df['Personality'].value_counts()


df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})



df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})



df['Personality'].value_counts()


df = df.drop(columns=['id'])



df.info()


correlations = df.corr(numeric_only=True)['Personality'].drop('Personality').sort_values(ascending=False)

# Print
print("ğŸ”� Correlation of each feature with Personality (1=Extrovert, 0=Introvert):")
print(correlations)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

X = df.drop(columns=['Personality'])
y = df['Personality']

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Compute scale_pos_weight for XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Define models with imbalance handling
models = {
    "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              scale_pos_weight=scale_pos_weight, random_state=42),
    "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
    "CatBoost": CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n==================== {name} ====================")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)

    acc = accuracy_score(y_val, y_pred)
    print(f"âœ… Accuracy: {acc:.4f}")
    print("ğŸ“Š Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Introvert", "Extrovert"]))
    print("ğŸ“‰ Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# Clean + encode train
train_df = train_df.dropna()
binary_map = {'Yes': 1, 'No': 0}
train_df['Stage_fear'] = train_df['Stage_fear'].map(binary_map)
train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map(binary_map)
train_df['Personality'] = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# Encode test
test_df['Stage_fear'] = test_df['Stage_fear'].map(binary_map)
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map(binary_map)

# Split features/target
X = train_df.drop(columns=['id', 'Personality'])
y = train_df['Personality']
X_test = test_df.drop(columns=['id'])
test_ids = test_df['id']

# Split train/val for metrics
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute missing values (robust)
imputer = SimpleImputer(strategy="mean")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_full_imp = imputer.fit_transform(X)
X_test_imp = imputer.transform(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_full_scaled = scaler.fit_transform(X_full_imp)
X_test_scaled = scaler.transform(X_test_imp)

# Define top 3 models
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
lgbm = LGBMClassifier(class_weight='balanced', random_state=42)
cat = CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=42)

# Soft voting ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('lgbm', lgbm),
        ('cat', cat)
    ],
    voting='soft'
)

# Train on training split, evaluate on validation
ensemble.fit(X_train_scaled, y_train)
val_preds = ensemble.predict(X_val_scaled)

# âœ… Print evaluation metrics
print("\n==================== ENSEMBLE VALIDATION METRICS ====================")
acc = accuracy_score(y_val, val_preds)
print(f"âœ… Accuracy: {acc:.4f}")
print("ğŸ“Š Classification Report:")
print(classification_report(y_val, val_preds, target_names=["Introvert", "Extrovert"]))
print("ğŸ“‰ Confusion Matrix:")
print(confusion_matrix(y_val, val_preds))

# âœ… Retrain on full data for test submission
ensemble.fit(X_full_scaled, y)

# Predict on test
test_probs = ensemble.predict_proba(X_test_scaled)
test_preds = (test_probs[:, 1] >= 0.5).astype(int)

# Prepare submission
label_map = {0: 'Introvert', 1: 'Extrovert'}
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': [label_map[p] for p in test_preds]
})
submission.to_csv("submission.csv", index=False)
print("\nâœ… Submission saved as submission.csv")



sub_df=pd.read_csv('/kaggle/working/submission.csv')


sub_df.head()




