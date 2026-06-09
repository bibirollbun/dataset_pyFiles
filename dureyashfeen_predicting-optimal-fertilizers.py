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


# ğŸ“¦ Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ğŸ“‚ Step 2: Load Datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
print("Train shape:", train.shape)
print("Test shape:", test.shape)


# ğŸ§¼ Step 3: Unified Preprocessing Function (Train + Test)

def preprocess_train_test(train_df, test_df, scale_data=True, encode_target=True):
    train = train_df.copy()
    test = test_df.copy()
    
    # Drop ID column
    if 'id' in train.columns:
        train_ids = train.pop('id')
    else:
        train_ids = None
    if 'id' in test.columns:
        test_ids = test.pop('id')
    else:
        test_ids = None

    # Separate target
    target_col = 'Fertilizer Name'
    y = train[target_col]
    train = train.drop(columns=[target_col])

    # Combine for consistent encoding
    combined = pd.concat([train, test], axis=0)

    # Identify columns
    categorical_cols = ['Soil Type', 'Crop Type']
    numeric_cols = combined.drop(columns=categorical_cols).columns

    # Impute
    num_imputer = SimpleImputer(strategy='median')
    combined[numeric_cols] = num_imputer.fit_transform(combined[numeric_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    combined[categorical_cols] = cat_imputer.fit_transform(combined[categorical_cols])

    # One-hot encoding
    combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

    # Scale
    if scale_data:
        scaler = StandardScaler()
        combined[numeric_cols] = scaler.fit_transform(combined[numeric_cols])

    # Split back
    X_train = combined.iloc[:len(train_df)]
    X_test = combined.iloc[len(train_df):]

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y) if encode_target else y

    return X_train, y_encoded, X_test, label_encoder, test_ids


# ğŸ› ï¸� Step 4: Preprocess Data
X_train, y_train, X_test, le, test_ids = preprocess_train_test(train, test)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ğŸ”€ Step 5: Split for Validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


# ğŸ¤– Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)

y_pred = model.predict(X_val)


# ğŸ“Š Feature Importance (Random Forest)
importances = model.feature_importances_
feature_names = X_train.columns

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='viridis')
plt.title("Top 15 Feature Importances - Random Forest")
plt.tight_layout()
plt.show()


# Reload raw train data for label visualizations
train_raw = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")

# Crop Type distribution
plt.figure(figsize=(10,4))
sns.countplot(data=train_raw, x='Crop Type', order=train_raw['Crop Type'].value_counts().index)
plt.title("Crop Type Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Soil Type distribution
plt.figure(figsize=(10,4))
sns.countplot(data=train_raw, x='Soil Type', order=train_raw['Soil Type'].value_counts().index)
plt.title("Soil Type Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# âš¡ Optional: XGBoost Model (often outperforms RF)
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_tr, y_tr)
xgb_pred = xgb_model.predict(X_val)

# Evaluation
print("XGBoost Accuracy:", accuracy_score(y_val, xgb_pred))
print("\nClassification Report (XGBoost):\n", classification_report(y_val, xgb_pred))



xgb_test_preds = xgb_model.predict(X_test)
xgb_decoded_preds = le.inverse_transform(xgb_test_preds)

# Submission from XGBoost
xgb_submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': xgb_decoded_preds
})
xgb_submission.to_csv("xgb_fertilizer_submission.csv", index=False)
print("XGBoost submission saved as xgb_fertilizer_submission.csv")


# ğŸ“ˆ Step 7: Evaluation
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# âœ… Step 8: Predict on Test Set
test_preds = model.predict(X_test)
decoded_preds = le.inverse_transform(test_preds)

# Prepare submission
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': decoded_preds
})
submission.head()


# ğŸ’¾ Step 9: Save Submission File
submission.to_csv("fertilizer_submission.csv", index=False)
print("Submission file saved as fertilizer_submission.csv")

