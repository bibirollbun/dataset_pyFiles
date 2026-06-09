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


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold


# --- Load Datasets ---
print(" Loading datasets...")

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

# --- Merge transaction + identity ---
print(" Merging transaction and identity datasets...")
train = train_transaction.merge(train_identity, on='TransactionID', how='left')
test = test_transaction.merge(test_identity, on='TransactionID', how='left')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# --- Basic EDA: Missing Values Before Preprocessing ---
print("\n Checking Missing Values before preprocessing...")
missing = train.isnull().mean().sort_values(ascending=False)
missing = missing[missing > 0]
plt.figure(figsize=(10, 6))
missing.head(30).plot(kind='barh')
plt.title("Top Missing Values (Before Cleaning)")
plt.show()

# --- Data Type Overview ---
print("\n Data types BEFORE preprocessing:")
print(train.dtypes.value_counts())
print(test.dtypes.value_counts())

# --- Rename columns: '-' to '_' ---
print("\n Replacing '-' with '_' in test columns...")
test.columns = test.columns.str.replace('-', '_', regex=False)

# --- Drop columns with >90% missing values ---
missing_thresh = 0.90
missing_cols = train.isnull().mean()
missing_cols = missing_cols[missing_cols > missing_thresh].index.tolist()

print(f"\n Dropping columns with >90% missing values: {missing_cols}")

train.drop(columns=missing_cols, inplace=True, errors='ignore')
test.drop(columns=missing_cols, inplace=True, errors='ignore')

# --- Drop specific noisy columns manually ---
cols_to_drop = ['id_24', 'id_25', 'id_08', 'id_07', 'id_21', 'id_26', 'id_22', 'id_23', 'id_27']
print(f" Dropping noisy columns: {cols_to_drop}")

train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
test.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# --- Handle missing values ---
print("\n Filling missing values...")
for df in [train, test]:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('missing')  # for categorical
        else:
            df[col] = df[col].fillna(-999)        # for numerical

# --- Label Encoding ---
print("\n Label Encoding categorical columns...")

# Identify object columns
cat_cols_train = train.select_dtypes(include=['object']).columns.tolist()
cat_cols_test = test.select_dtypes(include=['object']).columns.tolist()

label_encoders = {}

for col in cat_cols_train:
    if col in train.columns:
        le = LabelEncoder()
        combined_data = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined_data)
        train[col] = le.transform(train[col].astype(str))
        label_encoders[col] = le

for col in cat_cols_train:
    if col in test.columns:
        le = label_encoders.get(col)
        if le:
            test[col] = test[col].astype(str).apply(lambda x: x if x in le.classes_ else 'missing')
            test[col] = le.transform(test[col])

# --- Final Checks ---
print("\n Preprocessing Done.")
print(f"Train shape after preprocessing: {train.shape}")
print(f"Test shape after preprocessing: {test.shape}")
print("\nTrain Data Types AFTER preprocessing:")
print(train.dtypes.value_counts())
print("\nTest Data Types AFTER preprocessing:")
print(test.dtypes.value_counts())


# --- Set random seed ---
SEED = 42
np.random.seed(SEED)

# --- EDA Functions ---

def plot_class_distribution(train, target_column):
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_column, data=train)
    plt.title('Fraud vs Non-Fraud')
    plt.xlabel('Is Fraud (1=Yes, 0=No)')
    plt.ylabel('Count')
    plt.show()

# --- EDA Execution ---

# Assume train and test are already loaded and preprocessed from your given code

target = 'isFraud'

print("\n Running EDA...")
plot_class_distribution(train, target)






# --- Separate features and target ---
X = train.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])  # drop target and IDs
y = train['isFraud']

# Align test columns
X_test = test.drop(columns=['TransactionID', 'TransactionDT'], errors='ignore')

# --- Split into train/validation ---
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Define XGBoost model with GPU settings ---
model = xgb.XGBClassifier(
    n_estimators=5000,
    learning_rate=0.005,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False,
    tree_method='gpu_hist',  
    predictor='gpu_predictor', 
    random_state=42,
    gamma=1,
    reg_alpha=0.5,
    reg_lambda=0.5,
    n_jobs=-1
)

# --- Train with early stopping ---
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=100,
    verbose=100
)





# --- Predict on validation ---
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"\nValidation Accuracy: {accuracy:.5f}")

# --- Feature Importance Plot ---
xgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.title('Top 20 Feature Importances (Gain)')
plt.show()


# --- Predict on Test ---
test_preds = model.predict_proba(X_test)[:, 1]  # probability of class 1 (fraud)

# --- Prepare Submission ---
submission = pd.DataFrame({
    'TransactionID': test['TransactionID'],
    'isFraud': test_preds
})

# --- Save Submission File ---
submission.to_csv('Xgb_submission_final.csv', index=False)
print("\nSubmission file 'Xgb_submission.csv' created!")


import joblib


joblib.dump(model, 'xgb_classifier_model.joblib')



# Save test set to CSV
X_test.to_csv('processed_test.csv', index=False)


