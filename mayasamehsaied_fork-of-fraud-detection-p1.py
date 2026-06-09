import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



!pip install -Uq scikit-learn==1.3.0



from sklearn.metrics import roc_auc_score, f1_score, classification_report


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


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')



train_transaction.head()


train_identity.head()


# تحديد المسار
folder = '/kaggle/input/ieee-fraud-detection'

# تحميل ملفات test
test_trans = pd.read_csv(f'{folder}/test_transaction.csv')
test_id = pd.read_csv(f'{folder}/test_identity.csv')

# عرض شكل البيانات
test_trans.head(), test_id.head()


# نظرة سريعة على حجم البيانات
print("Transaction shape:", test_trans.shape)
print("Identity shape:", test_id.shape)

# نسبة القيم المفقودة في test_transaction
missing_trans = test_trans.isnull().mean().sort_values(ascending=False)
print("\nMissing values in test_transaction:")
print(missing_trans[missing_trans > 0][:10])  # أول 10 أعمدة فيها missing

# نسبة القيم المفقودة في test_identity
missing_id = test_id.isnull().mean().sort_values(ascending=False)
print("\nMissing values in test_identity:")
print(missing_id[missing_id > 0][:10])  # أول 10 أعمدة فيها missing


# حذف الأعمدة اللي فيها missing values بنسبة أكتر من 90%
missing_ratio_trans = test_trans.isnull().mean()
missing_ratio_id = test_id.isnull().mean()

cols_to_drop_trans = missing_ratio_trans[missing_ratio_trans > 0.9].index
cols_to_drop_id = missing_ratio_id[missing_ratio_id > 0.9].index

test_trans = test_trans.drop(columns=cols_to_drop_trans)
test_id = test_id.drop(columns=cols_to_drop_id)

print("Remaining columns in test_transaction:", len(test_trans.columns))
print("Remaining columns in test_identity:", len(test_id.columns))


# Fill NaNs in test_transaction
for col in test_trans.columns:
    if test_trans[col].dtype == 'object':
        test_trans[col] = test_trans[col].fillna('missing')
    else:
        test_trans[col] = test_trans[col].fillna(test_trans[col].mean())

# Fill NaNs in test_identity
for col in test_id.columns:
    if test_id[col].dtype == 'object':
        test_id[col] = test_id[col].fillna('missing')
    else:
        test_id[col] = test_id[col].fillna(test_id[col].mean())

print("✅ Missing values filled successfully.")


# first, we will explore the data for el training
# missing values
missing_values_count = train_transaction.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_transaction.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)


train = train_transaction.merge(train_identity, on='TransactionID', how='left')


test = test_trans.merge(test_id, on='TransactionID', how='left')



missing_percent = train.isnull().sum() / train.shape[0]
columns_to_drop = missing_percent[missing_percent > 0.90].index
train.drop(columns=columns_to_drop, axis=1, inplace=True)



# Fill remaining missing values with median
train_transaction = train_transaction.fillna(train_transaction.median(numeric_only=True))


del missing_values_count, total_cells, total_missing



def coorelation_analysis(cols,title='Coorelation Analysis',size=(12,12)):
    cols = sorted(cols)
    fig,axes = plt.subplots(1,1,figsize=size)
    df_corr = train_transaction[cols].corr()
    sns.heatmap(df_corr,annot=True,cmap='RdBu_r')
    axes.title.set_text(title)
    plt.show()


cols = ['V3', 'V9', 'V5', 'V11', 'V10', 'V8', 'V7', 'V6', 'V4', 'V2', 'V1']
coorelation_analysis(cols,title='Coorelation Analysis: V1-V11')


# Transaction Amount Distribution by Fraud (histogram) (Helps identify suspicious amount ranges.)
plt.figure(figsize=(10,6))
sns.histplot(data=train_transaction, x='TransactionAmt', hue='isFraud', bins=100, kde=True, log_scale=True)
plt.title("Transaction Amount Distribution by Fraud")
plt.xlabel("Transaction Amount (log scale)")
plt.ylabel("Frequency")
plt.show()



# Helps catch time-based fraud patterns.
train_transaction['TransactionDT_Hour'] = (train_transaction['TransactionDT'] // 3600) % 24
plt.figure(figsize=(10,6))
sns.histplot(data=train_transaction, x='TransactionDT_Hour', hue='isFraud', bins=24, kde=False)
plt.title("Hourly Transaction Frequency by Fraud")
plt.xlabel("Hour of Day")
plt.ylabel("Transaction Count")
plt.show()



numeric_cols = train_transaction.select_dtypes(include=['number']).drop(columns=['isFraud']).corrwith(train_transaction['isFraud'])
top_features = numeric_cols.abs().sort_values(ascending=False).head(10).index

plt.figure(figsize=(10,8))
sns.heatmap(train_transaction[top_features.tolist() + ['isFraud']].corr(), annot=True, cmap='coolwarm')
plt.title("Top Correlated Features with Fraud")
plt.show()



 train = train.drop(['TransactionID', 'TransactionDT'], axis=1)  # Drop ID cols
train.fillna(-999, inplace=True)  # Fill missing values


test = test.drop(['TransactionID', 'TransactionDT'], axis=1)
test.fillna(-999, inplace=True)


from sklearn.preprocessing import LabelEncoder

# 1. Encode train and store encoders
cat_cols = train.select_dtypes(include='object').columns
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    encoders[col] = le 

# 2. Apply encoders to test
for col in cat_cols:
    if col in test.columns:
        test[col] = test[col].astype(str)
    else:
        test[col] = 'unknown'

    if col in encoders:
        le = encoders[col]

        # Handle unseen labels
        test[col] = test[col].apply(lambda x: x if x in le.classes_ else 'unknown')

        if 'unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'unknown')

        test[col] = le.transform(test[col])
    else:
        test[col] = -999



from sklearn.preprocessing import LabelEncoder

def label_encode(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    return df




X = train.drop('isFraud', axis=1)
features = X.columns.tolist()

y = train['isFraud']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#  Random Forest model with balanced class weights to fix el data
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_preds_prob = rf_model.predict_proba(X_val)[:, 1]
rf_preds = rf_model.predict(X_val)


print("Random Forest ROC AUC:", roc_auc_score(y_val, rf_preds_prob))
print("Random Forest F1 Score:", f1_score(y_val, rf_preds))


# we will use lightgbm to boost el model 
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds_prob = lgb_model.predict_proba(X_val)[:, 1]
lgb_preds = lgb_model.predict(X_val)


params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 10,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbosity': -1}


param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
    'max_depth': [5, 10],
    'min_child_samples': [20, 50]
}



print("LightGBM ROC AUC:", roc_auc_score(y_val, lgb_preds_prob))
print("LightGBM F1 Score:", f1_score(y_val, lgb_preds))


# next we will average the predicted prob. to reduce individual weaknesses and improve overall performance 
combined_preds_prob = (rf_preds_prob + lgb_preds_prob) / 2
combined_preds = (combined_preds_prob > 0.5).astype(int)


# These are arrays of predicted probabilities from each model
rf_preds_prob = rf_model.predict_proba(X_val)[:, 1]
lgb_preds_prob = lgb_model.predict_proba(X_val)[:, 1]



combined_preds_prob = (rf_preds_prob + lgb_preds_prob) / 2



# لو اكبر من ال 0.5 يبقى fraud لو اقل تبقى 0
combined_preds = (combined_preds_prob > 0.5).astype(int)


# evaluation finally
roc_auc_score(y_val, combined_preds_prob)
f1_score(y_val, combined_preds)




test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

# Step 2: Merge on TransactionID
test = test_transaction.merge(test_identity, how='left', on='TransactionID')

 # Add missing columns (like id_01 to id_38) if they don't exist in test
for col in features:
    if col not in test.columns:
        test[col] = -999  # Or any default fill value used in training



# Add missing columns
missing_cols = set(features) - set(test.columns)
for col in missing_cols:
    test[col] = -999  

# Remove extra columns
extra_cols = set(test.columns) - set(features)
test = test.drop(columns=extra_cols)

# Ensure correct order
test = test[features]



X_test = test[features]
X_test = X_test.fillna(-999)



# Encode training data
X = label_encode(X)

# Encode test data (same features)
X_test = label_encode(X_test)




rf_model.fit(X, y)



X_test = label_encode(X_test)
rf_model.predict_proba(X_test)



# Get predicted probabilities
rf_test_probs = rf_model.predict_proba(X_test)[:, 1]
lgb_test_probs = lgb_model.predict_proba(X_test)[:, 1]

combined_test_probs = (rf_test_probs + lgb_test_probs) / 2




pd.DataFrame({'combined_probs': combined_test_probs[:10]})



rf_test_probs = rf_model.predict_proba(X_test)[:, 1]
lgb_test_probs = lgb_model.predict_proba(X_test)[:, 1]
combined_test_probs = (rf_test_probs + lgb_test_probs) / 2

submission = pd.DataFrame({
    'TransactionID': test_transaction['TransactionID'],
    'isFraud': combined_test_probs
})



submission.head()



submission.to_csv("submission.csv", index=False)



import matplotlib.pyplot as plt

plt.hist(combined_test_probs, bins=50)
plt.title("Predicted Fraud Probabilities")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()


