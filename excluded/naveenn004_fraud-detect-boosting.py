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


train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")

train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

train = train_transaction.merge(train_identity, on="TransactionID", how="left")
test = test_transaction.merge(test_identity, on="TransactionID", how="left")

print("Train Shape:",train.shape)
print("Test Shape:",test.shape)


train_transaction.shape


train_identity.shape


train_transaction.info()


train_transaction.select_dtypes("object").info()


train_identity.info()


train_identity.select_dtypes("object").columns


train.head()


test.head()


train.info()


train.isna().sum().sort_values(ascending=False).head(20)


train["isFraud"].value_counts()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(5,5))
sns.countplot(x="isFraud", data=train, palette="coolwarm")
plt.title("Fraudulent vs. Non-Fraudulent Transactions")
plt.xlabel("isFraud (1 = Fraud, 0 = No Fraud)")
plt.show


plt.figure(figsize=(7, 5))
sns.histplot(train['TransactionAmt'], bins=100, kde=True, color='blue')
plt.title("Distribution of Transaction Amount")
plt.show()



# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


missing_values = missing_values_table(train)
missing_values.head(15)


null_cols = [col for col in train.columns if train[col].isna().sum()/train.shape[0] > 0.9]
null_cols_test = [col for col in test.columns if test[col].isna().sum()/test.shape[0] > 0.9]
print(len(null_cols))
print(len(null_cols_test))


ul_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0]>0.9]
ul_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0]>0.9]
print(len(ul_cols))
print(len(ul_cols_test))


onev_cols = [col for col in train.columns if train[col].nunique() <= 1]
onev_cols_test = [col for col in test.columns if test[col].nunique() <=1]
print(onev_cols)
print(onev_cols_test)


cols_to_drop = list(set(null_cols+ul_cols+onev_cols))
cols_to_drop.remove('isFraud')
len(cols_to_drop)


cols_to_drop_test = list(set(null_cols_test+ul_cols_test+onev_cols_test))
len(cols_to_drop_test)


train = train.drop(cols_to_drop, axis =1 )


test = test.drop(cols_to_drop_test, axis=1)


train.shape


test.shape


others = set(train.columns) - set(test.columns)  
others


train_labes = train['isFraud']
train,test = train.align(test, join='inner',axis=1)



train['isFraud'] = train_labes
print("Train shape",train.shape)
print("Test Shape",test.shape)


cat_cols = train.select_dtypes(include=['object', 'category']).columns
cat_cols_test = train.select_dtypes(include=['object','category']).columns
print(f"Categorical Columns: {list(cat_cols)}")



for col in cat_cols:
    print(f"{col}: {train[col].nunique()} unique values")
    print(train[col].unique()[:10])  # Show first 10 unique values
    print("-" * 50)



for col in ['ProductCD','DeviceType']:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='isFraud', data=train, palette="coolwarm")
    plt.title(f"Fraud Distribution by {col}")
    plt.show()



w = [col for col in cat_cols if train[col].nunique() <=2]
train[w].apply(lambda x: x.unique().tolist())


w1 = [col for col in cat_cols_test if train[col].nunique() <= 2]
train[w1].apply(lambda x: x.unique().tolist())


from sklearn.preprocessing import LabelEncoder
for col in w:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


cat_cols = cat_cols.drop(w)


cat_cols_test = cat_cols_test.drop(w1)


cat_cols = cat_cols.drop(['DeviceInfo','P_emaildomain'])


from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    labelencoder = LabelEncoder()
    train[col] = labelencoder.fit_transform(train[col])
    test[col] = labelencoder.transform(test[col])


train.shape


missing_values_table(train).head()


#train = pd.get_dummies(train, columns=cat_cols)
#test = pd.get_dummies(test, columns=cat_cols)


# Group by card1 and compute transaction statistics
train['card1_mean'] = train.groupby('card1')['TransactionAmt'].transform('mean')
train['card1_std'] = train.groupby('card1')['TransactionAmt'].transform('std')

test['card1_mean'] = test.groupby('card1')['TransactionAmt'].transform('mean')
test['card1_std'] = test.groupby('card1')['TransactionAmt'].transform('std')

# Repeat for addr1 (billing address)
train['addr1_mean'] = train.groupby('addr1')['TransactionAmt'].transform('mean')
train['addr1_std'] = train.groupby('addr1')['TransactionAmt'].transform('std')

test['addr1_mean'] = test.groupby('addr1')['TransactionAmt'].transform('mean')
test['addr1_std'] = test.groupby('addr1')['TransactionAmt'].transform('std')



# Convert transaction time to hours
train['Transaction_hour'] = (train['TransactionDT'] // 3600) % 24
test['Transaction_hour'] = (test['TransactionDT'] // 3600) % 24

# Flag for night transactions (higher fraud risk)
train['is_night'] = train['Transaction_hour'].apply(lambda x: 1 if x < 6 else 0)
test['is_night'] = test['Transaction_hour'].apply(lambda x: 1 if x < 6 else 0)



# Count number of times a card appears in the dataset
train['card1_count'] = train['card1'].map(train['card1'].value_counts())
test['card1_count'] = test['card1'].map(test['card1'].value_counts())



# Define high-risk emails
high_risk_emails = ['mail.com', 'rambler.ru', 'yahoo.com']

train['email_risk'] = train['P_emaildomain'].apply(lambda x: 1 if x in high_risk_emails else 0)
test['email_risk'] = test['P_emaildomain'].apply(lambda x: 1 if x in high_risk_emails else 0)



train['DeviceInfo_count'] = train['DeviceInfo'].map(train['DeviceInfo'].value_counts())
test['DeviceInfo_count'] = test['DeviceInfo'].map(test['DeviceInfo'].value_counts())



train.shape


#train = pd.get_dummies(train, columns = ['DeviceInfo','P_emaildomain'])
#test = pd.get_dummies(test, columns = ['DeviceInfo','P_emaildomain'])


train['P_emaildomain_enc'] = train.groupby('P_emaildomain')['isFraud'].transform('mean')
test['P_emaildomain_enc'] = test['P_emaildomain'].map(train.groupby('P_emaildomain')['isFraud'].mean())

train['DeviceInfo_enc'] = train.groupby('DeviceInfo')['isFraud'].transform('mean')
test['DeviceInfo_enc'] = test['DeviceInfo'].map(train.groupby('DeviceInfo')['isFraud'].mean())

# Drop original categorical columns
train.drop(['P_emaildomain', 'DeviceInfo'], axis=1, inplace=True)
test.drop(['P_emaildomain', 'DeviceInfo'], axis=1, inplace=True)



train_labels = train['isFraud']
train, test = train.align(test, join='inner',axis =1)
train['isFraud'] = train_labels
print(train.shape)
print(test.shape)


print(f"Final Train Shape: {train.shape}, Test Shape: {test.shape}")
train.head()



print(train.shape)


print(test.shape)


X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']


X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)


del train


def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)   

# Cleaning infinite values to NaN
X = clean_inf_nan(X)
X_test = clean_inf_nan(X_test )


X.shape


import gc
gc.collect()


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Split train data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to LightGBM dataset format
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)



X_train.shape


# Define parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'random_state': 42
}

# Train model using early stopping with callbacks (for newer LightGBM versions)
callbacks = [
    lgb.early_stopping(50),
    lgb.log_evaluation(100)
]

model = lgb.train(
    params, 
    train_data, 
    valid_sets=[train_data, val_data],  
    valid_names=['train', 'valid'],  
    num_boost_round=1000, 
    callbacks=callbacks
)

# Evaluate model
y_val_pred = model.predict(X_val)
auc_score = roc_auc_score(y_val, y_val_pred)
print(f'Validation AUC Score: {auc_score:.4f}')


# Get feature importance
feature_importance = model.feature_importance()
feature_names = X_train.columns

# Create DataFrame
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Sort by importance
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(20)  # Show top 20

# Plot
plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title('Top 20 Most Important Features')
plt.show()



test_IDS = test['TransactionID']
test_IDS


test = test.drop(['TransactionDT', 'TransactionID'], axis=1)


# Predict on test data
test_preds = model.predict(test)

# Create submission file for Kaggle
submission = pd.DataFrame({
    'TransactionID': test_IDS,
    'isFraud': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved! ✅")





