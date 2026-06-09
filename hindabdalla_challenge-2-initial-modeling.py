import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Load datasets from Kaggle input directory
app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

# Load additional datasets for feature engineering
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
pos_cash = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
credit_card = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
installments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')

# Encode categorical features
for df in [app_train, app_test]:
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))


params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}


# Aggregate numerical features in bureau dataset
def aggregate_bureau(bureau):
    numeric_cols = bureau.select_dtypes(include=['number']).columns.tolist()
    if 'SK_ID_CURR' not in numeric_cols:
        numeric_cols.append('SK_ID_CURR')
    bureau_agg = bureau[numeric_cols].groupby('SK_ID_CURR').agg(['mean', 'sum', 'max', 'min']).reset_index()
    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
    
    print("Bureau Aggregated Columns:", bureau_agg.columns)
    
    if 'SK_ID_CURR' not in bureau_agg.columns:
        print("Error: 'SK_ID_CURR' missing after aggregation. First few rows:")
        print(bureau_agg.head())
        return pd.DataFrame()
    
    return bureau_agg

bureau_features = aggregate_bureau(bureau)
if bureau_features.empty:
    print("Warning: bureau_features is empty. Skipping merge.")
else:
    print("bureau_features columns:", bureau_features.columns)
    app_train = app_train.merge(bureau_features, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(bureau_features, on='SK_ID_CURR', how='left')

# Visualization: Histogram of bureau loan counts per customer
plt.figure(figsize=(10, 6))
sns.histplot(bureau['SK_ID_CURR'].value_counts(), bins=50, kde=True)
plt.title('Histogram of Bureau Loan Counts per Customer')
plt.xlabel('Number of Loans per Customer')
plt.ylabel('Frequency')
plt.show()





numeric_cols = pos_cash.select_dtypes(include=['number']).columns.tolist()
pos_features = pos_cash[numeric_cols].groupby('SK_ID_CURR').agg(['mean']).reset_index()
pos_features.columns = ['POS_' + '_'.join(col).strip() for col in pos_features.columns.values]
if 'POS_SK_ID_CURR' in pos_features.columns:
    pos_features.rename(columns={'POS_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)
    app_train = app_train.merge(pos_features, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(pos_features, on='SK_ID_CURR', how='left')

# Visualization: Bar chart of mean POS cash balance per customer
plt.figure(figsize=(12, 6))
pos_cash_balance_mean = pos_cash.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].mean()
pos_cash_balance_mean.nlargest(20).plot(kind='bar')
plt.title('Mean POS Cash Balance per Customer')
plt.xlabel('Customer ID')
plt.ylabel('Mean POS Cash Balance')
plt.show()


numeric_cols = credit_card.select_dtypes(include=['number']).columns.tolist()
credit_card_features = credit_card[numeric_cols].groupby('SK_ID_CURR').agg(['mean']).reset_index()
credit_card_features.columns = ['CC_' + '_'.join(col).strip() for col in credit_card_features.columns.values]
if 'CC_SK_ID_CURR' in credit_card_features.columns:
    credit_card_features.rename(columns={'CC_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)
    app_train = app_train.merge(credit_card_features, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(credit_card_features, on='SK_ID_CURR', how='left')

# Visualization: Line plot of average credit card balance over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=credit_card.groupby('MONTHS_BALANCE')['AMT_BALANCE'].mean())
plt.title('Average Credit Card Balance Over Time')
plt.xlabel('Months Balance')
plt.ylabel('Average Balance')
plt.show()


numeric_cols = installments.select_dtypes(include=['number']).columns.tolist()
installments_features = installments[numeric_cols].groupby('SK_ID_CURR').agg(['mean']).reset_index()
installments_features.columns = ['INST_' + '_'.join(col).strip() for col in installments_features.columns.values]
if 'INST_SK_ID_CURR' in installments_features.columns:
    installments_features.rename(columns={'INST_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)
    app_train = app_train.merge(installments_features, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(installments_features, on='SK_ID_CURR', how='left')


X = app_train.drop(columns=['SK_ID_CURR', 'TARGET'])
y = app_train['TARGET']
X_test = app_test.drop(columns=['SK_ID_CURR'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], callbacks=[lgb.early_stopping(50)])

# Make predictions
preds = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({'SK_ID_CURR': app_test['SK_ID_CURR'], 'TARGET': preds})
submission.to_csv('submission.csv', index=False)

# Visualization: Histogram of prediction probabilities
plt.figure(figsize=(10,6))
sns.histplot(preds, bins=50, kde=True)
plt.title('Distribution of Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()

print("Submission file saved!")




