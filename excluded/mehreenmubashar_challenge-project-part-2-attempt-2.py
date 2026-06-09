import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Load the data
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
pos_cash_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
credit_card_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
installments_payments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
application_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

# Bureau Aggregation
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'DAYS_CREDIT': 'mean', 
    'CREDIT_DAY_OVERDUE': 'sum', 
    'AMT_CREDIT_SUM': 'sum', 
    'AMT_CREDIT_SUM_DEBT': 'sum'
}).reset_index()

# Bureau Balance Aggregation
bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].nunique().reset_index()
bureau_balance_agg.rename(columns={'STATUS': 'NUM_UNIQUE_STATUS'}, inplace=True)

# Merge bureau_balance with bureau
bureau = bureau.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left')

# Aggregate bureau_balance information per SK_ID_CURR
bureau_final_agg = bureau.groupby('SK_ID_CURR').agg({'NUM_UNIQUE_STATUS': 'sum'}).reset_index()

# POS_CASH_balance Aggregation
pos_cash_balance_agg = pos_cash_balance.groupby('SK_ID_CURR').agg({
    'MONTHS_BALANCE': 'count', 
    'CNT_INSTALMENT_FUTURE': 'sum'
}).reset_index()

# Credit Card Balance Aggregation
credit_card_balance_agg = credit_card_balance.groupby('SK_ID_CURR').agg({
    'AMT_BALANCE': 'sum', 
    'AMT_CREDIT_LIMIT_ACTUAL': 'mean'
}).reset_index()

# Installments Payments Aggregation
installments_payments_agg = installments_payments.groupby('SK_ID_CURR').agg({
    'AMT_PAYMENT': 'sum', 
    'AMT_INSTALMENT': 'sum'
}).reset_index()

# Aggregating useful numerical features from bureau
useful_bureau_features = ['SK_ID_CURR', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'AMT_CREDIT_SUM']

bureau_filtered = bureau[useful_bureau_features].groupby('SK_ID_CURR').agg(['mean']).reset_index()
bureau_filtered.columns = ['SK_ID_CURR', 'bureau_DAYS_CREDIT_mean', 'bureau_DAYS_CREDIT_ENDDATE_mean', 'bureau_AMT_CREDIT_SUM_mean']

# Counting previous loans
previous_loan_counts = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count().reset_index()
previous_loan_counts.rename(columns={'SK_ID_BUREAU': 'previous_loan_counts'}, inplace=True)

# Merge all aggregated data into a single DataFrame
data = application_train[['SK_ID_CURR', 'TARGET']].merge(bureau_agg, on='SK_ID_CURR', how='left') \
    .merge(bureau_final_agg, on='SK_ID_CURR', how='left') \
    .merge(pos_cash_balance_agg, on='SK_ID_CURR', how='left') \
    .merge(credit_card_balance_agg, on='SK_ID_CURR', how='left') \
    .merge(installments_payments_agg, on='SK_ID_CURR', how='left') \
    .merge(bureau_filtered, on='SK_ID_CURR', how='left') \
    .merge(previous_loan_counts, on='SK_ID_CURR', how='left')

# Handling missing values
data.fillna(0, inplace=True)

# Prepare the feature matrix and target vector
X = data.drop(columns=['SK_ID_CURR', 'TARGET'])
y = data['TARGET']

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Model
model = xgb.XGBClassifier(
    objective='binary:logistic', 
    eval_metric='logloss', 
    learning_rate=0.1, 
    max_depth=6, 
    n_estimators=1000, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)

# Train the model
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50, verbose=100)

# Predict on validation set
y_pred_valid = model.predict_proba(X_valid)[:, 1]

# Evaluate the model using AUC
roc_auc = roc_auc_score(y_valid, y_pred_valid)
print(f'ROC AUC: {roc_auc:.4f}')

# Load and process test data
test_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

# Merge test data with aggregated features
test_data = test_data[['SK_ID_CURR']].merge(bureau_agg, on='SK_ID_CURR', how='left') \
    .merge(bureau_final_agg, on='SK_ID_CURR', how='left') \
    .merge(pos_cash_balance_agg, on='SK_ID_CURR', how='left') \
    .merge(credit_card_balance_agg, on='SK_ID_CURR', how='left') \
    .merge(installments_payments_agg, on='SK_ID_CURR', how='left') \
    .merge(bureau_filtered, on='SK_ID_CURR', how='left') \
    .merge(previous_loan_counts, on='SK_ID_CURR', how='left')

# Handle missing values in test data
test_data.fillna(0, inplace=True)

# Prepare test features
X_test = test_data.drop(columns=['SK_ID_CURR'])

# Making predictions for test data
y_pred_test = model.predict_proba(X_test)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'SK_ID_CURR': test_data['SK_ID_CURR'], 'TARGET': y_pred_test})
submission.to_csv('submission.csv', index=False)
print("Submission file saved: submission.csv")


