import pandas as pd
import numpy as np

# Load CSV Files
application_train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
previous_application = pd.read_csv("/kaggle/input/home-credit-default-risk/previous_application.csv")
installments_payments = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")
bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bureau_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")
credit_card_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")



# Merge application_train with previous_application using SK_ID_CURR
df1 = application_train[['SK_ID_CURR', 'TARGET']].merge(
    previous_application[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']], on='SK_ID_CURR', how='left')

# Count total and approved applications
df1_grouped = df1.groupby('NAME_CONTRACT_STATUS')['TARGET'].value_counts(normalize=True).unstack().reset_index()
df1_grouped.columns = ['NAME_CONTRACT_STATUS', 'Repaid (0)', 'Defaulted (1)']

# Save to CSV for Tableau
df1_grouped.to_csv("loan_approval_vs_previous_applications.csv", index=False)


# Merge previous applications with installments payments
df2 = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].merge(
    installments_payments[['SK_ID_PREV', 'DAYS_ENTRY_PAYMENT']], on='SK_ID_PREV', how='left')

# Calculate average days late per client
df2_grouped = df2.groupby('SK_ID_CURR')['DAYS_ENTRY_PAYMENT'].mean().reset_index()
df2_grouped.columns = ['SK_ID_CURR', 'Avg_Days_Late']

# Merge with application_train
df2_final = df2_grouped.merge(application_train[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR', how='left')

# Save to CSV for Tableau
df2_final.to_csv("loan_repayment_vs_previous_apps.csv", index=False)


# Merge bureau with bureau_balance
df3 = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].merge(
    bureau_balance[['SK_ID_BUREAU', 'MONTHS_BALANCE']], on='SK_ID_BUREAU', how='left')

# Compute average months of credit history
df3_grouped = df3.groupby('SK_ID_CURR')['MONTHS_BALANCE'].mean().reset_index()
df3_grouped.columns = ['SK_ID_CURR', 'Avg_Credit_History_Months']

# Merge with application_train
df3_final = df3_grouped.merge(application_train[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR', how='left')

# Save to CSV for Tableau
df3_final.to_csv("credit_history_vs_default_rate.csv", index=False)



# Merge application_train with credit_card_balance
df4 = credit_card_balance.groupby('SK_ID_CURR')['AMT_BALANCE'].mean().reset_index()
df4.columns = ['SK_ID_CURR', 'Avg_Credit_Card_Balance']

# Merge with application_train
df4_final = df4.merge(application_train[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR', how='left')

# Save to CSV for Tableau
df4_final.to_csv("credit_card_utilization_vs_default.csv", index=False)


