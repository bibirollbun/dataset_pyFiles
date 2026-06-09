# =====================
#  1. IMPORT LIBRARIES
# =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# If you are in Jupyter/Colab, enable inline plots:
# %matplotlib inline

# Aesthetic setup for Seaborn
sns.set(style='whitegrid')


# =========================
#  2. LOAD THE DATA FRAMES
# =========================
# Adjust these paths to your environment:
app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
prev_app = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
inst_pay = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
credit_card_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')


# ============================================
#  3. CHART #1: app_train + previous_app
# ============================================
# Goal: Visualize the distribution of how many previous loans each client had,
#       and check if there's any relationship to default (TARGET).

# Count the number of previous applications per SK_ID_CURR
prev_counts = prev_app.groupby('SK_ID_CURR').size().reset_index(name='Prev_App_Count')

# Merge with application_train
app_train_merged = app_train.merge(prev_counts, on='SK_ID_CURR', how='left')
# Fill missing with 0 (meaning no previous apps found)
app_train_merged['Prev_App_Count'] = app_train_merged['Prev_App_Count'].fillna(0)

# --- Plot 1: Distribution of the number of previous applications ---
plt.figure(figsize=(10,6))
sns.histplot(data=app_train_merged, x='Prev_App_Count', bins=30, kde=False)
plt.title('Distribution of Number of Previous Applications')
plt.xlabel('Number of Previous Applications')
plt.ylabel('Count')
plt.show()

# --- Plot 2: Average default rate by number of previous applications ---
target_by_prev = app_train_merged.groupby('Prev_App_Count')['TARGET'].mean().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(x='Prev_App_Count', y='TARGET', data=target_by_prev, palette='viridis')
plt.title('Average Default Rate by # of Previous Applications')
plt.xlabel('Number of Previous Applications')
plt.ylabel('Average Default Rate')
plt.show()


# Chart 2: application_train + previous_application + installments_payments
###########################################
# Goal: Compute and visualize the average installment payment ratio per applicant.
# We join installments_payments with previous_application (via SK_ID_PREV), then aggregate by SK_ID_CURR.

# Merge installments payments with previous application info using SK_ID_PREV
prev_inst = prev_app[['SK_ID_PREV', 'SK_ID_CURR']].merge(inst_pay, on='SK_ID_PREV', how='left')

# If the merge creates a duplicate SK_ID_CURR column (e.g., 'SK_ID_CURR_x'), rename it back to 'SK_ID_CURR'
if 'SK_ID_CURR_x' in prev_inst.columns:
    prev_inst = prev_inst.rename(columns={'SK_ID_CURR_x': 'SK_ID_CURR'})

# Calculate the payment ratio for each installment record: AMT_PAYMENT / AMT_INSTALMENT
prev_inst['Payment_Ratio'] = prev_inst['AMT_PAYMENT'] / prev_inst['AMT_INSTALMENT']

# Aggregate: compute the average payment ratio for each applicant (group by SK_ID_CURR)
payment_ratio_agg = prev_inst.groupby('SK_ID_CURR')['Payment_Ratio'].mean().reset_index()

# Merge the aggregated payment ratio back into the main application data
app_train_inst = app_train.merge(payment_ratio_agg, on='SK_ID_CURR', how='left')

# Plot the distribution of the average payment ratio
plt.figure(figsize=(10,6))
sns.histplot(app_train_inst['Payment_Ratio'].dropna(), bins=30, kde=True)
plt.title('Distribution of Average Payment Ratio from Installments')
plt.xlabel('Average Payment Ratio')
plt.ylabel('Count')
plt.show()

# Compare the payment ratio distribution by default status using a boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='TARGET', y='Payment_Ratio', data=app_train_inst)
plt.title('Payment Ratio Distribution by Default Status')
plt.xlabel('Default (TARGET)')
plt.ylabel('Average Payment Ratio')
plt.show()


print(prev_inst.columns.tolist())


# ===============================================================
#  5. CHART #3: app_train + bureau + bureau_balance
# ===============================================================
# Goal: Examine credit bureau records from "bureau" and monthly statuses from "bureau_balance."
#       We can, for example, see how long each bureau record was tracked and compare to default.

# 5.1 Aggregate bureau_balance: count how many monthly entries exist per SK_ID_BUREAU
bb_agg = bureau_balance.groupby('SK_ID_BUREAU').size().reset_index(name='BB_Months_Count')

# 5.2 Merge with the bureau dataframe
bureau_merged = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
bureau_merged['BB_Months_Count'] = bureau_merged['BB_Months_Count'].fillna(0)

# 5.3 Aggregate by SK_ID_CURR: the average number of months monitored (BB_Months_Count)
bureau_agg = bureau_merged.groupby('SK_ID_CURR')['BB_Months_Count'].mean().reset_index()

# 5.4 Merge with main app_train
app_train_bureau = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')

# --- Plot 1: Distribution of the average bureau monitoring months ---
plt.figure(figsize=(10,6))
sns.histplot(app_train_bureau['BB_Months_Count'].dropna(), bins=30, kde=True)
plt.title('Distribution of Avg. # of Months in Bureau Balance')
plt.xlabel('Average # of Months')
plt.ylabel('Count')
plt.show()

# --- Plot 2: Boxplot by TARGET (default) ---
plt.figure(figsize=(8,6))
sns.boxplot(x='TARGET', y='BB_Months_Count', data=app_train_bureau)
plt.title('Bureau Monitoring Duration by Default Status')
plt.xlabel('Default (TARGET)')
plt.ylabel('Average # of Months in Bureau Balance')
plt.show()



# ===============================================================
#  6. CHART #4: app_train + credit_card_balance
# ===============================================================
# Goal: Look at the average credit card balance by client and compare for defaults vs non-defaults.

# 6.1 Aggregate credit card balance: average AMT_BALANCE per SK_ID_CURR
cc_bal_agg = credit_card_balance.groupby('SK_ID_CURR')['AMT_BALANCE'].mean().reset_index()
cc_bal_agg.rename(columns={'AMT_BALANCE':'Avg_Credit_Card_Balance'}, inplace=True)

# 6.2 Merge with the main training data
app_train_cc = app_train.merge(cc_bal_agg, on='SK_ID_CURR', how='left')

# --- Plot 1: Distribution of the average credit card balance ---
plt.figure(figsize=(10,6))
sns.histplot(app_train_cc['Avg_Credit_Card_Balance'].dropna(), bins=30, kde=True)
plt.title('Distribution of Average Credit Card Balance')
plt.xlabel('Average Credit Card Balance')
plt.ylabel('Count')
plt.show()

# --- Plot 2: Boxplot by default status ---
plt.figure(figsize=(8,6))
sns.boxplot(x='TARGET', y='Avg_Credit_Card_Balance', data=app_train_cc)
plt.title('Credit Card Balance by Default Status')
plt.xlabel('Default (TARGET)')
plt.ylabel('Average Credit Card Balance')
plt.show()

print("All 4 visual analytics sections completed successfully!")



# -------------------------------------------------------------------
app_train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
prev_app = pd.read_csv("/kaggle/input/home-credit-default-risk/previous_application.csv")
inst_pay = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")
bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bureau_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")
credit_card_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")

# ======================================================
# 1) PREVIOUS APPLICATION COUNT PER CLIENT
# ======================================================
prev_counts = prev_app.groupby("SK_ID_CURR").size().reset_index(name="Prev_App_Count")

# ======================================================
# 2) AVERAGE INSTALLMENT PAYMENT RATIO
# ======================================================
# Merge prev_app and inst_pay on SK_ID_PREV
prev_inst = prev_app[['SK_ID_PREV', 'SK_ID_CURR']].merge(inst_pay, on='SK_ID_PREV', how='left')

# If there's a duplicate SK_ID_CURR column, rename it
if "SK_ID_CURR_x" in prev_inst.columns:
    prev_inst.rename(columns={"SK_ID_CURR_x": "SK_ID_CURR"}, inplace=True)
if "SK_ID_CURR_y" in prev_inst.columns:
    prev_inst.drop(columns=["SK_ID_CURR_y"], inplace=True)

# Payment ratio for each installment row
prev_inst["Payment_Ratio"] = prev_inst["AMT_PAYMENT"] / prev_inst["AMT_INSTALMENT"]

# Average payment ratio per SK_ID_CURR
inst_ratio_agg = prev_inst.groupby("SK_ID_CURR")["Payment_Ratio"].mean().reset_index()

# Rename column to clarify
inst_ratio_agg.rename(columns={"Payment_Ratio": "Avg_Installment_Payment_Ratio"}, inplace=True)

# ======================================================
# 3) BUREAU + BUREAU_BALANCE (AVG # OF MONTHS)
# ======================================================
# Count how many monthly entries exist for each bureau record
bb_agg = bureau_balance.groupby("SK_ID_BUREAU").size().reset_index(name="BB_Months_Count")

# Merge that back with the bureau data
bureau_merged = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
bureau_merged["BB_Months_Count"] = bureau_merged["BB_Months_Count"].fillna(0)

# Average # of months tracked per SK_ID_CURR
bureau_agg = bureau_merged.groupby("SK_ID_CURR")["BB_Months_Count"].mean().reset_index()
bureau_agg.rename(columns={"BB_Months_Count": "Avg_Bureau_Balance_Months"}, inplace=True)

# ======================================================
# 4) AVERAGE CREDIT CARD BALANCE
# ======================================================
cc_bal_agg = credit_card_balance.groupby("SK_ID_CURR")["AMT_BALANCE"].mean().reset_index()
cc_bal_agg.rename(columns={"AMT_BALANCE": "Avg_Credit_Card_Balance"}, inplace=True)

# ======================================================
# 5) MERGE EVERYTHING INTO A SINGLE TABLE
# ======================================================
# Start from application_train (since that includes TARGET)
df_final = app_train[["SK_ID_CURR", "TARGET"]].copy()

# Add each aggregate
df_final = df_final.merge(prev_counts, on="SK_ID_CURR", how="left")
df_final = df_final.merge(inst_ratio_agg, on="SK_ID_CURR", how="left")
df_final = df_final.merge(bureau_agg, on="SK_ID_CURR", how="left")
df_final = df_final.merge(cc_bal_agg, on="SK_ID_CURR", how="left")

# Optionally fill NAs with zero for these aggregates if you prefer
# df_final[["Prev_App_Count","Avg_Installment_Payment_Ratio","Avg_Bureau_Balance_Months","Avg_Credit_Card_Balance"]] = \
#     df_final[["Prev_App_Count","Avg_Installment_Payment_Ratio","Avg_Bureau_Balance_Months","Avg_Credit_Card_Balance"]].fillna(0)

# ======================================================
# 6) SAVE TO A SINGLE CSV
# ======================================================
df_final.to_csv("final_table.csv", index=False)
print("Single CSV file 'final_table.csv' created successfully!")


