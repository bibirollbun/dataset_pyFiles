pip install pandas numpy scikit-learn matplotlib seaborn


import pandas as pd


app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
credit = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')


numeric_cols = credit.select_dtypes(include='number').columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'MONTHS_BALANCE']]
credit_agg = credit.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])[numeric_cols].mean().reset_index()


credit_pivot = credit_agg.pivot(index='SK_ID_CURR', columns='MONTHS_BALANCE', values=numeric_cols)

credit_pivot.columns = [f"{col}_{month}" for col, month in credit_pivot.columns]
credit_pivot.reset_index(inplace=True)


credit_summary = credit.groupby('SK_ID_CURR')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'last']).reset_index()

credit_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in credit_summary.columns]


# Recalculate just to be safe
numeric_cols = credit.select_dtypes(include='number').columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'MONTHS_BALANCE']]

# Proper aggregation
credit_summary = credit.groupby('SK_ID_CURR')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'last'])

# Flatten the MultiIndex column names
credit_summary.columns = ['_'.join([col[0], col[1]]) for col in credit_summary.columns]
credit_summary.reset_index(inplace=True)  # <- this brings back SK_ID_CURR as a column



# Merge with application data
train_merged = app_train.merge(credit_summary, how='left', on='SK_ID_CURR')
test_merged = app_test.merge(credit_summary, how='left', on='SK_ID_CURR')



# Extract baseline data
baseline = credit[credit['MONTHS_BALANCE'] == -12]
baseline_agg = baseline.groupby('SK_ID_CURR')[numeric_cols].mean().reset_index()



import seaborn as sns
import matplotlib.pyplot as plt

# Compare a sample feature across two time points
feature = 'AMT_BALANCE'

plt.figure(figsize=(10, 5))
sns.kdeplot(credit[credit['MONTHS_BALANCE'] == -12][feature], label='-12 Months')
sns.kdeplot(credit[credit['MONTHS_BALANCE'] == -1][feature], label='-1 Month')
plt.title(f'Distribution of {feature} Over Time')
plt.legend()
plt.show()



import pandas as pd

app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
print("Rows:", app_train.shape[0])
print("Columns:", app_train.shape[1])
print("\nColumns:\n", app_train.columns.tolist())
print("\nTarget value counts:\n", app_train['TARGET'].value_counts())




