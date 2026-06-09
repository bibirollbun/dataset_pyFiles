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


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import warnings
warnings.filterwarnings('ignore')

train= pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
test= pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
previous= pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
bureau= pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_balance= pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
pos_cash= pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
cc_balance= pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
installments= pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')


print(train.shape)
print(test.shape)
print(previous.shape)
print(bureau.shape)
print(bureau_balance.shape)
print(pos_cash.shape)
print(cc_balance.shape)
print(installments.shape)


for col in train.columns:
    print(col + ": " + str(train[col].dtype))


for col in test.columns:
    print(col + ": " + str(test[col].dtype))


for col in bureau.columns:
    print(col + ": " + str(bureau[col].dtype))


for col in bureau_balance.columns:
    print(col + ": " + str(bureau_balance[col].dtype))


for col in pos_cash.columns:
    print(col + ": " + str(pos_cash[col].dtype))


for col in cc_balance.columns:
    print(col + ": " + str(cc_balance[col].dtype))


for col in installments.columns:
    print(col + ": " + str(installments[col].dtype))


train.isnull().sum()


test.isnull().sum()


previous.isnull().sum()


bureau.isnull().sum()


bureau_balance.isnull().sum()


pos_cash.isnull().sum()


cc_balance.isnull().sum()


installments.isnull().sum()


train.head(10)


test.head(10)


previous.head(10)


bureau.head(10)


bureau_balance.head(10)


pos_cash.head(10)


cc_balance.head(10)


installments.head(10)


(train == 'XNA').sum()



train['is_train'] = 1
test['is_train'] = 0
test['TARGET'] = None



df = pd.concat([train, test], axis=0).reset_index(drop=True)


df.replace('XNA', np.nan, inplace=True)


(df['CODE_GENDER'] == 'XNA').sum()


df.head(5)



df.shape


(df == 365243).sum()


(previous == 365243).sum()



(bureau == 365243).sum()


(bureau_balance == 365243).sum()


(pos_cash == 365243).sum()


(cc_balance == 365243).sum()


(installments == 365243).sum()


df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)


previous['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
previous['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
previous['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
previous['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
previous['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)


missings=df.isnull().sum()/len(df)*100
to_impute = missings[(missings>0)&(missings<2)].sort_values()
print("Columns to impute:")
print(to_impute)


numericals = ['DAYS_LAST_PHONE_CHANGE','CNT_FAM_MEMBERS','AMT_ANNUITY','AMT_GOODS_PRICE','EXT_SOURCE_2','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']
for col in numericals:
    df[col].fillna(df[col].median(), inplace=True)


for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        print(col + ": " + str(df[col].unique()))
        print()



binary_map = {'Y': 1, 'Yes': 1, 'N': 0, 'No': 0}
binary_cols = ['EMERGENCYSTATE_MODE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

for col in binary_cols:
    df[col] = df[col].map(binary_map)


for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        print(col + ": " + str(df[col].unique()))
        print()


df['FONDKAPREMONT_MODE'].value_counts(dropna=False)


df['FONDKAPREMONT_MODE'].replace('not specified', np.nan, inplace=True)



df['NAME_FAMILY_STATUS'].value_counts(dropna=False)



df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan)


for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        print(col + ": " + str(df[col].unique()))
        print()


for col in bureau.columns:
    if bureau[col].dtype == 'object' or bureau[col].dtype.name == 'category':
        print(col + ": " + str(bureau[col].unique()))
        print()


bureau['CREDIT_TYPE'].value_counts(dropna=False)


bureau['CREDIT_TYPE'] = bureau['CREDIT_TYPE'].replace('Unknown type of loan', np.nan)


for col in bureau_balance.columns:
    if bureau_balance[col].dtype == 'object' or bureau_balance[col].dtype.name == 'category':
        print(col + ": " + str(bureau_balance[col].unique()))
        print()



bureau_balance['STATUS'].value_counts(dropna=False)


for col in previous.columns:
    if previous[col].dtype == 'object' or previous[col].dtype.name == 'category':
        print(col + ": " + str(previous[col].unique()))
        print()


previous['FLAG_LAST_APPL_PER_CONTRACT'] = previous['FLAG_LAST_APPL_PER_CONTRACT'].map(binary_map)


previous['FLAG_LAST_APPL_PER_CONTRACT'].value_counts(dropna=False)


unknown_cols=['NAME_CASH_LOAN_PURPOSE','NAME_PAYMENT_TYPE',
                    'CODE_REJECT_REASON','NAME_CLIENT_TYPE',
                    'NAME_GOODS_CATEGORY','NAME_PORTFOLIO',
                    'NAME_PRODUCT_TYPE','NAME_SELLER_INDUSTRY',
                    'NAME_YIELD_GROUP']

for col in unknown_cols:
    previous[col] = previous[col].replace({'XNA':'Unknown', 'XAP':'Unknown'})


for col in previous.columns:
    if previous[col].dtype=='object' or previous[col].dtype.name=='category':
        print(col+ ": " +str(previous[col].unique()))
        print()


for col in pos_cash.columns:
    if pos_cash[col].dtype=='object' or pos_cash[col].dtype.name=='category':
        print(col + ": " + str(pos_cash[col].unique()))
        print()


pos_cash['NAME_CONTRACT_STATUS'].value_counts(dropna=False)


pos_cash['NAME_CONTRACT_STATUS'].replace('XNA', np.nan, inplace=True)


def agg_and_prefix(df_src, group_key, agg_dict, prefix):
    agg=df_src.groupby(group_key).agg(agg_dict)
    agg.columns = [f"{prefix}_{c[0]}_{c[1].upper()}" for c in agg.columns]
    agg.reset_index(inplace=True)
    return agg


severity={'C': 0, 'X': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
bureau_balance['STATUS_SEV']=bureau_balance['STATUS'].map(severity).astype('int8')

bb_agg= agg_and_prefix(
    bureau_balance,
    'SK_ID_BUREAU',
    {
        'MONTHS_BALANCE':['min', 'max', 'size'],
        'STATUS_SEV':['max', 'mean']
    },
    'BBAL'
)


bureau=bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

bureau_agg=agg_and_prefix(
    bureau,
    'SK_ID_CURR',
    {
        'AMT_CREDIT_SUM':['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT':['mean', 'sum'],
        'DAYS_CREDIT':['min', 'max', 'mean'],
        'CREDIT_ACTIVE':['nunique'],        
        'BBAL_STATUS_SEV_MAX':['max'],            
        'BBAL_STATUS_SEV_MEAN':['mean'],            
        'BBAL_MONTHS_BALANCE_SIZE':['mean']         
    },
    'BURO'
)



def approved_ratio(x):
    return (x=='Approved').mean()
approved_ratio.__name__='APPROVED_RATIO' 

previous['APP_CREDIT_PERC']=previous['AMT_APPLICATION']/previous['AMT_CREDIT']

previous_agg=agg_and_prefix(
    previous,
    'SK_ID_CURR',
    {
        'AMT_CREDIT':['mean', 'max', 'sum'],
        'AMT_ANNUITY':['mean'],
        'APP_CREDIT_PERC':['mean'],
        'CNT_PAYMENT':['sum'],
        'NAME_CONTRACT_STATUS': approved_ratio
    },
    'PREV'
)



problem_set={'Returned to the store', 'Amortized debt', 'Demand'}
pos_cash['STATUS_FLAG'] = pos_cash['NAME_CONTRACT_STATUS'].map(
    lambda s: 2 if s in problem_set else (1 if s=='Completed' else 0)
)

pos_cash_agg=agg_and_prefix(
    pos_cash,
    'SK_ID_CURR',
    {
        'CNT_INSTALMENT_FUTURE': ['sum', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'STATUS_FLAG': ['max', 'mean']
    },
    'POS'
)


installments['PAYMENT_PERC']=(installments['AMT_PAYMENT']/installments['AMT_INSTALMENT']).clip(0, 2)             
installments['PAYMENT_DIFF']=installments['AMT_INSTALMENT']-installments['AMT_PAYMENT']

installments_agg = agg_and_prefix(
    installments,
    'SK_ID_CURR',
    {
        'PAYMENT_PERC':['mean', 'std'],
        'PAYMENT_DIFF':['sum', 'mean'],
        'DAYS_ENTRY_PAYMENT':['mean']
    },
    'INST'
)


cc_balance['UTIL']=cc_balance['AMT_BALANCE']/cc_balance['AMT_CREDIT_LIMIT_ACTUAL']
cc_balance_agg=agg_and_prefix(
    cc_balance,
    'SK_ID_CURR',
    {
        'UTIL':['mean', 'max'],
        'AMT_PAYMENT_CURRENT':['sum', 'mean'],
        'CNT_DRAWINGS_CURRENT':['sum', 'mean'],
        'SK_DPD':['max', 'mean']
    },
    'CC'
)


for block in [bureau_agg, previous_agg, pos_cash_agg, installments_agg, cc_balance_agg]:
    df=df.merge(block, on='SK_ID_CURR', how='left')



df.head(5)


df.replace([np.inf, -np.inf], np.nan, inplace=True)


df.shape


for col in df.columns:
    if df[col].dtype=='object' or df[col].dtype.name=='category':
        print(col+ ": "+str(df[col].unique()))
        print()


onehot_encode_cols=[
    'NAME_TYPE_SUITE',
    'NAME_INCOME_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE',
    'WALLSMATERIAL_MODE',
    'ORGANIZATION_TYPE',
    'CODE_GENDER',
    'NAME_CONTRACT_TYPE',
    'NAME_EDUCATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START',
    'OCCUPATION_TYPE'
]
df=pd.get_dummies(df,columns=onehot_encode_cols,dummy_na=True,drop_first=False)


df.shape


df['is_train']=df['TARGET'].notna().astype(int)
train_df=df[df['is_train']==1].copy()
test_df=df[df['is_train']==0].copy()

X_train=train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
y_train=train_df['TARGET'].astype(int)
X_test=test_df.drop(columns=['TARGET', 'SK_ID_CURR'])


n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

out_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(),
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        early_stopping_rounds=100,
        verbose=100
    )

    out_preds[va_idx] = model.predict_proba(X_va)[:, 1]
    test_preds      += model.predict_proba(X_test)[:, 1] / n_splits

    print(f"Fold {fold} AUC: {roc_auc_score(y_va, out_preds[va_idx]):.5f}")

print("Overall  CV AUC:", roc_auc_score(y_train, out_preds).round(5))



from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_train, out_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.5f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


submission_path = '/kaggle/working/submission.csv'

# Delete the old submission file if it exists
if os.path.exists(submission_path):
    os.remove(submission_path)
    print("Old submission.csv deleted successfully.")
else:
    print("No existing submission.csv found — nothing to delete.")


submission = pd.DataFrame({
    'SK_ID_CURR':test_df['SK_ID_CURR'],
    'TARGET':test_preds
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(submission.head(10))

