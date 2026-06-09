# CÃ i Ä‘áº·t cÃ¡c thÆ° viá»‡n cáº§n thiáº¿t
!pip install -q numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost catboost


import importlib.util

# Danh sÃ¡ch cÃ¡c thÆ° viá»‡n cáº§n kiá»ƒm tra phiÃªn báº£n
libraries = [
    "numpy", "pandas", "matplotlib", "seaborn", "gc", "time", "warnings", 
    "datetime", "sklearn", "lightgbm", "xgboost", "catboost"
]

# Kiá»ƒm tra vÃ  hiá»ƒn thá»‹ phiÃªn báº£n cá»§a tá»«ng thÆ° viá»‡n
for lib in libraries:
    spec = importlib.util.find_spec(lib)
    if spec is not None:
        module = importlib.import_module(lib)
        version = getattr(module, '__version__', 'KhÃ´ng cÃ³ thÃ´ng tin phiÃªn báº£n')
        print(f"{lib}: {version}")
    else:
        print(f"{lib}: KhÃ´ng Ä‘Æ°á»£c cÃ i Ä‘áº·t")



# Import cÃ¡c thÆ° viá»‡n
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


# Cáº¥u hÃ¬nh hiá»ƒn thá»‹
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
warnings.filterwarnings('ignore')



# Thiáº¿t láº­p style cho biá»ƒu Ä‘á»“
plt.style.use('ggplot')
sns.set_style('whitegrid')


# Ä�á»�c dá»¯ liá»‡u giao dá»‹ch
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


train_transaction.head()


# Ä�á»�c dá»¯ liá»‡u Ä‘á»‹nh danh
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')


train_identity.head()


test_identity.head()


# Chuáº©n hÃ³a tÃªn cá»™t trong test_identity: thay dáº¥u gáº¡ch ngang (-) thÃ nh dáº¥u gáº¡ch dÆ°á»›i (_)
test_identity.columns = [col.replace('-', '_') for col in test_identity.columns]


# Kiá»ƒm tra kÃ­ch thÆ°á»›c dá»¯ liá»‡u
print(f'Train Transaction: {train_transaction.shape}')
print(f'Test Transaction: {test_transaction.shape}')
print(f'Train Identity: {train_identity.shape}')
print(f'Test Identity: {test_identity.shape}')


# Káº¿t há»£p dá»¯ liá»‡u giao dá»‹ch vÃ  Ä‘á»‹nh danh
train = train_transaction.merge(train_identity, on='TransactionID', how='left')
test = test_transaction.merge(test_identity, on='TransactionID', how='left')


# Kiá»ƒm tra kÃ­ch thÆ°á»›c sau khi káº¿t há»£p
print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')


# Xem thÃ´ng tin cÆ¡ báº£n cá»§a dá»¯ liá»‡u
train.head()


# XÃ³a cÃ¡c DataFrame gá»‘c Ä‘á»ƒ giáº£i phÃ³ng RAM
del train_transaction, train_identity, test_transaction, test_identity
gc.collect()


# Kiá»ƒm tra phÃ¢n bá»‘ cá»§a biáº¿n má»¥c tiÃªu
plt.figure(figsize=(10, 6))
sns.countplot(x='isFraud', data=train)
plt.title('PhÃ¢n bá»‘ cá»§a biáº¿n má»¥c tiÃªu isFraud')
plt.show()


# TÃ­nh tá»· lá»‡ gian láº­n
fraud_ratio = train['isFraud'].mean() * 100
print(f'Tá»· lá»‡ giao dá»‹ch gian láº­n: {fraud_ratio:.2f}%')
print(f'Sá»‘ lÆ°á»£ng giao dá»‹ch gian láº­n: {train["isFraud"].sum()}')
print(f'Tá»•ng sá»‘ giao dá»‹ch: {len(train)}')


# Kiá»ƒm tra giÃ¡ trá»‹ thiáº¿u
missing_train = (train.isnull().sum() / len(train)) * 100
missing_test = (test.isnull().sum() / len(test)) * 100


missing_df = pd.DataFrame({'Train': missing_train, 'Test': missing_test})
missing_df = missing_df[missing_df.sum(axis=1) > 0].sort_values('Train', ascending=False)


# Hiá»ƒn thá»‹ cÃ¡c cá»™t cÃ³ giÃ¡ trá»‹ thiáº¿u > 50%
print("CÃ¡c cá»™t cÃ³ giÃ¡ trá»‹ thiáº¿u > 50%:")
print(missing_df[missing_df['Train'] > 50].head(20))


# Chuyá»ƒn Ä‘á»•i TransactionDT thÃ nh ngÃ y
START_DATE = '2017-12-01' # Bá»™ dá»¯ liá»‡u Ä‘Æ°á»£c cho lÃ  báº¯t Ä‘áº§u vÃ o ngÃ y nÃ y
start_date = datetime.strptime(START_DATE, '%Y-%m-%d')


def convert_to_datetime(x):
    return start_date + pd.Timedelta(seconds=x)


train['TransactionDate'] = train['TransactionDT'].apply(convert_to_datetime)
test['TransactionDate'] = test['TransactionDT'].apply(convert_to_datetime)


# ThÃªm cÃ¡c Ä‘áº·c trÆ°ng thá»�i gian
for df in [train, test]:
    df['Day'] = df['TransactionDate'].dt.day
    df['Month'] = df['TransactionDate'].dt.month
    df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
    df['Hour'] = df['TransactionDate'].dt.hour


# PhÃ¢n tÃ­ch giao dá»‹ch gian láº­n theo ngÃ y trong tuáº§n
plt.figure(figsize=(12, 6))
fraud_by_day = train.groupby('DayOfWeek')['isFraud'].mean() * 100
sns.barplot(x=fraud_by_day.index, y=fraud_by_day.values)
plt.title('Tá»· lá»‡ gian láº­n theo ngÃ y trong tuáº§n')
plt.xlabel('NgÃ y trong tuáº§n (0: Thá»© 2, 6: Chá»§ nháº­t)')
plt.ylabel('Tá»· lá»‡ gian láº­n (%)')
plt.show()


# PhÃ¢n tÃ­ch giao dá»‹ch gian láº­n theo giá»�
plt.figure(figsize=(16, 6))
fraud_by_hour = train.groupby('Hour')['isFraud'].mean() * 100
sns.barplot(x=fraud_by_hour.index, y=fraud_by_hour.values)
plt.title('Tá»· lá»‡ gian láº­n theo giá»�')
plt.xlabel('Giá»� trong ngÃ y')
plt.ylabel('Tá»· lá»‡ gian láº­n (%)')
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(data=train, x='TransactionAmt', hue='isFraud', bins=50, log_scale=True)
plt.title('PhÃ¢n bá»‘ TransactionAmt theo isFraud')
plt.show()


# ThÃªm Ä‘áº·c trÆ°ng pháº§n tháº­p phÃ¢n cá»§a TransactionAmt
train['TransactionAmt_decimal'] = train['TransactionAmt'] - np.floor(train['TransactionAmt'])
test['TransactionAmt_decimal'] = test['TransactionAmt'] - np.floor(test['TransactionAmt'])


plt.figure(figsize=(12, 6))
sns.histplot(data=train, x='TransactionAmt_decimal', hue='isFraud', bins=20)
plt.title('PhÃ¢n bá»‘ pháº§n tháº­p phÃ¢n cá»§a TransactionAmt theo isFraud')
plt.show()


# Táº¡o UID dá»±a trÃªn card1 vÃ  addr1
for df in [train, test]:
    df['uid1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
    
    # Táº¡o thÃªm cÃ¡c UID khÃ¡c dá»±a trÃªn cÃ¡c káº¿t há»£p khÃ¡c nhau
    df['uid2'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['card2'].astype(str)
    df['uid3'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['card2'].astype(str) + '_' + df['card3'].astype(str)
    df['uid4'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['card2'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)



# Kiá»ƒm tra sá»‘ lÆ°á»£ng UID duy nháº¥t
print(f"Sá»‘ lÆ°á»£ng uid1 duy nháº¥t trong táº­p train: {train['uid1'].nunique()}")
print(f"Sá»‘ lÆ°á»£ng uid2 duy nháº¥t trong táº­p train: {train['uid2'].nunique()}")
print(f"Sá»‘ lÆ°á»£ng uid3 duy nháº¥t trong táº­p train: {train['uid3'].nunique()}")
print(f"Sá»‘ lÆ°á»£ng uid4 duy nháº¥t trong táº­p train: {train['uid4'].nunique()}")


# TÃ­nh tá»· lá»‡ gian láº­n theo uid1
uid_fraud = train.groupby('uid1')['isFraud'].mean().reset_index()
uid_fraud.columns = ['uid1', 'fraud_rate']


uid_fraud.head()


# TÃ­nh sá»‘ lÆ°á»£ng giao dá»‹ch theo uid1
uid_count = train.groupby('uid1').size().reset_index()
uid_count.columns = ['uid1', 'transaction_count']


# Káº¿t há»£p thÃ´ng tin
uid_stats = uid_fraud.merge(uid_count, on='uid1', how='left')


# Hiá»ƒn thá»‹ thá»‘ng kÃª
print("Thá»‘ng kÃª vá»� tá»· lá»‡ gian láº­n theo UID:")
print(uid_stats.describe())


# Váº½ biá»ƒu Ä‘á»“ phÃ¢n bá»‘ tá»· lá»‡ gian láº­n theo UID
plt.figure(figsize=(12, 6))
sns.histplot(uid_stats['fraud_rate'], bins=20)
plt.title('PhÃ¢n bá»‘ tá»· lá»‡ gian láº­n theo UID')
plt.xlabel('Tá»· lá»‡ gian láº­n')
plt.show()


# Váº½ biá»ƒu Ä‘á»“ phÃ¢n tÃ¡n giá»¯a sá»‘ lÆ°á»£ng giao dá»‹ch vÃ  tá»· lá»‡ gian láº­n
plt.figure(figsize=(12, 6))
sns.scatterplot(x='transaction_count', y='fraud_rate', data=uid_stats, alpha=0.5)
plt.title('Má»‘i quan há»‡ giá»¯a sá»‘ lÆ°á»£ng giao dá»‹ch vÃ  tá»· lá»‡ gian láº­n theo UID')
plt.xlabel('Sá»‘ lÆ°á»£ng giao dá»‹ch')
plt.ylabel('Tá»· lá»‡ gian láº­n')
plt.xscale('log')
plt.show()


# Táº¡o Ä‘áº·c trÆ°ng tá»•ng há»£p theo uid1
def create_aggregated_features(df, group_var, agg_cols):
    """
    Táº¡o Ä‘áº·c trÆ°ng tá»•ng há»£p theo biáº¿n nhÃ³m
    """
    for col in agg_cols:
        # Bá»� qua náº¿u cá»™t khÃ´ng tá»“n táº¡i
        if col not in df.columns:
            continue
            
        # Kiá»ƒm tra kiá»ƒu dá»¯ liá»‡u
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            continue
            
        # TÃ­nh cÃ¡c thá»‘ng kÃª
        prefix = f'{col}_{group_var}'
        
        # TÃ­nh giÃ¡ trá»‹ trung bÃ¬nh
        df[f'{prefix}_mean'] = df.groupby([group_var])[col].transform('mean')
        
        # TÃ­nh Ä‘á»™ lá»‡ch chuáº©n
        df[f'{prefix}_std'] = df.groupby([group_var])[col].transform('std')
        
        # TÃ­nh giÃ¡ trá»‹ lá»›n nháº¥t
        df[f'{prefix}_max'] = df.groupby([group_var])[col].transform('max')
        
        # TÃ­nh giÃ¡ trá»‹ nhá»� nháº¥t
        df[f'{prefix}_min'] = df.groupby([group_var])[col].transform('min')
        
        # TÃ­nh sá»‘ lÆ°á»£ng giÃ¡ trá»‹ duy nháº¥t
        df[f'{prefix}_nunique'] = df.groupby([group_var])[col].transform('nunique')
    
    # TÃ­nh sá»‘ lÆ°á»£ng giao dá»‹ch theo nhÃ³m
    df[f'{group_var}_count'] = df.groupby([group_var])['TransactionID'].transform('count')
    
    return df


# Danh sÃ¡ch cÃ¡c cá»™t sá»‘ Ä‘á»ƒ táº¡o Ä‘áº·c trÆ°ng tá»•ng há»£p
numeric_cols = ['TransactionAmt', 'TransactionAmt_decimal', 'Day', 'Hour']


# ThÃªm cÃ¡c cá»™t D (timedelta)
d_cols = [col for col in train.columns if col.startswith('D')]
numeric_cols.extend(d_cols)


# ThÃªm cÃ¡c cá»™t C (counting)
c_cols = [col for col in train.columns if col.startswith('C')]
numeric_cols.extend(c_cols)


# Táº¡o Ä‘áº·c trÆ°ng tá»•ng há»£p cho táº­p train vÃ  test
for df in [train, test]:
    df = create_aggregated_features(df, 'uid1', numeric_cols)


# Kiá»ƒm tra cÃ¡c Ä‘áº·c trÆ°ng má»›i
new_features = [col for col in train.columns if 'uid1' in col and col != 'uid1']
print(f"Sá»‘ lÆ°á»£ng Ä‘áº·c trÆ°ng tá»•ng há»£p má»›i: {len(new_features)}")
print("Má»™t sá»‘ Ä‘áº·c trÆ°ng tá»•ng há»£p má»›i:")
print(new_features[:10])


# Xá»­ lÃ½ giÃ¡ trá»‹ thiáº¿u
def handle_missing_values(df):
    """
    Xá»­ lÃ½ giÃ¡ trá»‹ thiáº¿u cho cÃ¡c cá»™t sá»‘ vÃ  phÃ¢n loáº¡i
    """
    for col in df.columns:
        # Bá»� qua TransactionID vÃ  isFraud
        if col in ['TransactionID', 'isFraud', 'TransactionDate']:
            continue
            
        # Xá»­ lÃ½ cá»™t sá»‘
        if df[col].dtype != 'object':
            # Thay tháº¿ giÃ¡ trá»‹ thiáº¿u báº±ng -999
            df[col] = df[col].fillna(-999)
        else:
            # Thay tháº¿ giÃ¡ trá»‹ thiáº¿u báº±ng 'missing'
            df[col] = df[col].fillna('missing')
    
    return df


# Xá»­ lÃ½ giÃ¡ trá»‹ thiáº¿u cho táº­p train vÃ  test
train = handle_missing_values(train)
test = handle_missing_values(test)


# Kiá»ƒm tra láº¡i giÃ¡ trá»‹ thiáº¿u
print("Sá»‘ lÆ°á»£ng giÃ¡ trá»‹ thiáº¿u sau khi xá»­ lÃ½:")
print(f"Train: {train.isnull().sum().sum()}")
print(f"Test: {test.isnull().sum().sum()}")


# MÃ£ hÃ³a Ä‘áº·c trÆ°ng phÃ¢n loáº¡i
def label_encode(df_train, df_test, cols):
    """
    MÃ£ hÃ³a Ä‘áº·c trÆ°ng phÃ¢n loáº¡i báº±ng LabelEncoder,
    xá»­ lÃ½ trÆ°á»�ng há»£p cá»™t cÃ³ thá»ƒ chá»‰ tá»“n táº¡i á»Ÿ má»™t trong hai DataFrame.
    Fit encoder trÃªn táº¥t cáº£ giÃ¡ trá»‹ cÃ³ thá»ƒ cÃ³ á»Ÿ cáº£ train/test náº¿u cá»™t tá»“n táº¡i.
    """
    # Táº¡o báº£n sao Ä‘á»ƒ trÃ¡nh cáº£nh bÃ¡o SettingWithCopyWarning
    df_train = df_train.copy()
    df_test = df_test.copy()

    print("--- Starting Flexible Label Encoding ---")
    for col in cols:
        # XÃ¡c Ä‘á»‹nh xem cá»™t cÃ³ tá»“n táº¡i khÃ´ng
        train_col_exists = col in df_train.columns
        test_col_exists = col in df_test.columns

        # XÃ¡c Ä‘á»‹nh xem cÃ³ pháº£i kiá»ƒu object á»Ÿ nÆ¡i nÃ³ tá»“n táº¡i khÃ´ng
        # Chá»‰ cáº§n lÃ  object á»Ÿ Ã­t nháº¥t 1 nÆ¡i Ä‘á»ƒ xem xÃ©t mÃ£ hÃ³a
        is_object_somewhere = (train_col_exists and df_train[col].dtype == 'object') or \
                              (test_col_exists and df_test[col].dtype == 'object')

        # Náº¿u cá»™t khÃ´ng tá»“n táº¡i á»Ÿ Ä‘Ã¢u cáº£, hoáº·c khÃ´ng pháº£i lÃ  object á»Ÿ báº¥t ká»³ Ä‘Ã¢u nÃ³ tá»“n táº¡i -> bá»� qua
        if not (train_col_exists or test_col_exists) or not is_object_somewhere:
            # print(f"Skipping column '{col}': Does not exist or not object type where it exists.")
            continue

        print(f"Processing column: '{col}'")

        # --- Fitting ---
        le = LabelEncoder()
        values_to_fit = []

        # Láº¥y giÃ¡ trá»‹ tá»« train náº¿u cá»™t tá»“n táº¡i vÃ  lÃ  object
        if train_col_exists and df_train[col].dtype == 'object':
            # Chuyá»ƒn sang str Ä‘á»ƒ xá»­ lÃ½ NaN vÃ  cÃ¡c kiá»ƒu khÃ¡c, láº¥y giÃ¡ trá»‹ .values
            values_to_fit.extend(list(df_train[col].astype(str).values))
            print(f"  Found object column '{col}' in train.")

        # Láº¥y giÃ¡ trá»‹ tá»« test náº¿u cá»™t tá»“n táº¡i vÃ  lÃ  object
        if test_col_exists and df_test[col].dtype == 'object':
            # Chuyá»ƒn sang str Ä‘á»ƒ xá»­ lÃ½ NaN vÃ  cÃ¡c kiá»ƒu khÃ¡c, láº¥y giÃ¡ trá»‹ .values
            values_to_fit.extend(list(df_test[col].astype(str).values))
            print(f"  Found object column '{col}' in test.")
        elif test_col_exists:
             print(f"  Column '{col}' exists in test but is not object type ({df_test[col].dtype}).")
        else:
             print(f"  Column '{col}' does not exist in test.")


        # Chá»‰ fit náº¿u cÃ³ giÃ¡ trá»‹ há»£p lá»‡ (trÃ¡nh lá»—i vá»›i cá»™t trá»‘ng hoáº·c toÃ n NaN)
        # Láº¥y táº­p há»£p cÃ¡c giÃ¡ trá»‹ duy nháº¥t Ä‘á»ƒ fit
        unique_values = pd.Series(values_to_fit).unique()
        if len(unique_values) > 0:
            print(f"  Fitting LabelEncoder for '{col}' on {len(unique_values)} unique values.")
            le.fit(unique_values) # Fit trÃªn cÃ¡c giÃ¡ trá»‹ duy nháº¥t Ä‘Ã£ thu tháº­p

            # --- Transforming ---
            # Ã�p dá»¥ng transform cho train náº¿u cá»™t tá»“n táº¡i vÃ  lÃ  object
            if train_col_exists and df_train[col].dtype == 'object':
                try:
                    df_train[col] = le.transform(df_train[col].astype(str))
                    print(f"  Transformed train column '{col}'.")
                except ValueError as e:
                    print(f"  ERROR transforming train column '{col}': {e}. Check for unseen values.")
                    # Xá»­ lÃ½ lá»—i náº¿u cáº§n (vÃ­ dá»¥: gÃ¡n giÃ¡ trá»‹ Ä‘áº·c biá»‡t cho giÃ¡ trá»‹ khÃ´ng tháº¥y)
                    # df_train[col] = df_train[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1) # VÃ­ dá»¥

            # Ã�p dá»¥ng transform cho test náº¿u cá»™t tá»“n táº¡i vÃ  lÃ  object
            if test_col_exists and df_test[col].dtype == 'object':
                try:
                    df_test[col] = le.transform(df_test[col].astype(str))
                    print(f"  Transformed test column '{col}'.")
                except ValueError as e:
                    print(f"  ERROR transforming test column '{col}': {e}. Check for unseen values.")
                    # Xá»­ lÃ½ lá»—i náº¿u cáº§n
                    # df_test[col] = df_test[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1) # VÃ­ dá»¥
        else:
            print(f"  Skipping fitting/transforming for '{col}': No valid values found.")

    print("--- Finished Flexible Label Encoding ---")
    # KhÃ´ng nÃªn cÃ³ df_train.head() trong hÃ m, hÃ£y gá»�i nÃ³ bÃªn ngoÃ i sau khi hÃ m cháº¡y xong
    return df_train, df_test


# Danh sÃ¡ch cÃ¡c cá»™t phÃ¢n loáº¡i
categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
categorical_cols


# ThÃªm cÃ¡c cá»™t M (match)
m_cols = [col for col in train.columns if col.startswith('M')]
categorical_cols.extend(m_cols)


# ThÃªm cÃ¡c cá»™t id_12-id_38
id_cols = [col for col in train.columns if col.startswith('id_') and int(col.split('_')[1]) >= 12]
categorical_cols.extend(id_cols)


# ThÃªm cÃ¡c cá»™t UID
uid_cols = [col for col in train.columns if col.startswith('uid')]
categorical_cols.extend(uid_cols)


# Loáº¡i bá»� cÃ¡c cá»™t Ä‘Ã£ Ä‘Æ°á»£c xá»­ lÃ½
categorical_cols = list(set(categorical_cols))
categorical_cols


train.head()


test.head()


# MÃ£ hÃ³a Ä‘áº·c trÆ°ng phÃ¢n loáº¡i
train, test = label_encode(train, test, categorical_cols)


train.head()


test.head()


print([col for col in train.columns if train[col].dtype == 'object'])
print([col for col in test.columns if test[col].dtype == 'object'])


# Kiá»ƒm tra káº¿t quáº£
print(f"Sá»‘ lÆ°á»£ng cá»™t phÃ¢n loáº¡i Ä‘Ã£ mÃ£ hÃ³a: {len(categorical_cols)}")
print("Má»™t sá»‘ cá»™t phÃ¢n loáº¡i Ä‘Ã£ mÃ£ hÃ³a:")
print(categorical_cols[:10])


# Loáº¡i bá»� cÃ¡c cá»™t khÃ´ng cáº§n thiáº¿t
cols_to_drop = ['TransactionID', 'TransactionDate']


# Chuáº©n bá»‹ dá»¯ liá»‡u cho mÃ´ hÃ¬nh
X = train.drop(cols_to_drop + ['isFraud'], axis=1)
y = train['isFraud']
X_test = test.drop(cols_to_drop, axis=1)


# XÃ³a train vÃ  test Ä‘á»ƒ giáº£i phÃ³ng RAM
del train
gc.collect()


# Chia táº­p train thÃ nh táº­p huáº¥n luyá»‡n vÃ  táº­p xÃ¡c thá»±c
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train.head()


y_train.head()


# Kiá»ƒm tra kÃ­ch thÆ°á»›c dá»¯ liá»‡u
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")


# XÃ³a X vÃ  y Ä‘á»ƒ giáº£i phÃ³ng RAM
# del X, y
# gc.collect()


del X_train, X_val
gc.collect()


# Thiáº¿t láº­p tham sá»‘ cho LightGBM
lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_jobs': -1,
    'learning_rate': 0.01,
    'num_leaves': 256,
    'max_depth': 8,
    'tree_learner': 'serial',
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'min_child_weight': 1,
    'min_child_samples': 20,
    'scale_pos_weight': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1
}


# Táº¡o táº­p dá»¯ liá»‡u LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)


# Huáº¥n luyá»‡n mÃ´ hÃ¬nh
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    num_boost_round=10000,
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
    # dá»«ng early_stopping_rounds=100 náº¿u lÃ  phiÃªn báº£n < 4.0.0
    # tÆ°á»£ng tá»±, verbose_eval=100 thay cho lgb.log_evaluation náº¿u...
)


# LÆ°u mÃ´ hÃ¬nh
# lgb_model.save_model('lightgbm_model')


# Dá»± Ä‘oÃ¡n trÃªn táº­p xÃ¡c thá»±c
lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)


# Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh
lgb_val_auc = roc_auc_score(y_val, lgb_val_pred)
print(f"LightGBM Validation AUC: {lgb_val_auc:.6f}")


# Hiá»ƒn thá»‹ Ä‘áº·c trÆ°ng quan trá»�ng
lgb_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgb_model.feature_importance(importance_type='gain')
})
lgb_importance = lgb_importance.sort_values('Importance', ascending=False).reset_index(drop=True)


plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=lgb_importance.head(20))
plt.title('LightGBM Feature Importance (Top 20)')
plt.tight_layout()
plt.show()


del lgb_val, lgb_val_pred, lgb_val_auc, lgb_importance
gc.collect()


# Ä�Ã¡nh giÃ¡ trÃªn táº­p dá»¯ liá»‡u test
lgb_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)


# LÆ°u káº¿t quáº£ dá»± Ä‘oÃ¡n
# np.save('/kaggle/working/lgb_test_pred.npy', lgb_test_pred)


del lgb_model, lgb_train
gc.collect()


# Thiáº¿t láº­p tham sá»‘ cho XGBoost
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.05,
    'max_depth': 8,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0.1,
    'lambda': 0.1,
    'tree_method': 'hist',
    'nthread': -1,
    'scale_pos_weight': 1,
    'seed': 42
}


# Táº¡o táº­p dá»¯ liá»‡u XGBoost
xgb_train = xgb.DMatrix(X_train, y_train)
xgb_val = xgb.DMatrix(X_val, y_val)


# Thiáº¿t láº­p danh sÃ¡ch Ä‘Ã¡nh giÃ¡
evallist = [(xgb_train, 'train'), (xgb_val, 'val')]


# Huáº¥n luyá»‡n mÃ´ hÃ¬nh
xgb_model = xgb.train(
    xgb_params,
    xgb_train,
    num_boost_round=10000,
    evals=evallist,
    early_stopping_rounds=100,
    verbose_eval=100
)


# xgb_model.save_model('xgboost_model')


# Dá»± Ä‘oÃ¡n trÃªn táº­p xÃ¡c thá»±c vá»›i early stopping
xgb_val_pred = xgb_model.predict(xgb.DMatrix(X_val), iteration_range=(0, xgb_model.best_iteration + 1))


# Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh
xgb_val_auc = roc_auc_score(y_val, xgb_val_pred)
print(f"XGBoost Validation AUC: {xgb_val_auc:.6f}")


# Hiá»ƒn thá»‹ Ä‘áº·c trÆ°ng quan trá»�ng
importance_dict = xgb_model.get_score(importance_type='gain')
all_features = X_train.columns
importance_values = [importance_dict.get(feat, 0) for feat in all_features]
xgb_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': importance_values
})
xgb_importance = xgb_importance.sort_values('Importance', ascending=False).reset_index(drop=True)


plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_importance.head(20))
plt.title('XGBoost Feature Importance (Top 20)')
plt.tight_layout()
plt.show()


del xgb_val, xgb_val_pred, xgb_val_auc, xgb_importance
gc.collect()


# Ä�Ã¡nh giÃ¡ trÃªn táº­p dá»¯ liá»‡u test
xgb_test_pred = xgb_model.predict(xgb.DMatrix(X_test), iteration_range=(0, xgb_model.best_iteration + 1))


# LÆ°u káº¿t quáº£ dá»± Ä‘oÃ¡n
# np.save('/kaggle/working/xgb_test_pred.npy', xgb_test_pred)


del xgb_model, xgb_train
gc.collect()


# Thiáº¿t láº­p tham sá»‘ cho CatBoost
cat_params = {
    'iterations': 10000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 10,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'scale_pos_weight': 1,
    'eval_metric': 'AUC',
    'metric_period': 100,
    'od_type': 'Iter',
    'od_wait': 100,
    'random_seed': 42,
    'allow_writing_files': False,
    'task_type': 'CPU',
    'verbose': 100
}


# Huáº¥n luyá»‡n mÃ´ hÃ¬nh
cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
    early_stopping_rounds=100,
    verbose=False
)


# cat_model.save_model('catboost_model')


# Dá»± Ä‘oÃ¡n trÃªn táº­p xÃ¡c thá»±c
cat_val_pred = cat_model.predict_proba(X_val)[:, 1]


# Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh
cat_val_auc = roc_auc_score(y_val, cat_val_pred)
print(f"CatBoost Validation AUC: {cat_val_auc:.6f}")


# Hiá»ƒn thá»‹ Ä‘áº·c trÆ°ng quan trá»�ng
cat_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': cat_model.get_feature_importance()
})
cat_importance = cat_importance.sort_values('Importance', ascending=False).reset_index(drop=True)


plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=cat_importance.head(20))
plt.title('CatBoost Feature Importance (Top 20)')
plt.tight_layout()
plt.show()


del cat_val_pred, cat_val_auc, cat_importance
gc.collect()


# Ä�Ã¡nh giÃ¡ trÃªn táº­p dá»¯ liá»‡u test
cat_test_pred = cat_model.predict_proba(X_test)[:, 1]


# LÆ°u káº¿t quáº£ dá»± Ä‘oÃ¡n
# np.save('/kaggle/working/cat_test_pred.npy', cat_test_pred)


del cat_model
gc.collect()


'''# Load láº¡i model (do cháº¡y má»™t láº§n 3 mÃ´ hÃ¬nh khÃ¡ tá»‘n ram nÃªn thá»±c táº¿ Ä‘Ã£ cháº¡y 3 mÃ´ hÃ¬nh trÃªn á»Ÿ 3 láº§n khÃ¡c nhau)
lgb_model = lgb.Booster(model_file='/kaggle/input/model-for-ieee-fraud-detection/lightgbm_model')
lgb_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
np.save('/kaggle/working/lgb_test_pred.npy', lgb_test_pred)
del lgb_model
gc.collect()

xgb_model = xgb.Booster(model_file='/kaggle/input/model-for-ieee-fraud-detection/xgboost_model.json')
xgb_test_pred = xgb_model.predict(xgb.DMatrix(X_test), iteration_range=(0, xgb_model.best_iteration + 1))
np.save('/kaggle/working/xgb_test_pred.npy', xgb_test_pred)
del xgb_model
gc.collect()

cat_model = CatBoostClassifier().load_model('/kaggle/input/model-for-ieee-fraud-detection/catboost_model')
cat_test_pred = cat_model.predict_proba(X_test)[:, 1]
np.save('/kaggle/working/cat_test_pred.npy', cat_test_pred)
del cat_model
gc.collect()'''


# Load láº¡i káº¿t quáº£ prediction
lgb_test_pred = np.load('/kaggle/input/model-for-ieee-fraud-detection/lgb_test_pred.npy', allow_pickle=True)
xgb_test_pred = np.load('/kaggle/input/model-for-ieee-fraud-detection/xgb_test_pred.npy', allow_pickle=True)
cat_test_pred = np.load('/kaggle/input/model-for-ieee-fraud-detection/cat_test_pred.npy', allow_pickle=True)


# Káº¿t há»£p dá»± Ä‘oÃ¡n vá»›i trá»�ng sá»‘
# Dá»±a trÃªn káº¿t quáº£ xÃ¡c thá»±c, chÃºng ta cÃ³ thá»ƒ Ä‘iá»�u chá»‰nh trá»�ng sá»‘
weights = [0.1, 0.8, 0.1]  # LightGBM, XGBoost, CatBoost
test_pred = weights[0] * lgb_test_pred + weights[1] * xgb_test_pred + weights[2] * cat_test_pred


# XÃ³a cÃ¡c biáº¿n dá»± Ä‘oÃ¡n riÃªng láº» Ä‘á»ƒ giáº£i phÃ³ng RAM
del lgb_test_pred, xgb_test_pred, cat_test_pred
gc.collect()


# Háº­u xá»­ lÃ½ theo khÃ¡ch hÃ ng (UID)
# TÃ­nh dá»± Ä‘oÃ¡n trung bÃ¬nh cho má»—i UID
test_uid_pred = test[['TransactionID', 'uid1']].copy()
test_uid_pred['prediction'] = test_pred


del X_test
gc.collect()


# TÃ­nh dá»± Ä‘oÃ¡n trung bÃ¬nh cho má»—i UID
uid_mean_pred = test_uid_pred.groupby('uid1')['prediction'].transform('mean')


# Thay tháº¿ dá»± Ä‘oÃ¡n gá»‘c báº±ng dá»± Ä‘oÃ¡n trung bÃ¬nh theo UID
test_pred_pp = test_pred #+ 1 * uid_mean_pred.values


# XÃ³a cÃ¡c biáº¿n khÃ´ng cáº§n thiáº¿t
del test_pred, test_uid_pred, uid_mean_pred
gc.collect()


# Táº¡o file ná»™p bÃ i
submission = pd.DataFrame({
    'TransactionID': test['TransactionID'],
    'isFraud': test_pred_pp
})


# LÆ°u file ná»™p bÃ i
submission.to_csv('/kaggle/working/submission.csv', index=False)


# Hiá»ƒn thá»‹ thÃ´ng tin vá»� file ná»™p bÃ i
print("ThÃ´ng tin vá»� file ná»™p bÃ i:")
print(submission.describe())
print(f"Sá»‘ lÆ°á»£ng giao dá»‹ch: {len(submission)}")


# XÃ³a submission Ä‘á»ƒ giáº£i phÃ³ng RAM
del submission, test_pred_pp
gc.collect()


# Thiáº¿t láº­p KFold
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)


oof_preds_lgb = np.zeros(X.shape[0])
test_preds_lgb = np.zeros(X_test.shape[0])
feature_importance_df_lgb = pd.DataFrame()


# Thiáº¿t láº­p tham sá»‘ LightGBM (giá»¯ nguyÃªn hoáº·c tinh chá»‰nh náº¿u cáº§n)
lgb_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "n_jobs": -1,
    "learning_rate": 0.01,
    "num_leaves": 256,
    "max_depth": 8, # Giáº£m Ä‘á»™ sÃ¢u má»™t chÃºt Ä‘á»ƒ trÃ¡nh overfitting
    "tree_learner": "serial",
    "colsample_bytree": 0.7, # Giáº£m má»™t chÃºt
    "subsample": 0.7, # Giáº£m má»™t chÃºt
    "min_child_weight": 1,
    "min_child_samples": 20,
    "scale_pos_weight": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42
}


print("Training LightGBM...")
# VÃ²ng láº·p qua cÃ¡c fold
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"Current Fold: {fold_+1}")
    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])

    # Huáº¥n luyá»‡n mÃ´ hÃ¬nh
    clf = lgb.train(
        lgb_params,
        trn_data,
        valid_sets=[trn_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=10000,
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
    )

    # Dá»± Ä‘oÃ¡n OOF
    oof_preds_lgb[val_idx] = clf.predict(X.iloc[val_idx], num_iteration=clf.best_iteration)

    # Dá»± Ä‘oÃ¡n trÃªn táº­p test
    test_preds_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / NFOLDS

    # LÆ°u feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = X.columns
    fold_importance_df["Importance"] = clf.feature_importance(importance_type="gain")
    fold_importance_df["Fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df], axis=0)

    del clf, trn_data, val_data, fold_importance_df
    gc.collect()


# LÆ°u káº¿t quáº£ dá»± Ä‘oÃ¡n
# np.save('/kaggle/working/oof_preds_lgb_new.npy', oof_preds_lgb)
np.save('/kaggle/working/test_preds_lgb_new.npy', test_preds_lgb)


# Ä�Ã¡nh giÃ¡ OOF AUC
oof_auc_lgb = roc_auc_score(y, oof_preds_lgb)
print(f"LightGBM OOF AUC: {oof_auc_lgb:.6f}")


# Hiá»ƒn thá»‹ Ä‘áº·c trÆ°ng quan trá»�ng trung bÃ¬nh
mean_importance_lgb = feature_importance_df_lgb[["Feature", "Importance"]].groupby("Feature").mean().sort_values(by="Importance", ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=mean_importance_lgb.Importance.head(20), y=mean_importance_lgb.head(20).index)
plt.title("LightGBM Feature Importance (Average over folds)")
plt.tight_layout()
plt.show()


del oof_preds_lgb, feature_importance_df_lgb, test_preds_lgb
gc.collect()


# Thiáº¿t láº­p KFold
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)


X = X.astype('float32')
y = y.astype('int32')
X_test = X_test.astype('float32')


# Thiáº¿t láº­p KFold (sá»­ dá»¥ng láº¡i folds Ä‘Ã£ táº¡o)
oof_preds_xgb = np.zeros(X.shape[0])
test_preds_xgb = np.zeros(X_test.shape[0])
feature_importance_df_xgb = pd.DataFrame()


# Thiáº¿t láº­p tham sá»‘ XGBoost (giá»¯ nguyÃªn hoáº·c tinh chá»‰nh)
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.02, # Giáº£m learning rate
    "max_depth": 8,
    "min_child_weight": 1,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "alpha": 0.1,
    "lambda": 0.1,
    "tree_method": "gpu_hist",
    "nthread": -1,
    "scale_pos_weight": 1,
    "seed": 42
}


print("\nTraining XGBoost...")
# VÃ²ng láº·p qua cÃ¡c fold
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"Current Fold: {fold_+1}")
    
    # Táº¡o táº­p huáº¥n luyá»‡n vÃ  validation
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    
    # Táº¡o DMatrix
    train_data = xgb.DMatrix(X_train, label=y_train)
    valid_data = xgb.DMatrix(X_valid, label=y_valid)
    
    # Thiáº¿t láº­p watchlist
    watchlist = [(train_data, "train"), (valid_data, "val")]
    
    # Huáº¥n luyá»‡n mÃ´ hÃ¬nh
    clf = xgb.train(
        xgb_params,
        train_data,
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    # Dá»± Ä‘oÃ¡n OOF
    oof_preds_xgb[val_idx] = clf.predict(valid_data, iteration_range=(0, clf.best_iteration + 1))
    
    # Dá»± Ä‘oÃ¡n trÃªn táº­p test
    test_preds_xgb += clf.predict(xgb.DMatrix(X_test), iteration_range=(0, clf.best_iteration + 1)) / NFOLDS
    
    # LÆ°u feature importance
    fold_importance_df = pd.DataFrame.from_dict(clf.get_score(importance_type="gain"), orient="index", columns=["Importance"])
    fold_importance_df["Feature"] = fold_importance_df.index
    fold_importance_df["Fold"] = fold_ + 1
    feature_importance_df_xgb = pd.concat([feature_importance_df_xgb, fold_importance_df], axis=0)
    
    # Giáº£i phÃ³ng bá»™ nhá»›
    del clf, train_data, valid_data, X_train, y_train, X_valid, y_valid, fold_importance_df
    gc.collect()


# LÆ°u káº¿t quáº£ dá»± Ä‘oÃ¡n
# np.save('/kaggle/working/oof_preds_xgb_new.npy', oof_preds_xgb)
np.save('/kaggle/working/test_preds_xgb_new.npy', test_preds_xgb)


# Ä�Ã¡nh giÃ¡ OOF AUC
oof_auc_xgb = roc_auc_score(y, oof_preds_xgb)
print(f"XGBoost OOF AUC: {oof_auc_xgb:.6f}")


# Hiá»ƒn thá»‹ Ä‘áº·c trÆ°ng quan trá»�ng (Náº¿u báº¡n muá»‘n, cáº§n xá»­ lÃ½ pháº§n lÆ°u importance á»Ÿ trÃªn)
mean_importance_xgb = feature_importance_df_xgb[["Feature", "Importance"]].groupby("Feature").mean().sort_values(by="Importance", ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=mean_importance_xgb.Importance.head(20), y=mean_importance_xgb.head(20).index)
plt.title("XGBoost Feature Importance (Average over folds)")
plt.tight_layout()
plt.show()


del oof_preds_xgb, feature_importance_df_xgb, test_preds_xgb
gc.collect()


# Thiáº¿t láº­p KFold
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)


# Khá»Ÿi táº¡o máº£ng dá»± Ä‘oÃ¡n
oof_preds_cat = np.zeros(X.shape[0], dtype=np.float32)
test_preds_cat = np.zeros(X_test.shape[0], dtype=np.float32)
feature_importance_df_cat = pd.DataFrame()


# Thiáº¿t láº­p tham sá»‘ CatBoost
cat_params = {
    "iterations": 10000,
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 10,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.7,
    "scale_pos_weight": 1,
    "eval_metric": "AUC",
    "metric_period": 100,
    "od_type": "Iter",
    "od_wait": 200,
    "random_seed": 42,
    "allow_writing_files": False,
    "task_type": "GPU",
    "verbose": 100
}


# Náº¿u muá»‘n cháº¡y tá»‘t hÆ¡n trÃªn catboost, hÃ£y xem xÃ©t thÃªm cat_features á»Ÿ pháº§n fit
# nhÆ°ng chÃº Ã½ ráº±ng cÃ¡c features Ä‘Æ°á»£c chá»‰ Ä‘á»‹nh nÃ y chá»‰ cÃ³ thá»ƒ thuá»™c kiá»ƒu int hoáº·c str


print("\nTraining CatBoost...")
# VÃ²ng láº·p qua cÃ¡c fold
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"Current Fold: {fold_+1}")
    X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Huáº¥n luyá»‡n mÃ´ hÃ¬nh
    clf = CatBoostClassifier(**cat_params)
    clf.fit(
        X_trn, y_trn,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False
    )

    # Dá»± Ä‘oÃ¡n OOF
    oof_preds_cat[val_idx] = clf.predict_proba(X_val)[:, 1]
    
    # Dá»± Ä‘oÃ¡n trÃªn táº­p test
    test_preds_cat += clf.predict_proba(X_test)[:, 1] / NFOLDS

    # LÆ°u feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = X.columns
    fold_importance_df["Importance"] = clf.get_feature_importance()
    fold_importance_df["Fold"] = fold_ + 1
    feature_importance_df_cat = pd.concat([feature_importance_df_cat, fold_importance_df], axis=0)

    # Giáº£i phÃ³ng bá»™ nhá»›
    del clf, X_trn, y_trn, X_val, y_val, fold_importance_df
    gc.collect()


# LÆ°u káº¿t quáº£ dá»± Ä‘oÃ¡n
# np.save('/kaggle/working/oof_preds_cat_new.npy', oof_preds_cat)
np.save('/kaggle/working/test_preds_cat_new.npy', test_preds_cat)


# Ä�Ã¡nh giÃ¡ OOF AUC
oof_auc_cat = roc_auc_score(y, oof_preds_cat)
print(f"CatBoost OOF AUC: {oof_auc_cat:.6f}")


# Hiá»ƒn thá»‹ Ä‘áº·c trÆ°ng quan trá»�ng trung bÃ¬nh
mean_importance_cat = feature_importance_df_cat[["Feature", "Importance"]].groupby("Feature").mean().sort_values(by="Importance", ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=mean_importance_cat.Importance.head(20), y=mean_importance_cat.head(20).index)
plt.title("CatBoost Feature Importance (Average over folds)")
plt.tight_layout()
plt.show()


del oof_preds_cat, feature_importance_df_cat, test_preds_cat
gc.collect()


'''# In OOF AUC cá»§a tá»«ng mÃ´ hÃ¬nh
print(f"LightGBM OOF AUC: {oof_auc_lgb:.6f}")
print(f"XGBoost OOF AUC: {oof_auc_xgb:.6f}")
print(f"CatBoost OOF AUC: {oof_auc_cat:.6f}")

# Káº¿t há»£p dá»± Ä‘oÃ¡n OOF Ä‘á»ƒ tÃ¬m trá»�ng sá»‘ (vÃ­ dá»¥ Ä‘Æ¡n giáº£n)
# Báº¡n cÃ³ thá»ƒ dÃ¹ng cÃ¡c phÆ°Æ¡ng phÃ¡p tá»‘i Æ°u hÃ³a phá»©c táº¡p hÆ¡n (nhÆ° scipy.optimize)
# Hoáº·c dá»±a vÃ o kinh nghiá»‡m/thá»­ nghiá»‡m Ä‘á»ƒ chá»�n trá»�ng sá»‘
oof_combined = 0.4 * oof_preds_lgb + 0.4 * oof_preds_xgb + 0.2 * oof_preds_cat
oof_auc_combined = roc_auc_score(y, oof_combined)
print(f"Combined OOF AUC (0.4 LGB, 0.4 XGB, 0.2 CAT): {oof_auc_combined:.6f}")'''


# XÃ³a cÃ¡c biáº¿n lá»›n tá»« cÃ¡c bÆ°á»›c trÆ°á»›c (náº¿u cÃ²n)
try:
    del X, X_test, y
    gc.collect()
except NameError:
    pass


# Load láº¡i cÃ¡c máº£ng dá»± Ä‘oÃ¡n Ä‘Ã£ lÆ°u trá»¯
test_preds_lgb = np.load('/kaggle/input/model-for-ieee-fraud-detection/test_preds_lgb_new.npy').astype(np.float32)
test_preds_xgb = np.load('/kaggle/input/model-for-ieee-fraud-detection/test_preds_xgb_new.npy').astype(np.float32)
test_preds_cat = np.load('/kaggle/input/model-for-ieee-fraud-detection/test_preds_cat_new.npy').astype(np.float32)


# Káº¿t há»£p dá»± Ä‘oÃ¡n trÃªn táº­p test
weights = [0.1, 0.6, 0.3]  # LGB, XGB, CAT
test_pred_final = weights[0] * test_preds_lgb + weights[1] * test_preds_xgb + weights[2] * test_preds_cat


# Load TransactionID tá»« file test_transaction.csv, chá»‰ láº¥y cá»™t cáº§n thiáº¿t
transaction_ids = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv", usecols=["TransactionID"])


# Táº¡o file ná»™p bÃ i
submission = pd.DataFrame({
    "TransactionID": transaction_ids["TransactionID"].astype(np.int32),
    "isFraud": test_pred_final
})


# LÆ°u file ná»™p bÃ i
submission.to_csv("submission_kfold.csv", index=False)


# Hiá»ƒn thá»‹ thÃ´ng tin vá»� file ná»™p bÃ i
print("\nThÃ´ng tin vá»� file ná»™p bÃ i:")
print(submission.describe())
print(f"Sá»‘ lÆ°á»£ng giao dá»‹ch: {len(submission)}")


# Váº½ biá»ƒu Ä‘á»“ phÃ¢n bá»‘ dá»± Ä‘oÃ¡n
plt.figure(figsize=(12, 6))
plt.hist(submission["isFraud"], bins=50)
plt.title("PhÃ¢n bá»‘ dá»± Ä‘oÃ¡n cuá»‘i cÃ¹ng")
plt.xlabel("XÃ¡c suáº¥t gian láº­n")
plt.ylabel("Sá»‘ lÆ°á»£ng giao dá»‹ch")
plt.show()


# Giáº£i phÃ³ng bá»™ nhá»›
del test_preds_lgb, test_preds_xgb, test_preds_cat, test_pred_final, transaction_ids, submission
gc.collect()




