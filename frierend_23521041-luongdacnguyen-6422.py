# load cÃ¡c thÆ° viá»‡n
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, mode
from sklearn.preprocessing import LabelEncoder
from datetime import time


# Load data
base_path = "/kaggle/input/ds-108-p-21-assigment-06"

# Ä�á»�c tá»«ng file CSV vÃ o DataFrame
df_delay_4_6 = pd.read_csv(f"{base_path}/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv", encoding='utf-8')
df_not_delay_4_6 = pd.read_csv(f"{base_path}/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv", encoding='utf-8')

print(df_delay_4_6.shape)
print(df_not_delay_4_6.shape)


# Ná»‘i delay vÃ  not delay thÃ nh data hoÃ n chá»‰nh
df_full_4_6 = pd.concat([df_delay_4_6, df_not_delay_4_6], ignore_index=True)
print(df_full_4_6.shape)


# Load codebook
codebook = pd.read_excel(
    "/kaggle/input/m-company-delay-prediction/Sample codebook of Delay Prediction task.xlsx",
    sheet_name="Columns Details",
    header=0,
    index_col=0
)


print(f"sá»‘ thuá»™c tÃ­nh Ä‘Æ°á»£c liá»‡t kÃª trong codebook: {codebook.shape[0]}")
print(f"sá»‘ thuá»™c tÃ­nh cá»§a df_full_4_6: {df_full_4_6.shape[1]}")


from datetime import datetime

# Ghi láº¡i thá»�i Ä‘iá»ƒm ingest hiá»‡n táº¡i
ingest_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("Ingest timestamp:", ingest_timestamp)

# Option: gáº¯n timestamp vÃ o toÃ n bá»™ DataFrame
codebook['ingest_timestamp'] = ingest_timestamp
df_full_4_6['ingest_timestamp'] = ingest_timestamp


# Version theo thá»�i gian ingest
data_version = "v_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
print("Data version:", data_version)

# CÃ³ thá»ƒ thÃªm vÃ o metadata hoáº·c tÃªn file lÆ°u
codebook['data_version'] = data_version
df_full_4_6['data_version'] = data_version


def check_variables_in_information(codebook, df_full_7_9):
    """
    Kiá»ƒm tra trong df_full_7_9 cÃ³ bao nhiÃªu biáº¿n Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong codebook,
    vÃ  bao nhiÃªu biáº¿n khÃ´ng Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a (unexpected), bá»� qua hai cá»™t ká»¹ thuáº­t.

    Args:
    - codebook: DataFrame chá»©a danh sÃ¡ch cÃ¡c biáº¿n há»£p lá»‡ trong cá»™t 'Attributions'
    - df_full_7_9: DataFrame chá»©a dá»¯ liá»‡u thá»±c táº¿

    Returns:
    - None (chá»‰ in ra thÃ´ng tin chi tiáº¿t)
    """
    # Táº­p biáº¿n há»£p lá»‡ Ä‘Ã£ Ä‘á»‹nh nghÄ©a
    defined_variables = set(codebook["Attributions"].str.strip())

    # CÃ¡c cá»™t cáº§n bá»� qua trong quÃ¡ trÃ¬nh kiá»ƒm tra
    ignored_columns = {"data_version", "ingest_timestamp"}

    # Táº­p biáº¿n thá»±c táº¿ trong dá»¯ liá»‡u (sau khi loáº¡i bá»� cÃ¡c cá»™t bá»‹ bá»� qua)
    info_variables = set(df_full_7_9.columns.str.strip()) - ignored_columns

    # Biáº¿n cÃ³ trong df_full_7_9 mÃ  cÅ©ng Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a
    matched_variables = info_variables & defined_variables

    # Biáº¿n cÃ³ trong df_full_7_9 nhÆ°ng KHÃ”NG Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a
    unexpected_variables = info_variables - defined_variables

    print(f"Tá»•ng sá»‘ biáº¿n trong df_full_7_9 (bá»� qua cá»™t ká»¹ thuáº­t): {len(info_variables)}")
    print(f"Sá»‘ biáº¿n Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong codebook: {len(matched_variables)}")
    print(f"Sá»‘ biáº¿n KHÃ”NG Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong codebook: {len(unexpected_variables)}")

    print("\nDanh sÃ¡ch biáº¿n Ä�Æ¯á»¢C Ä‘á»‹nh nghÄ©a:")
    print(sorted(matched_variables))

    if unexpected_variables:
        print("\nğŸ”º Danh sÃ¡ch biáº¿n KHÃ”NG Ä�Æ¯á»¢C Ä‘á»‹nh nghÄ©a:")
        print(sorted(unexpected_variables))
    else:
        print("\nğŸ�‰ KhÃ´ng cÃ³ biáº¿n nÃ o báº¥t ngá»�! Má»�i thá»© Ä‘á»�u Ä‘Ãºng chuáº©n Ä‘á»‹nh nghÄ©a.")


check_variables_in_information(codebook, df_full_4_6)


# Táº¡o DataFrame chá»©a 10 dÃ²ng má»›i
new_rows = pd.DataFrame([
    {
        'Attributions': 'ACTUAL_SHIP_DAYS',
        'Data Type': 'int64',
        'Group Attributions': 'ThÃ´ng tin váº­n chuyá»ƒn',
        'Description': 'Sá»‘ ngÃ y lÃ m viá»‡c thá»±c táº¿ tá»« thá»�i Ä‘iá»ƒm Ä‘áº·t hÃ ng Ä‘áº¿n ngÃ y dá»± kiáº¿n giao (VSD)',
        'Miá»�n giÃ¡ trá»‹': None
    },
    {
        'Attributions': 'EXPENSIVE_FLG',
        'Data Type': 'int64',
        'Group Attributions': 'ThÃ´ng tin sáº£n pháº©m',
        'Description': 'Cá»� Ä‘Ã¡nh dáº¥u máº·t hÃ ng cÃ³ giÃ¡ trá»‹ cao (1: Ä‘áº¯t | 0: khÃ´ng Ä‘áº¯t)',
        'Miá»�n giÃ¡ trá»‹': None
    },
    {
        'Attributions': 'HAZARD_FLG',
        'Data Type': 'int64',
        'Group Attributions': 'ThÃ´ng tin sáº£n pháº©m',
        'Description': 'Cá»� Ä‘Ã¡nh dáº¥u máº·t hÃ ng nguy hiá»ƒm (1: cÃ³ nguy cÆ¡ | 0: an toÃ n bÃ¬nh thÆ°á»�ng)',
        'Miá»�n giÃ¡ trá»‹': None
    },
    {
        'Attributions': 'HEAVY_FLG',
        'Data Type': 'int64',
        'Group Attributions': 'ThÃ´ng tin sáº£n pháº©m',
        'Description': 'Cá»� Ä‘Ã¡nh dáº¥u máº·t hÃ ng náº·ng (1: náº·ng | 0: nháº¹)',
        'Miá»�n giÃ¡ trá»‹': None
    },
    {
        'Attributions': 'IO_UNFIT_FLG',
        'Data Type': 'int64',
        'Group Attributions': 'MÃ£ lÃ½ do vÃ  lá»—i',
        'Description': 'Cá»� Ä‘Ã¡nh dáº¥u â€œunfitâ€� ná»™i bá»™ khi giao dá»‹ch liÃªn cÃ´ng ty, tá»± Ä‘á»™ng loáº¡i bá»� cÃ¡c dÃ²ng hÃ ng khÃ´ng phÃ¹ há»£p',
        'Miá»�n giÃ¡ trá»‹': None
    },
    {
        'Attributions': 'PRODUCT_ASSORT',
        'Data Type': 'object',
        'Group Attributions': 'ThÃ´ng tin sáº£n pháº©m',
        'Description': 'Loáº¡i sáº£n pháº©m, phÃ¢n nhÃ³m theo tÃ­nh cháº¥t vÃ  dÃ²ng hÃ ng',
        'Miá»�n giÃ¡ trá»‹': None
    },
    {
        'Attributions': 'SPECIFY_PRODUCTION_DAYS',
        'Data Type': 'int64',
        'Group Attributions': 'ThÃ´ng tin sáº£n pháº©m',
        'Description': 'Sá»‘ ngÃ y sáº£n xuáº¥t Ä‘Æ°á»£c chá»‰ Ä‘á»‹nh bá»Ÿi bá»™ pháº­n front desk (chá»‰ set khi cÃ³ yÃªu cáº§u Ä‘áº·c biá»‡t)',
        'Miá»�n giÃ¡ trá»‹': None
    },
    {
        'Attributions': 'SPECIFY_SHIP_DAYS',
        'Data Type': 'int64',
        'Group Attributions': 'ThÃ´ng tin váº­n chuyá»ƒn',
        'Description': 'Sá»‘ ngÃ y váº­n chuyá»ƒn Ä‘Æ°á»£c chá»‰ Ä‘á»‹nh bá»Ÿi bá»™ pháº­n front desk (chá»‰ set khi cÃ³ yÃªu cáº§u Ä‘áº·c biá»‡t)',
        'Miá»�n giÃ¡ trá»‹': None
    },
    {
        'Attributions': 'SUPPLIER_CATEGORY_CD',
        'Data Type': 'int64',
        'Group Attributions': 'ThÃ´ng tin nhÃ  cung cáº¥p',
        'Description': 'MÃ£ loáº¡i Supplier (phÃ¢n nhÃ³m nhÃ  cung cáº¥p theo danh má»¥c ná»™i bá»™)',
        'Miá»�n giÃ¡ trá»‹': None
    },
    {
        'Attributions': 'WEIGHT_UNIT',
        'Data Type': 'object',
        'Group Attributions': 'ThÃ´ng tin sáº£n pháº©m',
        'Description': 'Ä�Æ¡n vá»‹ khá»‘i lÆ°á»£ng (g, kgâ€¦)',
        'Miá»�n giÃ¡ trá»‹': None
    }
])

# Táº¡o codebook_full = codebook cÅ© + 10 dÃ²ng má»›i
codebook_full = pd.concat([codebook, new_rows], ignore_index=True)

# Giá»¯ láº¡i Ä‘Ãºng 5 cá»™t mong muá»‘n
codebook_full = codebook_full[['Attributions', 'Data Type', 'Group Attributions', 'Description', 'Miá»�n giÃ¡ trá»‹']]

# Kiá»ƒm tra vÃ  Chuáº©n hÃ³a kiá»ƒu dá»¯ liá»‡u trong Codebook cho phÃ¹ há»£p

# Cáº­p nháº­t giÃ¡ trá»‹ trong cá»™t 'Data Type' náº¿u Attributions lÃ  order date hoáº·c 'VSD' 
codebook_full.loc[codebook_full['Attributions'] == 'VSD', 'Data Type'] = 'DateTime'
codebook_full.loc[codebook_full['Attributions'] == 'Order date', 'Data Type'] = 'DateTime'

# Xuáº¥t ra file Excel
codebook_full.sort_values(by='Group Attributions', ascending=False).to_excel('codebook_full_final.xlsx', index=False)


codebook_full.tail(5)


import os

output_dir = 'Data for practice'
os.mkdir(output_dir)

output_path = os.path.join(output_dir, 'df_full_4_6.csv')
df_full_4_6.to_csv(output_path, index=False)

print(f"ğŸ“„ File CSV Ä‘Ã£ Ä‘Æ°á»£c lÆ°u táº¡i: {output_path}")


# In ra tÃªn cá»™t vÃ  dtype tÆ°Æ¡ng á»©ng trong DataFrame df_full_4_6
codebook_full[['Attributions', 'Data Type']].head()


def convert_column_dtypes_from_codebook(df: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyá»ƒn Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cá»§a cÃ¡c cá»™t trong DataFrame Ä‘áº§u vÃ o dá»±a trÃªn Ä‘á»‹nh nghÄ©a tá»« codebook.

    Args:
        df (pd.DataFrame): DataFrame chá»©a dá»¯ liá»‡u gá»‘c cáº§n chuyá»ƒn kiá»ƒu dá»¯ liá»‡u (vd: df_full_4_6).
        codebook (pd.DataFrame): DataFrame Ä‘á»‹nh nghÄ©a kiá»ƒu dá»¯ liá»‡u cho tá»«ng cá»™t, 
            yÃªu cáº§u cÃ³ cÃ¡c cá»™t:
                - 'Attributions': tÃªn cá»™t trong df
                - 'Data Type': kiá»ƒu dá»¯ liá»‡u Ä‘Ã­ch (int64, float, object, DateTime...)

    Returns:
        pd.DataFrame: DataFrame Ä‘áº§u vÃ o sau khi Ä‘Ã£ cá»‘ gáº¯ng chuyá»ƒn Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cÃ¡c cá»™t tÆ°Æ¡ng á»©ng.

    In ra:
        - âœ… Vá»›i má»—i cá»™t Ä‘á»•i thÃ nh cÃ´ng â†’ cÃ³ thá»ƒ in ra tÃªn cá»™t vÃ  kiá»ƒu Ä‘Ã£ chuyá»ƒn
        - â�Œ Vá»›i má»—i cá»™t lá»—i â†’ in ra tÃªn cá»™t, kiá»ƒu Ä‘á»‹nh chuyá»ƒn, vÃ  lÃ½ do lá»—i (exception message)

    Notes:
        - CÃ¡c kiá»ƒu dá»¯ liá»‡u há»— trá»£ gá»“m: int, float, object/string, DateTime
        - Vá»›i kiá»ƒu int sá»­ dá»¥ng kiá»ƒu nullable 'Int64' cá»§a pandas Ä‘á»ƒ giá»¯ láº¡i giÃ¡ trá»‹ NaN náº¿u cÃ³
        - Náº¿u kiá»ƒu dá»¯ liá»‡u khÃ´ng Ä‘Æ°á»£c nháº­n diá»‡n hoáº·c lá»—i Ã©p kiá»ƒu â†’ cá»™t sáº½ bá»‹ bá»� qua

    VÃ­ dá»¥ dÃ¹ng:
        df_full_4_6 = convert_column_dtypes_from_codebook(df_full_4_6, codebook_full)
    """
    for _, row in codebook.iterrows():
        col = row['Attributions']
        dtype = row['Data Type']

        try:
            if dtype.lower() in ['int', 'int64']:
                df[col] = pd.to_numeric(df[col], errors='raise').astype('Int64')
            elif dtype.lower() in ['float', 'float64']:
                df[col] = pd.to_numeric(df[col], errors='raise').astype('float')
            elif dtype.lower() in ['object', 'string']:
                df[col] = df[col].astype(str)
            elif dtype.lower() in ['datetime', 'datetime64', 'datetime64[ns]']:
                try:
                    # Cá»‘ gáº¯ng parse kiá»ƒu ISO8601 trÆ°á»›c
                    df[col] = pd.to_datetime(df[col], format='ISO8601', errors='raise')
                    print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ DateTime (ISO8601)")
                except Exception as e_iso:
                    try:
                        # Cá»‘ gáº¯ng vá»›i format kiá»ƒu Excel: "4/1/2022  12:00:00 AM"
                        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y %I:%M:%S %p", errors='raise')
                        print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ DateTime (Excel-style)")
                    except Exception as e_excel:
                        try:
                            # Cuá»‘i cÃ¹ng lÃ  mixed
                            df[col] = pd.to_datetime(df[col], format='mixed', errors='raise')
                            print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ DateTime (mixed format)")
                        except Exception as e_mixed:
                            print(f"Lá»—i khi Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cho cá»™t: {col} â†’ DateTime | ISO8601: {e_iso} | Excel: {e_excel} | mixed: {e_mixed}")

            else:
                print(f"KhÃ´ng rÃµ kiá»ƒu dá»¯ liá»‡u '{dtype}' cho cá»™t '{col}', bá»� qua.")
                continue

            print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ {dtype}")
        
        except Exception as e:
            print(f"Lá»—i khi Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cho cá»™t: {col} â†’ {dtype} | LÃ½ do: {e}")

    return df


df_full_4_6 = convert_column_dtypes_from_codebook(df_full_4_6, codebook_full)


df_full_4_6['OTHER AREA SHIP DIV'] = df_full_4_6['OTHER AREA SHIP DIV'].replace(' ', 0).fillna(0)


df_full_4_6 = df_full_4_6.drop(df_full_4_6.index[170932])


print(df_full_4_6['REASON_CD'].unique())  


# Náº¿u muá»‘n convert cá»™t sang float (cáº©n tháº­n lá»—i), cÃ³ thá»ƒ lÃ m sáº¡ch trÆ°á»›c:
df_full_4_6['REASON_CD'] = df_full_4_6['REASON_CD'].replace(r'^\s*$', pd.NA, regex=True)  # thay chuá»—i trá»‘ng/cÃ¡ch báº±ng NA

# Tuy nhiÃªn vÃ¬ cá»™t nÃ y lÃ  dá»¯ liá»‡u tÆ°Æ¡ng lai, ta sáº½ loáº¡i bá»�:
df_full_4_6 = df_full_4_6.drop(columns=['REASON_CD'])  # loáº¡i bá»� hoÃ n toÃ n cá»™t REASON_CD
codebook_full = codebook_full[codebook_full['Attributions'] != 'REASON_CD']


df_full_4_6 = convert_column_dtypes_from_codebook(df_full_4_6, codebook_full)


# TÃ¡ch Order date
df_full_4_6['order_year'] = df_full_4_6['Order date'].dt.year
df_full_4_6['order_month'] = df_full_4_6['Order date'].dt.month
df_full_4_6['order_dayofweek'] = df_full_4_6['Order date'].dt.dayofweek
df_full_4_6['is_order_weekend'] = df_full_4_6['order_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# TÃ¡ch VSD
df_full_4_6['vsd_year'] = df_full_4_6['VSD'].dt.year
df_full_4_6['vsd_month'] = df_full_4_6['VSD'].dt.month
df_full_4_6['vsd_dayofweek'] = df_full_4_6['VSD'].dt.dayofweek
df_full_4_6['is_vsd_weekend'] = df_full_4_6['vsd_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# TÃ­nh khoáº£ng cÃ¡ch giá»¯a VSD vÃ  Order date (náº¿u há»£p lÃ½ vá»� logic nghiá»‡p vá»¥)
df_full_4_6['days_to_vsd'] = (df_full_4_6['VSD'] - df_full_4_6['Order date']).dt.days


cols_to_convert = ['SUPPLIER_CATEGORY_CD', 'CLASSIFY_CD', 'CUST_CD', 'SHIP DECISION NO']
df_full_4_6[cols_to_convert] = df_full_4_6[cols_to_convert].astype('object')


df_full_4_6['SOUF_RCV_NO'].value_counts()[:3]


df_full_4_6['Ship Mode'].value_counts()


# Thay cÃ¡c chuá»—i 'nan' (chuá»—i chá»¯, khÃ´ng pháº£i NaN tháº­t sá»±) thÃ nh np.nan
df_full_4_6['SOUF_RCV_NO'] = df_full_4_6['SOUF_RCV_NO'].replace('nan', np.nan)
df_full_4_6['Ship Mode'] = df_full_4_6['Ship Mode'].replace('nan', np.nan)


def convert_to_time(x):
    try:
        x_str = str(int(x)).zfill(6)
        hour = int(x_str[:2])
        minute = int(x_str[2:4])
        second = int(x_str[4:6])
        return time(hour, minute, second)
    except:
        return pd.NaT

# Ã�p dá»¥ng
df_full_4_6['SO_TIME'] = df_full_4_6['SO_TIME'].apply(convert_to_time)


# Kiá»ƒm tra káº¿t quáº£
print("âœ… Ä�Ã£ chuáº©n hÃ³a cá»™t 'SO_TIME' vá»� kiá»ƒu thá»�i gian thá»±c táº¿.")
df_full_4_6['SO_TIME'].head()


# Ä�áº£m báº£o cá»™t SO_TIME Ä‘Ã£ lÃ  kiá»ƒu datetime.time
# Náº¿u chÆ°a: df_full_4_6['SO_TIME'] = df_full_4_6['SO_TIME'].apply(convert_to_time)

# Táº¡o Ä‘áº·c trÆ°ng tá»« SO_TIME
df_full_4_6['so_hour'] = df_full_4_6['SO_TIME'].apply(lambda x: x.hour if pd.notnull(x) else np.nan)
df_full_4_6['so_minute'] = df_full_4_6['SO_TIME'].apply(lambda x: x.minute if pd.notnull(x) else np.nan)
df_full_4_6['so_second'] = df_full_4_6['SO_TIME'].apply(lambda x: x.second if pd.notnull(x) else np.nan)

# Buá»•i sÃ¡ng hay chiá»�u
df_full_4_6['is_morning'] = df_full_4_6['so_hour'].apply(lambda x: 1 if x < 12 else 0 if pd.notnull(x) else np.nan)

# GÃ¡n time slot theo khung giá»� sinh hoáº¡t
def assign_time_slot(hour):
    if pd.isnull(hour): return np.nan
    elif 0 <= hour < 6: return 'midnight'
    elif 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 18: return 'afternoon'
    else: return 'evening'

df_full_4_6['time_slot'] = df_full_4_6['so_hour'].apply(assign_time_slot)

# Binned theo khung 4 tiáº¿ng
df_full_4_6['time_bin'] = pd.cut(df_full_4_6['so_hour'],
                                 bins=[-1, 3, 7, 11, 15, 19, 23],
                                 labels=['0-3h', '4-7h', '8-11h', '12-15h', '16-19h', '20-23h'])



def print_data_overview(df, columns=None):
    if columns is not None:
        df = df[columns]

    num_rows = df.shape[0]
    num_cols = df.shape[1]
    total_cells = num_rows * num_cols
    percent_missing = df.isnull().sum().sum() / total_cells * 100
    percent_duplicate = df.duplicated().sum() / num_rows * 100

    print(f"Tá»•ng quan: {num_rows} dÃ²ng, {num_cols} cá»™t")
    print(f"âš ï¸�  {percent_missing:.4f}% missing | ğŸ”� {percent_duplicate:.4f}% duplicate\n")


# Tá»•ng quan dá»¯ liá»‡u
print_data_overview(df_full_4_6)


print(df_full_4_6.select_dtypes(exclude='object').columns)


df_full_4_6.describe()


df_full_4_6.isnull().sum().sort_values(ascending=False)[:5]


df_full_4_6.duplicated().sum()


missing_info = df_full_4_6.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_4_6)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


# Tá»•ng quan dá»¯ liá»‡u
print_data_overview(df_full_4_6)


# Chuyá»ƒn 'Order date' lÃ m index
df_plot = df_full_4_6.set_index('Order date')

# Resample theo ngÃ y (cÃ³ thá»ƒ thay báº±ng 'W' cho tuáº§n hoáº·c 'M' cho thÃ¡ng náº¿u muá»‘n smooth hÆ¡n)
label_1_by_day = df_plot[df_plot['label'] == 1].resample('D').size()

plt.figure(figsize=(14, 6))
plt.plot(label_1_by_day.index, label_1_by_day.values, color='red', label='Sá»‘ lÆ°á»£ng label = 1')

plt.title('Sá»‘ láº§n xuáº¥t hiá»‡n label = 1 theo thá»�i gian')
plt.xlabel('Thá»�i gian')
plt.ylabel('Táº§n suáº¥t')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Ä�áº£m báº£o cá»™t 'Order date' lÃ  datetime
df_full_4_6['Order date'] = pd.to_datetime(df_full_4_6['Order date'])

df_plot = df_full_4_6.set_index('Order date')

# Resample theo ngÃ y (cÃ³ thá»ƒ Ä‘á»•i 'D' -> 'W' hoáº·c 'M' náº¿u cáº§n mÆ°á»£t hÆ¡n)
label_1_by_day = df_plot[df_plot['label'] == 1].resample('D').size()
label_0_by_day = df_plot[df_plot['label'] == 0].resample('D').size()

plt.figure(figsize=(14, 6))
plt.plot(label_1_by_day.index, label_1_by_day.values, color='red', label='Label = 1')
plt.plot(label_0_by_day.index, label_0_by_day.values, color='blue', label='Label = 0')

plt.title('Táº§n suáº¥t label = 1 (Ä‘á»�) vÃ  label = 0 (xanh) theo thá»�i gian')
plt.xlabel('Thá»�i gian')
plt.ylabel('Sá»‘ lÆ°á»£ng')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Láº¥y top 50 GLOBAL_NO xuáº¥t hiá»‡n nhiá»�u nháº¥t
top_50_global_no = df_full_4_6['GLOBAL_NO'].value_counts().head(50)

global_no_labels = df_full_4_6.groupby('GLOBAL_NO')['label'].min()

colors = ['red' if global_no_labels[global_no] == 1 else 'blue' for global_no in top_50_global_no.index]

plt.figure(figsize=(18, 8))
bars = plt.bar(top_50_global_no.index.astype(str), top_50_global_no.values, color=colors)

# Gáº¯n nhÃ£n sá»‘ lÆ°á»£ng lÃªn tá»«ng cá»™t
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
             str(int(height)), ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.title('Top 50 GLOBAL_NO - PhÃ¢n mÃ u theo Label (1: Ä‘á»�, 0: xanh)', fontsize=14)
plt.xlabel('GLOBAL_NO')
plt.ylabel('Sá»‘ láº§n xuáº¥t hiá»‡n')
plt.xticks(rotation=75, ha='right', fontsize=8)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



grouped = df_full_4_6.groupby(['SUPPLIER_CD', 'label']).size().unstack(fill_value=0)


top10_suppliers = grouped.sum(axis=1).sort_values(ascending=False).head(20)

top_grouped = grouped.loc[top10_suppliers.index]
plt.figure(figsize=(10, 6))

plt.bar(top_grouped.index.astype(str), top_grouped[0], label='Label 0', color='lightcoral')
plt.bar(top_grouped.index.astype(str), top_grouped[1], bottom=top_grouped[0], label='Label 1', color='cornflowerblue')

plt.xlabel('SUPPLIER_CD')
plt.ylabel('Sá»‘ Ä‘Æ¡n hÃ ng')
plt.title('Stacked Bar Chart: Top 20 Supplier theo Label')
plt.xticks(rotation=45)
plt.legend(title='Label')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


vc = df_full_4_6['SPECIFY_SHIP_DAYS'].value_counts()
df_ngang = vc.to_frame().T
df_ngang.reset_index(drop=True, inplace=True)

df_ngang


vc = df_full_4_6['ACTUAL_SHIP_DAYS'].value_counts()
df_ngang = vc.to_frame().T
df_ngang.reset_index(drop=True, inplace=True)

df_ngang


grouped = df_full_4_6.groupby(['ACTUAL_SHIP_DAYS', 'label']).size().unstack(fill_value=0)
grouped = grouped.sort_index().head(50)

plt.figure(figsize=(14, 6))

plt.bar(grouped.index.astype(str), grouped[0], label='Label 0', color='orchid')
plt.bar(grouped.index.astype(str), grouped[1], bottom=grouped[0], label='Label 1', color='mediumslateblue')

plt.xlabel('ACTUAL_SHIP_DAYS')
plt.ylabel('Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng')
plt.title('Stacked Bar Chart: PhÃ¢n phá»‘i sá»‘ ngÃ y giao hÃ ng thá»±c táº¿ theo Label')
plt.xticks(rotation=45)
plt.legend(title='Label')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



grouped = df_full_4_6.groupby(['SPECIFY_SHIP_DAYS', 'label']).size().unstack(fill_value=0)

# BÆ°á»›c 2: Giá»›i háº¡n top 50 ngÃ y Ä‘áº§u tiÃªn (Ä‘áº£m báº£o Ä‘Ã£ sort)
grouped = grouped.sort_index().head(50)

# BÆ°á»›c 3: Váº½ stacked bar chart
plt.figure(figsize=(14, 6))

plt.bar(grouped.index.astype(str), grouped[0], label='Label 0', color='orchid')
plt.bar(grouped.index.astype(str), grouped[1], bottom=grouped[0], label='Label 1', color='mediumslateblue')

plt.xlabel('SPECIFY_SHIP_DAYS')
plt.ylabel('Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng')
plt.title('Stacked Bar Chart: PhÃ¢n phá»‘i sá»‘ ngÃ y Sá»‘ ngÃ y váº­n chuyá»ƒn Ä‘Æ°á»£c chá»‰ Ä‘á»‹nh bá»Ÿi bá»™ pháº­n front desk theo Label')
plt.xticks(rotation=45)
plt.legend(title='Label')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# CÃ¡c cá»™t cáº§n váº½
cols = ['DIRECT SHIP FLG', 'DELI_DIV', 'Ship Mode']
label_col = 'label'

# Táº¡o Figure vá»›i 3 subplot náº±m trÃªn 1 dÃ²ng
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, col in enumerate(cols):
    ax = axes[i]
    
    # NhÃ³m vÃ  Ä‘áº¿m theo giÃ¡ trá»‹ + label
    grouped = df_full_4_6.groupby([col, label_col]).size().unstack(fill_value=0)

    # Ä�áº£m báº£o cÃ³ cáº£ label 0 vÃ  1
    if 0 not in grouped.columns:
        grouped[0] = 0
    if 1 not in grouped.columns:
        grouped[1] = 0

    grouped = grouped[[0, 1]]  # Giá»¯ thá»© tá»±

    # TÃ­nh sá»‘ lÆ°á»£ng giÃ¡ trá»‹ unique
    n_unique = df_full_4_6[col].nunique()

    # Váº½ stacked bar chart
    ax.bar(grouped.index.astype(str), grouped[0], label='Label 0', color='lightcoral')
    ax.bar(grouped.index.astype(str), grouped[1], bottom=grouped[0], label='Label 1', color='steelblue')
    
    # Gáº¯n tiÃªu Ä‘á»� cÃ³ sá»‘ lÆ°á»£ng unique
    ax.set_title(f'{col} (Unique: {n_unique})')
    ax.set_xticklabels(grouped.index.astype(str), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

# GÃ¡n nhÃ£n chung
fig.suptitle('PhÃ¢n phá»‘i Label theo DIRECT_SHIP_FLG, DELI_DIV vÃ  Ship Mode', fontsize=16)
fig.text(0.5, 0.04, 'GiÃ¡ trá»‹', ha='center')
fig.text(0.04, 0.5, 'Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng', va='center', rotation='vertical')

# ThÃªm chÃº thÃ­ch
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout(rect=[0.03, 0.05, 0.95, 0.95])
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# CÃ¡c cá»™t flag cáº§n váº½ + 1 cá»™t rá»�i ráº¡c thÃªm
cols = ['HEAVY_FLG', 'HAZARD_FLG', 'EXPENSIVE_FLG', 'Stock class']
label_col = 'label'

# Táº¡o Figure vá»›i 4 subplot náº±m trÃªn 1 dÃ²ng
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

for i, col in enumerate(cols):
    ax = axes[i]
    
    # NhÃ³m vÃ  Ä‘áº¿m theo giÃ¡ trá»‹ + label
    grouped = df_full_4_6.groupby([col, label_col]).size().unstack(fill_value=0)

    # Ä�áº£m báº£o cÃ³ cáº£ label 0 vÃ  1
    for lbl in [0, 1]:
        if lbl not in grouped.columns:
            grouped[lbl] = 0

    grouped = grouped[[0, 1]]  # Ä�áº£m báº£o Ä‘Ãºng thá»© tá»±

    # TÃ­nh sá»‘ lÆ°á»£ng giÃ¡ trá»‹ unique
    n_unique = df_full_4_6[col].nunique()

    # Váº½ stacked bar chart
    x = grouped.index.astype(str)
    bar1 = ax.bar(x, grouped[0], label='Label 0', color='lightcoral')
    bar2 = ax.bar(x, grouped[1], bottom=grouped[0], label='Label 1', color='steelblue')
    
    # Gáº¯n sá»‘ lÆ°á»£ng lÃªn Ä‘áº§u má»—i cá»™t
    for j in range(len(x)):
        total = grouped[0].iloc[j] + grouped[1].iloc[j]
        ax.text(j, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Gáº¯n tiÃªu Ä‘á»� cÃ³ sá»‘ lÆ°á»£ng unique
    ax.set_title(f'{col} (Unique: {n_unique})')
    ax.set_xticklabels(x, rotation=30, ha='center')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

# GÃ¡n nhÃ£n chung
fig.suptitle('PhÃ¢n phá»‘i Label theo cÃ¡c cá»™t Flag vÃ  Stock Class', fontsize=16)
fig.text(0.5, 0.04, 'GiÃ¡ trá»‹', ha='center')
fig.text(0.04, 0.5, 'Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng', va='center', rotation='vertical')

# ThÃªm chÃº thÃ­ch
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout(rect=[0.03, 0.05, 0.95, 0.95])
plt.show()


top_weights = df_full_4_6['WEIGHT PER PIECE'].value_counts().head(30).index

# B2: Lá»�c dá»¯ liá»‡u theo top_weights
df_top = df_full_4_6[df_full_4_6['WEIGHT PER PIECE'].isin(top_weights)]

# B3: NhÃ³m theo weight vÃ  label
grouped = df_top.groupby(['WEIGHT PER PIECE', 'label']).size().unstack(fill_value=0)

# Ä�áº£m báº£o cÃ³ cáº£ label 0 vÃ  1
for lbl in [0, 1]:
    if lbl not in grouped.columns:
        grouped[lbl] = 0

grouped = grouped[[0, 1]]  # Ä�Ãºng thá»© tá»±

# B4: Váº½ stacked bar chart
plt.figure(figsize=(12, 6))
x = grouped.index.astype(str)

bar1 = plt.bar(x, grouped[0], label='Label 0 (KhÃ´ng trá»…)', color='mediumseagreen')
bar2 = plt.bar(x, grouped[1], bottom=grouped[0], label='Label 1 (Bá»‹ trá»…)', color='tomato')

# ThÃªm sá»‘ lÆ°á»£ng tá»•ng má»—i cá»™t
for i in range(len(x)):
    total = grouped[0].iloc[i] + grouped[1].iloc[i]
    plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.title('PhÃ¢n phá»‘i Label theo cÃ¡c trá»�ng lÆ°á»£ng phá»• biáº¿n nháº¥t')
plt.xlabel('WEIGHT PER PIECE')
plt.ylabel('Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


df_full_4_6['WEIGHT_UNIT'].value_counts()


top_10_modes = df_full_4_6['Ship Mode'].value_counts().head(20).index

# Lá»�c dá»¯ liá»‡u chá»‰ giá»¯ láº¡i cÃ¡c Ship Mode náº±m trong top 10
filtered_df = df_full_4_6[df_full_4_6['Ship Mode'].isin(top_10_modes)]

# Ä�áº¿m theo Ship Mode vÃ  label
shipmode_counts = filtered_df.groupby(['Ship Mode', 'label']).size().unstack(fill_value=0)

# Ä�áº£m báº£o Ä‘Ãºng thá»© tá»±
shipmode_counts = shipmode_counts.loc[top_10_modes]  # giá»¯ Ä‘Ãºng thá»© tá»± top 10
shipmode_counts = shipmode_counts[[0, 1]]  # label 0 trÆ°á»›c, rá»“i label 1

# Váº½ stacked bar chart
plt.figure(figsize=(12, 6))
x = shipmode_counts.index.astype(str)
bar1 = plt.bar(x, shipmode_counts[0], label='Label 0', color='mediumseagreen')
bar2 = plt.bar(x, shipmode_counts[1], bottom=shipmode_counts[0], label='Label 1', color='tomato')

# Gáº¯n sá»‘ lÆ°á»£ng lÃªn má»—i cá»™t
for i in range(len(x)):
    total = shipmode_counts[0].iloc[i] + shipmode_counts[1].iloc[i]
    plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Trang trÃ­
plt.title('ğŸš› PhÃ¢n phá»‘i Ship Mode theo Label (Top 10 phá»• biáº¿n nháº¥t)', fontsize=14)
plt.xlabel('Ship Mode')
plt.ylabel('Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


cols = ['SO_DAY_OF_WEEK', 'SO_DAY_OF_MONTH']
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

for i, col in enumerate(cols):
    ax = axes[i]
    
    # NhÃ³m dá»¯ liá»‡u theo giÃ¡ trá»‹ + label
    grouped = df_full_4_6.groupby([col, 'label']).size().unstack(fill_value=0)

    # Ä�áº£m báº£o cáº£ label 0 vÃ  1 Ä‘á»�u cÃ³
    for lbl in [0, 1]:
        if lbl not in grouped.columns:
            grouped[lbl] = 0

    grouped = grouped[[0, 1]]  # Ä�Ãºng thá»© tá»±
    
    # TÃ­nh sá»‘ lÆ°á»£ng giÃ¡ trá»‹ unique
    n_unique = df_full_4_6[col].nunique()
    
    # Váº½ stacked bar chart
    x = grouped.index.astype(str)
    bar1 = ax.bar(x, grouped[0], label='Label 0 (KhÃ´ng trá»…)', color='mediumseagreen')
    bar2 = ax.bar(x, grouped[1], bottom=grouped[0], label='Label 1 (Bá»‹ trá»…)', color='tomato')
    
    # In sá»‘ lÆ°á»£ng trÃªn má»—i cá»™t
    for j in range(len(x)):
        total = grouped[0].iloc[j] + grouped[1].iloc[j]
        ax.text(j, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Gáº¯n tiÃªu Ä‘á»�
    ax.set_title(f'{col} (Unique: {n_unique})')
    ax.set_xlabel(col)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

fig.suptitle('PhÃ¢n phá»‘i Ä�Æ¡n hÃ ng theo NgÃ y trong tuáº§n & NgÃ y trong thÃ¡ng', fontsize=16)
fig.text(0.5, 0.04, 'GiÃ¡ trá»‹ ngÃ y', ha='center')
fig.text(0.04, 0.5, 'Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng', va='center', rotation='vertical')

# ThÃªm chÃº thÃ­ch
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout(rect=[0.03, 0.05, 0.95, 0.95])
plt.show()



# BÆ°á»›c 1: TÃ­nh SO_HOUR táº¡m thá»�i, khÃ´ng thÃªm vÃ o df_full_4_6
temp_df = df_full_4_6.copy()
temp_df['SO_HOUR'] = temp_df['SO_TIME'].apply(lambda x: x.hour if pd.notna(x) else None)

# BÆ°á»›c 2: Group theo giá»� vÃ  label
hourly_counts = temp_df.groupby(['SO_HOUR', 'label']).size().unstack(fill_value=0)

# Ä�áº£m báº£o Ä‘á»§ 24 giá»� (ká»ƒ cáº£ giá»� khÃ´ng cÃ³ Ä‘Æ¡n)
all_hours = pd.Series(range(0, 24), name='SO_HOUR')
hourly_counts = hourly_counts.reindex(all_hours, fill_value=0)

# BÆ°á»›c 3: Váº½ stacked bar chart
plt.figure(figsize=(14, 6))
bars_0 = plt.bar(hourly_counts.index, hourly_counts[0], label='KhÃ´ng trá»… (label=0)', color='mediumseagreen')
bars_1 = plt.bar(hourly_counts.index, hourly_counts[1], bottom=hourly_counts[0], label='Trá»… (label=1)', color='tomato')

# Gáº¯n sá»‘ lÆ°á»£ng tá»•ng lÃªn Ä‘á»‰nh
for i in hourly_counts.index:
    total = hourly_counts.loc[i, 0] + hourly_counts.loc[i, 1]
    if total > 0:
        plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

# CÃ i Ä‘áº·t biá»ƒu Ä‘á»“
plt.xlabel('Giá»� Ä‘áº·t hÃ ng (0â€“23h)')
plt.ylabel('Sá»‘ Ä‘Æ¡n hÃ ng')
plt.title('PhÃ¢n phá»‘i Ä‘Æ¡n hÃ ng theo giá»� Ä‘áº·t (SO_TIME) vÃ  Label (Trá»… / KhÃ´ng trá»…)')
plt.xticks(range(24))
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


col = 'Consider count hodiday Saturday'

# Gom nhÃ³m theo sá»‘ ngÃ y Ä‘áº·c biá»‡t + label
grouped = df_full_4_6.groupby([col, 'label']).size().unstack(fill_value=0)

# Ä�áº£m báº£o Ä‘á»§ 2 nhÃ£n label
for lbl in [0, 1]:
    if lbl not in grouped.columns:
        grouped[lbl] = 0

grouped = grouped[[0, 1]]  # Ä‘Ãºng thá»© tá»±

# Láº¥y cÃ¡c giÃ¡ trá»‹ phá»• biáº¿n Ä‘áº§u tiÃªn (trÃ¡nh quÃ¡ dÃ i)
grouped = grouped.sort_index().head(10)

# Báº¯t Ä‘áº§u váº½
plt.figure(figsize=(10, 6))

x = grouped.index.astype(str)
bar1 = plt.bar(x, grouped[0], label='Label 0 (KhÃ´ng trá»…)', color='mediumseagreen')
bar2 = plt.bar(x, grouped[1], bottom=grouped[0], label='Label 1 (Bá»‹ trá»…)', color='tomato')

# Gáº¯n sá»‘ lÆ°á»£ng lÃªn Ä‘áº§u cá»™t
for i in range(len(x)):
    total = grouped[0].iloc[i] + grouped[1].iloc[i]
    plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

# ThÃªm info
plt.title(f'PhÃ¢n phá»‘i Ä‘Æ¡n hÃ ng theo Sá»‘ ngÃ y lá»… vÃ  ngÃ y nghá»‰ cuá»‘i tuáº§n tá»« ngÃ y Ä‘áº·t hÃ ng Ä‘áº¿n ngÃ y giao hÃ ng\nBiáº¿n: "{col}" (Unique: {df_full_4_6[col].nunique()})', fontsize=14)
plt.xlabel('Sá»‘ ngÃ y lá»… + Thá»© 7')
plt.ylabel('Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


def get_highly_dominant_columns(df, threshold=0.99):
    """
    Tráº£ vá»� danh sÃ¡ch cÃ¡c cá»™t mÃ  1 giÃ¡ trá»‹ chiáº¿m â‰¥ threshold (default 0.99 = 99%)
    """
    dominant_cols = []

    for col in df.columns:
        value_counts = df[col].value_counts(normalize=True, dropna=False)
        if not value_counts.empty:
            top_freq = value_counts.values[0]
            if top_freq >= threshold:
                dominant_val = value_counts.idxmax()
                dominant_cols.append((col, top_freq, dominant_val))

    return dominant_cols


dominant_columns_info = get_highly_dominant_columns(df_full_4_6, threshold=0.99)

for col, freq, val in dominant_columns_info:
    print(f"ğŸ§© {col}: '{val}' chiáº¿m {freq:.2%}")


# Gá»�i hÃ m
dominant_columns_info = get_highly_dominant_columns(df_full_4_6, threshold=0.99)

# In danh sÃ¡ch trÆ°á»›c khi xoÃ¡
print("ğŸ“Œ CÃ¡c cá»™t sáº½ bá»‹ drop (1 giÃ¡ trá»‹ chiáº¿m â‰¥99%):")
for col, freq, val in dominant_columns_info:
    print(f"ğŸ§© {col}: '{val}' chiáº¿m {freq:.2%}")

# Thá»±c hiá»‡n drop
columns_to_drop = [col for col, _, _ in dominant_columns_info]
df_full_4_6 = df_full_4_6.drop(columns=columns_to_drop)

print(f"\nâœ… Ä�Ã£ xoÃ¡ {len(columns_to_drop)} cá»™t. DataFrame giá»� cÃ²n {df_full_4_6.shape[1]} cá»™t.")


missing_info = df_full_4_6.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_4_6)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


df_full_4_6.drop(columns=['SOUF_RCV_NO'], inplace=True)


missing_info = df_full_4_6.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_4_6)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


# Chuáº©n bá»‹ dá»¯ liá»‡u biá»ƒu Ä‘á»“
cols_to_check = ['SHIP DECISION NO', 'Ship Mode']
missing_stats = []

for label_value in [0, 1]:
    subset = df_full_4_6[df_full_4_6['label'] == label_value]
    total_rows = len(subset)

    for col in cols_to_check:
        missing_count = subset[col].isnull().sum()
        missing_percent = (missing_count / total_rows) * 100
        missing_stats.append({
            'Label': f'label = {label_value}',
            'Column': col,
            'Missing %': missing_percent
        })

# Táº¡o DataFrame tá»« káº¿t quáº£
missing_df = pd.DataFrame(missing_stats)

# Váº½ biá»ƒu Ä‘á»“
plt.figure(figsize=(8, 5))
for col in cols_to_check:
    subset_df = missing_df[missing_df['Column'] == col]
    plt.bar(subset_df['Label'], subset_df['Missing %'], label=col)

plt.ylabel('Missing Value (%)')
plt.title('Tá»‰ lá»‡ Missing Value theo label')
plt.legend(title='Column')
plt.ylim(0, max(missing_df['Missing %']) + 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


df_delay_4_6['SHIP DECISION NO'].isnull().sum()


df_full_4_6['SHIP DECISION NO'].value_counts()


# Thay giÃ¡ trá»‹ missing báº±ng -1 cho 2 cá»™t Ä‘Æ°á»£c phÃ¢n tÃ­ch
df_full_4_6['SHIP DECISION NO'] = df_full_4_6['SHIP DECISION NO'].fillna(-1)
df_full_4_6['Ship Mode'] = df_full_4_6['Ship Mode'].fillna(-1)

# In thÃ´ng bÃ¡o xÃ¡c nháº­n
print("âœ… Ä�Ã£ thay tháº¿ missing values trong 'SHIP DECISION NO' vÃ  'Ship Mode' báº±ng -1.")


missing_info = df_full_4_6.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_4_6)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


# Missing quÃ¡ Ã­t nÃªn ta Ä‘Æ¡n giáº£n lÃ  drop cÃ¡c dÃ²ng bá»‹ missing.
# Drop cÃ¡c dÃ²ng bá»‹ thiáº¿u á»Ÿ SUPPLIER_DIV
df_full_4_6 = df_full_4_6.dropna(subset=['SUPPLIER_DIV'])


missing_info = df_full_4_6.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_4_6)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


print_data_overview(df_full_4_6)


# Liá»‡t kÃª táº¥t cáº£ cÃ¡c dtype cá»§a cÃ¡c cá»™t trong dataframe
print(df_full_4_6.dtypes)


for col in df_full_4_6.select_dtypes(include='object').columns:
    df_full_4_6[col] = df_full_4_6[col].astype(str)

df_full_4_6.to_parquet('/kaggle/working/Data for practice/df_full_4_6_sliver.parquet', index=False)














# Load data
base_path = r"/kaggle/input/ds-108-p-21-assigment-06"

# Ä�á»�c tá»«ng file CSV vÃ o DataFrame
df_delay_7_9 = pd.read_csv(f"{base_path}/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv", encoding='utf-8')
df_not_delay_7_9 = pd.read_csv(f"{base_path}/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv", encoding='utf-8')

print(df_delay_7_9.shape)
print(df_not_delay_7_9.shape)


# Ná»‘i delay vÃ  not delay thÃ nh data hoÃ n chá»‰nh
df_full_7_9 = pd.concat([df_delay_7_9, df_not_delay_7_9], ignore_index=True)
print(df_full_7_9.shape)


# Load codebook
codebook = pd.read_excel(
    "/kaggle/input/m-company-delay-prediction/Sample codebook of Delay Prediction task.xlsx",
    sheet_name="Columns Details",
    header=0,
    index_col=0
)


print(f"sá»‘ thuá»™c tÃ­nh Ä‘Æ°á»£c liá»‡t kÃª trong codebook: {codebook.shape[0]}")
print(f"sá»‘ thuá»™c tÃ­nh cá»§a df_full_7_9: {df_full_7_9.shape[1]}")


from datetime import datetime

# Ghi láº¡i thá»�i Ä‘iá»ƒm ingest hiá»‡n táº¡i
ingest_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("Ingest timestamp:", ingest_timestamp)

# Option: gáº¯n timestamp vÃ o toÃ n bá»™ DataFrame
codebook['ingest_timestamp'] = ingest_timestamp
df_full_7_9['ingest_timestamp'] = ingest_timestamp


# Version theo thá»�i gian ingest
data_version = "v_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
print("Data version:", data_version)

# CÃ³ thá»ƒ thÃªm vÃ o metadata hoáº·c tÃªn file lÆ°u
codebook['data_version'] = data_version
df_full_7_9['data_version'] = data_version


def check_variables_in_information(codebook, df_full_7_9):
    """
    Kiá»ƒm tra trong df_full_7_9 cÃ³ bao nhiÃªu biáº¿n Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong codebook,
    vÃ  bao nhiÃªu biáº¿n khÃ´ng Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a (unexpected), bá»� qua hai cá»™t ká»¹ thuáº­t.

    Args:
    - codebook: DataFrame chá»©a danh sÃ¡ch cÃ¡c biáº¿n há»£p lá»‡ trong cá»™t 'Attributions'
    - df_full_7_9: DataFrame chá»©a dá»¯ liá»‡u thá»±c táº¿

    Returns:
    - None (chá»‰ in ra thÃ´ng tin chi tiáº¿t)
    """
    # Táº­p biáº¿n há»£p lá»‡ Ä‘Ã£ Ä‘á»‹nh nghÄ©a
    defined_variables = set(codebook["Attributions"].str.strip())

    # CÃ¡c cá»™t cáº§n bá»� qua trong quÃ¡ trÃ¬nh kiá»ƒm tra
    ignored_columns = {"data_version", "ingest_timestamp"}

    # Táº­p biáº¿n thá»±c táº¿ trong dá»¯ liá»‡u (sau khi loáº¡i bá»� cÃ¡c cá»™t bá»‹ bá»� qua)
    info_variables = set(df_full_7_9.columns.str.strip()) - ignored_columns

    # Biáº¿n cÃ³ trong df_full_7_9 mÃ  cÅ©ng Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a
    matched_variables = info_variables & defined_variables

    # Biáº¿n cÃ³ trong df_full_7_9 nhÆ°ng KHÃ”NG Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a
    unexpected_variables = info_variables - defined_variables

    print(f"Tá»•ng sá»‘ biáº¿n trong df_full_7_9 (bá»� qua cá»™t ká»¹ thuáº­t): {len(info_variables)}")
    print(f"Sá»‘ biáº¿n Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong codebook: {len(matched_variables)}")
    print(f"Sá»‘ biáº¿n KHÃ”NG Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong codebook: {len(unexpected_variables)}")

    print("\nDanh sÃ¡ch biáº¿n Ä�Æ¯á»¢C Ä‘á»‹nh nghÄ©a:")
    print(sorted(matched_variables))

    if unexpected_variables:
        print("\nğŸ”º Danh sÃ¡ch biáº¿n KHÃ”NG Ä�Æ¯á»¢C Ä‘á»‹nh nghÄ©a:")
        print(sorted(unexpected_variables))
    else:
        print("\nğŸ�‰ KhÃ´ng cÃ³ biáº¿n nÃ o báº¥t ngá»�! Má»�i thá»© Ä‘á»�u Ä‘Ãºng chuáº©n Ä‘á»‹nh nghÄ©a.")


check_variables_in_information(codebook, df_full_7_9)


import os

output_dir = '/kaggle/working/Data for practice'
output_path = os.path.join(output_dir, 'df_full_7_9.csv')
df_full_7_9.to_csv(output_path, index=False)

print(f"ğŸ“„ File CSV Ä‘Ã£ Ä‘Æ°á»£c lÆ°u táº¡i: {output_path}")


# In ra tÃªn cá»™t vÃ  dtype tÆ°Æ¡ng á»©ng trong DataFrame df_full_7_9
codebook[['Attributions', 'Data Type']].head()


# Cáº­p nháº­t giÃ¡ trá»‹ trong cá»™t 'Data Type' náº¿u Attributions lÃ  order date hoáº·c 'VSD' 
codebook.loc[codebook['Attributions'] == 'VSD', 'Data Type'] = 'DateTime'
codebook.loc[codebook['Attributions'] == 'Order date', 'Data Type'] = 'DateTime'


def convert_column_dtypes_from_codebook(df: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyá»ƒn Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cá»§a cÃ¡c cá»™t trong DataFrame Ä‘áº§u vÃ o dá»±a trÃªn Ä‘á»‹nh nghÄ©a tá»« codebook.

    Args:
        df (pd.DataFrame): DataFrame chá»©a dá»¯ liá»‡u gá»‘c cáº§n chuyá»ƒn kiá»ƒu dá»¯ liá»‡u (vd: df_full_7_9).
        codebook (pd.DataFrame): DataFrame Ä‘á»‹nh nghÄ©a kiá»ƒu dá»¯ liá»‡u cho tá»«ng cá»™t, 
            yÃªu cáº§u cÃ³ cÃ¡c cá»™t:
                - 'Attributions': tÃªn cá»™t trong df
                - 'Data Type': kiá»ƒu dá»¯ liá»‡u Ä‘Ã­ch (int64, float, object, DateTime...)

    Returns:
        pd.DataFrame: DataFrame Ä‘áº§u vÃ o sau khi Ä‘Ã£ cá»‘ gáº¯ng chuyá»ƒn Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cÃ¡c cá»™t tÆ°Æ¡ng á»©ng.

    In ra:
        - âœ… Vá»›i má»—i cá»™t Ä‘á»•i thÃ nh cÃ´ng â†’ cÃ³ thá»ƒ in ra tÃªn cá»™t vÃ  kiá»ƒu Ä‘Ã£ chuyá»ƒn
        - â�Œ Vá»›i má»—i cá»™t lá»—i â†’ in ra tÃªn cá»™t, kiá»ƒu Ä‘á»‹nh chuyá»ƒn, vÃ  lÃ½ do lá»—i (exception message)

    Notes:
        - CÃ¡c kiá»ƒu dá»¯ liá»‡u há»— trá»£ gá»“m: int, float, object/string, DateTime
        - Vá»›i kiá»ƒu int sá»­ dá»¥ng kiá»ƒu nullable 'Int64' cá»§a pandas Ä‘á»ƒ giá»¯ láº¡i giÃ¡ trá»‹ NaN náº¿u cÃ³
        - Náº¿u kiá»ƒu dá»¯ liá»‡u khÃ´ng Ä‘Æ°á»£c nháº­n diá»‡n hoáº·c lá»—i Ã©p kiá»ƒu â†’ cá»™t sáº½ bá»‹ bá»� qua

    VÃ­ dá»¥ dÃ¹ng:
        df_full_7_9 = convert_column_dtypes_from_codebook(df_full_7_9, codebook_full)
    """
    for _, row in codebook.iterrows():
        col = row['Attributions']
        dtype = row['Data Type']

        try:
            if dtype.lower() in ['int', 'int64']:
                df[col] = pd.to_numeric(df[col], errors='raise').astype('Int64')
            elif dtype.lower() in ['float', 'float64']:
                df[col] = pd.to_numeric(df[col], errors='raise').astype('float')
            elif dtype.lower() in ['object', 'string']:
                df[col] = df[col].astype(str)
            elif dtype.lower() in ['datetime', 'datetime64', 'datetime64[ns]']:
                try:
                    # Cá»‘ gáº¯ng parse kiá»ƒu ISO8601 trÆ°á»›c
                    df[col] = pd.to_datetime(df[col], format='ISO8601', errors='raise')
                    print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ DateTime (ISO8601)")
                except Exception as e_iso:
                    try:
                        # Cá»‘ gáº¯ng vá»›i format kiá»ƒu Excel: "4/1/2022  12:00:00 AM"
                        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y %I:%M:%S %p", errors='raise')
                        print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ DateTime (Excel-style)")
                    except Exception as e_excel:
                        try:
                            # Cuá»‘i cÃ¹ng lÃ  mixed
                            df[col] = pd.to_datetime(df[col], format='mixed', errors='raise')
                            print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ DateTime (mixed format)")
                        except Exception as e_mixed:
                            print(f"Lá»—i khi Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cho cá»™t: {col} â†’ DateTime | ISO8601: {e_iso} | Excel: {e_excel} | mixed: {e_mixed}")

            else:
                print(f"KhÃ´ng rÃµ kiá»ƒu dá»¯ liá»‡u '{dtype}' cho cá»™t '{col}', bá»� qua.")
                continue

            print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ {dtype}")
        
        except Exception as e:
            print(f"Lá»—i khi Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cho cá»™t: {col} â†’ {dtype} | LÃ½ do: {e}")

    return df


df_full_7_9 = convert_column_dtypes_from_codebook(df_full_7_9, codebook)


# TÃ¡ch Order date
df_full_7_9['order_year'] = df_full_7_9['Order date'].dt.year
df_full_7_9['order_month'] = df_full_7_9['Order date'].dt.month
df_full_7_9['order_dayofweek'] = df_full_7_9['Order date'].dt.dayofweek
df_full_7_9['is_order_weekend'] = df_full_7_9['order_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# TÃ¡ch VSD
df_full_7_9['vsd_year'] = df_full_7_9['VSD'].dt.year
df_full_7_9['vsd_month'] = df_full_7_9['VSD'].dt.month
df_full_7_9['vsd_dayofweek'] = df_full_7_9['VSD'].dt.dayofweek
df_full_7_9['is_vsd_weekend'] = df_full_7_9['vsd_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# TÃ­nh khoáº£ng cÃ¡ch giá»¯a VSD vÃ  Order date (náº¿u há»£p lÃ½ vá»� logic nghiá»‡p vá»¥)
df_full_7_9['days_to_vsd'] = (df_full_7_9['VSD'] - df_full_7_9['Order date']).dt.days


df_full_7_9['vsd_year'].value_counts()


print(df_full_7_9['REASON_CD'].unique())  


# Tuy nhiÃªn vÃ¬ cá»™t nÃ y lÃ  dá»¯ liá»‡u tÆ°Æ¡ng lai, ta sáº½ loáº¡i bá»�:
df_full_7_9 = df_full_7_9.drop(columns=['REASON_CD'])  # loáº¡i bá»� hoÃ n toÃ n cá»™t REASON_CD
codebook = codebook[codebook['Attributions'] != 'REASON_CD']


cols_to_convert = ['CLASSIFY_CD', 'CUST_CD', 'SHIP DECISION NO']
df_full_7_9[cols_to_convert] = df_full_7_9[cols_to_convert].astype('object')


df_full_7_9.isnull().sum().sort_values(ascending=False)[:7]


df_full_7_9['QTUF_RCV_NO'].value_counts().sort_values(ascending=False)


df_full_7_9['SOUF_RCV_NO'].value_counts().sort_values(ascending=False)


df_full_7_9['Ship Mode'].value_counts().sort_values(ascending=False)


# Thay cÃ¡c chuá»—i 'nan' (chuá»—i chá»¯, khÃ´ng pháº£i NaN tháº­t sá»±) thÃ nh np.nan
df_full_7_9['SOUF_RCV_NO'] = df_full_7_9['SOUF_RCV_NO'].replace('nan', np.nan)
df_full_7_9['Ship Mode'] = df_full_7_9['Ship Mode'].replace('nan', np.nan)


def convert_to_time(x):
    try:
        x_str = str(int(x)).zfill(6)
        hour = int(x_str[:2])
        minute = int(x_str[2:4])
        second = int(x_str[4:6])
        return time(hour, minute, second)
    except:
        return pd.NaT

# Ã�p dá»¥ng
df_full_7_9['SO_TIME'] = df_full_7_9['SO_TIME'].apply(convert_to_time)


# Kiá»ƒm tra káº¿t quáº£
print("âœ… Ä�Ã£ chuáº©n hÃ³a cá»™t 'SO_TIME' vá»� kiá»ƒu thá»�i gian thá»±c táº¿.")
df_full_7_9['SO_TIME'].head()


# Ä�áº£m báº£o cá»™t SO_TIME Ä‘Ã£ lÃ  kiá»ƒu datetime.time
# Náº¿u chÆ°a: df_full_7_9['SO_TIME'] = df_full_7_9['SO_TIME'].apply(convert_to_time)

# Táº¡o Ä‘áº·c trÆ°ng tá»« SO_TIME
df_full_7_9['so_hour'] = df_full_7_9['SO_TIME'].apply(lambda x: x.hour if pd.notnull(x) else np.nan)
df_full_7_9['so_minute'] = df_full_7_9['SO_TIME'].apply(lambda x: x.minute if pd.notnull(x) else np.nan)
df_full_7_9['so_second'] = df_full_7_9['SO_TIME'].apply(lambda x: x.second if pd.notnull(x) else np.nan)

# Buá»•i sÃ¡ng hay chiá»�u
df_full_7_9['is_morning'] = df_full_7_9['so_hour'].apply(lambda x: 1 if x < 12 else 0 if pd.notnull(x) else np.nan)

# GÃ¡n time slot theo khung giá»� sinh hoáº¡t
def assign_time_slot(hour):
    if pd.isnull(hour): return np.nan
    elif 0 <= hour < 6: return 'midnight'
    elif 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 18: return 'afternoon'
    else: return 'evening'

df_full_7_9['time_slot'] = df_full_7_9['so_hour'].apply(assign_time_slot)

# Binned theo khung 4 tiáº¿ng
df_full_7_9['time_bin'] = pd.cut(df_full_7_9['so_hour'],
                                 bins=[-1, 3, 7, 11, 15, 19, 23],
                                 labels=['0-3h', '4-7h', '8-11h', '12-15h', '16-19h', '20-23h'])



def print_data_overview(df, columns=None):
    if columns is not None:
        df = df[columns]

    num_rows = df.shape[0]
    num_cols = df.shape[1]
    total_cells = num_rows * num_cols
    percent_missing = df.isnull().sum().sum() / total_cells * 100
    percent_duplicate = df.duplicated().sum() / num_rows * 100

    print(f"Tá»•ng quan: {num_rows} dÃ²ng, {num_cols} cá»™t")
    print(f"âš ï¸�  {percent_missing:.4f}% missing | ğŸ”� {percent_duplicate:.4f}% duplicate\n")


# Tá»•ng quan dá»¯ liá»‡u
print_data_overview(df_full_7_9)


print(df_full_7_9.select_dtypes(exclude='object').columns)


df_full_7_9.describe()


df_full_7_9.isnull().sum().sort_values(ascending=False)[:7]


df_full_7_9.duplicated().sum()


missing_info = df_full_7_9.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_7_9)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 7 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(7)


# Tá»•ng quan dá»¯ liá»‡u
print_data_overview(df_full_7_9)


# Chuyá»ƒn 'Order date' lÃ m index
df_plot = df_full_7_9.set_index('Order date')
# Resample theo ngÃ y (cÃ³ thá»ƒ thay báº±ng 'W' cho tuáº§n hoáº·c 'M' cho thÃ¡ng náº¿u muá»‘n smooth hÆ¡n)
label_1_by_day = df_plot[df_plot['label'] == 1].resample('D').size()

plt.figure(figsize=(14, 6))
plt.plot(label_1_by_day.index, label_1_by_day.values, color='red', label='Sá»‘ lÆ°á»£ng label = 1')

plt.title('Sá»‘ láº§n xuáº¥t hiá»‡n label = 1 theo thá»�i gian')
plt.xlabel('Thá»�i gian')
plt.ylabel('Táº§n suáº¥t')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Ä�áº£m báº£o cá»™t 'Order date' lÃ  datetime
df_full_7_9['Order date'] = pd.to_datetime(df_full_7_9['Order date'])

df_plot = df_full_7_9.set_index('Order date')

# Resample theo ngÃ y (cÃ³ thá»ƒ Ä‘á»•i 'D' -> 'W' hoáº·c 'M' náº¿u cáº§n mÆ°á»£t hÆ¡n)
label_1_by_day = df_plot[df_plot['label'] == 1].resample('D').size()
label_0_by_day = df_plot[df_plot['label'] == 0].resample('D').size()

plt.figure(figsize=(14, 6))
plt.plot(label_1_by_day.index, label_1_by_day.values, color='red', label='Label = 1')
plt.plot(label_0_by_day.index, label_0_by_day.values, color='blue', label='Label = 0')

plt.title('Táº§n suáº¥t label = 1 (Ä‘á»�) vÃ  label = 0 (xanh) theo thá»�i gian')
plt.xlabel('Thá»�i gian')
plt.ylabel('Sá»‘ lÆ°á»£ng')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


len(df_full_7_9['GLOBAL_NO'])


df_full_7_9['GLOBAL_NO'].value_counts()


# Láº¥y top 50 GLOBAL_NO xuáº¥t hiá»‡n nhiá»�u nháº¥t
top_50_global_no = df_full_7_9['GLOBAL_NO'].value_counts().head(50)

global_no_labels = df_full_7_9.groupby('GLOBAL_NO')['label'].min()

colors = ['red' if global_no_labels[global_no] == 1 else 'blue' for global_no in top_50_global_no.index]

plt.figure(figsize=(18, 8))
bars = plt.bar(top_50_global_no.index.astype(str), top_50_global_no.values, color=colors)

# Gáº¯n nhÃ£n sá»‘ lÆ°á»£ng lÃªn tá»«ng cá»™t
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
             str(int(height)), ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.title('Top 50 GLOBAL_NO - PhÃ¢n mÃ u theo Label (1: Ä‘á»�, 0: xanh)', fontsize=14)
plt.xlabel('GLOBAL_NO')
plt.ylabel('Sá»‘ láº§n xuáº¥t hiá»‡n')
plt.xticks(rotation=75, ha='right', fontsize=8)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


grouped = df_full_7_9.groupby(['SUPPLIER_CD', 'label']).size().unstack(fill_value=0)


top10_suppliers = grouped.sum(axis=1).sort_values(ascending=False).head(20)

top_grouped = grouped.loc[top10_suppliers.index]
plt.figure(figsize=(10, 6))

plt.bar(top_grouped.index.astype(str), top_grouped[0], label='Label 0', color='lightcoral')
plt.bar(top_grouped.index.astype(str), top_grouped[1], bottom=top_grouped[0], label='Label 1', color='cornflowerblue')

plt.xlabel('SUPPLIER_CD')
plt.ylabel('Sá»‘ Ä‘Æ¡n hÃ ng')
plt.title('Stacked Bar Chart: Top 20 Supplier theo Label')
plt.xticks(rotation=45)
plt.legend(title='Label')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# CÃ¡c cá»™t cáº§n váº½
cols = ['DIRECT SHIP FLG', 'DELI_DIV', 'Ship Mode']
label_col = 'label'

# Táº¡o Figure vá»›i 3 subplot náº±m trÃªn 1 dÃ²ng
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, col in enumerate(cols):
    ax = axes[i]
    
    # NhÃ³m vÃ  Ä‘áº¿m theo giÃ¡ trá»‹ + label
    grouped = df_full_7_9.groupby([col, label_col]).size().unstack(fill_value=0)

    # Ä�áº£m báº£o cÃ³ cáº£ label 0 vÃ  1
    if 0 not in grouped.columns:
        grouped[0] = 0
    if 1 not in grouped.columns:
        grouped[1] = 0

    grouped = grouped[[0, 1]]  # Giá»¯ thá»© tá»±

    # TÃ­nh sá»‘ lÆ°á»£ng giÃ¡ trá»‹ unique
    n_unique = df_full_7_9[col].nunique()

    # Váº½ stacked bar chart
    ax.bar(grouped.index.astype(str), grouped[0], label='Label 0', color='lightcoral')
    ax.bar(grouped.index.astype(str), grouped[1], bottom=grouped[0], label='Label 1', color='steelblue')
    
    # Gáº¯n tiÃªu Ä‘á»� cÃ³ sá»‘ lÆ°á»£ng unique
    ax.set_title(f'{col} (Unique: {n_unique})')
    ax.set_xticklabels(grouped.index.astype(str), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

# GÃ¡n nhÃ£n chung
fig.suptitle('PhÃ¢n phá»‘i Label theo DIRECT_SHIP_FLG, DELI_DIV vÃ  Ship Mode', fontsize=16)
fig.text(0.5, 0.04, 'GiÃ¡ trá»‹', ha='center')
fig.text(0.04, 0.5, 'Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng', va='center', rotation='vertical')

# ThÃªm chÃº thÃ­ch
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout(rect=[0.03, 0.05, 0.95, 0.95])
plt.show()


# CÃ¡c cá»™t flag cáº§n váº½ + 1 cá»™t rá»�i ráº¡c thÃªm
cols = ['DIRECT SHIP FLG', 'PACKING RANK', 'SPECIAL DIV', 'Stock class']
label_col = 'label'

# Táº¡o Figure vá»›i 4 subplot náº±m trÃªn 1 dÃ²ng
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

for i, col in enumerate(cols):
    ax = axes[i]
    
    # NhÃ³m vÃ  Ä‘áº¿m theo giÃ¡ trá»‹ + label
    grouped = df_full_7_9.groupby([col, label_col]).size().unstack(fill_value=0)

    # Ä�áº£m báº£o cÃ³ cáº£ label 0 vÃ  1
    for lbl in [0, 1]:
        if lbl not in grouped.columns:
            grouped[lbl] = 0

    grouped = grouped[[0, 1]]  # Ä�áº£m báº£o Ä‘Ãºng thá»© tá»±

    # TÃ­nh sá»‘ lÆ°á»£ng giÃ¡ trá»‹ unique
    n_unique = df_full_7_9[col].nunique()

    # Váº½ stacked bar chart
    x = grouped.index.astype(str)
    bar1 = ax.bar(x, grouped[0], label='Label 0', color='lightcoral')
    bar2 = ax.bar(x, grouped[1], bottom=grouped[0], label='Label 1', color='steelblue')
    
    # Gáº¯n sá»‘ lÆ°á»£ng lÃªn Ä‘áº§u má»—i cá»™t
    for j in range(len(x)):
        total = grouped[0].iloc[j] + grouped[1].iloc[j]
        ax.text(j, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Gáº¯n tiÃªu Ä‘á»� cÃ³ sá»‘ lÆ°á»£ng unique
    ax.set_title(f'{col} (Unique: {n_unique})')
    ax.set_xticklabels(x, rotation=30, ha='center')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

# GÃ¡n nhÃ£n chung
fig.suptitle('PhÃ¢n phá»‘i Label theo cÃ¡c cá»™t Flag vÃ  Stock Class', fontsize=16)
fig.text(0.5, 0.04, 'GiÃ¡ trá»‹', ha='center')
fig.text(0.04, 0.5, 'Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng', va='center', rotation='vertical')

# ThÃªm chÃº thÃ­ch
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout(rect=[0.03, 0.05, 0.95, 0.95])
plt.show()


top_weights = df_full_7_9['WEIGHT PER PIECE'].value_counts().head(30).index

# B2: Lá»�c dá»¯ liá»‡u theo top_weights
df_top = df_full_7_9[df_full_7_9['WEIGHT PER PIECE'].isin(top_weights)]

# B3: NhÃ³m theo weight vÃ  label
grouped = df_top.groupby(['WEIGHT PER PIECE', 'label']).size().unstack(fill_value=0)

# Ä�áº£m báº£o cÃ³ cáº£ label 0 vÃ  1
for lbl in [0, 1]:
    if lbl not in grouped.columns:
        grouped[lbl] = 0

grouped = grouped[[0, 1]]  # Ä�Ãºng thá»© tá»±

# B4: Váº½ stacked bar chart
plt.figure(figsize=(12, 6))
x = grouped.index.astype(str)

bar1 = plt.bar(x, grouped[0], label='Label 0 (KhÃ´ng trá»…)', color='mediumseagreen')
bar2 = plt.bar(x, grouped[1], bottom=grouped[0], label='Label 1 (Bá»‹ trá»…)', color='tomato')

# ThÃªm sá»‘ lÆ°á»£ng tá»•ng má»—i cá»™t
for i in range(len(x)):
    total = grouped[0].iloc[i] + grouped[1].iloc[i]
    plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.title('PhÃ¢n phá»‘i Label theo cÃ¡c trá»�ng lÆ°á»£ng phá»• biáº¿n nháº¥t')
plt.xlabel('WEIGHT PER PIECE')
plt.ylabel('Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


top_10_modes = df_full_7_9['Ship Mode'].value_counts().head(20).index

# Lá»�c dá»¯ liá»‡u chá»‰ giá»¯ láº¡i cÃ¡c Ship Mode náº±m trong top 10
filtered_df = df_full_7_9[df_full_7_9['Ship Mode'].isin(top_10_modes)]

# Ä�áº¿m theo Ship Mode vÃ  label
shipmode_counts = filtered_df.groupby(['Ship Mode', 'label']).size().unstack(fill_value=0)

# Ä�áº£m báº£o Ä‘Ãºng thá»© tá»±
shipmode_counts = shipmode_counts.loc[top_10_modes]  # giá»¯ Ä‘Ãºng thá»© tá»± top 10
shipmode_counts = shipmode_counts[[0, 1]]  # label 0 trÆ°á»›c, rá»“i label 1

# Váº½ stacked bar chart
plt.figure(figsize=(12, 6))
x = shipmode_counts.index.astype(str)
bar1 = plt.bar(x, shipmode_counts[0], label='Label 0', color='mediumseagreen')
bar2 = plt.bar(x, shipmode_counts[1], bottom=shipmode_counts[0], label='Label 1', color='tomato')

# Gáº¯n sá»‘ lÆ°á»£ng lÃªn má»—i cá»™t
for i in range(len(x)):
    total = shipmode_counts[0].iloc[i] + shipmode_counts[1].iloc[i]
    plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Trang trÃ­
plt.title('PhÃ¢n phá»‘i Ship Mode theo Label (Top 10 phá»• biáº¿n nháº¥t)', fontsize=14)
plt.xlabel('Ship Mode')
plt.ylabel('Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


cols = ['SO_DAY_OF_WEEK', 'SO_DAY_OF_MONTH']
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

for i, col in enumerate(cols):
    ax = axes[i]
    
    # NhÃ³m dá»¯ liá»‡u theo giÃ¡ trá»‹ + label
    grouped = df_full_7_9.groupby([col, 'label']).size().unstack(fill_value=0)

    # Ä�áº£m báº£o cáº£ label 0 vÃ  1 Ä‘á»�u cÃ³
    for lbl in [0, 1]:
        if lbl not in grouped.columns:
            grouped[lbl] = 0

    grouped = grouped[[0, 1]]  # Ä�Ãºng thá»© tá»±
    
    # TÃ­nh sá»‘ lÆ°á»£ng giÃ¡ trá»‹ unique
    n_unique = df_full_7_9[col].nunique()
    
    # Váº½ stacked bar chart
    x = grouped.index.astype(str)
    bar1 = ax.bar(x, grouped[0], label='Label 0 (KhÃ´ng trá»…)', color='mediumseagreen')
    bar2 = ax.bar(x, grouped[1], bottom=grouped[0], label='Label 1 (Bá»‹ trá»…)', color='tomato')
    
    # In sá»‘ lÆ°á»£ng trÃªn má»—i cá»™t
    for j in range(len(x)):
        total = grouped[0].iloc[j] + grouped[1].iloc[j]
        ax.text(j, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Gáº¯n tiÃªu Ä‘á»�
    ax.set_title(f'{col} (Unique: {n_unique})')
    ax.set_xlabel(col)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

fig.suptitle('PhÃ¢n phá»‘i Ä�Æ¡n hÃ ng theo NgÃ y trong tuáº§n & NgÃ y trong thÃ¡ng', fontsize=16)
fig.text(0.5, 0.04, 'GiÃ¡ trá»‹ ngÃ y', ha='center')
fig.text(0.04, 0.5, 'Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng', va='center', rotation='vertical')

# ThÃªm chÃº thÃ­ch
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout(rect=[0.03, 0.05, 0.95, 0.95])
plt.show()


# BÆ°á»›c 1: TÃ­nh SO_HOUR táº¡m thá»�i, khÃ´ng thÃªm vÃ o df_full_7_9
temp_df = df_full_7_9.copy()
temp_df['SO_HOUR'] = temp_df['SO_TIME'].apply(lambda x: x.hour if pd.notna(x) else None)

# BÆ°á»›c 2: Group theo giá»� vÃ  label
hourly_counts = temp_df.groupby(['SO_HOUR', 'label']).size().unstack(fill_value=0)

# Ä�áº£m báº£o Ä‘á»§ 24 giá»� (ká»ƒ cáº£ giá»� khÃ´ng cÃ³ Ä‘Æ¡n)
all_hours = pd.Series(range(0, 24), name='SO_HOUR')
hourly_counts = hourly_counts.reindex(all_hours, fill_value=0)

# BÆ°á»›c 3: Váº½ stacked bar chart
plt.figure(figsize=(14, 6))
bars_0 = plt.bar(hourly_counts.index, hourly_counts[0], label='KhÃ´ng trá»… (label=0)', color='mediumseagreen')
bars_1 = plt.bar(hourly_counts.index, hourly_counts[1], bottom=hourly_counts[0], label='Trá»… (label=1)', color='tomato')

# Gáº¯n sá»‘ lÆ°á»£ng tá»•ng lÃªn Ä‘á»‰nh
for i in hourly_counts.index:
    total = hourly_counts.loc[i, 0] + hourly_counts.loc[i, 1]
    if total > 0:
        plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

# CÃ i Ä‘áº·t biá»ƒu Ä‘á»“
plt.xlabel('Giá»� Ä‘áº·t hÃ ng (0â€“23h)')
plt.ylabel('Sá»‘ Ä‘Æ¡n hÃ ng')
plt.title('PhÃ¢n phá»‘i Ä‘Æ¡n hÃ ng theo giá»� Ä‘áº·t (SO_TIME) vÃ  Label (Trá»… / KhÃ´ng trá»…)')
plt.xticks(range(24))
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



col = 'Consider count hodiday Saturday'

# Gom nhÃ³m theo sá»‘ ngÃ y Ä‘áº·c biá»‡t + label
grouped = df_full_7_9.groupby([col, 'label']).size().unstack(fill_value=0)

# Ä�áº£m báº£o Ä‘á»§ 2 nhÃ£n label
for lbl in [0, 1]:
    if lbl not in grouped.columns:
        grouped[lbl] = 0

grouped = grouped[[0, 1]]  # Ä‘Ãºng thá»© tá»±

# Láº¥y cÃ¡c giÃ¡ trá»‹ phá»• biáº¿n Ä‘áº§u tiÃªn (trÃ¡nh quÃ¡ dÃ i)
grouped = grouped.sort_index().head(10)

# Báº¯t Ä‘áº§u váº½
plt.figure(figsize=(10, 6))

x = grouped.index.astype(str)
bar1 = plt.bar(x, grouped[0], label='Label 0 (KhÃ´ng trá»…)', color='mediumseagreen')
bar2 = plt.bar(x, grouped[1], bottom=grouped[0], label='Label 1 (Bá»‹ trá»…)', color='tomato')

# Gáº¯n sá»‘ lÆ°á»£ng lÃªn Ä‘áº§u cá»™t
for i in range(len(x)):
    total = grouped[0].iloc[i] + grouped[1].iloc[i]
    plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

# ThÃªm info
plt.title(f'PhÃ¢n phá»‘i Ä‘Æ¡n hÃ ng theo Sá»‘ ngÃ y lá»… vÃ  ngÃ y nghá»‰ cuá»‘i tuáº§n tá»« ngÃ y Ä‘áº·t hÃ ng Ä‘áº¿n ngÃ y giao hÃ ng\nBiáº¿n: "{col}" (Unique: {df_full_7_9[col].nunique()})', fontsize=14)
plt.xlabel('Sá»‘ ngÃ y lá»… + Thá»© 7')
plt.ylabel('Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


def get_highly_dominant_columns(df, threshold=0.99):
    """
    Tráº£ vá»� danh sÃ¡ch cÃ¡c cá»™t mÃ  1 giÃ¡ trá»‹ chiáº¿m â‰¥ threshold (default 0.99 = 99%)
    """
    dominant_cols = []

    for col in df.columns:
        value_counts = df[col].value_counts(normalize=True, dropna=False)
        if not value_counts.empty:
            top_freq = value_counts.values[0]
            if top_freq >= threshold:
                dominant_val = value_counts.idxmax()
                dominant_cols.append((col, top_freq, dominant_val))

    return dominant_cols


dominant_columns_info = get_highly_dominant_columns(df_full_7_9, threshold=0.99)

for col, freq, val in dominant_columns_info:
    print(f"ğŸ§© {col}: '{val}' chiáº¿m {freq:.2%}")


# Gá»�i hÃ m
dominant_columns_info = get_highly_dominant_columns(df_full_7_9, threshold=0.99)

# In danh sÃ¡ch trÆ°á»›c khi xoÃ¡
print("ğŸ“Œ CÃ¡c cá»™t sáº½ bá»‹ drop (1 giÃ¡ trá»‹ chiáº¿m â‰¥99%):")
for col, freq, val in dominant_columns_info:
    print(f"ğŸ§© {col}: '{val}' chiáº¿m {freq:.2%}")

# Thá»±c hiá»‡n drop
columns_to_drop = [col for col, _, _ in dominant_columns_info]
df_full_7_9 = df_full_7_9.drop(columns=columns_to_drop)

print(f"\nâœ… Ä�Ã£ xoÃ¡ {len(columns_to_drop)} cá»™t. DataFrame giá»� cÃ²n {df_full_7_9.shape[1]} cá»™t.")


missing_info = df_full_7_9.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_7_9)) * 100

missing_info.sort_values(by='missing_count', ascending=False).head(7)


df_full_7_9.drop(columns=['SOUF_RCV_NO'], inplace=True)


missing_info = df_full_7_9.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_7_9)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


# Chuáº©n bá»‹ dá»¯ liá»‡u biá»ƒu Ä‘á»“
cols_to_check = ['SHIP DECISION NO', 'Ship Mode']
missing_stats = []

for label_value in [0, 1]:
    subset = df_full_7_9[df_full_7_9['label'] == label_value]
    total_rows = len(subset)

    for col in cols_to_check:
        missing_count = subset[col].isnull().sum()
        missing_percent = (missing_count / total_rows) * 100
        missing_stats.append({
            'Label': f'label = {label_value}',
            'Column': col,
            'Missing %': missing_percent
        })

# Táº¡o DataFrame tá»« káº¿t quáº£
missing_df = pd.DataFrame(missing_stats)

# Váº½ biá»ƒu Ä‘á»“
plt.figure(figsize=(8, 5))
for col in cols_to_check:
    subset_df = missing_df[missing_df['Column'] == col]
    plt.bar(subset_df['Label'], subset_df['Missing %'], label=col)

plt.ylabel('Missing Value (%)')
plt.title('Tá»‰ lá»‡ Missing Value theo label')
plt.legend(title='Column')
plt.ylim(0, max(missing_df['Missing %']) + 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


df_delay_7_9['SHIP DECISION NO'].isnull().sum()


df_full_7_9['SHIP DECISION NO'].value_counts()


# Thay giÃ¡ trá»‹ missing báº±ng -1 cho 2 cá»™t Ä‘Æ°á»£c phÃ¢n tÃ­ch
df_full_7_9['SHIP DECISION NO'] = df_full_7_9['SHIP DECISION NO'].fillna(-1)
df_full_7_9['Ship Mode'] = df_full_7_9['Ship Mode'].fillna(-1)

# In thÃ´ng bÃ¡o xÃ¡c nháº­n
print("âœ… Ä�Ã£ thay tháº¿ missing values trong 'SHIP DECISION NO' vÃ  'Ship Mode' báº±ng -1.")


missing_info = df_full_7_9.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_7_9)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


grouped = df_full_7_9.groupby(['OTHER AREA SHIP DIV', 'label']).size().unstack(fill_value=0)

# Ä�áº£m báº£o Ä‘Ãºng thá»© tá»± label [0, 1]
for lbl in [0, 1]:
    if lbl not in grouped.columns:
        grouped[lbl] = 0
grouped = grouped[[0, 1]]

# BÆ°á»›c 2: Váº½ stacked bar chart
x = grouped.index.astype(str)
plt.figure(figsize=(14, 6))
bar0 = plt.bar(x, grouped[0], label='KhÃ´ng trá»… (label=0)', color='mediumseagreen')
bar1 = plt.bar(x, grouped[1], bottom=grouped[0], label='Trá»… (label=1)', color='tomato')

# Gáº¯n sá»‘ lÆ°á»£ng lÃªn Ä‘á»‰nh cá»™t
for i, name in enumerate(x):
    total = grouped.iloc[i, 0] + grouped.iloc[i, 1]
    plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

# CÃ i Ä‘áº·t biá»ƒu Ä‘á»“
n_unique = df_full_7_9['OTHER AREA SHIP DIV'].nunique()
plt.title(f'ğŸ“¦ PhÃ¢n phá»‘i Ä‘Æ¡n hÃ ng theo OTHER AREA SHIP DIV (Unique: {n_unique})', fontsize=14)
plt.xlabel('OTHER AREA SHIP DIV')
plt.ylabel('Sá»‘ Ä‘Æ¡n hÃ ng')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


df_full_7_9.drop(columns=['OTHER AREA SHIP DIV'], inplace=True)


# Missing quÃ¡ Ã­t nÃªn ta Ä‘Æ¡n giáº£n lÃ  drop cÃ¡c dÃ²ng bá»‹ missing.
# Drop cÃ¡c dÃ²ng bá»‹ thiáº¿u á»Ÿ SUPPLIER_DIV
df_full_7_9 = df_full_7_9.dropna(subset=['SUPPLIER_DIV'])


missing_info = df_full_7_9.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_full_7_9)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


df_full_7_9.duplicated().sum()


# Liá»‡t kÃª táº¥t cáº£ cÃ¡c dtype cá»§a cÃ¡c cá»™t trong dataframe
print(df_full_7_9.dtypes)


for col in df_full_7_9.select_dtypes(include='object').columns:
    df_full_7_9[col] = df_full_7_9[col].astype(str)
    
df_full_7_9.to_parquet('/kaggle/working/Data for practice/df_full_7_9_sliver.parquet', index=False)





# load cÃ¡c thÆ° viá»‡n
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, mode
from sklearn.preprocessing import LabelEncoder
from datetime import time


# Load data
base_path = "/kaggle/input/ds-108-p-21-assigment-06"

# Ä�á»�c tá»«ng file CSV vÃ o DataFrame
df_10 = pd.read_csv(f"{base_path}/PILOT_10.csv", encoding='utf-8')

print(df_10.shape)


# Load codebook
codebook = pd.read_excel(
    "/kaggle/input/m-company-delay-prediction/Sample codebook of Delay Prediction task.xlsx",
    sheet_name="Columns Details",
    header=0,
    index_col=0
)


print(f"sá»‘ thuá»™c tÃ­nh Ä‘Æ°á»£c liá»‡t kÃª trong codebook: {codebook.shape[0]}")
print(f"sá»‘ thuá»™c tÃ­nh cá»§a df_10: {df_10.shape[1]}")


from datetime import datetime

# Ghi láº¡i thá»�i Ä‘iá»ƒm ingest hiá»‡n táº¡i
ingest_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("Ingest timestamp:", ingest_timestamp)

# Option: gáº¯n timestamp vÃ o toÃ n bá»™ DataFrame
codebook['ingest_timestamp'] = ingest_timestamp
df_10['ingest_timestamp'] = ingest_timestamp


# Version theo thá»�i gian ingest
data_version = "v_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
print("Data version:", data_version)

# CÃ³ thá»ƒ thÃªm vÃ o metadata hoáº·c tÃªn file lÆ°u
codebook['data_version'] = data_version
df_10['data_version'] = data_version


def check_variables_in_information(codebook, df_10):
    """
    Kiá»ƒm tra trong df_10 cÃ³ bao nhiÃªu biáº¿n Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong codebook,
    vÃ  bao nhiÃªu biáº¿n khÃ´ng Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a (unexpected), bá»� qua hai cá»™t ká»¹ thuáº­t.

    Args:
    - codebook: DataFrame chá»©a danh sÃ¡ch cÃ¡c biáº¿n há»£p lá»‡ trong cá»™t 'Attributions'
    - df_10: DataFrame chá»©a dá»¯ liá»‡u thá»±c táº¿

    Returns:
    - None (chá»‰ in ra thÃ´ng tin chi tiáº¿t)
    """
    # Táº­p biáº¿n há»£p lá»‡ Ä‘Ã£ Ä‘á»‹nh nghÄ©a
    defined_variables = set(codebook["Attributions"].str.strip())

    # CÃ¡c cá»™t cáº§n bá»� qua trong quÃ¡ trÃ¬nh kiá»ƒm tra
    ignored_columns = {"data_version", "ingest_timestamp"}

    # Táº­p biáº¿n thá»±c táº¿ trong dá»¯ liá»‡u (sau khi loáº¡i bá»� cÃ¡c cá»™t bá»‹ bá»� qua)
    info_variables = set(df_10.columns.str.strip()) - ignored_columns

    # Biáº¿n cÃ³ trong df_10 mÃ  cÅ©ng Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a
    matched_variables = info_variables & defined_variables

    # Biáº¿n cÃ³ trong df_10 nhÆ°ng KHÃ”NG Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a
    unexpected_variables = info_variables - defined_variables

    print(f"Tá»•ng sá»‘ biáº¿n trong df_10 (bá»� qua cá»™t ká»¹ thuáº­t): {len(info_variables)}")
    print(f"Sá»‘ biáº¿n Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong codebook: {len(matched_variables)}")
    print(f"Sá»‘ biáº¿n KHÃ”NG Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong codebook: {len(unexpected_variables)}")

    print("\nDanh sÃ¡ch biáº¿n Ä�Æ¯á»¢C Ä‘á»‹nh nghÄ©a:")
    print(sorted(matched_variables))

    if unexpected_variables:
        print("\nDanh sÃ¡ch biáº¿n KHÃ”NG Ä�Æ¯á»¢C Ä‘á»‹nh nghÄ©a:")
        print(sorted(unexpected_variables))
    else:
        print("\nKhÃ´ng cÃ³ biáº¿n nÃ o báº¥t ngá»�! Má»�i thá»© Ä‘á»�u Ä‘Ãºng chuáº©n Ä‘á»‹nh nghÄ©a.")


check_variables_in_information(codebook, df_10)


# In ra tÃªn cá»™t vÃ  dtype tÆ°Æ¡ng á»©ng trong DataFrame df_10
codebook[['Attributions', 'Data Type']].head()


# Cáº­p nháº­t giÃ¡ trá»‹ trong cá»™t 'Data Type' náº¿u Attributions lÃ  order date hoáº·c 'VSD'
codebook.loc[codebook['Attributions'] == 'VSD', 'Data Type'] = 'DateTime'
codebook.loc[codebook['Attributions'] == 'Order date', 'Data Type'] = 'DateTime'


def convert_column_dtypes_from_codebook(df: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyá»ƒn Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cá»§a cÃ¡c cá»™t trong DataFrame Ä‘áº§u vÃ o dá»±a trÃªn Ä‘á»‹nh nghÄ©a tá»« codebook.

    Args:
        df (pd.DataFrame): DataFrame chá»©a dá»¯ liá»‡u gá»‘c cáº§n chuyá»ƒn kiá»ƒu dá»¯ liá»‡u (vd: df_10).
        codebook (pd.DataFrame): DataFrame Ä‘á»‹nh nghÄ©a kiá»ƒu dá»¯ liá»‡u cho tá»«ng cá»™t, 
            yÃªu cáº§u cÃ³ cÃ¡c cá»™t:
                - 'Attributions': tÃªn cá»™t trong df
                - 'Data Type': kiá»ƒu dá»¯ liá»‡u Ä‘Ã­ch (int64, float, object, DateTime...)

    Returns:
        pd.DataFrame: DataFrame Ä‘áº§u vÃ o sau khi Ä‘Ã£ cá»‘ gáº¯ng chuyá»ƒn Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cÃ¡c cá»™t tÆ°Æ¡ng á»©ng.

    In ra:
        - âœ… Vá»›i má»—i cá»™t Ä‘á»•i thÃ nh cÃ´ng â†’ cÃ³ thá»ƒ in ra tÃªn cá»™t vÃ  kiá»ƒu Ä‘Ã£ chuyá»ƒn
        - â�Œ Vá»›i má»—i cá»™t lá»—i â†’ in ra tÃªn cá»™t, kiá»ƒu Ä‘á»‹nh chuyá»ƒn, vÃ  lÃ½ do lá»—i (exception message)

    Notes:
        - CÃ¡c kiá»ƒu dá»¯ liá»‡u há»— trá»£ gá»“m: int, float, object/string, DateTime
        - Vá»›i kiá»ƒu int sá»­ dá»¥ng kiá»ƒu nullable 'Int64' cá»§a pandas Ä‘á»ƒ giá»¯ láº¡i giÃ¡ trá»‹ NaN náº¿u cÃ³
        - Náº¿u kiá»ƒu dá»¯ liá»‡u khÃ´ng Ä‘Æ°á»£c nháº­n diá»‡n hoáº·c lá»—i Ã©p kiá»ƒu â†’ cá»™t sáº½ bá»‹ bá»� qua

    VÃ­ dá»¥ dÃ¹ng:
        df_10 = convert_column_dtypes_from_codebook(df_10, codebook_full)
    """
    for _, row in codebook.iterrows():
        col = row['Attributions']
        dtype = row['Data Type']

        try:
            if dtype.lower() in ['int', 'int64']:
                df[col] = pd.to_numeric(df[col], errors='raise').astype('Int64')
            elif dtype.lower() in ['float', 'float64']:
                df[col] = pd.to_numeric(df[col], errors='raise').astype('float')
            elif dtype.lower() in ['object', 'string']:
                df[col] = df[col].astype(str)
            elif dtype.lower() in ['datetime', 'datetime64', 'datetime64[ns]']:
                try:
                    # Cá»‘ gáº¯ng parse kiá»ƒu ISO8601 trÆ°á»›c
                    df[col] = pd.to_datetime(df[col], format='ISO8601', errors='raise')
                    print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ DateTime (ISO8601)")
                except Exception as e_iso:
                    try:
                        # Cá»‘ gáº¯ng vá»›i format kiá»ƒu Excel: "4/1/2022  12:00:00 AM"
                        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y %I:%M:%S %p", errors='raise')
                        print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ DateTime (Excel-style)")
                    except Exception as e_excel:
                        try:
                            # Cuá»‘i cÃ¹ng lÃ  mixed
                            df[col] = pd.to_datetime(df[col], format='mixed', errors='raise')
                            print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ DateTime (mixed format)")
                        except Exception as e_mixed:
                            print(f"Lá»—i khi Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cho cá»™t: {col} â†’ DateTime | ISO8601: {e_iso} | Excel: {e_excel} | mixed: {e_mixed}")

            else:
                print(f"KhÃ´ng rÃµ kiá»ƒu dá»¯ liá»‡u '{dtype}' cho cá»™t '{col}', bá»� qua.")
                continue

            print(f"Ä�Ã£ Ä‘á»•i kiá»ƒu dá»¯ liá»‡u thÃ nh cÃ´ng cho cá»™t: {col} â†’ {dtype}")
        
        except Exception as e:
            print(f"Lá»—i khi Ä‘á»•i kiá»ƒu dá»¯ liá»‡u cho cá»™t: {col} â†’ {dtype} | LÃ½ do: {e}")

    return df


df_10 = convert_column_dtypes_from_codebook(df_10, codebook)


# TÃ¡ch Order date
df_10['order_year'] = df_10['Order date'].dt.year
df_10['order_month'] = df_10['Order date'].dt.month
df_10['order_dayofweek'] = df_10['Order date'].dt.dayofweek
df_10['is_order_weekend'] = df_10['order_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# TÃ¡ch VSD
df_10['vsd_year'] = df_10['VSD'].dt.year
df_10['vsd_month'] = df_10['VSD'].dt.month
df_10['vsd_dayofweek'] = df_10['VSD'].dt.dayofweek
df_10['is_vsd_weekend'] = df_10['vsd_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# TÃ­nh khoáº£ng cÃ¡ch giá»¯a VSD vÃ  Order date (náº¿u há»£p lÃ½ vá»� logic nghiá»‡p vá»¥)
df_10['days_to_vsd'] = (df_10['VSD'] - df_10['Order date']).dt.days


df_10['vsd_year'].value_counts()


print(df_10['REASON_CD'].unique()) 


# Tuy nhiÃªn vÃ¬ cá»™t nÃ y lÃ  dá»¯ liá»‡u tÆ°Æ¡ng lai, ta sáº½ loáº¡i bá»�:
df_10 = df_10.drop(columns=['REASON_CD'])  # loáº¡i bá»� hoÃ n toÃ n cá»™t REASON_CD
codebook = codebook[codebook['Attributions'] != 'REASON_CD']


cols_to_convert = ['CLASSIFY_CD', 'CUST_CD', 'SHIP DECISION NO']
df_10[cols_to_convert] = df_10[cols_to_convert].astype('object')


df_10.isnull().sum().sort_values(ascending=False)[:7]


df_10['QTUF_RCV_NO'].value_counts().sort_values(ascending=False)


df_10['SOUF_RCV_NO'].value_counts().sort_values(ascending=False)


df_10['Ship Mode'].value_counts().sort_values(ascending=False)


# Thay cÃ¡c chuá»—i 'nan' (chuá»—i chá»¯, khÃ´ng pháº£i NaN tháº­t sá»±) thÃ nh np.nan
df_10['SOUF_RCV_NO'] = df_10['SOUF_RCV_NO'].replace('nan', np.nan)
df_10['Ship Mode'] = df_10['Ship Mode'].replace('nan', np.nan)


def convert_to_time(x):
    try:
        x_str = str(int(x)).zfill(6)
        hour = int(x_str[:2])
        minute = int(x_str[2:4])
        second = int(x_str[4:6])
        return time(hour, minute, second)
    except:
        return pd.NaT

# Ã�p dá»¥ng
df_10['SO_TIME'] = df_10['SO_TIME'].apply(convert_to_time)


# Kiá»ƒm tra káº¿t quáº£
print("âœ… Ä�Ã£ chuáº©n hÃ³a cá»™t 'SO_TIME' vá»� kiá»ƒu thá»�i gian thá»±c táº¿.")
df_10['SO_TIME'].head()


# Ä�áº£m báº£o cá»™t SO_TIME Ä‘Ã£ lÃ  kiá»ƒu datetime.time
# Náº¿u chÆ°a: df_10['SO_TIME'] = df_10['SO_TIME'].apply(convert_to_time)

# Táº¡o Ä‘áº·c trÆ°ng tá»« SO_TIME
df_10['so_hour'] = df_10['SO_TIME'].apply(lambda x: x.hour if pd.notnull(x) else np.nan)
df_10['so_minute'] = df_10['SO_TIME'].apply(lambda x: x.minute if pd.notnull(x) else np.nan)
df_10['so_second'] = df_10['SO_TIME'].apply(lambda x: x.second if pd.notnull(x) else np.nan)

# Buá»•i sÃ¡ng hay chiá»�u
df_10['is_morning'] = df_10['so_hour'].apply(lambda x: 1 if x < 12 else 0 if pd.notnull(x) else np.nan)

# GÃ¡n time slot theo khung giá»� sinh hoáº¡t
def assign_time_slot(hour):
    if pd.isnull(hour): return np.nan
    elif 0 <= hour < 6: return 'midnight'
    elif 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 18: return 'afternoon'
    else: return 'evening'

df_10['time_slot'] = df_10['so_hour'].apply(assign_time_slot)

# Binned theo khung 4 tiáº¿ng
df_10['time_bin'] = pd.cut(df_10['so_hour'],
                                 bins=[-1, 3, 7, 11, 15, 19, 23],
                                 labels=['0-3h', '4-7h', '8-11h', '12-15h', '16-19h', '20-23h'])


def print_data_overview(df, columns=None):
    if columns is not None:
        df = df[columns]

    num_rows = df.shape[0]
    num_cols = df.shape[1]
    total_cells = num_rows * num_cols
    percent_missing = df.isnull().sum().sum() / total_cells * 100
    percent_duplicate = df.duplicated().sum() / num_rows * 100

    print(f"Tá»•ng quan: {num_rows} dÃ²ng, {num_cols} cá»™t")
    print(f"âš ï¸�  {percent_missing:.4f}% missing | ğŸ”� {percent_duplicate:.4f}% duplicate\n")


# Tá»•ng quan dá»¯ liá»‡u
print_data_overview(df_10)


print(df_10.select_dtypes(exclude='object').columns)


df_10.describe()


df_10.isnull().sum().sort_values(ascending=False)[:7]


df_10.duplicated().sum()


missing_info = df_10.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_10)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 7 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(7)


# Tá»•ng quan dá»¯ liá»‡u
print_data_overview(df_10)


def get_highly_dominant_columns(df, threshold=0.99):
    """
    Tráº£ vá»� danh sÃ¡ch cÃ¡c cá»™t mÃ  1 giÃ¡ trá»‹ chiáº¿m â‰¥ threshold (default 0.99 = 99%)
    """
    dominant_cols = []

    for col in df.columns:
        value_counts = df[col].value_counts(normalize=True, dropna=False)
        if not value_counts.empty:
            top_freq = value_counts.values[0]
            if top_freq >= threshold:
                dominant_val = value_counts.idxmax()
                dominant_cols.append((col, top_freq, dominant_val))

    return dominant_cols


dominant_columns_info = get_highly_dominant_columns(df_10, threshold=0.99)

for col, freq, val in dominant_columns_info:
    print(f"ğŸ§© {col}: '{val}' chiáº¿m {freq:.2%}")


# Gá»�i hÃ m
dominant_columns_info = get_highly_dominant_columns(df_10, threshold=0.99)

# In danh sÃ¡ch trÆ°á»›c khi xoÃ¡
print("ğŸ“Œ CÃ¡c cá»™t sáº½ bá»‹ drop (1 giÃ¡ trá»‹ chiáº¿m â‰¥99%):")
for col, freq, val in dominant_columns_info:
    print(f"ğŸ§© {col}: '{val}' chiáº¿m {freq:.2%}")

# Thá»±c hiá»‡n drop
columns_to_drop = [col for col, _, _ in dominant_columns_info]
df_10 = df_10.drop(columns=columns_to_drop)

print(f"\nâœ… Ä�Ã£ xoÃ¡ {len(columns_to_drop)} cá»™t. DataFrame giá»� cÃ²n {df_10.shape[1]} cá»™t.")


missing_info = df_10.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_10)) * 100

missing_info.sort_values(by='missing_count', ascending=False).head(7)


df_10.drop(columns=['SOUF_RCV_NO'], inplace=True)


df_10.drop(columns=['OTHER AREA SHIP DIV'], inplace=True)


missing_info = df_10.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_10)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


# Thay giÃ¡ trá»‹ missing báº±ng -1 cho 2 cá»™t Ä‘Æ°á»£c phÃ¢n tÃ­ch
df_10['SUPPLIER_DIV'] = df_10['SUPPLIER_DIV'].fillna(-1)
df_10['Ship Mode'] = df_10['Ship Mode'].fillna(-1)

# In thÃ´ng bÃ¡o xÃ¡c nháº­n
print("âœ… Ä�Ã£ thay tháº¿ missing values trong 'SUPPLIER_DIV' vÃ  'Ship Mode' báº±ng -1.")


missing_info = df_10.isnull().sum().to_frame(name='missing_count')
missing_info['missing_percent'] = (missing_info['missing_count'] / len(df_10)) * 100

# Sáº¯p xáº¿p vÃ  in ra top 5 cá»™t missing nhiá»�u nháº¥t
missing_info.sort_values(by='missing_count', ascending=False).head(5)


df_10.duplicated().sum()


# Liá»‡t kÃª táº¥t cáº£ cÃ¡c dtype cá»§a cÃ¡c cá»™t trong dataframe
print(df_10.dtypes)


for col in df_10.select_dtypes(include='object').columns:
    df_10[col] = df_10[col].astype(str)
    
df_10.to_parquet('/kaggle/working/Data for practice/df_10_sliver.parquet', index=False)





import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.utils import resample

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import f1_score

import shap
import optuna


import os
base_data_url_folder = '/kaggle/working/Data for practice'

# Ä�á»�c dá»¯ liá»‡u tá»« cÃ¡c file parquet
df_4_6 = pd.read_parquet(os.path.join(base_data_url_folder, "df_full_4_6_sliver.parquet"))
df_7_9 = pd.read_parquet(os.path.join(base_data_url_folder, "df_full_7_9_sliver.parquet"))
df_10 = pd.read_parquet(os.path.join(base_data_url_folder, "df_10_sliver.parquet"))


# Giáº£ Ä‘á»‹nh dá»¯ liá»‡u vÃ o thÃ¡ng 7 khÃ´ng cÃ²n phÃ¹ há»£p cho training vÃ  tetsing thÃ¡ng 10, vÃ¬ nÃ³ khÃ´ng cÃ²n tÃ­nh má»›i
# nÃªn ta chá»‰ dÃ¹ng thÃ¡ng 8 vÃ  thÃ¡ng 9
df_7_9 = df_7_9[df_7_9['Order date'].dt.month != 7] 


print("DataFrame 4-6:")
print(df_4_6.shape)

print("\nDataFrame 7-9:")
print(df_7_9.shape)

print("\nDataFrame 10:")
print(df_10.shape)


print(df_4_6.isnull().sum().sort_values(ascending=False).head())
print(df_7_9.isnull().sum().sort_values(ascending=False).head())
print(df_10.isnull().sum().sort_values(ascending=False).head())


def compare_delay_labels(df1, df2, df1_name='df1', df2_name='df2', label_col='label', show_table=True):
    """
    So sÃ¡nh sá»‘ lÆ°á»£ng label = 0 (khÃ´ng delay) vÃ  label = 1 (delay) giá»¯a hai DataFrame.

    Parameters:
    - df1, df2: Hai DataFrame cáº§n so sÃ¡nh
    - df1_name, df2_name: TÃªn DataFrame hiá»ƒn thá»‹ trong báº£ng
    - label_col: TÃªn cá»™t chá»©a label (máº·c Ä‘á»‹nh lÃ  'label')
    - show_table: Náº¿u True thÃ¬ in báº£ng ra console

    Returns:
    - summary_table: DataFrame dáº¡ng 2x2 chá»©a káº¿t quáº£ thá»‘ng kÃª
    """
    # Ä�áº¿m sá»‘ lÆ°á»£ng label
    count_1 = df1[label_col].value_counts().sort_index()
    count_2 = df2[label_col].value_counts().sort_index()

    # Táº¡o báº£ng thá»‘ng kÃª
    summary_table = pd.DataFrame({
        df1_name: count_1,
        df2_name: count_2
    }).fillna(0).astype(int)

    # Ä�áº·t tÃªn index dá»… Ä‘á»�c
    summary_table.index = ['KhÃ´ng delay (0)', 'Delay (1)']

    if show_table:
        print(f"Báº£ng so sÃ¡nh sá»‘ lÆ°á»£ng Ä‘Æ¡n delay vÃ  khÃ´ng delay giá»¯a {df1_name} vÃ  {df2_name}:")
        # print(summary_table)

    return summary_table


compare_delay_labels(df_4_6, df_7_9, df1_name='df_4_6', df2_name='df_7_9')


def compare_columns(df1, df2, df1_name='df1', df2_name='df2', verbose=True):
    """
    So sÃ¡nh cÃ¡c cá»™t giá»¯a hai DataFrame vÃ  in ra nhá»¯ng cá»™t chá»‰ cÃ³ á»Ÿ má»™t trong hai.

    Parameters:
    - df1, df2: Hai DataFrame cáº§n so sÃ¡nh
    - df1_name, df2_name: TÃªn hiá»ƒn thá»‹ cá»§a hai DataFrame (tÃ¹y chá»�n)
    - verbose: Náº¿u True sáº½ in káº¿t quáº£ ra mÃ n hÃ¬nh

    Returns:
    - Tuple gá»“m (only_in_df1, only_in_df2): hai táº­p há»£p tÃªn cá»™t chá»‰ cÃ³ á»Ÿ tá»«ng DataFrame
    """
    cols_df1 = set(df1.columns)
    cols_df2 = set(df2.columns)

    only_in_df1 = cols_df1 - cols_df2
    only_in_df2 = cols_df2 - cols_df1

    if verbose:
        print(f"CÃ¡c cá»™t chá»‰ cÃ³ trong {df1_name}:")
        print(only_in_df1 if only_in_df1 else "âœ… KhÃ´ng cÃ³, quÃ¡ lÃ  Ä‘á»“ng bá»™!")

        print(f"\nCÃ¡c cá»™t chá»‰ cÃ³ trong {df2_name}:")
        print(only_in_df2 if only_in_df2 else "âœ… KhÃ´ng cÃ³, siÃªu khá»›p luÃ´n!")

    return only_in_df1, only_in_df2


only_in_10, only_in_7_9 = compare_columns(df_10, df_7_9, df1_name='df_10', df2_name='df_7_9')


# Loáº¡i bá»� cÃ¡c cá»™t khÃ´ng cÃ³ trong df_7_9 khá»�i df_10, trá»« cá»™t 'ID'
cols_to_drop_in_10 = [col for col in only_in_10 if col != 'ID']
df_10 = df_10.drop(columns=cols_to_drop_in_10)
print("Ä�Ã£ loáº¡i bá»� cÃ¡c cá»™t khÃ´ng cáº§n thiáº¿t khá»�i df_10, giá»¯ láº¡i 'ID'!")

# Loáº¡i bá»� cÃ¡c cá»™t khÃ´ng cÃ³ trong df_10 khá»�i df_7_9, trá»« cá»™t 'label'
cols_to_drop_in_7_9 = [col for col in only_in_7_9 if col != 'label']
df_7_9 = df_7_9.drop(columns=cols_to_drop_in_7_9)
print("Ä�Ã£ loáº¡i bá»� cÃ¡c cá»™t khÃ´ng cáº§n thiáº¿t khá»�i df_7_9, giá»¯ láº¡i 'label'!")


def compare_column_dtypes(df1, df2, df1_name='df1', df2_name='df2'):
    """
    So sÃ¡nh dtype cá»§a cÃ¡c cá»™t giá»¯a hai DataFrame:
    - BÃ¡o cÃ¡c cá»™t chá»‰ cÃ³ á»Ÿ má»™t trong hai DataFrame.
    - BÃ¡o cÃ¡c cá»™t cÃ³ cÃ¹ng tÃªn nhÆ°ng dtype khÃ¡c nhau.
    """

    cols_df1 = set(df1.columns)
    cols_df2 = set(df2.columns)

    only_in_df1 = cols_df1 - cols_df2
    only_in_df2 = cols_df2 - cols_df1
    common_cols = cols_df1 & cols_df2

    # 1. BÃ¡o cá»™t chá»‰ cÃ³ á»Ÿ má»™t bÃªn
    if only_in_df1:
        print(f"âš ï¸� CÃ¡c cá»™t chá»‰ cÃ³ trong {df1_name} mÃ  KHÃ”NG cÃ³ trong {df2_name}: {only_in_df1}")
    if only_in_df2:
        print(f"âš ï¸� CÃ¡c cá»™t chá»‰ cÃ³ trong {df2_name} mÃ  KHÃ”NG cÃ³ trong {df1_name}: {only_in_df2}")
    if not only_in_df1 and not only_in_df2:
        print("âœ… Hai DataFrame cÃ³ cÃ¹ng danh sÃ¡ch cá»™t (khÃ´ng tÃ­nh dtype).")

    # 2. So sÃ¡nh dtype cÃ¡c cá»™t trÃ¹ng
    dtype_mismatches = []
    for col in common_cols:
        dtype1 = df1[col].dtype
        dtype2 = df2[col].dtype
        if dtype1 != dtype2:
            dtype_mismatches.append((col, dtype1, dtype2))

    if dtype_mismatches:
        print(f"\nğŸš¨ CÃ¡c cá»™t cÃ³ kiá»ƒu dá»¯ liá»‡u KHÃ”NG khá»›p giá»¯a {df1_name} vÃ  {df2_name}:")
        for col, dtype1, dtype2 in dtype_mismatches:
            print(f"ğŸ”¸ Cá»™t '{col}': {df1_name} -> {dtype1}, {df2_name} -> {dtype2}")
    else:
        print("\nğŸ�‰ Táº¥t cáº£ cÃ¡c cá»™t chung Ä‘á»�u cÃ³ dtype khá»›p nhau! Tuyá»‡t zá»�i Ã´ng máº·t giá»�i ğŸŒ�")

    return {
        "only_in_df1": only_in_df1,
        "only_in_df2": only_in_df2,
        "dtype_mismatches": dtype_mismatches
    }


compare_column_dtypes(df_7_9, df_10, df1_name='df_7_9', df2_name='df_10')


def plot_stacked_bar_top(df, target_col, label_col='label', top_n=30):
    """
    Váº½ stacked bar chart phÃ¢n phá»‘i label theo giÃ¡ trá»‹ phá»• biáº¿n nháº¥t cá»§a 1 cá»™t rá»�i ráº¡c.

    Parameters:
    - df: DataFrame Ä‘áº§u vÃ o
    - target_col: tÃªn cá»™t rá»�i ráº¡c muá»‘n váº½
    - label_col: tÃªn cá»™t nhÃ£n (default = 'label')
    - top_n: sá»‘ lÆ°á»£ng giÃ¡ trá»‹ phá»• biáº¿n nháº¥t muá»‘n hiá»ƒn thá»‹ (default = 30)
    """
    # B1: TÃ­nh sá»‘ lÆ°á»£ng giÃ¡ trá»‹ unique toÃ n cá»™t
    n_unique = df[target_col].nunique()

    # B2: Láº¥y top N giÃ¡ trá»‹ phá»• biáº¿n nháº¥t
    top_values = df[target_col].value_counts().head(top_n).index

    # B3: Lá»�c dá»¯ liá»‡u
    df_top = df[df[target_col].isin(top_values)]

    # B4: NhÃ³m vÃ  Ä‘áº¿m theo target_col + label
    grouped = df_top.groupby([target_col, label_col]).size().unstack(fill_value=0)

    # Ä�áº£m báº£o cÃ³ Ä‘á»§ label
    for lbl in [0, 1]:
        if lbl not in grouped.columns:
            grouped[lbl] = 0
    grouped = grouped[[0, 1]]

    # B5: Váº½ biá»ƒu Ä‘á»“
    plt.figure(figsize=(12, 6))
    x = grouped.index.astype(str)

    plt.bar(x, grouped[0], label='Label 0 (KhÃ´ng trá»…)', color='mediumseagreen')
    plt.bar(x, grouped[1], bottom=grouped[0], label='Label 1 (Bá»‹ trá»…)', color='tomato')

    # ThÃªm tá»•ng sá»‘ lÃªn trÃªn
    for i in range(len(x)):
        total = grouped.iloc[i].sum()
        plt.text(i, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Trang trÃ­ biá»ƒu Ä‘á»“
    plt.title(f'PhÃ¢n phá»‘i Label theo {target_col} (Top {top_n} phá»• biáº¿n)\nSá»‘ lÆ°á»£ng giÃ¡ trá»‹ unique: {n_unique}', fontsize=14)
    plt.xlabel(target_col)
    plt.ylabel('Sá»‘ lÆ°á»£ng Ä‘Æ¡n hÃ ng')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# # Lá»�c cá»™t há»£p lá»‡ 
# valid_cols = [col for col in df_7_9.columns]

# # Váº½ biá»ƒu Ä‘á»“ cho má»—i cá»™t há»£p lá»‡
# for col in valid_cols:

#     print(f'\nğŸ“¦ Ä�ang váº½ cho cá»™t: {col} - DataFrame: df_7_9')
#     plot_stacked_bar_top(df_7_9, col)


def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        # print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF


iv_full, woe_full = iv_woe(data=df_7_9, target='label', bins=10, show_woe=False)
iv_sorted = iv_full.sort_values(by='IV', ascending=False).reset_index(drop=True)

print(iv_sorted)


# NgÆ°á»¡ng IV tháº¥p
iv_threshold = 0.05

# Láº¥y danh sÃ¡ch cÃ¡c biáº¿n cÃ³ IV tháº¥p hÆ¡n ngÆ°á»¡ng
low_iv_vars = iv_sorted[iv_sorted['IV'] < iv_threshold]['Variable'].tolist()

# In ra (náº¿u muá»‘n kiá»ƒm tra)
print("CÃ¡c trÆ°á»�ng cÃ³ IV tháº¥p hÆ¡n 0.05:", low_iv_vars) 


# Danh sÃ¡ch cÃ¡c cá»™t cáº§n loáº¡i bá»�
cols_to_remove = ['GLOBAL_NO', # VÃ¬ trong df_7_9 Ä‘Ã¢y lÃ  khÃ³a chÃ­nh
                  'SHIP DECISION NO', # vÃ¬ dá»¯ liá»‡u chá»‰ táº­p trung vÃ o 1 giÃ¡ trá»‹ (-1) á»Ÿ cáº£ 2 bá»™ data vÃ  giÃ¡ trá»‹ unique cá»±c cao
                  #'PRODUCT_CD', # GiÃ¡ trá»‹ unique cá»±c kÃ¬ cao
                  #'INNER_CD', # GiÃ¡ trá»‹ unique cá»±c kÃ¬ cao
                  'SO_TIME', # kiá»ƒu time 
                  'order_dayofweek', # trÃ¹ng vá»›i SO_DAY_OF_WEEK 
                  'so_second', # Ä‘Æ¡n vá»‹ quÃ¡ nhá»� khÃ´ng cÃ³ Ã½ nghÄ©a 
                  'so_minute ', # Ä‘Æ¡n vá»‹ quÃ¡ nhá»� khÃ´ng cÃ³ Ã½ nghÄ©a 
                  'Order date', # Kiá»ƒu datetime
                  'VSD', # Kiá»ƒu datetime
                  'SUPPLIER_DIV',
                  # 'CUST_CD', # giÃ¡ trá»‹ unique cá»±c kÃ¬ cao
                  ]


# Gá»™p 2 danh sÃ¡ch vÃ  loáº¡i bá»� trÃ¹ng láº·p (Ä‘áº·c biá»‡t lÆ°u Ã½ xÃ³a khoáº£ng tráº¯ng thá»«a!)
combined_cols_to_drop = set([col.strip() for col in cols_to_remove + low_iv_vars])


def drop_useless_columns(df, columns_to_drop, inplace=False, verbose=True):
    """
    Loáº¡i bá»� cÃ¡c cá»™t khÃ´ng Ä‘Ã³ng gÃ³p giÃ¡ trá»‹ cho quÃ¡ trÃ¬nh huáº¥n luyá»‡n, vÃ­ dá»¥: ID, khÃ³a ngoáº¡i, mÃ£ code...

    Parameters:
    - df: DataFrame cáº§n xá»­ lÃ½
    - columns_to_drop: Danh sÃ¡ch tÃªn cá»™t cáº§n loáº¡i bá»� (list hoáº·c set)
    - inplace: Náº¿u True thÃ¬ thay Ä‘á»•i trá»±c tiáº¿p trÃªn df, náº¿u False thÃ¬ tráº£ vá»� báº£n sao
    - verbose: Náº¿u True thÃ¬ in ra cÃ¡c cá»™t Ä‘Ã£ bá»‹ loáº¡i bá»�

    Returns:
    - df_clean: DataFrame sau khi loáº¡i bá»� (náº¿u inplace=False)
    """
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    
    if verbose:
        if existing_cols:
            print(f"ğŸ§¹ Ä�ang loáº¡i bá»� cÃ¡c cá»™t: {existing_cols}")
        else:
            print("âœ… KhÃ´ng cÃ³ cá»™t nÃ o trong danh sÃ¡ch cáº§n loáº¡i bá»� tá»“n táº¡i trong DataFrame!")

    if inplace:
        df.drop(columns=existing_cols, inplace=True)
    else:
        return df.drop(columns=existing_cols)


# Ã�p dá»¥ng hÃ m xÃ³a
# drop_useless_columns(df_4_6, combined_cols_to_drop, inplace=True)
drop_useless_columns(df_7_9, combined_cols_to_drop, inplace=True)


drop_useless_columns(df_10, combined_cols_to_drop, inplace=True)


df_7_9.info()


df_10.info()


def split_data(df, label_col='label', test_size=0.2, val_size=0.1, random_state=42):
    """
    Chia má»™t DataFrame thÃ nh ba táº­p: train, validation vÃ  test, giá»¯ phÃ¢n phá»‘i lá»›p khÃ´ng Ä‘á»•i (stratify).

    Parameters:
        df (pd.DataFrame): Dá»¯ liá»‡u Ä‘áº§u vÃ o cÃ³ cá»™t nhÃ£n.
        label_col (str): TÃªn cá»™t nhÃ£n (default='label').
        test_size (float): Tá»· lá»‡ táº­p test trÃªn toÃ n bá»™ dá»¯ liá»‡u (default=0.2).
        val_size (float): Tá»· lá»‡ táº­p validation trÃªn toÃ n bá»™ dá»¯ liá»‡u (default=0.1).
        random_state (int): Seed Ä‘á»ƒ tÃ¡i táº¡o káº¿t quáº£ chia (default=42).

    Returns:
        df_train (pd.DataFrame): Táº­p huáº¥n luyá»‡n.
        df_val (pd.DataFrame): Táº­p validation.
        df_test (pd.DataFrame): Táº­p kiá»ƒm tra.
    """
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Chia train+val vÃ  test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      stratify=y,
                                                      random_state=random_state)
    
    # Chia train vÃ  val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, 
                                                      test_size=val_ratio, 
                                                      stratify=y_temp,
                                                      random_state=random_state)
    
    # Káº¿t há»£p láº¡i thÃ nh df
    df_train = X_train.copy()
    df_train[label_col] = y_train

    df_val = X_val.copy()
    df_val[label_col] = y_val

    df_test = X_test.copy()
    df_test[label_col] = y_test

    return df_train, df_val, df_test


def undersample_df(df, label_col='label', ratio=1/10, random_state=42):
    """
    Thá»±c hiá»‡n under sampling lá»›p Ä‘a sá»‘ Ä‘á»ƒ Ä‘áº¡t tá»· lá»‡ mong muá»‘n so vá»›i lá»›p thiá»ƒu sá»‘.

    Parameters:
        df (pd.DataFrame): Dá»¯ liá»‡u Ä‘áº§u vÃ o cÃ³ cá»™t nhÃ£n.
        label_col (str): TÃªn cá»™t nhÃ£n (default='label').
        ratio (float): Tá»· lá»‡ giá»¯a lá»›p thiá»ƒu sá»‘ / lá»›p Ä‘a sá»‘ mong muá»‘n (default=1/10).
        random_state (int): Seed Ä‘á»ƒ tÃ¡i táº¡o káº¿t quáº£ láº¥y máº«u (default=42).

    Returns:
        df_balanced (pd.DataFrame): DataFrame sau khi under sampling, Ä‘Ã£ Ä‘Æ°á»£c shuffle.
    """
    label_counts = df[label_col].value_counts()
    minority_label = label_counts.idxmin()
    majority_label = label_counts.idxmax()
    
    n_minority = label_counts.min()
    n_majority = int(n_minority / ratio)

    df_minority = df[df[label_col] == minority_label]
    df_majority = df[df[label_col] == majority_label]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=n_majority,
                                       random_state=random_state)

    df_balanced = pd.concat([df_minority, df_majority_downsampled])
    return df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


def split_and_undersample(df, label_col='label', test_size=0.2, val_size=0.1, ratios=[1/10, 1/20, 1/5]):
    """
    Chia táº­p dá»¯ liá»‡u thÃ nh train/val/test vÃ  thá»±c hiá»‡n under sampling trÃªn táº­p train vá»›i nhiá»�u tá»· lá»‡.

    Parameters:
        df (pd.DataFrame): Dá»¯ liá»‡u Ä‘áº§u vÃ o cÃ³ cá»™t nhÃ£n.
        label_col (str): TÃªn cá»™t nhÃ£n (default='label').
        test_size (float): Tá»· lá»‡ táº­p test trÃªn toÃ n bá»™ dá»¯ liá»‡u (default=0.2).
        val_size (float): Tá»· lá»‡ táº­p validation trÃªn toÃ n bá»™ dá»¯ liá»‡u (default=0.1).
        ratios (list of float): Danh sÃ¡ch cÃ¡c tá»· lá»‡ thiá»ƒu sá»‘/Ä‘a sá»‘ cáº§n under sampling (default=[1/10, 1/20, 1/5]).

    Returns:
        dict: Tá»« Ä‘iá»ƒn vá»›i cÃ¡c key:
            - 'train_under_10', 'train_under_20', ...: táº­p train sau khi under sampling.
            - 'val': táº­p validation.
            - 'test': táº­p kiá»ƒm tra.
    """
    df_train, df_val, df_test = split_data(df, label_col=label_col, test_size=test_size, val_size=val_size)

    result = {}
    for r in ratios:
        undersampled_train = undersample_df(df_train, label_col=label_col, ratio=r)
        key = f'train_under_{int(1/r)}'
        result[key] = undersampled_train

    result['val'] = df_val
    result['test'] = df_test
    return result


# Chia vÃ  under df_7_9
results_7_9 = split_and_undersample(df_7_9, ratios=[1/10, 1/20, 1/5])


# df_7_9
train_7_9_1_5  = results_7_9['train_under_5']
train_7_9_1_10 = results_7_9['train_under_10']
train_7_9_1_20 = results_7_9['train_under_20']
val_7_9 = results_7_9['val']
test_7_9 = results_7_9['test']


print(val_7_9.isnull().sum().sort_values(ascending=False).head())


# def cast_category_dtype(*dfs):
#     """
#     Tráº£ vá»� cÃ¡c DataFrame má»›i Ä‘Ã£ Ã©p cÃ¡c cá»™t object -> category.
#     """
#     converted = []
#     for df in dfs:
#         df_copy = df.copy()
#         for col in df_copy.select_dtypes(['object']).columns:
#             df_copy[col] = df_copy[col].astype('category')
#         converted.append(df_copy)
#     return converted


# hÃ m chuyá»ƒn Ä‘á»•i kiá»ƒu dá»¯ liá»‡u object thÃ nh categorical - phá»¥c vá»¥ cho má»™t sá»‘ mÃ´ hÃ¬nh cáº§n phÃ¢n biá»‡t categorical
def align_category_columns(train_df, val_df, test_df): 
    # Láº¥y cÃ¡c cá»™t object hoáº·c category tá»« train
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in cat_cols:
        # Ã‰p kiá»ƒu category cho táº¥t cáº£
        categories = train_df[col].astype('category').cat.categories
        train_df[col] = train_df[col].astype('category').cat.set_categories(categories)
        val_df[col] = val_df[col].astype('category').cat.set_categories(categories)
        test_df[col] = test_df[col].astype('category').cat.set_categories(categories)
    
    return train_df, val_df, test_df


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Huáº¥n luyá»‡n mÃ´ hÃ¬nh LGBMClassifier, tá»± Ä‘á»™ng xá»­ lÃ½ categorical features, 
    vÃ  in bÃ¡o cÃ¡o phÃ¢n loáº¡i & AUC trÃªn táº­p test.
    """

    # ğŸ”� TÃ¬m cÃ¡c cá»™t dáº¡ng object/category
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    print(f"ğŸ“Œ Categorical columns detected: {list(categorical_cols)}")

    # ğŸ”„ Chuyá»ƒn sang category dtype
    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
        X_val[col]   = X_val[col].astype('category')
        X_test[col]  = X_test[col].astype('category')

    # ğŸŒŸ Khá»Ÿi táº¡o mÃ´ hÃ¬nh
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        objective='binary',
        random_state=42
    )

    # ğŸš€ Huáº¥n luyá»‡n
    model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    categorical_feature=categorical_cols.tolist(),
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
        ]
    )


    # ğŸ”® Dá»± Ä‘oÃ¡n
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ğŸ“Š BÃ¡o cÃ¡o
    print("\nğŸ“Š Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))

    # ğŸ�¯ AUC
    auc = roc_auc_score(y_test, y_proba)
    print(f"ğŸ�¯ ROC AUC: {auc:.4f}")

    return model



# HÃ m tÃ¡ch cÃ¡c biáº¿n Ä‘á»™c láº­p vÃ  phá»¥ thuá»™c Ä‘á»ƒ cháº¡y LightGBM
def run_lightgbm_from_df(train_df, val_df, test_df, label_col='label'):
    """
    Parameters:
        train_df (pd.DataFrame): Táº­p dá»¯ liá»‡u huáº¥n luyá»‡n, bao gá»“m cá»™t nhÃ£n.
        val_df (pd.DataFrame): Táº­p dá»¯ liá»‡u validation, bao gá»“m cá»™t nhÃ£n.
        test_df (pd.DataFrame): Táº­p dá»¯ liá»‡u kiá»ƒm tra, bao gá»“m cá»™t nhÃ£n.
        label_col (str): TÃªn cá»™t chá»©a nhÃ£n phÃ¢n loáº¡i (default='label').

    Returns:
        model: MÃ´ hÃ¬nh LightGBM Ä‘Ã£ Ä‘Æ°á»£c huáº¥n luyá»‡n.
    """
    return train_lightgbm(
        train_df.drop(columns=label_col), train_df[label_col],
        val_df.drop(columns=label_col), val_df[label_col],
        test_df.drop(columns=label_col), test_df[label_col]
    )


print("\n==================== ğŸ§ª Tá»ˆ Lá»† 1:5 ====================")
train_7_9_1_5, val_7_9, test_7_9 = align_category_columns(train_7_9_1_5, val_7_9, test_7_9)
model_lightgbm_7_9_1_5 = run_lightgbm_from_df(train_7_9_1_5, val_7_9, test_7_9)

print("\n==================== ğŸ§ª Tá»ˆ Lá»† 1:10 ====================")
train_7_9_1_10, val_7_9, test_7_9 = align_category_columns(train_7_9_1_10, val_7_9, test_7_9)
model_lightgbm_7_9_1_10 = run_lightgbm_from_df(train_7_9_1_10, val_7_9, test_7_9)

print("\n==================== ğŸ§ª Tá»ˆ Lá»† 1:20 ====================")
train_7_9_1_20, val_7_9, test_7_9 = align_category_columns(train_7_9_1_20, val_7_9, test_7_9)
model_lightgbm_7_9_1_20 = run_lightgbm_from_df(train_7_9_1_20, val_7_9, test_7_9)


model_to_use = model_lightgbm_7_9_1_10
corresponding_training_set = train_7_9_1_10

# Láº¥y danh sÃ¡ch cÃ¡c cá»™t features mÃ  mÃ´ hÃ¬nh Ä‘Ã£ Ä‘Æ°á»£c huáº¥n luyá»‡n
features_columns = corresponding_training_set.drop(columns=['label']).columns

print(f"Ä�ang sá»­ dá»¥ng model Ä‘Æ°á»£c train trÃªn táº­p: train_7_9_1_10")
print(f"CÃ¡c features mÃ  mÃ´ hÃ¬nh sá»­ dá»¥ng Ä‘á»ƒ dá»± Ä‘oÃ¡n: {len(features_columns)} cá»™t")

# Chuáº©n bá»‹ dá»¯ liá»‡u df_10 cho viá»‡c dá»± Ä‘oÃ¡n
# ThÃªm .copy() Ä‘á»ƒ trÃ¡nh cÃ¡c cáº£nh bÃ¡o khi thay Ä‘á»•i dá»¯ liá»‡u á»Ÿ bÆ°á»›c sau
ids_10 = df_10['ID']
df_10_prepared = df_10[features_columns].copy()

print(f"\nÄ�Ã£ chuáº©n bá»‹ xong dá»¯ liá»‡u df_10 vá»›i {df_10_prepared.shape[1]} features.")

# Ä�á»’NG Bá»˜ KIá»‚U Dá»® LIá»†U CATEGORICAL
print("\nBáº¯t Ä‘áº§u Ä‘á»“ng bá»™ kiá»ƒu dá»¯ liá»‡u categorical...")
# Láº¥y cÃ¡c cá»™t categorical tá»« táº­p train gá»‘c Ä‘á»ƒ lÃ m tham chiáº¿u
cat_cols = corresponding_training_set.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    # Láº¥y cÃ¡c "loáº¡i" (categories) Ä‘Ã£ biáº¿t tá»« táº­p train
    known_categories = corresponding_training_set[col].astype('category').cat.categories
    
    # Ã‰p kiá»ƒu vÃ  Ã¡p dá»¥ng cÃ¡c "loáº¡i" Ä‘Ã£ biáº¿t Ä‘Ã³ vÃ o cá»™t tÆ°Æ¡ng á»©ng cá»§a df_10
    # Ä�iá»�u nÃ y Ä‘áº£m báº£o df_10 cÃ³ chÃ­nh xÃ¡c cÃ¡c category mÃ  mÃ´ hÃ¬nh Ä‘Ã£ há»�c
    df_10_prepared[col] = df_10_prepared[col].astype('category').cat.set_categories(known_categories)

print(f"Ä�á»“ng bá»™ hoÃ n táº¥t cho {len(cat_cols)} cá»™t: {list(cat_cols)}")
# Káº¾T THÃšC BÆ¯á»šC Má»šI 

# == Dá»° Ä�OÃ�N VÃ€ XUáº¤T Káº¾T QUáº¢ ===

# DÃ¹ng mÃ´ hÃ¬nh Ä‘Ã£ chá»�n Ä‘á»ƒ dá»± Ä‘oÃ¡n trÃªn dá»¯ liá»‡u Ä‘Ã£ chuáº©n bá»‹
print("\nBáº¯t Ä‘áº§u dá»± Ä‘oÃ¡n trÃªn bá»™ dá»¯ liá»‡u thÃ¡ng 10...")
predicted_labels = model_to_use.predict(df_10_prepared)
print("Dá»± Ä‘oÃ¡n hoÃ n táº¥t!")

# Táº¡o DataFrame káº¿t quáº£
output_df_10 = pd.DataFrame({
    'ID': ids_10,
    'label': predicted_labels
})

# LÆ°u DataFrame káº¿t quáº£ ra file CSV
output_filename = '23521041_LuongDacNguyen_6422.csv'
output_df_10.to_csv(output_filename, index=False)

print(f"\nÄ�Ã£ lÆ°u káº¿t quáº£ thÃ nh cÃ´ng vÃ o file: {output_filename}")
print("Xem trÆ°á»›c 5 dÃ²ng Ä‘áº§u cá»§a file káº¿t quáº£:")
print(output_df_10.head())




