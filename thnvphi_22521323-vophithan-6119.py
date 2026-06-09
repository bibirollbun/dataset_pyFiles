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


import pandas as pd


df_delay_4_6 = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv")
df_delay_7_9 = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv")
df_not_delay_4_6 = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv")
df_not_delay_7_9 = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv")


df = pd.concat([df_delay_4_6, df_delay_7_9, df_not_delay_4_6, df_not_delay_7_9], ignore_index=True)


df.shape


df.info()


df_example = pd.read_csv("/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv")


col = []
for i in df.columns:
    if i not in df_example.columns:
        print(i)
        col.append(i)


col.remove('label')


df = df.drop(labels=col, axis=1)


df.columns


def parse_date(s):
    try:
        return pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S')
    except:
        try:
            return pd.to_datetime(s, format='%Y-%m-%d')
        except:
            return pd.NaT

df['Order date'] = df['Order date'].apply(parse_date)


import datetime as dt
df['year'] = df['Order date'].dt.year
df['month'] = df['Order date'].dt.month
df['day'] = df['Order date'].dt.day
df['weekday'] = df['Order date'].dt.weekday
df['hour'] = df['Order date'].dt.hour
df['minute'] = df['Order date'].dt.minute


df['VSD'] = df['VSD'].apply(parse_date)


import datetime as dt
df['year_vsd'] = df['VSD'].dt.year
df['month_vsd'] = df['VSD'].dt.month
df['day_vsd'] = df['VSD'].dt.day
df['weekday_vsd'] = df['VSD'].dt.weekday
df['hour_vsd'] = df['VSD'].dt.hour
df['minute_vsd'] = df['VSD'].dt.minute


df = df.drop(['Order date', 'VSD'], axis=1)


df.columns


# Giáº£ sá»­ df lÃ  DataFrame cá»§a báº¡n
nan_columns = df.columns[df.isna().any()].tolist()

print("CÃ¡c cá»™t chá»©a giÃ¡ trá»‹ NaN:")
print(nan_columns)



df['QTUF_RCV_NO'] = df['QTUF_RCV_NO'].fillna(0)


df['SOUF_RCV_NO'].unique()


df['SOUF_RCV_NO'] = df['SOUF_RCV_NO'].fillna(0)
df['SOUF_RCV_NO'] = df['SOUF_RCV_NO'].apply(lambda x: pd.to_numeric(x, errors='ignore'))


df['REASON_CD'] = df['REASON_CD'].replace(['   ', np.nan], 0)
df['REASON_CD'] = df['REASON_CD'].astype(int)


df['SUPPLIER_DIV'] = df['SUPPLIER_DIV'].fillna(0)


df['SHIP DECISION NO'] = df['SHIP DECISION NO'].fillna(0)


df['Ship Mode'] = df['Ship Mode'].fillna('0')


df['Consider count hodiday Saturday'] = df['Consider count hodiday Saturday'].replace(' ', 0)
df['Consider count hodiday Saturday'] = df['Consider count hodiday Saturday'].astype(int)


df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].replace([' ', '1', np.nan], [0, 1, 0])
df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].astype(int)


df = df.drop(['year', 'hour', 'minute', 'hour_vsd', 'minute_vsd'], axis=1)


label_col = ['SUBSIDIARY_CD', 'GLOBAL_NO', 'CLASSIFY_CD', 'CUST_CD', 'BRAND_CD',
       'INNER_CD', 'SUPPLIER_CD', 'Stock class', 'OTHER AREA SHIP DIV', 'PACKING RANK', 'PRODUCT_CD','LOGICAL PLANT', 'DELI_DIV', 'Ship Mode','SHIP DECISION NO','SUPPLIER_DIV', 'REASON_CD', 'SOUF_RCV_NO','QTUF_RCV_NO']


from sklearn.preprocessing import LabelEncoder

df_v2 = df.copy()
# Giáº£ sá»­ df lÃ  DataFrame vÃ  'label' lÃ  biáº¿n má»¥c tiÃªu
le = LabelEncoder()

# Láº·p qua cÃ¡c cá»™t dáº¡ng object vÃ  label encode
for col in label_col:
    df_v2[col] = le.fit_transform(df_v2[col].astype(str))



correlations = df_v2.corr(numeric_only=True)['label'].drop('label')


import seaborn as sns
import matplotlib.pyplot as plt

# Giáº£ sá»­ df Ä‘Ã£ Label Encode, vÃ  'label' lÃ  biáº¿n má»¥c tiÃªu

# TÃ­nh há»‡ sá»‘ tÆ°Æ¡ng quan
# Sáº¯p xáº¿p theo giÃ¡ trá»‹ tuyá»‡t Ä‘á»‘i giáº£m dáº§n
corrs_sorted = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

# Váº½ biá»ƒu Ä‘á»“ cá»™t ngang
plt.figure(figsize=(10, 6))
sns.barplot(x=corrs_sorted.values, y=corrs_sorted.index, palette='viridis')

# Ghi nhÃ£n
plt.title('Ä�á»™ tÆ°Æ¡ng quan giá»¯a cÃ¡c cá»™t vÃ  biáº¿n label')
plt.xlabel('Há»‡ sá»‘ tÆ°Æ¡ng quan')
plt.ylabel('TÃªn cá»™t')
plt.tight_layout()
plt.show()



top_20_cols = correlations.abs().sort_values(ascending=False).head(20).index.tolist()

print("20 cá»™t cÃ³ tÆ°Æ¡ng quan cao nháº¥t vá»›i label:")
print(top_20_cols)


# from sklearn.preprocessing import LabelEncoder


# le = LabelEncoder()

# # Láº·p qua cÃ¡c cá»™t dáº¡ng object vÃ  label encode
# for col in label_col:
#     df[col] = le.fit_transform(df[col].astype(str))


from sklearn.preprocessing import OrdinalEncoder

# Khá»Ÿi táº¡o encoder vá»›i cáº¥u hÃ¬nh cho unseen labels
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Ã�p dá»¥ng cho cÃ¡c cá»™t cáº§n label encode
df[label_col] = encoder.fit_transform(df[label_col].astype(str))



X = df[top_20_cols]
y = df['label']


# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# # Giáº£ sá»­ báº¡n Ä‘Ã£ cÃ³ X vÃ  y (dá»¯ liá»‡u Ä‘Ã£ xá»­ lÃ½ sáºµn)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Danh sÃ¡ch cÃ¡c mÃ´ hÃ¬nh phÃ¢n loáº¡i
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Random Forest": RandomForestClassifier(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
# }

# results = []

# # Huáº¥n luyá»‡n vÃ  Ä‘Ã¡nh giÃ¡ cÃ¡c mÃ´ hÃ¬nh
# for name, model in models.items():
#     print(f"\nğŸ”� Ä�ang cháº¡y mÃ´ hÃ¬nh: {name}")
    
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     # Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"âœ… Accuracy: {accuracy:.4f}")
#     print("ğŸ“Š Classification Report:")
#     print(classification_report(y_test, y_pred))
    
#     results.append({
#         "model": name,
#         "accuracy": accuracy
#     })

# # Tá»•ng há»£p káº¿t quáº£
# print("\nğŸ“ˆ Tá»•ng há»£p káº¿t quáº£:")
# final_df = pd.DataFrame(results).sort_values(by='accuracy', ascending=False)
# print(final_df)
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# # Giáº£ sá»­ báº¡n Ä‘Ã£ cÃ³ X vÃ  y (dá»¯ liá»‡u Ä‘Ã£ xá»­ lÃ½ sáºµn)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Danh sÃ¡ch cÃ¡c mÃ´ hÃ¬nh phÃ¢n loáº¡i
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Random Forest": RandomForestClassifier(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
#     "LightGBM": LGBMClassifier(),
#     "CatBoost": CatBoostClassifier(verbose=0)
# }

# results = []

# # Huáº¥n luyá»‡n vÃ  Ä‘Ã¡nh giÃ¡ cÃ¡c mÃ´ hÃ¬nh
# for name, model in models.items():
#     print(f"\nğŸ”� Ä�ang cháº¡y mÃ´ hÃ¬nh: {name}")
    
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     # Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"âœ… Accuracy: {accuracy:.4f}")
#     print("ğŸ“Š Classification Report:")
#     print(classification_report(y_test, y_pred))
    
#     results.append({
#         "model": name,
#         "accuracy": accuracy
#     })

# # Tá»•ng há»£p káº¿t quáº£
# print("\nğŸ“ˆ Tá»•ng há»£p káº¿t quáº£:")
# final_df = pd.DataFrame(results).sort_values(by='accuracy', ascending=False)
# print(final_df)



# from sklearn.ensemble import RandomForestClassifier

# # Táº¡o mÃ´ hÃ¬nh vá»›i cÃ¡c tham sá»‘ Ä‘Ã£ chá»‰ Ä‘á»‹nh
# rf = RandomForestClassifier(
#     n_estimators=200,    # Sá»‘ lÆ°á»£ng cÃ¢y
#     max_depth=10,        # Ä�á»™ sÃ¢u tá»‘i Ä‘a cá»§a má»—i cÃ¢y
#     min_samples_split=2, # Sá»‘ máº«u tá»‘i thiá»ƒu Ä‘á»ƒ chia nÃºt
#     min_samples_leaf=1,  # Sá»‘ máº«u tá»‘i thiá»ƒu á»Ÿ lÃ¡
#     max_features='sqrt', # Sá»‘ lÆ°á»£ng tÃ­nh nÄƒng xem xÃ©t khi chia nÃºt
#     random_state=42,     # Ä�áº£m báº£o káº¿t quáº£ cÃ³ thá»ƒ tÃ¡i láº­p
#     bootstrap=True       # Sá»­ dá»¥ng bootstrap sampling
# )

# # Huáº¥n luyá»‡n mÃ´ hÃ¬nh
# rf.fit(X, y)



import xgboost as xgb
params = {
    'booster': 'gbtree',
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'lambda': 1,
    'alpha': 0.5,
    'scale_pos_weight': 1
}

# Khá»Ÿi táº¡o mÃ´ hÃ¬nh XGBoost
model_xgb = xgb.XGBClassifier(**params)
model_xgb.fit(X, y)


df_example.info()


def preprocess(df):
    df['Order date'] = df['Order date'].apply(parse_date)
    df['year'] = df['Order date'].dt.year
    df['month'] = df['Order date'].dt.month
    df['day'] = df['Order date'].dt.day
    df['weekday'] = df['Order date'].dt.weekday
    df['hour'] = df['Order date'].dt.hour
    df['minute'] = df['Order date'].dt.minute
    df['VSD'] = df['VSD'].apply(parse_date)
    df['year_vsd'] = df['VSD'].dt.year
    df['month_vsd'] = df['VSD'].dt.month
    df['day_vsd'] = df['VSD'].dt.day
    df['weekday_vsd'] = df['VSD'].dt.weekday
    df['hour_vsd'] = df['VSD'].dt.hour
    df['minute_vsd'] = df['VSD'].dt.minute
    df = df.drop(['Order date', 'VSD'], axis=1)
    df['QTUF_RCV_NO'] = df['QTUF_RCV_NO'].fillna(0)
    df['SOUF_RCV_NO'] = df['SOUF_RCV_NO'].fillna(0)
    df['SOUF_RCV_NO'] = df['SOUF_RCV_NO'].apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df['REASON_CD'] = df['REASON_CD'].replace(['   ', np.nan], 0)
    df['REASON_CD'] = df['REASON_CD'].astype(int)
    df['SUPPLIER_DIV'] = df['SUPPLIER_DIV'].fillna(0)
    df['SHIP DECISION NO'] = df['SHIP DECISION NO'].fillna(0)
    df['Ship Mode'] = df['Ship Mode'].fillna('0')
    df['Consider count hodiday Saturday'] = df['Consider count hodiday Saturday'].replace(' ', 0)
    df['Consider count hodiday Saturday'] = df['Consider count hodiday Saturday'].astype(int)
    df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].replace([' ', '1', np.nan], [0, 1, 0])
    df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].astype(int)
    df = df.drop(['year', 'hour', 'minute', 'hour_vsd', 'minute_vsd'], axis=1)
    return df


df_process = preprocess(df_example)


def label_encoder(df):
    df[label_col] = encoder.transform(df[label_col].astype(str))
    return df


df_label = label_encoder(df_process)


# def model(df):
#     X = df[top_20_cols]
#     X = X.fillna(0)
#     df['label'] = rf.predict(X)
#     return df


def model(df):
    X = df[top_20_cols]
    X = X.fillna(0)
    df['label'] = model_xgb.predict(X)
    return df


df_final = model(df_label)


df_final = df_final[['ID', 'label']]


df_final.to_csv("submission.csv", index= False)




