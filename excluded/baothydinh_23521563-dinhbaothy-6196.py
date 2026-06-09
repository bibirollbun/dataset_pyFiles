import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


A_delay = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', low_memory=False)
A_not_delay = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv', low_memory=False)

A_delay['label'] = 1
A_not_delay['label'] = 0

A = pd.concat([A_delay, A_not_delay], axis=0)
A.shape


B_delay = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', low_memory=False)
B_not_delay = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv', low_memory=False)

B_delay['label'] = 1
B_not_delay['label'] = 0

B = pd.concat([B_delay, B_not_delay], axis=0)
B.shape


submission = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv', low_memory=False)
submission['label'] = np.nan
submission.shape


AB_diff = set(A.columns) - set(B.columns)
print(AB_diff)


for col in AB_diff:
    A.drop(col, axis=1, inplace=True)


submission_diff = set(submission.columns) - set(B.columns)
print(submission_diff)


A['ID'] = np.nan
B['ID'] = np.nan


A['Order date'].head(), B['Order date'].head(), submission['Order date'].head()


A['Order date'] = pd.to_datetime(A['Order date'], format='%Y-%m-%d %H:%M:%S')
B['Order date'] = pd.to_datetime(B['Order date'], format='%Y-%m-%d')
submission['Order date'] = pd.to_datetime(submission['Order date'], format='%m/%d/%Y')


for col in A.columns:
    if col not in ['ID', 'label']:
        if (A[col].dtype != B[col].dtype) or (A[col].dtype != submission[col].dtype):
            print(f'{col}: {A[col].dtype} {B[col].dtype} {submission[col].dtype}')


A['GLOBAL_NO'].unique(), B['GLOBAL_NO'].unique(), submission['GLOBAL_NO'].unique()


A['GLOBAL_NO'] = A['GLOBAL_NO'].astype(str)
B['GLOBAL_NO'] = B['GLOBAL_NO'].astype(str)
submission['GLOBAL_NO'] = submission['GLOBAL_NO'].astype(str)


A['Consider count hodiday Saturday'].unique(), B['Consider count hodiday Saturday'].unique(), submission['Consider count hodiday Saturday'].unique()


A['Consider count hodiday Saturday'] = A['Consider count hodiday Saturday'].astype(str).str.strip()
A['Consider count hodiday Saturday'] = A['Consider count hodiday Saturday'].replace('', '0')
A['Consider count hodiday Saturday'] = A['Consider count hodiday Saturday'].astype(np.int64)


A['OTHER AREA SHIP DIV'].unique(), B['OTHER AREA SHIP DIV'].unique(), submission['OTHER AREA SHIP DIV'].unique()


A['OTHER AREA SHIP DIV'] = A['OTHER AREA SHIP DIV'].astype(str).str.strip()
A['OTHER AREA SHIP DIV'] = A['OTHER AREA SHIP DIV'].replace('', np.nan)
A['OTHER AREA SHIP DIV'] = A['OTHER AREA SHIP DIV'].astype(np.float64)


A['SO_DAY_OF_WEEK'].unique(), B['SO_DAY_OF_WEEK'].unique(), submission['SO_DAY_OF_WEEK'].unique()


submission['SO_DAY_OF_WEEK'] = submission['SO_DAY_OF_WEEK'].astype(str).str.strip()
submission['SO_DAY_OF_WEEK'] = submission['SO_DAY_OF_WEEK'].astype(float).astype(np.int64)


A['REASON_CD'].unique(), B['REASON_CD'].unique(), submission['REASON_CD'].unique()


A['REASON_CD'] = A['REASON_CD'].astype(str).str.strip()
A['REASON_CD'] = A['REASON_CD'].replace('', np.nan)
A['REASON_CD'] = A['REASON_CD'].astype(np.float64)


A['SO_TIME'].unique(), B['SO_TIME'].unique(), submission['SO_TIME'].unique()


submission['SO_TIME'] = submission['SO_TIME'].astype(np.int64)


A['source'] = np.nan
B['source'] = np.nan
submission['source'] = 'submission'
df = pd.concat([A, B, submission], axis=0)
df.info()


identify_cols = ['ID', 'label', 'source']


duplicate_cols = []
for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2 and col1 not in duplicate_cols and col2 not in duplicate_cols:
            if df[col1].equals(df[col2]):
                print(f'{col1} = {col2}')
                duplicate_cols.append(col2)
                break


for col in duplicate_cols:
    df.drop(col, axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)


df.drop(['REASON_CD', 'SOUF_RCV_NO', 'QTUF_RCV_NO', 'VSD'], inplace=True, axis=1)


df.shape


df.columns


id_cols = ['SUBSIDIARY_CD', 'GLOBAL_NO', 'CLASSIFY_CD', 'CUST_CD',
       'BRAND_CD', 'INNER_CD', 'SUPPLIER_CD','PRODUCT_CD', 'SHIP DECISION NO', 'Ship Mode']
for col in id_cols:
    nunique = df[col].nunique()
    percent = round(nunique * 100 / df.shape[0], 4)
    print(f'{col}: {nunique} - {percent}%')


# column with 1 unique value
df.drop(['SUBSIDIARY_CD'], inplace=True, axis=1)


#labels column with large unique values and small frequency
df.drop(['GLOBAL_NO', 'PRODUCT_CD', 'SHIP DECISION NO'], axis=1, inplace=True)


df['BRAND_CD'].value_counts()* 100 / df.shape[0]


df['BRAND_CD_MSM1'] = df['BRAND_CD'].apply(lambda x: 1 if x == 'MSM1' else 0)
df.drop(['BRAND_CD'], axis=1, inplace=True)


id_cols = list(set(id_cols) - {'BRAND_CD', 'GLOBAL_NO', 'PRODUCT_CD', 'SHIP DECISION NO', 'SUBSIDIARY_CD'})
id_cols


#group rare classes into 'other'
for col in id_cols:
    df[col] = df[col].fillna('other')
    df[col] = df[col].astype('object')
    vc = df[col].value_counts()

    # Tìm các nhãn xuất hiện ít hơn 1%
    rare = set(vc[vc * 100 / len(df) <= 1].index)
    # Gán các nhãn hiếm thành 'other'
    mask = df[col].isin(rare)
    df.loc[mask, col] = 'other'
    # Chuyển đổi lại về category
    df[col] = df[col].astype('category')
    print(f'{col} --> {df[col].unique()}')


df['Order date'].dtypes, df['SO_DAY_OF_MONTH'].dtypes, df['SO_DAY_OF_WEEK'].dtypes


df['SO_DAY_OF_MONTH'] = df['SO_DAY_OF_MONTH'].astype('category')
df['SO_DAY_OF_WEEK'] = df['SO_DAY_OF_WEEK'].astype('category')


df['Order_month'] = df['Order date'].dt.month.astype('category')
df['Order_week'] = pd.cut(
    df['SO_DAY_OF_MONTH'],
    bins=[0, 7, 14, 21, 31],
    labels=['1-7', '8-14', '15-21', '22-31'],
    include_lowest=True
)


df['SO_TIME'] = df['SO_TIME'].astype(str).str.zfill(6)
df['SO_TIME'] = pd.to_datetime(df['SO_TIME'], format='%H%M%S').dt.time
df['SO_TIME'].head(), df['SO_TIME'].dtype


df['Order_hour'] = df['SO_TIME'].apply(lambda x: int(str(x).split(':')[0]))
df['Order_hour'] = df['Order_hour'].astype('category')
df['Order_hour'] = df['SO_TIME'].apply(lambda x: int(str(x).split(':')[0]))
df['Order_time_slot'] = pd.cut(
    df['Order_hour'],
    bins=[0, 6, 12, 18, 24],
    labels=['0-6', '6-12', '12-18', '18-24'],
    include_lowest=True
)


df[['Order date', 'Order_month', 'SO_DAY_OF_MONTH', 'Order_week', 'SO_DAY_OF_WEEK', 'SO_TIME', 'Order_hour', 'Order_time_slot']].head()


df.drop(['Order date', 'Order_hour', 'SO_TIME', 'SO_DAY_OF_MONTH', 'Order_hour'], axis=1, inplace=True)
print(df[['Order_month', 'Order_week', 'SO_DAY_OF_WEEK', 'Order_time_slot']].dtypes)


na_percent = df.drop(columns=identify_cols).isna().mean()
print(na_percent[na_percent > 0].sort_values(ascending=False))


df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].fillna(0)
df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].replace('1.0', 1)
df['OTHER AREA SHIP DIV'] = df['OTHER AREA SHIP DIV'].astype(np.int64)
df['OTHER AREA SHIP DIV'].isna().sum(), df['OTHER AREA SHIP DIV'].value_counts()


df['SUPPLIER_DIV'].isna().sum(), df['SUPPLIER_DIV'].value_counts()


df['SUPPLIER_DIV'] = df['SUPPLIER_DIV'].fillna(0)
df['SUPPLIER_DIV'] = df['SUPPLIER_DIV'].replace('1.0', 1)
df['SUPPLIER_DIV'] = df['SUPPLIER_DIV'].replace('2.0', 2)
df['SUPPLIER_DIV'] = df['SUPPLIER_DIV'].replace('3.0', 3)
df['SUPPLIER_DIV'] = df['SUPPLIER_DIV'].replace('4.0', 4)
df['SUPPLIER_DIV'] = df['SUPPLIER_DIV'].astype(int)
df['SUPPLIER_DIV'].value_counts()


bin_cols = []
for col in df.drop(columns=identify_cols).columns:
    if df[col].nunique() == 2:
        print(f'{col} - {df[col].dtypes}')
        bin_cols.append(col)


cat_cols = []
for col in df.columns:
    if df[col].nunique() > 2 and df[col].nunique() <= 20:
        print(col, '-', df[col].nunique(), '-', df[col].dtypes)
        cat_cols.append(col)


cat_cols = list(set(cat_cols) - {'Consider count hodiday Saturday'})
df[cat_cols] = df[cat_cols].astype('category')


continous_cols = []
for col in set(df.columns) - set(cat_cols) - set(bin_cols) - set(id_cols) - set(identify_cols):
    if df[col].dtype != 'object':
        print(col, '-', df[col].dtype, '(', df[col].nunique(), ')')
        continous_cols.append(col)


countinous_cols = set(continous_cols) - {'Consider count hodiday Saturday'}
pd.set_option('display.float_format', '{:.2f}'.format)
df[list(countinous_cols)].describe()


print('% PACKQTY == 0:', (df['PACK QTY'] == 0).sum()*100 / df.shape[0])


# Change PACK QTY to binary
df['PACK_QTY_0'] = df['PACK QTY'].apply(lambda x: 1 if x == 0 else 0)
df['PACK_QTY_0'].dtype
bin_cols.append('PACK_QTY_0')
df.drop('PACK QTY', axis=1, inplace=True)


cols_to_normalize = list(set(countinous_cols) - {'PACK QTY'})

plt.figure(figsize=(15, 2.5))
for i, col in enumerate(cols_to_normalize):
    plt.subplot(1, 5, i + 1)
    df[col].hist(bins=50)
    plt.title(col)
plt.show()


df.info()


# one-hot encoding for Random Forest
# df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


submission_df = df[df['source'] == 'submission'].drop(columns={'source', 'label'})
submission_df.shape


df = df[df['source'] != 'submission']
df.drop(columns={'source', 'ID'}, inplace=True)
df.info()


import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


X = df.drop(columns='label')
y = df['label']


submission_ids = submission_df['ID'].copy()
X_test = submission_df.drop(columns=['ID'])
X_test[cols_to_normalize] = np.log1p(X_test[cols_to_normalize])


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
threshold = 0.2


def evaluate_model(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"        Pred 0   Pred 1")
    print(f"True 0{cm[0,0]:8} {cm[0,1]:8}")
    print(f"True 1{cm[1,0]:8} {cm[1,1]:8}")


print("===== K-Fold Validation =====")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold} training...")

    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

    X_train[cols_to_normalize] = np.log1p(X_train[cols_to_normalize])
    X_val[cols_to_normalize] = np.log1p(X_val[cols_to_normalize])

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'verbosity': -1,
        'num_threads': 4,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50)],
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test, num_iteration=model.best_iteration)
    evaluate_model(y_val, (val_preds > threshold).astype(int))

# Evaluate OOF
print("\n==> Final OOF Evaluation:")
evaluate_model(y, (oof_preds > threshold).astype(int))


submission = pd.DataFrame({
    'ID': submission_ids,
    'label': test_preds
})
submission['label'] = (submission['label'] > threshold).astype(int)


submission.to_csv('submission_LGBM.csv', index=False)


print("===== K-Fold Random Forest Validation =====")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold} training...")

    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

    # Normalize selected columns
    X_train[cols_to_normalize] = np.log1p(X_train[cols_to_normalize])
    X_val[cols_to_normalize] = np.log1p(X_val[cols_to_normalize])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    evaluate_model(y_val, (val_pred > threshold).astype(int))

    oof_preds[val_idx] = val_pred
    test_preds += test_pred

# Average test predictions
test_preds /= skf.get_n_splits()

# Final OOF evaluation
print("\n==> Final OOF Evaluation:")
evaluate_model(y, (oof_preds > threshold).astype(int))


submission = pd.DataFrame({
    'ID': submission_ids,
    'label': test_preds
})
submission['label'] = (submission['label'] > threshold).astype(int)
print(submission['label'].value_counts())


submission.to_csv('submission_RF.csv', index=False)

