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


df_not_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
df_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')


data_train = pd.concat([df_not_delay_7_9, df_delay_7_9], axis=0).drop_duplicates().reset_index(drop=True)


data_train.columns


data_test = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv')


data_train.head()


data_train['OTHER AREA SHIP DIV'] = data_train['OTHER AREA SHIP DIV'].replace(' ', -1)
data_test['OTHER AREA SHIP DIV'] = data_test['OTHER AREA SHIP DIV'].replace(' ', -1)


data_train['OTHER AREA SHIP DIV'] = data_train['OTHER AREA SHIP DIV'].fillna(-1)
data_train['SUPPLIER_DIV'] = data_train['SUPPLIER_DIV'].fillna(-1)
data_train['REASON_CD'] = data_train['REASON_CD'].fillna(-1)
data_train['VSD'] = data_train['VSD'].fillna(-1)
data_train['Ship Mode'] = data_train['Ship Mode'].fillna(-1)
data_train['SHIP DECISION NO'] = data_train['OTHER AREA SHIP DIV'].fillna(-1)
data_train['Order date'] = data_train['Order date'].fillna(-1)

data_test['OTHER AREA SHIP DIV'] = data_test['OTHER AREA SHIP DIV'].fillna(-1)
data_test['SUPPLIER_DIV'] = data_test['SUPPLIER_DIV'].fillna(-1)
data_test['REASON_CD'] = data_test['REASON_CD'].fillna(-1)
data_test['VSD'] = data_test['VSD'].fillna(-1)
data_test['Ship Mode'] = data_test['Ship Mode'].fillna(-1)
data_test['SHIP DECISION NO'] = data_test['OTHER AREA SHIP DIV'].fillna(-1)
data_test['Order date'] = data_test['Order date'].fillna(-1)


remove_features = ['SUBSIDIARY_CD', 'GLOBAL_NO', 'BRAND_CD', 'Sales order line number', 
                   'Stock class', 'ALLOCATION QTY', 'SPECIAL DIV', 'LOGICAL PLANT', 
                   'PURCHASE AMOUNT', 'DIRECT SHIP FLG', 'SHIP DECISION NO', 'PACK QTY', 
                   'SUPPLIER_DIV', 'SPECIAL_DIV', 'SO_DAY_OF_MONTH', 'SO_DAY_OF_WEEK', 
                    'SO_TIME', 'QTUF_RCV_NO', 'SOUF_RCV_NO', 'REASON_CD']
for col in data_train.columns:
    if col in remove_features:
        data_train.drop(col, axis=1, inplace=True)
    else: 
        continue

for col in data_test.columns:
    if col in remove_features:
        data_test.drop(col, axis=1, inplace=True)
    else: 
        continue


data_train['Consider count hodiday Saturday'] = data_train['Consider count hodiday Saturday'].replace(' ', '-1', inplace=True)
data_train['Consider count hodiday Saturday'] = data_train['Consider count hodiday Saturday'].fillna(-1)
data_train['Consider count hodiday Saturday'] = data_train['Consider count hodiday Saturday'].astype('int')

data_test['Consider count hodiday Saturday'] = data_test['Consider count hodiday Saturday'].replace(' ', '-1', inplace=True)
data_test['Consider count hodiday Saturday'] = data_test['Consider count hodiday Saturday'].fillna(-1)
data_test['Consider count hodiday Saturday'] = data_test['Consider count hodiday Saturday'].astype('int')


data_train.columns


data_train['OTHER AREA SHIP DIV'] = data_train['OTHER AREA SHIP DIV'].astype('int')

data_test['OTHER AREA SHIP DIV'] = data_test['OTHER AREA SHIP DIV'].astype('int')


data_train


!pip install unidecode


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import unidecode


categorical_cols = ['Ship Mode', 'PRODUCT ATTRIBUTION', 'PRODUCT_CD', 'PACKING RANK'
                    , 'OTHER AREA SHIP DIV', 'Consider count hodiday Saturday', 'SUPPLIER_CD',
                   'INNER_CD', 'CUST_CD', 'DELI_DIV', 'CLASSIFY_CD']

for col in categorical_cols:
    # Lowercase + deunicode
    data_train[col] = data_train[col].astype(str).str.lower().apply(unidecode.unidecode)
    data_test[col] = data_test[col].astype(str).str.lower().apply(unidecode.unidecode)

    # Count Encoding
    counts = data_train[col].value_counts().to_dict()
    data_train[col] = data_train[col].map(counts)
    data_test[col] = data_test[col].map(counts)


data_train['WEIGHT'] = data_train['SO QTY'] * data_train['WEIGHT PER PIECE']
data_test['WEIGHT'] = data_test['SO QTY'] * data_test['WEIGHT PER PIECE']

data_train.drop(columns=['SO QTY', 'WEIGHT PER PIECE'], inplace=True)
data_test.drop(columns=['SO QTY', 'WEIGHT PER PIECE'], inplace=True)


numerical_cols = ['WEIGHT', 'SUPPLIER INV AMOUNT']

# Thay thế missing bằng -1
data_train[numerical_cols] = data_train[numerical_cols].fillna(-1)
data_test[numerical_cols] = data_test[numerical_cols].fillna(-1)


def remove_outliers_iqr(df, cols):
    df_cleaned = df.copy()
    for col in cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Giữ lại các hàng không phải outlier
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

# Áp dụng loại bỏ outlier
data_train= remove_outliers_iqr(data_train, numerical_cols)



for col in numerical_cols:
    col_min = data_train[col].min()
    col_max = data_train[col].max()  

    # Min-Max normalization
    data_train[col] = (data_train[col] - col_min) / (col_max - col_min)
    data_test[col] = (data_test[col] - col_min) / (col_max - col_min)


# Chuyển đổi sang datetime
data_train['Order date'] = pd.to_datetime(data_train['Order date'], errors='coerce')
data_train['VSD'] = pd.to_datetime(data_train['VSD'], errors='coerce')

data_test['Order date'] = pd.to_datetime(data_test['Order date'], errors='coerce')
data_test['VSD'] = pd.to_datetime(data_test['VSD'], errors='coerce')

# Tạo cột isWeekday
data_train['isWeekday'] = data_train['Order date'].dt.weekday < 5
data_test['isWeekday'] = data_test['Order date'].dt.weekday < 5


# Tạo thuộc tính day_range
data_train['day_range'] = (data_train['VSD'] - data_train['Order date']).dt.days
data_test['day_range'] = (data_test['VSD'] - data_test['Order date']).dt.days

# Loại bỏ hai cột gốc
data_train.drop(columns=['Order date', 'VSD'], inplace=True)
data_test.drop(columns=['Order date', 'VSD'], inplace=True)

data_train['delivery_weekday'] = (data_train['day_range'] < 0).astype(int)
data_test['delivery_weekday'] = (data_test['day_range'] < 0).astype(int)


supplier_mean = data_train.groupby('SUPPLIER_CD')['SUPPLIER INV AMOUNT'].mean().to_dict()
data_train['SUPPLIER_AVG_AMT'] = data_train['SUPPLIER_CD'].map(supplier_mean)
data_test['SUPPLIER_AVG_AMT'] = data_test['SUPPLIER_CD'].map(supplier_mean)


from sklearn.utils import resample

# Undersampling trên tập train
df = data_train.copy()

# Tách các lớp
df_delay = df[df['label'] == 1]        # delay
df_not_delay = df[df['label'] == 0]    # not_delay

# Undersample not_delay để đạt tỉ lệ 1:20
n_delay = len(df_delay)
n_not_delay_target = n_delay * 20
df_not_delay_downsampled = resample(
    df_not_delay,
    replace=False, 
    n_samples=n_not_delay_target,
    random_state=42
)

# Kết hợp lại
df_balanced = pd.concat([df_delay, df_not_delay_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)


from sklearn.model_selection import train_test_split

X_train = df_balanced.drop('label', axis=1)
y_train = df_balanced['label']

from sklearn.model_selection import train_test_split

X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)


X_test = data_test


import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score

def objective_lgbm(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'num_leaves': trial.suggest_int('num_leaves', 15, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    }

    model = LGBMClassifier(**params)
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_valid_split, y_valid_split)],
        eval_metric='f1_macro',
        callbacks=[
            early_stopping(stopping_rounds=30),
            log_evaluation(period=0)  # tắt log trong vòng lặp
        ]
    )
    preds = model.predict(X_valid_split, num_iteration=model.best_iteration_)
    return f1_score(y_valid_split, preds, average='macro')

# Tối ưu hóa
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgbm, n_trials=50)

# Huấn luyện mô hình tốt nhất
best_lgb = LGBMClassifier(
    **study_lgb.best_params,
    n_estimators=1000,
    random_state=42,
    n_jobs=-1
)

best_lgb.fit(
    X_train, y_train,
    eval_set=[(X_valid_split, y_valid_split)],
    eval_metric='f1_macro',
    callbacks=[
        early_stopping(stopping_rounds=30),
        log_evaluation(period=50)  # in log mỗi 50 vòng lặp
    ]
)

# Dự đoán
lgb_preds = best_lgb.predict(X_test, num_iteration=best_lgb.best_iteration_)



import pandas as pd

submission = pd.DataFrame({
    'ID': range(1, len(lgb_preds) + 1),
    'label': lgb_preds
})

submission.to_csv('submission.csv', index=False)

