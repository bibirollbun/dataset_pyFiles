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


!pip install lightautoml --q


import seaborn as sns
import matplotlib.pyplot as plt


import logging
logging.basicConfig(level=logging.INFO)

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, log_loss


data = pd.read_csv('/kaggle/input/optimizingdefaultmodelbyfirstpaymentdefault/kaggle_dataset.csv')
data.head()


target = data.target
target.value_counts()


target.value_counts(normalize=True)


sns.countplot(x=target)



data.info()


num_cols = data.select_dtypes(include='number').columns
#data[num_cols]


data[num_cols].describe(percentiles=[0.01, 0.99])



corr = data[num_cols].corr()
corr


plt.figure(figsize=(12, 10))
sns.heatmap(corr, 
            #annot=True,  
            cmap='coolwarm',  
            center=0,  
            square=True,  
            fmt='.2f',  
            linewidths=0.5)
plt.title('Матрица корреляции')
plt.show()



abs_corr = corr.abs()

threshold = 0.6
high_corr = abs_corr[abs_corr > threshold]

# удалим дубликаты и единичные корреляции
high_corr_pairs = high_corr.unstack().dropna()
high_corr_pairs = high_corr_pairs[high_corr_pairs < 1] 

high_corr_pairs.sort_values(ascending=False, inplace=True)

print(high_corr_pairs)



data.isna().mean().sort_values(ascending=False)



missing_data = pd.DataFrame({
    'count': data.isna().sum(),
    'percent': (data.isna().sum() / len(data)) * 100
})

missing_data = missing_data[missing_data['count'] > 0]
missing_data = missing_data.sort_values('count', ascending=False)


fig, ax1 = plt.subplots(figsize=(12, 7))


bars = ax1.bar(missing_data.index, missing_data['count'], 
               color='skyblue', alpha=0.7, label='Количество пропусков')


ax1.set_ylabel('Количество пропусков', fontsize=12)
ax1.set_xlabel('Столбцы', fontsize=12)
ax1.tick_params(axis='x', rotation=45)


for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
              f'{missing_data["percent"].iloc[i]:.1f}%',
              ha='center', va='bottom', fontsize=9)

plt.title('Пропущенные значения по столбцам', fontsize=14, pad=20)
plt.tight_layout()
plt.show()



corr_series = data[num_cols.drop('target')].corrwith(target).sort_values()
plt.figure(figsize=(12, 8))
colors = sns.color_palette('coolwarm', len(corr_series))

bars = sns.barplot(
    y=corr_series.index,
    x=corr_series.values,
    palette=colors,
    edgecolor='black',
    linewidth=0.8
)

plt.xlabel('Коэффициент корреляции (Пирсона)', fontsize=12)
plt.ylabel('Признаки', fontsize=12)
plt.title('Корреляция признаков с целевой переменной', fontsize=14, pad=20)
plt.axvline(0, color='gray', linewidth=1, linestyle='--')  # линия нуля

for i, v in enumerate(corr_series.values):
    plt.text(v + 0.01 if v >= 0 else v - 0.05, i + 0.1, f'{v:.3f}', 
             color='black', fontsize=9, ha='center', va='center')

plt.tight_layout()
plt.show()








RANDOM_STATE = 42
TARGET_COL = 'target'

TEST_SIZE = 0.15
VAL_SIZE = 0.15

TIMEOUT_FAST = 600     # 10 минут
TIMEOUT_FULL = 1800    # 30 минут
CPU_LIMIT = 4



data = data.drop(columns=['ID'],axis=1)
df = data.copy()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

val_relative_size = VAL_SIZE / (1 - TEST_SIZE)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=val_relative_size,
    stratify=y_temp,
    random_state=RANDOM_STATE
)



train_data = X_train.copy()
train_data[TARGET_COL] = y_train

val_data = X_val.copy()
val_data[TARGET_COL] = y_val

test_data = X_test.copy()
test_data[TARGET_COL] = y_test



task = Task(
    name='binary',
    metric='auc'
)

automl_fast = TabularAutoML(
    task=task,
    timeout=TIMEOUT_FAST,
    cpu_limit=CPU_LIMIT,
    general_params={
        'use_algos': [['lgb']]
    },
    reader_params={
        'random_state': RANDOM_STATE
    }
)

logging.info("Training FAST LAMA baseline")

oof_fast = automl_fast.fit_predict(
    train_data,
    roles={'target': TARGET_COL}
)



val_pred_fast = automl_fast.predict(val_data).data[:, 0]

roc_auc_fast = roc_auc_score(y_val, val_pred_fast)
logloss_fast = log_loss(y_val, val_pred_fast)

print(f"FAST LAMA | ROC-AUC: {roc_auc_fast:.4f}, LogLoss: {logloss_fast:.4f}")



automl_full_numeric = TabularAutoML(
    task=task,
    timeout=TIMEOUT_FULL,
    cpu_limit=CPU_LIMIT,

    general_params={
 
        'use_algos': [
            ['lgb', 'linear','rf']
        ]
    },

    reader_params={
        'random_state': RANDOM_STATE
    },

    # Препроцессинг числовых признаков
    lgb_params={
        'default_params': {
            'num_leaves': 64,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_data_in_leaf': 20,
            'seed': RANDOM_STATE
        }
    }
)

oof_full_numeric = automl_full_numeric.fit_predict(
    train_data,
    roles={'target': TARGET_COL}
)


val_pred_full_numeric = automl_full_numeric.predict(val_data).data[:, 0]

roc_auc_full_numeric = roc_auc_score(y_val, val_pred_full_numeric)
logloss_full_numeric = log_loss(y_val, val_pred_full_numeric)

print(
    f"FULL numeric LAMA | "
    f"ROC-AUC: {roc_auc_full_numeric:.4f}, "
    f"LogLoss: {logloss_full_numeric:.4f}"
)






TARGET_COL = 'target'

bool_cols = X_train.select_dtypes(include='bool').columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()



missing_frac = X_train.isna().mean()
high_missing_cols = missing_frac[missing_frac > 0.45].index.tolist()

X_train_f = X_train.drop(columns=high_missing_cols)
X_val_f   = X_val.drop(columns=high_missing_cols)
X_test_f  = X_test.drop(columns=high_missing_cols)

num_cols = X_train_f.select_dtypes(include=['int64', 'float64']).columns.tolist()
bool_cols = X_train_f.select_dtypes(include='bool').columns.tolist()


bool_modes = X_train_f[bool_cols].mode().iloc[0]

X_train_f[bool_cols] = X_train_f[bool_cols].fillna(bool_modes)
X_val_f[bool_cols]   = X_val_f[bool_cols].fillna(bool_modes)
X_test_f[bool_cols]  = X_test_f[bool_cols].fillna(bool_modes)




num_medians = X_train_f[num_cols].median()

X_train_f[num_cols] = X_train_f[num_cols].fillna(num_medians)
X_val_f[num_cols]   = X_val_f[num_cols].fillna(num_medians)
X_test_f[num_cols]  = X_test_f[num_cols].fillna(num_medians)



X_train_f[bool_cols] = X_train_f[bool_cols].astype(int)
X_val_f[bool_cols]   = X_val_f[bool_cols].astype(int)
X_test_f[bool_cols]  = X_test_f[bool_cols].astype(int)



corr_matrix = X_train_f.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

corr_with_target = X_train_f.apply(
    lambda col: np.corrcoef(col, y_train)[0, 1]
).abs()
to_drop = set()

for col in upper.columns:
    for row in upper.index:
        if upper.loc[row, col] > 0.8:
            drop_col = row if corr_with_target[row] < corr_with_target[col] else col
            to_drop.add(drop_col)

X_train_f = X_train_f.drop(columns=list(to_drop))
X_val_f   = X_val_f.drop(columns=list(to_drop))
X_test_f  = X_test_f.drop(columns=list(to_drop))



model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_f, y_train)



val_pred = model.predict_proba(X_val_f)[:, 1]

roc_auc = roc_auc_score(y_val, val_pred)
logloss = log_loss(y_val, val_pred)

print(f"Custom pipeline | ROC-AUC: {roc_auc:.4f}, LogLoss: {logloss:.4f}")



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_f)
X_val_scaled = scaler.transform(X_val_f)
X_test_scaled = scaler.transform(X_test_f)

model_logreg = LogisticRegression(random_state=42, max_iter=1000)
model_logreg.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

auc = roc_auc_score(y_val, y_pred_proba)
print(f"ROC-AUC на тестовой выборке: {auc:.4f}")



TARGET_COL = 'target'

bool_cols = X_train.select_dtypes(include='bool').columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()


bool_modes = X_train[bool_cols].mode().iloc[0]
num_medians = X_train[num_cols].median()

for df_ in [X_train, X_val, X_test]:
    df_[bool_cols] = df_[bool_cols].fillna(bool_modes)
    df_[num_cols] = df_[num_cols].fillna(num_medians)
    df_[bool_cols] = df_[bool_cols].astype(int)

# добавляем missing indicators - должно дать прирост
for col in num_cols:
    X_train[f'{col}_miss'] = X_train[col].isna().astype(int)
    X_val[f'{col}_miss'] = X_val[col].isna().astype(int)
    X_test[f'{col}_miss'] = X_test[col].isna().astype(int)



lgb_model = LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.015,
    num_leaves=128,
    max_depth=-1,
    min_child_samples=10,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train)

val_pred = lgb_model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, val_pred)
logloss = log_loss(y_val, val_pred)

print(f"Custom pipeline | ROC-AUC: {roc_auc:.4f}, LogLoss: {logloss:.4f}")



data = pd.read_csv('/kaggle/input/optimizingdefaultmodelbyfirstpaymentdefault/kaggle_dataset.csv')

full_data = pd.concat([X_train_f, X_test_f,X_val_f], axis=0, ignore_index=True)

predictions = model.predict_proba(full_data)[:, 0]



# ignore_index=True — сбрасывает индексы, чтобы получился непрерывный диапазон 0, 1, 2, ...


submission = pd.DataFrame({
    'ID': data['ID'],           
    'TARGET': predictions      
})

submission.to_csv('submission.csv', index=False)




