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


# Загрузка данных
df_train = pd.read_parquet("/kaggle/input/alpha-summer-challenge/train.pa")
df_txn   = pd.read_parquet("/kaggle/input/alpha-summer-challenge/df_transaction.pa")


def pivot_client_mcc(df_txn: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    df_pivot = df_txn[['client_num', 'mcc_code', 'amount']].groupby(
        ['client_num', 'mcc_code']).agg(sum_amount=('amount','sum')).reset_index()
    df_pivot_format = pd.pivot_table(
        df_pivot, values='sum_amount', index=['client_num'], columns=['mcc_code']).fillna(0)
    df = (df_pivot_format.T/df_pivot_format.sum(axis=1)).T.reset_index()
    return df.merge(df_train, on='client_num', how='left')


df_merged = pivot_client_mcc(df_txn, df_train)


df_merged


# Сумма вероятностей отчислений по МСС-коду по всем клиентам
df_valid = pd.DataFrame(df_merged.sum(axis=0).sort_values().iloc[:-2], columns=["total_probs"])
# Доля суммы вероятностей
df_valid["frac"] = df_valid["total_probs"]/df_valid["total_probs"].sum()*100
df_valid["frac"] = df_valid["frac"].apply(lambda x: f'{x:.5f}')
# Процент пустых MMC-кодов по всем клиентам
df_valid['Empty MCC'] = (df_merged == 0).sum(axis=0).iloc[1:-1]/df_merged.shape[0]*100



df_valid.sort_values(by='frac', ascending=False)


df_merged.shape


df_with_target = df_merged[df_merged['target'].notna()]
df_without_target = df_merged[df_merged['target'].isna()]



df_without_target


df_with_target.shape


from sklearn.model_selection import train_test_split

# Разделение признаков и целевую переменную
X = df_with_target.drop(columns=['client_num','target'])
y = df_with_target['target']

# Разделение на тестовую и валидационную выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Обучение модели
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced', 
    random_state=42
)
clf.fit(X_train, y_train)


# Предсказание
y_pred = clf.predict(X_test)

# Оценка
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred))


# Веса по классам
class_weights = {
    0: 1.00,
    1: 0.72,
    2: 0.52,
    3: 0.37,
    4: 0.27,
    5: 0.19,
    6: 0.14,
    7: 0.1
}

def wmae(y_true, y_pred, weights_map):
    weights = np.array([weights_map.get(y, 0.0) for y in y_true])
    abs_errors = np.abs(y_true - y_pred)
    return np.sum(weights * abs_errors) / np.sum(weights)

wmae_score = wmae(y_test, y_pred, class_weights)
print("WMAE:", wmae_score)


y_pred


X_test_2 = df_without_target.drop(columns=['client_num', 'target'])
y_pred_2 = clf.predict(X_test_2)


with open('/kaggle/working/pred.txt','w') as f:
    f.write(f'{y_pred_2}')


y_pred_2


forecast = df_without_target.reset_index()[['client_num']]


forecast['target'] = y_pred_2.astype(int).tolist()


forecast.to_csv('/kaggle/working/submission.csv', index=False)


forecast


df_train


df_full = pd.concat([df_train, forecast], axis=0).reset_index(drop=True)


df_full.to_csv('/kaggle/working/submission_full.csv', index=False)


df_full




