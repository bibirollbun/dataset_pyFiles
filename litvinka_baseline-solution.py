import pandas as pd
import numpy as np


# Загрузка данных
df_train = pd.read_parquet("/kaggle/input/alpha-summer-challenge/train.pa")
df_txn   = pd.read_parquet("/kaggle/input/alpha-summer-challenge/df_transaction.pa")


# Определяем клиентов в тесте
test_clients = (
    df_txn.loc[~df_txn.client_num.isin(df_train.client_num), "client_num"]
    .unique()
)


# Вычисляем распределение классов в train
class_probs = (
    df_train["target"]
    .value_counts(normalize=True)
    .sort_index()
    .values
)


# Для каждого тестового клиента случайно выбираем класс согласно распределению классов в обучающей выборке
rng = np.random.default_rng(seed=42)

submission = pd.DataFrame({
    "client_num": test_clients,
    "target": rng.choice(np.arange(7), size=len(test_clients), p=class_probs)
})


# Сохраняем сабмит
submission.to_csv("submission.csv", index=False)

