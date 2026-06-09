%%capture
requirements = """
fireducks>=1.3.1
"""

!echo "{requirements}" > requirements.txt
%pip install -r requirements.txt


from typing import Optional
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

import numpy as np
# fast data processing with Pandas API support
import fireducks.pandas as pd
import matplotlib.pyplot as plt

# models and training
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna


# Загрузка данных

df_train = pd.read_parquet("/kaggle/input/alpha-summer-challenge/train.pa")
df_txn   = pd.read_parquet("/kaggle/input/alpha-summer-challenge/df_transaction.pa")


# Определяем клиентов в тесте

test_clients = (
    df_txn.loc[~df_txn.client_num.isin(df_train.client_num), "client_num"]
    .unique()
)


# Посмотрим на объемы обучающей и тестовой выборки
# Выведим количество клиентов и количество транзакций

def show_info(name: str, cl_num: int, txn_num: int):
    print(f"{name}:\t{cl_num:,} clients\t{txn_num:,} txn")


train_txn_num = df_txn.client_num.isin(df_train.client_num).sum()

show_info("Train", len(df_train), train_txn_num)
show_info("Test", len(test_clients), train_txn_num)
show_info("Total", len(df_train) + len(test_clients), len(df_txn))


# Смотрим распределение таргетов
(
    df_train["target"]
    .value_counts(normalize=True)
    .sort_index()
    .reset_index()
    .plot.bar(x="target", y="proportion", rot=0)
)
plt.show()


# Посмотрим временной период
max_date_time, min_date_time = df_txn['date_time'].max(), df_txn['date_time'].min()

print("Максимальная дата:", max_date_time)
print("Минимальная дата:", min_date_time)
print("Длина периода:", max_date_time - min_date_time)


max_date_time_txn = df_txn.loc[df_txn['date_time'] == max_date_time, "date_time"]
print(
    "Количество транзакций в максимальный момент времени:",
    len(max_date_time_txn)
)


# Для дальнейшего удобства вычтем у них 1 секунду, 
# чтобы масимальная дата (2024-10-01) была не включительна
df_txn.loc[max_date_time_txn.index, "date_time"] -= pd.Timedelta(seconds=1)


def get_window_features(
    wind_df: pd.DataFrame,
    wind_postfix: Optional[str] = None,
    with_mcc_codes: bool = True
) -> pd.DataFrame:
    """
        Функция для получения фичей (см. описание фичей выше) за определённое окно

        Params:
            wind_df: pd.DataFrame
                Исходные данные за определённое окно
            wind_postfix: str or None, deafult None
                Постфикс, который будет добавлен названиям всех фичей
            with_mcc_codes: bool, deafult True
                Необходимо ли рассчитывать группу фичей в разрезе всех MCC-кодов

        Returns:
            DataFrame с фичами за окно
    """
    start_period_dt = wind_df["date_time"].min()
    num_days = (wind_df["date_time"].max() - start_period_dt).days + 1
    print(f"Calc features for window {num_days} days - START")
    
    # General features (cnt, sum, min, max, avg)
    general_df = wind_df.groupby("client_num").agg(
        txn_cnt=("amount", "count"),
        txn_sum=("amount", "sum"),
        txn_avg=("amount", "mean"),
        txn_median=('amount', 'median'),
        txn_min_sum=("amount", "min"),
        txn_max_sum=("amount", "max")
    )
    general_df["wind_txn_cnt_avg"] = general_df["txn_cnt"] / num_days
    general_df["wind_txn_sum_avg"] = general_df["txn_sum"] / num_days

    # Features by dates
    days_df = wind_df.groupby(["client_num", "txn_date"]).agg(
        txn_cnt=("amount", 'count'),
        txn_sum=("amount", 'sum')
    ).reset_index()

    days_df["prev_txn_date"] = (
        days_df
        .sort_values(by=['txn_date'], ascending=True)
        .groupby("client_num")['txn_date'].shift(1)
    )
    days_df["prev_txn_date"] = days_df["prev_txn_date"].fillna(
        start_period_dt.date() - relativedelta(days=1)
    )
    days_df["txn_gap_days"] = (days_df['txn_date'] - days_df['prev_txn_date']).dt.days

    days_df = days_df.groupby(["client_num"]).agg(
        txn_days_cnt=("txn_date", "count"),
        day_txn_cnt_min=("txn_cnt", "min"),
        day_txn_cnt_max=("txn_cnt", "max"),
        day_txn_cnt_avg=("txn_cnt", "mean"),
        day_txn_cnt_median=('txn_cnt', 'median'),
        day_txn_sum_min=("txn_sum", "min"),
        day_txn_sum_max=("txn_sum", "max"),
        day_txn_sum_avg=("txn_sum", "mean"),
        day_txn_sum_median=('txn_sum', 'median'),
        txn_max_days_gap=("txn_gap_days", "max"),
        txn_avg_days_gap=("txn_gap_days", "mean")
    )

    if with_mcc_codes:
        # MCC features: cnt/sum by each mcc code
        mcc_df = wind_df.groupby(["client_num", "mcc_code"]).agg(
            txn_cnt=("amount", "count"),
            txn_sum=("amount", "sum")
        ).reset_index()
    
        mcc_df = mcc_df.pivot(
            index='client_num', columns='mcc_code', 
            values=["txn_cnt", "txn_sum"]
        )
        mcc_df.columns = ["mcc_" + "_".join(col[::-1]) for col in mcc_df.columns]

    # Join to result DataFrame
    res_df = general_df.merge(days_df, left_index=True, right_index=True, how='inner')
    if with_mcc_codes:
        res_df = res_df.merge(mcc_df, left_index=True, right_index=True, how='inner')
    res_df = res_df.fillna(0)

    if wind_postfix:
        res_df.columns = ["_".join([col, wind_postfix]) for col in res_df.columns]

    print(f"Calc features for window {num_days} days - DONE")

    return res_df


def get_diff_features(
    period_features: pd.DataFrame,
    next_period_df: pd.DataFrame, 
    wind_postfix: Optional[str] = None
) -> pd.DataFrame:
    """
        Функция для получения diff фичей между текущим окном и предыдущим

        Params:
            period_features:
                Рассчитанные фичи для текущего периода (см. get_window_features)
            next_period_df:
                Исходные данные для следующего периода
            wind_postfix: str or None, deafult None
                    Постфикс, который будет добавлен названиям всех фичей
        Returns:
            DataFrame с diff фичaми
    """
    print(f"Calc diff features for window {wind_postfix} - START")

    next_period_features = (
        next_period_df
        .groupby(["client_num", "txn_date"])
        .agg(
            txn_cnt=("amount", 'count'),
            txn_sum=("amount", 'sum')
        )
        .reset_index()
        .groupby("client_num")
        .agg(
            next_txn_days_cnt=("txn_date", "count"),
            next_txn_cnt=("txn_cnt", 'sum'),
            next_txn_sum=("txn_sum", 'sum')
        )
    )

    features_list = ["txn_days_cnt", "txn_cnt", "txn_sum"]
    period_cols = ["_".join([feat_name, wind_postfix]) for feat_name in features_list]
    
    periods_df = next_period_features.merge(
        period_features[period_cols], 
        left_index=True, right_index=True,
        how='outer'
    )
    periods_df = periods_df.fillna(0)
    
    for period_col, feature_name in zip(period_cols, features_list):
        periods_df[f"diff_{feature_name}"] = periods_df[period_col] - periods_df[f"next_{feature_name}"]    
        periods_df.drop(columns=[period_col, f"next_{feature_name}"], inplace=True)
    
    if wind_postfix:
        periods_df.columns = ["_".join([col, wind_postfix]) for col in periods_df.columns]

    print(f"Calc diff features for window {wind_postfix} - DONE")
    
    return periods_df    

def get_transactions_features(df_txn: pd.DataFrame, score_date: date) -> pd.DataFrame:
    """
        Функция для рассчёта и получения полного DataFrame с фичами 
        на основе исходных данных о транзакциях
        Все null значения заполняются нулями
    
        Params:
            df_txn: pd.DataFrame
                DataFrame с исходными данными по транзакциям
            score_date: date
                Дата расчёта, первый день месяца

        Returns:
            DataFrame с фичами для обучения модели
    """
    df_txn["txn_date"] = df_txn['date_time'].dt.date

    windows = [
        # Structure: (<num units>, <units>, <with_mcc_codes flag (optional)>)
        # (7, "days", False),
        (14, "days"),
        (1, "months"),
        # (45, "days"),
        (3, "months")
    ]

    wind_features = []
    for window in windows:
        wind_postfix = f"{window[0]}{window[1][0]}"
        if window != (3, "months"):
            start_wind_date = score_date - relativedelta(**{window[1]: window[0]})
            wind_df = df_txn[df_txn["txn_date"] >= start_wind_date]
        else:
            start_wind_date = None
            wind_df = df_txn

        with_mcc_codes = len(window) <= 2 or window[2]

        wind_features_df = get_window_features(
            wind_df, wind_postfix=wind_postfix, with_mcc_codes=with_mcc_codes
        )
        wind_features.append(wind_features_df)

        if start_wind_date:
            next_wind_start_date = start_wind_date - relativedelta(**{window[1]: window[0]})
            next_wind_df = df_txn[
                (df_txn["txn_date"] >= next_wind_start_date) & 
                (df_txn["txn_date"] < start_wind_date)
            ]
            wind_diff_features_df = get_diff_features(
                wind_features_df, next_wind_df, wind_postfix=wind_postfix
            )
            wind_features.append(wind_diff_features_df)

    # Other features
    last_txn = df_txn.groupby("client_num").agg(last_txn_date=("date_time", 'max'))
    last_txn["last_txn_days"] = (score_date - last_txn['last_txn_date']).dt.days
    last_txn.drop(columns=["last_txn_date"], inplace=True)

    # Join result DataFrame
    print("Join features - START")

    res_df = last_txn
    for wind_features_df in wind_features:
        res_df = res_df.merge(wind_features_df, left_index=True, right_index=True, how='left')

    res_df = res_df.fillna(0).reset_index()

    print("Join features - DONE")
    
    return res_df


%%time
# Считаем фичи
score_date = max_date_time.date()
print("Score date:", score_date)

features_df = get_transactions_features(df_txn, score_date=score_date)
print("Features shape:", features_df.shape)


features_df.head(10)


# Посмотрим распрделение некоторых из полученных фичей

plot_features = [
    'last_txn_days', ('txn_cnt_3m', False),
    'txn_days_cnt_14d', 'txn_days_cnt_1m',
    'txn_days_cnt_3m', 'txn_max_days_gap_14d',
    'txn_max_days_gap_1m', 'txn_max_days_gap_3m'
]
subplot_size = (round(len(plot_features) / 2), 2)
fig_size = (12, 4 * subplot_size[0])
plt.figure(figsize=fig_size)


def plot_subplot(feature_name: str, subplot_num: int, set_ticks: bool = True):
    if not set_ticks:
        xticks = None
    elif feature_name.split('_')[-1] == "7d":
        xticks=[i for i in range(8)]
    elif feature_name.split('_')[-1] == "14d":
        xticks=[i * 2 for i in range(8)]
    elif feature_name.split('_')[-1] == "1m":
        xticks=[i * 3 for i in range(11)]
    else:
        xticks=[i * 10 for i in range(10)]

    plt.subplot(*subplot_size, subplot_num)
    features_df.groupby(feature_name)['client_num'].count().plot(xticks=xticks)


for i, feature_name in enumerate(plot_features):
    if isinstance(feature_name, tuple):
        plot_subplot(feature_name[0], i + 1, set_ticks=feature_name[1])
    else:
        plot_subplot(feature_name, i + 1)


# Веса классов для WMAE
CLASS_WEIGHTS = {
    0: 1.0, 1: 0.72, 2: 0.52, 3: 0.37,
    4: 0.27, 5: 0.19, 6: 0.14
}


# Добавляем веса к таргетам

df_train['weight'] = df_train['target'].map(CLASS_WEIGHTS)
df_train.head()


# Получаем полный датасет для трейна
train_features = features_df.merge(df_train, on='client_num', how='inner')
print("Train shape:", train_features.shape)


# Делим на обучающую и валидационную выборку с стратификацией по таргету
# Объём на валидацию - 15%
X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
    train_features.drop(columns=['client_num', 'target', 'weight']), 
    train_features['target'], 
    train_features['weight'],
    test_size=0.15,
    random_state=42,
    stratify=train_features['target']
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)


# Создаем DMatrix с весами

dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
dval = xgb.DMatrix(X_val, label=y_val, weight=weights_val)


# Преобразование в классы с учетом границ
def continuous_to_class(preds):
    # Обеспечиваем попадание в диапазон [0, 7]
    clipped = np.clip(preds, 0, 6)
    rounded = np.round(clipped)
    return rounded.astype(int)


# Задаём расчёт метрики для xgb
def wmae_xgb(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()

    # Расчет взвешенной MAE
    preds_class = continuous_to_class(preds)
    errors = np.abs(labels - preds_class)
    wmae = np.mean(weights * errors)

    return 'wmae', wmae


# Общие параметры модели, остальные будут подобраны с использованием Optuna
MODEL_PARAMS = {
    'objective': 'reg:absoluteerror',
    "booster": "gbtree",
    'eval_metric': 'mae',
    'tree_method': 'hist',
    "device": "cuda"
}
# Список различных сидов для обучения модели для получения более стабильных результатов
SEEDS = [
    42, 672, 961, 956, 389, 
    387, 438, 379, 755, 173
]


def train_batch_models(params, dtrain, dval = None, num_boost_round=1000, verbose_eval=500):
    """
        Функция для обучения списка моделей с заданными параметрами

        Params:
            params: Dict[str, Any]
                Список всех параметров для обучения модели
            dtrain: xgb.DMatrix
                Тренировочный набор данных
            dval: xgb.DMatrix or None, default None
                Валидационный набор данных
                Если None, то валидация не проводится
            num_boost_round: int, default 1000
                Количество эпох для обучения модели (максимальное)
            verbose_eval: int, default 500
                Метрики моделей на эпохах, кратным данному значению, будут логироваться
        Returns:
            Возвращается tuple из двух элементов
            Первый элемент список всех обученных моделей
            Второй элемент - результирующее значение метрики WMAE на валидации
            Если dval не был передан, то во втором элементе будет лежать 0
    """
    models = []
    predicts = []
    for seed in SEEDS:
        params['seed'] = seed

        evals = [(dtrain, 'train')]
        if dval is not None:
            evals.append((dval, 'val'))
    
        model = xgb.train(
            params, dtrain, 
            num_boost_round=num_boost_round,
            evals=evals,
            custom_metric=wmae_xgb,
            early_stopping_rounds=50,
            verbose_eval=verbose_eval
        )
        models.append(model)

        if dval is not None:
            val_preds_cont = model.predict(dval)
            _, score = wmae_xgb(val_preds_cont, dval)
            predicts.append(val_preds_cont)

    if predicts:
        models_pred_cls = continuous_to_class(np.mean(predicts, axis=0))
        _, res_wmae = wmae_xgb(models_pred_cls, dval)
    else:
        res_wmae = 0
    return models, res_wmae


# Обучение моделей с подбором гиперпараметров
def objective(trial):
    params = {
        # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
        "lambda": trial.suggest_float("lambda", 1e-5, 1.0, log=True),
        'gamma': trial.suggest_float("gamma", 1e-5, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'max_depth': trial.suggest_int("max_depth", 4, 10),
        'min_child_weight': trial.suggest_int("min_child_weight", 10, 100)
    }
    params.update(MODEL_PARAMS)

    _, score = train_batch_models(params, dtrain, dval)
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


study.best_params


# Задаём оптимальную конфигурацию

params = {
    'learning_rate': 0.09231095550711414,
     'lambda': 0.2104939648855454,
     'gamma': 0.0011188229126978212,
     'alpha': 0.06379171385162212,
     'subsample': 0.5846624679146472,
     'colsample_bytree': 0.8128808827251961,
     'max_depth': 10,
     'min_child_weight': 81
}

params.update(MODEL_PARAMS)

models, models_wmae = train_batch_models(params, dtrain, dval)
print("All models wmae:", models_wmae)


# Важность признаков
def show_models_features_importance(models, num_features=10):
    rows_per_model = num_features + 3
    messages = [""] * (rows_per_model * len(models) // 2)
    
    for i, model in enumerate(models):
        importance = model.get_score(importance_type='gain')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
        indx = rows_per_model * (i // 2)
        messages[indx] += f"\t№{i+1} - Топ {num_features} важных признаков:\t"
    
        for f_i, (feat, gain) in enumerate(sorted_importance[:num_features]):
            col_name = train_features.columns[int(feat[1:]) + 1]
            if len(col_name) < 16:
                col_name += "\t"
            messages[indx + f_i + 2] += f"\t{col_name}\t{gain:.4f}\t"

    for mes in messages:
        print(mes)

show_models_features_importance(models)


# Создаём DMatrix для всего набора данных

dtrain_full = xgb.DMatrix(
    train_features.drop(columns=['client_num', 'target', 'weight']), 
    label=train_features['target'], 
    weight=train_features['weight']
)


# Выбираем фичи по тестовым клиентам и создаём Dmatrix
test_features = features_df[features_df['client_num'].isin(test_clients)]

dtest = xgb.DMatrix(
    test_features.drop(columns=['client_num'])
)


# Обучаем модель на всех данных и прогоняем на тесте
params = {
    'learning_rate': 0.09,
    'lambda': 0.22,
    'gamma': 0.002,
    'alpha': 0.064,
    'subsample': 0.55,
    'colsample_bytree': 0.7,
    'max_depth': 8,
    'min_child_weight': 90
}

params.update(MODEL_PARAMS)

models, _ = train_batch_models(
    params, dtrain_full,
    # Ограничиваем num_boost_round=1500, чтобы сильно не переобучалась
    num_boost_round=1500, 
    verbose_eval=1000
)

# Итоговый prediction как средний предикт по всем обученным моделям
models_pred = [model.predict(dtest) for model in models]
models_pred_cls = continuous_to_class(np.mean(models_pred, axis=0))

submission = pd.DataFrame({
    "client_num": test_features['client_num'],
    "target": models_pred_cls
})
submission.head()


# Выведим распределением таргетов на тесте 
# в сравнении с распределением таргетов на обучающей выборке
train_target_proportion = (
    df_train["target"]
    .value_counts(normalize=True)
    .reset_index()
)
train_target_proportion.columns = ["target", "train_proportion"]

submission_target_proportion = (
    submission['target']
    .value_counts(normalize=True)
    .reset_index()
)
submission_target_proportion.columns = ["target", "submission_proportion"]

both_target_proportion = train_target_proportion.merge(
    submission_target_proportion, on='target', how='outer'
).sort_values("target")

both_target_proportion.plot.bar(
    x="target", 
    y=["train_proportion", "submission_proportion"], 
    rot=0
)


show_models_features_importance(models)


# Сохраняем сабмит, забираем Top-1 на контесте
submission.to_csv("submission.csv", index=False)

