!pip install git+https://github.com/VsevolodL27/tabular_dae@main


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import gridspec
from scipy.special import erfinv
from numba import njit
from sklearn.model_selection import KFold
from multiprocessing import Pool, cpu_count
from tabular_dae import DAE
from tabular_dae.model import load

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
submission_df = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/sample_submission.csv')


def eval_gini(y_true, y_prob):
    """Функция для расчета метрики Джини"""
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def gini_lgb(preds, dtrain):
    """Функция для расчета Джини на тестовой выборке"""
    y = list(dtrain.get_label())
    score = eval_gini(y, preds) / eval_gini(y, y)
    return 'gini', score, True


# Оставляем все колонки, кроме 'id','target'
col = [c for c in train_df.columns if c not in ['id','target']]
# Удаляем колонки содержащие calc
col = [c for c in col if not c.startswith('ps_calc_')]

# Собираем айдишники
id_test = test_df['id'].values
id_train = train_df['id'].values

# Подготовка датасета
y = train_df['target']
X = train_df[col]
y_valid_pred = 0*y
X_test = test_df.drop(['id'], axis=1)[col]
y_test_pred = 0


# Инициализация модели
dae = DAE(
    body_network='deepstack',
    body_network_cfg=dict(hidden_size=768),
    swap_noise_probas=.15,
    device='cuda',
)

# Обучение модели на полной выборке
df = pd.concat([X, X_test], axis=0)

# Обучение модели
dae.fit(df, batch_size=100_000, max_epochs=150, verbose=1, optimizer_params={'lr': 3e-4})


# Сохраняем модель
dae.save('/kaggle/working/dae_150_768.pkl')


# Гиперпараметры для обучения
MAX_ROUNDS = 1200
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.024


# Установка параметров для LGBM Classifier
params = {
    'learning_rate': LEARNING_RATE,
    'max_depth': 4,
    'lambda_l1': 16.7,
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': .7,
    'is_training_metric': False,
    'verbosity': -1,
    'seed': 99
}


# Параметры для разбиения на фолды
K = 5
kf = KFold(n_splits=K, random_state=1, shuffle=True)


for i, (train_index, test_index) in enumerate(kf.split(train_df)): 

    # Разбиение на тестовый и валидационные фолды
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index].copy()
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    test = test_df.copy()[col]
    print( "\nFold ", i)

    # Применение DAE
    X_train = dae.transform(X_train)
    X_valid = dae.transform(X_valid)
    test = dae.transform(test)
    
    # Запуск модели на фолдах
    if OPTIMIZE_ROUNDS:
        fit_model = lgb.train(
                               params,
                               lgb.Dataset(X_train, label=y_train),
                               MAX_ROUNDS,
                               lgb.Dataset(X_valid, label=y_valid),
                               feval=gini_lgb,
                               early_stopping_rounds=200
                             )
        print( " Best iteration = ", fit_model.best_iteration )
        pred = fit_model.predict(X_valid, num_iteration=fit_model.best_iteration)
        test_pred = fit_model.predict(test[col], num_iteration=fit_model.best_iteration)
    else:
        fit_model = lgb.train(
                               params,
                               lgb.Dataset(X_train, label=y_train),
                               MAX_ROUNDS,
                             )
        pred = fit_model.predict(X_valid)
        test_pred = fit_model.predict(test)

    # Сохраняем валидационные предсказания для текущего фолда
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = (np.exp(pred) - 1.0).clip(0,1)

    # Накопление прогнозов тестового набора
    y_test_pred += (np.exp(test_pred) - 1.0).clip(0,1)

# Усредненные прогнозы тестового набора
y_test_pred /= K

# Вывод значений метрики
print( "\nGini for full training set:" )
eval_gini(y, y_valid_pred)


# Сохранение валидационных предсказаний для использования в стэкинге
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('lgb_valid.csv', float_format='%.6f', index=False)


# Создания submission файла
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('lgb_submit_dae.csv', float_format='%.6f', index=False)

