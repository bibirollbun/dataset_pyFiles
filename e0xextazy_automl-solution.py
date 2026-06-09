!pip3 install -U lightautoml > null


import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import torch
import gc

warnings.filterwarnings('ignore')

# Константы
RANDOM_STATE = 42
N_FOLDS = 5
TIMEOUT = 4 * 3600 # 4 часа на каждую нофигурацию + 4 часа на все остальное
N_THREADS = 4
TARGET_NAME = 'target'

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


# COMPETITION METRIC FROM Konstantin Yakovlev
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def group_stratify_train_test_split(data, test_size=0.2, random_state=RANDOM_STATE):
    customers_target = data.groupby('customer_ID')[TARGET_NAME].first().reset_index()
    
    train_customers, test_customers = train_test_split(
        customers_target['customer_ID'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=customers_target[TARGET_NAME]
    )
    
    train_mask = data['customer_ID'].isin(train_customers)
    test_mask = data['customer_ID'].isin(test_customers)
    
    train_set = data[train_mask].copy()
    test_set = data[test_mask].copy()
    
    print(f"Train размер: {train_set.shape}")
    print(f"Test размер: {test_set.shape}")
    
    return train_set, test_set


train_data = pd.read_parquet('/kaggle/input/amex-parquet/train_data.parquet')
print(f"Размер train данных: {train_data.shape}")


train_data, val_data = group_stratify_train_test_split( # Так как в тесте новые юзеры, то в треин и вал тоже не должно быть пересечений по юзерам + стратификация
    train_data, 
    test_size=0.2,
    random_state=RANDOM_STATE
)


task = Task('binary')
roles = {
    'target': TARGET_NAME,
    'drop': ['customer_ID']
}
automl_basic = TabularAutoML(
    task=task,
    timeout=TIMEOUT,
    cpu_limit=N_THREADS,
    general_params={'use_algos': [['linear_l2', 'lgb', 'cb']]},
    reader_params={
        'n_jobs': 4,
        'cv': N_FOLDS,
        'random_state': RANDOM_STATE,
        'advanced_roles': False
    }
)


oof_pred_basic = automl_basic.fit_predict(
    train_data, 
    roles=roles, 
    verbose=3
)


val_pred_basic = automl_basic.predict(val_data)


val_score_basic = amex_metric_mod(
    val_data[TARGET_NAME].values, 
    val_pred_basic.data[:, 0]
)

print(f"\nРезультаты конфигурации 1:")
print(f"Validation score: {val_score_basic:.6f}")


del oof_pred_basic, val_pred_basic
gc.collect()


automl_advanced = TabularAutoML(
    task=task,
    timeout=TIMEOUT,
    cpu_limit=N_THREADS,
    general_params={
        'use_algos': [['xgb', 'lgb_tuned', 'cb_tuned'], ['mlp']],
    },
    reader_params={
        'n_jobs': N_THREADS,
        'cv': N_FOLDS,
        'random_state': 2 * RANDOM_STATE,
        'advanced_roles': True,
    },
    tuning_params={'max_tuning_iter': "auto", 'max_tuning_time': 300},
    selection_params={'mode': 1}
)


oof_pred_advanced = automl_advanced.fit_predict(
    train_data, 
    roles=roles, 
    verbose=3
)


val_pred_advanced = automl_advanced.predict(val_data)


val_score_advanced = amex_metric_mod(
    val_data[TARGET_NAME].values, 
    val_pred_advanced.data[:, 0]
)

print(f"\nРезультаты конфигурации 2:")
print(f"Validation score: {val_score_advanced:.6f}")


del oof_pred_advanced, val_pred_advanced, val_data, train_data
gc.collect()


if val_score_basic > val_score_advanced:
    best_model = automl_basic
    best_val_score = val_score_basic
    best_config = "Конфигурация 1"
else:
    best_model = automl_advanced
    best_val_score = val_score_advanced
    best_config = "Конфигурация 2"

print(f"ЛУЧШАЯ МОДЕЛЬ: {best_config}")
print(f"Лучший validation score: {best_val_score:.6f}")

del automl_basic, automl_advanced
gc.collect()


from tqdm.auto import tqdm

batch_size = 1000000
results = []

for chunk in tqdm(pd.read_csv('/kaggle/input/amex-default-prediction/test_data.csv', chunksize=batch_size), total=(11000000 // batch_size) + 1):
    customer_ids = chunk['customer_ID']
    chunk_for_pred = chunk.drop('customer_ID', axis=1)
    preds = best_model.predict(chunk_for_pred)
    
    results.append(pd.DataFrame({
        'customer_ID': customer_ids,
        'prediction': preds.data[:, 0] if hasattr(preds, 'data') else preds[:, 0]
    }))

submission = pd.concat(results, ignore_index=True)
submission.groupby(["customer_ID"]).agg("mean").reset_index().to_csv("submission.csv", index=False)




