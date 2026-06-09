!pip install -U -q lightautoml


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task


data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
data


N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 36000
TARGET_NAME = 'y'


np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


train_data, test_data = train_test_split(
    data,
    test_size=TEST_SIZE,
    stratify=data[TARGET_NAME],
    random_state=RANDOM_STATE
)

print(f'Data is splitted. Parts sizes: train_data = {train_data.shape}, test_data = {test_data.shape}')

train_data.head()


roles = {
    'target': TARGET_NAME,
    'drop': ['id']
}


task = Task(
    'binary',
    loss = 'logloss',
    metric = 'auc'
)


automl = TabularAutoML(
    task = task,
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
)


oof_preds = automl.fit_predict(train_data, roles = roles, verbose = 1)


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


test_predictions = automl.predict(test)


pd.DataFrame({
    'id': test['id'],
    'y': test_predictions.data.reshape(-1)
}).to_csv('submission.csv', index=False)

