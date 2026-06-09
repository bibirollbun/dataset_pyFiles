!pip install lightautoml &> /dev/null


import numpy as np
import pandas as pd

from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

train_data.head(3)


N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TARGET_NAME = 'accident_risk'
TIMEOUT = 3600


task = Task('reg', loss='mse', metric='mse')

roles = {
    'target': TARGET_NAME,
    'drop': ['id']
}

automl = TabularUtilizedAutoML(
    task = task,
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
)


%%time

automl.fit_predict(train_data, roles = roles, verbose = 1)
test_predictions = automl.predict(test_data)


preds = automl.predict(test_data)
test_data["accident_risk"] = preds.data[:, 0]
submission_df = test_data[['id', 'accident_risk']]
submission_df.to_csv("baseline__submission.csv", index=False)

