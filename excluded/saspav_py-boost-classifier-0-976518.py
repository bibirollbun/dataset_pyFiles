import os
import shutil

# Скачиваем все .csv файлы
output_dir = '.'
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        full_path = os.path.join(dirname, filename)
        if filename != 'sample_submission.csv':
            shutil.copy(full_path, output_dir)
            print(f'Скачан файл: {filename}')


# Скачиваем дополнительные модули из репо
!wget -q https://raw.githubusercontent.com/saspav/python_for_pro/main/final_cat/some_functions_clf.py > /dev/null 2>&1


!pip install py-boost -q 2>/dev/null


import re
import random
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor


from py_boost import GradientBoosting

import warnings

warnings.filterwarnings("ignore")

from some_functions_clf import (SEED, make_train_valid, DataTransform, 
                                train_valid_model, make_submit)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Зафиксируем сиды
np.random.seed(SEED)
random.seed(SEED)

# Целевая переменная
target = 'Personality'

train, valid, test = make_train_valid(add_original_df=True)

imputer, imputer_params = IterativeImputer, dict(
    initial_strategy='most_frequent',
    min_value=0,
    max_value=15,
    skip_complete=True,
    max_iter=10,
    random_state=SEED,
)

dts = DataTransform(set_category=False, set_num_int=False,
                    preprocessor=imputer, **imputer_params)

fill_nan_cat = True

# Применяем трансформации
train = dts.fit_transform(train, fill_nan_cat=fill_nan_cat)
valid = dts.transform(valid, fill_nan_cat=fill_nan_cat)

# Трансформируем признаки тестовой выборки
test = dts.transform(test, fill_nan_cat=fill_nan_cat)

# Колонки для моделей
model_columns = dts.all_features


pb_best_grid = dict(loss='bce', metric='auc',
                    ntrees=1000, max_depth=4, verbose=100,
                    lr=.01, lambda_l2=1, subsample=.8, colsample=.8,
                    min_data_in_leaf=10, min_gain_to_split=0, max_bin=256)

pb, metrics, opt_thrs = train_valid_model(GradientBoosting, 'PB', pb_best_grid,
                                          train, valid, model_columns, target)
metrics


# threshold = opt_thrs
threshold = 0.5

params_imputer = sorted(imputer_params.items())[:2]
submit_postfix = f"_{str(imputer).split('.')[-1][:-2]}-{params_imputer}-thr={threshold}"
submit_postfix = '_'.join(re.findall(r'\w+', submit_postfix))
submit_postfix = submit_postfix + ('', '_nan_cat')[fill_nan_cat]

# Вызываем функцию для формирования сабмита: передаем обученную модель
file_submit = make_submit(pb, test[model_columns], threshold, 
                          dts.reverse_mapping, postfix=submit_postfix)

shutil.copy(file_submit, 'submission.csv')

