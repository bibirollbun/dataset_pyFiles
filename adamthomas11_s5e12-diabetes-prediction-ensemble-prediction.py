# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


cb = pd.read_csv('/kaggle/input/s5e12-diabetes-prediction-catboost/diabetes_pred_cb.csv')
lgbm = pd.read_csv('/kaggle/input/s5e12-diabetes-prediction-lgbm/diabetes_pred_lgbm.csv')

print(cb.head())
print(lgbm.head())


cb = cb.sort_values('id').reset_index(drop=True)
lgbm = lgbm.sort_values('id').reset_index(drop=True)

assert (cb['id'].values == lgbm['id'].values).all()


cb_preds = cb['diagnosed_diabetes'].values
lgbm_preds = lgbm['diagnosed_diabetes'].values

print('CB mean:', cb_preds.mean(), 'std:', cb_preds.std())
print('LGBM mean:', lgbm_preds.mean(), 'std:', lgbm_preds.std())


corr = np.corrcoef(cb_preds, lgbm_preds)[0, 1]
print('Prediction correlation:', corr)


abs_diff = np.abs(cb_preds - lgbm_preds)

print('Mean abs diff:', abs_diff.mean())
print('95th percentile diff:', np.percentile(abs_diff, 95))
print('Max diff:', abs_diff.max())


ensemble_preds = 0.5 * cb_preds + 0.5 * lgbm_preds


submission = pd.DataFrame({
    'id': cb['id'],
    'diagnosed_diabetes': ensemble_preds
})

submission.to_csv('submission.csv', index=False)

submission


plt.hist(cb_preds, bins=50, alpha=0.5, label='CatBoost')
plt.hist(lgbm_preds, bins=50, alpha=0.5, label='LGBM')
plt.legend()
plt.title('Prediction distributions')
plt.show()

