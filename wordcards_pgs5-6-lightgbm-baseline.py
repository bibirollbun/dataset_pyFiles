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


import warnings
warnings.simplefilter('ignore')

import polars as pl
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

from lightgbm import LGBMClassifier, early_stopping

from tqdm import tqdm


PATH = '../input/playground-series-s5e6/'
train = pl.read_csv(PATH + 'train.csv')
test = pl.read_csv(PATH + 'test.csv')


def base_encoder(input_df):
    out_df = input_df.select(['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']).with_columns(
        [pl.Series(input_df[col]).cast(pl.Categorical) for col in ['Soil Type', 'Crop Type']]
    )

    return out_df

x0 = base_encoder(train)
test_x0  = base_encoder(test)

le = LabelEncoder()
y = le.fit_transform(train['Fertilizer Name'])

N_CLASS = train['Fertilizer Name'].n_unique()


# convert pl.DataFrame to numpy
feature_names  = x0.columns
cat_columns = [name for name, dtype in x0.schema.items() if dtype == pl.Categorical]

# convert categorical features to integer
all_x = pl.concat([x0, test_x0], how='vertical').with_columns(
    [pl.col(c).to_physical() for c in cat_columns]
)

x = all_x[:len(train)].to_numpy()
test_x = all_x[len(train):].to_numpy()


x0.dtypes


for i in range(x.shape[1]):
    print(x[:,i].dtype)


def single_apk(y, oof):
    sorted_oof = np.argsort(oof, axis=1)[:, ::-1][:, :3]

    score = 0
    for i in range(3):
        score += accuracy_score(y, sorted_oof[:, i]) / (i+1)

    return score


N_FOLDS = 5

oof = np.zeros((len(train), N_CLASS))
pred = np.zeros((len(test), N_CLASS))

logloss = []
map3 = []
iterations = []

fi_df = pl.DataFrame()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

model = LGBMClassifier(
    n_estimators=10000,
    max_depth=5,
    colsample_bytree=0.5,
    importance_type='gain',
    random_state=0,
    verbose=-1
)

for i, (train_idx, valid_idx) in tqdm(enumerate(skf.split(x, y))):
    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]

    model.fit(x_train, y_train,
              eval_set=(x_valid, y_valid),
              feature_name=feature_names,
              categorical_feature=cat_columns,
              callbacks=[early_stopping(stopping_rounds=100, verbose=False)])

    oof[valid_idx, :] = model.predict_proba(x_valid)
    pred += model.predict_proba(test_x) / N_FOLDS
    fi_df = pl.concat([fi_df,
                       pl.DataFrame({'feature': feature_names, 'importance':model.feature_importances_, 'fold': i})])

    logloss.append(log_loss(y_valid, oof[valid_idx, :]))
    map3.append(single_apk(y_valid, oof[valid_idx, :]))
    iterations.append(model.best_iteration_)

fold_df = pl.DataFrame({
    'fold': range(N_FOLDS),
    'Logloss': logloss,
    'MAP@3': map3,
    'iterations': iterations
})

display(fold_df)
total_logloss = log_loss(y, oof)
total_map3 = single_apk(y, oof)
print(f"Total: Logloss={total_logloss:.4f}, MAP@3={total_map3:.4f}")


sorted_pred = np.argsort(pred, axis=1)[:, ::-1]

pl.DataFrame({
    'id': test['id'],
    'pred1': le.inverse_transform(sorted_pred[:, 0]),
    'pred2': le.inverse_transform(sorted_pred[:, 1]),
    'pred3': le.inverse_transform(sorted_pred[:, 2])
}).with_columns(
    Fertilizer= pl.col('pred1') + ' ' + pl.col('pred2') + ' ' + pl.col('pred3')
).select(['id', 'Fertilizer']).rename({'Fertilizer': 'Fertilizer Name'}).write_csv('submission.csv')


_order = fi_df.group_by('feature').agg(pl.col('importance').mean().alias('mean_FI')).sort(
    'mean_FI', descending=True)['feature']

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxenplot(y='feature', x='importance', data=fi_df.to_pandas(), orient='h', order=_order, ax=ax);

