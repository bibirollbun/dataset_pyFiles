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


%pip -q install git+https://github.com/iseedeep/deeprage.git@main
from deeprage.core import val_pie, val_bar, val_all_hist, compare_columns


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


df_train.drop('id',axis=1,inplace=True)
test_id = df_test['id']
df_test.drop('id', axis=1, inplace=True)


df_train


df_train.dtypes


df_train.shape, df_test.shape


df_train.isna().sum()


df_test.isna().sum()


val_pie(df_train, 'Soil Type')
val_bar(df_train, 'Crop Type')


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,10))
sns.heatmap(df_train.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Heatmap: Numeric cols')
plt.show()


for col in ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df_train, x='Fertilizer Name', y=col)
    plt.title(f'{col} by Fertilizer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le_soil = LabelEncoder()
le_crop = LabelEncoder()

le.fit(df_train['Fertilizer Name'])
le_soil.fit(pd.concat([df_train['Soil Type'], df_test['Soil Type']]))
le_crop.fit(pd.concat([df_train['Crop Type'], df_test['Crop Type']]))

df_train['Soil Type'] = le_soil.transform(df_train['Soil Type'])
df_train['Crop Type'] = le_crop.transform(df_train['Crop Type'])
df_train['Fertilizer Name'] = le.transform(df_train['Fertilizer Name'])
df_test['Soil Type'] = le_soil.transform(df_test['Soil Type'])
df_test['Crop Type'] = le_crop.transform(df_test['Crop Type'])

feature_cols = [
    'Soil Type', 'Crop Type',
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Potassium', 'Phosphorous'
]


X = df_train.drop('Fertilizer Name', axis=1)
y = df_train['Fertilizer Name']
X_test = df_test[feature_cols]


import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

"""
def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    for train_idx, valid_idx in cv.split(X, y):
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y[train_idx])
        dvalid = lgb.Dataset(X.iloc[valid_idx], label=y[valid_idx])
        model = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=100)
        preds = np.argmax(model.predict(X.iloc[valid_idx]), axis=1)
        accs.append(accuracy_score(y[valid_idx], preds))
    return np.mean(accs)
"""


#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=10)
#best_params = study.best_params


best_params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.1924,
    'num_leaves': 56,
    'max_depth': 6,
    'seed': 42
}


train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=100,
    callbacks=[lgb.log_evaluation(10)]
)


preds = final_model.predict(X_test)
top_3_preds = np.argsort(preds, axis=1)[:, -3:][:, ::-1]


top_3_labels = [le.inverse_transform(row) for row in top_3_preds]
submission_labels = [' '.join(map(str, labels)) for labels in top_3_labels]


submission = pd.DataFrame({
    'id': test_id,
    'Fertilizer Name': submission_labels
})
submission.to_csv('submission.csv', index=False)


submission




