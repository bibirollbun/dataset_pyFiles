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


train_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')


train_df.head(5)


train_df.dtypes


test_df = test_df.drop('price', axis=1)


test_df.head(5)


train_df.isnull().sum()


test_df.isnull().sum()


import sys
import os

file_path = os.path.abspath('/kaggle/input/russian-car-plates-prices-prediction/supplemental_english.py') 
sys.path.append(file_path)

from supplemental_english import REGION_CODES, GOVERNMENT_CODES


max_len = max(len(v) for v in REGION_CODES.values())
for k in REGION_CODES:
    while len(REGION_CODES[k]) < max_len:
        REGION_CODES[k].append(None)


region_codes_df = pd.DataFrame(REGION_CODES)
region_codes_df = region_codes_df.melt(var_name='region', value_name='region_code').dropna().reset_index(drop=True)
region_codes_df['region_code'] = region_codes_df['region_code'].astype(str)

region_codes_df.head(5)


region_codes_df.dtypes


# GOVERNMENT_CODES
# dizionario Python in cui le chiavi sono tuple che rappresentano categorie di targhe automobilistiche russe, 
# e i valori sono tuple che descrivono il significato e l'uso di ciascun tipo di targa.

records = []

for (letters, (num_from, num_to), region), (description, is_forbidden, road_advantage, significance) in GOVERNMENT_CODES.items():
    records.append({
        "letters": letters,
        "number_from": num_from,
        "number_to": num_to,
        "region": region,
        "description": description,
        "is_forbidden": is_forbidden,
        "road_advantage": road_advantage,
        "significance": significance
    })


government_codes_df = pd.DataFrame(records)
government_codes_df.head()


import re

def parse_plate(plate):
    match = re.match(r'([A-Z])(\d{3})([A-Z]{2})(\d{2,3})$', plate)
    if match:
        return match.groups()  # letters1, number, letters2, region_code
    return None, None, None, None

train_df[['letter1', 'number', 'letter2', 'region_code']] = (
    train_df['plate'].apply(lambda p: pd.Series(parse_plate(p)))
)
test_df[['letter1', 'number', 'letter2', 'region_code']] = (
    test_df['plate'].apply(lambda p: pd.Series(parse_plate(p)))
)


train_df.head(5)


test_df.head(5)


train_df['plate_letters'] = train_df['letter1'] + train_df['letter2']
train_df['plate_number'] = pd.to_numeric(train_df['number'], errors='coerce')
train_df['region_code'] = train_df['region_code'].astype(str)
government_codes_df['region'] = government_codes_df['region'].astype(str)

# Filtro preventivo: join solo dove lettere e regioni corrispondono
merged = train_df.merge(
    government_codes_df,
    left_on=['plate_letters', 'region_code'],
    right_on=['letters', 'region'],
    how='left',
    suffixes=('', '_gov')
)

# Verifica se plate_number rientra nel range
merged['is_government'] = (
    (merged['plate_number'] >= merged['number_from']) &
    (merged['plate_number'] <= merged['number_to'])
)

# Dove non c'è match, metti False
merged['is_government'] = merged['is_government'].fillna(False)

# Riporta nel train_df finale
train_df['is_government'] = merged['is_government']
train_df = train_df.drop(['letter1', 'letter2', 'number'], axis=1)


test_df['plate_letters'] = test_df['letter1'] + test_df['letter2']
test_df['plate_number'] = pd.to_numeric(test_df['number'], errors='coerce')
test_df['region_code'] = test_df['region_code'].astype(str)
government_codes_df['region'] = government_codes_df['region'].astype(str)

# Filtro preventivo: join solo dove lettere e regioni corrispondono
merged = test_df.merge(
    government_codes_df,
    left_on=['plate_letters', 'region_code'],
    right_on=['letters', 'region'],
    how='left',
    suffixes=('', '_gov')
)

# Verifica se plate_number rientra nel range
merged['is_government'] = (
    (merged['plate_number'] >= merged['number_from']) &
    (merged['plate_number'] <= merged['number_to'])
)

# Dove non c'è match, metti False
merged['is_government'] = merged['is_government'].fillna(False)

# Riporta nel train_df finale
test_df['is_government'] = merged['is_government']
test_df = test_df.drop(['letter1', 'letter2', 'number'], axis=1)


train_df.head(5)


train_df = train_df.merge(region_codes_df, on='region_code', how='left')
test_df = test_df.merge(region_codes_df, on='region_code', how='left')


train_df.head(5)


test_df.head(5)


from sklearn.preprocessing import LabelEncoder

cols = ['plate_letters', 'is_government', 'region']

for col in cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

all_plates = pd.concat([train_df['plate'], test_df['plate']]).astype(str)
le = LabelEncoder()
le.fit(all_plates)
train_df['plate'] = le.transform(train_df['plate'].astype(str))
test_df['plate'] = le.transform(test_df['plate'].astype(str))


train_df.head(5)


test_df.head(5)


def date_change(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday  # 0 = Monday, 6 = Sunday
    df = df.drop(['date'], axis=1)
    return df


train_df = date_change(train_df)
test_df = date_change(test_df)
train_df.head(5)


train_df.dtypes


test_df.dtypes


train_df['region_code'] = train_df['region_code'].astype('int64')
test_df['region_code'] = test_df['region_code'].astype('int64')


test_ids = test_df['id']
y = train_df['price']
X = train_df[['plate', 'region_code', 'plate_letters', 'plate_number', 'is_government', 'region', 'year', 'month']]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train), len(y_train), len(X_test), len(y_test)


X_train


y_train


import numpy as np
from sklearn.metrics import make_scorer

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    smape_val = np.where(denominator == 0, 0, diff / denominator)
    return np.mean(smape_val) * 100


smape_scorer = make_scorer(smape, greater_is_better=False)


correlation = X.assign(target=y).corr()['target'].drop('target')
print(correlation.sort_values(ascending=False))


from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

models = {
    'xgb': {
        'model': XGBRegressor(device='cuda', tree_method='hist'),
        'params': {
            'n_estimators': [1000, 1500],
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1],
        }
    },
    'catboost': {
        'model': CatBoostRegressor(task_type='GPU', devices='0'),
        'params': {
            'n_estimators': [1000, 1500],
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1],
        }
    }
}

best_models = {}

for model_name, config in models.items():
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        scoring=smape_scorer,
        cv=5,
        verbose=100
    )

    grid_search.fit(X_train, y_train, verbose=False)

    best_models[model_name] = {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
    
print(f"\n**{model_name.upper()}**")
print("Migliori parametri:", grid_search.best_params_)
print("Miglior ROC AUC (CV):", grid_search.best_score_)


model = XGBRegressor(
    colsample_bytree=1.0, 
    gamma=0, 
    learning_rate=0.01, 
    max_depth=6, 
    n_estimators=1500, 
    reg_alpha=0, 
    reg_lambda=2, 
    subsample=0.8
)

model.fit(X_train, y_train, verbose=False)
pred = model.predict(X_test)
smape(y_test, pred)


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

best_xgb = model #best_models['xgb']['best_model']
best_catboost = best_models['catboost']['best_model']

meta_model = CatBoostRegressor(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.05,
    task_type='GPU', 
    devices='0',
    verbose=100
)

stacked_model = StackingRegressor(
    estimators=[
        ('xgb', best_xgb),
        ('catboost', best_catboost)
    ],
    final_estimator=meta_model,  
    cv=5, 
    passthrough=False,  
)

stacked_model.fit(X_train, y_train)


stacked_preds = stacked_model.predict(X_test)
stacked_smape = smape(y_test, stacked_preds)
print(f"\nSMAPE dello Stacking Ensemble: {stacked_smape:.4f}")


test_sub = test_df[['plate', 'region_code', 'plate_letters', 'plate_number', 'is_government', 'region', 'year', 'month']]


submission_predictions = stacked_model.predict(test_sub)


submission = pd.DataFrame({'id': test_ids.values, 'price': submission_predictions})
submission.head(5)


submission.to_csv('/kaggle/working/russian_plate_submission.csv', index=False)

