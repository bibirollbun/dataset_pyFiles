!pip install --upgrade scikit-learn -q


import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import root_mean_squared_error as rmse

from catboost import Pool, CatBoostRegressor

RANDOM_STATE = 42


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')

train, test = train_test_split(train, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)


RMV = ['id', 'Price']
DISCRETE = ['Compartments']
CONTINUOUS = ['Weight Capacity (kg)']
CATEGORICAL = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


# Categorical NaN must be treated as a different category, same for Discrete
for col in CATEGORICAL+DISCRETE:
    train[col].fillna('NAN')
    # train_extra[col].fillna('NAN')
    test[col].fillna('NAN')


train_original = train.copy()
test_original = test.copy()


# CatBoost Training Utility
def train_catboost(X, y, kf):        
    models = []
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
        # CatBoost Dataset
        train_pool = Pool(X_train, label=y_train)
        val_pool = Pool(X_val, label=y_val)
    
        # Model
        model = CatBoostRegressor(
            iterations=500,
            learning_rate = 0.1,
            depth = 6, 
            verbose=100, 
            loss_function='RMSE'
        )

        # Training
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, use_best_model=True)
    
        # Store the model
        models.append(model)
        scores.append(model.best_score_['validation']['RMSE'])
        
    print(f'Mean RMSE: {np.mean(scores):.4f}')

    best_model_idx = np.argmin(scores)
    best_model = models[best_model_idx]

    return best_model


# Shared By all encodings
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


train = train_original.copy()
test = test_original.copy()

for col in CATEGORICAL:
    encoder = OrdinalEncoder(handle_unknown='error', dtype=np.float32)
    train[col] = encoder.fit_transform(train[[col]])
    test[col] = encoder.transform(test[[col]])

X_train, X_test = train.drop(columns=RMV), test.drop(columns=RMV)
y_train, y_test = train['Price'], test['Price']

oe_model = train_catboost(X_train, y_train, kf)

print('\n'*3)
print('*'*26)
RMSE = rmse(oe_model.predict(X_test), y_test)
print(f'OE RMSE: {RMSE:.4f}')


train = train_original.copy()
test = test_original.copy()

for col in CATEGORICAL:
    enc = OneHotEncoder(handle_unknown='error', dtype=np.float32, sparse_output=False)
    train[col] = enc.fit_transform(train[[col]])
    test[col] = enc.transform(test[[col]])

X_train, X_test = train.drop(columns=RMV), test.drop(columns=RMV)
y_train, y_test = train['Price'], test['Price']

ohe_model = train_catboost(X_train, y_train, kf)

print('\n'*3)
print('*'*26)
RMSE = rmse(ohe_model.predict(X_test), y_test)
print(f'OHE RMSE: {RMSE:.4f}')


train = train_original.copy()
test = test_original.copy()

for col in CATEGORICAL:
    enc = TargetEncoder()
    train[col] = enc.fit_transform(train[[col]], train['Price'])
    test[col] = enc.transform(test[[col]])

X_train, X_test = train.drop(columns=RMV), test.drop(columns=RMV)
y_train, y_test = train['Price'], test['Price']

te_model = train_catboost(X_train, y_train, kf)

print('\n'*3)
print('*'*26)
RMSE = rmse(te_model.predict(X_test), y_test)
print(f'TE RMSE: {RMSE:.4f}')

