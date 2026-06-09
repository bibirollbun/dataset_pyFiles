import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

train.head(3)


from itertools import combinations

feature_cols = [col for col in train.columns if col not in ['id', 'BeatsPerMinute']]

for col1, col2 in combinations(feature_cols, 2):
    train[f'{col1}_x_{col2}'] = train[col1] * train[col2]
    test[f'{col1}_x_{col2}'] = test[col1] * test[col2]

print('Train Shape:', train.shape)
print('Test Shape:', test.shape)

train.head()


TARGET = 'BeatsPerMinute' 
FEATURES = [col for col in train.columns if col not in ['id', 'BeatsPerMinute']]
print(len(FEATURES),'Features:\n', FEATURES)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


X = train[FEATURES]
y = train[TARGET]


FOLDS = 5
SEED = 42

params = {
    "objective": "regression_l2",  
    "metric": "rmse",              
    "learning_rate": 0.002,
    # "max_depth": 5,
    "bagging_fraction": 0.8,       
    "feature_fraction": 0.8,       
    "seed": SEED,
    "boosting_type": "gbdt",
    "verbose": -1,                 
    "n_jobs": -1,                
}


oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print(f"--- Fold {fold+1} ---")

    X_train = train.iloc[train_idx][FEATURES].copy()
    y_train = train.iloc[train_idx][TARGET]
    
    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx][TARGET]
    X_test = test[FEATURES].copy()

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=10_000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=True),
            lgb.log_evaluation(period=200)
        ]
    )

    oof_preds[val_idx] = model.predict(X_valid, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / FOLDS


pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv('oof_lgbm_product.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv('test_lgbm_product.csv', index=False)

