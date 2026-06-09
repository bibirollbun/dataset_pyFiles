import pandas as pd
import numpy as np
import warnings


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head(3)


train.describe()


train.info()


train.isna().sum()


test.head(3)


test.describe()


test.info()


test.isna().sum()


train['was_contacted_before'] = train['pdays'].apply(lambda x: 1 if x > 0 else 0)
test['was_contacted_before'] = test['pdays'].apply(lambda x: 1 if x > 0 else 0)


train.loc[train['pdays'] == -1, 'pdays'] = 99999
test.loc[test['pdays'] == -1, 'pdays'] = 99999


cat_cols=train.select_dtypes(include=['object']).columns
num_cols=train.select_dtypes(include=['int']).columns

print(f'Total Categorical Columns {len(cat_cols)}')
print(f'Total Numerical Columns {len(num_cols)}')


test_id=test['id']


X = train.drop(['y', 'id'], axis=1)
y = train['y']
test=test.drop('id',axis=1)


cat_cols = X.select_dtypes(include=['object']).columns.tolist()
cat_cols.append('was_contacted_before')


from sklearn.model_selection import StratifiedKFold,train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
import optuna


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'random_state': 42,
        'eval_metric': 'AUC',
        'verbose': 0,
        'early_stopping_rounds': 50,
        'task_type': 'GPU',       
        'devices': '0'
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    return roc_auc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5, show_progress_bar=True)
best_params = study.best_trial.params
print("\nBest Hyperparameters from Optuna:")
print(best_params)


best_params['random_state'] = 42
best_params['eval_metric'] = 'AUC'
best_params['verbose'] = 100
best_params['early_stopping_rounds'] = 50


NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"\n--- Starting Fold {fold_ + 1}/{NFOLDS} ---")
    
    trn_data = X.iloc[trn_idx]
    val_data = X.iloc[val_idx]
    trn_y = y.iloc[trn_idx]
    val_y = y.iloc[val_idx]
    
    model = CatBoostClassifier(**best_params)
    
    model.fit(trn_data, trn_y, cat_features=cat_cols,
              eval_set=(val_data, val_y), use_best_model=True)
    
    oof_preds[val_idx] = model.predict_proba(val_data)[:, 1]
    test_preds += model.predict_proba(test)[:, 1] / NFOLDS


print("OOF (Out-of-Fold) ROC AUC Score:", roc_auc_score(y, oof_preds))


submission = pd.DataFrame({'id': test_id, 'y': test_preds})

submission.to_csv('submission.csv', index=False)

