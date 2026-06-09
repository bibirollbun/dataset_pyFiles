import pandas as pd
import numpy as np 
import warnings


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.shape


train.info()


train.describe().T


train.isna().sum()


test.shape


test.info()


test.describe()


test.isna().sum()


train.dtypes


test_id=test['id']


train['was_contacted_before'] = train['pdays'].apply(lambda x: 1 if x > 0 else 0)
test['was_contacted_before'] = test['pdays'].apply(lambda x: 1 if x > 0 else 0)
train.loc[train['pdays'] == -1, 'pdays'] = 99999
test.loc[test['pdays'] == -1, 'pdays'] = 99999


from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import optuna


X = train.drop(['y', 'id'], axis=1)
y = train['y']
test = test.drop('id', axis=1)


cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)


X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test)
print("Preprocessing complete. All data is now numerical.")


def lgb_objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMClassifier(**params)
    
    model.fit(X_train, y_train,
              callbacks=[lgb.early_stopping(50, verbose=False)],
              eval_set=[(X_test, y_test)]
            )
    
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    return roc_auc


print("Starting LightGBM Hyperparameter Tuning...")
study = optuna.create_study(direction='maximize')
study.optimize(lgb_objective, n_trials=5, show_progress_bar=True)
best_params = study.best_trial.params
print("\nBest LightGBM Hyperparameters:")
print(best_params)


best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['seed'] = 42
best_params['verbose'] = -1
best_params['n_jobs'] = -1


NFOLDS = 10
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))


for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_processed, y)):
    print(f"\n--- Starting Fold {fold_ + 1}/{NFOLDS} ---")
    
    model = lgb.LGBMClassifier(**best_params)
                               
    model.fit(X_processed[trn_idx], y.iloc[trn_idx],
                 callbacks=[lgb.early_stopping(50, verbose=False)],
                 eval_set=[(X_processed[val_idx], y.iloc[val_idx])]
                )
    
    oof_preds[val_idx] = model.predict_proba(X_processed[val_idx])[:, 1]
    test_preds += model.predict_proba(test_processed)[:, 1] / NFOLDS



print("OOF (Out-of-Fold) ROC AUC Score:", roc_auc_score(y, oof_preds))


submission = pd.DataFrame({'id': test_id, 'y': test_preds})
submission.to_csv('/kaggle/working/submission.csv',index=False)

