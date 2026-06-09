%load_ext cudf.pandas


import numpy as np
import pandas as pd

df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col = 'id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col = 'id')
df_orig = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')


df_train.info()


df_test.info()


df_orig.info()


df_orig = df_orig.set_index(np.arange(14633) + 698886)
df = pd.concat([df_train, df_test, df_orig])

df['LogVocalContent'] = df.VocalContent.apply(np.log)
df['LogAcousticQuality'] = df.AcousticQuality.apply(np.log)
df['LogInstrumentalScore'] = df.InstrumentalScore.apply(np.log)
df['LogLivePerformanceLikelihood'] = df.LivePerformanceLikelihood.apply(np.log)
df['DivExpAudioLoudness'] = 1 / df.AudioLoudness.apply(np.exp)

df_train = df.iloc[df_train.index]
df_test = df.iloc[df_test.index]
df_orig = df.iloc[df_orig.index]


df_train.describe()


import optuna
from optuna.pruners import SuccessiveHalvingPruner
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df_train.copy()
y = X.pop('BeatsPerMinute')

def objective(trial):
    xgb_model = XGBRegressor(
        max_depth = trial.suggest_int('max_depth', 2, 10),
        learning_rate = trial.suggest_float('learning_rate', 1e-2, 1e-1, log = True),
        n_estimators = trial.suggest_int('n_estimators', 1000, 4000),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.2, 1.0),
        subsample = trial.suggest_float('subsample', 0.2, 1.0),
        reg_alpha = trial.suggest_float('reg_alpha', 1e-4, 1e2, log = True),
        reg_lambda = trial.suggest_float('reg_lambda', 1e-4, 1e2, log = True),
        early_stopping_rounds = 50,
        eval_metric = 'rmse',
        n_jobs = 4,
        random_state = 1
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 1)
    xgb_model.fit(
        X_train, y_train,
        eval_set = [(X_valid, y_valid)],
        verbose = False
    )
    preds_valid = xgb_model.predict(X_valid)
    return mean_squared_error(y_valid, preds_valid, squared = False)

# study = optuna.create_study(direction = 'minimize', pruner = SuccessiveHalvingPruner())
# study.optimize(objective, n_trials = 100)
# best_params = study.best_trial.params


from sklearn.model_selection import cross_val_score

# X_train = pd.concat([df_train, df_orig])
X_train = df_train
y_train = X_train.pop('BeatsPerMinute')
X_test = df_test.drop('BeatsPerMinute', axis = 1)

best_params = {'max_depth': 7, 'learning_rate': 0.020510032522613342, 'n_estimators': 1190, 'min_child_weight': 10, 'colsample_bytree': 0.9439375408198358, 'subsample': 0.7235503096394793, 'reg_alpha': 0.004321280213219448, 'reg_lambda': 65.81629415232665}
xgb_model_best = XGBRegressor(**best_params)

scores = cross_val_score(xgb_model_best, X_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print('Scores:', scores)
print('Mean:', -scores.mean())


xgb_model_best.fit(X_train, y_train)
preds_test = xgb_model_best.predict(X_test)
output = pd.DataFrame({'id': X_test.index, 'BeatsPerMinute': preds_test})
output.to_csv('submission.csv', index = False)

