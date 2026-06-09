import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

from xgboost import XGBClassifier


train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
orig_path = "/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv"
submit_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"


train_df = pd.read_csv(train_path, index_col = "id")
train_df.info()


test_df = pd.read_csv(test_path)
test_df.info()


orig_df = pd.read_csv(orig_path)
orig_df.info()


orig_df.columns = orig_df.columns.str.replace(' ', '')
orig_df = orig_df[train_df.columns].copy()
orig_df['rainfall'] = orig_df['rainfall'].map({'no': 0, 'yes': 1})

orig_df = orig_df.fillna(method='bfill')
test_df = test_df.fillna(method='bfill')


train_df.loc[train_df['maxtemp'] < train_df['mintemp']][['day', 'mintemp','maxtemp', 'temparature']]


for df_name, df in [('train', train_df), ('original', orig_df), ('test', test_df)]:
    num_of_duplicates = df.duplicated().sum()
    if num_of_duplicates != 0:
        print(f'The {df_name} dataset has {num_of_duplicates} duplicates. They need to be dropped.')
    else:
        print(f'The {df_name} dataset has no duplicates')


train_comb_df = pd.concat([train_df, orig_df], ignore_index=True)
orig_df.info()


def preprocess(df):
    df['dew_humidity'] = df['dewpoint']*df['humidity']
    df['temp_previous_day'] = df['temparature'].shift(1).fillna(0)
    df['temp_next_day'] = df['temparature'].shift(-1).fillna(0)
    df['humidity_previous_day'] = df['humidity'].shift(1).fillna(0)
    df['pressure_previous_day'] = df['pressure'].shift(1).fillna(0)
    df['day_bins'] = pd.cut(df['day'], bins=12, labels=range(1, 13))
    X = df.copy()
    try:
        y = X.pop('rainfall')
        return X, y
    except:
        return X


ts = test_df.copy()
X_ts = preprocess(ts)

tr_c = train_comb_df.copy()
X, y = preprocess(tr_c)

column_trans = make_column_transformer(
    (OneHotEncoder(), X.select_dtypes('object').columns.tolist()),
    remainder='passthrough', 
    sparse_threshold=0)
pd.DataFrame(column_trans.fit_transform(X), columns=X.columns).head()


def objective(trial):
    xgb_param_grid = {
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0,  step=0.05),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 10000, step=100),
        'eta': trial.suggest_float('eta', 0.01, 0.1,  step=0.01),
        'reg_alpha': trial.suggest_int('reg_alpha', 1, 50),
        'reg_lambda': trial.suggest_int('reg_lambda', 5, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
    }
    model = make_pipeline(column_trans, 
                          XGBClassifier(**xgb_param_grid))
   
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=8)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, preds)
    
    return auc_score
    
def find_best_params():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=36000, show_progress_bar=True)
    best_study_params = study.best_params
    trial = study.best_trial

    print('best params: {}'.format(best_study_params))
    return best_study_params


params = find_best_params()


X_ts = preprocess(ts)
cat_clf_pipe = make_pipeline(column_trans,
                             XGBClassifier(**params)
                            ).fit(X, y)

y_hat_ts = cat_clf_pipe.predict_proba(X_ts)[:, 1]

test_pred_df = pd.Series(y_hat_ts)


sub_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub_df = sub_raw.copy()
sub_df['rainfall'] = y_hat_ts

display(sub_df.head(10))

sub_df.to_csv('submission.csv', index=False)
print('The file is ready for submission')

